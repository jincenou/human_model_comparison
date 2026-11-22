"""Fixed source-mask extraction: retain every native BG Gaussian with F > 0.

Masks select unchanged native rows, never erase pixels from exported predictions.
The masked whole-scene panel is explicitly a display-only diagnostic.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
sys.path[:0] = [str(H / 'scripts'), str(R / 'scripts')]
from extract_streetgs_human_background_v1 import KEYS, read, save, sha, render
from run_background_vehicle_scene import DriveStudio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', required=True, choices=['HN0646', 'HN0108', 'HN0273', 'HN0317'])
    args = ap.parse_args()
    sid = args.id
    torch.set_num_threads(2)
    out = H / 'reports/streetgs_human_mask_all_v2' / sid
    out.mkdir(parents=True, exist_ok=False)
    (out / 'probe_script.py').write_text(Path(__file__).read_text())
    start = time.time()
    validation_path = H / 'exports/streetgs_human_probe_v1/objects' / sid / 'validation.json'
    rec = read(validation_path)
    assert rec['status'] == 'passed'
    assert sha(rec['bundle']) == rec['bundle_sha256']
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    old = torch.load(rec['bundle'], map_location='cpu', weights_only=False)
    camera, binding = old['camera'], old['binding']
    assert sha(rec['camera_contract']) == rec['camera_contract_sha256']
    assert sha(camera['source_rgb']) == camera['source_rgb_sha256']
    mask_path = camera['source_mask']
    assert sha(mask_path) == camera['source_mask_sha256']
    width, height = camera['render_wh']
    truth = np.asarray(Image.open(mask_path).convert('L')) > 127
    assert truth.shape == (height, width) and truth.any()
    Image.fromarray((truth * 255).astype('uint8')).save(out / 'selection_mask_full.png')
    mask_evidence = dict(source_reference_mask=mask_path, source_reference_mask_sha256=sha(mask_path),
                         conditioning='Frozen exact-source reference mask used for annotation-assisted row extraction',
                         prediction_erased_with_reference_mask=False, reference_mask_read=True)

    native = DriveStudio(rec)
    trainer, dataset = native.trainer, native.dataset
    from datasets.base.pixel_source import get_rays
    import models.trainers.base as base
    frame, camid = binding['source_frame'], binding['native_camera_id']
    index = frame * dataset.pixel_source.num_cams + dataset.pixel_source.camera_data[camid].unique_cam_idx
    assert index == binding['source_camera_index']
    info, ci = dataset.pixel_source.get_image(index)
    info = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in info.items()}
    ci = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ci.items()}
    pose = native.T @ np.asarray(camera['T_world_from_camera_opencv'])
    camera_error = float(np.abs(ci['camera_to_world'].cpu().numpy() - pose).max())
    assert camera_error < 1e-4
    ci.update(camera_to_world=torch.tensor(pose, dtype=torch.float32, device='cuda'),
              intrinsics=torch.tensor(camera['K_raw'], dtype=torch.float32, device='cuda'),
              width=torch.tensor(width, device='cuda'), height=torch.tensor(height, device='cuda'))
    x, y = torch.meshgrid(torch.arange(width, device='cuda'), torch.arange(height, device='cuda'), indexing='xy')
    origins, directions, norm = get_rays(x.flatten(), y.flatten(), ci['camera_to_world'], ci['intrinsics'])
    info = {k: info[k].flatten()[0].expand(height, width) for k in ('img_idx', 'normed_time', 'frame_idx')}
    info.update(origins=origins.reshape(height, width, 3), viewdirs=directions.reshape(height, width, 3),
                direction_norm=norm.reshape(height, width, 1), pixel_coords=torch.stack([y / height, x / width], dim=-1))
    n_background = len(native.model._means)
    raster = base.rasterization
    capture = {}

    def recording_raster(**kwargs):
        result = raster(**kwargs)
        if not capture and len(kwargs['means']) >= n_background:
            assert torch.equal(kwargs['means'][:n_background], native.model._means)
            capture.update(kwargs={k: v.detach() if isinstance(v, torch.Tensor) else v
                                   for k, v in kwargs.items() if k not in ['colors', 'backgrounds']},
                           alpha=result[1].detach())
        return result

    base.rasterization = recording_raster
    try:
        with torch.no_grad():
            result = trainer(info, ci, novel_view=True)
            full = (result['rgb'].cpu().numpy().clip(0, 1) * 255).round().astype('uint8')
            Image.fromarray(full).save(out / 'full_scene_source_camera.png')
            del result
    finally:
        base.rasterization = raster
    assert capture
    mask = torch.from_numpy(truth).cuda()
    kwargs = capture['kwargs']
    n_all = len(kwargs['means'])
    native_alpha = capture['alpha']
    expected = np.array([float(native_alpha[0, :, :, 0][mask].double().sum()), float(native_alpha[0, :, :, 0][~mask].double().sum())])
    reuse = None
    if sid == 'HN0273':
        reuse_dir = H / 'reports/streetgs_human_scene_audit_v1/visible_source_reference_HN0273'
        prior = read(reuse_dir / 'audit.json')
        assert prior['status'] == 'passed'
        assert prior['source_reference_mask_sha256'] == camera['source_mask_sha256']
        assert prior['checkpoint_sha256'] == rec['checkpoint_sha256']
        assert prior['source_rgb_sha256'] == camera['source_rgb_sha256']
        assert prior['native_background_count'] == n_background and prior['native_scene_count'] == n_all
        prior_weights = reuse_dir / 'native_contributions.npz'
        z = np.load(prior_weights, allow_pickle=False)
        fg, bg = z['foreground_weight'], z['background_weight']
        assert fg.shape == bg.shape == (n_background,)
        assert np.isclose(fg.sum(dtype=np.float64), prior['fg_mass_background'], rtol=1e-12)
        alpha_error, constant_error = prior['alpha_max_difference'], prior['constant_feature_max_difference']
        conservation = prior['contribution_conservation_relative_error']
        reuse = dict(audit=str(reuse_dir/'audit.json'), audit_sha256=sha(reuse_dir/'audit.json'),
                     weights=str(prior_weights), weights_sha256=sha(prior_weights),
                     reused_background_weights=True, repeated_contribution_backward=False)
    else:
        colors = torch.ones((n_all, 3), device='cuda', dtype=kwargs['means'].dtype, requires_grad=True)
        kwargs.update(colors=colors, render_mode='RGB', sh_degree=None, absgrad=False)
        rendered, alpha, _ = raster(**kwargs)
        alpha_error = float((alpha-native_alpha).abs().max())
        constant_error = float((rendered-alpha).abs().max())
        assert alpha_error < 1e-5 and constant_error < 1e-5
        loss = rendered[0, :, :, 0][mask].sum() + rendered[0, :, :, 1][~mask].sum()
        weights = torch.autograd.grad(loss, colors, only_inputs=True)[0][:, :2].detach().cpu().numpy()
        assert np.isfinite(weights).all() and (weights >= 0).all()
        actual = weights.astype(np.float64).sum(0)
        conservation = (np.abs(actual-expected) / np.maximum(expected, 1.)).tolist()
        assert max(conservation) < 2e-4
        fg, bg = weights[:n_background].T
        np.savez_compressed(out/'native_contributions.npz', foreground_weight=fg, background_weight=bg)
        del weights, loss, rendered, alpha, colors
    assert np.isfinite(fg).all() and np.isfinite(bg).all() and (fg >= 0).all() and (bg >= 0).all()
    total_bg_fg = float(fg.sum(dtype=np.float64))
    non_bg_fg = float(expected[0]-total_bg_fg)
    assert non_bg_fg >= -max(expected[0]*2e-4, 1e-3)
    original_ids = old['original_indices'].numpy()
    variants = dict(majority=np.flatnonzero(fg > bg).astype(np.int64), all_visible=np.flatnonzero(fg > 0).astype(np.int64))
    assert all(len(ids) > 0 for ids in variants.values())
    # Removing zero entries changes NumPy's reduction grouping by roundoff only.
    assert np.all(fg[fg <= 0] == 0)
    assert np.isclose(float(fg[variants['all_visible']].sum(dtype=np.float64)), total_bg_fg, rtol=1e-12, atol=1e-9)
    checkpoint = torch.load(rec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    bundles = {}
    variant_audits = {}
    for name, ids in variants.items():
        dest = out/name; dest.mkdir()
        np.save(dest/'native_indices.npy', ids, allow_pickle=False)
        native_indices = torch.as_tensor(ids, device='cuda')
        canonical = {k: getattr(native.model, k)[native_indices].detach().cpu().clone().contiguous() for k in KEYS}
        assert all(torch.equal(canonical[k], checkpoint['models']['Background'][k][ids]) for k in KEYS)
        pack = dict(old, canonical=canonical, original_indices=torch.as_tensor(ids),
                    schema='streetgs_human_mask_all_v2', selection_variant=name, mask_evidence=mask_evidence)
        bundle_path = dest/'native_background_subset.pt'
        torch.save(pack, bundle_path)
        bundles[name] = bundle_path
        variant_audits[name] = dict(selected_count=len(ids), selected_overlap_old_count=int(np.isin(ids, original_ids).sum()),
                                   fg_mass_selected=float(fg[ids].sum(dtype=np.float64)), bg_mass_selected=float(bg[ids].sum(dtype=np.float64)),
                                   fg_share_native_background=float(fg[ids].sum(dtype=np.float64)/total_bg_fg),
                                   parameters_equal_checkpoint=True, bundle=str(bundle_path), bundle_sha256=sha(bundle_path))
    del checkpoint, capture, kwargs, native_alpha, native, trainer, dataset
    torch.cuda.empty_cache()
    images = {}
    for name, path in bundles.items():
        pack = torch.load(path, map_location='cpu', weights_only=False)
        rgb, alpha = render(pack)
        Image.fromarray(rgb).save(path.parent/'native_actor_source.png')
        Image.fromarray(alpha).save(path.parent/'native_actor_alpha.png')
        reloaded = torch.load(path, map_location='cpu', weights_only=False)
        rgb2, alpha2 = render(reloaded)
        assert np.array_equal(rgb, rgb2) and np.array_equal(alpha, alpha2)
        variant_audits[name]['reload_exact'] = True
        images[name] = rgb
    oldrgb = np.asarray(Image.open(rec['actor_preview']).convert('RGB'))
    masked = np.where(truth[..., None], full, 255).astype('uint8')
    Image.fromarray(masked).save(out/'full_scene_masked_DISPLAY_ONLY.png')
    roi = tuple(camera['roi_xyxy'])
    sheet = Image.new('RGB', (512*5, 552), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'), 22)
    panels = [('Native full scene', full), ('Scene + mask (display only)', masked), ('Old box subset', oldrgb),
              ('Foreground > background', images['majority']), ('All foreground > 0', images['all_visible'])]
    for i, (name, arr) in enumerate(panels):
        draw.text((i*512+12, 8), name, fill='black', font=font)
        sheet.paste(Image.fromarray(arr).crop(roi).resize((512,512), Image.Resampling.BICUBIC), (i*512,40))
    sheet.save(out/'comparison.jpg', quality=95)
    evidence = dict(id=sid, status='passed', rules=dict(majority='F > B', all_visible='F > 0 exactly; no positive threshold'),
                    **mask_evidence, checkpoint=rec['checkpoint'], checkpoint_sha256=rec['checkpoint_sha256'],
                    source_rgb=camera['source_rgb'], source_rgb_sha256=camera['source_rgb_sha256'],
                    camera_contract=rec['camera_contract'], camera_contract_sha256=rec['camera_contract_sha256'],
                    source_camera_error=camera_error, camera_policy='Frozen source camera; learned CamPose disabled',
                    native_background_count=n_background, native_scene_count=n_all,
                    fg_mass_full_scene=float(expected[0]), fg_mass_background=total_bg_fg,
                    fg_mass_nonbackground_raw=non_bg_fg,
                    fg_share_nonbackground=float(non_bg_fg/max(expected[0],1.)),
                    nonbackground_selection_policy='Never included in exported subsets; preserve native fullscene occlusion for contribution estimation',
                    old_selected_count=len(original_ids), old_fg_mass=float(fg[original_ids].sum(dtype=np.float64)),
                    old_bg_mass=float(bg[original_ids].sum(dtype=np.float64)), variants=variant_audits,
                    alpha_max_difference=alpha_error, constant_feature_max_difference=constant_error,
                    contribution_conservation_relative_error=conservation, prior_contribution_reuse=reuse,
                    formal_metrics_updated=False, gaussian_parameters_modified=False, trained=False,
                    limitation='F>0 recovers all visible BG contributions in this mask, including mixed/background Gaussian tails; neither clean 3D segmentation nor dynamic human reconstruction is guaranteed.',
                    comparison=str(out/'comparison.jpg'), script_sha256=sha(__file__), elapsed_s=time.time()-start)
    save(out/'audit.json', evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
