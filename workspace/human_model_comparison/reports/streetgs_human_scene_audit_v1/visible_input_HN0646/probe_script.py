"""One fixed source-input-mask extraction diagnostic; native parameters untouched."""
from pathlib import Path
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
sys.path[:0] = [str(H / 'scripts'), str(R / 'scripts')]
from extract_streetgs_human_background_v1 import KEYS, read, save, sha, render
from run_background_vehicle_scene import DriveStudio


def main():
    torch.set_num_threads(2)
    sid = 'HN0646'
    out = H / 'reports/streetgs_human_scene_audit_v1/visible_input_HN0646'
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    selection_path = H / 'inputs/benchmark_human100_v3/selection.json'
    obj = next(x for x in read(selection_path)['objects'] if x['id'] == sid)
    rec = read(H / 'exports/streetgs_human_probe_v1/objects' / sid / 'validation.json')
    assert rec['status'] == 'passed' and sha(rec['bundle']) == rec['bundle_sha256']
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    old = torch.load(rec['bundle'], map_location='cpu', weights_only=False)
    camera, binding = old['camera'], old['binding']
    assert sha(camera['source_rgb']) == camera['source_rgb_sha256']
    mapping_path, mask_path = obj['paths']['frame_mapping'], obj['paths']['mask_npz']
    mapping = read(mapping_path)['frames'][25]
    assert mapping['slot'] == 25 and not mapping['interpolated']
    assert mapping['left_native_index'] == mapping['right_native_index'] == obj['reference_frame']['native_index'] == 12
    assert mapping['source_path'] == camera['source_rgb']
    assert obj['t_ref_us'] == camera['timestamp_us']
    local = np.load(mask_path, allow_pickle=False)['masks'][25] > 0
    x0, y0, x1, y1 = obj['reference_frame']['crop_xyxy']
    assert [x0, y0, x1, y1] == [916, 261, 1070, 541]
    assert local.shape == (y1-y0, x1-x0)
    assert np.array_equal(mapping['undistorted_to_input'], obj['reference_frame']['undistorted_to_input'])
    width, height = camera['render_wh']
    truth = np.zeros((height, width), dtype=bool)
    truth[y0:y1, x0:x1] = local
    Image.fromarray((truth * 255).astype('uint8')).save(out / 'input_mask_full.png')

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
    assert float(np.abs(ci['camera_to_world'].cpu().numpy() - pose).max()) < 1e-4
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
    kwargs = capture['kwargs']
    n_all = len(kwargs['means'])
    colors = torch.ones((n_all, 3), device='cuda', dtype=kwargs['means'].dtype, requires_grad=True)
    kwargs.update(colors=colors, render_mode='RGB', sh_degree=None, absgrad=False)
    rendered, alpha, _ = raster(**kwargs)
    alpha_error = float((alpha - capture['alpha']).abs().max())
    constant_error = float((rendered - alpha).abs().max())
    assert alpha_error < 1e-5 and constant_error < 1e-5
    mask = torch.from_numpy(truth).cuda()
    loss = rendered[0, :, :, 0][mask].sum() + rendered[0, :, :, 1][~mask].sum()
    weights = torch.autograd.grad(loss, colors, only_inputs=True)[0][:, :2].detach().cpu().numpy()
    assert np.isfinite(weights).all() and (weights >= 0).all()
    expected = np.array([float(alpha[0, :, :, 0][mask].double().sum()), float(alpha[0, :, :, 0][~mask].double().sum())])
    actual = weights.astype(np.float64).sum(0)
    conservation = np.abs(actual-expected) / np.maximum(expected, 1.)
    assert float(conservation.max()) < 2e-4
    fg, bg = weights[:n_background].T
    selected = np.flatnonzero(fg > bg).astype(np.int64)
    assert len(selected) > 0
    old_ids = old['original_indices'].numpy()
    np.savez_compressed(out / 'native_contributions.npz', foreground_weight=fg, background_weight=bg)
    np.save(out / 'native_indices.npy', selected, allow_pickle=False)
    canonical = {k: getattr(native.model, k)[torch.as_tensor(selected, device='cuda')].detach().cpu().clone().contiguous() for k in KEYS}
    ck = torch.load(rec['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    assert all(torch.equal(canonical[k], ck['models']['Background'][k][selected]) for k in KEYS)
    pack = dict(old, canonical=canonical, original_indices=torch.as_tensor(selected),
                schema='streetgs_human_input_visible_majority_diagnostic_v1')
    torch.save(pack, out / 'native_background_subset.pt')
    del ck, rendered, alpha, loss, colors, capture, kwargs, native, trainer, dataset
    torch.cuda.empty_cache()
    rgb, alpha_png = render(pack)
    Image.fromarray(rgb).save(out / 'native_actor_source.png')
    Image.fromarray(alpha_png).save(out / 'native_actor_alpha.png')
    reloaded = torch.load(out / 'native_background_subset.pt', map_location='cpu', weights_only=False)
    rgb2, alpha2 = render(reloaded)
    assert np.array_equal(rgb, rgb2) and np.array_equal(alpha_png, alpha2)
    oldrgb = np.asarray(Image.open(rec['actor_preview']).convert('RGB'))
    source = np.asarray(Image.open(camera['source_rgb']).convert('RGB'))
    roi = tuple(camera['roi_xyxy'])
    sheet = Image.new('RGB', (512*4, 552), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, arr) in enumerate([('Source RGB', source), ('Native full scene', full), ('Old box subset', oldrgb), ('Input-mask contribution subset', rgb)]):
        draw.text((i*512+12, 12), name, fill='black')
        sheet.paste(Image.fromarray(arr).crop(roi).resize((512, 512), Image.Resampling.BICUBIC), (i*512, 40))
    sheet.save(out / 'comparison.jpg', quality=95)
    audit = dict(id=sid, status='passed', rule='native visible foreground contribution > background contribution, one fixed source input mask',
                 source_selection=str(selection_path), source_selection_sha256=sha(selection_path),
                 mask_npz=mask_path, mask_npz_sha256=sha(mask_path), mask_key='masks', mask_slot=25,
                 frame_mapping=mapping_path, frame_mapping_sha256=sha(mapping_path), raw_crop_xyxy=[x0,y0,x1,y1],
                 exact_native_index=12, timestamp_us=obj['t_ref_us'], source_rgb=camera['source_rgb'], source_rgb_sha256=camera['source_rgb_sha256'],
                 checkpoint=rec['checkpoint'], checkpoint_sha256=rec['checkpoint_sha256'],
                 native_background_count=n_background, native_scene_count=n_all, old_selected_count=len(old_ids), selected_count=len(selected),
                 selected_overlap_old_count=int(np.isin(selected, old_ids).sum()),
                 fg_mass_background=float(fg.sum(dtype=np.float64)), fg_mass_old=float(fg[old_ids].sum(dtype=np.float64)),
                 fg_mass_selected=float(fg[selected].sum(dtype=np.float64)), bg_mass_old=float(bg[old_ids].sum(dtype=np.float64)),
                 bg_mass_selected=float(bg[selected].sum(dtype=np.float64)), alpha_max_difference=alpha_error,
                 constant_feature_max_difference=constant_error, contribution_conservation_relative_error=conservation.tolist(),
                 selected_parameters_equal_checkpoint=True, reload_exact=True, gaussian_parameters_modified=False,
                 reference_mask_read=False, formal_metrics_updated=False, trained=False,
                 limitation='Single input-view visible extraction; may retain background surfaces behind the person and cannot recover missing human geometry or motion.',
                 comparison=str(out/'comparison.jpg'), bundle_sha256=sha(out/'native_background_subset.pt'),
                 script_sha256=sha(__file__), elapsed_s=time.time()-start)
    save(out/'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
