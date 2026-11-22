"""Render unchanged StreetGS scenes at the frozen human source exposures."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
OUT = H / 'reports/streetgs_human_scene_audit_v1'
sys.path[:0] = [str(H / 'scripts'), str(R / 'scripts')]
from extract_streetgs_human_background_v1 import read, save, sha
from run_background_vehicle_scene import DriveStudio


def png(t, path):
    a = t.detach().cpu().numpy()
    assert np.isfinite(a).all()
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    a = (a.clip(0, 1) * 255).round().astype('uint8')
    Image.fromarray(a).save(path)
    return a


def main():
    p = argparse.ArgumentParser(); p.add_argument('--id', required=True); args = p.parse_args()
    torch.set_num_threads(2)
    sid = args.id
    rec = read(H / 'exports/streetgs_human_probe_v1/objects' / sid / 'validation.json')
    assert rec['status'] == 'passed'
    pack = torch.load(rec['bundle'], map_location='cpu', weights_only=False)
    assert sha(rec['bundle']) == rec['bundle_sha256']
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    camera, binding = pack['camera'], pack['binding']
    assert sha(camera['source_rgb']) == camera['source_rgb_sha256']
    out = OUT / sid; out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    native = DriveStudio(rec)
    trainer, dataset = native.trainer, native.dataset
    from datasets.base.pixel_source import get_rays
    from models.gaussians.basics import dataclass_gs
    frame, camid = binding['source_frame'], binding['native_camera_id']
    index = frame * dataset.pixel_source.num_cams + dataset.pixel_source.camera_data[camid].unique_cam_idx
    assert index == binding['source_camera_index']
    info, ci = dataset.pixel_source.get_image(index)
    info = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in info.items()}
    ci = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in ci.items()}
    pose = native.T @ np.asarray(camera['T_world_from_camera_opencv'])
    error = float(np.abs(ci['camera_to_world'].cpu().numpy() - pose).max())
    assert error < 1e-4
    width, height = camera['render_wh']
    ci.update(camera_to_world=torch.tensor(pose, dtype=torch.float32, device='cuda'),
              intrinsics=torch.tensor(camera['K_raw'], dtype=torch.float32, device='cuda'),
              width=torch.tensor(width, device='cuda'), height=torch.tensor(height, device='cuda'))
    x, y = torch.meshgrid(torch.arange(width, device='cuda'), torch.arange(height, device='cuda'), indexing='xy')
    origins, directions, norm = get_rays(x.flatten(), y.flatten(), ci['camera_to_world'], ci['intrinsics'])
    info = {k: info[k].flatten()[0].expand(height, width) for k in ('img_idx', 'normed_time', 'frame_idx')}
    info.update(origins=origins.reshape(height, width, 3), viewdirs=directions.reshape(height, width, 3),
                direction_norm=norm.reshape(height, width, 1), pixel_coords=torch.stack([y / height, x / width], dim=-1))
    paths = {}
    with torch.no_grad():
        # The regular scene renderer includes every native node, Sky and Affine.
        for name, novel in [('scene_native', False), ('scene_source_camera', True)]:
            result = trainer(info, ci, novel_view=novel)
            path = out / (name + '.png'); png(result['rgb'], path); paths[name] = str(path)
            del result
        cam = trainer.process_camera(ci, info['img_idx'].flatten()[0], novel_view=True)
        gs = native.model.get_gaussians(cam)
        assert torch.equal(gs['_means'], native.model._means)
        for name, ix in [('background_source_camera', None), ('subset_source_camera', pack['original_indices'].cuda())]:
            selected = dataclass_gs(**{k: v if ix is None else v[ix] for k, v in gs.items()}, detach_keys=[], extras=None)
            result, _ = trainer.render_gaussians(gs=selected, cam=cam, near_plane=trainer.render_cfg.near_plane,
                far_plane=trainer.render_cfg.far_plane, render_mode='RGB+ED', radius_clip=trainer.render_cfg.get('radius_clip', 0.))
            alpha, raw = result['opacity'], result['rgb_gaussians']
            bias = trainer.affine_transformation(torch.zeros_like(raw), info)
            rgb = trainer.affine_transformation(raw, info) - bias * (1 - alpha) + (1 - alpha)
            path = out / (name + '.png'); arr = png(rgb, path); paths[name] = str(path)
            if ix is not None:
                alpha_arr = png(alpha, out / 'subset_alpha.png')
                old = np.asarray(Image.open(rec['actor_preview']))
                delta = np.abs(arr.astype(int) - old.astype(int))
                old_a = np.asarray(Image.open(rec['actor_alpha']))
                replay = dict(rgb_max=int(delta.max()), rgb_mean=float(delta.mean()),
                              alpha_max=int(np.abs(alpha_arr.astype(int) - old_a.astype(int)).max()))
                assert replay['rgb_max'] <= 1 and replay['alpha_max'] <= 1, replay
        corrected = trainer.process_camera(ci, info['img_idx'].flatten()[0], novel_view=False)
        pose_delta = np.linalg.inv(cam.camtoworlds.cpu().numpy()) @ corrected.camtoworlds.cpu().numpy()
    source_row = read(H / 'reports/streetgs_human_probe_v1/static_rows' / (sid + '.json'))
    result = dict(id=sid, scene=rec['scene'], status='passed', checkpoint=rec['checkpoint'],
        checkpoint_sha256=rec['checkpoint_sha256'], source_rgb=camera['source_rgb'],
        source_rgb_sha256=camera['source_rgb_sha256'], native_frame=frame, native_camera_id=camid,
        reference=source_row['reference'], images=paths,
        image_sha256={k: sha(v) for k, v in paths.items()}, old_subset_replay=replay,
        source_camera_error=error, native_CamPose_delta=pose_delta.tolist(),
        model_nodes=list(trainer.models), scene_gaussian_classes=list(trainer.gaussian_classes),
        background_gaussians=len(native.model._means), subset_gaussians=len(pack['original_indices']),
        native_scene_policy='Unmodified full native trainer: all nodes, Sky, Affine, learned CamPose',
        source_camera_policy='Same full trainer with CamPose disabled to match frozen source calibration',
        trained=False, gaussian_parameters_modified=False, reference_mask_used_in_render=False,
        script_sha256=sha(__file__), elapsed_s=time.time()-start)
    save(out / 'audit.json', result)
    print(sid, 'passed', replay, flush=True)


if __name__ == '__main__':
    main()
