"""AH raw-camera pilot with native-renderer equivalence, never automatic acceptance."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from gsplat import rasterization

from audit_static_source_cameras_v1 import H, sha, read, checked


def tensor(value):
    return torch.as_tensor(np.array(value), dtype=torch.float32, device='cuda').contiguous()


def render(gs, view, K, wh, near=.1, far=500.):
    rgbd, alpha, _ = rasterization(
        means=gs[:, :3].contiguous(), opacities=gs[:, 3].contiguous(),
        scales=gs[:, 4:7].contiguous(), quats=gs[:, 7:11].contiguous(),
        colors=gs[:, 11:].contiguous(), viewmats=tensor(view)[None], Ks=tensor(K)[None],
        width=wh[0], height=wh[1], near_plane=near, far_plane=far,
        packed=False, backgrounds=torch.ones((1,3), dtype=torch.float32, device='cuda'),
        render_mode='RGB+ED')
    if not torch.isfinite(rgbd).all() or not torch.isfinite(alpha).all():
        raise ValueError('Non-finite native AH rasterization')
    return rgbd[0,...,:3], alpha[0,...,0]


def difference(a, b):
    d = torch.abs(a-b)
    return dict(max=float(d.max()), mean=float(d.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-id', choices=['HN0004','HN0035','HN0039','HN0040'], required=True)
    sid = parser.parse_args().human_id
    rp = H / 'configs/human_source_common_v1/ah_registration_pilot_v1' / (sid+'.json')
    reg = read(rp)
    for path_key, hash_key in [('camera_contract','camera_contract_sha256'),
                               ('native_ply','native_ply_sha256'),('ahc_camera','ahc_camera_sha256'),
                               ('audit_record','audit_record_sha256')]:
        checked(reg[path_key], reg[hash_key])
    camera, ahc = read(reg['camera_contract']), read(reg['ahc_camera'])
    if reg['identity_key'] != camera['identity_key'] or reg['timestamp_us'] != camera['timestamp_us']:
        raise ValueError('Camera identity/exposure mismatch')
    repo = H.parent / 'assetbench/repos/ah'
    sys.path.insert(0, str(repo / 'benchmark'))
    from utils import load_ply, render_from_cameras
    gaussian_batch = load_ply(reg['native_ply']).cuda()
    gs = gaussian_batch[0]
    if gs.ndim != 2 or gs.shape[1] != 14 or not len(gs) or not torch.isfinite(gs).all():
        raise ValueError('Invalid native AH Gaussian payload')
    T = np.asarray(reg['T_source_camera_from_export'])
    scale = reg['physical_scale_estimate']
    R = T[:3,:3] / scale
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
        raise ValueError('Not a proper similarity')
    # Preserve the checkpoint coordinates and express the raw source camera in
    # that coordinate frame. This avoids altering SH/covariance during scoring.
    equivalent_view = np.eye(4)
    equivalent_view[:3,:3] = R
    equivalent_view[:3,3] = T[:3,3] / scale
    # Independent transformed-cloud route must rotate full covariance as well.
    mapped = gs.clone()
    mapped[:,:3] = gs[:,:3] @ tensor(T[:3,:3]).T + tensor(T[:3,3])
    mapped[:,4:7] *= scale
    q = gs[:,7:11].cpu().numpy()
    rotated = (Rotation.from_matrix(R) * Rotation.from_quat(q[:,[1,2,3,0]])).as_quat()
    mapped[:,7:11] = tensor(rotated[:,[3,0,1,2]])
    with torch.inference_mode():
        native_rgb, native_alpha = render(gs, reg['native_view'], reg['native_K'], [512,512])
        official_rgb, official_alpha = render_from_cameras(gaussian_batch, np.asarray(ahc['cam_directions']),
            ahc['dists'], float(np.asarray(ahc['fov']).reshape(-1)[0]), np.asarray(ahc['lwh']),
            output_size=512, bg_color=(1.,1.,1.))
        native_check = dict(rgb=difference(native_rgb, official_rgb[0].permute(1,2,0)),
                            alpha=difference(native_alpha, official_alpha[0,0]))
        rgb, alpha = render(gs, equivalent_view, camera['K_raw'], camera['render_wh'], near=.1/scale, far=500./scale)
        mapped_rgb, mapped_alpha = render(mapped, np.eye(4), camera['K_raw'], camera['render_wh'])
        transform_check = dict(rgb=difference(rgb, mapped_rgb), alpha=difference(alpha, mapped_alpha))
    out = H / 'exports/ah_source_camera_pilot_v1' / sid
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('source_rgb',rgb),('source_alpha',alpha),('native_white_rgb',native_rgb),('native_white_alpha',native_alpha)]:
        array = np.floor(value.clamp(0,1).cpu().numpy() * 255 + .5).astype(np.uint8)
        Image.fromarray(array).save(out / (name+'.png'))
    numerics_ok = all(v['max'] <= .002 for v in native_check.values()) and all(
        v['max'] <= .05 and v['mean'] <= .0001 for v in transform_check.values())
    report = dict(status='rendered_pending_registration_and_visual_review' if numerics_ok else 'renderer_equivalence_needs_investigation',
                  id=sid, registration=str(rp), registration_sha256=sha(rp),
                  script_sha256=sha(__file__), native_benchmark_renderer_sha256=sha(repo/'benchmark/utils.py'),
                  camera_contract=reg['camera_contract'], camera_contract_sha256=reg['camera_contract_sha256'],
                  native_renderer_equivalence=native_check, similarity_coordinate_equivalence=transform_check,
                  renderer_numerics_passed=numerics_ok, full_covariance_rotated=True, preserved_sh_degree=0,
                  registration_is_weak_perspective_initialization=True,
                  registration_limitations=reg['protocol']['limitation'],
                  K_raw=camera['K_raw'], render_wh=camera['render_wh'],
                  prediction_encoding='white_composited_uint8',
                  source_rgb=str(out/'source_rgb.png'), source_rgb_sha256=sha(out/'source_rgb.png'),
                  source_alpha=str(out/'source_alpha.png'), source_alpha_sha256=sha(out/'source_alpha.png'),
                  common_camera_valid=False, metric_eligible=False, scores_produced=False)
    (out/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
