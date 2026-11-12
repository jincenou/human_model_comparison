"""Prepare an explicit AH weak-perspective initialization, not a registration pass.

Use AHC's predicted orientation/physical scale and the historical input crop.
Transfer its origin and frontoparallel center-depth plane to the true intrinsics
using one proper Sim(3). Off-plane points are NOT projectively equivalent; preserve
and record that difference. No predicted bbox, score search, or joint edits.
"""
import json
import sys
from pathlib import Path
import numpy as np

from audit_static_source_cameras_v1 import H, sha, read


def initialization(native_view, native_k, raw_k, raw_to_input, physical_scale):
    view, kn, kr, affine = map(lambda x: np.asarray(x, dtype=np.float64),
                              (native_view, native_k, raw_k, raw_to_input))
    if not np.isfinite([physical_scale]).all() or physical_scale <= 0:
        raise ValueError('Positive physical scale required')
    if not np.allclose(view[3], [0, 0, 0, 1]) or not np.allclose(view[:3, :3].T @ view[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError('Native camera must be rigid')
    if np.linalg.det(view[:3, :3]) < 0:
        raise ValueError('Reflected camera')
    # AHC's orbit camera looks at the asset origin. Its native center lies on z.
    if np.max(np.abs(view[:2, 3])) > 1e-5 or view[2, 3] <= 0:
        raise ValueError('AHC source camera is not origin-targeted')
    crop_k = affine @ kr
    ratio = np.array([crop_k[0, 0] / kn[0, 0], crop_k[1, 1] / kn[1, 1]])
    if not np.allclose(ratio, ratio[0], rtol=1e-6, atol=1e-8):
        raise ValueError('Anisotropic focal conversion cannot be hidden in a similarity')
    if not np.allclose(kn[[0,1],[1,0]], 0) or not np.allclose(crop_k[[0,1],[1,0]], 0):
        raise ValueError('Skew requires a separate derivation')
    origin_ray = np.linalg.solve(kr, np.linalg.solve(affine, kn[:, 2]))
    origin_ray /= origin_ray[2]
    new_depth = physical_scale * view[2, 3] * ratio[0]
    transform = np.eye(4)
    transform[:3, :3] = physical_scale * view[:3, :3]
    transform[:3, 3] = origin_ray * new_depth
    return transform


def project(points, transform, K):
    xyz = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    pixels = xyz @ np.asarray(K).T
    return pixels[:, :2] / pixels[:, 2:3]


def main():
    # Use the very same installed orbit-camera implementation as AH's renderer.
    from kiui.cam import orbit_camera
    audit = H / 'reports/static_source_camera_audit_v1'
    manifest = read(audit / 'manifest.json')
    if manifest['passed'] != 100 or manifest['failed']:
        raise ValueError('Input/camera audit incomplete')
    out = H / 'configs/human_source_common_v1/ah_registration_pilot_v1'
    out.mkdir(exist_ok=False)
    records = []
    for sid in ['HN0004', 'HN0035', 'HN0039', 'HN0040']:
        record_path = audit / (sid + '.json')
        if sha(record_path) != manifest['records'][sid]['sha256']:
            raise ValueError('Changed audit record')
        record = read(record_path)
        ah = record['ah']
        cam = read(record['current_camera'])
        direction = np.asarray(ah['source_direction'][0])
        azimuth = np.rad2deg(np.arctan2(direction[0], -direction[1])) + 90
        elevation = -np.rad2deg(np.arcsin(direction[2]))
        pose = orbit_camera(elevation, azimuth, radius=ah['normalized_camera_distance'])
        pose[:3, 1:3] *= -1
        native_view = np.linalg.inv(pose)
        scale = max(ah['lwh_prediction'])
        K = np.asarray(cam['K_raw'])
        affine = np.asarray(ah['raw_to_generation_edge_affine'])
        T = initialization(native_view, ah['K_prediction'], K, affine, scale)
        # Check an independent 5x5 grid through the native origin plane and two
        # planes away from it. Only the center-depth plane has exact equivalence.
        xy = np.array([(x,y) for x in np.linspace(-.4,.4,5) for y in np.linspace(-.4,.4,5)])
        diagnostics = []
        for depth in [-.25, 0., .25]:
            camera_relative = np.column_stack([xy, np.full(len(xy), depth)])
            points = camera_relative @ native_view[:3, :3]
            native_pixels = project(points, native_view, np.asarray(ah['K_prediction']))
            expected_h = np.column_stack([native_pixels, np.ones(len(points))]) @ np.linalg.inv(affine).T
            expected = expected_h[:, :2] / expected_h[:, 2:3]
            actual = project(points, T, K)
            diagnostics.append(dict(native_depth_offset=depth,
                                    raw_pixel_difference_max=float(np.abs(actual-expected).max()),
                                    raw_pixel_difference_mean=float(np.abs(actual-expected).mean())))
        if diagnostics[1]['raw_pixel_difference_max'] > 1e-3:
            raise ValueError('Origin-plane derivation failed')
        singular = np.linalg.svd(T[:3, :3], compute_uv=False)
        if not np.allclose(singular, scale, rtol=1e-6) or np.linalg.det(T[:3, :3]) <= 0:
            raise ValueError('Transform deforms or reflects the asset')
        world = np.asarray(cam['T_world_from_camera_opencv']) @ T
        reg = dict(schema='ah_analytic_source_initialization_v1', id=sid,
                   status='candidate_pending_actual_camera_render_and_review',
                   identity_key=record['identity_key'], timestamp_us=record['timestamp_us'],
                   camera_contract=record['current_camera'], camera_contract_sha256=record['current_camera_sha256'],
                   audit_record=str(record_path), audit_record_sha256=sha(record_path),
                   native_ply=ah['ply'], native_ply_sha256=ah['ply_sha256'],
                   ahc_camera=ah['source_camera'], ahc_camera_sha256=ah['source_camera_sha256'],
                   native_view=native_view.tolist(), native_K=ah['K_prediction'],
                   physical_scale_estimate=scale, T_source_camera_from_export=T.tolist(),
                   T_world_from_export=world.tolist(), plane_transfer_diagnostics=diagnostics,
                   protocol=dict(orientation='unchanged AHC predicted source orientation',
                                 scale='AHC max(lwh) physical scale estimate; not ground truth',
                                 translation='origin-ray and depth transfer from historical crop/focal ratio',
                                 pixel_convention='continuous crop-edge coordinates; integer-center alternative retained in audit',
                                 limitation='weak-perspective initialization; exact only on the origin-depth plane; not a calibrated recovery',
                                 forbidden_steps_used=[], joint_pose_modified=False, prediction_bbox_used=False,
                                 rgb_or_metric_optimization=False, true_holdout=False),
                   script_sha256=sha(__file__), metric_eligible=False, common_camera_valid=False, scores_produced=False)
        path = out / (sid + '.json')
        path.write_text(json.dumps(reg, indent=2, allow_nan=False) + '\n')
        records.append(dict(id=sid, path=str(path), sha256=sha(path), diagnostics=diagnostics))
    (out / 'manifest.json').write_text(json.dumps(dict(status='four_initializations_prepared_not_accepted', records=records), indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
