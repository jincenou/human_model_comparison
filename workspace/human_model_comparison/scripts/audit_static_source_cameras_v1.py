"""Trace historical static-method inputs to raw cameras without certifying registration.

This audit never changes generation inputs, assets, reference qualification, or scores.
The original TRELLIS preprocessing method is executed on the existing RGBA inputs
without importing its model, CUDA, or downloading/removing a background.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

H = Path(__file__).resolve().parents[1]
P = H.parent / 'vehicle_model_comparison/model_preparation/20261109_2100_CST'
TRELLIS = P / 'sources/trellis_original_legacy_20261110'
PIPELINE = TRELLIS / 'trellis/pipelines/trellis_image_to_3d.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Content hash mismatch: {path}')
    return actual


def crop_affines(box, output_wh):
    """Record both conventions explicitly; this function does not choose a renderer.

    Edge coordinates map the crop edges to the resized image edges. Integer sample
    centers additionally use Pillow's half-pixel sampling offset. These are not
    interchangeable without checking the downstream rasterizer's convention.
    """
    left, top, right, bottom = map(float, box)
    if not (right > left and bottom > top and min(output_wh) > 0):
        raise ValueError('Empty crop or output')
    sx, sy = output_wh[0] / (right - left), output_wh[1] / (bottom - top)
    edge = np.array([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1.]])
    centers = edge.copy()
    centers[:2, 2] += .5 * np.array([sx, sy]) - .5
    return edge, centers


def load_original_preprocessor():
    tree = ast.parse(PIPELINE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TrellisImageTo3DPipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_image')
    namespace = {'np': np, 'Image': Image}
    # Only this locally inspected method is extracted. Valid RGBA inputs take its
    # alpha branch, so model imports, rembg, network and CUDA are never invoked.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(PIPELINE), 'exec'), namespace)
    return namespace['preprocess_image']


def trellis_conditioning_trace(image, original_preprocess):
    if image.mode != 'RGBA':
        raise ValueError('Historical input must be RGBA; do not invoke rembg')
    alpha = np.asarray(image)[..., 3]
    if np.all(alpha == 255):
        raise ValueError('Opaque input would invoke unrecorded background removal')
    points = np.argwhere(alpha > .8 * 255)
    if not len(points):
        raise ValueError('No original TRELLIS foreground')
    y0, x0 = points.min(0)
    y1, x1 = points.max(0)
    center = [(x0 + x1) / 2., (y0 + y1) / 2.]
    size = int(max(x1 - x0, y1 - y0) * 1.2)
    float_box = [center[0] - size // 2, center[1] - size // 2,
                 center[0] + size // 2, center[1] + size // 2]
    # Pillow Image._crop rounds each edge to an integer, including half-to-even.
    effective_box = [int(round(v)) for v in float_box]
    rgba = image.crop(effective_box).resize((518, 518), Image.Resampling.LANCZOS)
    values = np.asarray(rgba).astype(np.float32) / 255
    reconstructed = (values[..., :3] * values[..., 3:4] * 255).astype(np.uint8)
    original = np.asarray(original_preprocess(SimpleNamespace(), image.copy()))
    if not np.array_equal(original, reconstructed):
        raise ValueError('Reconstructed preprocessing differs from original method')
    edge, centers = crop_affines(effective_box, (518, 518))
    return dict(alpha_threshold_byte=204, alpha_rule='strictly greater than 0.8 * 255',
                bbox_inclusive_xyxy=[int(x0), int(y0), int(x1), int(y1)],
                requested_crop_xyxy=float_box, effective_crop_xyxy=effective_box,
                output_wh=[518, 518], resize='Pillow LANCZOS RGBA',
                background='RGB multiplied by resized alpha; native black conditioning',
                exact_original_preprocess_pixel_match=True,
                conditioning_rgb_pixel_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                input_to_conditioning_edge_affine=edge.tolist(),
                input_to_conditioning_integer_center_affine=centers.tolist())


def audit_row(row, ref, preprocess):
    sid = row['sample_id']
    for other in (ref,):
        if row['identity_key'] != other['identity_key'] or row['timestamp_us'] != other['timestamp_us']:
            raise ValueError('Identity/exposure mismatch')
    old_path = Path(row['camera_candidate'])
    checked(old_path, row['camera_candidate_sha256'])
    checked(ref['camera_contract'], ref['camera_contract_sha256'])
    old, current = read(old_path), read(ref['camera_contract'])
    if not np.array_equal(old['K_raw'], current['K_raw']) or not np.array_equal(
            old['T_world_from_camera_opencv'], current['T_world_from_camera_opencv']):
        raise ValueError('Raw camera changed between generation and reference review')
    for file_key, hash_key in [('image_rgba', 'image_rgba_sha256'), ('image', 'source_image_sha256'),
                               ('mask', 'source_mask_sha256'), ('raw_source_path', 'raw_source_sha256')]:
        checked(row[file_key], row[hash_key])
    checked(current['source_rgb'], current['source_rgb_sha256'])
    if row['raw_source_sha256'] != current['source_rgb_sha256']:
        raise ValueError('Generation and evaluation use different raw RGB')
    edge, centers = crop_affines(old['roi_xyxy'], (512, 512))
    with Image.open(row['image_rgba']) as rgba:
        trace = trellis_conditioning_trace(rgba, preprocess)
    trace['raw_to_conditioning_edge_affine'] = (np.array(trace['input_to_conditioning_edge_affine']) @ edge).tolist()
    trace['raw_to_conditioning_integer_center_affine'] = (np.array(trace['input_to_conditioning_integer_center_affine']) @ centers).tolist()
    run = H / 'runs/trellis_original_human_native_v1' / sid
    state, reload = read(run / 'state.json'), read(run / 'reload_validation.json')
    if state['input_sha256'] != row['image_rgba_sha256'] or reload['sample_id'] != sid:
        raise ValueError('TRELLIS native identity/input mismatch')
    checked(run / 'native_gaussians.pt', state['output_hashes']['native_gaussians.pt'])
    if reload['output_sha256'] != state['output_hashes']['native_gaussians.pt']:
        raise ValueError('TRELLIS reload payload mismatch')
    job_name = 'human_pilot_v1_HN0004' if sid == 'HN0004' else f'human_native_v1_{sid}'
    job_path = H / 'runs/ah' / job_name
    inp = read(H / 'inputs/ah/jobs' / job_name / 'source_manifest.json')
    job, validation = read(job_path / 'job.json'), read(job_path / 'reload_validation/validation.json')
    if inp['identity_key'] != ref['identity_key'] or inp['timestamp_us'] != ref['timestamp_us']:
        raise ValueError('AH identity/exposure mismatch')
    for key, digest in [('image', 'source_image_sha256'), ('mask', 'source_mask_sha256')]:
        checked(inp[key], inp[digest])
    ply = Path(job['output_ply'])
    checked(ply, validation['ply_sha256'])
    camera_path = ply.parent / 'ahc_source_prediction.json'
    ahc = read(camera_path)
    direction = np.asarray(ahc['cam_directions'], dtype=float)
    distance, fov = float(np.asarray(ahc['dists']).flat[0]), float(np.asarray(ahc['fov']).flat[0])
    lwh = np.asarray(ahc['lwh'], dtype=float)
    if direction.shape != (1, 3) or abs(np.linalg.norm(direction[0]) - 1) > 1e-3:
        raise ValueError('AH source direction invalid')
    if not (0 < fov < 180 and distance > 0 and lwh.shape == (3,) and np.isfinite(lwh).all() and min(lwh) > 0):
        raise ValueError('AH source intrinsic/scale invalid')
    ah_box = inp.get('roi_xyxy', old['roi_xyxy'])
    ah_edge, ah_centers = crop_affines(ah_box, (512, 512))
    raw_k = np.asarray(old['K_raw'])
    f_native = 256. / np.tan(np.deg2rad(fov) / 2)
    estimated_k = np.array([[f_native, 0, 256], [0, f_native, 256], [0, 0, 1.]])
    # This maps camera rays only. Unequal singular values demonstrate why using
    # it as a 3D asset transform would deform the geometry rather than register it.
    ray_map = np.linalg.inv(raw_k) @ np.linalg.inv(ah_edge) @ estimated_k
    singular = np.linalg.svd(ray_map, compute_uv=False)
    focal_ratio = (ah_edge @ raw_k)[0, 0] / f_native
    return dict(id=sid, identity_key=ref['identity_key'], timestamp_us=ref['timestamp_us'],
                status='input_and_camera_trace_valid_registration_still_required',
                historical_camera=str(old_path), historical_camera_sha256=sha(old_path),
                current_camera=ref['camera_contract'], current_camera_sha256=ref['camera_contract_sha256'],
                historical_roi=old['roi_xyxy'], current_roi=current['roi_xyxy'],
                roi_changed_after_generation=old['roi_xyxy'] != current['roi_xyxy'],
                historical_mask_differs_from_reviewed_mask=row['exact_mask_sha256'] != ref['reference_mask_sha256'],
                raw_camera_unchanged=True, raw_to_generation_edge_affine=edge.tolist(),
                raw_to_generation_integer_center_affine=centers.tolist(),
                ah=dict(input_manifest=str(H / 'inputs/ah/jobs' / job_name / 'source_manifest.json'),
                        source_camera=str(camera_path), source_camera_sha256=sha(camera_path),
                        ply=str(ply), ply_sha256=validation['ply_sha256'],
                        source_direction=direction.tolist(), distance_prediction=distance,
                        lwh_prediction=lwh.tolist(), normalized_camera_distance=distance / float(max(lwh)),
                        fov_prediction_degrees=fov, K_prediction=estimated_k.tolist(),
                        raw_to_generation_edge_affine=ah_edge.tolist(),
                        raw_to_generation_integer_center_affine=ah_centers.tolist(),
                        K_raw_in_generation_edge_coordinates=(ah_edge @ raw_k).tolist(),
                        raw_crop_to_predicted_focal_ratio=float(focal_ratio),
                        ray_map_singular_values=singular.tolist(),
                        ray_map_is_similarity=bool(np.allclose(singular, singular[0], rtol=1e-5, atol=1e-7)),
                        registration_status='estimated source direction exists; actual-camera global registration unvalidated'),
                trellis_original=dict(native_payload=str(run / 'native_gaussians.pt'),
                                      native_payload_sha256=reload['output_sha256'],
                                      source_camera_estimate_present=False,
                                      preview_camera='fixed yaw 0 or 0.29670597, pitch 0.2, radius 2, fov 40 degrees',
                                      preview_camera_is_source_evidence=False,
                                      conditioning=trace,
                                      registration_status='semantic/root orientation evidence and global registration required'),
                reference_eligible=ref['reference_eligible'], common_camera_valid=False,
                metric_eligible=False, scores_produced=False)


def main():
    output = H / 'reports/static_source_camera_audit_v1'
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = H / 'inputs/human_static_native_v1/manifest.json'
    refs_path = H / 'configs/human_source_common_v1/reference_qualification_frozen_v1.json'
    manifest, refs = read(manifest_path), read(refs_path)
    checked(refs['selection'], refs['selection_sha256'])
    if manifest['selection_sha256'] != refs['selection_sha256']:
        raise ValueError('Selection mismatch')
    by_id = {r['id']: r for r in refs['objects']}
    rows = manifest['objects']
    if len(rows) != 100 or len({r['sample_id'] for r in rows}) != 100 or {r['sample_id'] for r in rows} != set(by_id):
        raise ValueError('Population must be the same frozen 100')
    preprocess = load_original_preprocessor()
    records = []
    for row in rows:
        try:
            record = audit_row(row, by_id[row['sample_id']], preprocess)
        except Exception as error:
            record = dict(id=row['sample_id'], status='audit_failed', reason=f'{type(error).__name__}: {error}',
                          common_camera_valid=False, metric_eligible=False)
        records.append(record)
        (output / (row['sample_id'] + '.json')).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    passed = [r for r in records if r['status'] != 'audit_failed']
    ratios = [r['ah']['raw_crop_to_predicted_focal_ratio'] for r in passed]
    report = dict(schema='static_source_camera_input_audit_v1', population=100, passed=len(passed),
                  failed=100-len(passed), failures=[r for r in records if r['status']=='audit_failed'],
                  historical_roi_changed_count=sum(r['roi_changed_after_generation'] for r in passed),
                  historical_mask_changed_count=sum(r['historical_mask_differs_from_reviewed_mask'] for r in passed),
                  ah_ray_maps_not_similarity=sum(not r['ah']['ray_map_is_similarity'] for r in passed),
                  ah_raw_crop_to_predicted_focal_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                  trellis_preprocessing_exact_pixel_matches=len(passed),
                  registration_accepted=0, metric_eligible=0, scores_produced=False,
                  script_sha256=sha(__file__), static_manifest_sha256=sha(manifest_path),
                  reference_qualification_sha256=sha(refs_path), original_preprocessor_sha256=sha(PIPELINE),
                  records={r['id']:dict(path=str(output/(r['id']+'.json')), sha256=sha(output/(r['id']+'.json'))) for r in records})
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, indent=2))
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
