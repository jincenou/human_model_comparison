"""Observed-surface geometry of independently reloaded StreetGS human subsets.

This adds one method without changing old references, scores, or protocols.
Background subsets remain static: no GT trajectory or pose deformation is added.
The height-conditioned and native-metric panels exactly follow the existing
human scene-Gaussian evaluation, including source-camera-centred scale changes.
"""
import argparse
import collections
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
HG = H / 'evaluation_static_cd_fscore_time_v1'
G = R / 'evaluation_geometry_expanded_20261114_v2'
EXPORT = H / 'exports/streetgs_human_probe_v1/objects'
OUT = H / 'reports/streetgs_human_geometry_v1'
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(HG / 'scripts'), str(G / 'scripts'), str(R / 'scripts'),
               str(R / 'evaluation_nuscenes_v1/scripts'), str(H / 'scripts')]
from common import read, sha, checked, save, rigid_part, selection
from scene_depth import SceneDepth
from pilot import measure
from depth import query
from geometry_core import metrics, selftest


def prepare():
    """Bind to the unchanged 100-person reference and original 84-person panel."""
    old = read(HG / 'protocol.json')
    checked(HG / 'reference_manifest.json', old['reference_manifest_sha256'])
    for path, digest in old['code'].items():
        checked(path, digest)
    _, frozen = selection()
    refs = read(HG / 'reference_manifest.json')
    assert refs['selection_sha256'] == old['selection_sha256']
    assert {r['id'] for r in refs['rows']} == set(frozen)
    assert collections.Counter(r['candidate_panel'] for r in refs['rows']) == {
        'main': 84, 'low_support_exploratory': 16}
    code = {**old['code'], str(SCRIPT): sha(SCRIPT)}
    for name in ['common.py', 'evaluate.py']:
        path = HG / 'scripts' / name
        code[str(path)] = sha(path)
    protocol = dict(
        version='streetgs_human_background_observed_surface_v1',
        method='streetgs', ids=sorted(frozen),
        selection_sha256=old['selection_sha256'],
        reference_manifest=str(HG / 'reference_manifest.json'),
        reference_manifest_sha256=sha(HG / 'reference_manifest.json'),
        inherited_protocol=str(HG / 'protocol.json'),
        inherited_protocol_sha256=sha(HG / 'protocol.json'),
        rules={k: old[k] for k in ['reference', 'reference_qualification',
            'main_panel', 'scale_rule', 'annotation_information', 'prohibited',
            'native_extent', 'depth', 'metrics']},
        export_root=str(EXPORT),
        input='Independent original Background Gaussian subset, unchanged raw '
              'positions/quaternions/log-scales/opacity-logits/SH; no new motion',
        renderer='Retained DriveStudio rasterization, SceneDepth omnire backend '
                 'reused because StreetGS uses the identical renderer',
        aggregation='Own-valid per fixed reference panel; person equal weight; '
                    'valid empty F-score=0, CD=null and extended mean=infinity; '
                    'technical failures separately counted',
        old_scores_modified=False, code=code)
    path = OUT / 'protocol.json'
    if path.exists():
        existing = read(path)
        for key, value in protocol.items():
            assert existing[key] == value, ('Protocol changed', key)
        return existing
    protocol.update(created_at=time.time(), selftest=selftest())
    save(path, protocol)
    return protocol


def load_export(sid, ref):
    """No original checkpoint is needed for rendering, only the certified subset."""
    import torch
    folder = EXPORT / sid
    validation_path = folder / 'validation.json'
    validation = read(validation_path)
    assert validation['status'] == 'passed' and validation['reload_verified'] is True
    path = folder / 'native_background_subset.pt'
    assert Path(validation['bundle']).resolve() == path.resolve()
    checked(path, validation['bundle_sha256'])
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    assert bundle.get('id', bundle.get('sample_id')) == sid
    assert bundle['identity_key'] == ref['identity_key']
    assert validation['timestamp_us'] == validation['frozen_raw_timestamp_us'] == ref['t_ref_us']
    assert validation['camera_contract_sha256'] == ref['camera_contract_sha256']
    assert bundle['camera']['timestamp_us'] == ref['t_ref_us']
    assert bundle['camera']['identity_key'] == ref['identity_key']
    assert bundle['camera'] == read(ref['camera_contract'])
    raw = bundle['canonical']
    count = len(raw['_means'])
    assert count > 0, 'Empty Gaussian payload is not a rendered empty prediction'
    for key in ['_means', '_quats', '_scales', '_opacities', '_features_dc', '_features_rest']:
        assert len(raw[key]) == count and bool(torch.isfinite(raw[key]).all()), key
    assert raw['_means'].shape == (count, 3)
    assert raw['_scales'].shape in [(count, 3), (count, 1)]
    assert raw['_quats'].shape == (count, 4)
    norm = torch.linalg.vector_norm(raw['_quats'], dim=-1, keepdim=True)
    assert bool((norm > 0).all())
    scales = raw['_scales'].exp()
    if scales.shape[1] == 1:
        scales = scales.repeat(1, 3)
    arrays = dict(means=raw['_means'].float().contiguous(),
                  quats=(raw['_quats'] / norm).float().contiguous(),
                  scales=scales.float().contiguous(),
                  opacity=raw['_opacities'].sigmoid().float().reshape(-1).contiguous())
    assert all(bool(torch.isfinite(a).all()) for a in arrays.values())
    assert bool((arrays['scales'] > 0).all())
    Tworld = np.asarray(bundle['T_world_from_native'], dtype=float)
    assert Tworld.shape == (4, 4)
    _, gauge = rigid_part(Tworld)
    assert abs(gauge - 1.) < 1e-6, 'Background input must retain native metric units'
    np.testing.assert_allclose(Tworld[3], [0, 0, 0, 1], atol=1e-8)
    cfg = bundle['config']['trainer']['render']
    for key in ['packed', 'absgrad', 'sparse_grad', 'antialiased']:
        assert key in cfg, key
    # SceneDepth chooses the native DriveStudio rasterizer using this backend
    # name; all published provenance and result labels remain StreetGS.
    pack = dict(arrays=arrays, method='omnire', render_cfg=cfg,
                T_world_from_native=Tworld)
    provenance = dict(bundle=str(path), bundle_sha256=sha(path),
                      validation=str(validation_path), validation_sha256=sha(validation_path),
                      raw_gaussian_count=count, identity_key=ref['identity_key'],
                      renderer_backend='DriveStudio/SceneDepth',
                      input_parameters_modified=False, motion_added=False)
    return bundle, validation, pack, provenance


def alpha_equivalence(adapter, sourceT, camera, validation, out):
    """Use the rectangular source camera; a square crop is not equivalent."""
    torch = adapter.torch
    alpha_path = Path(validation['actor_alpha'])
    checked(alpha_path, validation['actor_alpha_sha256'])
    width, height = camera['render_wh']
    original = np.asarray(Image.open(alpha_path).convert('L'))
    assert original.shape == (height, width)
    with torch.inference_mode():
        _, alpha, _ = adapter.rasterization(
            *adapter.arrays, torch.zeros_like(adapter.arrays[0]),
            viewmats=adapter.tensor(sourceT)[None], Ks=adapter.tensor(camera['K_raw'])[None],
            width=width, height=height,
            near_plane=adapter.pack['render_cfg']['near_plane'],
            far_plane=adapter.pack['render_cfg']['far_plane'],
            render_mode='RGB+ED', **adapter.options)
    a = alpha[0, ..., 0].detach().cpu().numpy()
    diff = abs((a * 255).round().clip(0, 255) - original.astype(float))
    check = dict(max=float(diff.max()), mean=float(diff.mean()),
                 source_alpha=str(alpha_path), source_alpha_sha256=sha(alpha_path),
                 source_camera_rectangular=True)
    save(out / 'source_alpha_check.json', check)
    assert check['max'] <= 2 and check['mean'] <= .01, ('Native source alpha mismatch', check)
    return check


def run(sid):
    protocol = prepare()
    assert sid in protocol['ids']
    ref = next(r for r in read(HG / 'reference_manifest.json')['rows'] if r['id'] == sid)
    out = OUT / 'runs' / sid
    rp = OUT / 'results' / (sid + '.json')
    if rp.exists() and read(rp)['status'] == 'completed':
        previous = read(rp)
        assert previous['protocol_sha256'] == sha(OUT / 'protocol.json')
        assert previous['queries_sha256'] == ref['queries_sha256']
        for key in ['certificate', 'returns']:
            checked(previous[key], previous[key + '_sha256'])
        return previous
    out.mkdir(parents=True, exist_ok=True)
    if rp.exists():
        save(out / ('previous_result_' + sha(rp) + '.json'), read(rp))
    rec = dict(id=sid, method='streetgs', identity_key=ref['identity_key'],
               t_ref_us=ref['t_ref_us'], reference_panel=ref['candidate_panel'],
               protocol_sha256=sha(OUT / 'protocol.json'),
               queries_sha256=ref['queries_sha256'], status='running',
               started_at=time.time(), panels={})
    save(rp, rec)
    try:
        assert shutil.disk_usage(H).free > 10 * 1024**3
        checked(ref['queries'], ref['queries_sha256'])
        checked(ref['camera_contract'], ref['camera_contract_sha256'])
        camera = read(ref['camera_contract'])
        Twc = np.asarray(camera['T_world_from_camera_opencv'])
        with np.load(ref['queries']) as z:
            arrays = {k: z[k] for k in z.files}
        bundle, validation, pack, evidence = load_export(sid, ref)
        adapter = SceneDepth(pack)
        Tworld = np.asarray(pack['T_world_from_native'])
        sourceT = np.linalg.inv(Twc) @ Tworld
        alpha_equivalence(adapter, sourceT, camera, validation, out)
        rng, al, hit, diag = query(adapter, arrays, Tworld)
        met, _, _, _ = metrics(arrays['q'], arrays['o'], arrays['u'], arrays['r'], rng, hit)
        np.savez_compressed(out / 'native_metric_returns.npz', range_pred=rng, alpha=al, hit=hit)
        rec['panels']['native_metric'] = dict(
            metrics=met, T_world_from_native=Tworld.tolist(), diagnostics=diag,
            returns=str(out / 'native_metric_returns.npz'),
            returns_sha256=sha(out / 'native_metric_returns.npz'))
        V, oldscale = rigid_part(sourceT)
        axes = Twc[:3, :3] @ V[:3, :3]
        extent = measure(adapter, axes, out / 'extent')
        if extent['status'] != 'passed':
            rec.update(status='extent_numerically_unresolved',
                       reason='Common native-alpha vertical extent convergence not reached',
                       extent=str(out / 'extent/measurement.json'))
        else:
            n = str(extent['precision']['fine_resolution'])
            height = extent['measurements'][n]['vertical_extent']
            scale = ref['annotation_vertical_extent_m'] / height
            assert np.isfinite(scale) and scale > 0
            A = Twc @ np.diag([scale, scale, scale, 1.]) @ V
            assert abs(scale * height - ref['annotation_vertical_extent_m']) < 1e-9
            cert = dict(
                id=sid, method='streetgs', identity_key=ref['identity_key'],
                t_ref_us=ref['t_ref_us'], protocol_sha256=rec['protocol_sha256'],
                reference_manifest_sha256=sha(HG / 'reference_manifest.json'),
                queries_sha256=ref['queries_sha256'], evidence=evidence,
                camera_contract_sha256=ref['camera_contract_sha256'],
                T_source_camera_from_native=sourceT.tolist(),
                source_similarity_gauge_removed=oldscale, T_world_from_native=A.tolist(),
                scale_m_per_native_unit=scale, native_vertical_extent=height,
                annotation_vertical_extent_m=ref['annotation_vertical_extent_m'],
                extent=str(out / 'extent/measurement.json'),
                extent_sha256=sha(out / 'extent/measurement.json'),
                translation_scaled=True,
                full_covariance_transformation='s^2 R Sigma R^T via equivalent camera',
                root_pose_edited=False, LiDAR_endpoint_fitting=False,
                added_geometry_pruning=False, motion_added=False)
            save(out / 'metric_scale_certificate.json', cert)
            rng, alpha, hit, diag = query(adapter, arrays, A)
            met, points, dpg, dgp = metrics(arrays['q'], arrays['o'], arrays['u'], arrays['r'], rng, hit)
            np.savez_compressed(out / 'returns.npz', range_pred=rng, alpha=alpha, hit=hit,
                                predicted_endpoints=points, distances_pg=dpg, distances_gp=dgp)
            rec['panels']['annotation_vertical_extent_conditioned'] = dict(metrics=met, diagnostics=diag)
            rec.update(status='completed', certificate=str(out / 'metric_scale_certificate.json'),
                       certificate_sha256=sha(out / 'metric_scale_certificate.json'),
                       returns=str(out / 'returns.npz'), returns_sha256=sha(out / 'returns.npz'))
        del adapter, pack, bundle
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as error:
        rec.update(status='technical_failure_pending_review', reason=repr(error),
                   traceback=traceback.format_exc())
    rec['finished_at'] = time.time()
    save(rp, rec)
    print(sid, rec['status'], rec.get('reason'), flush=True)
    return rec


def summary():
    protocol = prepare()
    rows = [read(p) for p in sorted((OUT / 'results').glob('*.json'))]
    assert len({r['id'] for r in rows}) == len(rows)
    assert all(r['id'] in protocol['ids'] and r['protocol_sha256'] == sha(OUT / 'protocol.json') for r in rows)
    panels = {}
    for panel in ['main', 'low_support_exploratory']:
        panels[panel] = {}
        for measurement in ['annotation_vertical_extent_conditioned', 'native_metric']:
            pairs = [(r, r['panels'][measurement]['metrics']) for r in rows
                     if r['reference_panel'] == panel and measurement in r.get('panels', {})]
            finite = [(r, m) for r, m in pairs if m['cd_m'] is not None]
            empty = sorted(r['id'] for r, m in pairs if m['status'] == 'empty_pred')
            cells = dict(CD_cm=dict(
                value=float(np.mean([m['cd_m'] for r, m in finite]) * 100) if finite else None,
                n=len(finite), ids=sorted(r['id'] for r, m in finite), unit='cm',
                valid_empty_ids=empty, valid_empty_n=len(empty),
                extended_mean='infinity' if empty else 'finite' if finite else None))
            for metric, key in [('F5', 'f1_5cm'), ('F10', 'f1_10cm'),
                                ('F20', 'f1_20cm'), ('RayHit', 'ray_hit')]:
                cells[metric] = dict(value=float(np.mean([m[key] for r, m in pairs]) * 100) if pairs else None,
                                     n=len(pairs), ids=sorted(r['id'] for r, m in pairs),
                                     unit='percent', valid_empty_n=len(empty))
            panels[panel][measurement] = cells
    counts = dict(collections.Counter(r['status'] for r in rows))
    result = dict(method='streetgs', status='incremental',
                  main_panel='main / annotation_vertical_extent_conditioned',
                  groups=panels, coverage=counts, expected=100,
                  unattempted_ids=sorted(set(protocol['ids']) - {r['id'] for r in rows}),
                  updated_at=time.time(), protocol_sha256=sha(OUT / 'protocol.json'),
                  old_scores_modified=False)
    if len(rows) == 100 and not any(r['status'] in ['running', 'technical_failure_pending_review'] for r in rows):
        result['status'] = 'completed'
    save(OUT / 'summary.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'run', 'summary', 'selftest'])
    ap.add_argument('--ids', nargs='+')
    args = ap.parse_args()
    if args.action == 'selftest':
        print(json.dumps(selftest(), indent=2))
    elif args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.action == 'summary':
        print(json.dumps(summary(), indent=2))
    else:
        assert args.ids, 'Use explicit frozen IDs; GPU/resource leases are held by the caller'
        for sid in args.ids:
            run(sid)
        summary()


if __name__ == '__main__':
    main()
