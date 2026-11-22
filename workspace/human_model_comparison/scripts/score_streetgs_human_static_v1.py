"""Score authentic StreetGS human exports with the existing final human protocol.

New output namespace only. Static appearance uses the accepted foreground-area
similarity; IoU and ED-P use the unwarped native render. Never invent a prediction
for a missing export, apply a reference mask to it, or alter old benchmark files.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
OUT = H / 'reports/streetgs_human_probe_v1'
EXPORT = H / 'exports/streetgs_human_probe_v1/objects'
METHOD = 'streetgs'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(p, digest):
    assert sha(p) == digest, 'Input hash differs: ' + str(p)


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def references():
    active = read(H / 'inputs/active_human_benchmark.json')
    checked(active['selection'], active['selection_sha256'])
    selection = {r['id']: r for r in read(active['selection'])['objects']}
    pointer = read(H / 'inputs/active_results.json')
    path = H / pointer['core_summary']
    summary = read(path)
    assert summary['selection_sha256'] == active['selection_sha256']
    refs = {}
    for row in summary['rows']:
        if not row.get('reference'):
            continue
        ref = row['reference']
        if row['id'] in refs:
            assert refs[row['id']] == ref, 'Conflicting method reference records'
        refs[row['id']] = ref
    assert set(refs) == set(selection) and len(selection) == 100
    for sid, ref in refs.items():
        assert ref['identity_key'] == selection[sid]['identity_key']
        assert ref['timestamp_us'] == selection[sid]['t_ref_us']
    return active, path, refs


def freeze():
    active, path, refs = references()
    align = R / 'reports/source_appearance_alignment_20261115_v1/run_comparison.py'
    code = [Path(__file__), H / 'scripts/human_source_metrics_v1.py', align,
            H / 'scripts/run_human_edp_common.py', H / 'scripts/human_edp_common.py']
    source = R / 'reports/source_appearance_alignment_20261115_v1/protocol.json'
    protocol = dict(
        version='streetgs_human_static_v1', method=METHOD,
        selection=active['selection'], selection_sha256=active['selection_sha256'],
        scoring_reference_summary=str(path), scoring_reference_summary_sha256=sha(path),
        ids=sorted(refs), reference_scope='Latest approved per-identity reference masks/ROI, same physical source cameras',
        appearance='Full-canvas own-alpha centroid translation and isotropic sqrt(Agt/Apred) scale, then existing GT ROI; no rotation or color fitting',
        silhouette='Unwarped prediction; same reference ROI and alpha byte>127',
        EDP='Unwarped common ROI, existing six-part DINOv3-L16 protocol',
        aggregation='Version A each method own valid identities; NA missingness, no zero imputation; empty valid rendered foreground remains scored',
        lpips=read(source)['lpips'], inherited_appearance_protocol=str(source),
        inherited_appearance_protocol_sha256=sha(source), code={str(p): sha(p) for p in code},
        old_scores_modified=False)
    p = OUT / 'static_protocol.json'
    if p.exists():
        assert read(p) == protocol, 'Frozen inputs changed; preserve this batch and create a new version'
    else:
        save(p, protocol)
    return protocol, refs


def validate(v, ref):
    """Check exported evidence; a passed render alone is not an independent asset."""
    assert v['status'] == 'passed'
    assert v.get('id', v.get('sample_id')) == ref['id']
    assert v['method'] == METHOD
    identity = v.get('identity_key') or 'nuscenes:' + v['actor_uuid']
    assert identity == ref['identity_key']
    assert v['timestamp_us'] == ref['timestamp_us']
    assert v['camera_contract_sha256'] == ref['camera_contract_sha256']
    checked(v['camera_contract'], v['camera_contract_sha256'])
    assert v['prediction_encoding'] == 'white_composited_uint8'
    assert v['reload_verified'] is True, 'Independent native asset reload has not passed'
    checked(v['bundle'], v['bundle_sha256'])
    for key in ['actor_preview', 'actor_alpha']:
        checked(v[key], v[key + '_sha256'])
    return v


def load_alignment():
    p = R / 'reports/source_appearance_alignment_20261115_v1/run_comparison.py'
    spec = importlib.util.spec_from_file_location('streetgs_shared_appearance_alignment', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(ids, device):
    import cv2
    import torch
    import human_source_metrics_v1 as core
    assert cv2.__version__ == '5.0.0', ('Frozen alignment runtime mismatch', cv2.__version__)
    protocol, refs = freeze()
    chosen = protocol['ids'] if not ids else ids
    assert len(chosen) == len(set(chosen)) and set(chosen) <= set(refs)
    align = load_alignment()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = protocol['lpips']
    metric = None
    for sid in chosen:
        q = refs[sid]
        vp = EXPORT / sid / 'validation.json'
        target = OUT / 'static_rows' / (sid + '.json')
        row = dict(id=sid, method=METHOD, reference=q,
                   protocol_sha256=sha(OUT / 'static_protocol.json'), metrics={})
        try:
            if not q['reference_eligible']:
                row.update(status='reference_ineligible', reason=q.get('na_reasons'))
            elif not vp.exists():
                row.update(status='export_pending', reason='No native export validation yet')
            else:
                v = read(vp)
                if v['status'] != 'passed':
                    row.update(status='native_export_unavailable', reason=v.get('reason', v['status']))
                else:
                    validate(v, q)
                    for key in ['reference_rgb', 'reference_mask', 'camera_contract']:
                        checked(q[key], q[key + '_sha256'])
                    signature = dict(validation_sha256=sha(vp), protocol_sha256=row['protocol_sha256'])
                    if target.exists() and read(target).get('status') == 'scored':
                        assert read(target)['signature'] == signature, 'Scored inputs changed; preserve old score first'
                        print(sid, 'cached', flush=True)
                        continue
                    raw = np.asarray(Image.open(q['reference_rgb']).convert('RGB'))
                    mask = np.asarray(Image.open(q['reference_mask']).convert('L'))
                    rgb = np.asarray(Image.open(v['actor_preview']).convert('RGB'))
                    alpha = np.asarray(Image.open(v['actor_alpha']).convert('L'))
                    assert raw.shape == rgb.shape and mask.shape == alpha.shape == raw.shape[:2]
                    assert list(rgb.shape[1::-1]) == read(q['camera_contract'])['render_wh']
                    if np.any(alpha == 0):
                        assert np.max(255 - rgb[alpha == 0]) <= 2, 'Nonwhite own-alpha background'
                    ref = core.prepare_reference(raw, mask)
                    assert ref['status'] == 'ready' and list(ref['roi']) == q['roi_xyxy']
                    pred = core.prepare_prediction(rgb, alpha, ref['roi'], v['prediction_encoding'])
                    transform, alignment = align.alignment(mask, alpha)
                    corrected = core.crop(align.warp(rgb, transform), ref['roi'])
                    if metric is None:
                        metric = core.AlexLPIPS(cfg['backbone'], cfg['linear'], cfg['backbone_sha256'], cfg['linear_sha256'], device)
                    metrics = dict(PSNR=core.psnr(ref['rgb'], corrected), SSIM=core.ssim(ref['rgb'], corrected),
                                   LPIPS=metric(ref['rgb'], corrected), IoU=core.iou(ref['mask'], pred['mask']),
                                   IoU_raw=core.iou(mask, alpha))
                    files = {}
                    for name, array in [('reference', ref['rgb']), ('reference_mask', ref['mask']),
                                        ('prediction_unwarped', pred['rgb']), ('prediction_mask', pred['mask']),
                                        ('prediction_aligned', corrected)]:
                        p = OUT / 'metric_images' / sid / (name + '.png')
                        p.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(array).save(p)
                        files[name] = str(p)
                        files[name + '_sha256'] = sha(p)
                    row.update(status='scored', metrics=metrics, signature=signature, files=files,
                               transform=transform.tolist(), alignment=alignment, source_validation=str(vp),
                               source_validation_sha256=sha(vp), real_heldout_eligible=False,
                               runtime=dict(executable=sys.executable, numpy=np.__version__, opencv=cv2.__version__))
        except Exception as error:
            # Never overwrite a successful row on a failed revalidation.
            if target.exists() and read(target).get('status') == 'scored':
                raise
            row.update(status='evaluation_error', reason=type(error).__name__ + ': ' + str(error))
        save(target, row)
        print(sid, row['status'], row.get('reason', ''), flush=True)
    summary()


def edp_prepare(ids, group):
    protocol, refs = freeze()
    old = H / read(H / 'inputs/active_results.json')['edp_root']
    oldcases = {c['key']: c for c in read(old / 'inputs.json')['cases']}
    root = OUT / 'edp' / group
    cases, valid = [], []
    for sid in ids or protocol['ids']:
        p = OUT / 'static_rows' / (sid + '.json')
        if not p.exists() or read(p)['status'] != 'scored':
            continue
        row = read(p)
        ref = oldcases[sid + '_reference']
        f = row['files']
        assert f['reference_sha256'] == ref['image_sha256']
        assert f['reference_mask_sha256'] == ref['mask_sha256']
        valid.append(sid)
        cases.extend([ref, dict(key=sid + '_' + METHOD, id=sid, method=METHOD,
            image=f['prediction_unwarped'], image_sha256=f['prediction_unwarped_sha256'],
            mask=f['prediction_mask'], mask_sha256=f['prediction_mask_sha256'],
            reference_quality_flags=ref['reference_quality_flags'],
            reference_excluded_parts=ref['reference_excluded_parts'],
            evidence=dict(score=str(p), score_sha256=sha(p)))])
    assert valid, 'No scored identities for ED-P'
    data = dict(ids=valid, cases=cases, protocol_sha256=sha(old / 'protocol.json'),
                selection_sha256=protocol['selection_sha256'])
    if (root / 'inputs.json').exists():
        assert read(root / 'inputs.json') == data, 'Different ED-P batch; choose a new --group'
        return
    for name in ['poses', 'features']:
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / 'protocol.json').write_bytes((old / 'protocol.json').read_bytes())
    save(root / 'inputs.json', data)


def edp_stage(action, group):
    import run_human_edp_common as original
    original.METHODS = [METHOD]
    getattr(original, action)(OUT / 'edp' / group)
    if action == 'score':
        summary()


def summary():
    protocol, refs = freeze()
    rows = {r['id']: r for p in (OUT / 'static_rows').glob('*.json') for r in [read(p)]}
    assert set(rows) <= set(refs)
    cells = {}
    for name in ['PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw']:
        ok = [r for r in rows.values() if r.get('metrics', {}).get(name, {}).get('status') == 'ok']
        inf = [r['id'] for r in rows.values() if r.get('metrics', {}).get(name, {}).get('status') == 'positive_infinity']
        cells[name] = dict(value=float(np.mean([r['metrics'][name]['value'] for r in ok])) if ok and not inf else None,
                           n=len(ok) + len(inf), ids=sorted([r['id'] for r in ok] + inf),
                           positive_infinity_ids=inf, extended_mean='infinity' if inf else None)
    edp = {}
    for p in (OUT / 'edp').glob('*/results.json'):
        for row in read(p)['rows']:
            assert row['id'] not in edp, 'Duplicate ED-P identity across batches'
            edp[row['id']] = row
    valid = [r for r in edp.values() if r['score']['status'] == 'ok']
    cells['EDP'] = dict(value=float(np.mean([r['score']['value'] for r in valid])) if valid else None,
                        n=len(valid), ids=sorted(r['id'] for r in valid),
                        valid_parts=sum(r['valid_part_count'] for r in valid))
    save(OUT / 'static_summary.json', dict(method=METHOD, expected=100,
         groups={METHOD: cells}, coverage=dict(collections.Counter(r['status'] for r in rows.values())),
         unvisited=sorted(set(refs) - set(rows)), edp_processed=len(edp), updated_at=time.time(),
         protocol_sha256=sha(OUT / 'static_protocol.json'), old_scores_modified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'score', 'summary', 'edp-prepare', 'edp-pose', 'edp-features', 'edp-score'])
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--group', default='all')
    args = parser.parse_args()
    assert args.group.replace('_', '').isalnum()
    if args.action == 'prepare':
        freeze()
    elif args.action == 'score':
        score(args.ids, args.device)
    elif args.action == 'summary':
        summary()
    elif args.action == 'edp-prepare':
        edp_prepare(args.ids, args.group)
    else:
        edp_stage(args.action.removeprefix('edp-'), args.group)
