"""Score declared 2D foreground postprocessing, preserving native 3D evidence.

The legacy metric reader calls its input file validation.json. Here that file
is explicitly a 2D scoring binding, never an independent Gaussian validation.
"""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import native_reaudit_common as common
import score_streetgs_human_static_v1 as original
import score_native_reaudit_static_v1 as shared

ROOT = common.ROOT / 'postprocess_v1'
METHOD = None
NATIVE_VALIDATE = shared._validate
STATUSES = {'postprocess_segmented', 'postprocess_no_person',
            'postprocess_empty_mask', 'postprocess_exception_fallback'}


@lru_cache(maxsize=1)
def contract():
    return common.freeze()


def prepare():
    c = contract()
    pp = ROOT / 'protocol.json'
    post = common.read(pp)
    assert post['selection_sha256'] == c['selection_sha256']
    assert post['ids'] == c['ids'] and post['expected'] == 300
    common.checked(post['input_manifest'], post['input_manifest_sha256'])
    source = c['inherited_protocols']['appearance']
    common.checked(source['path'], source['sha256'])
    paths = [Path(__file__), common.H / 'scripts/score_streetgs_human_static_v1.py',
             common.H / 'scripts/score_native_reaudit_static_v1.py',
             common.H / 'scripts/human_source_metrics_v1.py',
             common.H / 'scripts/human_edp_common.py',
             common.H / 'scripts/run_human_edp_common.py',
             common.R / 'reports/source_appearance_alignment_20261115_v1/run_comparison.py']
    protocol = dict(version='native_human_prediction_postprocess_static_v1',
        method=METHOD, ids=c['ids'], selection=c['selection'],
        selection_sha256=c['selection_sha256'],
        batch_contract=str(common.ROOT / 'batch_contract.json'),
        batch_contract_sha256=common.sha(common.ROOT / 'batch_contract.json'),
        postprocess_protocol=str(pp), postprocess_protocol_sha256=common.sha(pp),
        appearance='Unchanged own-alpha centroid translation and isotropic sqrt(Agt/Apred) scaling, then frozen ROI; no rotation/color fit',
        silhouette='Unwarped cleaned native prediction, same frozen ROI and alpha byte>127',
        EDP='Unwarped cleaned ROI, original six-part DINOv3-L16 protocol; predict new detector, pose and features',
        aggregation='Version A each method own metric-valid identities; valid empty predictions retained; ED-P NA not zero',
        lpips=common.read(source['path'])['lpips'],
        inherited_appearance_protocol=source['path'],
        inherited_appearance_protocol_sha256=source['sha256'],
        scoring_reference_summary=c['reference_summary'],
        scoring_reference_summary_sha256=c['reference_summary_sha256'],
        representation='2D postprocessing of independently verified native renders; not a changed Gaussian asset',
        geometry_changed=False, temporal_changed=False, reference_silhouette_erases_prediction=False,
        code={str(p): common.sha(p) for p in paths}, old_scores_modified=False)
    p = ROOT / 'metrics' / METHOD / 'static_protocol.json'
    if p.exists():
        assert common.read(p) == protocol, 'Changed frozen postprocess scoring protocol'
    else:
        common.save(p, protocol)
    return protocol, c['references']


def bind(ppath, ref):
    post = common.read(ppath)
    assert post['status'] in STATUSES and post['all_entries_retained_for_scoring']
    assert post['method'] == METHOD and post['id'] == ref['id']
    common.checked(post['protocol'], post['protocol_sha256'])
    assert Path(post['protocol']) == ROOT / 'protocol.json'
    common.checked(post['source_validation'], post['source_validation_sha256'])
    native = common.read(post['source_validation'])
    NATIVE_VALIDATE(native, ref)
    for key in ('identity_key', 'timestamp_us'):
        assert post[key] == native[key] == ref[key]
    assert post['roi_xyxy'] == ref['roi_xyxy']
    assert post['geometry_changed'] is False
    assert post['reference_mask_pixels_read'] is False
    assert post['original_source_photo_pixels_read'] is False
    assert post['gaussian_export_validation_claimed'] is False
    for pkey, nkey in [('original_rgb', 'actor_preview'), ('original_alpha', 'actor_alpha')]:
        assert post[pkey] == native[nkey] and post[pkey + '_sha256'] == native[nkey + '_sha256']
        common.checked(post[pkey], post[pkey + '_sha256'])
    f = post['outputs']
    for key in ('clean_rgb', 'clean_alpha', 'mask'):
        common.checked(f[key], f[key + '_sha256'])
    rgb = np.asarray(Image.open(post['original_rgb']).convert('RGB'))
    alpha = np.asarray(Image.open(post['original_alpha']).convert('L'))
    mask = np.asarray(Image.open(f['mask']).convert('L'))
    clean = np.asarray(Image.open(f['clean_rgb']).convert('RGB'))
    ca = np.asarray(Image.open(f['clean_alpha']).convert('L'))
    assert mask.shape == alpha.shape == ca.shape == rgb.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}
    keep = mask > 0
    assert np.array_equal(clean, np.where(keep[..., None], rgb, 255))
    assert np.array_equal(ca, alpha * keep)
    if post['status'] in {'postprocess_no_person', 'postprocess_empty_mask'}:
        assert not keep.any()
    if post['status'] == 'postprocess_exception_fallback':
        assert keep.all()
    return dict(schema='postprocessed_2d_scoring_input_v1', status='passed',
        id=ref['id'], method=METHOD, identity_key=ref['identity_key'],
        timestamp_us=ref['timestamp_us'], camera_contract_sha256=ref['camera_contract_sha256'],
        representation='2D postprocessing; native asset validation is separate',
        native_validation=post['source_validation'], native_validation_sha256=post['source_validation_sha256'],
        postprocess_record=str(ppath), postprocess_record_sha256=common.sha(ppath),
        postprocess_status=post['status'], gaussian_export_validation_claimed=False,
        prediction_encoding='white_composited_uint8', actor_preview=f['clean_rgb'],
        actor_preview_sha256=f['clean_rgb_sha256'], actor_alpha=f['clean_alpha'],
        actor_alpha_sha256=f['clean_alpha_sha256'])


def validate(v, ref):
    common.checked(v['postprocess_record'], v['postprocess_record_sha256'])
    assert bind(Path(v['postprocess_record']), ref) == v
    return v


def adapt_inputs(ids=None):
    ready = []
    for sid in ids or contract()['ids']:
        pp = ROOT / 'outputs' / METHOD / sid / 'postprocess.json'
        if not pp.exists():
            continue
        bound = bind(pp, contract()['references'][sid])
        p = original.EXPORT / sid / 'validation.json'
        if p.exists():
            assert common.read(p) == bound, 'Scoring input changed'
        else:
            common.save(p, bound)
        ready.append(sid)
    return ready


def main():
    global METHOD
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'score', 'edp-prepare', 'edp-pose', 'edp-features', 'edp-score', 'summary'])
    p.add_argument('--method', choices=common.METHODS, required=True)
    p.add_argument('--ids', nargs='+')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    METHOD = a.method
    original.METHOD = METHOD
    shared.METHOD = METHOD
    original.OUT = ROOT / 'metrics' / METHOD
    original.EXPORT = ROOT / 'metric_inputs' / METHOD
    original.freeze = prepare
    original.validate = validate
    if a.action == 'prepare':
        prepare()
    elif a.action == 'score':
        ready = adapt_inputs(a.ids)
        assert ready, 'Postprocessing outputs not ready'
        original.score(ready, a.device)
    elif a.action == 'edp-prepare':
        original.edp_prepare(a.ids, 'all')
        shared.reuse_reference_cache(original.OUT / 'edp' / 'all')
    elif a.action == 'summary':
        original.summary()
    else:
        shared.edp_stage(a.action.removeprefix('edp-'), 'all')


if __name__ == '__main__':
    main()
