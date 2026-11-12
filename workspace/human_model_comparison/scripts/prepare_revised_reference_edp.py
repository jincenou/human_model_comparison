"""Prepare revised static inputs for the unchanged, frozen ED-P implementation."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import human_metric_batch_v1 as b
import run_human_metric_pipeline_v1 as pipeline
import run_human_edp_common as edp


def prepare(source, ids, root):
    summary = b.read(source)
    assert len(ids) == len(set(ids)) and ids
    plan_path = summary['reference_revision_plan']
    b.verified(plan_path, summary['reference_revision_plan_sha256'])
    plan = b.read(plan_path)
    prior_path = source.parent/'prior_summary.json'
    b.verified(prior_path, plan['source_summary_sha256'])
    prior = b.read(prior_path)
    old_rows = {(r['id'], r['method']): r for r in prior['rows']}
    rows = {(r['id'], r['method']): r for r in summary['rows']}
    assert set(ids).issubset({c['id'] for c in plan['corrections']})
    root.mkdir(parents=True, exist_ok=False)
    for name in ['images', 'masks', 'poses', 'features']:
        (root/name).mkdir()
    frozen = b.read(b.C/'edp_protocol_frozen_v1.json')
    protocol_path = b.H/'reports/source_edp_common_v1/cohort_v1/protocol.json'
    b.verified(protocol_path, frozen['specification_sha256'])
    shutil.copyfile(protocol_path, root/'protocol.json')
    shutil.copyfile(source, root/'source_summary.json')
    cases = []
    for sid in ids:
        ref = rows[(sid, 'ours')]['reference']
        for k in ['reference_rgb', 'reference_mask', 'camera_contract']:
            b.verified(ref[k], ref[k+'_sha256'])
        reference = b.m.prepare_reference(np.array(Image.open(ref['reference_rgb']).convert('RGB')),
                                         np.array(Image.open(ref['reference_mask']).convert('L')))
        assert list(reference['roi']) == ref['roi_xyxy']
        images = [('reference', reference, dict(reference=ref))]
        for method in edp.METHODS:
            row = rows[(sid, method)]
            if row['status'] != 'scored':
                continue
            b.verified(row['result'], row['result_sha256'])
            result = b.read(row['result'])
            assert result['reference_mask_sha256'] == ref['reference_mask_sha256']
            assert result['roi_xyxy'] == ref['roi_xyxy']
            cert = pipeline.check_active_registration(old_rows[(sid, method)]['reference'], method, result)
            pred = b.m.prepare_prediction(np.array(Image.open(cert['source_rgb']).convert('RGB')),
                                         np.array(Image.open(cert['source_alpha']).convert('L')),
                                         reference['roi'], cert['prediction_encoding'])
            images.append((method, pred, dict(result=row['result'], result_sha256=row['result_sha256'],
                registration=result['registration_record'], registration_sha256=result['registration_sha256'])))
        for method, arrays, evidence in images:
            key = sid+'_'+method
            ip, mp = root/'images'/(key+'.png'), root/'masks'/(key+'.png')
            Image.fromarray(arrays['rgb']).save(ip); Image.fromarray(arrays['mask']).save(mp)
            flags = ref.get('reference_quality_flags', [])
            cases.append(dict(key=key, id=sid, method=method, image=str(ip), image_sha256=b.m.sha(ip),
                mask=str(mp), mask_sha256=b.m.sha(mp), evidence=evidence, reference_quality_flags=flags,
                reference_excluded_parts=['head'] if 'umbrella_occluded_input_state' in flags else []))
    b.save(root/'inputs.json', dict(ids=ids, cases=cases, summary_sha256=b.m.sha(root/'source_summary.json'),
        selection_sha256=summary['selection_sha256'], protocol_sha256=b.m.sha(root/'protocol.json'),
        preparation_script=str(Path(__file__).resolve()), preparation_script_sha256=b.m.sha(__file__),
        reference_revision_plan=str(plan_path), reference_revision_plan_sha256=b.m.sha(plan_path)))
    edp.verified_inputs(root)
    print(dict(prepared=len(cases), root=str(root), frozen_edp_protocol_unchanged=True), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--run-name', required=True)
    a = p.parse_args()
    assert a.run_name.replace('_', '').isalnum()
    prepare(a.summary.resolve(), a.ids, b.H/'reports/source_edp_common_v1'/a.run_name)
