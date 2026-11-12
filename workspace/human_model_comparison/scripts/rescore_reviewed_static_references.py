"""Version a reference correction without changing assets or camera registration.

The original camera contract remains the authority for raw K/T and timestamp.
The explicitly reviewed correction is the authority for the evaluation mask/ROI.
Old results are reproduced first; every already-scored method is then rescored.
No active benchmark pointer or historical result is overwritten.
"""
import argparse
import copy
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import human_metric_batch_v1 as b
import run_human_metric_pipeline_v1 as pipeline


def run(plan_path, out):
    plan = b.read(plan_path)
    b.verified(plan['source_summary'], plan['source_summary_sha256'])
    summary = b.read(plan['source_summary'])
    original = copy.deepcopy(summary)
    b.verified(summary['protocol'], summary['protocol_sha256'])
    protocol = b.read(summary['protocol'])
    b.verified(b.m.__file__, protocol['code_sha256'])
    assert plan['prediction_inputs_used_for_reference_review'] is False
    assert len({x['id'] for x in plan['corrections']}) == len(plan['corrections'])
    if shutil.disk_usage(b.H).free < 10 * 1024**3:
        raise RuntimeError('Data disk below 10 GiB')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out/'correction_plan.json')
    shutil.copyfile(plan['source_summary'], out/'prior_summary.json')
    import torch
    torch.set_num_threads(4)
    cfg = protocol['lpips']
    lpips = b.m.AlexLPIPS(cfg['backbone'], cfg['linear'],
                        cfg['backbone_sha256'], cfg['linear_sha256'], 'cpu')
    changes = []
    for correction in plan['corrections']:
        assert correction['visual_review_status'] == 'accepted_exact_static_exposure'
        sid = correction['id']
        rows = [r for r in summary['rows'] if r['id'] == sid]
        assert len(rows) == len(summary['coverage'])
        old_ref = rows[0]['reference']
        assert all(r['reference'] == old_ref for r in rows)
        for key in ['reference_rgb', 'reference_mask', 'camera_contract']:
            b.verified(old_ref[key], old_ref[key+'_sha256'])
        b.verified(correction['dynamic_qualification'], correction['dynamic_qualification_sha256'])
        q = b.read(correction['dynamic_qualification'])
        assert q['id'] == sid and q['status'] == 'reviewed_exact_exposure_references'
        frames = [f for f in q['frames'] if f['timestamp_us'] == old_ref['timestamp_us']]
        assert len(frames) == 1
        frame = frames[0]
        assert frame['source_rgb_sha256'] == old_ref['reference_rgb_sha256']
        for key in ['source_rgb', 'mask']:
            b.verified(frame[key], frame[key+'_sha256'])
        raw = np.array(Image.open(old_ref['reference_rgb']).convert('RGB'))
        old_mask = np.array(Image.open(old_ref['reference_mask']).convert('L'))
        mask = np.array(Image.open(frame['mask']).convert('L'))
        assert mask.shape == old_mask.shape == raw.shape[:2]
        ref_arrays = b.m.prepare_reference(raw, mask)
        assert ref_arrays['status'] == 'ready'
        folder = out/'media'/sid
        folder.mkdir(parents=True)
        mask_path = folder/'reference_raw_mask.png'
        shutil.copyfile(frame['mask'], mask_path)
        for name in ['rgb', 'mask']:
            Image.fromarray(ref_arrays[name]).save(folder/('reference_'+name+'.png'))
        ref = copy.deepcopy(old_ref)
        ref.update(reference_mask=str(mask_path), reference_mask_sha256=b.m.sha(mask_path),
                   roi_xyxy=list(ref_arrays['roi']),
                   reference_pixel_height=int(np.ptp(np.where(mask > 127)[0]))+1,
                   reference_revision=str(out/'correction_plan.json'),
                   reference_revision_sha256=b.m.sha(out/'correction_plan.json'),
                   evaluation_roi_authority='reviewed mask revision; camera contract retains historical ROI',
                   visual_note=correction['visual_note'])
        for row in rows:
            row['reference'] = ref
            if row['status'] != 'scored':
                continue  # Technical missingness is unchanged, never replaced by white images.
            b.verified(row['result'], row['result_sha256'])
            prior = b.read(row['result'])
            entry = dict(reference=old_ref, id=sid, method=row['method'], input_key=prior['input_key'],
                         registration_record=prior['registration_record'],
                         registration_sha256=prior['registration_sha256'])
            reproduced, _, _ = pipeline.score_active_pair(entry, lpips)
            for name in ['PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw']:
                assert reproduced['metrics'][name]['status'] == prior['metrics'][name]['status']
                if prior['metrics'][name]['status'] == 'ok':
                    np.testing.assert_allclose(reproduced['metrics'][name]['value'],
                                               prior['metrics'][name]['value'], rtol=0, atol=1e-6)
            cert = pipeline.check_active_registration(old_ref, row['method'], entry)
            rgb = np.array(Image.open(cert['source_rgb']).convert('RGB'))
            alpha = np.array(Image.open(cert['source_alpha']).convert('L'))
            assert rgb.shape == raw.shape and alpha.shape == mask.shape
            pred = b.m.prepare_prediction(rgb, alpha, ref_arrays['roi'], cert['prediction_encoding'])
            scores = dict(PSNR=b.m.psnr(ref_arrays['rgb'], pred['rgb']),
                          SSIM=b.m.ssim(ref_arrays['rgb'], pred['rgb']),
                          LPIPS=lpips(ref_arrays['rgb'], pred['rgb']),
                          IoU=b.m.iou(ref_arrays['mask'], pred['mask']), IoU_raw=b.m.iou(mask, alpha))
            scores['ED-P'] = b.m.na('reference_revision_requires_feature_recomputation')
            key = b.digest(dict(prior_result_sha256=row['result_sha256'], reference=ref,
                                script_sha256=b.m.sha(__file__)))
            result = dict(prior, input_key=key, metrics=scores,
                          roi_xyxy=ref['roi_xyxy'], reference_pixel_height=ref['reference_pixel_height'],
                          reference_mask_sha256=ref['reference_mask_sha256'],
                          reference_revision=ref['reference_revision'],
                          reference_revision_sha256=ref['reference_revision_sha256'],
                          evaluation_roi_authority=ref['evaluation_roi_authority'],
                          prior_result=row['result'], prior_result_sha256=row['result_sha256'],
                          old_scores_reproduced_atol=1e-6, raw_camera_and_predictions_unchanged=True,
                          execution_script_sha256=b.m.sha(__file__), lpips=lpips.provenance)
            result_path = out/'results'/row['method']/sid/(key+'.json')
            b.save(result_path, result)
            for name, array in [('rgb', pred['rgb']), ('mask', pred['mask'])]:
                Image.fromarray(array).save(folder/(row['method']+'_'+name+'.png'))
            changes.append(dict(id=sid, method=row['method'], before=prior['metrics'], after=scores))
            row.update(result=str(result_path), result_sha256=b.m.sha(result_path), input_key=key, metrics=scores)
        b.save(folder/'reference_revision.json', dict(reference=ref, prior_reference=old_ref,
            added_pixels=int(((mask > 127) & (old_mask <= 127)).sum()),
            removed_pixels=int(((mask <= 127) & (old_mask > 127)).sum()), correction=correction))
    by_method = {m: {} for m in summary['coverage']}
    for row in summary['rows']:
        if row['status'] == 'scored':
            b.verified(row['result'], row['result_sha256'])
            by_method[row['method']][row['id']] = b.read(row['result'])
    ids = sorted({r['id'] for r in summary['rows']})
    for group, metrics in summary['pairings'].items():
        for name in metrics:
            summary['pairings'][group][name] = b.m.paired_summary(
                {m: by_method[m] for m in group.split('+')}, name, ids)
    assert [(r['id'], r['method'], r['status']) for r in summary['rows']] == [
        (r['id'], r['method'], r['status']) for r in original['rows']]
    summary.update(reference_revision_plan=str(out/'correction_plan.json'),
                   reference_revision_plan_sha256=b.m.sha(out/'correction_plan.json'),
                   revision_status='core_rescored; ED-P and publication integration pending')
    b.save(out/'summary.json', summary)
    b.save(out/'changes.json', changes)
    with (out/'metrics.csv').open('w', newline='') as f:
        fields = ['id', 'method', 'status', 'reason', 'result', 'result_sha256',
                  'PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in summary['rows']:
            record = {k: row.get(k) for k in fields[:6]}
            record.update({k: row.get('metrics', {}).get(k, {}).get('value') for k in fields[6:]})
            writer.writerow(record)
    b.verified(plan['source_summary'], plan['source_summary_sha256'])
    print(dict(rescored=len(changes), output=str(out), active_benchmark_unchanged=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.plan, args.output.resolve())
