"""Merge independently recomputed reference revisions; keep old snapshots intact."""
import copy
import shutil
from pathlib import Path
from PIL import Image
import human_metric_batch_v1 as b


def main():
    h = b.H
    core = h/'reports/static_reference_revision_v2/core/summary.json'
    summary = b.read(core)
    root = h/'reports/source_edp_common_v1/cohort_reference_revision_v2'
    root.mkdir(exist_ok=False)
    old = h/'reports/source_edp_common_v1/cohort_v1'
    inputs = b.read(old/'inputs.json'); results = b.read(old/'results.json')
    b.verified(old/'inputs.json', results['inputs_sha256'])
    frozen = b.read(b.C/'edp_protocol_frozen_v1.json')
    b.verified(old/'protocol.json', frozen['specification_sha256'])
    shutil.copyfile(old/'protocol.json', root/'protocol.json')
    shutil.copyfile(core, root/'source_summary.json')
    cases = {c['key']: c for c in inputs['cases']}
    rows = {(r['id'], r['method']): r for r in results['rows']}
    prior_cases = copy.deepcopy(cases); prior_rows = copy.deepcopy(rows)
    image = Image.open(old/'part_review.jpg').convert('RGB')
    evidence = []; revised_ids = set()
    for name in ['static_reference_revision_v1', 'static_reference_revision_v2']:
        folder = h/'reports/source_edp_common_v1'/name
        b.verified(folder/'protocol.json', frozen['specification_sha256'])
        item = b.read(folder/'results.json'); manifest = b.read(folder/'inputs.json')
        b.verified(folder/'inputs.json', item['inputs_sha256'])
        b.verified(folder/'source_summary.json', manifest['summary_sha256'])
        assert not revised_ids.intersection(manifest['ids'])
        revised_ids.update(manifest['ids'])
        for c in manifest['cases']:
            for k in ['image', 'mask']: b.verified(c[k], c[k+'_sha256'])
            feature = folder/'features'/(c['key']+'.npz')
            meta = b.read(feature.with_suffix('.json'))
            b.verified(feature, meta['features_sha256'])
            b.verified(folder/'poses'/(c['key']+'.json'), meta['pose_sha256'])
            cases[c['key']] = c
        for row in item['rows']:
            rows[row['id'], row['method']] = row
        patch = Image.open(folder/'part_review.jpg').convert('RGB')
        for i, sid in enumerate(manifest['ids']):
            image.paste(patch.crop((0, i*250, 1456, (i+1)*250)),
                        (0, inputs['ids'].index(sid)*250))
        evidence.append(dict(root=str(folder), results_sha256=b.m.sha(folder/'results.json'),
                             inputs_sha256=b.m.sha(folder/'inputs.json'),
                             visual_review='All revised person/method part overlays viewed; model defects retained'))
    assert revised_ids == {'HN0116', 'HN0632', 'HN0838'}
    core_rows = {(r['id'], r['method']): r for r in summary['rows'] if r['status'] == 'scored'}
    assert set(rows) == set(core_rows) and len(rows) == 526
    for key, row in core_rows.items():
        case = cases[key[0]+'_'+key[1]]
        assert case['evidence']['result_sha256'] == row['result_sha256']
        b.verified(row['result'], row['result_sha256'])
        if key[0] not in revised_ids:
            assert rows[key] == prior_rows[key]
            assert case == prior_cases[case['key']]
    inputs.update(cases=[cases[c['key']] for c in inputs['cases']],
                  summary_sha256=b.m.sha(root/'source_summary.json'), revision_sources=evidence)
    b.save(root/'inputs.json', inputs)
    results.update(rows=[rows[(r['id'], r['method'])] for r in results['rows']],
                   inputs_sha256=b.m.sha(root/'inputs.json'), revision_sources=evidence,
                   status='reference_revisions_merged_with_unchanged_remaining_results')
    b.save(root/'results.json', results)
    image.save(root/'part_review.jpg', quality=94)
    b.save(h/'inputs/active_results.json', dict(
        core_summary=str(core.relative_to(h)), edp_root=str(root.relative_to(h)),
        gallery='reports/human_comparison_full_v2', advisor='reports/advisor_report_20261112_v2',
        policy='Approved reference revisions only; historical cohort v3 and advisor v1 remain immutable. Raw camera K/T, registration and model assets are unchanged.',
        corrected_reference_ids=sorted(revised_ids)))
    print(dict(merged_pairs=len(rows), revised_ids=sorted(revised_ids)), flush=True)


if __name__ == '__main__': main()
