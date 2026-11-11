"""Combine source-only visual reviews without freezing eligibility or changing inputs."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parents[1]
R = H / 'reports/reference_qualification_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    selection = H / 'inputs/benchmark_human100_v2/selection.json'
    assert sha(selection) == 'f6917e99f3e82d57b7ac2d141c50d8377aa953ec4ddb3e27192720d53af49f79'
    selected = {x['id']: x for x in read(selection)['objects']}
    assert len(selected) == 100
    rows, sources = {}, {}

    def source(path):
        sources[str(path)] = sha(path)
        return read(path)

    for path in sorted(R.glob('visual_review_batch*.json')):
        for item in source(path)['objects']:
            sid = item['id']
            assert sid in selected and sid not in rows
            assert item['timestamp_us'] == selected[sid]['t_ref_us']
            assert sha(item['raw']) == item['raw_sha256']
            assert sha(item['mask']) == item['mask_sha256']
            assert sha(item['evidence_sheet']) == item['evidence_sha256']
            rows[sid] = dict(id=sid, identity_key=item['identity_key'],
                timestamp_us=item['timestamp_us'], raw=item['raw'], raw_sha256=item['raw_sha256'],
                original_mask=item['mask'], original_mask_sha256=item['mask_sha256'],
                candidate_mask=item['mask'], candidate_mask_sha256=item['mask_sha256'],
                visual_status=item['visual_status'], note=item['visual_note'],
                review_history=[dict(record=str(path), status=item['visual_status'])],
                reference_ready=False, metric_eligible=False, scored=False)
    assert set(rows) == set(selected)

    def merge(path, items):
        for item in items:
            row = rows[item['id']]
            status = item.get('visual_status', item.get('status'))
            if item.get('correction_visual_passed'):
                status = 'correction_visual_passed_pending_reference_freeze'
            assert status
            if 'raw_sha256' in item:
                assert row['raw_sha256'] == item['raw_sha256']
            if 'base_mask_sha256' in item:
                assert row['original_mask_sha256'] == item['base_mask_sha256']
            if 'evidence' in item:
                assert sha(item['evidence']) == item['evidence_sha256']
            row['review_history'].append(dict(record=str(path), status=status))
            row['visual_status'] = status
            row['note'] = item.get('note', row['note'])
            for key in ('accessory_external_occlusion', 'external_occlusion_scope', 'final_eligibility_pending'):
                if key in item:
                    row[key] = item[key]
            # Rejected masks remain in their evidence folders, never promoted.
            if status == 'correction_visual_passed_pending_reference_freeze':
                assert sha(item['mask_path']) == item['mask_sha256']
                row['candidate_mask'] = item['mask_path']
                row['candidate_mask_sha256'] = item['mask_sha256']

    for rel in ['detail_review_v1/review.json', 'component_corrections_v1/visual_review.json',
                'detail_review_v1/HN0214_correction_review.json',
                'points_cpu_v2_review/visual_review.json', 'local_edits_v1_review/visual_review.json',
                'HN0861_accessory_union_v2_review.json', 'accessory_final_v3_visual_review.json',
                'temporal_context_visual_review_v1.json', 'remaining_four_visual_review_v1.json']:
        path = R / rel
        doc = source(path)
        if rel.startswith('component_'):
            assert sha(R/'component_corrections_v1/review.png') == doc['review_image_sha256']
        merge(path, doc['objects'] if 'objects' in doc else [doc])

    for row in rows.values():
        assert sha(row['original_mask']) == row['original_mask_sha256']
        assert sha(row['candidate_mask']) == row['candidate_mask_sha256']
        row['corrected_candidate'] = row['candidate_mask'] != row['original_mask']
        row['visual_passed'] = row['visual_status'] in (
            'visual_pass_pending_protocol_freeze', 'correction_visual_passed_pending_reference_freeze')
    pending = [x['id'] for x in rows.values() if not x['visual_passed'] and x['visual_status'] != 'ineligible_external_occlusion']
    occluded = [x['id'] for x in rows.values() if x['visual_status'] == 'ineligible_external_occlusion']
    result = dict(schema='reference_visual_candidate_status_v1',
        updated_at=datetime.now(timezone.utc).isoformat(), selection=str(selection), selection_sha256=sha(selection),
        selected=100, reviewed=100, visually_passed=sum(x['visual_passed'] for x in rows.values()),
        corrected_candidates=sum(x['corrected_candidate'] for x in rows.values()),
        pending_ids=sorted(pending), external_occlusion_ids=sorted(occluded),
        accessory_occlusion_requires_eligibility_review_ids=sorted(
            x['id'] for x in rows.values() if x.get('accessory_external_occlusion')),
        status_counts=dict(Counter(x['visual_status'] for x in rows.values())),
        formal_reference_ready=0, scores_produced=False, model_outputs_used=False,
        original_masks_unchanged=True, historical_generation_inputs_unchanged=True,
        scope='Candidate reference masks only. Freeze eligibility and common protocol separately; disclose differences from historical generation masks.',
        evidence_sha256=sources, objects=[rows[sid] for sid in sorted(rows)])
    output = R / 'current_candidates_v1.json'
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    tmp.replace(output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('objects', 'evidence_sha256')}))


if __name__ == '__main__':
    main()
