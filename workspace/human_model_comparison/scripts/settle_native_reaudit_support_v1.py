"""Verify all 300 final support bindings and distinguish candidate counts.

This does not approve geometry or publication; those have separate completion
checks. It never mutates a passed validation or any historical asset.
"""
import collections
from pathlib import Path

import native_reaudit_common as common


def main():
    contract = common.freeze()
    root = common.ROOT / 'resolved'
    rows, sources = [], {}
    for method in common.METHODS:
        choice_path = root / 'decisions' / (method + '.json')
        choices = common.read(choice_path)
        assert choices['expected'] == 100
        common.checked(choices['source_review'], choices['source_review_sha256'])
        sources[str(choice_path)] = common.sha(choice_path)
        decisions = {x['id']: x for x in choices['rows']}
        assert set(decisions) == set(contract['ids'])
        for sid in contract['ids']:
            p = root / 'exports' / method / sid / 'validation.json'
            v = common.read(p)
            assert v['status'] == 'passed' and v['reload_verified'] is True
            assert v['method'] == method and v.get('id', v.get('sample_id')) == sid
            assert v['source_reference'] == contract['references'][sid]
            assert v['selected_support'] == decisions[sid]['selected_support']
            common.checked(v['resolution'], v['resolution_sha256'])
            common.checked(v['candidate_validation'], v['candidate_validation_sha256'])
            candidate = common.read(v['candidate_validation'])
            assert candidate['bundle'] == v['bundle'] and candidate['bundle_sha256'] == v['bundle_sha256']
            for key in ['bundle', 'actor_preview', 'actor_alpha', 'camera_contract']:
                common.checked(v[key], v[key + '_sha256'])
            if method == 'streetgs':
                total, actor = v['selected_count'], 0
            elif method == 'omnire':
                total, actor = v['exported_gaussians'], v['original_actor_gaussians']
            else:
                total, actor = v['selected_gaussians'], v['actor_gaussians']
            selected = actor if v['selected_support'] == 'native_actor_only' else total
            assert selected > 0 and 0 <= actor <= total
            if v['selected_support'] == 'native_actor_only':
                assert v['actor_node_present']
                assert v['actor_only_reload']['rgb_max_byte_difference'] <= 1
                assert v['actor_only_reload']['independent_of_full_scene_checkpoint']
                if v.get('old_actor_replay'):
                    assert v['old_actor_replay']['alpha_max_byte_difference'] <= 1
            rows.append(dict(id=sid, method=method, identity_key=v['identity_key'], timestamp_us=v['timestamp_us'],
                validation=str(p), validation_sha256=common.sha(p), candidate_validation=v['candidate_validation'],
                selected_support=v['selected_support'], candidate_gaussian_count=total,
                selected_gaussian_count=selected, native_actor_gaussian_count=actor,
                selected_background_gaussian_count=selected-actor,
                original_bundle_reused_by_reference=True, gaussian_values_modified=False,
                semantic_decision=decisions[sid], metric_score_used_for_support_choice=False))
    assert len(rows) == 300 and len({(r['method'], r['id']) for r in rows}) == 300
    counts = {m: dict(collections.Counter(r['selected_support'] for r in rows if r['method'] == m)) for m in common.METHODS}
    result = dict(status='all_300_support_bindings_verified', frozen_id_count=100, method_record_count=300,
        selection_sha256=contract['selection_sha256'], source_hashes=sources, counts=counts, rows=rows,
        scope='Final native support and independent output bindings only; not a declaration that all people are complete or all metrics passed',
        count_policy='Original candidate validation Gaussian counts describe the conservative bundle. Use selected_gaussian_count here for the final actor-only view.',
        original_large_assets_duplicated=False, source_script=str(Path(__file__)), source_script_sha256=common.sha(__file__))
    common.save(root / 'resolution.json', result)
    print(counts)


if __name__ == '__main__':
    main()
