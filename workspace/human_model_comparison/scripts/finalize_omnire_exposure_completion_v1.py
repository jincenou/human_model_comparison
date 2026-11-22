"""Summarize isolated full-scene timing audit against frozen semantic decisions."""
from native_reaudit_omnire_v1 import REPORT,EXPORT,read,save,sha

def main():
    source=REPORT/'all_node_exposure_aux/summary.json'
    summary=read(source);semantic=read(REPORT/'semantic_review.json')
    assert summary['count']==100 and len(summary['rows'])==100
    decisions={r['id']:r for r in semantic['rows']};rows=[]
    for row in summary['rows']:
        decision=decisions[row['id']];validation=EXPORT/row['id']/'validation.json'
        assert sha(validation)==decision['immutable_validation_sha256']==row['validation_sha256']
        selected_bg=decision['decision']!='reuse_actor_only'
        rows.append(dict(id=row['id'],decision=decision['decision'],selected_background=selected_bg,
            source_timestamp_us=read(validation)['source_reference']['timestamp_us'],
            target_parameter_max_error=max(row['target_gaussian_parameter_max_errors'].values()),
            full_scene_corrected=row['full_scene_corrected'],full_scene_corrected_sha256=row['full_scene_corrected_sha256'],
            full_scene_label=row['corrected_full_scene_label'],
            auxiliary_audit=row['auxiliary_audit'],auxiliary_audit_sha256=row['auxiliary_audit_sha256'],
            old_background_selection_omitted_weight_fraction=row['background_omitted_weight_fraction'],
            independent_selection_mask_rgb_mae=row['independent_selection_mask_rgb_mae'],
            independent_selection_mask_rgb_max=row['independent_selection_mask_rgb_max'],
            independent_selection_mask_fraction_channels_over_1byte=row['independent_selection_mask_fraction_channels_over_1byte'],
            fixed_tolerance_flags=row['fixed_tolerance_flags'],
            selected_bg_requires_explicit_root_review=selected_bg and bool(row['fixed_tolerance_flags']),
            original_asset_unchanged=True))
    out=dict(status='all100_auxiliary_exposure_audit_complete',count=100,
        exposure_summary=str(source),exposure_summary_sha256=sha(source),
        frozen_semantic_review=str(REPORT/'semantic_review.json'),frozen_semantic_review_sha256=sha(REPORT/'semantic_review.json'),
        all_passed_validation_hashes_unchanged=True,
        max_target_parameter_error=max(r['target_parameter_max_error'] for r in rows),
        selected_background_ids=[r['id'] for r in rows if r['selected_background']],
        selected_background_diagnostic_flags=[r['id'] for r in rows if r['selected_bg_requires_explicit_root_review']],
        any_candidate_diagnostic_flags=[r['id'] for r in rows if r['fixed_tolerance_flags']],
        thresholds=summary['thresholds'],
        outcome='Use auxiliary corrected full-scene images for scene diagnostics. No original export, formal metric record, or frozen semantic decision was modified.',
        timing_policy='Every bracket-valid native instance root and deformation clock queried at source physical exposure. Unbracketed instance roots retain native source anchor and are explicitly counted. Target actor parameters are audited against the original independent bundle query.',
        revision_policy='Diagnostic threshold flags alone do not switch exports or remove difficult objects. Any selected-background difference must be reviewed by root and recorded as a separate revision, never an in-place change.',
        rows=rows)
    save(REPORT/'exposure_completion_audit.json',out)
    print({k:out[k] for k in ['count','max_target_parameter_error','selected_background_diagnostic_flags','any_candidate_diagnostic_flags']})

if __name__=='__main__':main()
