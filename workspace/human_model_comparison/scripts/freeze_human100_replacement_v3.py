"""Apply the user's explicit HN0035 replacement; immutable v2 remains historical."""
import copy,csv,json,datetime
from pathlib import Path
from audit_nuscenes_cohort_v2 import sha

H=Path(__file__).resolve().parents[1]
def main():
    oldpath=H/'inputs/benchmark_human100_v2/selection.json';old=json.loads(oldpath.read_text())
    auditpath=H/'reports/replacement_HN0035_20261112_v1/candidate_audit.json';a=json.loads(auditpath.read_text())
    assert sha(oldpath)==a['old_selection_sha256']
    folder=H/'references/replacement_v3/HN0646';meta=json.loads((folder/'metadata.json').read_text())
    assert sha(meta['mask_path'])==meta['mask_sha256'] and meta['timestamp_us']==a['t_ref_us']
    assert sha(a['reference_frame']['raw_source_path'])==a['reference_frame']['raw_source_sha256']
    assert sha(a['paths']['runtime'])==a['runtime_sha256']
    review=dict(id='HN0646',status='identity_and_mask_candidate_reviewed',
                inspected=['candidate_review.jpg','exact_mask_review.png'],
                observations=['Red top and diagonal white strap retained','Neighboring people excluded from mask','Small raw target and blur disclosed; final occlusion qualification pending'],
                identity_valid=True,reference_ready=False,metric_eligible=False)
    (auditpath.parent/'visual_review.json').write_text(json.dumps(review,indent=2)+'\n')
    oldrow=next(r for r in old['objects'] if r['id']=='HN0035')
    new=dict(benchmark_number=oldrow['benchmark_number'],id='HN0646',dataset='nuscenes',scene=a['scene'],identity_key=a['identity_key'],
             cohort='replacement_v3',replaces='HN0035',legacy_review_ordinal=None,candidate_review_number=None,
             selected=True,identity_selection_status='frozen_under_explicit_user_replacement_delegation',
             t_ref_us=a['t_ref_us'],reference_seconds_from_first_native=a['reference_seconds_from_first_native'],
             time_rule='nearest_retained_train_keyframe_to_median_native_exposure_earlier_tie',
             reference_frame=a['reference_frame'],paths=a['paths'],provenance_hashes=a['provenance_hashes'],
             input_budget=a['input_budget'],reference_mask_candidate=dict(metadata=str(folder/'metadata.json'),path=meta['mask_path'],sha256=meta['mask_sha256'],status='identity_isolated_final_reference_qualification_pending'),
             asset_produced=True,runtime_numeric_decode_valid=True,runtime_sha256=a['runtime_sha256'],
             reference_ready=False,reload_valid=False,identity_valid=True,common_camera_valid=False,metric_eligible=False,scored=False,
             pending_reason='Replacement identity frozen; native method completion, reference qualification and common-camera registration pending')
    objects=[new if r['id']=='HN0035' else r for r in old['objects']]
    assert len(objects)==100 and len({r['identity_key'] for r in objects})==100
    assert all(r['scene'] in old['allowed_scenes'] for r in objects)
    oldmap={r['id']:r for r in old['objects']};assert all(r==oldmap[r['id']] for r in objects if r['id']!='HN0646')
    bans={r['id'] for r in old['exclusions_this_review']}|set(old['historical_candidate_exclusions_preserved'])|{'HN0035'}
    assert not bans & {r['id'] for r in objects}
    out=H/'inputs/benchmark_human100_v3';out.mkdir(exist_ok=False)
    result=copy.deepcopy(old);result.update(schema='human100_nuscenes_frozen_v3',status='frozen_user_authorized_single_replacement',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),objects=objects,
        counts=dict(original_people_retained=74,previous_new_people_retained=25,replacement_people=1,people=100,allowed_scenes=100,represented_scenes=len({r['scene'] for r in objects})),
        supersedes=dict(path=str(oldpath),sha256=sha(oldpath)),
        user_authorization=dict(instruction='那就直接抛弃这个资产，后面你重新冻结一个资产然后重新跑几个模型',target='HN0035',scope='Replace exactly one identity and rerun corresponding methods; reuse the other99'),
        selection_policy='User-directed HN0035 exclusion supersedes original75 preservation for this person only; HN0646 chosen from unexcluded existing inventory within unchanged100 scenes',
        failure_policy='No further automatic replacements; this single exception was explicitly requested by the user',
        bias_disclosure='Cohort v3 replaces HN0035 after inspection of Ours appearance. Report this user-directed post-output curation; do not describe v3 as outcome-independent selection or silently pool with v2.',
        replacement=dict(removed='HN0035',added='HN0646',audit=str(auditpath),audit_sha256=sha(auditpath),others_unchanged=99),
        original_humans=[r for r in old['original_humans'] if r['id']!='HN0035'])
    result['exclusions_this_review']=old['exclusions_this_review']+[dict(id='HN0035',identity_key=oldrow['identity_key'],basis='explicit_user_exclusion_20261112',reason='User requested replacement after orientation/appearance inspection')]
    path=out/'selection.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'selection.sha256.json').write_text(json.dumps(dict(path=str(path),sha256=sha(path)),indent=2)+'\n')
    with (out/'identities.csv').open('w',newline='') as f:
        fields=['benchmark_number','id','cohort','scene','identity_key','t_ref_us'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in objects)
    pointer=H/'inputs/active_human_benchmark.json';assert not pointer.exists()
    pointer.write_text(json.dumps(dict(selection=str(path),selection_sha256=sha(path),population=100,removed_ids=['HN0035'],added_ids=['HN0646'],reuse_prior_ids=[r['id'] for r in objects if r['id']!='HN0646'],policy='New evaluation consumers must use this version; immutable v2 jobs may finish for retained identities. Never include HN0035 in v3 aggregation.'),indent=2)+'\n')
    print(json.dumps(dict(selection=str(path),removed='HN0035',added='HN0646',population=100,unchanged=99)))

if __name__=='__main__':main()
