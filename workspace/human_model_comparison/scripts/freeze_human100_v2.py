"""Apply the user's explicit review decision; freeze identities, not metric eligibility."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEHICLE = ROOT.parent/'vehicle_model_comparison'
DEST = ROOT/'inputs/benchmark_human100_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def save(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def main():
    selection_path=DEST/'selection.json'
    assert not selection_path.exists(),'Frozen selection already exists; do not overwrite'
    paths={
        'frozen_scene_selection':VEHICLE/'inputs/benchmark_nuscenes100_v1/selection.json',
        'original_time_manifest':VEHICLE/'inputs/benchmark_20261106/human_snapshot_manifest_v1.json',
        'cohort_audit':ROOT/'reports/nuscenes_cohort_audit_v2/audit.json',
        'candidate_review':ROOT/'reports/nuscenes_candidate_review_v2/manifest.json',
        'candidate_runtime_decode':ROOT/'reports/nuscenes_cohort_audit_v2/runtime_decode.json',
        'user_review':ROOT/'inputs/human100_user_review_20261110.json',
    }
    frozen=read(paths['frozen_scene_selection'])
    audit=read(paths['cohort_audit'])
    by_id={r['id']:r for r in audit['records']}
    review=read(paths['candidate_review'])
    by_number={r['review_number']:r for r in review['records']}
    decisions=read(paths['user_review'])
    assert decisions['review_manifest_sha256']==sha(paths['candidate_review'])
    assert by_number['C07']['id']=='HN0178'
    explicit={r['id'] for r in decisions['confirmed_exclusions']}
    assert len(explicit)==9
    additional={
        'C15':'真实参考帧噪声明显，人物细节受损；依据真实输入，不依据重建分数。',
        'C18':'真实参考人物衣着与暗部背景对比低，可辨细节不足；依据真实输入。',
        'C33':'目标儿童与旁边成人肢体接近及重叠，已有mask混入成人腿/鞋，精确轮廓判定困难。',
    }
    extra=[{'review_number':n,'id':by_number[n]['id'],'identity_key':by_number[n]['identity_key'],
            'basis':'user_delegated_source_quality_selection','reason':reason}
           for n,reason in additional.items()]
    excluded=explicit|{r['id'] for r in extra}
    additions=[r for r in review['records'] if r['id'] not in excluded]
    originals=frozen['existing_humans']
    assert len(review['records'])==37 and len(excluded)==12
    assert len(originals)==75 and len(additions)==25
    assert len({r['identity_key'] for r in originals+additions})==100
    assert not {r['id'] for r in originals}&excluded
    assert not {r['id'] for r in additions}&set(review['excluded_prior_ids'])
    scene_set={r['scene'] for r in frozen['scenes']}
    assert len(scene_set)==100
    assert all(r['scene'] in scene_set for r in originals+additions)
    for source in audit['source_files'].values():
        assert sha(source['path'])==source['sha256'],'Authoritative source manifest changed'
    for scene in frozen['scenes']:
        assert sha(scene['split'])==scene['split_sha256'],'Frozen scene split changed'
    times={r['id']:r for r in read(paths['original_time_manifest'])['objects'] if r['dataset']=='nuscenes'}
    decoded={r['id']:r for r in read(paths['candidate_runtime_decode'])['records']}
    objects=[]
    for number,row in enumerate(originals+additions,1):
        ident=row['id'];record=by_id[ident];old=number<=75
        assert not record['blocked_by_prior_exclusion'] and record['native_identity_matches']
        frame=record['reference_frame']
        assert sha(frame['raw_source_path'])==frame['raw_source_sha256']
        assert sha(frame['path'])==frame['sha256']
        if old:
            assert record['t_ref_us']==times[ident]['raw_timestamp_us']
            assert frame==times[ident]['native_frame']
        mask_dir=ROOT/'references/exact_exposure_v2'/ident
        mask=read(mask_dir/'metadata.json')
        assert mask['timestamp_us']==record['t_ref_us']
        assert mask['source_sha256']==frame['raw_source_sha256']
        assert mask['mask_sha256']==sha(mask_dir/'mask.png')
        obj={'benchmark_number':number,'id':ident,'dataset':'nuscenes','scene':row['scene'],
            'identity_key':row['identity_key'],'cohort':'original75' if old else 'new25',
            'legacy_review_ordinal':row.get('review_ordinal') if old else None,
            'candidate_review_number':None if old else row['review_number'],
            'selected':True,'identity_selection_status':'frozen_user_reviewed',
            't_ref_us':record['t_ref_us'],'reference_seconds_from_first_native':record['reference_seconds_from_first_native'],
            'time_rule':'original_frozen_time_preserved' if old else 'nearest_retained_train_keyframe_to_median_native_exposure_earlier_tie',
            'reference_frame':frame,'paths':record['paths'],'provenance_hashes':record['provenance_hashes'],
            'input_budget':{'real_native_exposures':record['real_exposures'],
                'native_duration_s':record['native_duration_s'],'model_input_slots':record['model_input_slots'],
                'interpolated_slots':record['interpolated_slots'],'model_input_duration_s':record['input_duration_s'],
                'generated_auxiliary_views':'pending_full_frontend_cache_audit'},
            'reference_mask_candidate':{'metadata':str(mask_dir/'metadata.json'),
                'path':mask['mask_path'],'sha256':mask['mask_sha256'],'status':'pending_final_visual_and_eligibility_review'},
            'asset_produced':True,'runtime_numeric_decode_valid':None if old else decoded[ident]['status']=='decoded_numeric_valid',
            'runtime_sha256':None if old else decoded[ident]['sha256'],
            'reference_ready':False,'reload_valid':False,'identity_valid':False,
            'common_camera_valid':False,'metric_eligible':False,'scored':False,
            'pending_reason':'Selection is frozen; reference qualification, native rendering and common-camera validation are separate gates.'}
        objects.append(obj)
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    decisions.update(status='resolved_applied',resolved_at=now,
        clarification_reply='指 HN0178；再补删3人，最终冻结100人',
        resolved_identifier={'178':'HN0178','duplicate_of':'C07'},
        additional_delegated_exclusions=extra,unique_current_exclusion_count=12,
        identity_freeze_performed=True,frozen_selection=str(selection_path))
    save(paths['user_review'],decisions)
    report={'schema':'human100_nuscenes_frozen_v2','status':'frozen_user_reviewed','frozen_at':now,
        'counts':{'original_people':75,'new_people':25,'people':100,'allowed_scenes':100,
                  'represented_scenes':len({r['scene'] for r in objects}),
                  'review_candidates':37,'user_explicit_unique_exclusions':9,'delegated_extra_exclusions':3},
        'user_authorization':{'instruction':decisions['user_instruction'],'clarification':decisions['clarification_reply'],
            'meaning':'Freeze these100 identities now; no further approval required for the25-person identity list.'},
        'selection_policy':'Keep original75 exactly. Apply user exclusions, deduplicate HN0178/C07, then remove3 source-quality cases under explicit delegation. No external-method scores used.',
        'failure_policy':'Never replace a frozen person because generation, extraction, rendering or scoring fails. Record NA and reason while retaining the denominator100.',
        'qualification_policy':'Identity-list approval is separate from reference_ready, reload_valid, common_camera_valid and metric_eligible; pending gates remain false.',
        'bias_disclosure':'Curated from existing assets and user visual review; this is not a random population sample. Prior cohort selection bias is retained.',
        'sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
        'allowed_scenes':[r['scene'] for r in frozen['scenes']],
        'original_humans':originals,
        'exclusions_this_review':decisions['confirmed_exclusions']+extra,
        'historical_candidate_exclusions_preserved':review['excluded_prior_ids'],
        'objects':objects}
    DEST.mkdir(parents=True,exist_ok=True)
    save(selection_path,report)
    with (DEST/'identities.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['benchmark_number','id','cohort','candidate_review_number','legacy_review_ordinal','scene','identity_key','t_ref_us']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in objects:writer.writerow({k:row.get(k) for k in fields})
    save(DEST/'selection.sha256.json',{'path':str(selection_path),'sha256':sha(selection_path)})
    print(json.dumps({'counts':report['counts'],'new25':[r['id'] for r in objects if r['cohort']=='new25'],
                      'selection':str(selection_path)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
