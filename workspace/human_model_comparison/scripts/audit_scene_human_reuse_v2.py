"""Read-only audit of existing actor exports and candidate scene checkpoints."""
import collections
import hashlib
import json
from pathlib import Path
import time

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    rows=json.loads(selection.read_text())['objects']
    sources={}
    for name in ['omnire_human_legacy_no_smpl_queue','omnire_reconstruction_only_queue',
                 'omnire_legacy_no_smpl_queue','nuscenes100_expansion_queue']:
        path=R/'state'/f'{name}.json'
        if path.exists():sources[name]=(path,json.loads(path.read_text()))
    records=[]
    for row in rows:
        rec=dict(id=row['id'],cohort=row['cohort'],scene=row['scene'],identity_key=row['identity_key'],t_ref_us=row['t_ref_us'],methods={})
        uuid=row['identity_key'].split(':')[-1]
        for method in ['omnire','splatad']:
            path=R/'exports'/f'{method}_human95'/'objects'/row['id']/'validation.json'
            result=dict(prior_validation=str(path),status='not_previously_extracted',scene_candidates=[],common_camera_valid=False,scored=False)
            if path.exists():
                v=json.loads(path.read_text());result.update(prior_status=v['status'],validation_sha256=sha(path))
                if v['status']=='passed':
                    issues=[]
                    for ok,reason in [(v['actor_uuid']==uuid,'identity_mismatch'),(v['scene']==row['scene'],'scene_mismatch'),
                                      (v.get('frozen_raw_timestamp_us')==row['t_ref_us'],'time_mismatch'),
                                      (Path(v['source_image']).resolve()==Path(row['reference_frame']['raw_source_path']).resolve(),'source_path_mismatch')]:
                        if not ok:issues.append(reason)
                    files={k:dict(path=v[k],exists=Path(v[k]).is_file(),bytes=Path(v[k]).stat().st_size if Path(v[k]).is_file() else None)
                           for k in ['checkpoint','bundle','actor_preview','actor_alpha']}
                    issues.extend(k+'_missing' for k,f in files.items() if not f['exists'])
                    result.update(status='existing_exact_native_artifacts_available' if not issues else 'prior_export_needs_repair',issues=issues,files=files,
                                  note='Path/time/identity and file existence checked; no new decoder execution or common-camera validation implied.')
                else:result.update(status='prior_native_unavailable',reason=v.get('reason'))
            if method=='omnire':
                for name,(q,d) in sources.items():
                    candidates=[]
                    if name=='nuscenes100_expansion_queue':
                        j=d.get('jobs',{}).get('omnire:'+row['scene']);candidates=[j] if j else []
                    elif row['scene'] in d.get('scenes',{}):candidates=[d['scenes'][row['scene']]]
                    for j in candidates:
                        if j.get('stage',j.get('status'))!='formal_validated':continue
                        cp=Path(j.get('checkpoint','/nonexistent'))
                        result['scene_candidates'].append(dict(source=str(q),checkpoint=str(cp),checkpoint_exists=cp.is_file(),
                            config=str(cp.parent/'config.yaml'),config_exists=(cp.parent/'config.yaml').is_file(),
                            candidate_only=True,instance_membership_pending=True))
            else:
                for vp in sorted((R/'reports').glob(f'splatad*formal_{row["scene"]}_validation.json')):
                    v=json.loads(vp.read_text())
                    if v.get('status')!='passed':continue
                    actors=[a for a in v.get('actors',[]) if a.get('uuid')==uuid]
                    cp=Path(v.get('checkpoint','/nonexistent'))
                    result['scene_candidates'].append(dict(source=str(vp),checkpoint=str(cp),checkpoint_exists=cp.is_file(),
                        exact_actor_present=bool(actors),actor_bundles=[dict(path=a.get('bundle'),exists=bool(a.get('bundle')) and Path(a['bundle']).is_file(),gaussians=a.get('gaussians')) for a in actors],
                        input_split_requires_provenance_audit=True))
            rec['methods'][method]=result
        records.append(rec)
    counts={method:{cohort:dict(collections.Counter(r['methods'][method]['status'] for r in records if r['cohort']==cohort))
                     for cohort in ['original75','new25']} for method in ['omnire','splatad']}
    extra={m:sum(any(x.get('checkpoint_exists') and (m=='omnire' or x.get('exact_actor_present')) for x in r['methods'][m]['scene_candidates']) for r in records if r['cohort']=='new25') for m in ['omnire','splatad']}
    output=dict(checked_at_unix=time.time(),selection=str(selection),selection_sha256=sha(selection),counts=counts,
                new25_with_candidate_checkpoint_or_splat_actor=extra,
                no_scene_training_started=True,records=records)
    out=H/'reports/scene_human_reuse_v2';out.mkdir(exist_ok=True)
    (out/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,new25_candidates=extra),ensure_ascii=False))


if __name__=='__main__':main()
