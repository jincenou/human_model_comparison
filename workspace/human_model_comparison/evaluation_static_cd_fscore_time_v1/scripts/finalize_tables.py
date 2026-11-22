"""Materialize complete coverage and same-person panels; pending stays pending."""
from collections import Counter
import math,time
import numpy as np
from common import *
PANEL='annotation_vertical_extent_conditioned'
LABELS=dict(ours='Ours',ah='AH',trellis_original='TRELLIS',trellis2='TRELLIS.2',omnire='OmniRe',splatad='SplatAD')
def summary(rows,method,condition,population,name):
    metrics=[r['panels'][condition]['metrics'] for r in rows];finite=[m['cd_m'] for m in metrics if m['cd_m'] is not None];empty=len(metrics)-len(finite)
    out=dict(set=name,condition=condition,reference_population=population,method=method,n=len(rows),n_finite=len(finite),n_empty=empty,cd_finite_mean_cm=float(np.mean(finite)*100) if finite else None,cd_finite_median_cm=float(np.median(finite)*100) if finite else None,cd_extended_mean_cm='inf' if empty else float(np.mean(finite)*100) if finite else None)
    for label,quantile in [('p25',.25),('p75',.75),('p90',.9),('max',1.)]:out[f'cd_finite_{label}_cm']=float(np.quantile(finite,quantile)*100) if finite else None
    for k in ['f1_5cm','f1_10cm','f1_20cm','precision_10cm','recall_10cm','ray_hit','ray_acc_5cm','ray_acc_10cm','ray_acc_20cm']:out[k]=float(np.mean([m[k] for m in metrics])) if metrics else None
    return out

def main():
    intake=read(E/'intake.json')['rows'];refs={r['id']:r for r in read(E/'reference_manifest.json')['rows']};accepted={(r['method'],r['id']):r for r in read(E/'registration_recovery/accepted.json')['rows']}
    scenes={(r['method'],r['id']):r for r in read(E/'validation/missing_scene_audit_final.json')['records']};recovery=read(E/'registration_recovery/local_views_summary.json')['rows'];result_rows=[];ledger=[]
    gaps=[j for j in intake if j['method'] in ['trellis_original','trellis2'] and j['asset_available'] and not j.get('T_source_camera_from_native')]
    gap_index={(j['method'],j['id']):i for i,j in enumerate(gaps)}
    recognizable={'HN0592','HN0866','HN0190','HN0272','HN0317','HN0263','HN0517','HN0215'}
    failure=[]
    for job in intake:
        sid,m=job['id'],job['method'];ref=refs[sid];key=(m,sid);path=E/'results'/m/(sid+'.json');d=read(path) if path.exists() else None
        row=dict(id=sid,method=m,identity_key=job['identity_key'],t_ref_us=job['t_ref_us'],scan_timestamp_us=ref['scan_timestamp_us'],delta_ms=ref['delta_ms'],reference_panel=ref['candidate_panel'],reference_count=ref['query_count'],reference_status=ref['reference_status'],asset_available=job['asset_available'],prior_2d_status=job['prior_2d_status'],registration_recovered=key in accepted,protocol_sha256=sha(E/'protocol.json'),queries_sha256=ref['queries_sha256'])
        if d and d['status']=='completed':
            met=d['panels'][PANEL]['metrics'];cert=read(d['certificate']);row.update(geometry_status='empty_pred' if met['status']=='empty_pred' else 'scored',registration_status='recovered_source_relation' if key in accepted else 'existing_source_relation_verified',depth_status='validated_native_adapter',scale_m_per_native_unit=cert['scale_m_per_native_unit'],native_height=cert['native_vertical_extent'],annotation_height=cert['annotation_vertical_extent_m'],result=str(path),result_sha256=sha(path),certificate=d['certificate'],certificate_sha256=d['certificate_sha256'],returns=d['returns'],returns_sha256=d['returns_sha256'],**met);result_rows.append(d)
        elif not job['asset_available']:
            if m=='trellis2':
                p=H/'runs/trellis2_human_native_v1'/sid/'state.json';state=read(p);row.update(geometry_status='NA_native_generation_failure',reason=state['reason'],evidence=str(p),evidence_sha256=sha(p))
            else:
                x=scenes[key];row.update(geometry_status=x['status'],reason=x['reason'],evidence=str(E/'validation/missing_scene_audit_final.json'),evidence_sha256=sha(E/'validation/missing_scene_audit_final.json'))
        elif key in gap_index and key not in accepted:
            i=gap_index[key];original=E/'registration_recovery/existing'/m/sid/'registration.json';zoom=E/'registration_recovery/zoom'/m/sid/'registration.json';local=[r for r in recovery if (r['method'],r['id'])==key]
            body_recognizable=sid in recognizable
            reason=('可辨认部分人体，但源图遮挡、肢体合并/缺失或双面纹理导致缺乏足够可靠的同一组3D—2D语义对应；统一修正及直接视觉复核后仍无法确认唯一全局关系。' if body_recognizable else '统一检测放大、多视图三角化及直接视觉复核后，仍缺少可确认的足量人体3D—2D对应；该组证据包含残片、水平/背景纹理面片及退化结构等情况。未将检测器失败单独作为判据，也不宣称未来任何人工对应方案都不可能恢复。')
            row.update(geometry_status='NA_global_correspondence_unresolved',registration_status='bounded_correction_and_visual_review_completed',reason=reason,evidence=str(E/'registration_review'/f'gaps_{i//8+1:02d}.jpg'),evidence_tile=i%8,evidence_sha256=sha(E/'registration_review'/f'gaps_{i//8+1:02d}.jpg'),original_correspondences=len(read(original)['joints']),zoom_correspondences=len(read(zoom)['joints']),maximum_local_correspondences=max(len(x['joints']) for x in local),reviewer='Codex direct review, not independent human blind review')
        else:row.update(geometry_status=d['status'] if d else 'queued',reason=d.get('reason') if d else 'eligible workload not yet complete')
        ledger.append(row)
        if row['geometry_status'].startswith('NA'):failure.append(row)
    assert len(ledger)==600 and len({(r['id'],r['method']) for r in ledger})==600
    csv_write(E/'geometry_ledger.csv',ledger);save(E/'geometry_ledger.json',dict(rows=ledger,updated_at=time.time()));csv_write(E/'tables/failure_reasons.csv',failure)
    coverage=[]
    for m in METHODS:
        for population in ['main','low_support_exploratory']:
            rows=[r for r in ledger if r['method']==m and (r['reference_panel']=='main')==(population=='main')];counts=Counter(r['geometry_status'] for r in rows)
            coverage.append(dict(method=m,reference_population=population,n_planned=len(rows),n_scored=counts['scored'],n_empty=counts['empty_pred'],n_na=sum(v for k,v in counts.items() if k.startswith('NA')),n_pending=sum(v for k,v in counts.items() if k not in ['scored','empty_pred'] and not k.startswith('NA')),n_recovered=sum(r['registration_recovered'] for r in rows)))
    csv_write(E/'tables/coverage.csv',coverage)
    sets={};stats=[];all_complete={m:{d['id']:d for d in result_rows if d['method']==m} for m in METHODS}
    for population in ['main','low_support_exploratory']:
        refids={sid for sid,r in refs.items() if (r['candidate_panel']=='main')==(population=='main')}
        for condition,methods in [(PANEL,METHODS),('native_metric',['omnire','splatad'])]:
            own={m:set(all_complete[m])&refids for m in methods};common=set.intersection(*own.values());finite={sid for sid in common if all(all_complete[m][sid]['panels'][condition]['metrics']['cd_m'] is not None for m in methods)}
            prefix=population+'/'+condition;sets[prefix+'/common_admission']=sorted(common);sets[prefix+'/common_finite_cd']=sorted(finite)
            for m in methods:
                sets[prefix+'/own/'+m]=sorted(own[m])
                for label,ids in [('own',own[m]),('common_admission',common),('common_finite_cd',finite)]:stats.append(summary([all_complete[m][sid] for sid in sorted(ids)],m,condition,population,label))
            if condition==PANEL:
                for baseline in METHODS[1:]:
                    pair=own['ours']&own[baseline];pf={sid for sid in pair if all(all_complete[m][sid]['panels'][condition]['metrics']['cd_m'] is not None for m in ['ours',baseline])}
                    for label,ids in [('admission',pair),('finite_cd',pf)]:
                        key=prefix+'/ours_vs_'+baseline+'/'+label;sets[key]=sorted(ids)
                        for m in ['ours',baseline]:stats.append(summary([all_complete[m][sid] for sid in sorted(ids)],m,condition,population,'ours_vs_'+baseline+'/'+label))
    save(E/'tables/common_and_paired_sets.json',sets);csv_write(E/'tables/geometry_summary.csv',stats)
    per_person=[]
    for d in result_rows:
        for cond,r in d['panels'].items():per_person.append(dict(id=d['id'],method=d['method'],reference_panel=d['reference_panel'],condition=cond,**r['metrics']))
    csv_write(E/'tables/geometry_per_person.csv',per_person)
    time_rows=[];time_stages=[]
    for job in read(E/'time_plan.json')['runs']:
        m,sid,repeat=job['method'],job['id'],job['repeat']
        if m=='ours':p=E/'timing/ours_runs'/f'{sid}_r{repeat}/timing.json'
        else:p=E/'timing'/('corrected_runs' if m=='ah' else 'runs')/m/f'{sid}_r{repeat}/timing.json'
        d=read(p) if p.exists() else dict(status='queued');row=dict(method=m,id=sid,repeat=repeat,status=d['status'],seconds=d.get('seconds'),failed_elapsed_seconds=d.get('failed_elapsed_seconds'),reason=d.get('reason'),current_or_failed_phase=d.get('failed_phase',d.get('current_phase')),record=str(p),record_sha256=sha(p) if p.exists() else None)
        if m=='ours':
            audit=p.parent/'independent_reload.json';row['independent_reload_status']=read(audit)['status'] if audit.exists() else 'pending';row['independent_reload_record']=str(audit)
        else:row['independent_reload_status']='passed' if d.get('independent_reload_passed') else 'pending'
        time_rows.append(row)
        for s in d.get('stages',[]):time_stages.append(dict(method=m,id=sid,repeat=repeat,stage=s.get('name',s.get('phase')),status=s['status'],seconds=s.get('seconds'),wall_seconds=s.get('wall_seconds'),excluded_load_seconds=sum(x['seconds'] for x in s.get('excluded_pretrained_load_intervals',[]))))
    csv_write(E/'tables/time_per_run.csv',time_rows);csv_write(E/'tables/time_per_stage.csv',time_stages);time_summary=[];time_people=[]
    for m in METHODS:
        rows=[r for r in time_rows if r['method']==m];values=[r['seconds'] for r in rows if r['status']=='success'];four=[r['seconds'] for r in rows if r['status']=='success' and r['repeat']==0]
        person_means=[]
        for sid in sorted({r['id'] for r in rows}):
            samples=[r for r in rows if r['id']==sid];v=[r['seconds'] for r in samples if r['status']=='success'];mean=float(np.mean(v)) if v else None
            time_people.append(dict(method=m,id=sid,n_planned=len(samples),n_success=len(v),mean_success_seconds=mean))
            if v:person_means.append(mean)
        time_summary.append(dict(method=m,n_planned=len(rows),n_success=len(values),n_failed=sum(r['status']=='failed' for r in rows),n_pending=sum(r['status'] not in ['success','failed'] for r in rows),n_successful_people=len(person_means),person_equal_weight_mean_seconds=float(np.mean(person_means)) if person_means else None,person_equal_weight_median_seconds=float(np.median(person_means)) if person_means else None,mean_success_seconds=float(np.mean(values)) if values else None,median_success_seconds=float(np.median(values)) if values else None,min_seconds=min(values) if values else None,max_seconds=max(values) if values else None,four_unique_person_mean_seconds=float(np.mean(four)) if four else None,scope='not_applicable_scene_training_not_measured' if m in ['omnire','splatad'] else 'fixed4persons_plus_anchor_repeat_native_generation'))
    csv_write(E/'tables/time_per_person.csv',time_people)
    csv_write(E/'tables/time_summary.csv',time_summary)
    status=dict(updated_at=time.time(),geometry=Counter(r['geometry_status'] for r in ledger),reference_main=84,reference_low=16,coverage=coverage,time_summary=time_summary,geometry_terminal=sum(r['geometry_status'] in ['scored','empty_pred'] or r['geometry_status'].startswith('NA') for r in ledger),time_terminal=sum(r['status'] in ['success','failed'] for r in time_rows),goal_complete=False)
    save(E/'progress.json',status);print(dict(geometry_terminal=status['geometry_terminal'],time_terminal=status['time_terminal'],geometry=dict(status['geometry'])),flush=True)

if __name__=='__main__':main()
