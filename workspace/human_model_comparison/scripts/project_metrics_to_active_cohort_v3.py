"""Project verified cached object scores onto the explicitly revised cohort.

No scores are recomputed or pose-adjusted. Historical files remain immutable;
the new summary keeps all100 and recomputes paired sets after removing HN0035.
"""
import csv,json
from pathlib import Path
import human_metric_batch_v1 as b

H=b.H
def main():
    pointer=H/'inputs/active_human_benchmark.json';active=b.read(pointer)
    b.verified(active['selection'],active['selection_sha256']);current=b.read(active['selection'])
    selected={r['id']:r for r in current['objects']};assert len(selected)==100
    previous=current['supersedes'];b.verified(previous['path'],previous['sha256'])
    old={r['id']:r for r in b.read(previous['path'])['objects']}
    retained=set(old)&set(selected)
    assert set(old)-set(selected)=={'HN0035'} and set(selected)-set(old)=={'HN0646'}
    assert len(retained)==99 and all(selected[k]==old[k] for k in retained)
    run=H/'reports/source_camera_metric_batch_v1';plan=b.load_plan(run)
    qualification=b.read(plan['qualification'])
    assert qualification['selection_sha256']==previous['sha256']
    methods=plan['methods'];records=[];by_method={method:{} for method in methods}
    verified_inputs={};results={};qualified={r['id']:r for r in qualification['objects']}
    for row in plan['rows']:
        sid=row['id'];method=row['method']
        if sid not in retained:continue
        ref=row['reference'];assert ref==qualified[sid]
        assert ref['identity_key']==selected[sid]['identity_key'] and ref['timestamp_us']==selected[sid]['t_ref_us']
        for k in ['camera_contract','reference_rgb','reference_mask']:
            path=ref[k];expected=ref[k+'_sha256']
            if path not in verified_inputs:b.verified(path,expected);verified_inputs[path]=expected
            else:assert verified_inputs[path]==expected
        output=b.result_path(run,row) if row['status']=='ready' else None
        rec=dict(id=sid,method=method,status=row['status'],reason=row.get('reason'),reference=ref)
        if output is not None and output.exists():
            value=b.read(output);assert value['status']=='scored' and value['input_key']==row['input_key']
            cert=b.check_registration(ref,method,row)
            assert value['reference_rgb_sha256']==ref['reference_rgb_sha256'] and value['reference_mask_sha256']==ref['reference_mask_sha256']
            assert value['prediction_rgb_sha256']==cert['source_rgb_sha256'] and value['prediction_alpha_sha256']==cert['source_alpha_sha256']
            assert value['registration_sha256']==row['registration_sha256']
            by_method[method][sid]=value;results[str(output)]=b.m.sha(output)
            rec.update(status='scored',reason=None,result=str(output),result_sha256=results[str(output)],input_key=row['input_key'],metrics=value['metrics'],reused_from_previous_cohort=True)
        records.append(rec)
    for method in methods:
        records.append(dict(id='HN0646',method=method,status='reference_and_registration_pending' if method!='splatad' else 'native_unavailable_deferred',
                            reason='Replacement reference qualification and method-specific camera certificate not yet frozen' if method!='splatad' else 'Native stationary actor filter; deferred by explicit user instruction',
                            metric_eligible=False,reference=None))
    assert len(records)==100*len(methods) and all(r['id']!='HN0035' for r in records)
    names=['PSNR','SSIM','LPIPS','IoU','IoU_raw'];pairings={}
    for chosen in [methods,['omnire','splatad']]:
        pairings['+'.join(chosen)]={name:b.m.paired_summary({m:by_method[m] for m in chosen},name,list(selected)) for name in names}
    out=H/'reports/source_camera_metric_cohort_v3';out.mkdir(exist_ok=False)
    summary=dict(status='verified_cached_scores_projected_to_active_cohort',denominator=100,
                 selection=active['selection'],selection_sha256=active['selection_sha256'],
                 historical_qualification=plan['qualification'],historical_qualification_sha256=plan['qualification_sha256'],
                 protocol=plan['protocol'],protocol_sha256=plan['protocol_sha256'],source_plan=str(run/'latest_plan.json'),source_plan_sha256=b.m.sha(run/'latest_plan.json'),
                 removed_ids=['HN0035'],added_ids=['HN0646'],retained_objects_identical=99,
                 reference_counts=dict(inherited_eligible=sum(qualified[k]['reference_eligible'] for k in retained),inherited_na=sum(not qualified[k]['reference_eligible'] for k in retained),replacement_pending=1),
                 coverage={k:len(v) for k,v in by_method.items()},pairings=pairings,rows=records,
                 cached_result_sha256=results,source_reconstruction_only=True,full_comparison_complete=False,
                 cohort_change_disclosure=current['bias_disclosure'],scores_recomputed=0,script_sha256=b.m.sha(__file__))
    b.save(out/'summary.json',summary)
    fields=['id','method','status','reason','result','result_sha256']+[x for n in names+['ED-P'] for x in [n,n+'_status']]
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in records:
            row={k:r.get(k) for k in fields[:6]}
            for name in names+['ED-P']:
                v=r.get('metrics',{}).get(name,dict(status='na',value=None));row[name+'_status']=v['status'];row[name]='inf' if v['status']=='positive_infinity' else v['value'] if v['status']=='ok' else 'NA'
            writer.writerow(row)
    print(json.dumps(dict(summary=str(out/'summary.json'),coverage=summary['coverage'],reference_counts=summary['reference_counts'],verified_cached_results=len(results),denominator=100)))

if __name__=='__main__':main()
