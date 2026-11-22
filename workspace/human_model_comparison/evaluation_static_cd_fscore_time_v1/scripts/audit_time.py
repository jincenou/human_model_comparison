"""Verify actual inputs, saved endpoints, failures and timing arithmetic without GPU."""
import time
import numpy as np
from common import *
from plyfile import PlyData
def main():
    plan=read(E/'time_plan.json');jobs=plan['runs'];ph=sha(E/'time_plan.json');assert len(jobs)==20
    intake={(r['method'],r['id']):r for r in read(E/'intake.json')['rows']};inputs={}
    for name in ['human_static_native_v1','human_static_replacement_v3']:
        for r in read(H/'inputs'/name/'manifest.json')['objects']:inputs[r['sample_id']]=r
    bindings=[];checks=[];events=[]
    for job in jobs:
        m,sid,repeat=job['method'],job['id'],job['repeat']
        p=E/'timing'/('ours_runs' if m=='ours' else 'corrected_runs' if m=='ah' else 'runs')
        p=(p/f'{sid}_r{repeat}' if m=='ours' else p/m/f'{sid}_r{repeat}')/'timing.json'
        if not p.exists():checks.append(dict(job,status='pending'));continue
        d=read(p);assert d['plan_sha256']==ph and all(d[k]==job[k] for k in ['method','id','repeat'])
        if m=='ours':
            b=read(p.parent/'input_binding.json')
            for raw,h in b['raw_files'].items():checked(raw,h)
            bindings.append(dict(job,real_input_count=len(b['raw_files']),files=b['raw_files'],record=str(p.parent/'input_binding.json')))
        else:
            item=inputs[sid];checked(item['image_rgba'],item['image_rgba_sha256']);assert d['input_sha256']==item['image_rgba_sha256']
            files={item['image_rgba']:item['image_rgba_sha256']}
            if m=='ah':
                trace=read(intake[m,sid]['source_trace']);src=read(trace['input_manifest'])
                for key,digest in [('image','source_image_sha256'),('mask','source_mask_sha256')]:checked(src[key],src[digest]);files[src[key]]=src[digest]
            bindings.append(dict(job,real_input_count=1,files=files,mask_scope='Frozen benchmark target isolation supplied as input; native method preprocessing remains timed.'))
        if d['status']=='failed':
            if m=='ours' and d.get('failed_phase')=='export' and len(d['stages'])==15 and not (p.parent/'timing/export.json').exists():
                checks.append(dict(job,status='pending_export_setup_recovery',record_sha256=sha(p)));continue
            assert d['failed_elapsed_seconds']>=0;checks.append(dict(job,status='failed_attempt_retained',reason=d.get('reason'),record_sha256=sha(p)));continue
        if d['status']!='success':checks.append(dict(job,status='pending'));continue
        checked(d['output'],d['output_sha256']);assert d['seconds']>0
        if m=='ours':
            reload=p.parent/'independent_reload.json'
            if not reload.exists():checks.append(dict(job,status='pending_independent_reload'));continue
            proof=read(reload);assert proof['status']=='passed' and proof['endpoint_sha256']==d['output_sha256'] and proof['timing_record_sha256']==sha(p)
            assert abs(sum(s['seconds'] for s in d['stages'])-d['seconds'])<1e-8
        else:
            assert d['independent_reload_passed'] and all(s['status']=='completed' for s in d['stages'])
            overhead=d['seconds']-sum(s['seconds'] for s in d['stages']);assert 0<=overhead<.2,overhead
            if m=='ah':
                vertex=PlyData.read(d['output'])['vertex'];assert len(vertex)>0
                assert all(np.isfinite(vertex[k]).all() for k in vertex.data.dtype.names)
                source=read(p.parent/'ahc_source_prediction.json');assert np.isfinite(np.asarray(source['c2w_relatives'])).all() and source['fov']>0 and source['dists']>0
            else:
                import torch
                z=torch.load(d['output'],map_location='cpu',weights_only=True)
                if m=='trellis_original':
                    assert set(z['params'])=={'_xyz','_features_dc','_scaling','_rotation','_opacity'};assert len(z['params']['_xyz'])>0
                    assert all(torch.isfinite(x).all() for x in z['params'].values())
                else:
                    assert all(k in z for k in ['vertices','faces','coords','attrs','layout','origin','voxel_size','voxel_shape'])
                    assert z['attrs'].numel()>0 and z['faces'].numel()>0 and z['vertices'].numel()>0
                    assert all(torch.isfinite(z[k]).all() for k in ['vertices','faces','coords','attrs']);assert 'alpha' in z['layout']
        checks.append(dict(job,status='passed',record=str(p),record_sha256=sha(p),endpoint_sha256=d['output_sha256'],seconds=d['seconds']))
    for p in sorted((E/'timing/runs/ah').glob('*/timing.json')):
        d=read(p);assert d['status']=='failed' and 'ndarray is not JSON serializable' in d['reason']
        events.append(dict(method='ah',id=d['id'],repeat=d['repeat'],classification='infrastructure_serialization_error_after_native_inference',seconds=d['failed_elapsed_seconds'],reason=d['reason'],native_ply_exists=(p.parent/'gaussians.ply').exists(),record=str(p),record_sha256=sha(p),resolution='fixed ndarray-to-list metadata wrapper; repeated same fixed inputs/seed/config, original attempt preserved'))
    for p in sorted((E/'state/history').glob('time_*.json')):
        d=read(p)
        if d['status']=='failed':events.append(dict(method=d['method'],classification='controller_setup_failure_before_formal_attempt',record=str(p),record_sha256=sha(p),exit_code=d['exit_code'],resolution='common duplicate status argument setup error corrected; no formal generation in this setup attempt'))
    for p in sorted((E/'timing/ours_runs').glob('*/history/timing.json.before_export_setup_fix')):
        d=read(p);events.append(dict(method='ours',id=d['id'],repeat=d['repeat'],classification='export_wrapper_setup_before_timer',completed_prefix_seconds=d['failed_elapsed_seconds'],failed_stage_measured_seconds=None,reason='NameError: torch undefined in decorated wrapper namespace',record=str(p),record_sha256=sha(p),resolution='inspect.unwrap; retained all15 completed generation stages and measured only remaining export/snapshot; no full retraining; intervals are staged, not continuous wall latency'))
    csv_write(E/'tables/time_failure_events.csv',events);save(E/'timing/input_bindings.json',dict(plan_sha256=ph,rows=bindings))
    pending=sum(r['status'].startswith('pending') for r in checks)
    save(E/'validation/time_audit.json',dict(status='passed' if not pending else 'in_progress',planned=20,terminal_verified=20-pending,rows=checks,infrastructure_events=events,script_sha256=sha(__file__),finished_at=time.time()))
    print(dict(terminal_verified=20-pending,pending=pending,infrastructure_events=len(events)))
if __name__=='__main__':main()
