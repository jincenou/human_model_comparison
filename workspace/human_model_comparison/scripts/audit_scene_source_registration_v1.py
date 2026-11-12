"""Certify existing scene renders against frozen identity, camera and physical time.

Checks only provenance/geometry/time/encoding; never uses image metric values to
select, align, scale, exclude or improve a prediction. Keeps failed rows as NA.
"""
from pathlib import Path
import json
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import human_source_metrics_v1 as m
H=Path(__file__).resolve().parents[1]
C=H/'configs/human_source_common_v1'
OUT=H/'reports/scene_source_registration_v1'
METHODS=['ours','ah','trellis_original','omnire','splatad']

def read(p):return json.loads(Path(p).read_text())
def source_for(method,sid):
    old=H/'exports/common_camera_human_original75_time_v1'/method/sid/'validation.json'
    if old.exists():return old,H/'scripts'/f'render_{method}_original75_time_v1.py'
    if method=='omnire':return H/'exports/common_camera_human_time_v2/omnire'/sid/'validation.json',H/'scripts/render_omnire_human_time_v2.py'
    return H/'exports/common_camera_human_v1/splatad'/sid/'validation.json',H/'scripts/render_splatad_human_common_v1.py'

def projection(c,v,method):
    audit=v['common_camera_audit'];T=np.array(audit['T_model_from_world'],np.float64)
    raw_pose=np.array(c['T_world_from_camera_opencv'],np.float64);raw_K=np.array(c['K_raw'],np.float64)
    np.testing.assert_allclose(audit['K_raw'],raw_K,rtol=0,atol=1e-9)
    np.testing.assert_allclose(audit['T_world_from_camera_opencv'],raw_pose,rtol=0,atol=1e-9)
    if method=='omnire':cam=np.array(v['camera_to_world']);K=np.array(v['camera_K'])
    else:cam=np.vstack([v['source_camera_to_world'],[0,0,0,1]])@np.diag([1.,-1.,-1.,1.]);K=np.array(v['source_camera_K'])
    assert T.shape==cam.shape==(4,4) and K.shape==(3,3)
    assert np.isfinite(T).all() and np.isfinite(cam).all() and np.isfinite(K).all()
    np.testing.assert_allclose(T[3],[0,0,0,1],rtol=0,atol=1e-9)
    assert abs(np.linalg.det(T[:3,:3])-1)<1e-4,'non-unit world scale or reflection'
    np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4,rtol=0)
    w,h=c['render_wh'];pixels=np.array([[x,y,1.] for x in [0.,w*.23,w*.5,w*.81,w-1.] for y in [0.,h*.17,h*.5,h*.79,h-1.]])
    rays=(np.linalg.inv(raw_K)@pixels.T).T
    points=np.concatenate([rays*z for z in [2.,10.,50.]])
    world=(raw_pose@np.c_[points,np.ones(len(points))].T).T
    projected=(np.linalg.inv(cam)@T@world.T).T[:,:3]@K.T
    uv=projected[:,:2]/projected[:,2,None]
    error=float(np.max(np.abs(uv-np.tile(pixels[:,:2],(3,1)))))
    pose_error=float(np.max(np.abs(cam-T@raw_pose)))
    assert error<.05 and pose_error<1e-3,(error,pose_error)
    assert audit['camera_pose_refinement'] is False and audit['render_time_shift_seconds']==0 and audit['rolling_shutter_seconds']==0
    assert audit['render_wh']==c['render_wh']==v['render_wh']
    return dict(probe_count=75,projection_max_error_px=error,pose_max_error=pose_error,
                allowed_projection_error_px=.05,allowed_pose_error=1e-3,model_world_transform_determinant=float(np.linalg.det(T[:3,:3])))

def certify(ref,method,p,script,time_audit_path=None):
    v=read(p);cp=Path(ref['camera_contract']);assert m.sha(cp)==ref['camera_contract_sha256'];c=read(cp)
    assert v['status']=='passed' and v['sample_id']==ref['id']
    assert v['actor_uuid']==ref['identity_key'].split(':',1)[1]
    assert v['selection_sha256']==c['selection_sha256'] and v['frozen_raw_timestamp_us']==ref['timestamp_us']==c['timestamp_us']
    assert Path(v['source_image']).resolve()==Path(ref['reference_rgb']).resolve()
    assert m.sha(v['source_image'])==ref['reference_rgb_sha256']
    assert m.sha(script)==v['extraction_script_sha256']
    audit=v['common_camera_audit'];assert m.sha(audit['contract'])==audit['contract_sha256']
    prior=read(audit['contract']);assert prior['identity_key']==ref['identity_key'] and prior['timestamp_us']==ref['timestamp_us']
    assert v.get('stored_canonical_parameters_equal') is True,'canonical-equality evidence missing'
    assert v.get('native_checkpoint_reloaded') is True,'native-reload evidence missing'
    assert v.get('state_time_query_valid') is True,'physical-time validation missing'
    geo=projection(c,v,method)
    if method=='omnire':
        t=v['temporal_query'];assert t['query_timestamp_us']==ref['timestamp_us'] and t['query_time_error_us']==0
        assert not t['joint_pose_optimization'] and not t['checkpoint_modified']
        assert 0<=t['fraction']<=1 and max(t['endpoint_max_errors'].values())<1e-5
        assert m.sha(time_audit_path or H/'reports/model_time_audit_v1/omnire.json')==t['time_audit_sha256']
        time_error=0
    else:
        t=v['state_time_audit'];assert t['timestamp_us']==ref['timestamp_us'] and t['frozen_time_inside_native_timeline']
        assert any(t['actor_present_at_brackets']) and t['native_means_transform_max_error']<1e-4
        physical=ref['timestamp_us']/1e6-t['time_offset_seconds']
        time_error=abs(t['native_query_seconds']-physical)
        assert time_error<1e-4 and abs(time_error-t['query_error_seconds'])<1e-10
        assert t['native_bracket_times'][0]-1e-5<=t['native_query_seconds']<=t['native_bracket_times'][1]+1e-5
    rgb_path=Path(v['actor_preview']);alpha_path=Path(v['actor_alpha'])
    rgb=np.asarray(Image.open(rgb_path).convert('RGB'));alpha=np.asarray(Image.open(alpha_path).convert('L'))
    assert list(rgb.shape[1::-1])==c['render_wh'] and rgb.shape[:2]==alpha.shape
    # Zero alpha should have white RGB under the inspected native compositing code.
    background_error=int(np.max(255-rgb[alpha==0])) if np.any(alpha==0) else 0
    assert background_error<=2,('nonwhite zero-alpha pixels',background_error)
    return dict(status='passed',registration_certified=True,identity=ref['id'],identity_key=ref['identity_key'],
        timestamp_us=ref['timestamp_us'],camera_sha256=ref['camera_contract_sha256'],method=method,
        render_validation=str(p),render_validation_sha256=m.sha(p),render_script=str(script),render_script_sha256=m.sha(script),
        camera_contract=str(cp),source_rgb=str(rgb_path),source_rgb_sha256=m.sha(rgb_path),source_alpha=str(alpha_path),source_alpha_sha256=m.sha(alpha_path),
        prediction_encoding='white_composited_uint8',encoding_evidence='inspected native own-alpha white composition; raw zero-alpha pixel check',
        zero_alpha_background_max_byte_error=background_error,geometry=geo,physical_query_error_seconds=time_error,
        native_decoder_and_covariance_retained=True,posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,
        scope='source reconstruction using native learned state and global dataset coordinate frame; not held-out or independent metric body geometry',
        real_heldout_eligible=False)

def main():
    assert not OUT.exists();qpath=C/'reference_qualification_frozen_v1.json';qualification=read(qpath)
    rows=[];records=[]
    for ref in qualification['objects']:
        for method in METHODS:
            row=dict(id=ref['id'],method=method,reference_eligible=ref['reference_eligible'],metric_eligible=False)
            if not ref['reference_eligible']:row.update(status='na',reason='reference_ineligible',detail=ref['na_reasons'])
            elif method not in ['omnire','splatad']:row.update(status='na',reason='source_registration_not_yet_validated')
            else:
                p,script=source_for(method,ref['id'])
                if not p.exists():row.update(status='na',reason='native_source_render_missing')
                else:
                    try:
                        rec=certify(ref,method,p,script);records.append(rec)
                        row.update(status='certified_for_core_metrics',metric_eligible=True,registration_record=f"{ref['id']}/{method}.json")
                    except (AssertionError,KeyError,ValueError,FileNotFoundError) as e:
                        row.update(status='na',reason='registration_evidence_incomplete_or_failed',detail=str(e),render_validation=str(p))
            rows.append(row)
    OUT.mkdir()
    for rec in records:
        dest=OUT/rec['identity']/(rec['method']+'.json');dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(rec,indent=2)+'\n')
        row=next(r for r in rows if r['id']==rec['identity'] and r['method']==rec['method'])
        row['registration_record']=str(dest);row['registration_sha256']=m.sha(dest)
    summary={method:dict(Counter(r['status'] for r in rows if r['method']==method)) for method in METHODS}
    output=dict(status='scene_source_registration_audited',observed_at_utc=datetime.now(timezone.utc).isoformat(),
        qualification=str(qpath),qualification_sha256=m.sha(qpath),selected_denominator=100,methods=METHODS,rows=rows,summary=summary,
        common_camera_scope='same raw canvas K/T and physical exposure; no independent prediction recentering or size fit',
        source_reconstruction_only=True,full_five_method_comparison_complete=False,script_sha256=m.sha(__file__))
    (OUT/'manifest.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(summary),flush=True)
    failures=[r for r in rows if r.get('reason')=='registration_evidence_incomplete_or_failed'];print(json.dumps(failures[:10]),flush=True)
def recover_legacy_splatad():
    """Recover missing audit fields from native checkpoints without rerendering.

    The historical renderer already asserted exact actor parameters and physical
    camera time. Repeat those checks on CPU and query the unchanged native actor
    interpolation; retain the original validation and RGB files as evidence.
    """
    import gc
    from types import SimpleNamespace
    import torch
    from nerfstudio.model_components.dynamic_actors import DynamicActors
    from nerfstudio.models.splatad import SplatADModel
    import nerfstudio.model_components.dynamic_actors as actor_source
    import nerfstudio.models.splatad as model_source
    torch.set_num_threads(4)
    active=read(H/'inputs/active_human_benchmark.json')
    assert m.sha(active['reference_qualification'])==active['reference_qualification_sha256']
    refs={r['id']:r for r in read(active['reference_qualification'])['objects']}
    summary=read(H/'reports/source_camera_metric_cohort_v3/summary.json')
    out=H/'reports/scene_source_registration_recovered';out.mkdir(exist_ok=True)
    script=H/'scripts/render_splatad_human_common_v1.py'
    assert m.sha(script)=='febcf5fb87003ec6b0e6a3315e82e1de3a4b0794d9a0ec6754a39bdf2a7cdced'
    jobs=[]
    for row in summary['rows']:
        if row['method']!='splatad' or row['status'] in ('scored','native_unavailable_deferred'):continue
        p=H/'exports/common_camera_human_v1/splatad'/row['id']/'validation.json'
        if p.exists():jobs.append((read(p)['checkpoint'],row['id'],p))
    checkpoint_path=None;checkpoint=None;state=None;rows=[];failures=[]
    for path,sid,p in sorted(jobs):
        try:
            ref=refs[sid];v=read(p);c=read(ref['camera_contract'])
            assert ref['reference_eligible'] and v['status']=='passed' and v['sample_id']==sid
            assert v['actor_uuid']==ref['identity_key'].split(':',1)[1]
            assert v['selection_sha256']==c['selection_sha256']
            assert v['frozen_raw_timestamp_us']==ref['timestamp_us']==c['timestamp_us']
            assert Path(v['source_image']).resolve()==Path(ref['reference_rgb']).resolve()
            assert m.sha(v['source_image'])==ref['reference_rgb_sha256']
            assert m.sha(ref['camera_contract'])==ref['camera_contract_sha256']
            assert m.sha(script)==v['extraction_script_sha256']
            camera_audit=v['common_camera_audit'];assert m.sha(camera_audit['contract'])==camera_audit['contract_sha256']
            prior=read(camera_audit['contract']);assert prior['identity_key']==ref['identity_key'] and prior['timestamp_us']==ref['timestamp_us']
            geo=projection(c,v,'splatad')
            if checkpoint_path!=path:
                state=None;checkpoint=None;gc.collect()
                assert m.sha(path)==v['checkpoint_sha256']
                checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=checkpoint['pipeline'];checkpoint_path=path
            assert checkpoint['step']==v['step']==30000
            assert m.sha(v['bundle'])==v['bundle_sha256']
            bundle=torch.load(v['bundle'],map_location='cpu',weights_only=False)
            assert bundle['uuid']==v['actor_uuid'] and Path(bundle['native_checkpoint']).resolve()==Path(path).resolve()
            index=bundle['actor_index'];mask=state['_model.gauss_params.id'].reshape(-1)==index
            assert int(mask.sum())==v['gaussians']
            for name,param in bundle['gauss_params'].items():
                assert torch.equal(state['_model.gauss_params.'+name][mask],param),name
            actors=DynamicActors.__new__(DynamicActors);torch.nn.Module.__init__(actors)
            for key,value in state.items():
                if key.startswith('_model.dynamic_actors.'):
                    setattr(actors,key.removeprefix('_model.dynamic_actors.'),value)
            actors.actor_editing=dict(lateral=0.,longitudinal=0.,rotation=0.,height=0.,index=-1.)
            actors.eval()
            query=torch.tensor(v['source_camera_time'],dtype=torch.float32).reshape(1,1)
            offset=camera_audit['native_time_offset_seconds'];physical=ref['timestamp_us']/1e6-offset
            error_time=abs(float(query.item())-physical);assert error_time<1e-4
            times=actors.unique_timestamps;assert bool(torch.all(times[1:]>times[:-1]))
            assert float(times[0])<=float(query.item())<=float(times[-1])
            right=min(int(torch.searchsorted(times,query.reshape(-1))[0]),len(times)-1);left=max(right-1,0)
            with torch.no_grad():
                pose,valid=actors.get_boxes2world(query,flatten=False);assert bool(valid[0,index])
                pose=pose[0,index];means=bundle['gauss_params']['means'];ids=bundle['gauss_params']['id']
                native,_=SplatADModel._get_actor_adjusted_means(SimpleNamespace(dynamic_actors=actors,training=False),means,query,ids,calc_vels=False)
                expected=means@pose[:3,:3].T+pose[:3,3]
                mean_error=float((native-expected).abs().max());assert mean_error<1e-4
            rgbp=Path(v['actor_preview']);alphap=Path(v['actor_alpha'])
            rgb=np.asarray(Image.open(rgbp).convert('RGB'));alpha=np.asarray(Image.open(alphap).convert('L'))
            assert list(rgb.shape[1::-1])==c['render_wh'] and rgb.shape[:2]==alpha.shape
            background=int(np.max(255-rgb[alpha==0])) if np.any(alpha==0) else 0;assert background<=2
            supplement=dict(status='passed',identity=sid,actor_uuid=v['actor_uuid'],
                historical_validation=str(p),historical_validation_sha256=m.sha(p),
                checkpoint=path,checkpoint_sha256=v['checkpoint_sha256'],bundle=v['bundle'],bundle_sha256=v['bundle_sha256'],
                stored_canonical_parameters_equal=True,native_checkpoint_reloaded_on_cpu=True,
                physical_timestamp_us=ref['timestamp_us'],native_query_seconds=float(query.item()),time_offset_seconds=offset,
                query_error_seconds=error_time,native_bracket_times=[float(times[left]),float(times[right])],
                actor_present_at_brackets=[bool(actors.actor_present_at_time[left,index]),bool(actors.actor_present_at_time[right,index])],
                native_box_to_model_at_query=pose.tolist(),native_means_transform_max_error=mean_error,
                source_sha256={str(x):m.sha(x) for x in [script,Path(actor_source.__file__),Path(model_source.__file__),Path(__file__)]},
                original_rgb_unchanged=True,original_validation_unchanged=True,no_scene_training=True)
            sp=out/sid/'splatad_supplement.json';sp.parent.mkdir(exist_ok=True)
            sp.write_text(json.dumps(supplement,indent=2)+'\n')
            cert=dict(status='passed',registration_certified=True,identity=sid,identity_key=ref['identity_key'],
                timestamp_us=ref['timestamp_us'],camera_sha256=ref['camera_contract_sha256'],method='splatad',
                render_validation=str(p),render_validation_sha256=m.sha(p),render_script=str(script),render_script_sha256=m.sha(script),
                source_rgb=str(rgbp),source_rgb_sha256=m.sha(rgbp),source_alpha=str(alphap),source_alpha_sha256=m.sha(alphap),
                camera_contract=ref['camera_contract'],prediction_encoding='white_composited_uint8',
                encoding_evidence='exact historical native own-alpha white renderer inspected; zero-alpha pixels checked',
                zero_alpha_background_max_byte_error=background,geometry=geo,physical_query_error_seconds=error_time,
                native_decoder_and_covariance_retained=True,posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,
                supplementary_audit=str(sp),supplementary_audit_sha256=m.sha(sp),real_heldout_eligible=False,
                scope='source reconstruction using native actor state and dataset coordinate frame; missing historical audit fields recovered from checkpoint')
            cp=out/sid/'splatad.json';cp.write_text(json.dumps(cert,indent=2)+'\n')
            rows.append(dict(id=sid,method='splatad',status='certified_for_core_metrics',registration_record=str(cp),registration_sha256=m.sha(cp)))
            print('Recovered',sid,flush=True)
        except Exception as e:
            import traceback
            failures.append(dict(id=sid,reason=repr(e),failed_check=traceback.extract_tb(e.__traceback__)[-1].line))
            print('Unavailable',failures[-1],flush=True)
    manifest=dict(status='existing_render_evidence_recovered',selection_sha256=active['selection_sha256'],rows=rows,failures=failures,script_sha256=m.sha(__file__))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(recovered=len(rows),failures=failures)),flush=True)


def certify_dynamic_anchor(sid):
    """Use an already rendered exact-exposure OmniRe video frame as source RGB."""
    active=read(H/'inputs/active_human_benchmark.json')
    assert m.sha(active['reference_qualification'])==active['reference_qualification_sha256']
    ref=next(r for r in read(active['reference_qualification'])['objects'] if r['id']==sid)
    cp=Path(ref['camera_contract']);assert m.sha(cp)==ref['camera_contract_sha256'];c=read(cp)
    folder=H/'reports/dynamic_source_clip_v1'/sid;clip=read(folder/'clip.json')
    assert clip['selection_sha256']==active['selection_sha256'] and clip['identity_key']==ref['identity_key']
    frame=next(f for f in clip['frames'] if f['timestamp_us']==ref['timestamp_us'])
    assert frame['source_rgb_sha256']==ref['reference_rgb_sha256']==m.sha(frame['source_rgb'])
    assert Path(frame['source_rgb']).resolve()==Path(ref['reference_rgb']).resolve()
    np.testing.assert_allclose(frame['K_raw'],c['K_raw'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(frame['T_world_from_camera_opencv'],c['T_world_from_camera_opencv'],rtol=0,atol=1e-9)
    vpath=folder/'omnire/validation.json';v=read(vpath);script=H/'scripts/render_dynamic_scene_clip.py'
    assert v['id']==sid and v['method']=='omnire' and v['clip_sha256']==m.sha(folder/'clip.json')
    assert v['script_sha256']==m.sha(script) and v['provenance']['canonical_parameters_exact'] is True
    assert m.sha(v['provenance']['checkpoint'])==v['provenance']['checkpoint_sha256']
    bundle=clip['dynamic_bundles']['omnire'];assert m.sha(bundle['path'])==bundle['sha256']
    r=v['frames'][frame['index']];assert r['index']==frame['index'] and r['timestamp_us']==ref['timestamp_us']
    bracket=frame['omnire'];assert 0<=bracket['fraction']<=1 and 0<=bracket['normalized_query']<=1
    assert r['native_appearance_frame'] in [bracket['lo'],bracket['hi']]
    for key in ['rgb','alpha']:assert m.sha(r[key])==r[key+'_sha256']
    rgb=np.array(Image.open(r['rgb']).convert('RGB'));alpha=np.array(Image.open(r['alpha']).convert('L'))
    assert list(rgb.shape[1::-1])==c['render_wh']==frame['render_wh'] and rgb.shape[:2]==alpha.shape
    background=int(np.max(255-rgb[alpha==0])) if np.any(alpha==0) else 0;assert background<=2
    trace=dict(id=sid,identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],frame_index=frame['index'],
        clip=str(folder/'clip.json'),clip_sha256=m.sha(folder/'clip.json'),real_frame=frame,
        native_render_validation=str(vpath),native_render_validation_sha256=m.sha(vpath),
        canonical_checkpoint_verified=True,reference_metric_values_used=False,
        time_policy='Same physical exposure and native continuous deformation/root interpolation as the accepted real-exposure clip',
        camera_policy='Same raw K/T in the saved clip, native model/world mapping in the exact hashed renderer',
        source_script=str(script),source_script_sha256=m.sha(script),script_sha256=m.sha(__file__))
    out=H/'reports/scene_source_registration_dynamic_anchor'/sid;out.mkdir(parents=True,exist_ok=False)
    tp=out/'trace.json';tp.write_text(json.dumps(trace,indent=2)+'\n')
    cert=dict(status='passed',registration_certified=True,identity=sid,identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],
        camera_sha256=ref['camera_contract_sha256'],camera_contract=str(cp),method='omnire',registration_kind='native_dynamic_source_anchor',
        render_validation=str(vpath),render_validation_sha256=m.sha(vpath),render_script=str(script),render_script_sha256=m.sha(script),
        source_rgb=r['rgb'],source_rgb_sha256=r['rgb_sha256'],source_alpha=r['alpha'],source_alpha_sha256=r['alpha_sha256'],
        prediction_encoding='white_composited_uint8',zero_alpha_background_max_byte_error=background,
        native_decoder_and_covariance_retained=True,posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,
        physical_query_error_seconds=0,trace_record=str(tp),trace_record_sha256=m.sha(tp),real_heldout_eligible=False,
        scope='Exact frozen real exposure extracted from the native dynamic source clip; not a selected best frame or held-out observation')
    p=out/'omnire.json';p.write_text(json.dumps(cert,indent=2)+'\n')
    manifest=dict(selection_sha256=active['selection_sha256'],rows=[dict(id=sid,method='omnire',status='certified_for_core_metrics',registration_record=str(p),registration_sha256=m.sha(p))])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Certified dynamic source anchor',sid,flush=True)


if __name__=='__main__':
    import sys
    if '--recover-legacy-splatad' in sys.argv:recover_legacy_splatad()
    elif '--dynamic-anchor' in sys.argv:certify_dynamic_anchor(sys.argv[sys.argv.index('--dynamic-anchor')+1])
    else:main()
