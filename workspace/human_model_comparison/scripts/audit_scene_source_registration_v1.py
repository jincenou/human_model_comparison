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

def certify(ref,method,p,script):
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
        assert m.sha(H/'reports/model_time_audit_v1/omnire.json')==t['time_audit_sha256']
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
if __name__=='__main__':main()
