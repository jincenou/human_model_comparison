"""Fixed TRELLIS observations -> source-keypoint rigid registration -> real-camera RGB.

One batch adapter, existing assets only. All GPU work is restricted to 4..7
(auxiliary leases, >=32 GiB free). No image-score fitting or human joint edits.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from audit_static_source_cameras_v1 import H,TRELLIS,read,sha,checked

ROOT=H/'reports/trellis_source_batch'
REG=H/'configs/human_source_common_v1/trellis_source_registration'
PY_RENDER=_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_recon02/bin/python')
PY_POSE=_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python')
PY_CONTROL=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python')

def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def selection():
    a=read(H/'inputs/active_human_benchmark.json');checked(a['selection'],a['selection_sha256'])
    return a,{r['id']:r for r in read(a['selection'])['objects']}

def audit(sid):
    p=H/'reports'/('static_source_camera_replacement_v3' if sid=='HN0646' else 'static_source_camera_audit_v1')/(sid+'.json')
    d=read(p);checked(d['current_camera'],d['current_camera_sha256']);checked(d['trellis_original']['native_payload'],d['trellis_original']['native_payload_sha256'])
    old_camera=read(d['current_camera']);active,_=selection()
    checked(active['reference_qualification'],active['reference_qualification_sha256'])
    ref=next(r for r in read(active['reference_qualification'])['objects'] if r['id']==sid)
    checked(ref['camera_contract'],ref['camera_contract_sha256']);current=read(ref['camera_contract'])
    np.testing.assert_array_equal(old_camera['K_raw'],current['K_raw'])
    np.testing.assert_array_equal(old_camera['T_world_from_camera_opencv'],current['T_world_from_camera_opencv'])
    assert d['identity_key']==ref['identity_key'] and d['timestamp_us']==ref['timestamp_us']
    d=dict(d,current_camera=ref['camera_contract'],current_camera_sha256=ref['camera_contract_sha256'])
    return p,d,current

def views(ids):
    import torch
    from PIL import Image
    sys.path.insert(0,str(TRELLIS))
    from trellis.representations.gaussian import Gaussian
    from trellis.utils import render_utils
    active,selected=selection()
    yaws=[2*np.pi*i/16 for i in range(16)]
    extr,ins=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws,[.2]*16,2.,40.)
    for sid in ids:
        assert sid in selected and shutil.disk_usage(H).free>10*1024**3
        p,d,c=audit(sid);payload=d['trellis_original']['native_payload']
        assert d['timestamp_us']==selected[sid]['t_ref_us'] and d['identity_key']==selected[sid]['identity_key']
        out=ROOT/'observations'/sid;out.mkdir(parents=True,exist_ok=False)
        packed=torch.load(payload,map_location='cpu',weights_only=True);assert packed['init_params']['sh_degree']==0
        gs=Gaussian(**packed['init_params'])
        for name,value in packed['params'].items():assert torch.isfinite(value).all();setattr(gs,name,value.cuda())
        renderer=render_utils.get_renderer(gs,resolution=512,bg_color=(1,1,1));cases=[]
        with torch.inference_mode():
            for i,(view,K) in enumerate(zip(extr,ins)):
                rgb=renderer.render(gs,view,K)['color'];assert torch.isfinite(rgb).all()
                image=np.clip(rgb.permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8)
                ip=out/f'view_{i:02d}.png';Image.fromarray(image).save(ip)
                kn=K.cpu().numpy().astype(np.float64);v=view.cpu().numpy().astype(np.float64)
                effective=np.array([[512*kn[0,0],0,512*kn[0,2]-.5],[0,512*kn[1,1],512*(1-kn[1,2])-.5],[0,0,1.]])
                cases.append(dict(case_id=f'{sid}_orbit_{i:02d}',id=sid,role='native_asset_view',view_index=i,image=str(ip),image_sha256=sha(ip),
                    native_payload=payload,native_payload_sha256=d['trellis_original']['native_payload_sha256'],identity_key=d['identity_key'],timestamp_us=d['timestamp_us'],
                    camera=dict(yaw_radians=yaws[i],pitch_radians=.2,radius=2.,fov_degrees=40.,normalized_intrinsics=kn.tolist(),world_to_camera=v.tolist(),
                                effective_pixel_K=effective.tolist(),projection_matrix=(effective@v[:3]).tolist(),image_wh=[512,512],source_orientation_known=False)))
        for name,value in packed['params'].items():assert torch.equal(value,getattr(gs,name).cpu())
        ref=c.get('reference_rgb',c.get('roi_reference_rgb'));assert ref and Image.open(ref).size==(512,512)
        l,t,r,b=c['roi_xyxy'];scale=(r-l)/512;assert r-l==b-t
        affine=[[scale,0,l+.5*scale-.5],[0,scale,t+.5*scale-.5],[0,0,1.]]
        cases.append(dict(case_id=sid+'_reference',id=sid,role='reference',image=ref,image_sha256=sha(ref),
            identity_key=d['identity_key'],timestamp_us=d['timestamp_us'],camera_contract=d['current_camera'],camera_contract_sha256=d['current_camera_sha256'],roi_to_raw_integer_center_affine=affine,raw_K=c['K_raw']))
        save(out/'views.json',dict(id=sid,cases=cases,audit=str(p),audit_sha256=sha(p),active_selection_sha256=active['selection_sha256'],
             parameter_equality_after_render=True,script_sha256=sha(__file__),protocol='same fixed16 cameras and native truncation as pilot; no source angle selection'))
        print(sid,'fixed16 views ready',flush=True)


def pose(ids):
    import torch
    from mmdet.apis import init_detector,inference_detector
    from mmpose.apis import init_model,inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    torch.set_num_threads(4)
    prep=H/'model_preparation/20261110';source=prep/'sources/exavatar/fitting/tools/mmpose'
    configs=[source/'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',source/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py']
    weights=[prep/'weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',prep/'weights/dw-ll_ucoco_384.pth']
    hashes=['235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9','0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07']
    for p,digest in zip(weights,hashes):checked(p,digest)
    detector=init_detector(str(configs[0]),str(weights[0]),device='cuda');detector.cfg=adapt_mmdet_pipeline(detector.cfg)
    estimator=init_model(str(configs[1]),str(weights[1]),device='cuda')
    for model,weight in zip([detector,estimator],weights):
        ckpt=torch.load(weight,map_location='cpu',weights_only=False);model.load_state_dict({k.removeprefix('module.'):v for k,v in ckpt.get('state_dict',ckpt).items()},strict=True);model.eval()
    for sid in ids:
        out=ROOT/'observations'/sid;manifest=read(out/'views.json');records=[]
        for case in manifest['cases']:
            checked(case['image'],case['image_sha256'])
            with torch.inference_mode():detected=inference_detector(detector,case['image']).pred_instances.cpu().numpy()
            boxes=np.c_[detected.bboxes,detected.scores];boxes=boxes[(detected.labels==0)&(detected.scores>.3)]
            if len(boxes):boxes=boxes[nms(boxes,.3)]
            rec=dict(case=case,detected_people=len(boxes),detections_xyxy_score=boxes.tolist(),keypoints=None,keypoint_scores=None,
                     confidence_is_visibility=False,source_keypoints_are_estimates=True,device='cuda',weights_sha256=hashes)
            if len(boxes)!=1:rec.update(status='keypoints_unavailable',reason='no_person_detected' if not len(boxes) else 'multiple_person_detections')
            else:
                with torch.inference_mode():pred=inference_topdown(estimator,case['image'],boxes[:,:4])
                assert len(pred)==1;xy=np.asarray(pred[0].pred_instances.keypoints)[0];scores=np.asarray(pred[0].pred_instances.keypoint_scores)[0]
                assert xy.shape==(133,2) and scores.shape==(133,) and np.isfinite(xy).all() and np.isfinite(scores).all()
                rec.update(status='observed',keypoints=xy.tolist(),keypoint_scores=scores.tolist())
                if case['role']=='reference':
                    raw=np.c_[xy,np.ones(len(xy))]@np.asarray(case['roi_to_raw_integer_center_affine']).T;rec['keypoints_raw_pixels']=(raw[:,:2]/raw[:,2:3]).tolist()
            rp=out/(case['case_id']+'.json');save(rp,rec);records.append(dict(record=str(rp),sha256=sha(rp),role=case['role']))
        save(out/'keypoints.json',dict(id=sid,records=records,detector_score_threshold=.3,detector_nms_iou=.3,script_sha256=sha(__file__),
             config_sha256={str(p):sha(p) for p in configs},weights_sha256={str(p):sha(p) for p in weights}))
        print(sid,'body keypoint observations ready',flush=True)


def register(ids):
    import cv2
    from analyze_registration_keypoints_v1 import triangulate
    active,selected=selection();REG.mkdir(exist_ok=True)
    for sid in ids:
        out=ROOT/'observations'/sid;kp=read(out/'keypoints.json');records=[]
        for entry in kp['records']:checked(entry['record'],entry['sha256']);records.append(read(entry['record']))
        reference=next(r for r in records if r['case']['role']=='reference');native=[r for r in records if r['case']['role']=='native_asset_view']
        try:
            assert reference['keypoints'] is not None,'source keypoints unavailable'
            joints=[];failures=[]
            for i in range(5,17):
                obs=[r for r in native if r['keypoints'] is not None and r['keypoint_scores'][i]>.5]
                if len(obs)<2 or reference['keypoint_scores'][i]<=.5:continue
                try:
                    geom=triangulate([r['case']['camera'] for r in obs],[r['keypoints'][i] for r in obs],[r['keypoint_scores'][i] for r in obs])
                    joints.append(dict(joint_index=i,geometry=geom,observations=len(obs)))
                except ValueError as error:failures.append(dict(joint_index=i,reason=str(error)))
            assert len(joints)>=6,'fewer than six available body correspondences'
            xyz=np.array([r['geometry']['xyz'] for r in joints],np.float64);uv=np.array([reference['keypoints_raw_pixels'][r['joint_index']] for r in joints],np.float64)
            cp=reference['case']['camera_contract'];checked(cp,reference['case']['camera_contract_sha256']);cam=read(cp);K=np.array(cam['K_raw'],np.float64)
            assert cam['timestamp_us']==selected[sid]['t_ref_us'] and cam['identity_key']==selected[sid]['identity_key']
            success,r,t=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);assert success
            r,t=cv2.solvePnPRefineLM(xyz,uv,K,None,r,t,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,30,1e-8))
            R=cv2.Rodrigues(r)[0];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t[:,0]
            assert np.isfinite(T).all() and np.linalg.det(R)>.999 and np.all((xyz@R.T+t[:,0])[:,2]>0)
            proj=cv2.projectPoints(xyz,r,t,K,None)[0][:,0];error=np.linalg.norm(proj-uv,axis=-1)
            payload=native[0]['case']['native_payload'];checked(payload,native[0]['case']['native_payload_sha256'])
            rp=out/(sid+'_reference.json')
            reg=dict(id=sid,identity_key=cam['identity_key'],timestamp_us=cam['timestamp_us'],method='trellis_original',active_selection_sha256=active['selection_sha256'],
                camera_contract=cp,camera_contract_sha256=sha(cp),native_payload=payload,native_payload_sha256=sha(payload),T_source_camera_from_export=T.tolist(),
                working_units='native TRELLIS units; translation in those units',metric_world_scale_certified=False,scale_gauge=1.,
                protocol='Body joints5..16 confidence>0.5; fixed sourceK; one SQPnP then LM(30,1e-8); no metric-guided selection',
                source_keypoints=str(rp),source_keypoints_sha256=sha(rp),joint_indices=[r['joint_index'] for r in joints],xyz_native=xyz.tolist(),
                target_pixels_raw=uv.tolist(),projected_pixels_raw=proj.tolist(),mean_reprojection_error_px=float(error.mean()),max_reprojection_error_px=float(error.max()),
                source_keypoints_used_for_registration=True,posthoc_pose_or_scale_fit=True,fit_scope='single rigid asset-to-camera transform, no joint pose or scale fit',
                reference_metric_values_used=False,prediction_bbox_used=False,pose_errors_preserved=True,asset_modified=False,real_heldout_eligible=False,
                registration_certified=False,metric_eligible=False,keypoint_manifest=str(out/'keypoints.json'),keypoint_manifest_sha256=sha(out/'keypoints.json'),
                script_sha256=sha(__file__),opencv_version=cv2.__version__)
            dest=REG/(sid+'.json');assert not dest.exists();save(dest,reg)
            save(out/'triangulation.json',dict(joints=joints,unavailable=failures));save(out/'registration_status.json',dict(id=sid,status='ready',registration=str(dest)))
            print(sid,'source registration ready',len(joints),float(error.mean()),flush=True)
        except Exception as error:
            save(out/'registration_status.json',dict(id=sid,status='unavailable',reason=str(error)));print(sid,'registration unavailable:',str(error),flush=True)


def render(ids):
    from register_trellis_source import render as one
    for sid in ids:
        st=read(ROOT/'observations'/sid/'registration_status.json')
        if st['status']!='ready':continue
        try:
            one(sid)
            certify(sid)
        except Exception as error:save(ROOT/'observations'/sid/'render_failure.json',dict(id=sid,reason=str(error)))


def certify(sid):
    active,selected=selection()
    prior=H/'reports/trellis_source_registration/certificates'/(sid+'.json')
    if prior.exists():
        old=read(prior);assert old['selection_sha256']==active['selection_sha256']
        return dict(id=sid,method='trellis_original',status='certified_for_core_metrics',registration_record=str(prior),registration_sha256=sha(prior))
    vp=H/'reports/trellis_source_registration'/sid/'validation.json';v=read(vp)
    rp=Path(v['registration']);checked(rp,v['registration_sha256']);reg=read(rp)
    script=H/'scripts/register_trellis_source.py';checked(script,v['script_sha256'])
    refs={r['id']:r for r in read(active['reference_qualification'])['objects']};ref=refs[sid]
    checked(ref['camera_contract'],ref['camera_contract_sha256']);c=read(ref['camera_contract'])
    assert v['camera_contract_sha256']==reg['camera_contract_sha256']==ref['camera_contract_sha256']
    assert reg['identity_key']==ref['identity_key'] and reg['timestamp_us']==ref['timestamp_us']==selected[sid]['t_ref_us']
    np.testing.assert_array_equal(v['K_raw'],c['K_raw']);assert v['render_wh']==c['render_wh']
    assert v['gaussian_parameters_unchanged'] and v['full_covariance_preserved'] and v['sh_degree']==0
    for key in ['native_payload','source_keypoints']:checked(reg[key],reg[key+'_sha256'])
    for key in ['source_rgb','source_alpha']:checked(v[key],v[key+'_sha256'])
    T=np.asarray(reg['T_source_camera_from_export']);np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
    assert np.isfinite(T).all() and np.linalg.det(T[:3,:3])>0
    protocol=H/'configs/human_source_common_v1/source_keypoint_rigid_pnp_v1.json'
    cert=dict(status='passed',registration_certified=True,identity=sid,identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],method='trellis_original',
        camera_sha256=ref['camera_contract_sha256'],camera_contract=ref['camera_contract'],
        registration_kind='source_keypoint_rigid_pnp_v1',registration_protocol=str(protocol),registration_protocol_sha256=sha(protocol),
        posthoc_pose_or_scale_fit=True,reference_metric_values_used=False,source_keypoints_used_for_registration=True,
        predicted_joint_pose_unchanged=True,prediction_bbox_used=False,real_heldout_eligible=False,
        trace_record=str(rp),trace_record_sha256=sha(rp),render_validation=str(vp),render_validation_sha256=sha(vp),
        render_script=str(script),render_script_sha256=sha(script),source_rgb=v['source_rgb'],source_rgb_sha256=v['source_rgb_sha256'],
        source_alpha=v['source_alpha'],source_alpha_sha256=v['source_alpha_sha256'],prediction_encoding='white_composited_uint8',
        metric_world_scale_certified=False,scope='Observed source appearance after declared rigid keypoint registration; no joint edits or quality-based exclusion',
        selection_sha256=active['selection_sha256'],certification_script_sha256=sha(__file__))
    dest=ROOT/'certificates'/(sid+'.json')
    if dest.exists():assert read(dest)==cert
    else:save(dest,cert)
    return dict(id=sid,method='trellis_original',status='certified_for_core_metrics',registration_record=str(dest),registration_sha256=sha(dest))


def index():
    active,selected=selection();rows=[];unavailable=[]
    for sid in selected:
        cert=H/'reports/trellis_source_registration/certificates'/(sid+'.json')
        if not cert.exists():cert=ROOT/'certificates'/(sid+'.json')
        if cert.exists():rows.append(dict(id=sid,method='trellis_original',status='certified_for_core_metrics',registration_record=str(cert),registration_sha256=sha(cert)))
        else:
            rp=ROOT/'observations'/sid/'render_failure.json';sp=ROOT/'observations'/sid/'registration_status.json'
            reason=read(rp).get('reason') if rp.exists() else read(sp).get('reason','render_pending') if sp.exists() else 'observation_stage_pending'
            unavailable.append(dict(id=sid,method='trellis_original',reason=reason))
    save(ROOT/'manifest.json',dict(selection=active['selection'],selection_sha256=active['selection_sha256'],rows=rows,unavailable=unavailable,
        posthoc_pose_or_scale_fit=True,reference_metric_values_used=False,registered=len(rows),updated_at=time.time()))
    return len(rows)


def acquire():
    from human_native_queue_v3 import shared_slot_available,legacy_owns
    if shutil.disk_usage(H).free<=11*1024**3:return None
    for i in range(4,8):
        lease=(H/'state'/f'gpu_auxiliary_slot_{i}.lock').open('a')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lease.close();continue
        ready,detail=shared_slot_available(str(i))
        if ready and detail['free_mib']>=32768 and not legacy_owns(str(i)):return lease,str(i),detail
        lease.close()
    return None


def worker(ids,gpu):
    from human_native_queue_v3 import resources
    env=resources.environment('trellis_original');env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',SPCONV_ALGO='native',ATTN_BACKEND='flash-attn')
    poseenv=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
    for action,py,ev in [('views',PY_RENDER,env),('pose',PY_POSE,poseenv),('register',PY_CONTROL,env),('render',PY_RENDER,env)]:
        assert shutil.disk_usage(H).free>10*1024**3,'Disk reserve reached'
        print('STAGE',action,ids,flush=True)
        subprocess.run([py,'-B','-u',str(Path(__file__).resolve()),action,'--ids',*ids],cwd=H,env=ev,check=True)


def launch(ids):
    active,selected=selection();ROOT.mkdir(exist_ok=True)
    sp=H/'state/trellis_source_batch.json'
    with (H/'state/trellis_source_batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not sp.exists(),'Audit an existing controller before restart'
        for sid in selected:
            if (H/'reports/trellis_source_registration'/sid/'validation.json').exists():certify(sid)
        index()
        remaining=ids or [sid for sid in selected if not (H/'reports/trellis_source_registration'/sid/'validation.json').exists()]
        assert all(sid in selected for sid in remaining)
        groups=[remaining[i:i+8] for i in range(0,len(remaining),8)]
        jobs=[dict(group=i,ids=group,status='queued') for i,group in enumerate(groups)]
        state=dict(pid=os.getpid(),status='running',selection=active['selection'],selection_sha256=active['selection_sha256'],jobs=jobs,gpu_pool=[4,5,6,7],max_workers=2,started_at=time.time(),script_sha256=sha(__file__))
        running={}
        def persist():state['updated_at']=time.time();save(sp,state)
        persist()
        while any(j['status'] in ('queued','running') for j in jobs):
            for i,(child,lease,stream,j) in list(running.items()):
                code=child.poll()
                if code is None:continue
                stream.close();lease.close();del running[i];j.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());state['certified']=index();persist()
            j=next((x for x in jobs if x['status']=='queued'),None)
            if j and len(running)<2:
                slot=acquire()
                if slot:
                    lease,gpu,detail=slot;assert sha(__file__)==state['script_sha256'];checked(active['selection'],active['selection_sha256'])
                    log=H/'logs'/f'trellis_source_batch_group{j["group"]:02d}.log';stream=log.open('x')
                    child=subprocess.Popen([PY_CONTROL,'-B','-u',str(Path(__file__).resolve()),'worker','--gpu',detail['uuid'],'--ids',*j['ids']],cwd=H,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),start_new_session=True)
                    j.update(status='running',pid=child.pid,gpu=gpu,gpu_uuid=detail['uuid'],started_at=time.time(),log=str(log));running[j['group']]=child,lease,stream,j;persist()
            if running or any(j['status']=='queued' for j in jobs):time.sleep(3)
        state.update(status='finished',finished_at=time.time(),certified=index());persist()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['views','pose','register','render','worker','launch']);ap.add_argument('--ids',nargs='*');ap.add_argument('--gpu');a=ap.parse_args()
    if a.action=='launch':launch(a.ids)
    elif a.action=='worker':worker(a.ids,a.gpu)
    else:globals()[a.action](a.ids)
