"""AH estimated-source mapping, rendering and certificates in one reusable adapter.

Keep AHC orientation and physical scale. Transfer the native origin plane through
the recorded input crop to true intrinsics; do not claim all-depth equivalence.
No RGB/mask/score fitting, predicted-bbox scaling, or articulation edits.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from audit_static_source_cameras_v1 import H, sha, read, checked
from prepare_ah_source_registration_pilot_v1 import initialization

PY = Path(_hmc_expand('${CONDA_ROOT}/envs/ab_ah/bin/python'))
ROOT = H / 'reports/ah_source_registration_current'
CONFIG = H / 'configs/human_source_common_v1/ah_registration_current'
EXPORT = H / 'exports/ah_source_camera_current'
STATE = H / 'state/ah_source_batch.json'
ALLOWED_GPUS = ['4', '5', '6', '7', '2', '3']
LIMITATION = ('AH predicts a different focal length from nuScenes. A proper similarity '
               'preserves AHC orientation/size and transfers its origin plane through the '
               'original input crop; off-plane projections need not equal native-camera '
               'projections. This is a fixed estimated-source adaptation, not independently '
               'verified metric body geometry or held-out evaluation.')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def current_reference():
    active = read(H/'inputs/active_human_benchmark.json')
    checked(active['selection'], active['selection_sha256'])
    summary = read(H/'reports/source_camera_metric_cohort_v3/summary.json')
    qp = Path(summary['qualification']); checked(qp, summary['qualification_sha256'])
    q = read(qp); assert q['selection_sha256'] == active['selection_sha256']
    return active, qp, {r['id']: r for r in q['objects']}


def prepare_one(ref):
    from kiui.cam import orbit_camera
    sid = ref['id']
    audit_path = H/'reports'/('static_source_camera_replacement_v3' if sid == 'HN0646' else 'static_source_camera_audit_v1')/(sid+'.json')
    audit = read(audit_path); ah = audit['ah']; cp = Path(ref['camera_contract'])
    checked(cp, ref['camera_contract_sha256']); c = read(cp)
    assert audit['identity_key'] == ref['identity_key'] == c['identity_key']
    assert audit['timestamp_us'] == ref['timestamp_us'] == c['timestamp_us']
    historical = read(audit['historical_camera'])
    for key in ['K_raw','T_world_from_camera_opencv','render_wh']:
        assert c[key] == historical[key], key
    inp = read(ah['input_manifest'])
    assert inp['identity_key'] == ref['identity_key'] and inp['timestamp_us'] == ref['timestamp_us']
    assert inp['raw_source_sha256'] == ref['reference_rgb_sha256']
    checked(inp['raw_source_path'], inp['raw_source_sha256'])
    for key, digest in [('image','source_image_sha256'),('mask','source_mask_sha256')]:
        checked(inp[key], inp[digest])
    checked(ah['ply'], ah['ply_sha256']); checked(ah['source_camera'], ah['source_camera_sha256'])
    job_name = 'human_pilot_v1_HN0004' if sid == 'HN0004' else 'human_native_v1_'+sid
    reload_path = H/'runs/ah'/job_name/'reload_validation/validation.json'
    reload = read(reload_path)
    assert reload['status'] == 'passed' and reload['ply_sha256'] == ah['ply_sha256']
    direction = np.asarray(ah['source_direction'][0])
    pose = orbit_camera(-np.rad2deg(np.arcsin(direction[2])),
                        np.rad2deg(np.arctan2(direction[0], -direction[1]))+90,
                        radius=ah['normalized_camera_distance'])
    pose[:3,1:3] *= -1; view = np.linalg.inv(pose)
    scale = float(max(ah['lwh_prediction']))
    T = initialization(view, ah['K_prediction'], c['K_raw'], ah['raw_to_generation_edge_affine'], scale)
    reg = dict(id=sid, identity_key=ref['identity_key'], timestamp_us=ref['timestamp_us'],
        camera_contract=str(cp), camera_contract_sha256=sha(cp), native_ply=ah['ply'], native_ply_sha256=ah['ply_sha256'],
        ahc_camera=ah['source_camera'], ahc_camera_sha256=ah['source_camera_sha256'],
        input_manifest=ah['input_manifest'], input_manifest_sha256=sha(ah['input_manifest']),
        native_reload_validation=str(reload_path), native_reload_validation_sha256=sha(reload_path),
        audit_record=str(audit_path), audit_record_sha256=sha(audit_path),
        native_view=view.tolist(), native_K=ah['K_prediction'], physical_scale_estimate=scale,
        T_source_camera_from_export=T.tolist(), T_world_from_export=(np.asarray(c['T_world_from_camera_opencv'])@T).tolist(),
        raw_to_generation_edge_affine=ah['raw_to_generation_edge_affine'], reference_roi_xyxy=ref['roi_xyxy'],
        protocol='ah_predicted_orientation_size_origin_plane_transfer', limitations=LIMITATION,
        source_pose_estimated=True, source_rgb_used_in_AHC=True, prediction_bbox_used=False,
        pose_scale_or_metric_fit=False, joint_pose_modified=False, script_sha256=sha(__file__))
    path = CONFIG/(sid+'.json'); assert not path.exists(); save(path, reg)
    return path


def certificate(sid, reg_path, validation, render_script, reused=False):
    reg = read(reg_path); v = read(validation); c = read(reg['camera_contract'])
    checked(reg['camera_contract'],reg['camera_contract_sha256'])
    checked(render_script,v['script_sha256'])
    assert v['id']==sid and v['K_raw']==c['K_raw'] and v['render_wh']==c['render_wh']
    oldreg = read(v['registration']); checked(v['registration'],v['registration_sha256'])
    np.testing.assert_allclose(oldreg['T_source_camera_from_export'],reg['T_source_camera_from_export'],rtol=0,atol=1e-8)
    assert oldreg['native_ply_sha256']==reg['native_ply_sha256']
    assert v['full_covariance_rotated'] and v['preserved_sh_degree']==0
    for key in ['source_rgb','source_alpha']:checked(v[key],v[key+'_sha256'])
    rgb=np.asarray(Image.open(v['source_rgb']).convert('RGB'));alpha=np.asarray(Image.open(v['source_alpha']).convert('L'))
    assert list(rgb.shape[1::-1])==c['render_wh'] and rgb.shape[:2]==alpha.shape
    background_error=int((255-rgb[alpha==0]).max()) if (alpha==0).any() else 0
    assert background_error<=2, 'Output does not use own-alpha white composition'
    rec=dict(status='passed',registration_certified=True,identity=sid,identity_key=reg['identity_key'],timestamp_us=reg['timestamp_us'],
        method='ah',camera_contract=reg['camera_contract'],camera_sha256=reg['camera_contract_sha256'],
        render_validation=str(validation),render_validation_sha256=sha(validation),render_script=str(render_script),render_script_sha256=sha(render_script),
        source_rgb=v['source_rgb'],source_rgb_sha256=v['source_rgb_sha256'],source_alpha=v['source_alpha'],source_alpha_sha256=v['source_alpha_sha256'],
        prediction_encoding='white_composited_uint8',zero_alpha_background_max_byte_error=background_error,
        posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,source_pose_estimated=True,independent_world_metric_geometry_certified=False,
        registration_kind=reg['protocol'],registration_limitations=LIMITATION,real_heldout_eligible=False,
        trace_record=str(reg_path),trace_record_sha256=sha(reg_path),native_reload_validation=reg['native_reload_validation'],
        native_reload_validation_sha256=reg['native_reload_validation_sha256'],native_gaussian_covariance_and_color_retained=True,
        preserved_sh_degree=0,predicted_pose_and_size_errors_preserved=True,pixel_differences_role='diagnostic_only_not_a_gate',
        source_render_reused=reused,scope='Source-observed reconstruction after fixed AH predicted-camera adaptation; original image was used by the method',
        certification_script_sha256=sha(__file__))
    path=ROOT/sid/'ah.json';assert not path.exists();save(path,rec)
    return dict(id=sid,method='ah',status='certified_for_core_metrics',registration_record=str(path),registration_sha256=sha(path))


def render_one(sid):
    import torch
    from render_ah_source_pilot_v2 import render
    reg_path=CONFIG/(sid+'.json');reg=read(reg_path);c=read(reg['camera_contract'])
    for key in ['native_ply','ahc_camera','camera_contract','audit_record','input_manifest','native_reload_validation']:
        checked(reg[key],reg[key+'_sha256'])
    sys.path.insert(0,str(H.parent/'assetbench/repos/ah/benchmark'))
    from utils import load_ply
    gs=load_ply(reg['native_ply']).cuda()[0]
    assert gs.ndim==2 and gs.shape[1]==14 and len(gs)>0 and torch.isfinite(gs).all()
    assert (gs[:,4:7]>0).all() and (torch.linalg.vector_norm(gs[:,7:11],dim=1)>0).all()
    T=np.asarray(reg['T_source_camera_from_export']);scale=reg['physical_scale_estimate']
    view=np.eye(4);view[:3,:3]=T[:3,:3]/scale;view[:3,3]=T[:3,3]/scale
    with torch.inference_mode():rgb,alpha=render(gs,view,c['K_raw'],c['render_wh'],near=.1/scale,far=500/scale)
    out=EXPORT/sid;out.mkdir(parents=True,exist_ok=False)
    for name,value in [('source_rgb',rgb),('source_alpha',alpha)]:Image.fromarray(np.floor(value.clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)).save(out/(name+'.png'))
    validation=out/'validation.json'
    save(validation,dict(status='rendered_using_fixed_estimated_source_mapping',id=sid,registration=str(reg_path),registration_sha256=sha(reg_path),
         script_sha256=sha(__file__),kernel=str(H/'scripts/render_ah_source_pilot_v2.py'),kernel_sha256=sha(H/'scripts/render_ah_source_pilot_v2.py'),
         full_covariance_rotated=True,covariance_handling='Keep original full covariance and move the camera by equivalent rigid pose; no Gaussian edits',
         preserved_sh_degree=0,gaussians=len(gs),K_raw=c['K_raw'],render_wh=c['render_wh'],prediction_encoding='white_composited_uint8',
         source_rgb=str(out/'source_rgb.png'),source_rgb_sha256=sha(out/'source_rgb.png'),source_alpha=str(out/'source_alpha.png'),source_alpha_sha256=sha(out/'source_alpha.png'),
         registration_limitations=LIMITATION,scores_produced=False))
    certificate(sid,reg_path,validation,Path(__file__))


def acquire_slot():
    from human_native_queue_v3 import shared_slot_available,legacy_owns
    if shutil.disk_usage(H).free<11*1024**3:return None
    for gpu in ALLOWED_GPUS:
        lease=(H/'state'/('gpu_auxiliary_slot_'+gpu+'.lock')).open('a')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lease.close();continue
        ready,detail=shared_slot_available(gpu)
        if ready and detail['free_mib']>=32768 and not legacy_owns(gpu):return lease,gpu,detail
        lease.close()
    return None


def main():
    from human_native_queue_v3 import resources
    with (H/'state/ah_source_batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists(),'Inspect existing process/state before restarting'
        active,qp,refs=current_reference();assert len(refs)==100
        ROOT.mkdir(exist_ok=False);CONFIG.mkdir(exist_ok=False)
        jobs=[];rows=[]
        state=dict(status='preparing',pid=os.getpid(),started_at=time.time(),selection_sha256=active['selection_sha256'],qualification=str(qp),qualification_sha256=sha(qp),
            gpu_pool=[2,3,4,5,6,7],gpu0_gpu1_prohibited=True,max_concurrent_workers=2,disk_floor_gib=10,
            protocol='ah_predicted_orientation_size_origin_plane_transfer',limitations=LIMITATION,jobs=jobs,scores_produced=False)
        def persist():
            state['updated_at']=time.time();state['counts']={s:sum(j['status']==s for j in jobs) for s in {j['status'] for j in jobs}}
            save(STATE,state);save(ROOT/'manifest.json',dict(status=state['status'],selection=active['selection'],selection_sha256=active['selection_sha256'],
                qualification=str(qp),qualification_sha256=sha(qp),rows=rows,failures=[j for j in jobs if j['status']=='failed'],registration_limitations=LIMITATION))
        persist()
        for sid,ref in refs.items():
            job=dict(id=sid,status='queued');jobs.append(job)
            try:
                rp=prepare_one(ref)
                if sid in ['HN0004','HN0039','HN0040','HN0646']:
                    folder='ah_source_camera_replacement_v3' if sid=='HN0646' else 'ah_source_camera_pilot_v2'
                    script='render_ah_source_replacement_v3.py' if sid=='HN0646' else 'render_ah_source_pilot_v2.py'
                    rows.append(certificate(sid,rp,H/'exports'/folder/sid/'validation.json',H/'scripts'/script,True))
                    job['status']='certified_reused_render'
            except Exception as error:job.update(status='failed',reason=repr(error))
            persist()
        assert all(next(j for j in jobs if j['id']==sid)['status']=='certified_reused_render' for sid in ['HN0004','HN0039','HN0040']), 'Three-person native/source pilot reuse must pass before batch'
        pins={str(p):sha(p) for p in [Path(__file__),H/'scripts/render_ah_source_pilot_v2.py',H/'scripts/prepare_ah_source_registration_pilot_v1.py',qp,Path(active['selection'])]}
        state.update(status='running',pinned_files=pins);persist();running={}
        while any(j['status'] in ['queued','running'] for j in jobs):
            for sid,(child,lease,stream,job) in list(running.items()):
                code=child.poll()
                if code is None:continue
                stream.close();lease.close();del running[sid];job.update(exit_code=code,finished_at=time.time())
                cp=ROOT/sid/'ah.json'
                if code==0 and cp.exists():
                    rows.append(dict(id=sid,method='ah',status='certified_for_core_metrics',registration_record=str(cp),registration_sha256=sha(cp)));job['status']='certified'
                else:job.update(status='failed',reason='See log; no automatic parameter search or regeneration')
                persist()
            job=next((j for j in jobs if j['status']=='queued'),None)
            if job and len(running)<2:
                slot=acquire_slot()
                if slot:
                    lease,gpu,detail=slot
                    for p,digest in pins.items():checked(p,digest)
                    sid=job['id'];log=H/'logs'/('ah_source_batch_'+sid+'.log');stream=log.open('x')
                    env=resources.environment('ah');env.update(CUDA_VISIBLE_DEVICES=detail['uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                    child=subprocess.Popen([str(PY),'-B','-u',str(Path(__file__).resolve()),'--worker',sid],cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),start_new_session=True)
                    job.update(status='running',pid=child.pid,gpu=gpu,gpu_uuid=detail['uuid'],started_at=time.time(),log=str(log));running[sid]=(child,lease,stream,job);persist()
            if running or any(j['status']=='queued' for j in jobs):time.sleep(2)
        state.update(status='finished_with_failures' if any(j['status']=='failed' for j in jobs) else 'finished',finished_at=time.time());persist()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    if args.worker:render_one(args.worker)
    else:main()
