"""Resumable native per-person geometric evaluation with exact artifact bindings."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,gc,os,subprocess,sys,time,traceback,shutil
import numpy as np
from PIL import Image
from common import *
from depth import GaussianDepth,VoxelDepth,query
from pilot import measure
from geometry_core import metrics

def helper_env():
    ev=dict(os.environ,TORCH_EXTENSIONS_DIR=str(H/'cache/torch_ext/ours'),CUDA_HOME=_hmc_expand('${CONDA_ROOT}/envs/ab_eval'));ev['PATH']=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin:')+ev['PATH'];return ev

def run(job):
    import torch
    sid,method=job['id'],job['method'];out=E/'runs'/method/sid;out.mkdir(parents=True,exist_ok=True)
    rp=E/'results'/method/(sid+'.json');ph=sha(E/'protocol.json');ref=next(r for r in read(E/'reference_manifest.json')['rows'] if r['id']==sid)
    if rp.exists():
        old=read(rp)
        if old['status']=='completed':
            assert old['protocol_sha256']==ph and old['queries_sha256']==ref['queries_sha256']
            checked(old['certificate'],old['certificate_sha256']);checked(old['returns'],old['returns_sha256']);return old
        save(out/f'previous_result_{sha(rp)}.json',old)
    record=dict(id=sid,method=method,identity_key=job['identity_key'],t_ref_us=job['t_ref_us'],reference_panel=ref['candidate_panel'],protocol_sha256=ph,queries_sha256=ref['queries_sha256'],status='running',started_at=time.time(),panels={})
    save(rp,record)
    try:
        assert shutil.disk_usage(E).free>10*1024**3
        checked(ref['queries'],ref['queries_sha256']);checked(job['camera_contract'],job['camera_contract_sha256']);cam=read(job['camera_contract']);Twc=np.asarray(cam['T_world_from_camera_opencv']);arrays=dict(np.load(ref['queries']))
        scene=method in ['omnire','splatad']
        if scene:
            from scene_depth import prepare,SceneDepth
            pack,preparation=prepare(job,out);adapter=SceneDepth(pack);Tworld=np.asarray(pack['T_world_from_native']);sourceT=np.linalg.inv(Twc)@Tworld
            evidence=dict(preparation=preparation,source_certificate_sha256=job['source_certificate_sha256'])
            # Native source cameras are rectangular. Gaussian covariance
            # projection clamps off-axis ratios using BOTH image dimensions;
            # rendering a square then cropping is not an equivalence test.
            with torch.inference_mode():
                _,native_alpha,_=adapter.rasterization(*adapter.arrays,torch.zeros_like(adapter.arrays[0]),viewmats=adapter.tensor(sourceT)[None],Ks=adapter.tensor(cam['K_raw'])[None],width=cam['render_wh'][0],height=cam['render_wh'][1],near_plane=.5 if method=='splatad' else .01,far_plane=1e10 if method=='splatad' else 1000.,render_mode='RGB+ED',**adapter.options)
            alpha=native_alpha[0,...,0].cpu().numpy()
            source=read(job['source_certificate']);old=np.array(Image.open(source['source_alpha']).convert('L'));diff=abs((alpha[:old.shape[0]]*255).round().clip(0,255)-old.astype(float))
            check=dict(max=float(diff.max()),mean=float(diff.mean()));save(out/'source_alpha_check.json',check)
            assert check['max']<=2 and check['mean']<=.01,('Native source alpha mismatch',check)
            rng,al,hit,diag=query(adapter,arrays,Tworld);met,_,_,_=metrics(arrays['q'],arrays['o'],arrays['u'],arrays['r'],rng,hit)
            record['panels']['native_metric']=dict(metrics=met,T_world_from_native=Tworld.tolist(),diagnostics=diag)
            np.savez_compressed(out/'native_metric_returns.npz',range_pred=rng,alpha=al,hit=hit)
            record['panels']['native_metric'].update(returns=str(out/'native_metric_returns.npz'),returns_sha256=sha(out/'native_metric_returns.npz'))
        else:
            adapter=VoxelDepth(job) if method=='trellis2' else GaussianDepth(job);sourceT=np.asarray(job['T_source_camera_from_native'])
            evidence=dict(asset=job['asset'],asset_sha256=job['asset_sha256'],source_trace=job.get('source_trace'),source_trace_sha256=job.get('source_trace_sha256'),recovery_certificate=job.get('recovery_certificate'))
        V,oldscale=rigid_part(sourceT);axes=Twc[:3,:3]@V[:3,:3];mo=out/'extent'
        if method=='trellis_original':
            mo.mkdir(exist_ok=True);p=mo/'native_activated_arrays.npz';np.savez_compressed(p,**{k:v.detach().cpu().numpy() for k,v in zip(['means','quats','scales','opacity'],adapter.arrays)})
            save(mo/'decoded_job.json',dict(job,decoded_arrays=str(p)));save(mo/'axes.json',axes.tolist())
            subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(E/'scripts/measure_decoded.py'),str(mo)],env=helper_env(),check=True);extent=read(mo/'measurement.json')
        elif method=='splatad':
            # Helper expects prepared state in its output root; use the same run.
            subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(E/'scripts/measure_splatad.py'),str(out)],env=helper_env(),check=True);extent=read(out/'measurement.json');mo=out
        else:extent=measure(adapter,axes,mo)
        assert extent['status']=='passed',('Extent numerical convergence',extent['status'])
        n=str(extent['precision']['fine_resolution']);height=extent['measurements'][n]['vertical_extent'];scale=ref['annotation_vertical_extent_m']/height;assert np.isfinite(scale) and scale>0
        A=Twc@np.diag([scale,scale,scale,1.])@V
        assert abs(scale*height-ref['annotation_vertical_extent_m'])<1e-9
        cert=dict(id=sid,method=method,identity_key=job['identity_key'],t_ref_us=job['t_ref_us'],protocol_sha256=ph,reference_manifest_sha256=sha(E/'reference_manifest.json'),queries_sha256=ref['queries_sha256'],evidence=evidence,camera_contract_sha256=job['camera_contract_sha256'],T_source_camera_from_native=sourceT.tolist(),source_similarity_gauge_removed=oldscale,T_world_from_native=A.tolist(),scale_m_per_native_unit=scale,native_vertical_extent=height,annotation_vertical_extent_m=ref['annotation_vertical_extent_m'],extent=str(mo/'measurement.json'),extent_sha256=sha(mo/'measurement.json'),translation_scaled=True,full_covariance_transformation='s^2 R Sigma R^T via equivalent camera',root_pose_edited=False)
        save(out/'metric_scale_certificate.json',cert)
        rng,alpha,hit,diag=query(adapter,arrays,A);met,p,dpg,dgp=metrics(arrays['q'],arrays['o'],arrays['u'],arrays['r'],rng,hit)
        np.savez_compressed(out/'returns.npz',range_pred=rng,alpha=alpha,hit=hit,predicted_endpoints=p,distances_pg=dpg,distances_gp=dgp)
        record['panels']['annotation_vertical_extent_conditioned']=dict(metrics=met,diagnostics=diag)
        record.update(status='completed',certificate=str(out/'metric_scale_certificate.json'),certificate_sha256=sha(out/'metric_scale_certificate.json'),returns=str(out/'returns.npz'),returns_sha256=sha(out/'returns.npz'),finished_at=time.time())
        del adapter;gc.collect();torch.cuda.empty_cache()
    except Exception as error:record.update(status='technical_failure_pending_review',reason=repr(error),traceback=traceback.format_exc(),finished_at=time.time())
    save(rp,record);return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--ids',nargs='*');a=p.parse_args()
    protocol=read(E/'protocol.json')
    for path,digest in protocol['code'].items():checked(path,digest)
    jobs=read(E/'intake.json')['rows'];recover=E/'registration_recovery/accepted.json'
    accepted={(r['method'],r['id']):r for r in read(recover)['rows']} if recover.exists() else {}
    for job in jobs:
        if job['method']!=a.method or (a.ids and job['id'] not in a.ids) or not job['asset_available']:continue
        if a.method not in ['omnire','splatad'] and not job.get('T_source_camera_from_native'):
            r=accepted.get((a.method,job['id']))
            if not r:continue
            job=dict(job,T_source_camera_from_native=r['T_source_camera_from_native'],recovery_certificate=r)
        result=run(job);print(job['id'],a.method,result['status'],result.get('reason'),flush=True)

if __name__=='__main__':main()
