"""Minimum scale/depth adapter pilot, deliberately does not calculate scores."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,sys,time,subprocess,os
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from common import *
from depth import GaussianDepth,VoxelDepth
from alpha_extent_measurement import measure_view,combine
from alpha_measurement_framing import gaussian_frame,depth_preserving_rig
from alpha_measurement_precision import assess,RESOLUTIONS

def combine_vertical(measures):
    """Six views are attempted; edge-on empty images do not invalidate a sheet.

    Height needs observed support in an opposing side-view pair. Rendered-empty
    views contribute no extrema. Clipped or technically failed views never pass.
    This common rule preserves scoreability of genuinely thin predictions.
    """
    names={m['name'] for m in measures if '2' in m['bounds']}
    assert any({a+'_pos',a+'_neg'}<=names for a in ['x','y']), 'No opposing vertical-support views'
    values=np.asarray([m['bounds']['2'] for m in measures if '2' in m['bounds']]);lo=float(values[:,0].min());hi=float(values[:,1].max())
    assert hi>lo
    result=dict(vertical_lower=lo,vertical_upper=hi,vertical_extent=hi-lo,vertical_support_views=sorted(names),rule='six_views_attempted; union_of_all_nonempty_vertical_alpha_support; opposing_side_pair_required')
    if len(measures)==6:result['full_dimensions']=combine(measures)
    return result

def assess_vertical(measurements,span):
    ns=sorted(map(int,measurements));out=dict(passed=False,boundary_fraction_limit=.01,fine_pixel_fraction_limit=.005)
    if len(ns)<2:return dict(out,reason='Two consecutive resolutions required')
    coarse,fine=ns[-2:];assert fine==2*coarse
    a,b=measurements[str(coarse)],measurements[str(fine)];h=b['vertical_extent']
    delta=max(abs(a[k]-b[k]) for k in ['vertical_lower','vertical_upper']);pitch=span/fine
    return dict(out,passed=delta<=.01*h and pitch<=.005*h,coarse_resolution=coarse,fine_resolution=fine,vertical_extent=h,
                boundary_max_difference=delta,fine_pixel_pitch=pitch,boundary_fraction_of_height=delta/h,fine_pixel_fraction_of_height=pitch/h)

def measure(adapter,axes,out):
    out.mkdir(parents=True,exist_ok=True)
    Q=np.eye(4);Q[:3,:3]=axes
    if adapter.arrays is not None:
        xyz,quat,scales,opacity=[x.detach().cpu().double().numpy() for x in adapter.arrays]
        rotations=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        sigma=np.sqrt(np.square(np.einsum('ij,njk->nik',axes,rotations)*scales[:,None,:]).sum(2));means=xyz@axes.T
        low=(means-6*sigma).min(0);high=(means+6*sigma).max(0)
        frame=gaussian_frame(means,sigma,opacity)
    else:
        vertices=adapter.mesh.vertices.cpu().double().numpy()@axes.T
        low=vertices.min(0);high=vertices.max(0);frame=dict(center=((low+high)/2).tolist(),span=1.2*float(np.ptp(vertices,axis=0).max()))
    data=dict(status='measuring',frame=frame,axes=axes.tolist(),measurements={},views=[],script_sha256=sha(__file__),started_at=time.time())
    for n in RESOLUTIONS:
        measures=[];errors=[]
        for view in depth_preserving_rig(frame['center'],frame['span'],n,low,high):
            ed,alpha=adapter.render(np.asarray(view['T_camera_from_semantic'])@Q,np.asarray(view['K_pixel_edges']),n,view['near'],view['far'],ortho=True)
            alpha=(alpha.clip(0,1)*255).round().astype('uint8');p=out/f'{n}_{view["name"]}.png';Image.fromarray(alpha).save(p)
            item=dict(view=view,path=str(p),sha256=sha(p),depth_diagnostics=getattr(adapter,'diagnostics',{}))
            try:item['measurement']=measure_view(alpha,view);measures.append(item['measurement'])
            except ValueError as error:item['error']=str(error);errors.append(str(error))
            data['views'].append(item)
        if all(e=='No alpha support above common threshold' for e in errors):
            try:data['measurements'][str(n)]=combine_vertical(measures)
            except AssertionError as error:data['vertical_error']=str(error)
        data['precision']=assess_vertical(data['measurements'],frame['span']);save(out/'measurement.json',data)
        if data['precision']['passed']:break
    data['status']='passed' if data['precision']['passed'] else 'numerical_extent_unresolved'
    data['finished_at']=time.time();save(out/'measurement.json',data)
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--id',required=True);a=p.parse_args()
    job=next(j for j in read(E/'intake.json')['rows'] if j['method']==a.method and j['id']==a.id)
    out=E/'pilot'/a.method/a.id;out.mkdir(parents=True,exist_ok=True)
    adapter=VoxelDepth(job) if a.method=='trellis2' else GaussianDepth(job)
    cam=read(job['camera_contract']);V,scale=rigid_part(job['T_source_camera_from_native'])
    axes=np.asarray(cam['T_world_from_camera_opencv'])[:3,:3]@V[:3,:3]
    if a.method=='trellis_original':
        decoded=out/'native_activated_arrays.npz'
        np.savez_compressed(decoded,**{k:v.detach().cpu().numpy() for k,v in zip(['means','quats','scales','opacity'],adapter.arrays)})
        copied=dict(job,decoded_arrays=str(decoded));save(out/'decoded_job.json',copied)
        save(out/'axes.json',axes.tolist())
        ev=dict(os.environ,TORCH_EXTENSIONS_DIR=str(H/'cache/torch_ext/ours'),CUDA_HOME=_hmc_expand('${CONDA_ROOT}/envs/ab_eval'))
        ev['PATH']=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin:')+ev['PATH']
        subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(E/'scripts/measure_decoded.py'),str(out)],env=ev,check=True)
        data=read(out/'measurement.json')
    else:data=measure(adapter,axes,out)
    checks={}
    if a.method=='trellis_original':
        n=512;K=np.asarray(cam['K_raw']).copy();K[:2]*=512/cam['render_wh'][0]
        # Use a full square virtual camera centred on the native asset.
        K=np.array([[700.,0,256],[0,700.,256],[0,0,1.]])
        views=read(H/'reports/trellis_source_batch/observations'/a.id/'views.json')['cases']
        v=np.asarray(next(x for x in views if x['role']=='native_asset_view')['camera']['world_to_camera'])
        ed,al=adapter.render(v,K,n,.01,1000)
        np.savez(out/'native_perspective_check.npz',ed=ed,alpha=al,V=v,K=K)
        subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(E/'scripts/measure_decoded.py'),str(out),'--compare'],env=ev,check=True)
        with np.load(out/'gsplat_perspective_check.npz') as z:ed2=z['ed'];al2=z['alpha']
        mask=(al>.5)&(al2>.5)
        checks=dict(alpha_max=float(abs(al-al2).max()),alpha_mean=float(abs(al-al2).mean()),depth_mean_on_joint_support=float(abs(ed[mask]-ed2[mask]).mean()),depth_max_on_joint_support=float(abs(ed[mask]-ed2[mask]).max()))
    if a.method=='trellis2':
        view=read(H/'reports/trellis2_source_registration/observations'/a.id/'views.json')['cases'][0]
        v=np.asarray(view['camera']['world_to_camera']);K=np.asarray(view['camera']['effective_pixel_K']);K[:2,2]+=.5
        ed,al=adapter.render(v,K,512,.01,1000,layers=8);diag8=adapter.diagnostics
        ed16,al16=adapter.render(v,K,512,.01,1000,layers=16);mask=(al>.5)&(al16>.5)
        from trellis2_source_adapter import basecolor_render
        import torch
        with torch.inference_mode():rgb,oldalpha=basecolor_render(adapter.mesh,adapter.tensor(v),adapter.tensor(view['camera']['normalized_intrinsics']),512,512)
        checks=dict(alpha_native_max=float(abs(al-oldalpha.cpu().numpy()).max()),alpha_native_mean=float(abs(al-oldalpha.cpu().numpy()).mean()),alpha_8_16_max=float(abs(al-al16).max()),depth_8_16_max=float(abs(ed[mask]-ed16[mask]).max()),layers8=diag8,layers16=adapter.diagnostics)
    save(out/'adapter_checks.json',dict(method=a.method,id=a.id,checks=checks,asset_sha256=job['asset_sha256'],depth_script_sha256=sha(E/'scripts/depth.py')))
    print(a.method,a.id,data['status'],checks,flush=True)

if __name__=='__main__':main()
