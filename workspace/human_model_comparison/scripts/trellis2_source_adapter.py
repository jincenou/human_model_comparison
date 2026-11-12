"""TRELLIS.2 native voxel attributes, shared keypoint registration, source camera.

The appearance variant is explicit: unlit native sRGB base color and native alpha,
eight depth layers, SSAA2. No new lighting, texture fitting or mesh simplification.
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
import trellis_source_batch as shared

H=shared.H
ROOT=H/'reports/trellis2_source_registration'
REG=H/'configs/human_source_common_v1/trellis2_source_registration'
SOURCE=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST/sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd'
PY_RENDER=_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2/bin/python')
read,save,sha,checked=shared.read,shared.save,shared.sha,shared.checked


def inputs(sid):
    active,selected=shared.selection();assert sid in selected
    q=next(r for r in read(active['reference_qualification'])['objects'] if r['id']==sid)
    checked(q['camera_contract'],q['camera_contract_sha256']);cam=read(q['camera_contract'])
    manifest=H/'inputs'/('human_static_replacement_v3' if sid=='HN0646' else 'human_static_native_v1')/'manifest.json'
    item=next(r for r in read(manifest)['objects'] if r['sample_id']==sid)
    assert item['identity_key']==q['identity_key'] and item['timestamp_us']==q['timestamp_us']
    checked(item['raw_source_path'],item['raw_source_sha256']);assert item['raw_source_sha256']==q['reference_rgb_sha256']
    folder=H/'runs/trellis2_human_native_v1'/sid;state=read(folder/'state.json');reload=read(folder/'reload_validation.json')
    assert state['sample_id']==sid and state['status']=='generated_pending_reload'
    assert reload['status']=='passed_pending_process_exit'
    assert state['input_sha256']==item['image_rgba_sha256'];checked(state['input'],state['input_sha256'])
    payload=folder/'native_voxel_mesh.pt';checked(payload,state['output_hashes'][payload.name])
    assert reload['native_checkpoint_sha256']==sha(payload)
    return active,q,cam,payload


def mesh_load(path):
    import torch
    sys.path.insert(0,str(SOURCE))
    from trellis2.representations import MeshWithVoxel
    p=torch.load(path,map_location='cpu',weights_only=True)
    for k in ['vertices','faces','coords','attrs']:
        assert torch.isfinite(p[k]).all();p[k]=p[k].cuda()
    p['layout']={k:slice(*v) for k,v in p['layout'].items()}
    return MeshWithVoxel(**p)


def basecolor_render(mesh,view,normalized_k,width,height):
    """Native rasterization and voxel lookup, with explicit own-alpha compositing."""
    import torch
    import torch.nn.functional as F
    import nvdiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
    context=dr.RasterizeCudaContext(device='cuda')
    vertices=mesh.vertices.unsqueeze(0)
    vh=torch.cat([vertices,torch.ones_like(vertices[...,:1])],dim=-1)
    proj=intrinsics_to_projection(normalized_k,.01,1000.)
    clip=vh@(proj@view).T
    wh=(height*2,width*2)
    rgb=torch.zeros((*wh,3),device='cuda');alpha=torch.zeros((*wh,1),device='cuda')
    coords=torch.cat([torch.zeros_like(mesh.coords[...,:1]),mesh.coords],dim=-1)
    with dr.DepthPeeler(context,clip,mesh.faces,wh) as peeler:
        for _ in range(8):
            rast,_=peeler.rasterize_next_layer();hit=rast[...,-1:]>0
            xyz=dr.interpolate(vertices,rast,mesh.faces)[0]
            xyz=((xyz-mesh.origin)/mesh.voxel_size).reshape(1,-1,3)
            attrs=grid_sample_3d(mesh.attrs,coords,mesh.voxel_shape,xyz,mode='trilinear')
            attrs=attrs.reshape(1,*wh,mesh.attrs.shape[-1])*hit
            color=attrs[0,...,mesh.layout['base_color']].clamp(0,1)
            opacity=attrs[0,...,mesh.layout['alpha']].clamp(0,1)
            weight=(1-alpha)*opacity;rgb+=weight*color;alpha+=weight
    rgb=rgb+1-alpha
    def down(x):
        return F.interpolate(x.permute(2,0,1)[None],(height,width),mode='bilinear',align_corners=False,antialias=True)[0]
    return down(rgb),down(alpha)[0]


def views(ids):
    import torch
    from PIL import Image
    sys.path.insert(0,str(SOURCE));from trellis2.utils import render_utils
    torch.set_num_threads(4)
    yaws=[2*np.pi*i/16 for i in range(16)]
    extr,intr=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws,[.2]*16,2.,40.)
    for sid in ids:
        active,q,cam,payload=inputs(sid);mesh=mesh_load(payload)
        out=ROOT/'observations'/sid;out.mkdir(parents=True,exist_ok=False);cases=[]
        with torch.inference_mode():
            for i,(view,K) in enumerate(zip(extr,intr)):
                rgb,alpha=basecolor_render(mesh,view,K,512,512)
                assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
                ip=out/f'view_{i:02d}.png';Image.fromarray(np.floor(rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)).save(ip)
                kn=K.cpu().numpy().astype(np.float64);v=view.cpu().numpy().astype(np.float64)
                effective=np.array([[512*kn[0,0],0,512*kn[0,2]-.5],[0,512*kn[1,1],512*(1-kn[1,2])-.5],[0,0,1.]])
                cases.append(dict(case_id=f'{sid}_orbit_{i:02d}',id=sid,role='native_asset_view',view_index=i,image=str(ip),image_sha256=sha(ip),native_payload=str(payload),native_payload_sha256=sha(payload),identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],camera=dict(yaw_radians=yaws[i],pitch_radians=.2,radius=2.,fov_degrees=40.,normalized_intrinsics=kn.tolist(),world_to_camera=v.tolist(),effective_pixel_K=effective.tolist(),projection_matrix=(effective@v[:3]).tolist(),image_wh=[512,512],source_orientation_known=False)))
                print(sid,'view',i,flush=True)
        reference=cam.get('reference_rgb',cam.get('roi_reference_rgb'));assert Image.open(reference).size==(512,512)
        l,t,r,b=cam['roi_xyxy'];scale=(r-l)/512
        cases.append(dict(case_id=sid+'_reference',id=sid,role='reference',image=reference,image_sha256=sha(reference),identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],roi_to_raw_integer_center_affine=[[scale,0,l+.5*scale-.5],[0,scale,t+.5*scale-.5],[0,0,1.]],raw_K=cam['K_raw']))
        save(out/'views.json',dict(id=sid,cases=cases,active_selection_sha256=active['selection_sha256'],script_sha256=sha(__file__),appearance_variant='native_base_color_unlit_own_alpha_depthpeel8_ssaa2',registration_cameras='same fixed16 as original TRELLIS'))
        del mesh;torch.cuda.empty_cache()


def pose(ids):
    shared.ROOT=ROOT
    shared.pose(ids)


def register(ids):
    import cv2
    from analyze_registration_keypoints_v1 import triangulate
    for sid in ids:
        active,q,cam,payload=inputs(sid);out=ROOT/'observations'/sid;kp=read(out/'keypoints.json');records=[]
        for item in kp['records']:checked(item['record'],item['sha256']);records.append(read(item['record']))
        reference=next(r for r in records if r['case']['role']=='reference');native=[r for r in records if r['case']['role']=='native_asset_view']
        try:
            assert reference['keypoints'] is not None,'source body keypoints unavailable'
            joints=[]
            for i in range(5,17):
                obs=[r for r in native if r['keypoints'] is not None and r['keypoint_scores'][i]>.5]
                if len(obs)<2 or reference['keypoint_scores'][i]<=.5:continue
                try:geom=triangulate([r['case']['camera'] for r in obs],[r['keypoints'][i] for r in obs],[r['keypoint_scores'][i] for r in obs])
                except ValueError:continue
                joints.append(dict(joint_index=i,geometry=geom))
            assert len(joints)>=6,'fewer than six body correspondences'
            xyz=np.array([x['geometry']['xyz'] for x in joints],np.float64);uv=np.array([reference['keypoints_raw_pixels'][x['joint_index']] for x in joints],np.float64);K=np.array(cam['K_raw'],np.float64)
            ok,r,t=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);assert ok
            r,t=cv2.solvePnPRefineLM(xyz,uv,K,None,r,t,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,30,1e-8))
            R=cv2.Rodrigues(r)[0];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t[:,0]
            assert np.isfinite(T).all() and np.linalg.det(R)>.999 and np.all((xyz@R.T+t[:,0])[:,2]>0)
            pixels=cv2.projectPoints(xyz,r,t,K,None)[0][:,0];error=np.linalg.norm(pixels-uv,axis=-1);rp=out/(sid+'_reference.json')
            record=dict(id=sid,identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],method='trellis2',selection_sha256=active['selection_sha256'],camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],native_payload=str(payload),native_payload_sha256=sha(payload),source_keypoints=str(rp),source_keypoints_sha256=sha(rp),T_source_camera_from_export=T.tolist(),xyz_native=xyz.tolist(),joint_indices=[x['joint_index'] for x in joints],target_pixels_raw=uv.tolist(),projected_pixels_raw=pixels.tolist(),reprojection_error_px=error.tolist(),reference_metric_values_used=False,prediction_bbox_used=False,asset_modified=False,predicted_joint_pose_unchanged=True,source_keypoints_used_for_registration=True,posthoc_pose_or_scale_fit=True,metric_world_scale_certified=False,real_heldout_eligible=False,scale_gauge=1.,script_sha256=sha(__file__))
            save(REG/(sid+'.json'),record);save(out/'registration_status.json',dict(status='ready',id=sid,joints=len(joints),mean_error_px=float(error.mean())))
            print(sid,'registered',len(joints),'joints',flush=True)
        except Exception as e:
            save(out/'registration_status.json',dict(status='unavailable',id=sid,reason=repr(e)));print(sid,'unavailable',repr(e),flush=True)


def projection_probe(view,K,width,height):
    """Verify actual interpolated camera rays; binary edge centroids can alias."""
    import torch
    import nvdiffrast.torch as dr
    from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
    kn=torch.tensor([[K[0,0]/width,0,(K[0,2]+.5)/width],[0,K[1,1]/height,1-(K[1,2]+.5)/height],[0,0,1]],device='cuda',dtype=torch.float32)
    perspective=intrinsics_to_projection(kn,.01,1000.);context=dr.RasterizeCudaContext(device='cuda');checks=[]
    yy,xx=torch.meshgrid(torch.arange(height,device='cuda'),torch.arange(width,device='cuda'),indexing='ij')
    for u,v in [(width*.25,height*.25),(width*.5,height*.5),(width*.72,height*.76)]:
        # Symmetric projected diamond around a known integer-centre target.
        pixels=np.array([[u-12,v],[u,v-12],[u+12,v],[u,v+12]])
        camera=np.c_[pixels,np.ones(4)]@np.linalg.inv(K).T*2
        native=(camera-view[:3,3])@view[:3,:3]
        verts=torch.tensor(np.c_[native,np.ones(4)],device='cuda',dtype=torch.float32)[None]
        faces=torch.tensor([[0,1,2],[0,2,3]],device='cuda',dtype=torch.int32)
        rast,_=dr.rasterize(context,verts@(perspective@torch.tensor(view,device='cuda',dtype=torch.float32)).T,faces,(height,width))
        mask=(rast[0,...,-1]>0).float();centre=torch.stack([(mask*xx).sum(),(mask*yy).sum()])/mask.sum()
        err=float(np.abs(centre.cpu().numpy()-[u,v]).max())
        # Project interpolated camera XYZ back through raw K at every covered
        # pixel. This distinguishes a camera shift from top-left edge coverage.
        xyz=dr.interpolate(torch.tensor(camera,device='cuda',dtype=torch.float32)[None],rast,faces)[0][0]
        projected=xyz@torch.tensor(K,device='cuda',dtype=torch.float32).T
        uv=projected[...,:2]/projected[...,2:].clamp_min(1e-10)
        error=(uv-torch.stack([xx,yy],dim=-1))[mask.bool()]
        ray_error=float(error.abs().max());assert ray_error<.002, f'Raw camera ray mismatch: {ray_error}'
        checks.append(dict(expected=[u,v],binary_coverage_centroid=centre.cpu().tolist(),centroid_error_px_diagnostic=err,
                           covered_pixels=int(mask.sum()),max_ray_reprojection_error_px=ray_error))
    return checks


def render(ids):
    import torch
    from PIL import Image
    torch.set_num_threads(4);sys.path.insert(0,str(SOURCE))
    for sid in ids:
        if read(ROOT/'observations'/sid/'registration_status.json')['status']!='ready':continue
        active,q,cam,payload=inputs(sid);reg=read(REG/(sid+'.json'));mesh=mesh_load(payload)
        width,height=cam['render_wh'];K=np.array(cam['K_raw']);T=np.array(reg['T_source_camera_from_export'])
        probe=projection_probe(T,K,width,height)
        kn=torch.tensor([[K[0,0]/width,0,(K[0,2]+.5)/width],[0,K[1,1]/height,1-(K[1,2]+.5)/height],[0,0,1]],device='cuda',dtype=torch.float32)
        with torch.inference_mode():rgb,alpha=basecolor_render(mesh,torch.tensor(T,device='cuda',dtype=torch.float32),kn,width,height)
        assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
        out=ROOT/'renders'/sid;out.mkdir(parents=True,exist_ok=False)
        for name,array in [('source_rgb',rgb.permute(1,2,0)),('source_alpha',alpha)]:Image.fromarray(np.floor(array.clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)).save(out/(name+'.png'))
        packed=torch.load(payload,map_location='cpu',weights_only=True)
        for k in ['vertices','faces','coords','attrs']:assert torch.equal(packed[k],getattr(mesh,k).cpu()),k
        save(out/'validation.json',dict(status='registered_native_basecolor_source_rendered',id=sid,method='trellis2',camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],registration=str(REG/(sid+'.json')),registration_sha256=sha(REG/(sid+'.json')),raw_camera_projection_checks=probe,render_wh=[width,height],source_rgb=str(out/'source_rgb.png'),source_rgb_sha256=sha(out/'source_rgb.png'),source_alpha=str(out/'source_alpha.png'),source_alpha_sha256=sha(out/'source_alpha.png'),prediction_encoding='white_composited_uint8',all_geometry_and_material_attributes_unchanged=True,appearance_variant='native_sRGB_basecolor_unlit_own_alpha_depthpeel8_ssaa2',script_sha256=sha(__file__)))
        print(sid,'source render complete',flush=True)


def worker(ids,gpu):
    """Resume completed stages, keeping each native asset and its pose intact."""
    from human_native_queue_v3 import resources
    ev=resources.environment('trellis2')
    ev.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    poseenv=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
    for sid in ids:inputs(sid)
    for action,py,env in [('views',PY_RENDER,ev),('pose',shared.PY_POSE,poseenv),('register',shared.PY_CONTROL,ev),('render',PY_RENDER,ev)]:
        pending=[]
        for sid in ids:
            out=ROOT/'observations'/sid
            if action=='views':
                done=out/'views.json'
                if done.exists():
                    manifest=read(done)
                    assert manifest['active_selection_sha256']==shared.selection()[0]['selection_sha256']
                    for case in manifest['cases']:checked(case['image'],case['image_sha256'])
            elif action=='pose':
                done=out/'keypoints.json'
                if done.exists():
                    for item in read(done)['records']:checked(item['record'],item['sha256'])
            elif action=='register':done=out/'registration_status.json'
            else:
                done=ROOT/'renders'/sid/'validation.json'
                if read(out/'registration_status.json')['status']!='ready':continue
            if not done.exists():pending.append(sid)
        if not pending:continue
        assert shutil.disk_usage(H).free>10*1024**3,'Data disk below 10 GiB'
        print('STAGE',action,pending,flush=True)
        subprocess.run([py,'-B','-u',str(Path(__file__).resolve()),action,'--ids',*pending],cwd=H,env=env,check=True)


def certify(sid):
    """Certify native color/alpha and declared rigid source registration."""
    vp=ROOT/'renders'/sid/'validation.json'
    if not vp.exists():return None
    active,q,cam,payload=inputs(sid)
    v=read(vp);rp=Path(v['registration']);checked(rp,v['registration_sha256']);reg=read(rp)
    assert v['id']==reg['id']==sid and v['method']==reg['method']=='trellis2'
    assert v['camera_contract_sha256']==reg['camera_contract_sha256']==q['camera_contract_sha256']
    assert reg['identity_key']==q['identity_key'] and reg['timestamp_us']==q['timestamp_us']
    assert v['render_wh']==cam['render_wh'] and v['all_geometry_and_material_attributes_unchanged']
    assert reg['native_payload_sha256']==sha(payload) and reg['asset_modified'] is False
    for key in ['source_keypoints','native_payload']:checked(reg[key],reg[key+'_sha256'])
    for key in ['source_rgb','source_alpha']:checked(v[key],v[key+'_sha256'])
    T=np.asarray(reg['T_source_camera_from_export']);np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
    assert np.isfinite(T).all() and np.linalg.det(T[:3,:3])>0
    script=ROOT/'source_archive'/(v['script_sha256']+'.py')
    if not script.exists():
        checked(__file__,v['script_sha256']);script.parent.mkdir(exist_ok=True);script.write_bytes(Path(__file__).read_bytes())
    checked(script,v['script_sha256'])
    protocol=H/'configs/human_source_common_v1/source_keypoint_rigid_pnp_v1.json'
    appearance=H/'configs/human_source_common_v1/trellis2_source_adapter_v1.json';ap=read(appearance)
    assert ap['status']=='frozen_before_scores' and ap['appearance_variant']==v['appearance_variant']
    assert v['script_sha256'] in ap['accepted_render_script_sha256']
    implementation=ROOT/'camera_implementation_check.json';check=read(implementation)
    assert check['status']=='passed' and max(x['max_ray_reprojection_error_px'] for x in check['checks'])<.002
    cert=dict(status='passed',registration_certified=True,identity=sid,identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],method='trellis2',
        camera_sha256=q['camera_contract_sha256'],camera_contract=q['camera_contract'],
        registration_kind='source_keypoint_rigid_pnp_v1',registration_protocol=str(protocol),registration_protocol_sha256=sha(protocol),
        posthoc_pose_or_scale_fit=True,reference_metric_values_used=False,source_keypoints_used_for_registration=True,
        predicted_joint_pose_unchanged=True,prediction_bbox_used=False,real_heldout_eligible=False,
        trace_record=str(rp),trace_record_sha256=sha(rp),render_validation=str(vp),render_validation_sha256=sha(vp),
        render_script=str(script),render_script_sha256=sha(script),source_rgb=v['source_rgb'],source_rgb_sha256=v['source_rgb_sha256'],
        source_alpha=v['source_alpha'],source_alpha_sha256=v['source_alpha_sha256'],prediction_encoding='white_composited_uint8',
        appearance_variant=v['appearance_variant'],appearance_protocol=str(appearance),appearance_protocol_sha256=sha(appearance),
        camera_implementation_check=str(implementation),camera_implementation_check_sha256=sha(implementation),
        metric_world_scale_certified=False,scope='Native sRGB base color and own alpha, unlit; source reconstruction after declared rigid keypoint registration; not full PBR shading',
        selection_sha256=active['selection_sha256'])
    dest=ROOT/'certificates'/(sid+'.json')
    if dest.exists():assert read(dest)==cert
    else:save(dest,cert)
    return dict(id=sid,method='trellis2',status='certified_for_core_metrics',registration_record=str(dest),registration_sha256=sha(dest))


def publish_index():
    """Publish a hashed immutable manifest, retaining all existing methods."""
    active,selected=shared.selection();rows=[];unavailable=[]
    for sid in selected:
        entry=certify(sid)
        if entry:rows.append(entry)
        else:
            sp=ROOT/'observations'/sid/'registration_status.json'
            status=read(sp) if sp.exists() else {}
            unavailable.append(dict(id=sid,method='trellis2',status=status.get('status','native_or_source_pending'),reason=status.get('reason','source_render_pending')))
    record=dict(selection_sha256=active['selection_sha256'],rows=rows,unavailable=unavailable)
    save(ROOT/'manifest.json',record)
    digest=sha(ROOT/'manifest.json');snapshot=ROOT/'index_snapshots'/(digest+'.json')
    if not snapshot.exists():snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes((ROOT/'manifest.json').read_bytes())
    if not rows:return 0
    with (H/'reports/source_camera_metric_cohort_v3/scorer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pointer=H/'inputs/active_human_benchmark.json';current=read(pointer)
        assert current['selection_sha256']==active['selection_sha256']
        new=dict(path=str(snapshot),sha256=digest)
        indexes=[x for x in current['registration_indexes'] if not Path(x['path']).is_relative_to(ROOT)]+[new]
        if indexes!=current['registration_indexes']:
            backup=H/'inputs/active_pointer_history'/(sha(pointer)+'.json');backup.parent.mkdir(exist_ok=True)
            if not backup.exists():backup.write_bytes(pointer.read_bytes())
            current['registration_indexes']=indexes;save(pointer,current)
    return len(rows)


def launch(ids):
    from human_native_queue_v3 import resources
    ROOT.mkdir(exist_ok=True)
    with (H/'state/trellis2_source_adapter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        slot=shared.acquire()
        if slot is None:raise RuntimeError('No eligible auxiliary GPU4–7 slot now; no task launched')
        lease,gpu,detail=slot
        try:
            ev=resources.environment('trellis2');ev.update(CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            poseenv=dict(os.environ,CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
            state=dict(status='running',pid=os.getpid(),ids=ids,gpu=gpu,gpu_uuid=detail['uuid'],started_at=time.time(),script_sha256=sha(__file__),stages=[]);sp=H/'state/trellis2_source_adapter.json';save(sp,state)
            for action,py,env in [('views',PY_RENDER,ev),('pose',shared.PY_POSE,poseenv),('register',shared.PY_CONTROL,ev),('render',PY_RENDER,ev)]:
                assert shutil.disk_usage(H).free>10*1024**3
                print('STAGE',action,flush=True)
                state['stage']=action;save(sp,state)
                result=subprocess.run([py,'-B','-u',str(Path(__file__).resolve()),action,'--ids',*ids],cwd=H,env=env)
                state['stages'].append(dict(action=action,exit_code=result.returncode));save(sp,state)
                if result.returncode:raise RuntimeError(action+' failed')
            state.update(status='finished',finished_at=time.time());save(sp,state)
        except Exception as e:
            if 'state' in locals():state.update(status='failed',reason=repr(e),finished_at=time.time());save(sp,state)
            raise
        finally:lease.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','views','pose','register','render','worker','publish']);p.add_argument('--ids',nargs='+',default=['HN0004']);p.add_argument('--gpu');a=p.parse_args()
    if a.action=='worker':
        assert a.gpu;worker(a.ids,a.gpu)
    elif a.action=='publish':print('Certified',publish_index())
    else:globals()[a.action](a.ids)
