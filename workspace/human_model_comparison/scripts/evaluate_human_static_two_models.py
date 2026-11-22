"""Incremental frozen100 source evaluation for the two added static generators.

Reuse the accepted source-keypoint SE(3), image alignment and common-ROI ED-P
implementations. No old metric file or active-cohort pointer is modified.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse, collections, contextlib, fcntl, hashlib, importlib.util, json
import os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

H=Path(__file__).resolve().parents[1]; R=H.parent/'vehicle_model_comparison'
P=R/'model_preparation/20261109_2100_CST'
GEN=H/'runs/human_static_two_models_20261115_v1'
OUT=H/'reports/human_static_two_models_source_20261115_v1'
ROOT=OUT; METHOD=None
S=Path(__file__).resolve()
RECON=P/'sources/recon02-f6720923260927333e22761fa722144d214733d9'
ENVS=Path(_hmc_expand('${CONDA_ROOT}/envs'))
PY={'control':str(ENVS/'ab_eval/bin/python'), 'reconviagen':str(ENVS/'ab_cmp_recon02/bin/python'),
    'hunyuan3d21':str(ENVS/'ab_cmp_hunyuan21/bin/python'),
    'pose':_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'),
    'feature':str(ENVS/'ab_cmp_trellis2/bin/python')}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def checked(p,h):assert sha(p)==h,str(p)
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def configure(method):
    global ROOT,METHOD
    METHOD=method;ROOT=OUT/method
def import_path(name,p):
    spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def freeze():
    active=read(H/'inputs/active_human_benchmark.json')
    checked(active['selection'],active['selection_sha256']);checked(active['reference_qualification'],active['reference_qualification_sha256'])
    gen=read(GEN/'manifest.json');assert gen['selection_sha256']==active['selection_sha256']
    q=read(active['reference_qualification']);ids=sorted(x['id'] for x in read(active['selection'])['objects'])
    assert len(ids)==100 and 'HN0646' in ids and 'HN0035' not in ids
    p=OUT/'protocol.json'
    if p.exists():
        d=read(p);assert d['selection_sha256']==active['selection_sha256'] and d['generation_manifest_sha256']==sha(GEN/'manifest.json')
        for f,h in d['code'].items():checked(f,h)
        return d
    code=[S,H/'scripts/trellis_source_batch.py',H/'scripts/analyze_registration_keypoints_v1.py',
          H/'scripts/human_source_metrics_v1.py',H/'scripts/run_human_edp_common.py',H/'scripts/human_edp_common.py',
          R/'reports/source_appearance_alignment_20261115_v1/run_comparison.py',R/'scripts/render_registered_orbit16.py']
    d=dict(status='frozen_before_new_scores',selection=active['selection'],selection_sha256=active['selection_sha256'],
        reference_qualification=active['reference_qualification'],reference_qualification_sha256=active['reference_qualification_sha256'],
        generation_manifest=str(GEN/'manifest.json'),generation_manifest_sha256=sha(GEN/'manifest.json'),ids=ids,
        source_registration='Existing TRELLIS recipe: fixed16 orbit views; RTMDet/RTMPose; body joints5..16 confidence>.5; triangulate; >=6 corresponding joints; SQPnP and LM30 with fixed real K; single SE3 only',
        camera_gauge='Recon native Gaussian units; Hunyuan Blender-imported geometry centered and uniformly scaled to max vertex extent1 only for camera units; no physical scale recovery claimed',
        appearance=dict(reconviagen='Native full covariance SH0 Gaussian RGB and own alpha',hunyuan3d21='Native GLB PBR, fixed existing Blender Cycles16 Standard/None lights in normalized units'),
        image_protocol='Existing full-canvas own-mask centroid translation and isotropic sqrt(Agt/Apred) scaling; PSNR/SSIM/LPIPS after fixed GT crop512; unwarped IoU; empty predictions retained',
        EDP='Existing unwarped common ROI six-part DINOv3-L16 protocol, no new image alignment',
        aggregation='VersionA own-valid; no zero imputation and no all-method intersection',
        quality_policy='No score-driven orientation search, no joint deformation, no per-object quality repair or quality-based exclusion',
        code={str(f):sha(f) for f in code},created_at=time.time())
    save(p,d);save(OUT/'queue.json',dict(jobs=[dict(method=m,id=sid,status='awaiting_generation') for m in ['hunyuan3d21','reconviagen'] for sid in ids],created_at=time.time()))
    return d

def inputs(sid):
    protocol=read(OUT/'protocol.json');checked(protocol['generation_manifest'],protocol['generation_manifest_sha256'])
    checked(protocol['reference_qualification'],protocol['reference_qualification_sha256'])
    assert sid in protocol['ids']
    q=next(x for x in read(protocol['reference_qualification'])['objects'] if x['id']==sid)
    for k in ['camera_contract','reference_rgb','reference_mask']:checked(q[k],q[k+'_sha256'])
    g=next(x for x in read(protocol['generation_manifest'])['objects'] if x['sample_id']==sid)
    assert g['identity_key']==q['identity_key'] and g['timestamp_us']==q['timestamp_us']
    folder=GEN/'objects'/METHOD/sid;result=read(folder/'result.json')
    assert result['status']=='completed' and result['method']==METHOD and result['sample_id']==sid
    assert result['manifest_sha256']==protocol['generation_manifest_sha256']
    checked(result['validation'],result['validation_sha256']);assert read(result['validation'])['status']=='passed'
    st=read(folder/'state.json')
    if METHOD=='reconviagen':
        assert len(st['inputs'])==1
        checked(st['inputs'][0]['path'],st['inputs'][0]['sha256']);assert st['inputs'][0]['sha256']==g['image_rgba_sha256']
    else:
        checked(st['input'],st['input_sha256']);assert st['input_sha256']==g['image_rgba_sha256']
    payload=folder/('native_gaussians.pt' if METHOD=='reconviagen' else 'demo_textured.glb')
    checked(payload,st['output_hashes'][payload.name] if 'output_hashes' in st else st['output_sha256'])
    return q,read(q['camera_contract']),payload,result

def native_cameras():
    out=[];f=512/(2*np.tan(np.deg2rad(40)/2));K=np.array([[f,0,255.5],[0,f,255.5],[0,0,1.]])
    for i in range(16):
        yaw=i*2*np.pi/16;eye=np.array([np.sin(yaw)*np.cos(.2),np.cos(yaw)*np.cos(.2),np.sin(.2)])*2
        z=-eye/np.linalg.norm(eye);x=np.cross(z,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(z,x)
        T=np.eye(4);T[:3,:3]=np.stack([x,y,z]);T[:3,3]=-T[:3,:3]@eye
        out.append(dict(name=f'orbit_{i:02d}',K=K.tolist(),wh=[512,512],T_camera_from_object=T.tolist(),near=.01,far=1000.,yaw=yaw))
    return out

class GaussianRenderer:
    def __init__(self,path):
        import torch
        # Upstream imports wheels/vggt relative to its checkout, as do the
        # already-validated generation and independent reload adapters.
        os.chdir(RECON);sys.path.insert(0,str(RECON))
        from trellis.representations.gaussian import Gaussian
        from trellis.renderers.gaussian_render import render,intrinsics_to_projection
        self.torch=torch;self.render_native=render;self.projection=intrinsics_to_projection;self.cls=Gaussian
        self.pack=torch.load(path,map_location='cpu',weights_only=True);assert self.pack['init_params']['sh_degree']==0
        self.gs=Gaussian(**self.pack['init_params'])
        for k,v in self.pack['params'].items():assert torch.isfinite(v).all();setattr(self.gs,k,v.cuda())
        self.normalization=np.eye(4)
    def camera(self,c):
        from types import SimpleNamespace
        t=self.torch;W,Hh=c['wh'];K=np.asarray(c['K']);T=np.asarray(c['T_camera_from_object'])
        kn=t.tensor([[K[0,0]/W,0,(K[0,2]+.5)/W],[0,K[1,1]/Hh,1-(K[1,2]+.5)/Hh],[0,0,1]],device='cuda',dtype=t.float32)
        v=t.tensor(T,device='cuda',dtype=t.float32);p=self.projection(kn,c['near'],c['far'])
        cam=SimpleNamespace(image_width=W,image_height=Hh,FoVx=2*np.arctan(W/(2*K[0,0])),FoVy=2*np.arctan(Hh/(2*K[1,1])),world_view_transform=v.T.contiguous(),full_proj_transform=(p@v).T.contiguous(),camera_center=t.linalg.inv(v)[:3,3])
        pipe=SimpleNamespace(kernel_size=.1,debug=False,compute_cov3D_python=False,convert_SHs_python=False)
        return cam,pipe
    def render(self,c,alpha=True):
        t=self.torch;cam,pipe=self.camera(c)
        check={}
        with t.inference_mode():
            rgb=self.render_native(cam,self.gs,pipe,t.ones(3,device='cuda'))['render']
            a=None
            if alpha:
                # Recon v0.2 ignores its override_color argument. Invoke the
                # exact same native rasterizer with explicit constant colors,
                # without changing SH coefficients, geometry or opacity.
                from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
                settings=GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
                    tanfovx=float(np.tan(cam.FoVx*.5)),tanfovy=float(np.tan(cam.FoVy*.5)),kernel_size=pipe.kernel_size,
                    subpixel_offset=t.zeros((cam.image_height,cam.image_width,2),device='cuda'),bg=t.zeros(3,device='cuda'),
                    scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,
                    sh_degree=self.gs.active_sh_degree,campos=cam.camera_center,prefiltered=False,debug=False)
                native_alpha,_=GaussianRasterizer(raster_settings=settings)(means3D=self.gs.get_xyz,means2D=t.zeros_like(self.gs.get_xyz),
                    shs=None,colors_precomp=t.ones_like(self.gs.get_xyz),opacities=self.gs.get_opacity,
                    scales=self.gs.get_scaling,rotations=self.gs.get_rotation,cov3D_precomp=None)
                assert float((native_alpha-native_alpha[:1]).abs().max())<1e-6
                a=native_alpha[0]
                # Independent compositing check. Upstream Python returns raw
                # float RGB; native forward.cu writes C + T*background without
                # output clamp or tone mapping. The relation holds even when
                # per-Gaussian SH colors exceed one.
                black=self.render_native(cam,self.gs,pipe,t.zeros(3,device='cuda'))['render']
                trans=rgb-black;channel_error=float((trans-trans[:1]).abs().max())
                assert channel_error<1e-5,channel_error
                replay_error=float((a-(1-trans.mean(dim=0))).abs().max());assert replay_error<1e-5,replay_error
                assert float(a.min())>=-1e-5 and float(a.max())<=1+1e-5
                for name,value in self.pack['params'].items():assert t.equal(value,getattr(self.gs,name).cpu()),name
                check=dict(alpha_extraction='same native rasterization kernel, colors_precomp=ones, black background',max_transmittance_channel_difference=channel_error,max_alpha_compositing_replay_error=replay_error,native_parameters_unchanged=True,override_color_used=False)
        assert t.isfinite(rgb).all() and (a is None or t.isfinite(a).all())
        return np.floor(rgb.clamp(0,1).permute(1,2,0).cpu().numpy()*255+.5).astype('uint8'),None if a is None else np.floor(a.clamp(0,1).cpu().numpy()*255+.5).astype('uint8'),check
    def probe(self,c):
        t=self.torch;cam,pipe=self.camera(c);W,Hh=c['wh'];K=np.array(c['K']);T=np.array(c['T_camera_from_object'])
        yy,xx=t.meshgrid(t.arange(Hh,device='cuda'),t.arange(W,device='cuda'),indexing='ij');out=[]
        g=self.cls(**self.pack['init_params']);g.from_scaling(t.full((1,3),max(.002,2*float(g.mininum_kernel_size)),device='cuda'));g.from_rotation(t.tensor([[1.,0,0,0]],device='cuda'));g.from_opacity(t.tensor([[.9]],device='cuda'));g.from_features(t.zeros((1,3,1),device='cuda'))
        for u,v in [(W*.25,Hh*.25),(W*.5,Hh*.5),(W*.72,Hh*.76)]:
            cp=np.linalg.inv(K)@np.array([u,v,1.])*2.;p=(cp-T[:3,3])@T[:3,:3];g.from_xyz(t.tensor(p[None],device='cuda',dtype=t.float32))
            with t.inference_mode():a=self.render_native(cam,g,pipe,t.zeros(3,device='cuda'),override_color=t.ones_like(g.get_xyz))['render'][0]
            xy=t.stack([(a*xx).sum(),(a*yy).sum()])/a.sum();err=float(np.max(np.abs(xy.cpu().numpy()-[u,v])));assert err<.1,err
            out.append(dict(pixel=[u,v],max_error_px=err))
        return out

class MeshRenderer:
    def __init__(self,path,dest,normalization=None):
        sys.path.insert(0,str(R/'scripts'));from render_global_orbit16 import GlobalMeshRenderer
        from render_registered_orbit16 import BlenderRenderer
        import bpy
        # The importer has already transformed GLB Y-up into Blender Z-up.
        tmp=GlobalMeshRenderer(path,dest);low,high=tmp.bounds(np.eye(3));extent=high-low
        if normalization is None:
            scale=1./float(extent.max());center=(low+high)/2;A=np.eye(4);A[:3,:3]*=scale;A[:3,3]=-center*scale
        else:A=np.asarray(normalization)
        self.normalization=A
        self.renderer=BlenderRenderer(dict(asset=path,transform=A,ref={'size_length_width_height_m':[1.,1.,1.]}),dest)
    def render(self,c,alpha=True):return self.renderer.render(c,{})
    def probe(self,c):
        # BlenderRenderer independently recovers K from its actual projection
        # matrix for every render and requires error <0.001 pixels.
        return [dict(check='actual Blender camera projection K',tolerance_px=.001)]

def backend(path,dest,normalization=None):return GaussianRenderer(path) if METHOD=='reconviagen' else MeshRenderer(path,dest,normalization)

def views(ids):
    for sid in ids:
        q,cam,path,result=inputs(sid);out=ROOT/'observations'/sid;out.mkdir(parents=True,exist_ok=True)
        if (out/'views.json').exists():continue
        render=backend(path,out);cases=[]
        for i,c in enumerate(native_cameras()):
            rgb,_,check=render.render(c,alpha=False);ip=out/f'view_{i:02d}.png';Image.fromarray(rgb).save(ip)
            K=np.asarray(c['K']);T=np.asarray(c['T_camera_from_object'])
            cases.append(dict(case_id=f'{sid}_orbit_{i:02d}',id=sid,role='native_asset_view',view_index=i,image=str(ip),image_sha256=sha(ip),native_payload=str(path),native_payload_sha256=sha(path),identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],camera=dict(world_to_camera=T.tolist(),effective_pixel_K=K.tolist(),projection_matrix=(K@T[:3]).tolist(),image_wh=[512,512],source_orientation_known=False)))
        ref=cam['reference_rgb'];assert Image.open(ref).size==(512,512)
        l,t,r,b=q['roi_xyxy'];scale=(r-l)/512
        cases.append(dict(case_id=sid+'_reference',id=sid,role='reference',image=ref,image_sha256=sha(ref),identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],roi_to_raw_integer_center_affine=[[scale,0,l+.5*scale-.5],[0,scale,t+.5*scale-.5],[0,0,1.]],raw_K=cam['K_raw']))
        save(out/'views.json',dict(id=sid,method=METHOD,cases=cases,normalization=render.normalization.tolist(),payload=str(path),payload_sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(S)))
        print(sid,'views complete',flush=True);del render

def pose(ids):
    import trellis_source_batch as shared
    shared.ROOT=ROOT;shared.pose(ids)

def register(ids):
    import cv2
    from analyze_registration_keypoints_v1 import triangulate
    for sid in ids:
        q,cam,path,result=inputs(sid);out=ROOT/'observations'/sid
        if (out/'registration_status.json').exists():continue
        try:
            kp=read(out/'keypoints.json');records=[]
            for e in kp['records']:checked(e['record'],e['sha256']);records.append(read(e['record']))
            ref=next(x for x in records if x['case']['role']=='reference');native=[x for x in records if x['case']['role']=='native_asset_view']
            assert ref['keypoints'] is not None,'source keypoints unavailable'
            joints=[]
            for i in range(5,17):
                obs=[x for x in native if x['keypoints'] is not None and x['keypoint_scores'][i]>.5]
                if len(obs)<2 or ref['keypoint_scores'][i]<=.5:continue
                try:geom=triangulate([x['case']['camera'] for x in obs],[x['keypoints'][i] for x in obs],[x['keypoint_scores'][i] for x in obs])
                except ValueError:continue
                joints.append(dict(joint_index=i,geometry=geom))
            assert len(joints)>=6,'fewer than six body correspondences'
            xyz=np.array([x['geometry']['xyz'] for x in joints],np.float64);uv=np.array([ref['keypoints_raw_pixels'][x['joint_index']] for x in joints],np.float64);K=np.array(cam['K_raw'],np.float64)
            ok,rv,tv=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);assert ok
            rv,tv=cv2.solvePnPRefineLM(xyz,uv,K,None,rv,tv,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,30,1e-8))
            rot=cv2.Rodrigues(rv)[0];T=np.eye(4);T[:3,:3]=rot;T[:3,3]=tv[:,0]
            assert np.isfinite(T).all() and np.linalg.det(rot)>.999 and np.all((xyz@rot.T+tv[:,0])[:,2]>0)
            px=cv2.projectPoints(xyz,rv,tv,K,None)[0][:,0];error=np.linalg.norm(px-uv,axis=-1)
            reg=dict(id=sid,method=METHOD,identity_key=q['identity_key'],timestamp_us=q['timestamp_us'],camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],payload=str(path),payload_sha256=sha(path),T_source_camera_from_export=T.tolist(),normalization=read(out/'views.json')['normalization'],xyz_native=xyz.tolist(),joints=joints,reference_keypoints=str(out/(sid+'_reference.json')),reference_keypoints_sha256=sha(out/(sid+'_reference.json')),reprojection_error_px=error.tolist(),mean_reprojection_error_px=float(error.mean()),asset_modified=False,predicted_joint_pose_unchanged=True,posthoc_pose_or_scale_fit=True,fit_scope='one rigid transform; fixed scale gauge; no metric optimization',reference_metric_values_used=False,metric_world_scale_certified=False,protocol_sha256=sha(OUT/'protocol.json'))
            save(ROOT/'registrations'/(sid+'.json'),reg);save(out/'registration_status.json',dict(status='ready',id=sid,method=METHOD,joints=len(joints)))
        except (AssertionError,ValueError,cv2.error) as exc:
            save(out/'registration_status.json',dict(status='unavailable',id=sid,method=METHOD,reason=str(exc),traceback=traceback.format_exc()))
        print(sid,'registration',read(out/'registration_status.json')['status'],flush=True)

def source(ids):
    for sid in ids:
        status=ROOT/'observations'/sid/'registration_status.json'
        if read(status)['status']!='ready':continue
        q,cam,path,result=inputs(sid);reg=read(ROOT/'registrations'/(sid+'.json'));out=ROOT/'renders'/sid;out.mkdir(parents=True,exist_ok=True)
        if (out/'validation.json').exists():continue
        checked(path,reg['payload_sha256']);renderer=backend(path,out,reg['normalization'])
        c=dict(name='source',K=cam['K_raw'],wh=cam['render_wh'],T_camera_from_object=reg['T_source_camera_from_export'],near=.01,far=1000.)
        probe=renderer.probe(c);rgb,alpha,check=renderer.render(c)
        ip=out/'source_rgb.png';ap=out/'source_alpha.png';Image.fromarray(rgb).save(ip);Image.fromarray(alpha).save(ap)
        save(out/'validation.json',dict(status='rendered',id=sid,method=METHOD,source_rgb=str(ip),source_rgb_sha256=sha(ip),source_alpha=str(ap),source_alpha_sha256=sha(ap),prediction_encoding='white_composited_uint8',camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],registration=str(ROOT/'registrations'/(sid+'.json')),registration_sha256=sha(ROOT/'registrations'/(sid+'.json')),payload=str(path),payload_sha256=sha(path),projection_probe=probe,projection_check=check,protocol_sha256=sha(OUT/'protocol.json')))
        print(sid,'source rendered',flush=True);del renderer

def score(ids):
    import torch
    import cv2
    assert cv2.__version__=='5.0.0',('Frozen alignment runtime mismatch',cv2.__version__)
    import human_source_metrics_v1 as core
    align=import_path('shared_appearance_alignment',R/'reports/source_appearance_alignment_20261115_v1/run_comparison.py')
    cfg=read(R/'reports/source_appearance_alignment_20261115_v1/protocol.json')['lpips']
    protocol=read(OUT/'protocol.json');checked(protocol['scoring_reference_summary'],protocol['scoring_reference_summary_sha256'])
    references={r['id']:r['reference'] for r in read(protocol['scoring_reference_summary'])['rows'] if r.get('reference')}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    metric=core.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'],'cuda')
    for sid in ids:
        target=OUT/'rows'/METHOD/(sid+'.json')
        if target.exists():continue
        q,cam,payload,result=inputs(sid)
        current=references[sid]
        for k in ['identity_key','timestamp_us','camera_contract_sha256']:assert current[k]==q[k],(sid,k)
        # The final old-model table uses three reviewed mask/ROI revisions.
        # Keep the established registration camera; apply the exact same final
        # evaluation references to every newly added method.
        q=current
        for k in ['reference_rgb','reference_mask']:checked(q[k],q[k+'_sha256'])
        status=read(ROOT/'observations'/sid/'registration_status.json')
        if status['status']!='ready':
            save(target,dict(id=sid,method=METHOD,status='registration_unavailable',reason=status['reason'],metrics={k:core.na(status['reason']) for k in ['PSNR','SSIM','LPIPS','IoU']}));continue
        v=read(ROOT/'renders'/sid/'validation.json')
        for k in ['source_rgb','source_alpha']:checked(v[k],v[k+'_sha256'])
        raw=np.asarray(Image.open(q['reference_rgb']).convert('RGB'));mask=np.asarray(Image.open(q['reference_mask']).convert('L'))
        rgb=np.asarray(Image.open(v['source_rgb']).convert('RGB'));alpha=np.asarray(Image.open(v['source_alpha']).convert('L'))
        assert raw.shape==rgb.shape and mask.shape==alpha.shape
        ref=core.prepare_reference(raw,mask);assert list(ref['roi'])==q['roi_xyxy']
        pred=core.prepare_prediction(rgb,alpha,ref['roi'],'white_composited_uint8');M,info=align.alignment(mask,alpha);corrected=core.crop(align.warp(rgb,M),ref['roi'])
        metrics={k:getattr(core,k.lower())(ref['rgb'],corrected) for k in ['PSNR','SSIM']};metrics['LPIPS']=metric(ref['rgb'],corrected);metrics['IoU']=core.iou(ref['mask'],pred['mask']);metrics['IoU_raw']=core.iou(mask,alpha)
        original={k:getattr(core,k.lower())(ref['rgb'],pred['rgb']) for k in ['PSNR','SSIM']};original['LPIPS']=metric(ref['rgb'],pred['rgb'])
        media=ROOT/'metric_images'/sid;media.mkdir(parents=True,exist_ok=True);files={}
        for name,a in [('reference',ref['rgb']),('reference_mask',ref['mask']),('prediction_unwarped',pred['rgb']),('prediction_mask',pred['mask']),('prediction_aligned',corrected)]:
            p=media/(name+'.png');Image.fromarray(a).save(p);files[name]=str(p);files[name+'_sha256']=sha(p)
        save(target,dict(status='scored',id=sid,method=METHOD,metrics=metrics,original=original,reference=q,source_render=v,files=files,transform=M.tolist(),alignment=info,alignment_runtime=dict(executable=sys.executable,opencv=cv2.__version__,numpy=np.__version__),protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=read(OUT/'protocol.json')['selection_sha256'],real_heldout_eligible=False))
        print(sid,'scored',flush=True)

def edp_prepare(ids,group):
    old=H/'reports/source_edp_common_v1/cohort_reference_revision_v2';oldcases={c['key']:c for c in read(old/'inputs.json')['cases']}
    root=ROOT/'edp'/group;root.mkdir(parents=True,exist_ok=True)
    if (root/'inputs.json').exists():return
    cases=[];valid=[]
    for sid in ids:
        row=read(OUT/'rows'/METHOD/(sid+'.json'))
        if row['status']!='scored':continue
        ref=oldcases[sid+'_reference'];f=row['files'];assert f['reference_sha256']==ref['image_sha256'] and f['reference_mask_sha256']==ref['mask_sha256']
        valid.append(sid);cases.append(ref);cases.append(dict(key=sid+'_'+METHOD,id=sid,method=METHOD,image=f['prediction_unwarped'],image_sha256=f['prediction_unwarped_sha256'],mask=f['prediction_mask'],mask_sha256=f['prediction_mask_sha256'],reference_quality_flags=ref['reference_quality_flags'],reference_excluded_parts=ref['reference_excluded_parts'],evidence=dict(score=str(OUT/'rows'/METHOD/(sid+'.json')),score_sha256=sha(OUT/'rows'/METHOD/(sid+'.json')))))
    for k in ['poses','features']:(root/k).mkdir(exist_ok=True)
    (root/'protocol.json').write_bytes((old/'protocol.json').read_bytes());save(root/'inputs.json',dict(ids=valid,cases=cases,protocol_sha256=sha(root/'protocol.json'),selection_sha256=read(OUT/'protocol.json')['selection_sha256']))

def edp(action,group):
    import run_human_edp_common as original
    root=ROOT/'edp'/group;manifest=read(root/'inputs.json')
    if not manifest['ids']:
        save(root/(action+'_empty.json'),dict(status='no_scored_pairs'));return
    original.METHODS=[METHOD]
    getattr(original,action)(root)

def summary():
    groups={};coverage={}
    jobs=read(OUT/'queue.json')['jobs'];settlement={}
    for method in ['hunyuan3d21','reconviagen']:
        rows=[read(p) for p in (OUT/'rows'/method).glob('*.json')];cells={}
        for metric in ['PSNR','SSIM','LPIPS','IoU','IoU_raw']:
            valid=[r for r in rows if r.get('metrics',{}).get(metric,{}).get('status')=='ok'];vals=[r['metrics'][metric]['value'] for r in valid]
            cells[metric]=dict(value=float(np.mean(vals)) if vals else None,n=len(vals),ids=sorted(r['id'] for r in valid))
        edprows=[r for p in (OUT/method/'edp').glob('*/results.json') for r in read(p)['rows']]
        assert len({r['id'] for r in edprows})==len(edprows),'Duplicate ED-P identities'
        ev=[r for r in edprows if r['score']['status']=='ok'];cells['EDP']=dict(value=float(np.mean([r['score']['value'] for r in ev])) if ev else None,n=len(ev),ids=sorted(r['id'] for r in ev),parts=sum(r['valid_part_count'] for r in ev))
        groups[method]=cells;coverage[method]=dict(collections.Counter(r['status'] for r in rows))
        counts=dict(collections.Counter(j['status'] for j in jobs if j['method']==method))
        pending=sum(counts.get(s,0) for s in ['awaiting_generation','running'])
        failed=counts.get('evaluation_stage_failed',0)
        settlement[method]=dict(status='completed' if not pending and not failed else 'needs_execution_review' if not pending else 'incremental',pending=pending,execution_failures=failed,counts=counts,expected=100)
    completed=all(s['status']=='completed' for s in settlement.values())
    save(OUT/'summary.json',dict(status='completed' if completed else 'incremental_new_static_methods',groups=groups,coverage=coverage,settlement=settlement,updated_at=time.time(),protocol_sha256=sha(OUT/'protocol.json'),old_scores_modified=False))

@contextlib.contextmanager
def queue():
    with (OUT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);q=read(OUT/'queue.json');yield q
        q.update(updated_at=time.time(),counts=dict(collections.Counter(j['status'] for j in q['jobs'])));save(OUT/'queue.json',q)

def environment(kind,gpu):
    ev=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
    ev.pop('PYTHONPATH',None)
    if kind=='reconviagen':
        ev.update(ATTN_BACKEND='flash-attn',SPCONV_ALGO='native',TORCH_HOME=_hmc_expand('${HMC_MODEL_CACHE}/trellis_runtime/torch'))
    if kind=='hunyuan3d21':ev.update(LD_LIBRARY_PATH=str(ENVS/'ab_cmp_hunyuan21/lib'),CUDA_VISIBLE_DEVICES='')
    return ev

def worker(gpu,batch_size,pilot_only=False,wait_for_lease=False):
    freeze();OUT.mkdir(parents=True,exist_ok=True)
    locks=[P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock',H/f'state/gpu_auxiliary_slot_{gpu}.lock',Path(f'/tmp/drivelab-gpu-locks/gpu{gpu}.lock')]
    def ready(claim):
            gen={(j['method'],j['sample_id']):j for j in read(GEN/'queue.json')['jobs']}
            jobs=[]
            with queue() as q:
                for j in q['jobs']:
                    if pilot_only and j['id']!='HN0004':continue
                    if j['status']!='awaiting_generation':continue
                    g=gen[(j['method'],j['id'])]
                    if g['status']=='failed':j.update(status='generation_failed',reason=g.get('reason'));continue
                    if g['status']!='completed':continue
                    if jobs and j['method']!=jobs[0]['method']:continue
                    if claim:j.update(status='running',pid=os.getpid(),gpu=gpu,started_at=time.time())
                    jobs.append(dict(j))
                    if len(jobs)>=batch_size:break
                pending=any(j['status'] in ['awaiting_generation','running'] and (not pilot_only or j['id']=='HN0004') for j in q['jobs'])
            return jobs,pending
    while True:
        jobs,pending=ready(False)
        if not jobs:
            if not pending:break
            save(OUT/'workers'/f'gpu{gpu}.json',dict(status='waiting_generation_no_gpu_lease',pid=os.getpid(),gpu=gpu,time=time.time()));time.sleep(20);continue
        if shutil.disk_usage(H).free<10*1024**3:
            save(OUT/'workers'/f'gpu{gpu}.json',dict(status='waiting_disk_10gib_no_gpu_lease',pid=os.getpid(),gpu=gpu,time=time.time()));time.sleep(30);continue
        with contextlib.ExitStack() as stack:
            files=[]
            save(OUT/'workers'/f'gpu{gpu}.json',dict(status='waiting_shared_gpu_lease',pid=os.getpid(),gpu=gpu,time=time.time(),pilot_only=pilot_only))
            for p in locks:
                p.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(p.open('a'))
                fcntl.flock(f,fcntl.LOCK_EX if wait_for_lease else fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
            jobs,pending=ready(True)
            if not jobs:continue
            method=jobs[0]['method'];ids=[j['id'] for j in jobs];group=hashlib.sha256((method+','.join(ids)).encode()).hexdigest()[:16]
            configure(method);steps=[('views',method),('pose','pose'),('register','control'),('source',method),('score','control'),('edp_prepare','control'),('edp_pose','pose'),('edp_features','feature'),('edp_score','control')]
            failed=None
            for action,kind in steps:
                log=OUT/'logs'/f'{group}_{action}.log';log.parent.mkdir(exist_ok=True)
                cmd=[PY[kind],'-B','-u',str(S),action,'--method',method,'--ids',*ids,'--group',group]
                save(OUT/'workers'/f'gpu{gpu}.json',dict(status='running',pid=os.getpid(),gpu=gpu,method=method,ids=ids,stage=action,started_at=time.time(),command=cmd))
                with log.open('a') as f:
                    child=subprocess.Popen(cmd,env=environment(kind,gpu),cwd=H,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(x.fileno() for x in files));code=child.wait()
                if code:failed=dict(stage=action,exit_code=code,log=str(log));break
            with queue() as q:
                for j in q['jobs']:
                    if j['method']==method and j['id'] in ids:j.update(status='evaluation_stage_failed' if failed else 'completed',failure=failed,finished_at=time.time(),group=group)
            summary()
        # Let queued GPU users acquire the shared slot at every completed group.
        # Waiting for future model outputs happens outside all three GPU leases.
        time.sleep(1)
    save(OUT/'workers'/f'gpu{gpu}.json',dict(status='completed',pid=os.getpid(),gpu=gpu,finished_at=time.time()));summary()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','views','pose','register','source','score','edp_prepare','edp_pose','edp_features','edp_score','summary','worker']);p.add_argument('--method',choices=['hunyuan3d21','reconviagen']);p.add_argument('--ids',nargs='+');p.add_argument('--group',default='pilot');p.add_argument('--gpu',type=int);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--pilot-only',action='store_true');p.add_argument('--wait-for-lease',action='store_true');a=p.parse_args()
    if a.method:configure(a.method)
    if a.action=='prepare':freeze()
    elif a.action=='worker':worker(a.gpu,a.batch_size,a.pilot_only,a.wait_for_lease)
    elif a.action=='summary':summary()
    elif a.action=='edp_prepare':edp_prepare(a.ids,a.group)
    elif a.action.startswith('edp_'):edp(a.action[4:],a.group)
    else:globals()[a.action](a.ids)
