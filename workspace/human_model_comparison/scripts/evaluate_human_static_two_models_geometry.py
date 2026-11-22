"""Incremental added-method human observed-surface CD/F-score, old protocol reused.

The source SE(3) is independently estimated from body keypoints. A single scale
uses annotated world-vertical extent, including scaled source translation. No
LiDAR fitting, posed-body edits, surface cleaning or generation-time retiming.
"""
import argparse,contextlib,collections,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
HG=H/'evaluation_static_cd_fscore_time_v1';G=R/'evaluation_geometry_expanded_20261114_v2'
OUT=H/'reports/human_static_two_models_geometry_20261115_v1';S=Path(__file__).resolve()
sys.path[:0]=[str(HG/'scripts'),str(H/'scripts'),str(G/'scripts'),str(R/'scripts'),str(R/'evaluation_nuscenes_v1/scripts')]
from common import read,sha,checked,save,rigid_part
from pilot import combine_vertical,assess_vertical
from depth import query
from geometry_core import metrics,selftest
from alpha_extent_measurement import measure_view
from alpha_measurement_framing import depth_preserving_rig
from alpha_measurement_precision import RESOLUTIONS
import evaluate_human_static_two_models as source
METHODS=['hunyuan3d21','reconviagen']

def prepare():
    old=read(HG/'protocol.json');refs=read(HG/'reference_manifest.json');src=read(source.OUT/'protocol.json')
    checked(HG/'reference_manifest.json',old['reference_manifest_sha256'])
    for path,digest in old['code'].items():checked(path,digest)
    assert src['selection_sha256']==old['selection_sha256']==refs['selection_sha256']
    assert set(src['ids'])=={r['id'] for r in refs['rows']} and len(src['ids'])==100
    assert collections.Counter(r['candidate_panel'] for r in refs['rows'])=={'main':84,'low_support_exploratory':16}
    p=OUT/'protocol.json'
    if p.exists():
        d=read(p)
        for path,digest in d['code'].items():checked(path,digest)
        checked(d['reference_manifest'],d['reference_manifest_sha256']);return d
    OUT.mkdir(parents=True,exist_ok=True)
    code={**old['code'],str(S):sha(S),str(source.S):sha(source.S)}
    for f in [G/'scripts/depth_adapters.py',G/'scripts/measure_registration.py',HG/'scripts/measure_decoded.py',HG/'scripts/common.py']:
        code[str(f)]=sha(f)
    d=dict(status='frozen_before_new_geometry_scores',version='human_two_added_static_methods_geometry_v1',selection_sha256=src['selection_sha256'],ids=src['ids'],reference_manifest=str(HG/'reference_manifest.json'),reference_manifest_sha256=sha(HG/'reference_manifest.json'),inherited_protocol=str(HG/'protocol.json'),inherited_protocol_sha256=sha(HG/'protocol.json'),rules={k:old[k] for k in ['reference','reference_qualification','main_panel','scale_rule','annotation_information','prohibited','native_extent','depth','metrics']},methods=METHODS,adapters=dict(hunyuan3d21='Native GLB PBR alpha extent in Blender world; exact GLB triangle/BVH queries with native material opacity and doubleSided, existing mesh adapter; GLB to Blender rotation applied exactly once',reconviagen='Original Recon native Gaussian parameters and rasterizer; explicit precomputed [1,z,0] for alpha and accumulated z; existing gsplat AA eps=.1 orthographic extent only, matching original TRELLIS metrology'),source_registration='New shared fixed16 views/body-keypoint SE3 source registrations; exact camera identity and payload content pinned per result',known_scale='Annotation world-vertical extent conditioned; not scale-free and not independent recovered metric scale',aggregation='Own-valid within each fixed reference support panel; equal person weight; valid empty F1=0, CD null and extended mean infinity; execution failures excluded and separately counted',generation_time='Not remeasured; retained original generation logs only',code=code,created_at=time.time(),selftest=selftest())
    save(p,d);save(OUT/'queue.json',dict(jobs=[dict(method=m,id=s,status='awaiting_source') for m in METHODS for s in src['ids']]))
    return d

def mesh_extent(path,axes,out):
    from measure_registration import mesh_renderer
    out.mkdir(parents=True,exist_ok=True);low,high,render=mesh_renderer(path,axes,out)
    frame=dict(center=((low+high)/2).tolist(),span=1.2*float((high-low).max()),rule='all_native_mesh_vertices_for_framing_only')
    data=dict(status='measuring',frame=frame,axes=axes.tolist(),measurements={},views=[],script_sha256=sha(S),started_at=time.time())
    for n in RESOLUTIONS:
        ms=[];errors=[]
        for view in depth_preserving_rig(frame['center'],frame['span'],n,low,high):
            alpha=render(view);p=out/f'{n}_{view["name"]}.png';Image.fromarray(alpha).save(p)
            item=dict(view=view,path=str(p),sha256=sha(p))
            try:item['measurement']=measure_view(alpha,view);ms.append(item['measurement'])
            except ValueError as e:item['error']=str(e);errors.append(str(e))
            data['views'].append(item)
        if all(e=='No alpha support above common threshold' for e in errors):
            try:data['measurements'][str(n)]=combine_vertical(ms)
            except AssertionError as e:data['vertical_error']=str(e)
        data['precision']=assess_vertical(data['measurements'],frame['span']);save(out/'measurement.json',data)
        if data['precision']['passed']:break
    data.update(status='passed' if data['precision']['passed'] else 'numerical_extent_unresolved',finished_at=time.time());save(out/'measurement.json',data);return data

class ReconDepth:
    def __init__(self,path):
        self.renderer=source.GaussianRenderer(path);self.gs=self.renderer.gs;self.torch=self.renderer.torch
        self.arrays=[self.gs.get_xyz,self.gs.get_rotation,self.gs.get_scaling,self.gs.get_opacity.reshape(-1)]
        self.diagnostics=dict(adapter='native_Recon_explicit_precomputed_depth_colors',alpha_kernel='same native covariance/opacity')
    def render(self,V,K,n,near,far,ortho=False):
        assert not ortho
        t=self.torch;ki=np.asarray(K).copy();ki[:2,2]-=.5
        c=dict(wh=[n,n],K=ki,T_camera_from_object=V,near=near,far=far)
        camera,pipe=self.renderer.camera(c)
        from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
        settings=GaussianRasterizationSettings(image_height=n,image_width=n,tanfovx=float(np.tan(camera.FoVx*.5)),tanfovy=float(np.tan(camera.FoVy*.5)),kernel_size=pipe.kernel_size,subpixel_offset=t.zeros((n,n,2),device='cuda'),bg=t.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,sh_degree=self.gs.active_sh_degree,campos=camera.camera_center,prefiltered=False,debug=False)
        v=t.as_tensor(V,device='cuda',dtype=t.float32);z=self.arrays[0]@v[2,:3]+v[2,3];colors=t.stack([t.ones_like(z),z,t.zeros_like(z)],dim=1)
        with t.inference_mode():im,_=GaussianRasterizer(raster_settings=settings)(means3D=self.gs.get_xyz,means2D=t.zeros_like(self.gs.get_xyz),shs=None,colors_precomp=colors,opacities=self.gs.get_opacity,scales=self.gs.get_scaling,rotations=self.gs.get_rotation,cov3D_precomp=None)
        a=im[0];ed=t.where(a>0,im[1]/a.clamp_min(1e-20),0)
        assert t.isfinite(a).all() and t.isfinite(ed).all()
        return ed.cpu().numpy(),a.cpu().numpy()
    def check(self,reg,out):
        c=source.native_cameras()[0];K=np.asarray(c['K']);V=np.asarray(c['T_camera_from_object'])
        _,alpha,detail=self.renderer.render(c);kp=K.copy();kp[:2,2]+=.5
        ed,a=self.render(V,kp,512,c['near'],c['far']);diff=abs((a.clip(0,1)*255).round().astype(float)-alpha.astype(float))
        assert float(diff.max())<=1 and float(diff.mean())<.001
        z=self.arrays[0].detach().cpu().numpy()@V[2,:3]+V[2,3];valid=a>.5
        assert valid.any() and np.all(ed[valid]>=z.min()-1e-5) and np.all(ed[valid]<=z.max()+1e-5)
        check=dict(alpha_native_uint8_max=float(diff.max()),alpha_native_uint8_mean=float(diff.mean()),depth_on_support_min=float(ed[valid].min()),depth_on_support_max=float(ed[valid].max()),gaussian_center_depth_range=[float(z.min()),float(z.max())],pixel_edges_to_integer_center='subtract 0.5 exactly once before native projection helper',alpha_validation=detail,native_parameters_unchanged=all(self.torch.equal(v,getattr(self.gs,k).cpu()) for k,v in self.renderer.pack['params'].items()))
        assert check['native_parameters_unchanged'];save(out/'adapter_check.json',check);return check

def recon_extent(adapter,path,axes,out):
    out.mkdir(parents=True,exist_ok=True);cache=out/'native_activated_arrays.npz'
    np.savez_compressed(cache,**{k:v.detach().cpu().numpy() for k,v in zip(['means','quats','scales','opacity'],adapter.arrays)})
    job=dict(method='reconviagen',asset=str(path),asset_sha256=sha(path),decoded_arrays=str(cache),decoded_arrays_sha256=sha(cache));save(out/'decoded_job.json',job);save(out/'axes.json',axes.tolist())
    ev=dict(os.environ,TORCH_EXTENSIONS_DIR=str(H/'cache/torch_ext/ours'),CUDA_HOME=str(source.ENVS/'ab_eval'));ev['PATH']=str(source.ENVS/'ab_eval/bin')+':'+ev['PATH']
    subprocess.run([source.PY['control'],'-B',str(HG/'scripts/measure_decoded.py'),str(out)],env=ev,check=True)
    result=read(out/'measurement.json');save(out/'activated_cache_provenance.json',dict(**job,cache_retained=False,reproduction='Load original native Gaussian using frozen source adapter, read get_xyz/get_rotation/get_scaling/get_opacity; no filtering'))
    cache.unlink();return result

def run(method,sid):
    protocol=prepare();source.configure(method);dest=OUT/'runs'/method/sid;dest.mkdir(parents=True,exist_ok=True);rp=OUT/'results'/method/(sid+'.json')
    if rp.exists() and read(rp)['status'] in ['completed','registration_unavailable','generation_failed']:return read(rp)
    ref=next(r for r in read(HG/'reference_manifest.json')['rows'] if r['id']==sid)
    rec=dict(status='running',id=sid,method=method,reference_panel=ref['candidate_panel'],protocol_sha256=sha(OUT/'protocol.json'),queries_sha256=ref['queries_sha256'],started_at=time.time(),panels={});save(rp,rec)
    try:
        assert shutil.disk_usage(H).free>10*1024**3
        status=read(source.ROOT/'observations'/sid/'registration_status.json')
        if status['status']!='ready':
            rec.update(status='registration_unavailable',reason=status['reason']);save(rp,rec);return rec
        q,cam,path,gen=source.inputs(sid);regpath=source.ROOT/'registrations'/(sid+'.json');reg=read(regpath)
        assert q['identity_key']==ref['identity_key']==reg['identity_key'] and q['timestamp_us']==ref['t_ref_us']==reg['timestamp_us']
        assert q['camera_contract_sha256']==ref['camera_contract_sha256']==reg['camera_contract_sha256']
        checked(path,reg['payload_sha256']);checked(ref['queries'],ref['queries_sha256'])
        # Our mesh registration is normalized Blender-world; native geometry
        # evaluation removes that similarity gauge, preserving its translation.
        T=np.asarray(reg['T_source_camera_from_export'])@np.asarray(reg['normalization']);V,gauge=rigid_part(T);Twc=np.asarray(cam['T_world_from_camera_opencv']);axes=Twc[:3,:3]@V[:3,:3];arrays=dict(np.load(ref['queries']))
        rec.update(identity_key=ref['identity_key'],t_ref_us=ref['t_ref_us'],source_registration=str(regpath),source_registration_sha256=sha(regpath))
        if method=='hunyuan3d21':extent=mesh_extent(path,axes,dest/'extent')
        else:
            adapter=ReconDepth(path);adapter.check(reg,dest);extent=recon_extent(adapter,path,axes,dest/'extent')
        if extent['status']!='passed':
            rec.update(status='extent_numerically_unresolved',reason='Common native-alpha extent precision not reached',extent=str(dest/'extent/measurement.json'));save(rp,rec);return rec
        height=extent['measurements'][str(extent['precision']['fine_resolution'])]['vertical_extent'];scale=ref['annotation_vertical_extent_m']/height;assert np.isfinite(scale) and scale>0
        A=Twc@np.diag([scale,scale,scale,1.])@V;assert abs(scale*height-ref['annotation_vertical_extent_m'])<1e-9
        cert=dict(id=sid,method=method,identity_key=ref['identity_key'],t_ref_us=ref['t_ref_us'],protocol_sha256=sha(OUT/'protocol.json'),reference_manifest_sha256=sha(HG/'reference_manifest.json'),queries_sha256=ref['queries_sha256'],asset=str(path),asset_sha256=sha(path),source_registration=str(regpath),source_registration_sha256=sha(regpath),camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],T_source_camera_from_native=T.tolist(),source_similarity_gauge_removed=gauge,T_world_from_native=A.tolist(),scale_m_per_native_unit=scale,native_vertical_extent=height,annotation_vertical_extent_m=ref['annotation_vertical_extent_m'],extent=str(dest/'extent/measurement.json'),extent_sha256=sha(dest/'extent/measurement.json'),translation_scaled=True,root_pose_edited=False,source_input_frame='Blender_import_world' if method=='hunyuan3d21' else 'native_Gaussian',LiDAR_endpoint_fitting=False)
        save(dest/'metric_scale_certificate.json',cert)
        if method=='hunyuan3d21':
            from depth_adapters import query_mesh
            rng,alpha,hit,diag=query_mesh(path,arrays,A,protocol['rules'])
        else:rng,alpha,hit,diag=query(adapter,arrays,A)
        met,points,dpg,dgp=metrics(arrays['q'],arrays['o'],arrays['u'],arrays['r'],rng,hit)
        np.savez_compressed(dest/'returns.npz',range_pred=rng,alpha=alpha,hit=hit,predicted_endpoints=points,distances_pg=dpg,distances_gp=dgp)
        rec['panels']['annotation_vertical_extent_conditioned']=dict(metrics=met,diagnostics=diag)
        rec.update(status='completed',certificate=str(dest/'metric_scale_certificate.json'),certificate_sha256=sha(dest/'metric_scale_certificate.json'),returns=str(dest/'returns.npz'),returns_sha256=sha(dest/'returns.npz'))
    except Exception as e:rec.update(status='execution_failure_pending_review',reason=repr(e),traceback=traceback.format_exc())
    rec['finished_at']=time.time();save(rp,rec);print(method,sid,rec['status'],rec.get('reason'),flush=True);return rec

def summary():
    groups={};coverage={};settlement={}
    for method in METHODS:
        rows=[read(p) for p in (OUT/'results'/method).glob('*.json')];coverage[method]=dict(collections.Counter(r['status'] for r in rows));panels={}
        for panel in ['main','low_support_exploratory']:
            rs=[r for r in rows if r['status']=='completed' and r['reference_panel']==panel];pairs=[(r,r['panels']['annotation_vertical_extent_conditioned']['metrics']) for r in rs];finite=[(r,m) for r,m in pairs if m['cd_m'] is not None];empty=[r['id'] for r,m in pairs if m['status']=='empty_pred'];cells={}
            cells['CD_cm']=dict(value=float(np.mean([m['cd_m'] for r,m in finite])*100) if finite else None,n=len(finite),ids=sorted(r['id'] for r,m in finite),unit='cm',valid_empty_ids=empty,valid_empty_n=len(empty),extended_mean='infinity' if empty else 'finite' if finite else None)
            for metric,key,mult in [('F5','f1_5cm',100),('F10','f1_10cm',100),('F20','f1_20cm',100),('RayHit','ray_hit',100)]:cells[metric]=dict(value=float(np.mean([m[key] for r,m in pairs])*mult) if pairs else None,n=len(pairs),ids=sorted(r['id'] for r,m in pairs),unit='percent',valid_empty_n=len(empty))
            panels[panel]=cells
        groups[method]=panels
        pending=100-sum(n for s,n in coverage[method].items() if s!='running');execution=coverage[method].get('execution_failure_pending_review',0)
        settlement[method]=dict(status='completed' if not pending and not execution else 'needs_execution_review' if not pending else 'incremental',expected=100,pending=pending,execution_failures=execution,counts=coverage[method])
    save(OUT/'summary.json',dict(status='completed' if all(x['status']=='completed' for x in settlement.values()) else 'incremental',groups=groups,coverage=coverage,settlement=settlement,main_panel='main / annotation_vertical_extent_conditioned',updated_at=time.time(),protocol_sha256=sha(OUT/'protocol.json'),old_scores_modified=False))

@contextlib.contextmanager
def lease(gpu):
    with contextlib.ExitStack() as stack:
        fs=[]
        for p in [source.P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock',H/f'state/gpu_auxiliary_slot_{gpu}.lock',Path(f'/tmp/drivelab-gpu-locks/gpu{gpu}.lock')]:
            p.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX);fs.append(f)
        yield fs

def worker(method,gpu=None,pilot_only=False):
    prepare();statuspath=OUT/'workers'/(method+'.json')
    while True:
        sourcejobs=[j for j in read(source.OUT/'queue.json')['jobs'] if j['method']==method and (not pilot_only or j['id']=='HN0004')];ready=[];pending=False
        for job in sourcejobs:
            rp=OUT/'results'/method/(job['id']+'.json')
            if rp.exists() and read(rp)['status']!='running':continue
            if job['status']=='completed':ready.append(job['id'])
            elif job['status']=='generation_failed':save(rp,dict(id=job['id'],method=method,status='generation_failed',reason=job.get('reason'),protocol_sha256=sha(OUT/'protocol.json')))
            elif job['status']=='evaluation_stage_failed':pending=True
            else:pending=True
        if not ready:
            summary()
            if not pending:break
            save(statuspath,dict(status='waiting_source_no_gpu_lease',pid=os.getpid(),gpu=gpu,updated_at=time.time()));time.sleep(20);continue
        sid=ready[0]
        if shutil.disk_usage(H).free<10*1024**3:time.sleep(30);continue
        save(statuspath,dict(status='waiting_gpu_lease' if method=='reconviagen' else 'running_cpu',id=sid,pid=os.getpid(),gpu=gpu,updated_at=time.time()))
        with (lease(gpu) if method=='reconviagen' else contextlib.nullcontext([])) as fs:
            cmd=[source.PY[method],'-B','-u',str(S),'run','--method',method,'--ids',sid];env=source.environment(method,gpu or 0)
            save(statuspath,dict(status='running',id=sid,pid=os.getpid(),gpu=gpu,command=cmd,updated_at=time.time()));log=OUT/'logs'/method/(sid+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                child=subprocess.Popen(cmd,env=env,cwd=H,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(x.fileno() for x in fs));code=child.wait()
            if code:save(OUT/'results'/method/(sid+'.json'),dict(id=sid,method=method,status='execution_failure_pending_review',reason=f'Worker subprocess exit {code}',log=str(log),protocol_sha256=sha(OUT/'protocol.json')))
        summary();time.sleep(1)
    save(statuspath,dict(status='completed',pid=os.getpid(),gpu=gpu,finished_at=time.time()));summary()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','summary','worker']);p.add_argument('--method',choices=METHODS);p.add_argument('--ids',nargs='+');p.add_argument('--gpu',type=int);p.add_argument('--pilot-only',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='summary':summary()
    elif a.action=='worker':worker(a.method,a.gpu,a.pilot_only)
    else:
        for sid in a.ids:run(a.method,sid)
        summary()
