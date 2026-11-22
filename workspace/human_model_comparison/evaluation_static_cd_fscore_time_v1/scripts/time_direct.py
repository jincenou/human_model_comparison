"""Preloaded direct-model generation timing, isolated native output endpoint."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,ast,inspect,os,random,subprocess,sys,textwrap,time,traceback
from types import SimpleNamespace
from common import *
P=R/'model_preparation/20261109_2100_CST'

def load(method):
    import torch
    from PIL import Image
    if method=='trellis_original':
        source=P/'sources/trellis_original_legacy_20261110';sys.path[:0]=[str(source),str(P/'scripts')];os.chdir(source)
        from gen_gs import GsConfig,GsBatchGenerator
        from trellis import models
        from trellis.pipelines import samplers
        from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
        pipeline=GsBatchGenerator(GsConfig(model_dir=Path(_hmc_expand('${HMC_MODEL_CACHE}/TRELLIS-image-large-gaussian')),seed=42,worker_count=1))._load_pipeline(models,samplers,TrellisImageTo3DPipeline)
        pipeline.cuda()
        def run(item,out,stage):
            with stage('preprocess_and_gaussian_inference'):
                gs=pipeline.run(Image.open(item['image_rgba']),seed=42,formats=['gaussian'])['gaussian'][0]
            with stage('native_serialization'):
                from recon02_serialization import save_native_gaussian
                save_native_gaussian(gs,out/'native_gaussians.ply')
                torch.save(dict(init_params=gs.init_params,params={k:getattr(gs,k).detach().cpu() for k in ['_xyz','_features_dc','_scaling','_rotation','_opacity']}),out/'native_gaussians.pt')
            return out/'native_gaussians.pt'
        return run,dict(source=str(source),sparse=pipeline.sparse_structure_sampler_params,slat=pipeline.slat_sampler_params,endpoint='native_gaussians.pt and equivalent PLY')
    if method=='trellis2':
        source=P/'sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd';sys.path.insert(0,str(source));os.chdir(source)
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(P/'weights/microsoft--TRELLIS.2-4B'));pipeline.cuda()
        assert pipeline.default_pipeline_type=='1024_cascade' and pipeline.low_vram
        def run(item,out,stage):
            with stage('preprocess_shape_and_material_voxel_generation'):
                mesh=pipeline.run(Image.open(item['image_rgba']),seed=42,pipeline_type='1024_cascade',max_num_tokens=49152)[0]
            with stage('necessary_simplify_and_native_serialization'):
                mesh.simplify(16777216)
                packed={k:getattr(mesh,k).detach().cpu() for k in ['vertices','faces','coords','attrs']}
                packed.update(origin=mesh.origin.tolist(),voxel_size=float(mesh.voxel_size),voxel_shape=list(mesh.voxel_shape),layout={k:[v.start,v.stop,v.step] for k,v in mesh.layout.items()})
                torch.save(packed,out/'native_voxel_mesh.pt')
            return out/'native_voxel_mesh.pt'
        return run,dict(source=str(source),pipeline_type='1024_cascade',low_vram=True,max_num_tokens=49152,seed=42,simplify=16777216,endpoint='native_voxel_mesh.pt with geometry AND material voxels; no UV/GLB')
    assert method=='ah'
    B=H.parent/'assetbench';source=B/'repos/ah';sys.path.insert(0,str(source));os.chdir(source)
    import run_inference as native
    tree=ast.parse((source/'run_inference.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='device' for t in n.targets))
    end=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='samples' for t in n.targets))
    args=SimpleNamespace(image_dir=True,data_root=None,precision='bf16',skip_gs_lifting=False,diffusion_checkpoint=str(B/'weights/ah/core/AH_multiview_diffusion.safetensors'),ahc_checkpoint=str(B/'weights/ah/core/AH_camera_estimator.safetensors'),lifting_checkpoint=str(B/'weights/ah/core/AH_tokengs_lifting.safetensors'),num_steps=30,cfg_scale=2.,offload_model_to_cpu=False)
    ns=dict(vars(native),args=args)
    exec(compile(ast.fix_missing_locations(ast.Module(body=main.body[start:end],type_ignores=[])),str(source/'run_inference.py'),'exec'),ns)
    io_source=inspect.getsource(native.save_mvd_outputs);io_tree=ast.parse(textwrap.dedent(io_source));fn=io_tree.body[0]
    preview=[n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='imageio.v2.mimwrite'];assert len(preview)==1
    fn.body=[n for n in fn.body if n is not preview[0]];io_ns=dict(native.save_mvd_outputs.__globals__)
    exec(compile(ast.fix_missing_locations(io_tree),inspect.getsourcefile(native.save_mvd_outputs),'exec'),io_ns)
    def run(item,out,stage):
        with stage('native_image_and_camera_preprocess'):
            source_item=native.load_image_sample([(item['ah_source_image'],item['ah_source_mask'])],ns['ahc_estimator'])
            data=ns['inference_preproc'](source_item,shuffle_inds=None);length=data.n_target+min(4,len(data.x)-data.n_target)
            for name in ['x','c2w_relatives','x_white_background','dists','fovs','plucker_image','relative_brightness']:setattr(data,name,getattr(data,name)[:length])
            if hasattr(data,'intrinsics'):data.intrinsics=data.intrinsics[:length]
        with stage('multiview_generation_and_required_handoff'):
            with torch.no_grad():result=ns['pipeline'](data_dict=data,num_inference_steps=30,guidance_scale=2.,flow_shift=1.,output_type='pil')
            fov=data.fovs[0].item();dist=data.dists[0].item();lwh=data.lwh if getattr(data,'lwh',None) is not None else None
            images=io_ns['save_mvd_outputs'](result['images'],fov,dist,lwh,str(out))
            native.gc.collect();torch.cuda.empty_cache()
        with stage('gaussian_lifting_and_native_serialization'):
            with torch.no_grad():gs=ns['lifting_runner'].run_lifting(images,fov,dist,lwh if lwh is not None else [1.,1.,1.])
            ns['lifting_runner'].save_ply(gs,str(out/'gaussians.ply'))
            # AHC-predicted camera and dimensions are required source metadata.
            save(out/'ahc_source_prediction.json',dict(fov=fov,dists=dist,lwh=lwh.detach().cpu().tolist() if hasattr(lwh,'detach') else lwh,c2w_relatives=data.c2w_relatives.detach().cpu().tolist()))
        return out/'gaussians.ply'
    return run,dict(source=str(source),source_sha256=sha(source/'run_inference.py'),settings=vars(args),endpoint='native Gaussian PLY plus required AHC camera metadata',io_edit='sole preview-video encoding removed; PNG/camera handoff retained')

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ah','trellis_original','trellis2'],required=True);a=p.parse_args()
    if a.method=='trellis_original':os.environ.update(SPCONV_ALGO='native',ATTN_BACKEND='flash-attn')
    if a.method=='trellis2':os.environ.update(PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OPENCV_IO_ENABLE_OPENEXR='1')
    import torch
    from contextlib import contextmanager
    torch.set_num_threads(4);plan=read(E/'time_plan.json');rows=[]
    for name in ['human_static_native_v1','human_static_replacement_v3']:rows+=read(H/'inputs'/name/'manifest.json')['objects']
    inputs={r['sample_id']:r for r in rows}
    # Resolve AH production input pairs independently of evaluation masks.
    for member in plan['objects']:
        sid=member['id'];job=next(j for j in read(E/'intake.json')['rows'] if j['id']==sid and j['method']=='ah')
        trace=read(job['source_trace']);source=read(trace['input_manifest']);item=inputs[sid]
        item.update(ah_source_image=source['image'],ah_source_mask=source['mask'])
        for key,digest in [('image','source_image_sha256'),('mask','source_mask_sha256')]:checked(source[key],source[digest])
    (E/'timing').mkdir(exist_ok=True);save(E/'timing'/f'{a.method}_loading.json',dict(status='loading_weights',pid=os.getpid(),started_at=time.time()))
    runner,settings=load(a.method)
    # Only generic CUDA setup; any first-use model kernel compilation is retained
    # in the first formal attempt and explicitly disclosed, never subtracted.
    torch.randn((64,64),device='cuda')@torch.randn((64,64),device='cuda');torch.cuda.synchronize()
    for job in [j for j in plan['runs'] if j['method']==a.method]:
        sid=job['id'];out=E/'timing/runs'/a.method/f'{sid}_r{job["repeat"]}';record_path=out/'timing.json'
        if record_path.exists():
            old=read(record_path)
            if old['status'] in ['success','failed']:
                assert old['plan_sha256']==sha(E/'time_plan.json') and old['script_sha256']==sha(__file__)
                continue
            raise RuntimeError('Incomplete prior attempt requires explicit recovery: '+str(out))
        out.mkdir(parents=True,exist_ok=True);assert not list(out.iterdir()),'Partial output requires review';item=inputs[sid];checked(item['image_rgba'],item['image_rgba_sha256'])
        record=dict(job,status='running',plan_sha256=sha(E/'time_plan.json'),script_sha256=sha(__file__),settings=settings,seed=42,
          gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),software=dict(torch=torch.__version__,cuda=torch.version.cuda),
          input_sha256=item['image_rgba_sha256'],stages=[],started_at=time.time(),load_average_before=os.getloadavg(),disk_free_bytes_before=__import__('shutil').disk_usage(E).free,cpu_stat_before=Path('/proc/stat').read_text().splitlines()[0],diskstats_before=Path('/proc/diskstats').read_text(),
          gpu_processes_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
          timing='weights preloaded; synchronized perf_counter; all necessary model-specific processing and serialization',first_use_compilation='retained if incurred; no estimated subtraction',real_input_count=1)
        save(record_path,record);start=None
        @contextmanager
        def stage(name):
            torch.cuda.synchronize();t=time.perf_counter();status='running'
            try:yield;status='completed'
            except BaseException:status='failed';raise
            finally:
                torch.cuda.synchronize();record['stages'].append(dict(name=name,status=status,seconds=time.perf_counter()-t))
        try:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            torch.cuda.synchronize();start=time.perf_counter();output=runner(item,out,stage)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            record.update(status='generated_pending_reload',seconds=seconds,output=str(output))
            # Independent reload outside interval, all attributes finite.
            if output.suffix=='.pt':
                packed=torch.load(output,map_location='cpu',weights_only=True)
                tensors=list(packed['params'].values()) if 'params' in packed else [packed[k] for k in ['vertices','faces','coords','attrs']]
                assert all(torch.isfinite(x).all() for x in tensors) and len(tensors[0])>0
            else:
                code="from plyfile import PlyData; import numpy as np,sys;v=PlyData.read(sys.argv[1])['vertex'];assert len(v)>0;assert all(np.isfinite(v[n]).all() for n in v.data.dtype.names)"
                subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-c',code,str(output)],check=True)
            record.update(status='success',output_sha256=sha(output),independent_reload_passed=True)
        except Exception as error:
            record.update(status='failed',reason=repr(error),traceback=traceback.format_exc())
            if start is not None:record['failed_elapsed_seconds']=time.perf_counter()-start
        record.update(finished_at=time.time(),load_average_after=os.getloadavg(),cpu_stat_after=Path('/proc/stat').read_text().splitlines()[0],diskstats_after=Path('/proc/diskstats').read_text(),gpu_processes_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))
        save(record_path,record);print(sid,record['status'],record.get('seconds'),record.get('reason'),flush=True)

if __name__=='__main__':main()
