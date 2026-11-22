"""Time the retained production functions in an isolated run root.

Measured compute intervals pause at named pretrained-model constructors/loaders,
then resume after synchronized loading. Generated-asset handoffs are never
excluded. This is a sum of preloaded-stage intervals, not online wall latency.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,ast,hashlib,importlib,inspect,json,os,sys,time,traceback,textwrap
from pathlib import Path
CODE=Path(_hmc_expand('${HMC_WORKSPACE}/human'));E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import common as native
import numpy as np

def save(p,d):native.save(p,d)
def patch_without_preview(module,name,start_text,end_text=None):
    original=inspect.unwrap(getattr(module,name));source=inspect.getsource(original);tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    begin=next(i for i,n in enumerate(fn.body) if ast.unparse(n).startswith(start_text))
    finish=next(i for i,n in enumerate(fn.body) if i>=begin and ast.unparse(n).startswith(end_text)) if end_text else len(fn.body)
    fn.body[begin:finish]=[]
    scope=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),str(inspect.getsourcefile(original)),'exec'),scope)
    setattr(module,name,scope[name])
    return dict(function=name,source_sha256=native.sha(inspect.getsourcefile(original)),removed_from=start_text,removed_until=end_text or 'function_end')

class Timer:
    def __init__(self,phase,torch):
        self.phase=phase;self.torch=torch;self.active=[];self.loads=[];self.stack=[];self.last=None;self.outer=None
    def sync(self):
        if self.torch.cuda.is_initialized():self.torch.cuda.synchronize()
    def reason(self,frame):
        # Training initialization and all generated-state loading stay inside.
        if self.phase in ['native','alpha','A','hull','configure','export','snapshot','native_input']:return None
        file=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name;local=frame.f_locals
        if '/fdanyone/model/loader.py' in file and name in ['load_denoiser','load_pose_encoder','load_vae','load_prompt_context']:return 'pretrained_'+name
        if name=='prepare_on_device' and '/fdanyone/model/loader.py' in file and not local['self'].turbo_lora_applied:return 'first_DiT_device_and_Turbo_weight_initialization'
        if '/sam2/build_sam.py' in file and name=='build_sam2_video_predictor':return 'SAM2_model_construction_and_weights'
        if name=='from_pretrained' and getattr(local.get('cls'),'__name__','')=='BiRefNet':return 'BiRefNet_pretrained_constructor'
        if name=='__init__' and file.endswith(('/preproc/vitpose.py','/preproc/vitfeat_extractor.py','/train_log/IFNet_HDv3.py')):return 'pretrained_constructor_'+Path(file).stem
        if self.phase=='motion' and '/hydra/_internal/instantiate/' in file and name=='instantiate':return 'GVHMR_model_construction'
        if name=='make_smplx' and '/hmr4d/' in file:return 'licensed_body_model_construction'
        if name=='load' and file.endswith('/torch/serialization.py'):
            path=str(local.get('f',''))
            if any(s in path for s in ['/inputs/checkpoints/','/models/','/rife_model/','/utils/body_model/']):return 'pretrained_weight_read:'+path
        if name=='load_state_dict' and file.endswith('/torch/nn/modules/module.py'):return 'strict_pretrained_parameter_assignment'
        return None
    def profile(self,frame,event,arg):
        if event=='call':
            why=self.reason(frame)
            if why:
                if not self.stack:
                    self.sync();now=time.perf_counter();self.active.append(dict(start=self.last,end=now,seconds=now-self.last));self.outer=(now,why,frame.f_code.co_filename,frame.f_code.co_firstlineno)
                self.stack.append(frame)
        elif event=='return' and self.stack and self.stack[-1] is frame:
            self.stack.pop()
            if not self.stack:
                self.sync();now=time.perf_counter();start,why,file,line=self.outer;self.loads.append(dict(reason=why,file=file,line=line,start=start,end=now,seconds=now-start));self.last=now
    def start(self):
        self.sync();self.last=time.perf_counter();self.started=self.last
        if self.phase not in ['native','alpha','A','hull','configure','export','snapshot','native_input']:sys.setprofile(self.profile)
    def stop(self):
        sys.setprofile(None);self.sync();now=time.perf_counter()
        if not self.stack:self.active.append(dict(start=self.last,end=now,seconds=now-self.last))
        return dict(seconds=sum(x['seconds'] for x in self.active),active_intervals=self.active,excluded_pretrained_load_intervals=self.loads,wall_seconds=now-self.started,complete_interval_accounting=not self.stack)

def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--phase',required=True);p.add_argument('--group',type=int);a=p.parse_args();phase=a.phase
    import torch
    torch.set_num_threads(2);patches=[]
    if phase in ['native_input','interpolate','segment']:
        import intake
        if phase=='native_input':
            # Native clip video is an audit preview; crop PNGs/camera records remain.
            original=intake.native;tree=ast.parse(inspect.getsource(original));fn=tree.body[0]
            fn.body=[n for n in fn.body if not (isinstance(n,ast.Expr) and ast.unparse(n).startswith("encode(out / 'observed_native.mp4'"))]
            ns=dict(original.__globals__);exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(original),'exec'),ns);intake.native=ns['native']
            patches.append('omit observed_native preview encoding only');call=lambda:intake.native(a.id)
        else:call=lambda:getattr(intake,phase)(a.id)
    elif phase in ['motion','condition']:
        sys.path.insert(0,str(CODE/'vendor/4danyone'));import intake_motion
        if phase=='motion':
            patches.append(patch_without_preview(intake_motion,'estimate','panels = []'));call=lambda:intake_motion.estimate(a.id)
        else:call=lambda:intake_motion.condition(a.id)
    elif phase=='frontend':
        import frontend
        # Same raw images, seed, masks and source recipe; record recreated bytes.
        registry=native.read(native.ROOT/'config/cases.json');spec=registry[a.id]
        old=spec['input_review']['input_sha256'];new=native.sha(spec['raw_input']);spec['input_review'].update(input_sha256=new,timing_rebuild_original_sha256=old,timing_rebuild_same_bytes=old==new)
        save(native.ROOT/'config/cases.json',registry);call=lambda:frontend.generate(a.id)
    elif phase in ['masks','hull','configure']:
        import dataset
        call=(lambda:dataset.masks(a.id,a.group)) if phase=='masks' else lambda:getattr(dataset,phase)(a.id)
    elif phase in ['native','alpha','A']:
        import reconstruction
        patches.append(dict(function='capture',source_sha256=native.sha(CODE/'reconstruction.py'),removed='RNG-preserving independent reference renders and their NPZ only; full state checkpoint and metadata retained'))
        original_source=(CODE/'reconstruction.py').read_text();tree=ast.parse(original_source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        begin=next(i for i,n in enumerate(fn.body) if ast.unparse(n).startswith("target = run / 'snapshots'"));fn.body=fn.body[begin:]
        fn.body=[n for n in fn.body if not (isinstance(n,ast.Expr) and ast.unparse(n).startswith('np.savez_compressed'))]
        ns=dict(reconstruction.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(CODE/'reconstruction.py'),'exec'),ns);reconstruction.capture=ns['capture']
        call=(lambda:reconstruction.train_native(a.id)) if phase=='native' else lambda:reconstruction.continue_training(a.id,phase)
    elif phase=='export':
        import assets
        # Independent renderer validation is excluded; complete native arrays,
        # SH/covariance conversion, necessary references/metadata remain.
        patches.append(patch_without_preview(assets,'export_asset','refs = {}','from PIL import Image'))
        call=lambda:assets.export_asset(a.id)
    elif phase=='snapshot':
        import importlib.util,shutil
        src=CODE.parent/'Hasim_trajectory_assets/src';spec=importlib.util.spec_from_file_location('hasim',src/'__init__.py',submodule_search_locations=[str(src)])
        mod=importlib.util.module_from_spec(spec);sys.modules['hasim']=mod;spec.loader.exec_module(mod)
        from hasim.shared.assets.library_v2 import LibraryAsset,RuntimeAsset
        def call():
            out=native.ASSET_ROOT/a.id;npz=out/'asset_4dgs.npz';source=native.ROOT/'outputs'/a.id/'asset';folder=out/'runtime'
            for name in ['point_cloud','deform','config.json','asset_camera_data.json','snapshots/53000.json','complete.json']:
                f=source/name;t=folder/name;t.parent.mkdir(parents=True,exist_ok=True)
                if f.is_dir():shutil.copytree(f,t)
                else:shutil.copy2(f,t)
            with np.load(npz) as z:
                arrays={k:z[v] for k,v in dict(frame_xyz='frame_xyz',frame_quaternions='frame_rotation',frame_scales='frame_scaling',opacities='opacity',sh='shs',timestamps='frame_timestamps').items()}
                arrays['opacities']=arrays['opacities'].reshape(-1)
            # Evaluate exactly the quality loader's nonlooping SPD interpolation
            # in exported coordinates; root transform remains explicit metadata.
            f=len(arrays['timestamps']);arrays.update(root_xy=np.zeros((f,2)),root_yaw=np.zeros(f),asset_to_standard=np.eye(4),ground_offset=np.array(0.))
            meta=dict(format='hasim_asset_runtime_v1',duration_s=float(arrays['timestamps'][-1]-arrays['timestamps'][0]))
            asset=LibraryAsset(a.id,'human','human',npz,meta,(npz,));runtime=RuntimeAsset(asset,arrays,[npz])
            row=next(r for r in json.loads((E.parent/'inputs/benchmark_human100_v3/selection.json').read_text())['objects'] if r['id']==a.id)
            elapsed=row['reference_seconds_from_first_native'];cloud=runtime.sample(elapsed);assert cloud.sh_degree==3
            np.savez_compressed(out/'frozen_time_native_export_coordinates.npz',means=cloud.means,quaternions_wxyz=cloud.quaternions,scales=cloud.scales,opacities=cloud.opacities,sh=cloud.sh,local_from_appearance=cloud.local_from_appearance)
            for key in ['root_motion','motion','conditioning']:assert Path(native.case(a.id)[key]).exists()
            shutil.copy2(native.case(a.id)['root_motion'],out/'root_motion.npz')
            for name in ['native.json','inputs/frame_mapping.json','inputs/cameras.npz']:
                p=native.ROOT/'inputs'/a.id/'clip0'/name;t=out/'source'/name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
            save(out/'time_endpoint.json',dict(id=a.id,t_ref_us=row['t_ref_us'],source_seconds=elapsed,clock='original nonlooping; no seam/speed/cycle',representation='complete SH3 and native covariance; quality RuntimeAsset SPD interpolation in original export coordinates',root='root-local geometry; full original root transform retained separately in root_motion.npz; not claimed metric world',native_updates=32999,alpha_updates=15000,complete_A_updates=5000,quality_loader_sha256=native.sha(src/'shared/assets/library_v2.py'),files={str(p.relative_to(out)):native.sha(p) for p in out.rglob('*') if p.is_file()}))
    else:raise ValueError(phase)
    native.gpu_identity();torch.cuda.synchronize();timer=Timer(phase,torch)
    tag=phase+(str(a.group) if a.group is not None else '');output=native.ROOT/'timing'/f'{tag}.json'
    record=dict(id=a.id,phase=tag,status='running',pid=os.getpid(),script_sha256=native.sha(__file__),source_code={str(p):native.sha(p) for p in CODE.glob('*.py')},preview_edits=patches,started_at=time.time(),torch=torch.__version__,cuda=torch.version.cuda,load_average_before=os.getloadavg())
    save(output,record);timer.start()
    try:
        with torch.no_grad() if phase in ['export','snapshot'] else __import__('contextlib').nullcontext():call()
        record['status']='completed'
    except BaseException as error:record.update(status='failed',reason=repr(error),traceback=traceback.format_exc());raise
    finally:
        record.update(timer.stop(),finished_at=time.time(),load_average_after=os.getloadavg());save(output,record)

if __name__=='__main__':main()
