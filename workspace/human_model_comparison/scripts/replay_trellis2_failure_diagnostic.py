"""Bounded official-generation replay; no GLB export or production replacement."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common_human_orbit import H, read, save, sha

P = H.parent / 'vehicle_model_comparison/model_preparation/20261109_2100_CST'
S = P / 'sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd'
ROOT = H / 'reports/trellis2_failure_root_cause_v1/controlled_replay'
TASKS = [('HN0046','original'), ('HN0004','original'), ('HN0214','original'), ('HN0214','corrected_mask')]


def worker():
    os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    sys.path.insert(0,str(S));os.chdir(S)
    import torch
    import numpy as np
    import transformers
    import cv2
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    from trellis2.representations import MeshWithVoxel
    from human_source_metrics_v1 import crop
    torch.set_num_threads(4)
    pipeline=Trellis2ImageTo3DPipeline.from_pretrained(str(P/'weights/microsoft--TRELLIS.2-4B'));pipeline.cuda()
    assert pipeline.low_vram and pipeline.default_pipeline_type=='1024_cascade'
    entries={r['sample_id']:r for r in read(H/'inputs/human_static_native_v1/manifest.json')['objects']}
    metric_rows={(r['id'],r['method']):r for r in read(H/'reports/source_camera_metric_cohort_v3/summary.json')['rows']}
    state=dict(status='running',pid=os.getpid(),tasks=[],seed=42,default_pipeline_type=pipeline.default_pipeline_type,
               pipeline_config_sha256=sha(P/'weights/microsoft--TRELLIS.2-4B/pipeline.json'),
               official_pipeline_sha256=sha(S/'trellis2/pipelines/trellis2_image_to_3d.py'),
               transformers_version=transformers.__version__,torch_version=torch.__version__,
               input_selection='One unchanged-mask failure, one success control, one substantial mask-revision case tested twice',
               production_assets_modified=False,official_generation_math_modified=False,
               geometry_measured_before_adapter_simplify_or_glb=True,glb_export=False)
    if (ROOT/'replay.json').exists():
        prior=read(ROOT/'replay.json')
        for key in ['seed','pipeline_config_sha256','official_pipeline_sha256','torch_version','transformers_version']:
            assert prior[key]==state[key],key
        state['tasks']=prior['tasks'];state['resumed_at']=time.time()
    save(ROOT/'replay.json',state)
    hdr=cv2.imread(str(S/'assets/hdri/forest.exr'),cv2.IMREAD_UNCHANGED);assert hdr is not None
    light=EnvMap(torch.tensor(cv2.cvtColor(hdr,cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda'))
    for sid,variant in TASKS:
        folder=ROOT/(sid+'_'+variant);folder.mkdir(exist_ok=True)
        item=next((r for r in state['tasks']if r['id']==sid and r['variant']==variant),None)
        if item and item['status']=='finished':continue
        entry=entries[sid];assert sha(entry['image_rgba'])==entry['image_rgba_sha256']
        image=Image.open(entry['image_rgba']);original=read(H/'runs/trellis2_human_native_v1'/sid/'state.json')
        assert original['input_sha256']==sha(entry['image_rgba'])
        if variant=='corrected_mask':
            # Diagnostic only: retain original exposure, original ROI and RGB.
            # Replace alpha with the reviewed mask; never feed generated edits.
            camera=read(entry['camera_candidate']);ref=metric_rows[(sid,'trellis2')]['reference']
            mask=np.array(Image.open(ref['reference_mask']).convert('L'))
            alpha=crop(mask,camera['roi_xyxy'],True)
            image=image.copy();image.putalpha(Image.fromarray(alpha))
        if (folder/'input.png').exists():
            assert item and sha(folder/'input.png')==item['input_sha256']
            assert np.array_equal(np.asarray(image),np.asarray(Image.open(folder/'input.png')))
        else:image.save(folder/'input.png')
        processed=pipeline.preprocess_image(image)
        if (folder/'preprocessed.png').exists():assert np.array_equal(np.asarray(processed),np.asarray(Image.open(folder/'preprocessed.png')))
        else:processed.save(folder/'preprocessed.png')
        if item is None:
            item=dict(id=sid,variant=variant,status='generating',input_sha256=sha(folder/'input.png'),
                  original_input_sha256=original['input_sha256'],started_at=time.time())
            state['tasks'].append(item)
        save(ROOT/'replay.json',state)
        if (folder/'native_voxel_mesh.pt').exists():
            packed=torch.load(folder/'native_voxel_mesh.pt',map_location='cpu',weights_only=True)
            restored=dict(packed)
            for key in ['vertices','faces','coords','attrs']:
                assert torch.isfinite(packed[key]).all();restored[key]=packed[key].cuda()
            restored['layout']={k:slice(*v)for k,v in packed['layout'].items()}
            mesh=MeshWithVoxel(**restored);item['reused_saved_generation']=True
        else:
            mesh=pipeline.run(image,seed=42)[0]
            packed={k:getattr(mesh,k).detach().cpu() for k in ['vertices','faces','coords','attrs']}
            packed.update(origin=mesh.origin.tolist(),voxel_size=float(mesh.voxel_size),voxel_shape=list(mesh.voxel_shape),
                      layout={k:[v.start,v.stop,v.step]for k,v in mesh.layout.items()})
            torch.save(packed,folder/'native_voxel_mesh.pt')
        xyz=packed['vertices'].double().numpy();center=xyz.mean(0)
        eigen=np.maximum(np.linalg.eigvalsh(xyz.T@xyz/len(xyz)-np.outer(center,center)),0)
        item.update(vertices=len(xyz),faces=len(packed['faces']),aabb_extent=np.ptp(xyz,axis=0).tolist(),
                    pca_min_to_max_std=float(np.sqrt(eigen[0]/max(eigen[-1],1e-20))),native_checkpoint_sha256=sha(folder/'native_voxel_mesh.pt'))
        old=torch.load(H/'runs/trellis2_human_native_v1'/sid/'native_voxel_mesh.pt',map_location='cpu',weights_only=True)
        item['comparison_to_production']={k:dict(same_shape=list(old[k].shape)==list(packed[k].shape),
            exact_tensor_equal=bool(torch.equal(old[k],packed[k])))for k in ['vertices','faces','coords','attrs']}
        item['status']='geometry_saved';save(ROOT/'replay.json',state)
        del old,packed,xyz
        if len(mesh.faces)<=16777216:
            ex,ins=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.,1.5707963268,3.1415926536],[.2]*3,2.,40.)
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(12345)
                frames=render_utils.render_frames(mesh,ex,ins,options={'resolution':512},envmap=light)
            for kind in ['base_color','alpha']:
                for index,array in enumerate(frames[kind]):Image.fromarray(array).save(folder/f'{kind}_{index}.png')
        item.update(status='finished',finished_at=time.time());save(ROOT/'replay.json',state)
        del mesh;torch.cuda.empty_cache()
    state.update(status='finished',finished_at=time.time());save(ROOT/'replay.json',state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.worker:worker();return
    if args.resume:
        prior=read(ROOT/'controller.json');assert prior['status']=='failed_retained'
        assert not Path('/proc',str(prior['pid'])).exists() and not Path('/proc',str(prior['worker_pid'])).exists()
    else:ROOT.mkdir(exist_ok=False)
    state=dict(status='waiting_gpu',pid=os.getpid(),started_at=time.time(),tasks=TASKS,script_sha256=sha(__file__))
    save(ROOT/'controller.json',state)
    import trellis_source_batch as shared
    slot=shared.acquire()
    while slot is None:time.sleep(5);slot=shared.acquire()
    lease,gpu,detail=slot
    try:
        assert shutil.disk_usage(H).free>10*1024**3
        sys.path.insert(0,str(P/'scripts'));import dispatch_rigid111_direct as resources
        env=resources.environment('trellis2');env.update(CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        command=[_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2/bin/python'),'-B','-u',str(Path(__file__).resolve()),'--worker']
        with (ROOT/'worker.log').open('ab'if args.resume else'xb')as stream:child=subprocess.Popen(command,cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        state.update(status='running',gpu=gpu,gpu_uuid=detail['uuid'],worker_pid=child.pid,command=command);save(ROOT/'controller.json',state)
        code=child.wait();state.update(status='finished'if code==0 else'failed_retained',exit_code=code,finished_at=time.time());save(ROOT/'controller.json',state)
    finally:lease.close()


if __name__=='__main__':main()
