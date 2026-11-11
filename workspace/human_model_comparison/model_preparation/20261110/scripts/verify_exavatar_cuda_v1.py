"""Wait for the pinned build, then verify CUDA kernels under a shared GPU lease."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parents[1]
R=P.parents[2]/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'))
STATE=P/'state/exavatar_cuda_validation_v1.json'


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker():
    import torch
    from pytorch3d.ops import knn_points, SubdivideMeshes
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
    torch.set_num_threads(2)
    tests=[]
    points=torch.tensor([[[0.,0.,0.],[1.,2.,0.],[-1.,.1,.5]]])
    targets=torch.tensor([[[.1,0.,0.],[.9,2.2,.1],[-1.1,.2,.4],[5.,5.,5.]]])
    cpu=knn_points(points,targets,K=1)
    gp=points.cuda().requires_grad_();gpu=knn_points(gp,targets.cuda(),K=1)
    assert torch.equal(cpu.idx,gpu.idx.cpu())
    assert torch.allclose(cpu.dists,gpu.dists.cpu(),atol=1e-6,rtol=1e-6)
    gpu.dists.sum().backward();assert torch.isfinite(gp.grad).all() and gp.grad.abs().sum()>0
    tests.append(dict(name='knn_cpu_cuda_values_indices_and_backward',passed=True))
    verts=torch.tensor([[[-.6,-.6,2.],[.6,-.6,2.],[0.,.6,2.]]],device='cuda',requires_grad=True)
    faces=torch.tensor([[[0,1,2]]],device='cuda')
    mesh=Meshes(verts=verts,faces=faces)
    pix,zbuf,bary,dists=rasterize_meshes(mesh,image_size=32,faces_per_pixel=1,bin_size=0)
    valid=pix>=0
    assert valid.any() and torch.allclose(zbuf[valid],torch.full_like(zbuf[valid],2.),atol=1e-5)
    zbuf[valid].sum().backward();assert torch.isfinite(verts.grad).all() and verts.grad.abs().sum()>0
    refined=SubdivideMeshes()(mesh.detach())
    assert refined.faces_packed().shape[0]==4 and refined.verts_packed().shape[0]==6
    tests.append(dict(name='triangle_mesh_cuda_raster_depth_backward_and_subdivision',passed=True,covered_pixels=int(valid.sum())))
    projection=torch.zeros(4,4,device='cuda');projection[0,0]=projection[1,1]=1.
    projection[2,2]=100./99.99;projection[2,3]=-1./99.99;projection[3,2]=1.
    settings=GaussianRasterizationSettings(image_height=48,image_width=48,tanfovx=1.,tanfovy=1.,
        bg=torch.ones(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),
        projmatrix=projection.T.contiguous(),sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
    raster=GaussianRasterizer(settings)
    def run(depth_loss=False):
        means=torch.tensor([[.03,.01,2.]],device='cuda',requires_grad=True)
        means2=torch.zeros_like(means,requires_grad=True)
        rgb=torch.tensor([[.2,.4,.7]],device='cuda',requires_grad=True)
        opacity=torch.tensor([[.8]],device='cuda',requires_grad=True)
        scales=torch.tensor([[.2,.15,.1]],device='cuda',requires_grad=True)
        rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True)
        color,radii,depth,weight=raster(means3D=means,means2D=means2,opacities=opacity,
            colors_precomp=rgb,scales=scales,rotations=rotation)
        assert color.shape==(3,48,48) and radii.shape==(1,) and int(radii[0])>0
        assert depth.numel()==48*48 and weight.numel()==48*48
        assert all(torch.isfinite(x).all() for x in (color,depth,weight))
        assert weight.min()>=0 and weight.max()<=1.00001
        visible=weight>.1;assert visible.any()
        # Upstream returns (sum(z*alpha*T))/(sum(alpha*T)+1e-6).
        expected=2.*(weight[visible]-1e-6)/weight[visible]
        assert torch.allclose(depth[visible],expected,atol=1e-4,rtol=1e-4)
        loss=depth[visible].mean() if depth_loss else color.square().mean()
        loss.backward()
        assert means.grad is not None and torch.isfinite(means.grad).all() and means.grad.abs().sum()>0
        if not depth_loss:
            for variable in (rgb,opacity,scales,rotation,means2):
                assert variable.grad is not None and torch.isfinite(variable.grad).all()
            assert rgb.grad.abs().sum()>0 and opacity.grad.abs().sum()>0
        return dict(name='gaussian_depth_backward' if depth_loss else 'gaussian_rgb_backward',passed=True,
            visible_pixels=int(visible.sum()),peak_weight=float(weight.max()),position_gradient_norm=float(means.grad.norm()))
    tests.extend([run(),run(depth_loss=True)])
    torch.cuda.synchronize()
    result=dict(status='cuda_extension_tests_passed',tests=tests,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_version=torch.__version__,script_sha256=sha(__file__),method_demo_validated=False,
        scope='PyTorch3D KNN, mesh rasterization/subdivision and ExAvatar depth-raster RGB/depth forward/backward only; not full model readiness.',
        native_mask_note='Weight output includes upstream 1e-6 stabilizer. Weight-only gradient is not tested; RGB/depth gradients tested independently.')
    save(P/'reports/exavatar_cuda_validation_v1.json',result);print(json.dumps(result))


def main():
    lock=(P/'state/exavatar_cuda_validation_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    sys.path.insert(0,str(R/'scripts'))
    import dispatch_rigid111_direct as resources
    state=dict(status='waiting_for_pytorch3d_build',pid=os.getpid(),started_at=time.time(),script_sha256=sha(__file__))
    save(STATE,state)
    while True:
        build=json.loads((P/'state/exavatar_pytorch3d_v1.json').read_text())
        if build['status']=='failed':
            state.update(status='dependency_build_failed',reason=build.get('reason'));save(STATE,state);return
        if build['status']=='extension_import_passed_gpu_runtime_pending':break
        state.update(build_pid=build['pid'],build_process_live=resources.live(build['pid']),updated_at=time.time());save(STATE,state)
        time.sleep(10)
    while True:
        state.update(status='waiting_for_shared_gpu_lease',updated_at=time.time());save(STATE,state)
        for gpu in ['4','1','0','2','3','5']:
            with (R/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock').open('a') as lease:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                ready,detail=resources.available(gpu)
                if not ready:continue
                assert sha(__file__)==state['script_sha256']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONNOUSERSITE='1')
                cmd=[str(ENV/'bin/python'),'-u',str(Path(__file__).resolve()),'--worker']
                with (P/'logs/exavatar_cuda_validation_worker_v1.log').open('x') as log:
                    child=subprocess.Popen(cmd,env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                    state.update(status='running',gpu=gpu,gpu_observation=detail,child_pid=child.pid);save(STATE,state)
                    code=child.wait()
                state.update(status='passed' if code==0 else 'failed',exit_code=code,finished_at=time.time());save(STATE,state);return
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker() if args.worker else main()
