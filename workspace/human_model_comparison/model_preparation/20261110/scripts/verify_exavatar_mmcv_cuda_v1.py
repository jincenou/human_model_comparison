"""CPU/CUDA MMCV operator parity under the shared lease, not a method demo."""
from human_asset_benchmark.paths import expand as _hmc_expand
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
STATE=P/'state/exavatar_mmcv_cuda_v1.json'


def save(path,data):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(path)


def worker():
    import torch,mmcv
    from mmcv.ops import nms,roi_align
    assert torch.__version__=='2.6.0+cu124' and mmcv.__version__=='2.1.0'
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[20.,20.,30.,30.]])
    scores=torch.tensor([.9,.8,.7])
    cpu=nms(boxes,scores,.5);gpu=nms(boxes.cuda(),scores.cuda(),.5)
    assert cpu[1].tolist()==gpu[1].cpu().tolist()==[0,2]
    assert torch.equal(cpu[0],gpu[0].cpu())
    feature=torch.arange(2*8*8,dtype=torch.float32).reshape(1,2,8,8)/128
    rois=torch.tensor([[0.,1.,1.,6.,6.],[0.,0.,0.,7.,7.]])
    a=feature.clone().requires_grad_();b=feature.cuda().requires_grad_()
    x=roi_align(a,rois,(3,3),1.,2,'avg',True)
    y=roi_align(b,rois.cuda(),(3,3),1.,2,'avg',True)
    assert torch.allclose(x,y.cpu(),atol=1e-6,rtol=1e-6)
    x.square().sum().backward();y.square().sum().backward()
    assert torch.allclose(a.grad,b.grad.cpu(),atol=1e-6,rtol=1e-6)
    result=dict(status='mmcv_cuda_tests_passed',nms_cpu_cuda_exact=True,
        roi_align_forward_max_error=float((x-y.cpu()).abs().max()),
        roi_align_backward_max_error=float((a.grad-b.grad.cpu()).abs().max()),
        torch_version=torch.__version__,mmcv_version=mmcv.__version__,
        full_method_demo_passed=False,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    save(P/'reports/exavatar_mmcv_cuda_v1.json',result);print(json.dumps(result))


def main():
    lock=STATE.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    build=json.loads((P/'state/exavatar_mmcv_v2.json').read_text())
    assert build['status']=='mmcv_import_cpu_nms_passed_gpu_pending'
    state=dict(status='waiting_shared_gpu_lease',pid=os.getpid(),started_at=time.time(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(STATE,state);sys.path.insert(0,str(R/'scripts'));import dispatch_rigid111_direct as resources
    while True:
        with (R/'runs/rigid111_gpu_locks/gpu_2.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            ready,detail=resources.available('2')
            if ready:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONNOUSERSITE='1')
                with (P/'logs/exavatar_mmcv_cuda_worker_v1.log').open('x') as log:
                    child=subprocess.Popen([_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'),str(Path(__file__)),'--worker'],cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                    state.update(status='running',child_pid=child.pid,gpu='2');save(STATE,state);code=child.wait()
                state.update(status='passed' if code==0 else 'failed',exit_code=code,finished_at=time.time());save(STATE,state);return
        time.sleep(5)


if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
