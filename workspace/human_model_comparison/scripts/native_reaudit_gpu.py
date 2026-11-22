"""Run one finite metric child with shared GPU leases and durable provenance."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1'
sys.path[:0]=[str(H/'evaluation_static_cd_fscore_time_v1/scripts'),str(R/'scripts'),str(R/'model_preparation/20261109_2100_CST/scripts')]
import run_gpu
import dispatch_rigid111_direct as resources
from native_reaudit_common import read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--env',required=True)
    p.add_argument('--key',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.key.replace('_','').replace('-','').isalnum()
    (ROOT/'jobs').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    sp=ROOT/'jobs'/(a.key+'.json')
    leases=[ROOT/'jobs'/(a.key+'.lock'),resources.P/'runs/rigid111_gpu_locks'/f'gpu_{a.gpu}.lock',H/'state'/f'gpu_auxiliary_slot_{a.gpu}.lock']
    handles=[]
    try:
        for p in leases:
            f=p.open('a');handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ready,detail=resources.available(a.gpu);assert ready,detail
        assert shutil.disk_usage(H).free>10*1024**3
        if sp.exists():
            prior=read(sp);assert prior['status']!='completed','Completed key cannot be silently rerun'
            save(ROOT/'jobs/history'/(a.key+'_'+sha(sp)+'.json'),prior)
        if a.env=='eval':
            py=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python');env=dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env['PATH']=str(Path(py).parent)+':'+env.get('PATH','')
        else:py,env=run_gpu.environment(a.env,detail['uuid'])
        if a.env=='pose':
            # mmdet/mmengine loads the already SHA-verified local checkpoints
            # before run_human_edp_common explicitly reloads their state_dict.
            env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        command=a.command[1:] if a.command and a.command[0]=='--' else a.command
        assert command
        cmd=[py,'-B','-u',*command]
        state=dict(status='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
                   started_at=time.time(),command=cmd,gpu=a.gpu,gpu_uuid=detail['uuid'],environment=a.env,script_sha256=sha(__file__))
        save(sp,state)
        with (ROOT/'logs'/(a.key+'.log')).open('a') as log:
            child=subprocess.Popen(cmd,cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(f.fileno() for f in handles[1:]))
            state['child_pid']=child.pid;save(sp,state);code=child.wait()
        state.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());save(sp,state)
        print(a.key,state['status'],flush=True)
        sys.exit(code)
    finally:
        for f in handles:f.close()


if __name__=='__main__':main()
