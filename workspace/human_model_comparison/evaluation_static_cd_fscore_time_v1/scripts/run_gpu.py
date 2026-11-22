"""One durable, locked GPU child; reuse configured interpreters without upgrades."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,fcntl,os,subprocess,sys,time,shutil
from common import *
P=R/'model_preparation/20261109_2100_CST'
sys.path[:0]=[str(H/'scripts'),str(R/'scripts'),str(P/'scripts')]

def environment(method,gpu):
    import dispatch_rigid111_direct as resources
    if method=='pose':
        env=os.environ.copy();py=_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python')
    elif method=='ours':
        from launch_ours_frozen_time_v2 import environment,PYTHON
        env=environment(gpu);py=PYTHON
    elif method=='omnire':
        from run_omnire_human_legacy_no_smpl_queue import environment,PYTHON
        env=environment(gpu);py=PYTHON
    elif method=='splatad':
        from run_splatad_human_train_only_queue import environment,PYTHON
        env=environment(gpu);py=PYTHON
    else:
        env=resources.environment(method);py=resources.ENVS/resources.METHODS[method]['env']/'bin/python'
    env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return str(py),env

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--key',required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    (E/'state').mkdir(exist_ok=True);(E/'logs').mkdir(exist_ok=True)
    with (E/'state'/f'{a.key}.lock').open('a') as own,(P/f'runs/rigid111_gpu_locks/gpu_{a.gpu}.lock').open('a') as lease,(H/'state'/f'gpu_auxiliary_slot_{a.gpu}.lock').open('a') as auxiliary:
        for f in [own,lease,auxiliary]:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import dispatch_rigid111_direct as resources
        ready,detail=resources.available(a.gpu);assert ready,detail
        assert shutil.disk_usage(E).free>10*1024**3
        py,env=environment(a.method,detail['uuid']);cmd=[py,'-B','-u',*a.command]
        state=dict(status='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],started_at=time.time(),command=cmd,method=a.method,gpu=a.gpu,uuid=detail['uuid'],resource=detail,script_sha256=sha(__file__))
        sp=E/'state'/f'{a.key}.json'
        if sp.exists():
            prior=read(sp);assert prior['status']!='completed','Completed job: inspect bindings rather than rerun'
            save(E/'state/history'/f'{a.key}_{sha(sp)}.json',prior)
        save(sp,state)
        with (E/'logs'/f'{a.key}.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),auxiliary.fileno()))
            state['child_pid']=child.pid;save(sp,state);code=child.wait()
        state.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());save(sp,state)
        print(state['status'],a.key,code,flush=True)

if __name__=='__main__':main()
