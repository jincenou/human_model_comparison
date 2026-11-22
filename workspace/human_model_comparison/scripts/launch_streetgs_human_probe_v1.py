"""Finite eight-GPU addition; existing inputs/results are never regenerated."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
O=H/'reports/streetgs_human_probe_v1';S=H/'scripts'
ENVS=Path(_hmc_expand('${CONDA_ROOT}/envs'))
sys.path.insert(0,str(R/'scripts'))
from dispatch_static_contribution_selection import environment,resources

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['source','geometry']);ap.add_argument('--gpus',nargs='+',default=[str(i) for i in range(8)]);a=ap.parse_args()
    assert read(O/'native_adapter_check.json')['status']=='passed'
    plan=read(O/'extraction_plan.json');statepath=O/(a.stage+'_queue.json')
    if a.stage=='source':
        groups={}
        for row in plan['rows']:groups.setdefault(row['scene'],[]).append(row['id'])
        jobs=[dict(scene=k,ids=v,status='queued') for k,v in groups.items()]
    else:
        jobs=[dict(ids=[r['id']],status='queued') for r in plan['rows'] if (H/'exports/streetgs_human_probe_v1/objects'/r['id']/'validation.json').exists() and read(H/'exports/streetgs_human_probe_v1/objects'/r['id']/'validation.json')['status']=='passed']
    assert not statepath.exists(),'Review previous queue instead of silently replacing it'
    state=dict(stage=a.stage,status='running',pid=os.getpid(),started_at=time.time(),jobs=jobs,gpus=a.gpus,disk_reserve_gib=10)
    guard=threading.Lock()
    def persist():state['updated_at']=time.time();save(statepath,state)
    persist()
    def worker(gpu):
        while True:
            with guard:
                if not any(j['status']=='queued' for j in jobs):return
            if shutil.disk_usage(H).free<10*1024**3:raise RuntimeError('10 GiB disk reserve reached')
            lease=(resources.P/'runs/rigid111_gpu_locks'/f'gpu_{gpu}.lock').open('a')
            try:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:time.sleep(5);continue
                if not resources.available(gpu)[0]:time.sleep(5);continue
                with guard:
                    job=next((j for j in jobs if j['status']=='queued'),None)
                    if job is None:return
                    job.update(status='running',gpu=gpu,started_at=time.time());persist()
                tag=job.get('scene') or job['ids'][0];logpath=O/'logs'/f'{a.stage}_{tag}.log';logpath.parent.mkdir(exist_ok=True)
                _,env=environment('streetgs',gpu);env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                try:
                    with logpath.open('w') as log:
                        def invoke(envname,script,args):
                            cmd=[str(ENVS/envname/'bin/python'),'-B','-u',str(S/script),*args]
                            ev=dict(env,PATH=str(ENVS/envname/'bin')+':'+env['PATH'])
                            proc=subprocess.Popen(cmd,env=ev,stdout=log,stderr=log,pass_fds=(lease.fileno(),))
                            with guard:job.update(child_pid=proc.pid,command=cmd,log=str(logpath));persist()
                            code=proc.wait()
                            if code:raise RuntimeError(f'{script}: exit {code}')
                        if a.stage=='source':
                            invoke('ab_cmp_omnire','extract_streetgs_human_background_v1.py',['export','--scene',job['scene']])
                            pending=[]
                            for sid in job['ids']:
                                v=read(H/'exports/streetgs_human_probe_v1/objects'/sid/'validation.json')
                                if v['status']=='exported_pending_reload':pending.append(sid)
                            if pending:invoke('ab_cmp_omnire','extract_streetgs_human_background_v1.py',['reload','--ids',*pending])
                            invoke('ab_eval','score_streetgs_human_static_v1.py',['score','--ids',*job['ids'],'--device','cuda'])
                        else:invoke('ab_cmp_omnire','score_streetgs_human_geometry_v1.py',['run','--ids',*job['ids']])
                    with guard:job.update(status='completed',finished_at=time.time());job.pop('child_pid',None);persist()
                except Exception as e:
                    with guard:job.update(status='failed',reason=repr(e),finished_at=time.time());persist()
            finally:lease.close()
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
    state.update(status='finished',finished_at=time.time());persist()
    print({status:sum(j['status']==status for j in jobs) for status in ['completed','failed','queued']},flush=True)

if __name__=='__main__':main()
