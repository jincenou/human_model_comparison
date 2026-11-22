"""Same frozen evaluation steps, GPU leases only around GPU work.

CPU Cycles views/source, registration, manifest preparation and ED-P reduction do
not reserve a GPU. Each ready group alternates methods when both are available.
"""
import argparse,contextlib,fcntl,hashlib,os,shutil,subprocess,time
from pathlib import Path
import evaluate_human_static_two_models as src

@contextlib.contextmanager
def gpu_lease(gpu):
    with contextlib.ExitStack() as stack:
        fs=[]
        for p in [src.P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock',src.H/f'state/gpu_auxiliary_slot_{gpu}.lock',Path(f'/tmp/drivelab-gpu-locks/gpu{gpu}.lock')]:
            p.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX);fs.append(f)
        yield fs

def worker(gpu,batch_size):
    src.freeze();last_method=None;statuspath=src.OUT/'workers'/f'gpu{gpu}.json'
    while True:
        if shutil.disk_usage(src.H).free<10*1024**3:
            src.save(statuspath,dict(status='waiting_disk_10gib_no_gpu_lease',pid=os.getpid(),gpu=gpu,time=time.time()));time.sleep(30);continue
        gen={(j['method'],j['sample_id']):j for j in src.read(src.GEN/'queue.json')['jobs']}
        with src.queue() as queue:
            ready=[]
            for j in queue['jobs']:
                if j['status']!='awaiting_generation':continue
                g=gen[(j['method'],j['id'])]
                if g['status']=='failed':j.update(status='generation_failed',reason=g.get('reason'));continue
                if g['status']=='completed':ready.append(j)
            method=next((j['method'] for j in ready if j['method']!=last_method),ready[0]['method'] if ready else None)
            jobs=[j for j in ready if j['method']==method][:batch_size]
            for j in jobs:j.update(status='running',pid=os.getpid(),gpu=gpu,started_at=time.time())
            ids=[j['id'] for j in jobs]
            pending=any(j['status'] in ['awaiting_generation','running'] for j in queue['jobs'])
        if not jobs:
            src.summary()
            if not pending:break
            src.save(statuspath,dict(status='waiting_generation_no_gpu_lease',pid=os.getpid(),gpu=gpu,time=time.time()));time.sleep(20);continue
        last_method=method;group=hashlib.sha256((method+','.join(ids)).encode()).hexdigest()[:16]
        steps=[('views',method),('pose','pose'),('register','control'),('source',method),('score','control'),('edp_prepare','control'),('edp_pose','pose'),('edp_features','feature'),('edp_score','control')]
        failed=None
        for action,kind in steps:
            needs_gpu=action in ['pose','score','edp_pose','edp_features'] or (method=='reconviagen' and action in ['views','source'])
            common=dict(pid=os.getpid(),gpu=gpu,method=method,ids=ids,stage=action,gpu_lease_required=needs_gpu,controller=str(Path(__file__).resolve()))
            src.save(statuspath,dict(common,status='waiting_shared_gpu_lease' if needs_gpu else 'running_cpu_without_gpu_lease',started_at=time.time()))
            with (gpu_lease(gpu) if needs_gpu else contextlib.nullcontext([])) as files:
                cmd=[src.PY[kind],'-B','-u',str(src.S),action,'--method',method,'--ids',*ids,'--group',group]
                src.save(statuspath,dict(common,status='running',started_at=time.time(),command=cmd))
                log=src.OUT/'logs'/f'{group}_{action}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as f:
                    child=subprocess.Popen(cmd,env=src.environment(kind,gpu),cwd=src.H,stdout=f,stderr=subprocess.STDOUT,pass_fds=tuple(x.fileno() for x in files));code=child.wait()
                if code:failed=dict(stage=action,exit_code=code,log=str(log));break
            # Give already queued users a scheduling opportunity after a GPU stage.
            if needs_gpu:time.sleep(.2)
        with src.queue() as queue:
            for j in queue['jobs']:
                if j['method']==method and j['id'] in ids:j.update(status='evaluation_stage_failed' if failed else 'completed',failure=failed,finished_at=time.time(),group=group)
        src.summary()
    src.save(statuspath,dict(status='completed',pid=os.getpid(),gpu=gpu,finished_at=time.time(),controller=str(Path(__file__).resolve())));src.summary()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args();worker(args.gpu,args.batch_size)
