"""Four CPU slots, unchanged human mesh geometry evaluator, one claim per person."""
import argparse,fcntl,os,subprocess,time
import evaluate_human_static_two_models_geometry as geo
src=geo.source

def run(slot):
    geo.prepare();out=geo.OUT;method='hunyuan3d21';status=out/'workers'/f'hunyuan3d21_cpu{slot}.json'
    with (out/f'cpu_slot_{slot}.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            jobs=[j for j in src.read(src.OUT/'queue.json')['jobs'] if j['method']==method];pending=False;executed=False
            for job in jobs:
                sid=job['id'];rp=out/'results'/method/(sid+'.json')
                if rp.exists() and src.read(rp)['status']!='running':continue
                pending=True
                if job['status'] not in ['completed','generation_failed','evaluation_stage_failed']:continue
                lp=out/'claims'/method/(sid+'.lock');lp.parent.mkdir(parents=True,exist_ok=True)
                with lp.open('a') as claim:
                    try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if rp.exists() and src.read(rp)['status']!='running':continue
                    if job['status'] in ['generation_failed','evaluation_stage_failed']:
                        src.save(rp,dict(id=sid,method=method,status='generation_failed' if job['status']=='generation_failed' else 'execution_failure_pending_review',reason='Upstream terminal '+job['status'],upstream_source_failure=job['status']=='evaluation_stage_failed',upstream_failure=job.get('failure'),protocol_sha256=src.sha(out/'protocol.json'),finished_at=time.time()));continue
                    cmd=[src.PY[method],'-B','-u',str(geo.S),'run','--method',method,'--ids',sid];log=out/'logs'/method/(sid+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    src.save(status,dict(status='running_cpu',id=sid,pid=os.getpid(),slot=slot,command=cmd,updated_at=time.time(),claim=str(lp)))
                    with log.open('a') as f:
                        child=subprocess.Popen(cmd,env=src.environment(method,0),cwd=src.H,stdout=f,stderr=subprocess.STDOUT,pass_fds=(claim.fileno(),));code=child.wait()
                    if code:src.save(rp,dict(id=sid,method=method,status='execution_failure_pending_review',reason=f'CPU worker child exit {code}',log=str(log),protocol_sha256=src.sha(out/'protocol.json'),finished_at=time.time()))
                    geo.summary();executed=True;break
            if not pending:break
            if not executed:src.save(status,dict(status='waiting_source_or_other_claim_no_gpu_lease',pid=os.getpid(),slot=slot,updated_at=time.time()));time.sleep(10)
            else:time.sleep(.2)
        geo.summary();src.save(status,dict(status='completed',pid=os.getpid(),slot=slot,finished_at=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=[0,1,2,3],required=True);a=p.parse_args();run(a.slot)
