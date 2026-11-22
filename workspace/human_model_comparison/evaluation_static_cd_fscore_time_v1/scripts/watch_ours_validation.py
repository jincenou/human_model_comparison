"""Wait without GPU reservation, then independently validate completed native endpoints."""
import subprocess,time,sys
from common import *
def main():
    finishing=E/'state/time_ours_finish.json'
    while finishing.exists() and read(finishing)['status']=='running':time.sleep(30)
    if finishing.exists():assert read(finishing)['status']=='completed','Export recovery controller requires review'
    for job in [j for j in read(E/'time_plan.json')['runs'] if j['method']=='ours']:
        root=E/'timing/ours_runs'/f'{job["id"]}_r{job["repeat"]}';p=root/'timing.json'
        while not p.exists() or read(p)['status'] not in ['success','failed']:
            state=E/'state/time_ours.json'
            if state.exists() and read(state)['status']=='failed':raise RuntimeError('Generation controller failed; inspect current per-run stage before resuming.')
            time.sleep(30)
        if read(p)['status']=='failed':print('Native attempt failed; retained:',job,flush=True);continue
        audit=root/'independent_reload.json'
        if audit.exists():assert read(audit)['status']=='passed';continue
        key=f'verify_ours_{job["id"]}_r{job["repeat"]}'
        cmd=[sys.executable,'-B','-u',str(E/'scripts/run_gpu.py'),'--method','ours','--gpu','0','--key',key,str(E/'scripts/verify_ours_dispatch.py'),'--run',str(root)]
        subprocess.run(cmd,check=True,cwd=H);assert read(audit)['status']=='passed'
        print('Independent endpoint passed:',job,flush=True)
    save(E/'validation/ours_time_endpoints.json',dict(status='completed',runs=[dict(job,root=str(E/'timing/ours_runs'/f'{job["id"]}_r{job["repeat"]}')) for job in read(E/'time_plan.json')['runs'] if job['method']=='ours'],finished_at=time.time()))
if __name__=='__main__':main()
