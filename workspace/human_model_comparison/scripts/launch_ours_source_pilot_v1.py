"""Bounded four-person source-camera pilot on the existing shared GPU pool."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from launch_ours_frozen_time_v2 import environment,PYTHON
H=Path(__file__).resolve().parents[1]
P=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
sys.path.insert(0,str(P/'scripts'))
import dispatch_rigid111_direct as resources

def main():
    sp=H/'state/ours_source_pilot_v1.json'
    guard=(H/'state/ours_source_pilot_v1.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not sp.exists(),'Audit existing state before restart'
    s=dict(status='waiting_shared_gpu_lease',pid=os.getpid(),gpu='4',started_at=time.time(),jobs=[])
    def save():
        t=sp.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(sp)
    save()
    try:
        with (P/'runs/rigid111_gpu_locks/gpu_4.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            while True:
                ready,obs=resources.available('4');s.update(status='checking_resources',observation=obs);save()
                if ready and shutil.disk_usage(H).free>11*1024**3:break
                time.sleep(5)
            for sid in ['HN0004','HN0035','HN0039','HN0040']:
                assert shutil.disk_usage(H).free>11*1024**3
                log=H/'logs'/('ours_source_pilot_v1_'+sid+'.log')
                command=[str(PYTHON),'-B','-u',str(H/'scripts/render_ours_source_pilot_v1.py'),'--human-id',sid]
                job=dict(id=sid,status='running',command=command,log=str(log));s['jobs'].append(job)
                with log.open('x') as f:
                    child=subprocess.Popen(command,env=environment(obs['uuid']),stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                    job['pid']=child.pid;s['status']='running';save();code=child.wait()
                job.update(status='rendered_pending_review' if code==0 else 'failed',exit_code=code);save()
                assert code==0, sid
        s['status']='rendered_pending_review'
    except BaseException as e:s.update(status='failed',error=repr(e));raise
    finally:s['finished_at']=time.time();save()

if __name__=='__main__':main()
