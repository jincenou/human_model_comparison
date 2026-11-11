"""Bounded source-camera pilot; wait without holding any GPU ahead of reservations."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from launch_ours_frozen_time_v2 import environment,PYTHON
H=Path(__file__).resolve().parents[1]
P=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
sys.path.insert(0,str(P/'scripts'))
import dispatch_rigid111_direct as resources
IDS=['HN0004','HN0035','HN0039','HN0040']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def acquire_available_slot():
 observations={}
 if shutil.disk_usage(H).free<10*1024**3:return None,{'disk':'below_10_GiB'}
 for i in range(8):
  gpu=str(i);lease=(P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock').open('a')
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:lease.close();observations[gpu]={'status':'shared_lease_busy'};continue
  try:
   ready,detail=resources.available(gpu);observations[gpu]=detail
   if ready:return (lease,gpu,detail),observations
  except BaseException:lease.close();raise
  lease.close()
 return None,observations

def main():
 sp=H/'state/ours_source_pilot_v2.json'
 with (H/'state/ours_source_pilot_v2.lock').open('a') as controller:
  fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not sp.exists(),'Audit existing controller before restart'
  prior=H/'state/ours_source_pilot_v1.json'
  if prior.exists():assert not resources.live(json.loads(prior.read_text()).get('pid'))
  script=H/'scripts/render_ours_source_pilot_v1.py';pinned={str(script):sha(script)}
  for sid in IDS:
   rp=H/'configs/human_source_common_v1/ours_registration_pilot_v1'/(sid+'.json');reg=json.loads(rp.read_text());pinned[str(rp)]=sha(rp)
   assert sha(reg['snapshot'])==reg['snapshot_sha256'] and sha(reg['camera_contract'])==reg['camera_contract_sha256']
   assert not (H/'exports/ours_source_camera_pilot_v1'/sid).exists()
  state=dict(status='waiting_shared_resources',pid=os.getpid(),started_at=time.time(),ids=IDS,jobs=[],code_and_registration_sha256=pinned,disk_floor_gib=10,resource_policy='respect all existing GPU locks, processes and reservations; never hold a GPU while waiting',scores_produced=False)
  def save():
   state['updated_at']=time.time();tmp=sp.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(sp)
  save()
  try:
   for sid in IDS:
    while True:
     slot,obs=acquire_available_slot();state.update(status='waiting_shared_resources',resource_observation=obs,next_id=sid);save()
     if slot:break
     time.sleep(15)
    lease,gpu,detail=slot
    with lease:
     for path,digest in pinned.items():assert sha(path)==digest
     assert shutil.disk_usage(H).free>=10*1024**3
     log=H/'logs'/('ours_source_pilot_v2_'+sid+'.log');cmd=[str(PYTHON),'-B','-u',str(script),'--human-id',sid]
     with log.open('x') as f:
      child=subprocess.Popen(cmd,cwd=H,env=environment(detail['uuid']),stdout=f,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
      job=dict(id=sid,pid=child.pid,gpu=gpu,gpu_uuid=detail['uuid'],status='running',log=str(log),started_at=time.time());state['jobs'].append(job);state['status']='running';save();code=child.wait()
     job.update(status='rendered_pending_acceptance' if code==0 else 'failed',exit_code=code,finished_at=time.time());save();assert code==0,sid
   state['status']='rendered_pending_acceptance'
  except BaseException as e:state.update(status='failed',reason=str(e));raise
  finally:state['finished_at']=time.time();save()
if __name__=='__main__':main()
