"""Isolated continuation of Ours after v1 queue's two-failure method gate.

The live mixed-method queue stays unchanged. Only its permanently gated Ours
pending/failed identities are consumed here, plus one retained successful control.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parents[1];P=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
Q=H/'runs/ours_reload_continuation_v3';OLD=H/'runs/human_native_queue_v1'
sys.path.insert(0,str(P/'scripts'))
import dispatch_rigid111_direct as resources
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def worker(sid,gpu):
 from launch_ours_frozen_time_v2 import environment,PYTHON
 env=environment(gpu);os.environ['CUDA_VISIBLE_DEVICES']=gpu
 import torch
 visible=torch.zeros(1,device='cuda')
 plan=read(Q/'plan.json')
 for path,digest in plan['code_hashes'].items():assert sha(path)==digest
 assert sid in plan['ids'] and sha(plan['selection'])==plan['selection_sha256']
 folder=H/'exports/ours_frozen_time_v1'/sid
 commands=[]
 if not folder.exists():commands.append(('sample',[str(PYTHON),'-B','-u',str(H/'scripts/export_ours_frozen_time_v1.py'),'--human-id',sid]))
 else:
  p=read(folder/'provenance.json');assert p['selection_sha256']==plan['selection_sha256']
  for name,digest in p['output_sha256'].items():assert sha(folder/name)==digest
 commands.append(('reload_v3',[str(PYTHON),'-B','-u',str(H/'scripts/render_ours_frozen_time_v3.py'),'--human-id',sid]))
 state=dict(id=sid,status='running',pid=os.getpid(),gpu_uuid=gpu,stages=[],started_at=time.time())
 try:
  for stage,cmd in commands:
   assert shutil.disk_usage(H).free>11*1024**3
   step=dict(stage=stage,command=cmd);state['stages'].append(step)
   child=subprocess.Popen(cmd,env=env,pass_fds=(int(os.environ['HUMAN_LEASE_FD']),));step['pid']=child.pid;save(Q/'workers'/f'{sid}.json',state)
   code=child.wait();step['exit_code']=code;assert code==0,(stage,code)
  validation=folder/'reload_review_v3/validation.json';r=read(validation)
  assert r['reload_valid'] and r['analytic_coordinate_check']['passed'] and r['repeat_render_exact_equal']
  state.update(status='passed',validation=str(validation),validation_sha256=sha(validation),metric_eligible=False)
 except BaseException as e:state.update(status='failed',reason=repr(e));raise
 finally:state['finished_at']=time.time();save(Q/'workers'/f'{sid}.json',state)
 del visible
def run():
 Q.mkdir(exist_ok=True);lock=(Q/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (Q/'state.json').exists()
 old=read(OLD/'state.json');assert sum(j['method']=='ours'and j['status']=='failed'for j in old['jobs'])>=2
 assert not any(j['method']=='ours'and j['status']=='running'for j in old['jobs'])
 code=read(OLD/'plan.json')['script_hashes'];old_script=H/'scripts/human_native_queue_v1.py';assert sha(old_script)==code[str(old_script)]
 assert "if failures>=2:return False" in old_script.read_text()
 ids=['HN0145','HN0161','HN0004']+sorted(j['sample_id']for j in old['jobs']if j['method']=='ours'and j['status']=='queued')
 assert len(ids)==75 and len(set(ids))==75
 selection=H/'inputs/benchmark_human100_v2/selection.json'
 plan=dict(ids=ids,pilots=ids[:3],selection=str(selection),selection_sha256=sha(selection),
  previous_queue=str(OLD/'state.json'),previous_controller_unchanged=True,
  code_hashes={str(p):sha(p)for p in [Path(__file__),H/'scripts/render_ours_frozen_time_v3.py',H/'scripts/ours_coordinate_contract_v1.py',H/'scripts/export_ours_frozen_time_v1.py']})
 save(Q/'plan.json',plan);(Q/'workers').mkdir();(Q/'logs').mkdir()
 state=dict(status='running',pid=os.getpid(),jobs=[dict(id=sid,status='queued')for sid in ids],slots={},started_at=time.time(),scores_produced=False)
 guard=threading.RLock()
 def snapshot():
  from collections import Counter
  state.update(updated_at=time.time(),counts=dict(Counter(j['status']for j in state['jobs'])));save(Q/'state.json',state)
 def slot(gpu):
  while True:
   with guard:
    if any(j['status']=='failed'for j in state['jobs'][:3]):return
    candidates=[j for j in state['jobs']if j['status']=='queued'and(j['id']in plan['pilots']or all(x['status']=='passed'for x in state['jobs'][:3]))]
    if not any(j['status']=='queued'for j in state['jobs']):return
   if not candidates:time.sleep(2);continue
   with (P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock').open('a')as lease:
    fcntl.flock(lease,fcntl.LOCK_EX)
    ready,detail=resources.available(gpu)
    if not ready or shutil.disk_usage(H).free<15*1024**3:
     with guard:state['slots'][gpu]=dict(status='waiting_resource',observation=detail);snapshot()
     continue
    with guard:
     candidates=[j for j in state['jobs']if j['status']=='queued'and(j['id']in plan['pilots']or all(x['status']=='passed'for x in state['jobs'][:3]))]
     if not candidates:continue
     job=candidates[0];job.update(status='running',gpu=gpu,started_at=time.time());snapshot()
    cmd=[_hmc_expand('${CONDA_ROOT}/envs/ab_ah/bin/python'),'-B','-u',str(Path(__file__)),'--worker',job['id'],'--gpu',detail['uuid']]
    with (Q/'logs'/f"{job['id']}.log").open('x')as log:
     child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),env=dict(os.environ,HUMAN_LEASE_FD=str(lease.fileno())))
     with guard:job['pid']=child.pid;state['slots'][gpu]=dict(status='running',id=job['id']);snapshot()
     code=child.wait()
    record=Q/'workers'/f"{job['id']}.json";r=read(record)if record.exists()else {}
    with guard:job.update(status='passed'if code==0 and r.get('status')=='passed'else'failed',reason=r.get('reason'),exit_code=code,worker_record=str(record),finished_at=time.time());snapshot()
 with ThreadPoolExecutor(max_workers=3)as pool:list(pool.map(slot,['2','3','4']))
 state.update(status='finished'if all(j['status']=='passed'for j in state['jobs'])else'finished_with_failures_or_blocked_pilot',finished_at=time.time());snapshot()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu');a=p.parse_args()
 if a.worker:worker(a.worker,a.gpu)
 else:run()
