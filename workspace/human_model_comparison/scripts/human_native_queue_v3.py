"""Resumable eight-slot native production queue v3, sharing existing GPU leases.

No formal scoring or new scene training. Failed jobs are preserved, never retried
implicitly. New static methods run HN0004 only until an explicit pilot gate exists.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
from collections import Counter
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
P=R/'model_preparation/20261109_2100_CST'
S=H/'scripts'; ENVS=Path(_hmc_expand('${CONDA_ROOT}/envs'))
Q=H/'runs/human_native_queue_v1'
MAX_METHOD_WORKERS = 8
PER_RUNNING_RESERVE_GIB = 0.5

def disk_launch_allowed(free_gib, jobs):
    running = sum(j['status'] == 'running' for j in jobs)
    return free_gib > 10 + PER_RUNNING_RESERVE_GIB * (running + 1)

sys.path.insert(0,str(P/'scripts'))
import dispatch_rigid111_direct as resources
sys.path.insert(0,str(R/'scripts'))

def allowed_gpus():
    policy=H/"state/gpu_resource_policy.json"
    return json.loads(policy.read_text())["allowed_gpu_indices"] if policy.exists() else list(range(8))

def shared_slot_available(gpu):
    if int(gpu) not in allowed_gpus():
        return False, dict(excluded_by_user=True, reservations=["user_reserved_gpu"], free_mib=0)
    # User explicitly permits sharing occupied GPUs. Keep exclusive task leases
    # and known reservations; permit other processes when measured free VRAM fits.
    _, detail = resources.available(gpu)
    ready = detail['free_mib'] >= 18000 and not detail['reservations']
    detail.update(sharing_authorized=True, required_free_mib=18000)
    return ready, detail

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)

def reviewed_sample_failures(state):
    """Keep audited sample failures terminal without stopping unrelated jobs."""
    path=Q/'failure_review.json'
    if not path.exists():return set()
    review=read(path);assert review['status']=='audited_sample_failures'
    jobs={j['key']:j for j in state['jobs']};keys=set()
    for item in review['failures']:
        job=jobs[item['key']]
        assert item['classification']=='native_empty_sparse_structure'
        assert item['continue_pending_jobs'] is True and item['retry_failed_job'] is False
        assert job['status']=='failed' and job['method']==item['method']=='trellis2'
        assert job['finished_at']==item['finished_at']
        for name in ['worker_record','log']:
            assert job[name]==item[name] and sha(item[name])==item[name+'_sha256']
        worker_record=read(item['worker_record'])
        assert worker_record['status']=='failed' and 'finished_at' in worker_record
        keys.add(item['key'])
    return keys

def plan():
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    rows=read(selection)['objects']; assert len(rows)==100
    jobs=[]
    def add(method,sid,existing=False):
        jobs.append(dict(key=f'{method}_{sid}',method=method,sample_id=sid,
            status='prior_pilot_passed' if existing else 'queued',
            metric_eligible=False,scores_produced=False))
    # Deterministic interleaving prevents one method monopolizing the queue.
    omni={x['id'] for x in read(H/'inputs/scene_extract_v2/omnire_plan.json')['objects'] if x['status']=='ready_for_native_decode'}
    splat={x['id'] for x in read(H/'inputs/scene_extract_v2/splatad_plan.json')['objects'] if x['status']=='ready_for_native_decode'}
    ids=['HN0004']+sorted(x['id'] for x in rows if x['id']!='HN0004')
    evidence={
        'ah':H/'runs/ah/human_pilot_v1_HN0004/reload_validation/validation.json',
        'omnire':H/'exports/common_camera_human_time_v2/omnire/HN0004/validation.json',
        'splatad':H/'exports/common_camera_human_v1/splatad/HN0004/validation.json',
        'ours':H/'exports/ours_frozen_time_v1/HN0004/reload_review_v2/validation.json'}
    for method,path in evidence.items():
        report=read(path)
        assert report.get('status')=='passed' or report.get('reload_valid') is True,(method,path)
    for sid in ids:
        for method in ['trellis_original','trellis2','ah','omnire','splatad','ours']:
            if method=='omnire' and sid not in omni:continue
            if method=='splatad' and sid not in splat:continue
            add(method,sid,sid=='HN0004' and method in ('ah','omnire','splatad','ours'))
    return dict(selection=str(selection),selection_sha256=sha(selection),gpu_pool=list(range(8)),
        scope='Native candidate asset production only; reference, pose and common-camera qualification remain pending',
        pilot_evidence={m:dict(path=str(p),sha256=sha(p)) for m,p in evidence.items()},
        script_hashes={str(p):sha(p) for p in [Path(__file__),S/'run_ah_human_pilot_v1.py',S/'reload_ah_human_pilot_v1.py',
           S/'extract_omnire_human100_v2.py',S/'render_omnire_human_time_v2.py',S/'omnire_time_query_v1.py',
           S/'extract_splatad_human100_v2.py',S/'render_splatad_human_common_v1.py',
           S/'export_ours_frozen_time_v1.py',S/'render_ours_frozen_time_v2.py',
           S/'run_trellis_original_human_native_v1.py',S/'run_trellis2_human_native_v1.py']},jobs=jobs)

def commands(job,gpu):
    method=job['method'];sid=job['sample_id'];env=None;cwd=H
    def pycmd(py,script,*args):return [str(py),'-B','-u',str(S/script),*map(str,args)]
    if method in ('trellis_original','trellis2'):
        env=resources.environment(method);py=ENVS/resources.METHODS[method]['env']/'bin/python'
        folder=H/f'runs/{method}_human_native_v1'/sid
        adapter='resume_trellis2_export_v1.py' if method=='trellis2' and sid in ('HN0221','HN0223') else f'run_{method}_human_native_v1.py'
        cmd=pycmd(py,adapter,'--sample-id',sid,'--run-dir',folder)
        stages=[('generate',cmd),('reload',cmd+['--reload-only'])]
        validation=folder/'reload_validation.json'
    elif method=='ah':
        env=resources.environment('ah');env['TORCH_HOME']=str(H.parent/'assetbench/cache/torch')
        key=f'human_native_v1_{sid}'
        env.update(AB_JOB_ID=key,AB_INPUT_ROOT=str(H/'inputs/ah/jobs'/key),AB_PHASE='human_native_production_v1')
        py=ENVS/'ab_ah/bin/python';folder=H/'runs/ah'/key
        stages=[('generate',pycmd(py,'run_ah_human_pilot_v1.py')),('reload',pycmd(py,'reload_ah_human_pilot_v1.py',key))]
        validation=folder/'reload_validation/validation.json'
    elif method=='omnire':
        from run_omnire_human_legacy_no_smpl_queue import environment,PYTHON,REPO
        env=environment(gpu);cwd=REPO;folder=H/'exports/omnire_human100_v2/objects'/sid
        stages=[('extract',pycmd(PYTHON,'extract_omnire_human100_v2.py','--human-id',sid)),
                ('time_camera_reload',pycmd(PYTHON,'render_omnire_human_time_v2.py','--human-id',sid))]
        validation=H/'exports/common_camera_human_time_v2/omnire'/sid/'validation.json'
    elif method=='splatad':
        from run_splatad_human_train_only_queue import environment,PYTHON
        env=environment(gpu);cwd=R;folder=H/'exports/splatad_human100_v2/objects'/sid
        stages=[('extract',pycmd(PYTHON,'extract_splatad_human100_v2.py','--human-id',sid)),
                ('camera_reload',pycmd(PYTHON,'render_splatad_human_common_v1.py','--human-id',sid))]
        validation=H/'exports/common_camera_human_v1/splatad'/sid/'validation.json'
    elif method=='ours':
        from launch_ours_frozen_time_v2 import environment,PYTHON
        env=environment(gpu);folder=H/'exports/ours_frozen_time_v1'/sid
        stages=[('sample_frozen_time',pycmd(PYTHON,'export_ours_frozen_time_v1.py','--human-id',sid)),
                ('reload_orbit',pycmd(PYTHON,'render_ours_frozen_time_v2.py','--human-id',sid))]
        validation=folder/'reload_review_v2/validation.json'
    else:raise ValueError(method)
    env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return stages,env,cwd,folder,validation

def worker(key,gpu):
    frozen=read(Q/'plan.json');assert sha(frozen['selection'])==frozen['selection_sha256']
    for p,digest in frozen['script_hashes'].items():assert sha(p)==digest,p
    job=next(x for x in frozen['jobs'] if x['key']==key)
    stages,env,cwd,folder,validation=commands(job,gpu)
    resume = job['method']=='trellis2' and job['sample_id'] in ('HN0221','HN0223')
    assert resume or not folder.exists(),f'Existing output needs audit: {folder}'
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    import torch
    assert torch.cuda.device_count()==1
    visible_guard=torch.zeros(1,device='cuda')
    state=dict(key=key,pid=os.getpid(),gpu_uuid=gpu,status='running',started_at=time.time(),stages=[])
    output=Q/'workers'/f'{key}.json'
    try:
        for name,cmd in stages:
            assert shutil.disk_usage(H).free>10*1024**3,'Disk reserve reached; no next stage'
            stage=dict(stage=name,command=cmd,started_at=time.time());state['stages'].append(stage);save(output,state)
            inherited_lease=int(os.environ['HUMAN_LEASE_FD'])
            child=subprocess.Popen(cmd,cwd=cwd,env=env,pass_fds=(inherited_lease,));stage['pid']=child.pid;save(output,state)
            code=child.wait();stage.update(exit_code=code,finished_at=time.time());save(output,state)
            assert code==0,f'{name} failed: exit {code}'
        report=read(validation)
        passed=report.get('reload_valid') is True or report.get('status') in ('passed','passed_pending_process_exit')
        state.update(status='native_passed' if passed else 'unavailable',validation=str(validation),validation_sha256=sha(validation),
            validation_status=report.get('status'),reason=report.get('reason'),metric_eligible=False,scores_produced=False)
    except BaseException as e:
        state.update(status='failed',reason=repr(e),traceback=traceback.format_exc());raise
    finally:state['finished_at']=time.time();save(output,state)
    del visible_guard

def legacy_owns(gpu):
    # This old two-worker dispatcher does not acquire shared flocks. Its live
    # workers currently own 6/7; reserve both until the controller exits.
    path=P/'runs/trellis2_vehicle432_v1/batch_state.json'
    if gpu in ('6','7') and path.exists():
        data=read(path)
        if resources.live(data.get('pid')):return True
    return False

def run():
    with ExitStack() as cleanup:
        # Resume existing plan and job records; all production scripts remain pinned.
        lock=cleanup.enter_context((Q/'controller.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(Q/'state.json');old_pid=state['pid']
        assert not resources.live(old_pid), 'Old controller must have exited before takeover'
        frozen=read(Q/'plan.json')
        assert sha(frozen['selection'])==frozen['selection_sha256']
        for p,digest in frozen['script_hashes'].items():assert sha(p)==digest,p
        assert state['plan_sha256']==sha(Q/'plan.json')
        reviewed_failures=reviewed_sample_failures(state)
        state.update(status='running',pid=os.getpid(),controller_script=str(Path(__file__)),
            controller_script_sha256=sha(__file__),prior_controller_pid=old_pid,
            gpu_pool=allowed_gpus(),disk_launch_floor_gib=10,per_running_reserve_gib=PER_RUNNING_RESERVE_GIB,max_method_workers=MAX_METHOD_WORKERS,resumed_at=time.time(),
            reviewed_sample_failures=sorted(reviewed_failures))
        children={};leases={}
        def snapshot():
            state.update(updated_at=time.time(),counts=dict(Counter(j['status'] for j in state['jobs'])))
            save(Q/'state.json',state)
        def live_worker(pid):
            p=Path('/proc')/str(pid)/'stat'
            try:return p.read_text().split(') ',1)[1].split()[0] not in ('Z','X')
            except FileNotFoundError:return False
        def eligible(j):
            if j['status']!='queued':return False
            method=j['method']
            if sum(x['status']=='failed' and x['method']==method and x['key'] not in reviewed_failures for x in state['jobs'])>=2:return False
            if sum(x['status']=='running' and x['method']==method for x in state['jobs'])>=MAX_METHOD_WORKERS:return False
            if method in ('trellis_original','trellis2') and j['sample_id']!='HN0004':
                p=Q/f'{method}_pilot_gate.json'
                if not p.exists():return False
                g=read(p)
                return bool(g.get('native_production_authorized') and g.get('selection_sha256')==frozen['selection_sha256'] and sha(g['validation'])==g['validation_sha256'])
            return True
        snapshot()
        while True:
            for j in state['jobs']:
                if j['status']!='running':continue
                key=j['key'];child=children.get(key)
                if child is not None:
                    code=child.poll()
                    if code is None:continue
                else:
                    if live_worker(j['child_pid']):continue
                    code=None
                rp=Q/'workers'/f'{key}.json';r=read(rp) if rp.exists() else {}
                terminal=r.get('status') in ('native_passed','unavailable','failed') and 'finished_at' in r
                if terminal and r['status']=='native_passed':
                    assert sha(r['validation'])==r['validation_sha256']
                j.update(status=r['status'] if terminal and code in (0,None) else 'failed',
                    exit_code=code,reason=r.get('reason') if terminal else 'worker exited without terminal validation',
                    worker_record=str(rp),finished_at=r.get('finished_at',time.time()),
                    adopted_worker_exit_code_unknown=child is None)
                if key in leases:leases.pop(key).close()
                children.pop(key,None);snapshot()
            if not any(j['status']=='running' or eligible(j) for j in state['jobs']):break
            for index in allowed_gpus():
                gpu=str(index)
                if any(j['status']=='running' and j.get('gpu')==gpu for j in state['jobs']):continue
                lease=cleanup.enter_context((P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock').open('a'))
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:lease.close();continue
                if legacy_owns(gpu):lease.close();continue
                ready,detail=shared_slot_available(gpu);disk=shutil.disk_usage(H).free/1024**3
                job=next((j for j in state['jobs'] if eligible(j)),None)
                if not ready or not disk_launch_allowed(disk,state['jobs']) or job is None:
                    state['slots'][gpu]=dict(status='waiting_resource_or_gate',ready=ready,free_gib=disk,observation=detail)
                    lease.close();continue
                key=job['key'];logpath=Q/'logs'/f'{key}.log'
                cmd=[str(ENVS/'ab_ah/bin/python'),'-B','-u',str(Path(__file__)),'--worker',key,'--gpu',detail['uuid']]
                if logpath.exists():
                    logpath=Q/'logs'/f'{key}_resume_{time.time_ns()}.log'
                with logpath.open('x') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),start_new_session=True,
                        env=dict(os.environ,HUMAN_LEASE_FD=str(lease.fileno())))
                children[key]=child;leases[key]=lease
                job.update(status='running',gpu=gpu,gpu_uuid=detail['uuid'],started_at=time.time(),child_pid=child.pid,log=str(logpath))
                state['slots'][gpu]=dict(status='running',key=key);snapshot()
            snapshot();time.sleep(5)
        state.update(status='native_queue_finished_or_needs_audit',finished_at=time.time());snapshot()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--gpu');a=parser.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    else:run()
