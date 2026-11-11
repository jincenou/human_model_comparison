"""Validate three deterministic original75 pilots, then existing-scene actor extraction."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

H = Path(__file__).resolve().parents[1]
R = H.parent/'vehicle_model_comparison'
P = R/'model_preparation/20261109_2100_CST'
Q = H/'runs/omnire_original75_time_v1'
sys.path.insert(0, str(R/'scripts'))
from run_omnire_human_legacy_no_smpl_queue import environment, REPO, PYTHON
sys.path.insert(0, str(P/'scripts'))
import dispatch_rigid111_direct as resources


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    Q.mkdir(exist_ok=True)
    lock = (Q/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (Q/'state.json').exists()
    plan_path = H/'inputs/scene_extract_original75_v1/omnire_plan.json'
    plan = read(plan_path)
    assert sha(plan['selection']) == plan['selection_sha256']
    rows = plan['objects']
    pilots, scenes = [], set()
    for row in rows:
        if row['scene'] not in scenes:
            scenes.add(row['scene']); pilots.append(row['id'])
        if len(pilots) == 3: break
    ids = pilots + [r['id'] for r in rows if r['id'] not in pilots]
    script = H/'scripts/render_omnire_original75_time_v1.py'
    pins = {str(p):sha(p) for p in [Path(__file__), script, plan_path,
        H/'scripts/omnire_time_query_v1.py', R/'scripts/common_camera_contract.py', Path(plan['time_audit'])]}
    state = dict(status='running', pid=os.getpid(), started_at=time.time(), gpu='5',
        pilots=pilots, code_and_input_sha256=pins, no_training=True, scores_produced=False,
        jobs=[dict(id=sid,status='queued') for sid in ids])
    (Q/'logs').mkdir(exist_ok=True)
    def save():
        from collections import Counter
        state.update(updated_at=time.time(), counts=dict(Counter(x['status'] for x in state['jobs'])))
        tmp=Q/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(Q/'state.json')
    save()
    for job in state['jobs']:
        if job['id'] not in pilots and any(j['status']!='passed' for j in state['jobs'][:3]):
            state['status']='pilot_gate_failed';save();return
        if sum(j['status']=='failed' for j in state['jobs']) >= 2:
            state['status']='technical_failure_gate';save();return
        output=H/'exports/common_camera_human_original75_time_v1/omnire'/job['id']
        assert not output.exists()
        while True:
            state['phase']='waiting_shared_gpu_lease';save()
            with (P/'runs/rigid111_gpu_locks/gpu_5.lock').open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                ready, detail=resources.available('5')
                state['resource_observation']=detail;save()
                if ready and shutil.disk_usage(H).free > 15*1024**3:
                    for path,digest in pins.items(): assert sha(path)==digest
                    assert sha(plan['selection'])==plan['selection_sha256']
                    cmd=[str(PYTHON),'-B','-u',str(script),'--human-id',job['id']]
                    logpath=Q/'logs'/(job['id']+'.log')
                    job.update(status='running',started_at=time.time(),command=cmd,log=str(logpath))
                    state['phase']='extracting';save()
                    with logpath.open('x') as log:
                        child=subprocess.Popen(cmd,cwd=REPO,env=environment(detail['uuid']),
                            stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                        job['pid']=child.pid;save();code=child.wait()
                    validation=output/'validation.json'
                    result=read(validation) if validation.exists() else {}
                    good=code==0 and result.get('status')=='passed'
                    if good:
                        assert result['stored_canonical_parameters_equal'] and result['state_time_query_valid']
                        assert result['common_camera_projection_valid'] and result['native_checkpoint_reloaded']
                    job.update(status='passed' if good else 'unavailable' if code==0 and result.get('status')=='unavailable' else 'failed',
                        exit_code=code,reason=result.get('reason'),finished_at=time.time())
                    if validation.exists():job.update(validation=str(validation),validation_sha256=sha(validation))
                    save();break
            time.sleep(5)
    state.update(status='finished' if all(x['status']=='passed' for x in state['jobs']) else 'finished_with_failures',finished_at=time.time())
    save()


if __name__ == '__main__':
    main()
