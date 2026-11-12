"""Bounded, resumable full-cohort dynamic preparation/render queue.

Reuses completed clips and live controllers. Failures are recorded once; this
queue never approves reference masks or fabricates missing metric scores.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common_human_orbit import H,read,save,sha,checked

PY=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'))
STATE=H/'state/dynamic_full_cohort_v1.json'


def live(pid):
    try:
        raw=Path(f'/proc/{int(pid)}/stat').read_text()
        return raw.rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError,TypeError,ValueError):return False


def acquire():
    from human_native_queue_v3 import shared_slot_available,legacy_owns
    if shutil.disk_usage(H).free<=10*1024**3:return None
    policy=read(H/'state/gpu_resource_policy.json')
    for i in policy['allowed_gpu_indices']:
        assert i not in policy['excluded_gpu_indices']
        lease=(H/'state'/f'gpu_auxiliary_slot_{i}.lock').open('a')
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lease.close();continue
        ready,detail=shared_slot_available(str(i))
        if ready and detail['free_mib']>=32768 and not legacy_owns(str(i)):return lease,str(i),detail
        lease.close()
    return None


def worker(sid):
    root=H/'reports/dynamic_source_clip_v1'/sid
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def command(script,*args):
        assert shutil.disk_usage(H).free>10*1024**3
        subprocess.run([str(PY),'-B','-u',str(H/'scripts'/script),*args],cwd=H,env=env,check=True)
    command('export_dynamic_source_assets.py','--ids',sid)
    if not (root/'clip.json').exists():command('prepare_dynamic_source_clip.py','prepare','--human-id',sid)
    if not (root/'temporal_support.json').exists():command('prepare_dynamic_source_clip.py','support','--human-id',sid)
    import launch_dynamic_source_clip as launcher
    launcher.shared.acquire=acquire
    sys.argv=[launcher.__file__,'--human-id',sid]
    launcher.main()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker');a=ap.parse_args()
    if a.worker:worker(a.worker);return
    lock=(H/'state/dynamic_full_cohort_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'Inspect the existing batch; do not duplicate it'
    active=read(H/'inputs/active_human_benchmark.json');checked(active['selection'],active['selection_sha256'])
    summary=read(H/'reports/source_camera_metric_cohort_v3/summary.json')
    assert summary['selection_sha256']==active['selection_sha256']
    ids=sorted(r['id']for r in read(active['selection'])['objects'])
    rows={(r['id'],r['method']):r for r in summary['rows']};jobs=[]
    for sid in ids:
        missing=[m for m in ['ours','omnire','splatad']if rows[sid,m]['status']!='scored']
        jobs.append(dict(id=sid,status='source_unavailable' if missing else 'queued',missing_methods=missing))
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),selection_sha256=active['selection_sha256'],
        jobs=jobs,gpu_pool=[2,3,4,5,6,7],max_workers=6,disk_floor_gib=10,reference_auto_approved=False,
        script_sha256=sha(__file__),selection_rule='All fixed100, source-common identities in ID order; no quality filtering')
    children={}
    while True:
        external=0
        for job in jobs:
            sid=job['id'];folder=H/'reports/dynamic_source_clip_v1'/sid
            if sid in children:
                child=children[sid];code=child.poll()
                if code is None:continue
                job.update(status='rendered_review_pending' if code==0 else 'failed_retained',exit_code=code,finished_at=time.time())
                del children[sid]
            if job['status'] not in ['queued','existing_live']:continue
            if (folder/'metrics/metrics.json').exists():job['status']='scored_existing';continue
            path=H/'state'/f'dynamic_source_{sid}.json'
            if path.exists():
                prior=read(path)
                if prior['status']=='rendered_reference_review_pending':job['status']='rendered_review_pending';continue
                if prior['status']=='failed':job.update(status='failed_retained',reason=prior.get('reason'));continue
                if live(prior.get('pid')):
                    job.update(status='existing_live',pid=prior['pid']);external+=1;continue
                job.update(status='stopped_requires_audit',reason='Existing controller is absent; retained outputs require inspection');continue
            job['status']='queued'
        state.update(updated_at=time.time(),disk_free_gib=shutil.disk_usage(H).free/1024**3)
        slots=max(0,6-len(children)-external)
        if state['disk_free_gib']>10:
            for job in [j for j in jobs if j['status']=='queued'][:slots]:
                sid=job['id'];log=H/'logs'/f'dynamic_full_{sid}.log'
                with log.open('x')as stream:
                    child=subprocess.Popen([str(PY),'-B','-u',str(Path(__file__).resolve()),'--worker',sid],cwd=H,
                        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4'),stdout=stream,stderr=subprocess.STDOUT)
                children[sid]=child;job.update(status='running',pid=child.pid,started_at=time.time(),log=str(log))
        state['counts']={s:sum(j['status']==s for j in jobs)for s in sorted({j['status']for j in jobs})}
        save(STATE,state)
        if not children and not any(j['status']in ['queued','existing_live']for j in jobs):
            state.update(status='batch_finished_reference_review_required',finished_at=time.time());save(STATE,state);return
        time.sleep(5)


if __name__=='__main__':main()
