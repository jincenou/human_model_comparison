"""Incremental orbit production over the active cohort; reuse completed assets.

The launcher owns the GPU lease. This controller limits concurrent launchers,
retains partial failures, and adopts only PID/start-time-matching live jobs.
No new model generation, alignment, or metric evaluation is performed here.
"""
import argparse
import collections
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from common_human_orbit import H,OUT,METHODS,read,save,sha,checked

LAUNCHER=H/'scripts/launch_common_human_orbit.py'
VERIFIED_OUTPUTS={}


def start_ticks(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,IndexError,ValueError):return None


def alive(job):
    pid=job.get('pid');tick=job.get('start_ticks')
    if not pid or tick is None or start_ticks(pid)!=tick:return False
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return str(LAUNCHER) in args and '--human-id '+job['id'] in args
    except OSError:return False


def output_status(root,sid,method):
    folder=root/sid/method;manifest=folder/'manifest.json'
    if not folder.exists():return 'missing'
    if not manifest.exists():return 'partial_requires_inspection'
    try:
        # Renderers publish one immutable manifest after all images. Avoid
        # hashing every finished image again on each scheduler heartbeat.
        stamp=(manifest.stat().st_mtime_ns,manifest.stat().st_size)
        if VERIFIED_OUTPUTS.get(str(manifest))==stamp:return 'rendered'
        m=read(manifest);protocol=read(root/sid/'protocol.json')
        assert m['id']==sid and m['method']==method
        assert m['protocol_sha256']==sha(root/sid/'protocol.json')
        assert m['timestamp_us']==protocol['timestamp_us'] and len(m['frames'])==16
        for i,row in enumerate(m['frames']):
            assert row['index']==i and row['angle_degrees']==i*22.5
            for key in ['rgb','alpha','rgb_roi','alpha_roi']:checked(row[key],row[key+'_sha256'])
        VERIFIED_OUTPUTS[str(manifest)]=stamp
        return 'rendered'
    except (OSError,ValueError,AssertionError,KeyError):return 'invalid_requires_inspection'


def pending_groups(rows,methods,root,active_ids,attempted):
    groups=collections.OrderedDict();inventory=[]
    for row in rows:
        sid,method=row['id'],row['method']
        if method not in methods:continue
        if sid in active_ids:state='running'
        elif row['status']!='scored':state='source_unavailable:'+row['status']
        else:state=output_status(root,sid,method)
        inventory.append(dict(id=sid,method=method,status=state))
        if state=='missing' and (sid,method) not in attempted:
            groups.setdefault(sid,[]).append(method)
    return groups,inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',choices=METHODS,default=['ours','ah','trellis_original','trellis2']);p.add_argument('--max-workers',type=int,default=4);a=p.parse_args()
    assert 1<=a.max_workers<=4
    key='common_human_orbit_batch_'+'_'.join(a.methods);path=H/'state'/(key+'.json')
    with (H/'state'/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=read(H/'inputs/active_human_benchmark.json');checked(active['selection'],active['selection_sha256'])
        selection=read(active['selection']);ids={r['id'] for r in selection['objects']};assert len(ids)==100
        state=read(path)if path.exists()else dict(jobs=[],created_at=time.time(),selection_sha256=active['selection_sha256'])
        assert state['selection_sha256']==active['selection_sha256']
        scripts=[LAUNCHER,H/'scripts/common_human_orbit.py',Path(__file__)]
        if any(m in ['ours','ah','trellis_original','trellis2']for m in a.methods):scripts.append(H/'scripts/render_common_human_orbit.py')
        if any(m in ['omnire','splatad']for m in a.methods):scripts.append(H/'scripts/render_common_scene_orbit.py')
        pinned={str(x):sha(x)for x in scripts}
        if state.get('pinned'):assert state['pinned']==pinned,'Changed batch implementation; inspect previous state first'
        state.update(status='running',pid=os.getpid(),start_ticks=start_ticks(os.getpid()),pinned=pinned,methods=a.methods,max_workers=a.max_workers,updated_at=time.time())
        handles={}
        while True:
            for script,digest in pinned.items():checked(script,digest)
            for job in state['jobs']:
                if job['status']!='running':continue
                child=handles.get(job['pid']);code=child.poll()if child else None
                if (child is not None and code is None)or(child is None and alive(job)):continue
                result=read(job['state'])if Path(job['state']).exists()else{}
                done=all(output_status(OUT,job['id'],m)=='rendered'for m in job['methods'])
                job.update(status='rendered'if done else'failed_retained',exit_code=code,launcher_status=result.get('status'),finished_at=time.time())
            summary=read(H/'reports/source_camera_metric_cohort_v3/summary.json')
            assert summary['selection_sha256']==active['selection_sha256']
            assert {r['id']for r in summary['rows']}==ids
            running=[j for j in state['jobs']if j['status']=='running']
            attempted={(j['id'],m)for j in state['jobs']for m in j['methods']}
            groups,inventory=pending_groups(summary['rows'],a.methods,OUT,{j['id']for j in running},attempted)
            # Jobs are serialized within each person and share one existing GPU
            # lease, so a person can never render the same method twice.
            for sid,methods in list(groups.items())[:max(0,a.max_workers-len(running))]:
                launchkey='common_human_orbit_'+sid+'_'+'_'.join(methods)
                launcherstate=H/'state'/(launchkey+'.json')
                job=dict(id=sid,methods=methods,status='running',state=str(launcherstate),started_at=time.time())
                if launcherstate.exists():
                    old=read(launcherstate);job.update(pid=old.get('pid'),start_ticks=old.get('start_ticks'))
                    if alive(job):job['adopted']=True
                    else:job.update(status='failed_retained',reason='Existing launcher state requires inspection; no automatic retry')
                else:
                    log=H/'logs'/(launchkey+'.log');command=[sys.executable,'-B','-u',str(LAUNCHER),'--human-id',sid,'--methods',*methods]
                    with log.open('ab')as stream:child=subprocess.Popen(command,cwd=H,stdout=stream,stderr=subprocess.STDOUT)
                    job.update(pid=child.pid,start_ticks=start_ticks(child.pid),log=str(log),command=command);handles[child.pid]=child
                state['jobs'].append(job)
                save(path,state)
            state.update(updated_at=time.time(),inventory=inventory,counts=dict(collections.Counter(r['status']for r in inventory)),source_coverage=summary['coverage'],denominator=100)
            active_jobs=[j for j in state['jobs']if j['status']=='running']
            if not active_jobs and not groups:
                state.update(status='current_ready_outputs_finished',finished_at=time.time(),visual_acceptance='Only explicitly reviewed pilot images are visually accepted; batch images await review')
                save(path,state);print(state['status'],state['counts'],flush=True);return
            save(path,state);time.sleep(10)


if __name__=='__main__':main()
