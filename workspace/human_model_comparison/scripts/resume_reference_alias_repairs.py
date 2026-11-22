"""Resume only the two batches stopped by the verified camera ROI-path alias."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import evaluate_human_static_two_models as src
import evaluate_human_static_two_models_geometry as geo

ROOT=src.OUT/'reference_alias_repair'
IDS=['HN0646','HN0651','HN0668','HN0685']
METHODS=['hunyuan3d21','reconviagen']


def launch(command,log,env=None):
    with log.open('a') as f:
        p=subprocess.Popen(command,cwd=src.H,env=env or os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(pid=p.pid,command=command,log=str(log),started_at=time.time())


def live(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'resume.lock').open('a')as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'resume.json').exists(),'One bounded metadata recovery only'
        src.freeze();proof=dict(status='resuming_evaluator_failures',pid=os.getpid(),scientific_script_sha256=src.sha(src.S),
                               protocol_sha256=src.sha(src.OUT/'protocol.json'),before_jobs=[],launches=[],geometry_archives=[])
        for method in METHODS:
            row=src.read(src.OUT/method/'observations/HN0646/views.json')
            assert row['reference_metadata_alias_repair']['aliases'][0]['sha256']=='df5ab6d364dc79e5076464139f54f5f9834e2b9f7c11a6b605107f3f3156b324'
        with src.queue()as queue:
            for j in queue['jobs']:
                if j['method'] not in METHODS or j['id']not in IDS:continue
                assert j['status']=='evaluation_stage_failed' and j['failure']['stage']=='views'
                log=Path(j['failure']['log']).read_text();assert "KeyError: 'reference_rgb'"in log
                proof['before_jobs'].append(dict(j))
                j.setdefault('previous_attempts',[]).append(dict(status=j['status'],failure=j['failure'],finished_at=j['finished_at']))
                j.update(status='awaiting_generation',failure=None,resume_reason='Verified existing ROI reference key alias; reuse same generated asset')
        assert len(proof['before_jobs'])==8
        for gpu in [4,5]:
            cmd=[src.PY['control'],'-B','-u',str(src.H/'scripts/run_human_static_two_models_staged.py'),'--gpu',str(gpu),'--batch-size','4']
            proof['launches'].append(launch(cmd,ROOT/f'source_gpu{gpu}.log'))
        src.save(ROOT/'resume.json',proof)
        while True:
            jobs=[j for j in src.read(src.OUT/'queue.json')['jobs']if j['method']in METHODS and j['id']in IDS]
            if any(j['status']=='evaluation_stage_failed' for j in jobs):
                proof.update(status='needs_review',jobs=jobs);src.save(ROOT/'resume.json',proof);return
            if all(j['status']=='completed'for j in jobs):break
            time.sleep(15)
        # Source statuses are now healthy; remove only their propagated error placeholders.
        for method in METHODS:
            for sid in IDS:
                p=geo.OUT/'results'/method/(sid+'.json')
                if p.exists():
                    record=src.read(p);assert record['status']=='execution_failure_pending_review'
                    dest=ROOT/(method+'__'+sid+'__old_geometry_failure.json');dest.write_bytes(p.read_bytes())
                    proof['geometry_archives'].append(dict(path=str(dest),sha256=src.sha(dest)));p.unlink()
        for slot in range(4):
            with (geo.OUT/f'cpu_slot_{slot}.lock').open('a')as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
            cmd=[src.PY['control'],'-B','-u',str(src.H/'scripts/run_human_static_geometry_cpu.py'),'--slot',str(slot)]
            proof['launches'].append(launch(cmd,ROOT/f'geometry_cpu{slot}.log'))
        statepath=geo.OUT/'workers/reconviagen.json'
        state=src.read(statepath) if statepath.exists()else{}
        if not live(state.get('pid',0)):
            cmd=[src.PY['control'],'-B','-u',str(geo.S),'worker','--method','reconviagen','--gpu','5']
            proof['launches'].append(launch(cmd,ROOT/'geometry_recon.log'))
        proof.update(status='source_recovered_geometry_resumed',after_jobs=jobs,finished_at=time.time(),
                     final_scientific_script_sha256=src.sha(src.S),final_protocol_sha256=src.sha(src.OUT/'protocol.json'))
        assert proof['scientific_script_sha256']==proof['final_scientific_script_sha256']
        assert proof['protocol_sha256']==proof['final_protocol_sha256']
        src.save(ROOT/'resume.json',proof)


if __name__=='__main__':main()
