"""Score every explicitly reviewed clip with the unchanged frozen CPU scorer.

Four independent people in parallel; no reference auto-approval or retry loop.
Re-running discovers newly reviewed people and preserves completed metrics.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

H = Path(__file__).resolve().parents[1]
PY = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python')


def read(p):
    return json.loads(p.read_text())


def main():
    lock = (H/'state/dynamic_scoring_batch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for q in sorted((H/'reports/dynamic_source_clip_v1').glob('HN*/reference_qualification.json')):
        if not read(q).get('reference_eligible'): continue
        if (q.parent/'metrics/metrics.json').exists() and (q.parent/'index.html').exists(): continue
        jobs.append(dict(id=q.parent.name,status='queued'))
    state = dict(status='running',pid=os.getpid(),started_at=time.time(),jobs=jobs,
                 workers=4,device='cpu',reference_auto_approved=False,disk_floor_gib=10)
    path = H/'state/dynamic_scoring_batch.json'
    def save():
        state['updated_at'] = time.time()
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
    def worker(sid):
        assert shutil.disk_usage(H).free > 10*1024**3
        log = H/'logs'/f'dynamic_score_{sid}_{time.time_ns()}.log'
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
        with log.open('x') as stream:
            folder=H/'reports/dynamic_source_clip_v1'/sid
            if not (folder/'metrics/metrics.json').exists():
                result = subprocess.run([PY,'-B','-u',str(H/'scripts/score_dynamic_source_clip.py'),
                        '--human-id',sid],cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode: return result.returncode,str(log)
            assert not read(folder/'metrics/metrics.json')['failures']
            result = subprocess.run([PY,'-B','-u',str(H/'scripts/prepare_dynamic_source_clip.py'),
                    'review','--human-id',sid],cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return result.returncode, str(log)
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(worker,j['id']):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                code, log = future.result()
                result = H/'reports/dynamic_source_clip_v1'/job['id']/'metrics/metrics.json'
                valid = code == 0 and result.exists() and not read(result)['failures']
                job.update(status='scored' if valid else 'failed_retained',exit_code=code,log=log)
            except Exception as error:
                job.update(status='failed_retained',reason=repr(error))
            job['finished_at']=time.time();save();print(job,flush=True)
    result = subprocess.run([PY,'-B',str(H/'scripts/export_dynamic_metric_tables.py')],cwd=H)
    state.update(status='finished',finished_at=time.time(),table_export_exit_code=result.returncode)
    save()


if __name__ == '__main__': main()
