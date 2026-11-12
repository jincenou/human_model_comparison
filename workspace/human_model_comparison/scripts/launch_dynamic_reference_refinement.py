"""Prepare all pending dynamic reference candidates, six shared GPU workers."""
from human_asset_benchmark.paths import expand as _hmc_expand
import concurrent.futures
import fcntl
import os
from pathlib import Path
import subprocess
import time
from review_dynamic_references import H, ROOT, read, save, sha


def main():
    lock=(H/'state/dynamic_reference_refinement.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[]
    for p in sorted(ROOT.glob('HN*/clip.json')):
        if (p.parent/'reference_qualification.json').exists():continue
        if (p.parent/'mask_candidates_video_seed_v1.json').exists():continue
        clip=read(p);camera=read(H/'configs/human_source_common_v1/reviewed_source_v2'/p.parent.name/'camera.json')
        matched=[f for f in clip['frames'] if f['timestamp_us']==camera['timestamp_us']]
        jobs.append(dict(id=p.parent.name,status='queued' if len(matched)==1 else 'seed_not_in_clip'))
    path=H/'state/dynamic_reference_refinement.json'
    assert not path.exists(), 'Inspect previous batch before restarting'
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),jobs=jobs,workers=6,
               gpu_pool=[2,3,4,5,6,7],reference_auto_approved=False,script_sha256=sha(__file__))
    save(path,state)
    def worker(sid):
        log=H/'logs'/f'dynamic_video_seed_{sid}_batch.log'
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
        with log.open('x') as stream:
            run=subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/hcmp_reference/bin/python'),'-B','-u',
                 str(H/'scripts/refine_dynamic_reference_video.py'),'--ids',sid],cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if run.returncode==0:
                run=subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',
                    str(H/'scripts/review_dynamic_references.py'),'sheets','--variant','video_seed_v1','--ids',sid],
                    cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return run.returncode,str(log)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(worker,j['id']):j for j in jobs if j['status']=='queued'}
        for future in concurrent.futures.as_completed(futures):
            job=futures[future]
            try:
                code,log=future.result();job.update(status='candidate_review_pending' if code==0 else 'failed_retained',exit_code=code,log=log)
            except Exception as error:job.update(status='failed_retained',reason=repr(error))
            job['finished_at']=time.time();state['updated_at']=time.time();save(path,state);print(job,flush=True)
    state.update(status='finished_candidates_require_review',finished_at=time.time());save(path,state)


if __name__=='__main__':main()
