"""Render one prepared real-exposure clip using the existing three methods."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import trellis_source_batch as shared

H=shared.H
sys.path.insert(0,str(H.parent/'vehicle_model_comparison/scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);args=p.parse_args();sid=args.human_id
    active,selected=shared.selection();assert sid in selected
    out=H/'reports/dynamic_source_clip_v1'/sid;clip=shared.read(out/'clip.json')
    assert clip['selection_sha256']==active['selection_sha256'] and clip['identity_key']==selected[sid]['identity_key']
    with (H/'state'/f'dynamic_source_{sid}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=H/'state'/f'dynamic_source_{sid}.json';assert not state_path.exists(),'Inspect prior task before restarting'
        state=dict(status='waiting_for_auxiliary_gpu',pid=os.getpid(),id=sid,started_at=time.time(),stages=[],script_sha256=shared.sha(__file__))
        shared.save(state_path,state)
        slot=shared.acquire()
        while slot is None:time.sleep(5);slot=shared.acquire()
        lease,gpu,detail=slot
        try:
            from launch_ours_frozen_time_v2 import environment as ours_environment,PYTHON as OURS_PY
            from run_omnire_human_legacy_no_smpl_queue import environment as omni_environment,PYTHON as OMNI_PY
            from run_splatad_human_train_only_queue import environment as splat_environment,PYTHON as SPLAT_PY
            maskenv=dict(os.environ,CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
            prepare=H/'scripts/prepare_dynamic_source_clip.py';scene=H/'scripts/render_dynamic_scene_clip.py'
            archive=out/'source_archive';archive.mkdir(exist_ok=True)
            for script in [prepare,scene]:
                copy=archive/(shared.sha(script)+'.py')
                if not copy.exists():copy.write_bytes(script.read_bytes())
            stages=[('masks',Path(_hmc_expand('${CONDA_ROOT}/envs/hcmp_reference/bin/python')),prepare,maskenv,out/'mask_candidates.json'),
                    ('ours',OURS_PY,prepare,ours_environment(detail['uuid']),out/'ours/validation.json'),
                    ('omnire',OMNI_PY,scene,omni_environment(detail['uuid']),out/'omnire/validation.json'),
                    ('splatad',SPLAT_PY,scene,splat_environment(detail['uuid']),out/'splatad/validation.json')]
            state.update(status='rendering',gpu=gpu,gpu_uuid=detail['uuid']);shared.save(state_path,state)
            for action,py,script,env,proof in stages:
                assert shutil.disk_usage(H).free>10*1024**3
                if proof.exists():
                    assert shared.read(proof)['clip_sha256']==shared.sha(out/'clip.json');continue
                command=[str(py),'-B','-u',str(script),action,'--human-id',sid]
                stage=dict(action=action,status='running',command=command,started_at=time.time());state['stages'].append(stage);shared.save(state_path,state)
                print('STAGE',action,sid,'GPU',gpu,flush=True)
                child=subprocess.Popen(command,cwd=H,env=env);stage['pid']=child.pid;shared.save(state_path,state);code=child.wait()
                stage.update(exit_code=code,status='finished' if code==0 else 'failed',finished_at=time.time());shared.save(state_path,state)
                if code:raise RuntimeError(action+' failed; retain outputs and inspect once')
            state.update(status='rendered_reference_review_pending',finished_at=time.time());shared.save(state_path,state)
        except Exception as error:
            state.update(status='failed',reason=repr(error),finished_at=time.time());shared.save(state_path,state);raise
        finally:lease.close()

if __name__=='__main__':main()
