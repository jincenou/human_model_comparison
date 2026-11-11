"""Take the existing GPU lease between rigid jobs, then extract one frozen human."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
P=R/'model_preparation/20261109_2100_CST'
sys.path.insert(0,str(R/'scripts'))
from run_splatad_human_train_only_queue import environment,PYTHON
REPO=R


def main():
    controller=(H/'state/splatad_human_common_v1.lock').open('a')
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=H/'exports/common_camera_human_v1/splatad/HN0004'
    assert not out.exists(), 'Existing extraction must be audited, not overwritten'
    state=dict(status='waiting_shared_gpu_lease',pid=os.getpid(),gpu='4',sample_id='HN0004',
               method='SplatAD existing-scene extraction only',started_at=time.time(),scores_produced=False)
    state_path=H/'state/splatad_human_common_v1.json'
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save()
    try:
        lease_path=P/'runs/rigid111_gpu_locks/gpu_4.lock'
        with lease_path.open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            spec=importlib.util.spec_from_file_location('rigid_resource_check',P/'scripts/dispatch_rigid111_direct.py')
            scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)
            while True:
                ready,detail=scheduler.available('4')
                state.update(status='checking_gpu_and_disk',resource_observation=detail);save()
                if ready and shutil.disk_usage(H).free>11*1024**3:break
                time.sleep(10)
            os.environ['CUDA_VISIBLE_DEVICES']=detail['uuid']
            import torch
            visible_lease=torch.zeros(1,device='cuda')
            command=[str(PYTHON),'-B','-u',str(H/'scripts/render_splatad_human_common_v1.py'),'--human-id','HN0004']
            log_path=H/'logs/splatad_human_common_v1_HN0004.log'
            state.update(status='extracting',command=command,log=str(log_path));save()
            with log_path.open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=environment(detail['uuid']),stdout=log,stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid);save();code=child.wait()
            state.update(exit_code=code)
            assert code==0, f'Extraction exited {code}; preserved log'
            validation=out/'validation.json';result=json.loads(validation.read_text())
            state.update(status='raw_camera_projection_passed_reference_qualification_pending' if result['status']=='passed' else 'native_unavailable',
                         validation=str(validation),result_status=result['status'],gaussians=result.get('gaussians'),reason=result.get('reason'))
            del visible_lease
    except BaseException as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state.update(finished_at=time.time());save()


if __name__=='__main__':main()
