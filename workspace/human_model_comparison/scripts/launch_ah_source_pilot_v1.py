"""Wait for an actually available shared GPU, then render four AH initializations."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from launch_ours_source_pilot_v2 import acquire_available_slot, resources, sha

H = Path(__file__).resolve().parents[1]
IDS = ['HN0004','HN0035','HN0039','HN0040']
PYTHON = Path(_hmc_expand('${CONDA_ROOT}/envs/ab_ah/bin/python'))


def main():
    state_path = H / 'state/ah_source_pilot_v1.json'
    with (H/'state/ah_source_pilot_v1.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():
            raise RuntimeError('Existing controller state requires audit; do not restart blindly')
        script = H / 'scripts/render_ah_source_pilot_v1.py'
        pin_paths = [Path(__file__),script,H/'scripts/audit_static_source_cameras_v1.py',
                     H/'scripts/launch_ours_source_pilot_v2.py',
                     H.parent/'assetbench/repos/ah/benchmark/utils.py',Path(resources.__file__)]
        for sid in IDS:
            path = H/'configs/human_source_common_v1/ah_registration_pilot_v1'/(sid+'.json')
            reg = json.loads(path.read_text())
            for key,digest in [('native_ply','native_ply_sha256'),('camera_contract','camera_contract_sha256'),
                               ('ahc_camera','ahc_camera_sha256'),('audit_record','audit_record_sha256')]:
                if sha(reg[key]) != reg[digest]:
                    raise RuntimeError('Changed input: '+reg[key])
                pin_paths.append(Path(reg[key]))
            pin_paths.append(path)
            if (H/'exports/ah_source_camera_pilot_v1'/sid).exists():
                raise RuntimeError('Pilot output already exists: '+sid)
        pinned = {str(path):sha(path) for path in pin_paths}
        state = dict(status='waiting_shared_resources',pid=os.getpid(),started_at=time.time(),ids=IDS,jobs=[],
                     code_and_input_sha256=pinned,disk_floor_gib=10,
                     scope='AH actual-camera initialization diagnostics, not training or scoring',
                     resource_policy='shared leases and live reservations respected; no lease held while waiting',
                     scores_produced=False)
        def save():
            state['updated_at']=time.time()
            temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
        save()
        try:
            for sid in IDS:
                while True:
                    slot,observation=acquire_available_slot()
                    state.update(status='waiting_shared_resources',next_id=sid,resource_observation=observation);save()
                    if slot:
                        break
                    time.sleep(15)
                lease,gpu,detail=slot
                with lease:
                    for path,digest in pinned.items():
                        if sha(path)!=digest:
                            raise RuntimeError('Pinned input changed: '+path)
                    if shutil.disk_usage(H).free<10*1024**3:
                        raise RuntimeError('Disk fell below 10 GiB before launch')
                    env=resources.environment('ah');env['CUDA_VISIBLE_DEVICES']=detail['uuid']
                    env['PYTHONUNBUFFERED']='1';env['PYTHONDONTWRITEBYTECODE']='1'
                    log=H/'logs'/('ah_source_pilot_v1_'+sid+'.log')
                    with log.open('x') as stream:
                        child=subprocess.Popen([str(PYTHON),'-B','-u',str(script),'--human-id',sid],
                            cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                        job=dict(id=sid,pid=child.pid,gpu=gpu,gpu_uuid=detail['uuid'],status='running',
                                 started_at=time.time(),log=str(log))
                        state['jobs'].append(job);state['status']='running';save()
                        code=child.wait()
                    job.update(exit_code=code,finished_at=time.time(),status='rendered_pending_acceptance' if code==0 else 'failed');save()
                    if code:
                        raise RuntimeError(f'{sid} render failed, see {log}')
            state['status']='rendered_pending_acceptance'
        except BaseException as error:
            state.update(status='failed',reason=str(error));raise
        finally:
            state['finished_at']=time.time();save()


if __name__=='__main__':
    main()
