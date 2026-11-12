"""Four fixed-view pilots, followed by CPU semantic observations; shared GPU leases."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from launch_ours_source_pilot_v2 import acquire_available_slot,resources,sha

H=Path(__file__).resolve().parents[1]
IDS=['HN0004','HN0035','HN0039','HN0040']


def main():
    state_path=H/'state/trellis_registration_views_v1.json'
    with (H/'state/trellis_registration_views_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():raise RuntimeError('Existing controller must be audited before restart')
        renderer=H/'scripts/render_trellis_registration_views_v1.py'
        keypoints=H/'scripts/run_registration_keypoints_v2.py'
        probe_path=H/'configs/human_source_common_v1/trellis_keypoint_probe_v1.json'
        probe=json.loads(probe_path.read_text())
        pin_paths=[Path(__file__),renderer,keypoints,probe_path,H/'scripts/audit_static_source_cameras_v1.py',
                   H/'scripts/launch_ours_source_pilot_v2.py',Path(resources.__file__)]
        pin_paths += [Path(p) for p in probe['source_sha256']]
        for path,digest in probe['source_sha256'].items():
            if sha(path)!=digest:raise RuntimeError('Changed camera source: '+path)
        for sid in IDS:
            audit_path=H/'reports/static_source_camera_audit_v1'/(sid+'.json')
            audit=json.loads(audit_path.read_text())
            payload=Path(audit['trellis_original']['native_payload'])
            if sha(payload)!=audit['trellis_original']['native_payload_sha256']:raise RuntimeError('Changed native asset')
            pin_paths += [audit_path,payload,payload.parent/'reload_0.png',payload.parent/'reload_validation.json']
            if (H/'exports/trellis_registration_views_v1'/sid).exists():raise RuntimeError('Existing view output: '+sid)
        for case in probe['cases']:
            if case['role']=='reference':
                if sha(case['image'])!=case['image_sha256']:raise RuntimeError('Changed reference image')
                pin_paths.append(Path(case['image']))
        pinned={str(p):sha(p) for p in pin_paths}
        state=dict(status='waiting_shared_resources',pid=os.getpid(),started_at=time.time(),ids=IDS,jobs=[],
                   pinned_sha256=pinned,disk_floor_gib=10,source_orientation_accepted=False,
                   scope='16 fixed native diagnostic views per person; then CPU keypoints; not common orbit or scoring')
        def save():
            state['updated_at']=time.time();tmp=state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
        def check_pins():
            for path,digest in pinned.items():
                if sha(path)!=digest:raise RuntimeError('Pinned content changed: '+path)
        save()
        try:
            for sid in IDS:
                while True:
                    slot,obs=acquire_available_slot()
                    state.update(status='waiting_shared_resources',resource_observation=obs,next_id=sid);save()
                    if slot:break
                    time.sleep(15)
                lease,gpu,detail=slot
                with lease:
                    check_pins()
                    if shutil.disk_usage(H).free<10*1024**3:raise RuntimeError('Disk below launch threshold')
                    env=resources.environment('trellis_original');env['CUDA_VISIBLE_DEVICES']=detail['uuid']
                    py=resources.ENVS/resources.METHODS['trellis_original']['env']/'bin/python'
                    log=H/'logs'/('trellis_registration_views_v1_'+sid+'.log')
                    with log.open('x') as stream:
                        child=subprocess.Popen([str(py),'-B','-u',str(renderer),'--human-id',sid],
                            cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                        job=dict(id=sid,pid=child.pid,gpu=gpu,status='running',started_at=time.time(),log=str(log))
                        state['jobs'].append(job);state['status']='rendering';save();code=child.wait()
                    job.update(status='native_views_rendered' if code==0 else 'failed',exit_code=code,finished_at=time.time());save()
                    if code:raise RuntimeError('Native diagnostic render failed: '+sid)
            # No GPU lease remains held in the CPU phase.
            check_pins()
            cases=[c for c in probe['cases'] if c['role']=='reference']
            manifests=[]
            for sid in IDS:
                path=H/'exports/trellis_registration_views_v1'/sid/'manifest.json'
                result=json.loads(path.read_text())
                if len(result['cases'])!=16 or not result['parameter_equality_after_render']:raise RuntimeError('Incomplete fixed views')
                cases.extend(result['cases']);manifests.append(dict(path=str(path),sha256=sha(path)))
            manifest_path=H/'exports/trellis_registration_views_v1/keypoint_manifest.json'
            with manifest_path.open('x') as stream:
                json.dump(dict(schema='trellis_fixed_view_semantic_inputs_v1',ids=IDS,cases=cases,
                               source_manifests=manifests,scope=state['scope'],metric_eligible=False),stream,indent=2)
            while shutil.disk_usage(H).free<10*1024**3:
                state['status']='waiting_disk_for_cpu_keypoints';save();time.sleep(15)
            cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                         HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
            log=H/'logs/trellis_registration_keypoints_v1.log'
            with log.open('x') as stream:
                child=subprocess.Popen([_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'),'-B','-u',str(keypoints),
                    '--manifest',str(manifest_path),'--output',str(H/'reports/trellis_registration_keypoints_v1')],
                    cwd=H,env=cpu_env,stdout=stream,stderr=subprocess.STDOUT)
                state.update(status='extracting_keypoints_cpu',keypoint_pid=child.pid,keypoint_log=str(log));save();code=child.wait()
            if code:raise RuntimeError('CPU semantic extraction failed')
            state.update(status='native_views_and_semantic_observations_ready_pending_registration_review',scores_produced=False)
        except BaseException as error:
            state.update(status='failed',reason=str(error));raise
        finally:state['finished_at']=time.time();save()


if __name__=='__main__':main()
