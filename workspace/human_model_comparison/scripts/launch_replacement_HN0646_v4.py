"""Independent replacement queue; never rewrite or interrupt pinned v2 jobs."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from launch_ah_source_pilot_v2 import acquire_available_slot as acquire_primary_slot
from human_native_queue_v3 import resources, ENVS
from launch_ours_frozen_time_v2 import environment as ours_environment
from audit_nuscenes_cohort_v2 import sha

H=Path(__file__).resolve().parents[1];S=H/'scripts';SID='HN0646'
sys.path.insert(0,str(H.parent/'vehicle_model_comparison/scripts'))

def commands(method,gpu):
    def cmd(py,script,*args):return [str(py),'-B','-u',str(S/script),*map(str,args)]
    cwd=H
    if method=='ours':
        py=ENVS/'ab_eval/bin/python';env=ours_environment(gpu)
        stages=[cmd(py,'export_ours_frozen_time_replacement_v3.py','--human-id',SID),cmd(py,'render_ours_frozen_time_v2.py','--human-id',SID)]
        validation=H/'exports/ours_frozen_time_v1'/SID/'reload_review_v2/validation.json'
    elif method=='ah':
        py=ENVS/'ab_ah/bin/python';env=resources.environment('ah');key='human_native_v1_'+SID
        env.update(AB_JOB_ID=key,AB_INPUT_ROOT=str(H/'inputs/ah/jobs'/key),AB_PHASE='replacement_cohort_v3',TORCH_HOME=str(H.parent/'assetbench/cache/torch'))
        stages=[cmd(py,'run_ah_human_replacement_v3.py'),cmd(py,'reload_ah_human_pilot_v1.py',key)]
        validation=H/'runs/ah'/key/'reload_validation/validation.json'
    elif method=='omnire':
        from run_omnire_human_legacy_no_smpl_queue import environment,PYTHON,REPO
        env=environment(gpu);cwd=REPO
        stages=[cmd(PYTHON,'extract_omnire_human_replacement_v3.py','--human-id',SID)]
        validation=H/'exports/omnire_human100_v2/objects'/SID/'validation.json'
    else:
        env=resources.environment(method);py=ENVS/resources.METHODS[method]['env']/'bin/python'
        folder=H/'runs'/f'{method}_human_native_v1'/SID
        command=cmd(py,f'run_{method}_human_replacement_v3.py','--sample-id',SID,'--run-dir',folder)
        stages=[command,command+['--reload-only']];validation=folder/'reload_validation.json'
    env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return stages,env,cwd,validation

def acquire_available_slot():
    slot, obs = acquire_primary_slot()
    if slot:
        return slot, obs
    # Explicit user authorization: co-share with existing jobs on all eight cards.
    # One auxiliary lease per card; reserve >=32 GiB measured free memory at launch.
    # Do not acquire or remove the occupied primary training lease.
    from human_native_queue_v3 import shared_slot_available, legacy_owns
    for i in range(8):
        lease = (H/'state'/f'gpu_auxiliary_slot_{i}.lock').open('a')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close(); continue
        ready, detail = shared_slot_available(str(i))
        if ready and detail['free_mib'] >= 32768 and not legacy_owns(str(i)):
            detail['lease_mode'] = 'user_authorized_auxiliary_coshare'
            detail['minimum_free_mib'] = 32768
            return (lease,str(i),detail), obs
        lease.close()
    return None, obs

def main():
    state_path=H/'state/replacement_HN0646_v4.json'
    with (H/'state/replacement_HN0646_v4.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not state_path.exists(),'Audit prior state; no implicit retry'
        selection=H/'inputs/benchmark_human100_v3/selection.json'
        active=json.loads((H/'inputs/active_human_benchmark.json').read_text());assert active['selection']==str(selection) and active['selection_sha256']==sha(selection)
        methods=['ours','ah','trellis_original','omnire','trellis2']
        paths=[selection,H/'inputs/human_static_replacement_v3/manifest.json',H/'inputs/scene_extract_replacement_v3/omnire_plan.json',Path(__file__)]
        for m in methods:
            stages,env,cwd,v=commands(m,'0')
            assert not v.exists(),str(v)
            for command in stages:paths.append(Path(command[3]))
        pins={str(p):sha(p) for p in paths}
        state=dict(status='waiting_resources',pid=os.getpid(),sample_id=SID,removed_id='HN0035',selection=str(selection),selection_sha256=sha(selection),
                   started_at=time.time(),jobs=[],pinned_files=pins,disk_floor_gib=10,launch_headroom_gib=1,
                   scope='Replacement-only native generation/extraction/reload; common-camera and metric acceptance separate',
                   existing99_reused=True,metric_eligible=False)
        def save():
            state['updated_at']=time.time();tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
        save()
        for method in methods:
            job=dict(method=method,status='waiting_resources',stages=[]);state['jobs'].append(job);save()
            while True:
                if shutil.disk_usage(H).free>11*1024**3:
                    slot,obs=acquire_available_slot()
                    state['resource_observation']=obs
                    if slot:break
                state.update(status='waiting_resources',next_method=method);save();time.sleep(10)
            lease,gpu,detail=slot
            with lease:
                try:
                    for p,digest in pins.items():assert sha(p)==digest,p
                    stages,env,cwd,validation=commands(method,detail['uuid'])
                    job.update(status='running',gpu=gpu,gpu_uuid=detail['uuid'],lease_mode=detail.get('lease_mode','primary'),resource_snapshot=detail,started_at=time.time());state['status']='running';save()
                    for i,command in enumerate(stages):
                        assert shutil.disk_usage(H).free>10*1024**3,'Disk reserve reached'
                        log=H/'logs'/f'replacement_v3_{method}_{SID}_{i}.log'
                        with log.open('x') as stream:
                            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),),start_new_session=True)
                            stage=dict(command=command,log=str(log),pid=child.pid,started_at=time.time());job['stages'].append(stage);save()
                            code=child.wait();stage.update(exit_code=code,finished_at=time.time());save();assert code==0, f'Stage {i} failed; see {log}'
                    report=json.loads(validation.read_text());passed=report.get('reload_valid') is True or report.get('status') in ('passed','passed_pending_process_exit')
                    job.update(status=('native_extracted_pending_independent_time_reload' if method=='omnire' else 'native_reload_passed') if passed else 'unavailable',
                               validation=str(validation),validation_sha256=sha(validation),reason=report.get('reason'))
                except Exception as error:job.update(status='failed',reason=repr(error))
                finally:job['finished_at']=time.time();save()
        state.update(status='native_stages_finished_pending_acceptance',finished_at=time.time());save()

if __name__=='__main__':main()
