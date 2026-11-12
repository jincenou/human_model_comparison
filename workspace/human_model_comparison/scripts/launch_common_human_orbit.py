"""Run existing renderers in their existing environments with an auxiliary GPU lease."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common_human_orbit import H,OUT,prepare,read,save,sha,checked
import trellis_source_batch as shared


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);p.add_argument('--methods',nargs='+',default=['ours','ah','trellis_original','trellis2']);a=p.parse_args()
    sid=a.human_id
    with (H/'state'/('common_human_orbit_'+sid+'_protocol.lock')).open('a')as protocol_lock:
        fcntl.flock(protocol_lock,fcntl.LOCK_EX);prepare(sid)
    key='common_human_orbit_'+sid+'_'+'_'.join(a.methods)
    with (H/'state'/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statepath=H/'state'/(key+'.json');assert not statepath.exists(),'Inspect existing process/output before relaunch'
        ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19])
        state=dict(status='waiting_for_auxiliary_gpu',pid=os.getpid(),start_ticks=ticks,id=sid,methods=a.methods,stages=[],started_at=time.time())
        save(statepath,state);slot=shared.acquire()
        while slot is None:time.sleep(5);slot=shared.acquire()
        lease,gpu,detail=slot
        try:
            sys.path.insert(0,str(H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST/scripts'))
            import dispatch_rigid111_direct as resources
            from launch_ours_frozen_time_v2 import environment as ours_environment,PYTHON as ours_python
            static=H/'scripts/render_common_human_orbit.py';scene=H/'scripts/render_common_scene_orbit.py';core=H/'scripts/common_human_orbit.py'
            renderers={scene if m in ['omnire','splatad']else static for m in a.methods}
            pinned={str(p):sha(p)for p in [core,*renderers]};state.update(status='rendering',gpu=gpu,gpu_uuid=detail['uuid'],pinned=pinned);save(statepath,state)
            for method in a.methods:
                for path,digest in pinned.items():checked(path,digest)
                assert shutil.disk_usage(H).free>10*1024**3
                if method=='ours':py=ours_python;env=ours_environment(detail['uuid'])
                elif method=='omnire':
                    from run_omnire_human_legacy_no_smpl_queue import environment,PYTHON
                    py=PYTHON;env=environment(detail['uuid'])
                elif method=='splatad':
                    from run_splatad_human_train_only_queue import environment,PYTHON
                    py=PYTHON;env=environment(detail['uuid'])
                else:
                    py=Path(_hmc_expand('${CONDA_ROOT}/envs'))/resources.METHODS[method]['env']/'bin/python'
                    env=resources.environment(method)
                env.update(CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                script=scene if method in ['omnire','splatad']else static
                command=[str(py),'-B','-u',str(script),'--human-id',sid,'--method',method]
                stage=dict(method=method,status='running',started_at=time.time(),command=command);state['stages'].append(stage);save(statepath,state)
                child=subprocess.Popen(command,cwd=H,env=env,pass_fds=(lease.fileno(),));stage['pid']=child.pid;save(statepath,state);code=child.wait()
                stage.update(exit_code=code,status='finished'if code==0 else'failed',finished_at=time.time());save(statepath,state)
                if code:raise RuntimeError(method+' orbit failed; retain outputs for inspection')
            state.update(status='rendered_pending_joint_review',finished_at=time.time());save(statepath,state)
        except Exception as error:
            state.update(status='failed',reason=repr(error),finished_at=time.time());save(statepath,state);raise
        finally:lease.close()


if __name__=='__main__':main()
