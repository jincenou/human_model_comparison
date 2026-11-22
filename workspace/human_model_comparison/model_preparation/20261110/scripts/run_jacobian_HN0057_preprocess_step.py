"""Run an explicitly selected native preprocessing step for one real pilot."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import argparse,fcntl,json,os,shutil,signal,subprocess,time
P=Path(__file__).resolve().parents[1];JA=P/'sources/jacobianavatar'
parser=argparse.ArgumentParser();parser.add_argument('--steps',required=True);parser.add_argument('--gpu',type=int,default=2);parser.add_argument('--attempt',default='1');args=parser.parse_args()
assert all(s in ['1a','2','mv','1b','3','4','5','6','7','finalize']for s in args.steps.split(','))
assert args.attempt.isalnum()and 0<=args.gpu<=7
tag='jacobian_HN0057_steps_'+args.steps.replace(',','_')+'_attempt'+args.attempt+'_20261113'
state_path=P/'state'/(tag+'.json')
def save(s):
    tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(state_path)
with (P/'state/jacobian_HN0057_preprocessing.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not state_path.exists()
    assert shutil.disk_usage('/mnt').free>10*1024**3
    env=dict(os.environ,PREP_PY=_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_jacobian_prep/bin/python'),
             SAPIENS_PY=_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_sapiens2/bin/python'),CONDA_BASE=_hmc_expand('${CONDA_ROOT}'),
             PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_jit'),
             LOG_DIR=str(P/'logs/jacobian_native_preprocess'),HF_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/huggingface'),
             TORCH_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/torch'))
    cuda=Path(_hmc_expand('${CONDA_ROOT}/envs/garden_sam3d'))
    env.update(CUDA_HOME=str(cuda),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',
               CPATH=str(cuda/'targets/x86_64-linux/include')+':'+str(cuda/'include'),
               LIBRARY_PATH=str(cuda/'targets/x86_64-linux/lib')+':'+str(cuda/'lib'))
    env['PATH']=_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_jacobian_prep/bin:')+str(cuda/'bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_jacobian_prep/lib:')+env.get('LD_LIBRARY_PATH','')
    command=['bash','preprocessing/pipeline_a_custom/run_scene.sh','HN0057_native_pilot_20261113',str(args.gpu),args.steps]
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),command=command,gpu=args.gpu,identity='HN0057',full_method_validated=False);save(state)
    try:
        with (P/'logs'/(tag+'.log')).open('x')as log:
            child=subprocess.Popen(command,cwd=JA,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['child_pid']=child.pid;save(state)
            while child.poll()is None:
                if shutil.disk_usage('/mnt').free<10*1024**3 or time.time()-state['started_at']>10800:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Disk floor or preprocessing deadline reached')
                time.sleep(5)
        state.update(status='native_step_passed'if child.returncode==0 else'failed',exit_code=child.returncode)
    except Exception as error:state.update(status='failed',reason=repr(error));raise
    finally:state['finished_at']=time.time();save(state)
