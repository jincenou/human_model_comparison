"""Continue the single-person native pilot after its live preprocessing finishes."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
import yaml
P=Path(__file__).resolve().parents[1];JA=P/'sources/jacobianavatar'
PRE=P/'state/jacobian_HN0057_steps_5_6_7_finalize_attempt2_20261113.json'
STATE=P/'state/jacobian_HN0057_training_20261113.json'
ROOT=P/'reports/jacobian_real_sequence_HN0057_20261113_training'
CONFIG=JA/'configs/datasets/Custom-HN0057_native_pilot_20261113.yaml'
def save(s):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
with STATE.with_suffix('.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists()
    s=dict(status='waiting_for_live_preprocessing',pid=os.getpid(),started_at=time.time(),identity='HN0057',gpu=4,
           scope='One real-person demo only; no cohort expansion',preprocessing_state=str(PRE),full_method_validated=False)
    save(s)
    try:
        while True:
            previous=json.loads(PRE.read_text())
            if previous['status']!='running':break
            os.kill(previous['pid'],0)
            assert time.time()-s['started_at']<14400,'Preprocessing wait exceeded 4 hours'
            time.sleep(10)
        assert previous['status']=='native_step_passed','Do not train with failed preprocessing'
        for folder,extension in [('masks','png'),('crop_normals_sapiens2','npy')]:
            assert len(list((ROOT/folder).glob('*.'+extension)))==30
        assert (ROOT/'normals').exists()and shutil.disk_usage('/mnt').free>10*1024**3
        chain={};config=CONFIG
        while config:
            chain[str(config)]=hashlib.sha256(config.read_bytes()).hexdigest();spec=yaml.safe_load(config.read_text())
            config=(config.parent/spec['extends']).resolve()if spec.get('extends')else None
        cuda=Path(_hmc_expand('${CONDA_ROOT}/envs/garden_sam3d'))
        env=dict(os.environ,PY=_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train/bin/python'),CONDA_BASE=_hmc_expand('${CONDA_ROOT}'),
                 CUDA_HOME=str(cuda),TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='4',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                 TORCH_EXTENSIONS_DIR=_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_train_jit'),
                 HF_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/huggingface'),TORCH_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/torch'))
        env['PATH']=_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train/bin:')+str(cuda/'bin')+':'+env.get('PATH','')
        env['LD_LIBRARY_PATH']=_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_jacobian_prep/lib:')+str(JA/'libs/MHR/.pixi/envs/default/lib')+':'+env.get('LD_LIBRARY_PATH','')
        env['CPATH']=str(cuda/'targets/x86_64-linux/include')+':'+str(cuda/'include')
        env['LIBRARY_PATH']=str(cuda/'targets/x86_64-linux/lib')+':'+str(cuda/'lib')
        command=['bash','scripts/usage/01_train_custom.sh',str(CONFIG),'4']
        s.update(status='native_training_running',command=command,config_chain_sha256=chain,training_started_at=time.time(),
                 native_frame_filter='Preserve official joint_valid_frames filtering; report its resulting input coverage, do not silently bypass it');save(s)
        with (P/'logs/jacobian_HN0057_native_training_20261113.log').open('x')as log:
            child=subprocess.Popen(command,cwd=JA,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            s['child_pid']=child.pid;save(s)
            while child.poll()is None:
                if shutil.disk_usage('/mnt').free<10*1024**3:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Data-disk floor reached; native run interrupted and recorded')
                time.sleep(10)
        s.update(status='trained_reload_pending'if child.returncode==0 else'failed',exit_code=child.returncode)
        for name in ['joint_valid_frames.txt','joint_invalid_frames.txt']:
            f=ROOT/name
            if f.exists():s[name]=f.read_text().splitlines()
    except Exception as error:s.update(status='failed',reason=repr(error));raise
    finally:s['finished_at']=time.time();save(s)
