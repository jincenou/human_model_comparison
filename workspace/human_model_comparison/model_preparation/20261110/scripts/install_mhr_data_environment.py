"""Install MHR's official locked runtime on the data disk; retain system weights."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl, hashlib, json, os, shutil, signal, subprocess, time
import yaml

P = Path(__file__).resolve().parents[1]
SOURCE = P / 'sources/jacobianavatar/libs/MHR'
ROOT = Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/mhr_runtime'))
TOOLS = Path(_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_pixi_tools'))
STATE = P / 'state/mhr_data_environment_20261113.json'

def save(state):
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(STATE)

with STATE.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), 'Inspect previous attempt before restarting'
    assert not TOOLS.exists(), 'Do not overwrite an existing environment'
    assert shutil.disk_usage('/mnt').free > 30 * 1024**3
    for name in ['tmp', 'conda_pkgs', 'cache', 'pixi_cache', 'uv_cache']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    lockfile = SOURCE / 'pixi.lock'
    digest = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    metadata = yaml.safe_load(lockfile.read_text())
    package = next(p for p in metadata['packages'] if p.get('name') == 'mhr')
    # Archive checkout has no Git HEAD; preserve upstream lock's package metadata
    # only. Actual source identity remains the separately hashed archive commit.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
               TMPDIR=str(ROOT/'tmp'), CONDA_PKGS_DIRS=str(ROOT/'conda_pkgs'),
               XDG_CACHE_HOME=str(ROOT/'cache'), PIXI_CACHE_DIR=str(ROOT/'pixi_cache'),
               UV_CACHE_DIR=str(ROOT/'uv_cache'), SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MHR=package['version'])
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','PYTHONPATH']:
        env.pop(key, None)
    runtime = SOURCE / '.pixi/envs/default'
    state = dict(status='running', pid=os.getpid(), started_at=time.time(), runtime=str(runtime),
                 lock_sha256=digest, archive_source=json.loads((SOURCE/'_source_archive_provenance.json').read_text())['revision'],
                 packaging_version_from_lock=package['version'], weights=_hmc_expand('Existing symlinks into ${HMC_WEIGHT_ROOT}'),
                 full_method_validated=False, stages=[])
    save(state)
    commands = [
        [_hmc_expand('${CONDA_ROOT}/bin/conda'),'create','-y','-p',str(TOOLS),'--override-channels','-c','conda-forge','pixi'],
        [str(TOOLS/'bin/pixi'),'install','--locked'],
        [str(runtime/'bin/uv'),'pip','install','--python',str(runtime/'bin/python'),'--no-deps',
         'smplx==0.1.28','scikit-learn==1.8.0','scipy==1.17.1','tqdm==4.67.3','joblib==1.5.3','threadpoolctl==3.6.0'],
        [str(runtime/'bin/python'),'-c','import mhr, smplx, sklearn; from mhr.mhr import MHR; import torch; m=MHR.from_files(device=torch.device("cpu"),lod=1); print("MHR native body load passed", m.character.mesh.faces.shape)'],
    ]
    try:
        for command in commands:
            stage = dict(command=command,status='running',started_at=time.time());state['stages'].append(stage);save(state)
            child = subprocess.Popen(command,cwd=SOURCE,env=env,start_new_session=True)
            stage['pid']=child.pid;save(state)
            while child.poll() is None:
                if shutil.disk_usage('/mnt').free < 10*1024**3 or time.time()-stage['started_at'] > 21600:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Disk floor or stage time limit reached')
                time.sleep(10)
            stage.update(exit_code=child.returncode,status='passed'if child.returncode==0 else'failed',finished_at=time.time());save(state)
            assert child.returncode==0, 'Inspect MHR installation log'
        assert hashlib.sha256(lockfile.read_bytes()).hexdigest()==digest
        egl=runtime/'lib/libEGL.so'
        if not egl.exists() and not egl.is_symlink() and (runtime/'lib/libEGL.so.1').exists():egl.symlink_to('libEGL.so.1')
        state['status']='runtime_installed_native_body_load_passed_conversion_pending'
    except Exception as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state['finished_at']=time.time();save(state)
