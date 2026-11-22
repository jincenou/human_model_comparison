"""Build remaining pinned native wheels without installing into active envs."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl, hashlib, json, os, shutil, signal, subprocess, time

P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_native_wheels'))
PREP=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_cuda_sources'))
ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train'))
CUDA=Path(_hmc_expand('${CONDA_ROOT}/envs/garden_sam3d'))
STATE=P/'state/jacobian_remaining_wheels_20261113.json'
def save(state):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
with STATE.with_suffix('.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    for folder in ['sources','wheels','tmp']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(ROOT).free>20*1024**3
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME=str(CUDA),FORCE_CUDA='1',MMCV_WITH_OPS='1',
             TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
             TMPDIR=str(ROOT/'tmp'),PIP_NO_INDEX='1')
    env['PATH']=str(ENV/'bin')+os.pathsep+str(CUDA/'bin')+os.pathsep+env.get('PATH','')
    env['CPATH']=str(CUDA/'targets/x86_64-linux/include')+os.pathsep+str(CUDA/'include')
    env['LIBRARY_PATH']=str(CUDA/'targets/x86_64-linux/lib')+os.pathsep+str(CUDA/'lib')
    state=dict(status='building',pid=os.getpid(),started_at=time.time(),torch='2.5.1+cu121',
               environment_mutated=False,runtime_validated=False,stages=[]);save(state)
    jobs=[]
    for name in ['simple_knn','pytorch3d']:
        meta=json.loads((PREP/(name+'.json')).read_text());assert meta['submodules_complete']
        assert hashlib.sha256(Path(meta['archive']).read_bytes()).hexdigest()==meta['sha256']
        jobs.append((name,Path(meta['path']),meta))
    for name,folder in [('chumpy','chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da'),('mmcv','mmcv-2.1.0')]:
        archive=P/'sources'/(folder+'.tar.gz')
        jobs.append((name,P/'sources'/folder,dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
    try:
        for name,source,meta in jobs:
            target=ROOT/'sources'/name
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('build','dist','*.egg-info','__pycache__'))
            command=[str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(ROOT/'wheels'),str(target)]
            stage=dict(name=name,source=str(source),provenance=meta,command=command,status='running',started_at=time.time());state['stages'].append(stage);save(state)
            with (P/'logs'/('jacobian_remaining_wheel_'+name+'_20261113.log')).open('x')as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                stage['pid']=child.pid;save(state)
                while child.poll()is None:
                    if shutil.disk_usage(ROOT).free<10*1024**3 or time.time()-stage['started_at']>7200:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise RuntimeError('Disk floor or build deadline reached')
                    time.sleep(10)
            stage.update(status='built'if child.returncode==0 else'failed',exit_code=child.returncode,finished_at=time.time());save(state)
            # Independent builds may continue; failed modules remain explicit.
        state.update(status='wheels_built_runtime_pending'if all(s['status']=='built'for s in state['stages'])else'partial_build_failures',
                     wheels={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in (ROOT/'wheels').glob('*.whl')})
    except Exception as error:state.update(status='failed',reason=repr(error));raise
    finally:state['finished_at']=time.time();save(state)
