"""Build the documented PyTorch3D 0.7.8 for the isolated Flexi fitting env."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,signal,subprocess,time
P=Path(__file__).resolve().parents[1]
ENV=Path(_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_flexi_fitting'))
CUDA=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2'))
ROOT=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/flexi_pytorch3d'))
STATE=P/'state/flexi_fitting_pytorch3d_20261113.json'
def save(s):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
with STATE.with_suffix('.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists()
    assert json.loads((P/'state/jacobian_data_env_flexi_fitting_20261113.json').read_text())['status']=='packages_installed_native_builds_pending'
    meta=json.loads((P/'sources/pytorch3d_V0.7.8_source.json').read_text())
    assert hashlib.sha256(Path(meta['archive']).read_bytes()).hexdigest()==meta['sha256']
    assert shutil.disk_usage('/mnt').free>15*1024**3
    for name in ['tmp','wheels']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    source=ROOT/'source';shutil.copytree(meta['path'],source)
    env=dict(os.environ,CUDA_HOME=str(CUDA),CUDA_VISIBLE_DEVICES='',TORCH_CUDA_ARCH_LIST='8.9',FORCE_CUDA='1',MAX_JOBS='4',
             TMPDIR=str(ROOT/'tmp'),PYTHONNOUSERSITE='1',PIP_NO_INDEX='1')
    env['PATH']=str(ENV/'bin')+os.pathsep+str(CUDA/'bin')+os.pathsep+env.get('PATH','')
    env['CPATH']=str(CUDA/'targets/x86_64-linux/include')+os.pathsep+str(CUDA/'include')
    s=dict(status='building',pid=os.getpid(),source=meta,environment=str(ENV),started_at=time.time(),runtime_validated=False);save(s)
    try:
        cmd=[str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(ROOT/'wheels'),str(source)]
        child=subprocess.Popen(cmd,env=env,start_new_session=True);s['build_pid']=child.pid;save(s)
        while child.poll()is None:
            if shutil.disk_usage('/mnt').free<10*1024**3 or time.time()-s['started_at']>7200:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Disk floor or build deadline reached')
            time.sleep(10)
        assert child.returncode==0,'Inspect official source build before changing versions'
        wheels=list((ROOT/'wheels').glob('pytorch3d-0.7.8-*.whl'));assert len(wheels)==1
        s.update(status='installing',wheel=str(wheels[0]),sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest());save(s)
        subprocess.run([str(ENV/'bin/python'),'-m','pip','install','--no-deps','--no-index',str(wheels[0])],env=env,check=True,timeout=300)
        s['status']='installed_gpu_runtime_pending'
    except Exception as error:s.update(status='failed',reason=repr(error));raise
    finally:s['finished_at']=time.time();save(s)
