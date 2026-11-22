"""Build local pinned wheels without mutating the environment being prepared."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import hashlib,json,os,subprocess,time,shutil,fcntl,argparse

P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_cuda_sources'))
ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train'))
parser=argparse.ArgumentParser()
parser.add_argument('--attempt',default='v1',choices=['v1','v2'])
args=parser.parse_args()
CUDA=Path(_hmc_expand('${CONDA_ROOT}/envs/omnire') if args.attempt=='v1' else _hmc_expand('${CONDA_ROOT}/envs/garden_sam3d'))
suffix='' if args.attempt=='v1' else '_v2'
STATE=P/('state/jacobian_cuda_wheels_20261113'+suffix+'.json')

def save(s):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(STATE)

with STATE.with_suffix('.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'Inspect existing build before restarting'
    assert shutil.disk_usage(ROOT).free>4*1024**3
    version=subprocess.check_output([str(CUDA/'bin/nvcc'),'--version'],text=True)
    assert 'release 12.1' in version
    wheels=ROOT/'wheels';wheels.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_HOME=str(CUDA),CUDA_VISIBLE_DEVICES='',FORCE_CUDA='1',
             TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2',PYTHONNOUSERSITE='1',
             PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT),PIP_NO_INDEX='1')
    env['PATH']=str(ENV/'bin')+os.pathsep+str(CUDA/'bin')+os.pathsep+env.get('PATH','')
    s=dict(status='building',pid=os.getpid(),started_at=time.time(),environment=str(ENV),
           cuda_compiler=str(CUDA/'bin/nvcc'),cuda_version=version,arch='8.9',
           environment_mutated=False,runtime_validated=False,stages=[]);save(s)
    try:
        if args.attempt=='v2':
            prior=json.loads((P/'state/jacobian_cuda_wheels_20261113.json').read_text())
            assert prior['status']=='failed'
            assert (CUDA/'include/cuda/std/type_traits').is_file()
        for name in (['nvdiffrast','diff_gauss'] if args.attempt=='v1' else ['diff_gauss']):
            metadata=json.loads((ROOT/(name+'.json')).read_text());assert metadata['submodules_complete']
            assert hashlib.sha256(Path(metadata['archive']).read_bytes()).hexdigest()==metadata['sha256']
            command=[str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation',
                     '--no-index','--wheel-dir',str(wheels),metadata['path']]
            stage=dict(name=name,command=command,started_at=time.time(),status='running',source=metadata)
            s['stages'].append(stage);save(s)
            with (P/'logs'/f'jacobian_wheel_{name}_20261113{suffix}.log').open('x')as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                stage['pid']=child.pid;save(s);code=child.wait()
            stage.update(status='built'if code==0 else'failed',exit_code=code,finished_at=time.time());save(s)
            assert code==0, 'Inspect native wheel build log'
        s.update(status='wheels_built_install_and_runtime_pending',wheels={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in wheels.glob('*.whl')})
    except Exception as error:s.update(status='failed',reason=repr(error));raise
    finally:s['finished_at']=time.time();save(s)
