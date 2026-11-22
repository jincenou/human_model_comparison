"""Fetch exact conversion wheels, verify PyPI hashes, then install offline."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import concurrent.futures, hashlib, json, os, subprocess, time
import requests

P=Path(__file__).resolve().parents[1]
SOURCE=P/'sources/jacobianavatar/libs/MHR'
ENV=SOURCE/'.pixi/envs/default'
ROOT=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/mhr_runtime/conversion_wheels'))
STATE=P/'state/mhr_conversion_dependencies_20261113.json'
assert not STATE.exists()
prior=json.loads((P/'state/mhr_data_environment_20261113.json').read_text())
assert prior['status']=='failed' and prior['stages'][1]['status']=='passed'
ROOT.mkdir(parents=True,exist_ok=True)
state=dict(status='downloading',pid=os.getpid(),started_at=time.time(),prior_failure='uv index requests timed out',files=[],full_method_validated=False)
def save():
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
save()
def fetch(spec):
    name,version=spec;s=requests.Session();s.trust_env=False
    error=None
    for attempt in range(3):
        try:
            r=s.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=(15,60));r.raise_for_status()
            candidates=[f for f in r.json()['urls'] if f['filename'].endswith('py3-none-any.whl')or('cp312-cp312-manylinux'in f['filename']and f['filename'].endswith('x86_64.whl'))]
            assert candidates,(name,version)
            f=candidates[0];dest=ROOT/f['filename'];tmp=dest.with_suffix('.part')
            with s.get(f['url'],timeout=(15,90),stream=True)as response:
                response.raise_for_status()
                with tmp.open('wb')as out:
                    for chunk in response.iter_content(1024*1024):out.write(chunk)
            assert hashlib.sha256(tmp.read_bytes()).hexdigest()==f['digests']['sha256']
            tmp.replace(dest)
            return dict(name=name,version=version,path=str(dest),url=f['url'],sha256=f['digests']['sha256'])
        except Exception as exc:error=exc
    raise error
try:
    specs=[('smplx','0.1.28'),('scikit-learn','1.8.0'),('scipy','1.17.1'),('tqdm','4.67.3'),('joblib','1.5.3'),('threadpoolctl','3.6.0')]
    with concurrent.futures.ThreadPoolExecutor(3)as pool:
        futures=[pool.submit(fetch,spec)for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();state['files'].append(result);save();print(result['name'],'verified',flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',UV_CACHE_DIR=str(ROOT.parent/'uv_cache'))
    env.pop('PYTHONPATH',None)
    state['status']='installing';save()
    command=[str(ENV/'bin/uv'),'pip','install','--python',str(ENV/'bin/python'),'--no-deps','--no-index']+[f['path']for f in state['files']]
    subprocess.run(command,cwd=SOURCE,env=env,check=True,timeout=300)
    state['status']='native_body_loading';save()
    code='import mhr,smplx,sklearn,torch; from mhr.mhr import MHR; from importlib.metadata import version; m=MHR.from_files(device=torch.device("cpu"),lod=1); print("MHR body load passed",m.character.mesh.faces.shape,"package metadata",version("mhr"))'
    subprocess.run([str(ENV/'bin/python'),'-B','-c',code],cwd=SOURCE,env=env,check=True,timeout=300)
    egl=ENV/'lib/libEGL.so'
    if not egl.exists()and not egl.is_symlink()and(ENV/'lib/libEGL.so.1').exists():egl.symlink_to('libEGL.so.1')
    state['status']='conversion_dependencies_installed_native_body_load_passed'
except Exception as error:state.update(status='failed',reason=repr(error));raise
finally:state['finished_at']=time.time();save()
