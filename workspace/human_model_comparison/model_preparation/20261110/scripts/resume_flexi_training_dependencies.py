"""Resume the verified partial Open3D wheel, then complete official packages."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
import requests
P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_flexi_training'))
ENV=Path(_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs/hcmp_flexi_training'))
STATE=P/'state/flexi_training_dependencies_resume_20261113.json'
assert not STATE.exists()
prior=json.loads((P/'state/jacobian_data_env_flexi_training_20261113.json').read_text());assert prior['status']=='failed'
state=dict(status='resuming_download',pid=os.getpid(),started_at=time.time(),full_method_validated=False)
def save():
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
save()
try:
    assert shutil.disk_usage('/mnt').free>12*1024**3
    availability=json.loads((P/'state/flexi_dependency_availability_20261113.json').read_text())
    wheel=next(f for r in availability if r['name']=='open3d'for f in r['files']if 'cp38-cp38-manylinux'in f['filename'])
    directory=ROOT/'wheelhouse';directory.mkdir(exist_ok=True);dest=directory/wheel['filename'];part=dest.with_suffix('.part')
    original=ROOT/'tmp/pip-unpack-04x_0k5q'/wheel['filename'];assert original.is_file()
    if not part.exists():shutil.copyfile(original,part)
    session=requests.Session();session.trust_env=False
    state.update(url=wheel['url'],sha256=wheel['sha256'],reused_partial_bytes=part.stat().st_size);save()
    for attempt in range(4):
        try:
            offset=part.stat().st_size
            with session.get(wheel['url'],headers={'Range':f'bytes={offset}-'},stream=True,timeout=(15,90))as response:
                response.raise_for_status()
                if response.status_code==206:
                    assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                    mode='ab'
                else:assert response.status_code==200;mode='wb'
                with part.open(mode)as f:
                    for chunk in response.iter_content(1024*1024):
                        assert shutil.disk_usage('/mnt').free>10*1024**3
                        f.write(chunk)
            assert hashlib.sha256(part.read_bytes()).hexdigest()==wheel['sha256']
            break
        except Exception:
            if attempt==3:raise
    part.replace(dest);state.update(status='installing',wheel=str(dest));save()
    env=dict(os.environ,PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(ROOT/'tmp'),PIP_CACHE_DIR=str(ROOT/'pip_cache'))
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(key,None)
    commands=[prior['stages'][-1]['command']+['--find-links',str(directory)],
              [str(ENV/'bin/python'),'-m','pip','install','--no-deps','--no-build-isolation',str(P/'sources/chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da')],
              [str(ENV/'bin/python'),'-m','pip','check']]
    state['stages']=[]
    for command in commands:
        stage=dict(command=command,status='running');state['stages'].append(stage);save()
        result=subprocess.run(command,env=env,timeout=21600)
        stage.update(status='passed'if result.returncode==0 else'failed',exit_code=result.returncode);save()
        assert result.returncode==0,'Inspect dependency result'
    state['status']='packages_installed_native_builds_pending'
except Exception as error:state.update(status='failed',reason=repr(error));raise
finally:state['finished_at']=time.time();save()
