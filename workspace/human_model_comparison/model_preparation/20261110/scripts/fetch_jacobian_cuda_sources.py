"""Fetch upstream-pinned CUDA sources without installing into an active env."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import tarfile
import time
import requests

P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_cuda_sources'))
JOBS=[('diff_gauss','slothfulxtx/diff-gaussian-rasterization','726449a8af5b933937acf1fc78deeee2beb0f879'),
      ('nvdiffrast','NVlabs/nvdiffrast','729261dc64c4241ea36efda84fbf532cc8b425b8'),
      ('pytorch3d','facebookresearch/pytorch3d','33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba')]
STATE=P/'state/jacobian_cuda_sources_20261113.json'


def fetch(job):
    name,repo,revision=job
    meta=ROOT/(name+'.json')
    if meta.exists():
        prior=json.loads(meta.read_text())
        assert prior['revision']==revision and Path(prior['path']).is_dir()
        return prior
    url=f'https://codeload.github.com/{repo}/tar.gz/{revision}'
    session=requests.Session();session.trust_env=False
    archive=ROOT/(name+'.tar.gz');part=archive.with_suffix('.part')
    start=time.time()
    with session.get(url,stream=True,timeout=(30,60)) as response:
        response.raise_for_status()
        with part.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                assert time.time()-start<300,'Source download deadline'
                stream.write(chunk)
    part.replace(archive)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target=(ROOT/member.name).resolve()
            assert ROOT.resolve() in target.parents and not(member.issym()or member.islnk())
        top=tar.getmembers()[0].name.split('/')[0]
        tar.extractall(ROOT)
    value=dict(name=name,url=url,revision=revision,path=str(ROOT/top),archive=str(archive),
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               submodules_complete=False,status='archive_ready_submodules_require_audit')
    meta.write_text(json.dumps(value,indent=2));print(name,'archive ready',flush=True)
    return value


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not STATE.exists(),'Inspect prior fetch state before restarting'
    s=dict(status='running',pid=os.getpid(),started_at=time.time(),jobs=[])
    STATE.write_text(json.dumps(s,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(fetch,j):j[0]for j in JOBS}
        for future in concurrent.futures.as_completed(futures):
            try:s['jobs'].append(future.result())
            except Exception as e:s['jobs'].append(dict(name=futures[future],status='failed',reason=repr(e)))
            STATE.write_text(json.dumps(s,indent=2))
    s.update(status='sources_fetched'if all(j['status']!='failed'for j in s['jobs'])else'partial_failure',finished_at=time.time())
    STATE.write_text(json.dumps(s,indent=2))
