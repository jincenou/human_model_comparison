"""Populate isolated Flexi tools from official archives, then apply its own adapter files."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,tarfile,time
import requests
P=Path(__file__).resolve().parents[1];F=P/'sources/flexiavatar/smplx_fitting';TOOLS=F/'tools'
ARCH=P/'sources/flexi_tool_archives';ARCH.mkdir(exist_ok=True)
STATE=P/'state/flexi_tool_sources_20261113.json';assert not STATE.exists()
state=dict(status='preparing',pid=os.getpid(),started_at=time.time(),sources=[],full_method_validated=False)
def save():
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
save()
def extract(name,archive,metadata):
    dest=TOOLS/name;assert not dest.exists()
    with tarfile.open(archive)as t:
        members=t.getmembers();top=members[0].name.split('/')[0]
        for m in members:
            parts=Path(m.name).parts;assert parts[0]==top
            if len(parts)==1:continue
            m.name=str(Path(*parts[1:]));assert (dest/m.name).resolve().is_relative_to(dest.resolve())
            assert not m.islnk(), 'Unexpected hard link in source archive'
            if m.issym():assert ((dest/m.name).parent/m.linkname).resolve().is_relative_to(dest.resolve())
            t.extract(m,dest)
    metadata.update(target=str(dest),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    return metadata
def fetch(job):
    name,repo,revision=job;s=requests.Session();s.trust_env=False
    if revision=='main':
        r=s.get('https://api.github.com/repos/'+repo+'/commits/main',timeout=(15,60));r.raise_for_status();revision=r.json()['sha']
    url='https://codeload.github.com/'+repo+'/tar.gz/'+revision
    r=s.get(url,timeout=(15,120));r.raise_for_status();archive=ARCH/(name+'-'+revision+'.tar.gz');archive.write_bytes(r.content)
    return extract(name,archive,dict(repository=repo,revision=revision,url=url))
try:
    previous=json.loads((P/'state/exavatar_fitting_sources_v1.json').read_text())
    for name in ['DECA','Depth-Anything-V2','segment-anything']:
        m=next(j for j in previous['jobs']if Path(j['target']).name==name)
        a=Path(m['archive']);assert hashlib.sha256(a.read_bytes()).hexdigest()==m['archive_sha256']
        state['sources'].append(extract(name,a,dict(repository=m['repository'],revision=m['revision'],source='Existing verified upstream archive, not ExAvatar-modified files')));save()
    with concurrent.futures.ThreadPoolExecutor(2)as pool:
        for result in pool.map(fetch,[('mmpose','open-mmlab/mmpose','v1.3.2'),('SMPLest-X','SMPLCap/SMPLest-X','main')]):
            state['sources'].append(result);save()
    subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B','copy_code.py'],cwd=TOOLS,check=True,timeout=60)
    state['adapter_files']={str(f.relative_to(TOOLS/'code_to_copy')):hashlib.sha256(f.read_bytes()).hexdigest()for f in (TOOLS/'code_to_copy').rglob('*')if f.is_file()}
    state['status']='official_sources_and_flexi_adapters_ready_weights_not_linked'
except Exception as error:state.update(status='failed',reason=repr(error));raise
finally:state['finished_at']=time.time();save()
