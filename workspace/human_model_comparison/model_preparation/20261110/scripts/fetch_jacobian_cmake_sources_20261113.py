"""Fetch the exact native-build dependencies; leave upstream options unchanged."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import concurrent.futures,hashlib,io,json,shutil,tarfile
import requests

P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_training_dependencies/pinned_cmake_sources'))
jobs=[
 ('robin_map','https://codeload.github.com/Tessil/robin-map/tar.gz/4ec1bf19c6a96125ea22062f38c2cf5b958e448e',None),
 ('eigen','https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz',None),
 ('tetgen','https://codeload.github.com/libigl/tetgen/tar.gz/e05aca7df74e3f531bc35733ed87d36d437266c5',None),
 ('triangle','https://codeload.github.com/libigl/triangle/tar.gz/62f02db9ab4ff4b62d5ff82a77c8ea458c84c23a',None),
 ('predicates','https://codeload.github.com/libigl/libigl-predicates/tar.gz/decb7bc1260e689cbe008109e3cc5d3a5a433aea',None),
 ('embree','https://codeload.github.com/embree/embree/tar.gz/v4.4.0',None),
 ('cgal','https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1-library.tar.xz','ea827f6778063e00554ae41f4c845492'),
 ('boost','https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz','ac857d73bb754b718a039830b07b9624'),
]
def fetch(job):
 name,url,md5=job;record=ROOT/(name+'.json')
 if record.exists():return json.loads(record.read_text())
 s=requests.Session();s.trust_env=False
 try:
  assert shutil.disk_usage(ROOT).free>7*1024**3
  r=s.get(url,timeout=(15,90));r.raise_for_status();raw=r.content
  if md5:assert hashlib.md5(raw).hexdigest()==md5
  archive=ROOT/(name+'.archive');archive.write_bytes(raw)
  with tarfile.open(fileobj=io.BytesIO(raw)) as t:
   members=t.getmembers()
   for m in members:
    assert (ROOT/m.name).resolve().is_relative_to(ROOT)
    if m.issym():assert ((ROOT/m.name).parent/m.linkname).resolve().is_relative_to(ROOT)
    if m.islnk():assert (ROOT/m.linkname).resolve().is_relative_to(ROOT)
   t.extractall(ROOT)
  path=ROOT/members[0].name.split('/')[0]
  result=dict(name=name,url=url,path=str(path),archive=str(archive),sha256=hashlib.sha256(raw).hexdigest(),expected_md5=md5,status='downloaded')
  record.write_text(json.dumps(result,indent=2));print(name,'downloaded',flush=True);return result
 except Exception as e:
  print(name,repr(e),flush=True);return dict(name=name,status='failed',reason=repr(e),url=url)

if __name__=='__main__':
 rows=list(concurrent.futures.ThreadPoolExecutor(3).map(fetch,jobs))
 # Populate the exact submodule discovered from GitHub's v2.7.0 tree.
 robin=next(r for r in rows if r['name']=='robin_map')
 if robin['status']=='downloaded':
  dest=ROOT/'nanobind-2.7.0/ext/robin_map'
  assert dest.is_dir() and not list(dest.iterdir())
  shutil.copytree(robin['path'],dest,dirs_exist_ok=True)
 # Existing pip process already carries the top-level source overrides.
 # Add child overrides to its existing CMake cache before its C++ stage starts.
 cache=ROOT.parent/'libigl-python-bindings-0fbf1c9a82b6986497cec1e98c717a33eaf23f83/build/cp39-cp39-linux_x86_64/CMakeCache.txt'
 if all(r['status']=='downloaded' for r in rows):
  content=cache.read_text()
  for r in rows:
   if r['name']=='robin_map':continue
   key='FETCHCONTENT_SOURCE_DIR_'+r['name'].upper()
   content='\n'.join(line for line in content.splitlines() if not line.startswith(key+':'))+'\n'+key+':PATH='+r['path']+'\n'
  cache.write_text(content)
 (P/'state/jacobian_cmake_sources_20261113.json').write_text(json.dumps(dict(files=rows,all_ready=all(r['status']=='downloaded' for r in rows)),indent=2))
