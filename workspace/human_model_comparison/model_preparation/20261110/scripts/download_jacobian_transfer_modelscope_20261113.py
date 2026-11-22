from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib,json,os,shutil,time
from pathlib import Path
import requests

P=Path(__file__).resolve().parents[1]
REPO='Xbotics_Embodied_AI_Club/Robot-imitation-learning'
NAME='根目录/dataset/body_models/smplx2smpl.pkl'
rows=json.loads((P/'reports/modelscope_extra_inventory_20261113.json').read_text())
meta=next(f for row in rows if row['repo']==REPO for f in row['body']['Data']['Files'] if f['Path']==NAME)
dest=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/jacobian_transfer_modelscope_20261113/smplx2smpl.pkl'))
dest.parent.mkdir(parents=True,exist_ok=True)
state_path=P/'state/jacobian_transfer_modelscope_20261113.json'
assert not state_path.exists()
state=dict(status='downloading',pid=os.getpid(),repo=REPO,file=meta,path=str(dest),bytes_received=0,runtime_validated=False)
def save():
 tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(state_path)
save()
try:
 assert shutil.disk_usage(dest.parent).free>meta['Size']+8*1024**3
 s=requests.Session();s.trust_env=False
 with s.get('https://modelscope.cn/api/v1/models/'+REPO+'/repo',params={'Revision':meta['Revision'],'FilePath':NAME},timeout=(20,60),stream=True) as r:
  r.raise_for_status();tmp=dest.with_suffix('.pkl.partial');assert not tmp.exists() and not dest.exists();h=hashlib.sha256();last=time.monotonic()
  with tmp.open('xb') as f:
   for block in r.iter_content(4*1024**2):
    if not block:continue
    assert shutil.disk_usage(dest.parent).free>7*1024**3
    f.write(block);h.update(block);state['bytes_received']+=len(block);assert state['bytes_received']<=meta['Size']
    if time.monotonic()-last>4:save();last=time.monotonic()
  assert state['bytes_received']==meta['Size'] and h.hexdigest()==meta['Sha256']
  tmp.replace(dest);state.update(status='downloaded_hash_verified_runtime_pending',sha256=h.hexdigest())
except Exception as e:
 state.update(status='failed',reason=repr(e));raise
finally:
 state['finished_at']=time.time();save()
