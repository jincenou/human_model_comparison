import json,time,os
from pathlib import Path
import requests
from download_exavatar_flame_models import P,DEST,INVENTORY,sha
names=['human_model_files/smplx/smplx_uv/LICENSE','human_model_files/smplx/smplx_uv/smplx_uv.obj','human_model_files/smplx/smplx_uv/smplx_uv.png']
meta={r['Path']:r for r in json.loads(INVENTORY.read_text())['Data']['Files']}
s=requests.Session();s.trust_env=False;rows=[]
for name in names:
 m=meta[name];p=DEST/Path(name).relative_to('human_model_files');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  response=s.get('https://modelscope.cn/api/v1/models/Damo_XR_Lab/LHMPP-Prior/repo',params={'Revision':m['Revision'],'FilePath':name},timeout=(30,60));response.raise_for_status();tmp=p.with_suffix(p.suffix+'.partial');assert not tmp.exists();tmp.write_bytes(response.content);assert tmp.stat().st_size==m['Size'] and sha(tmp)==m['Sha256'];tmp.replace(p)
 assert sha(p)==m['Sha256'];rows.append(dict(path=str(p),sha256=m['Sha256'],source=name,revision=m['Revision']));print(name,flush=True)
out=P/'state/exavatar_uv_assets_20261113.json';assert not out.exists();out.write_text(json.dumps(dict(status='downloaded_hash_verified_runtime_pending',files=rows),indent=2)+'\n')
