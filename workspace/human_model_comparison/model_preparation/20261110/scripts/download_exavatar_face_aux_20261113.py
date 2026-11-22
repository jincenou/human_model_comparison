from human_asset_benchmark.paths import expand as _hmc_expand
import json,time,os
from pathlib import Path
import requests
from download_exavatar_flame_models import P,DEST,INVENTORY,sha
names=['s3fd-619a316812.pth','2DFAN4-cd938726ad.zip']
DEST=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/torch_aux/hub/checkpoints'))
inventory=json.loads((P/'reports/modelscope_face_inventory_20261113.json').read_text())
meta={r['Path']:r for row in inventory if row['repo']=='geekane/s3fd619a316812' for r in row['body']['Data']['Files']}
s=requests.Session();s.trust_env=False;rows=[]
for name in names:
 m=meta[name];p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  response=s.get('https://modelscope.cn/api/v1/models/geekane/s3fd619a316812/repo',params={'Revision':m['Revision'],'FilePath':name},timeout=(30,60));response.raise_for_status();tmp=p.with_suffix(p.suffix+'.partial');assert not tmp.exists();tmp.write_bytes(response.content);assert tmp.stat().st_size==m['Size'] and sha(tmp)==m['Sha256'];tmp.replace(p)
 assert sha(p)==m['Sha256'];rows.append(dict(path=str(p),sha256=m['Sha256'],source=name,revision=m['Revision']));print(name,flush=True)
out=P/'state/exavatar_face_aux_20261113.json';assert not out.exists();out.write_text(json.dumps(dict(status='downloaded_hash_verified_runtime_pending',files=rows),indent=2)+'\n')
