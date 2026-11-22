"""Download missing FLAME files from the pinned ModelScope publisher snapshot."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import requests

P=Path(__file__).resolve().parents[1]
H=P.parent.parent
INVENTORY=H/'reports/model_readiness_20261112/LHMPP_remote_inventory.json'
DEST=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/exavatar_flame_modelscope'))
STATE=P/'state/exavatar_flame_download.json'
FILES=['human_model_files/flame/FLAME_NEUTRAL.pkl','human_model_files/flame/2019/generic_model.pkl','human_model_files/flame/FLAME_texture.npz']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def main():
    with (P/'state/exavatar_flame_download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists(),'Inspect previous download before resuming'
        meta={r['Path']:r for r in json.loads(INVENTORY.read_text())['Data']['Files']}
        DEST.mkdir(parents=True,exist_ok=True)
        state=dict(status='running',pid=os.getpid(),started_at=time.time(),repository='Damo_XR_Lab/LHMPP-Prior',
                   provenance='ModelScope publisher snapshot; checksums verified against repository metadata, not independent official FLAME digests',files=[],runtime_validated=False)
        def save():
            tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
        session=requests.Session();session.trust_env=False;save()
        try:
            for name in FILES:
                m=meta[name];dest=DEST/Path(name).relative_to('human_model_files');dest.parent.mkdir(parents=True,exist_ok=True)
                rec=dict(path=str(dest),source=name,metadata=m,status='downloading');state['files'].append(rec);save()
                assert shutil.disk_usage(DEST).free-m['Size']>10*1024**3
                if not dest.exists():
                    temp=dest.with_suffix(dest.suffix+'.partial');assert not temp.exists()
                    with session.get('https://modelscope.cn/api/v1/models/Damo_XR_Lab/LHMPP-Prior/repo',params={'Revision':m['Revision'],'FilePath':name},stream=True,timeout=(30,120)) as response:
                        response.raise_for_status()
                        with temp.open('xb') as stream:
                            for chunk in response.iter_content(4*1024**2):stream.write(chunk)
                            stream.flush();os.fsync(stream.fileno())
                    assert temp.stat().st_size==m['Size'] and sha(temp)==m['Sha256']
                    temp.replace(dest)
                assert dest.stat().st_size==m['Size'] and sha(dest)==m['Sha256']
                rec.update(status='downloaded_hash_verified',sha256=m['Sha256']);save();print(name,rec['status'],flush=True)
            state['status']='downloaded_hash_verified_runtime_pending'
        except Exception as error:
            state.update(status='failed',reason=repr(error));raise
        finally:state['finished_at']=time.time();save()

if __name__=='__main__':main()
