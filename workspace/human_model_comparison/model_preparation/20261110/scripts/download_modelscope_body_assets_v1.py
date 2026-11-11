"""Fetch pinned community-mirror body assets; no runtime or license claims."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import requests

P = Path(__file__).resolve().parents[1]
OUT = Path(_hmc_expand('${HMC_WEIGHT_ROOT}/modelscope_body_assets_v1'))
STATE = P / 'state/modelscope_body_assets_download_v1.json'
REPO = 'DaDing777/smpl-model'
FILES = ['smplx/SMPLX_MALE.npz', 'smplx/SMPLX_FEMALE.npz', 'smpl/SMPL_NEUTRAL.pkl']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (OUT / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads((P / 'reports/modelscope_body_assets_20261111_v1/DaDing777_smpl-model_files.json').read_text())
    meta = {f['Path']: f for f in inventory['Data']['Files']}
    state = dict(status='running', pid=os.getpid(), started_at=time.time(), repository=REPO,
                 provenance='Community mirror; checksums match mirror metadata, not independent official digests',
                 runtime_validated=False, files=[])
    def save():
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(tmp, STATE)
    session = requests.Session()
    session.trust_env = False
    save()
    try:
        for name in FILES:
            m = meta[name]
            dest = OUT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://modelscope.cn/api/v1/models/{REPO}/repo?' + urllib.parse.urlencode({'Revision': m['Revision'], 'FilePath': name})
            rec = dict(path=str(dest), metadata=m, url=url, status='downloading')
            state['files'].append(rec)
            save()
            if not dest.exists():
                assert shutil.disk_usage(OUT).free - m['Size'] > 10 * 1024**3
                temp = dest.with_suffix(dest.suffix + '.partial')
                assert not temp.exists(), 'Audit existing partial file before restart'
                with session.get(url, stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    with temp.open('xb') as f:
                        for chunk in r.iter_content(4 * 1024**2):
                            f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                assert temp.stat().st_size == m['Size']
                assert sha(temp) == m['Sha256']
                os.replace(temp, dest)
            assert dest.stat().st_size == m['Size'] and sha(dest) == m['Sha256']
            rec.update(status='downloaded_hash_verified', size=dest.stat().st_size, sha256=m['Sha256'])
            save()
            print(name, rec['status'], flush=True)
        state['status'] = 'downloaded_hash_verified_runtime_pending'
    except Exception as e:
        state.update(status='failed', error=repr(e))
        raise
    finally:
        state['finished_at'] = time.time()
        save()

if __name__ == '__main__':
    main()
