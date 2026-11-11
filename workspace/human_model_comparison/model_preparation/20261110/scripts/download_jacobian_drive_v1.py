"""Download explicitly identified public upstream assets; retain checksums and provenance."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse
import zipfile
import requests

P = Path(__file__).resolve().parents[1]
DEST = Path(_hmc_expand('${HMC_WEIGHT_ROOT}/jacobian_drive_v1'))
STATE = P / 'state/jacobian_drive_download_v1.json'


class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    lock = STATE.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists()
    DEST.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', pid=os.getpid(), started_at=time.time(),
                 script_sha256=sha(__file__), jobs=[], method_runtime_validated=False)
    def save():
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(STATE)
    jobs = [
        ('phmr_config', '1P3EEmBDeRRORhBkUhwPZclZxd7U3Z-yo', 'config.yaml', 1024*1024),
        ('phmr_checkpoint', '1uQwMCkkqtyQIwuBeWwE83XcOlISKbTHB', 'checkpoint.ckpt', 3*1024**3),
        ('waft_a2', '1RA4j0GlNCy7flmc3VIY71ZV-Kz-RYDox', 'a2.zip', 5*1024**3),
    ]
    for name, file_id, filename, limit in jobs:
        job = dict(name=name, file_id=file_id, status='downloading', bytes_received=0,
                   source='https://drive.google.com/uc?export=download&id='+file_id,
                   path=str(DEST/filename), checksum_basis='Locally computed SHA256 after official Google Drive transfer; upstream digest not published.')
        state['jobs'].append(job); save()
        try:
            assert shutil.disk_usage(DEST).free > limit + 12*1024**3
            with requests.Session() as session:
                response = session.get(job['source'], stream=True, timeout=(25,120))
                response.raise_for_status()
                if 'text/html' in response.headers.get('Content-Type', ''):
                    page = response.text; response.close()
                    form = Form(); form.feed(page)
                    assert form.action and urlparse(form.action).hostname == 'drive.usercontent.google.com', page[:300]
                    assert form.fields.get('id') == file_id
                    response = session.get(form.action, params=form.fields, stream=True, timeout=(25,120))
                    response.raise_for_status()
                assert 'text/html' not in response.headers.get('Content-Type',''), response.text[:300]
                length = int(response.headers.get('Content-Length',0))
                assert not length or length <= limit, (length,limit)
                output = DEST/filename; partial = output.with_suffix(output.suffix+'.part')
                assert not output.exists() and not partial.exists()
                last = time.monotonic()
                with partial.open('xb') as handle:
                    for block in response.iter_content(4*1024*1024):
                        if not block: continue
                        assert shutil.disk_usage(DEST).free > 12*1024**3
                        job['bytes_received'] += len(block)
                        assert job['bytes_received'] <= limit
                        handle.write(block)
                        if time.monotonic()-last > 5:
                            job['updated_at'] = time.time(); save(); last=time.monotonic()
                response.close()
                assert job['bytes_received'] > 0
                assert not length or job['bytes_received'] == length
                partial.replace(output)
            job.update(status='downloaded', sha256=sha(output), finished_at=time.time())
            if name == 'phmr_config':
                assert "BACKBONE: 'dinov2_vitl14'" in output.read_text()
            if name == 'phmr_checkpoint':
                # PyTorch state container integrity; no pickle execution here.
                with zipfile.ZipFile(output) as archive:
                    assert any(x.endswith('/data.pkl') for x in archive.namelist())
                    assert archive.testzip() is None
            if name == 'waft_a2':
                member = 'waftv2-ckpts/dinov3/spring.pth'
                with zipfile.ZipFile(output) as archive:
                    info = archive.getinfo(member)
                    assert info.file_size < 1024**3
                    weight = DEST/'waft_dinov3_spring.pth'
                    assert not weight.exists()
                    with archive.open(info) as src, weight.open('xb') as dst:
                        shutil.copyfileobj(src,dst,8*1024*1024)
                job['selected_member'] = dict(name=member,path=str(weight),bytes=weight.stat().st_size,sha256=sha(weight))
            job['status']='downloaded_container_validated_runtime_pending'
        except Exception as exc:
            job.update(status='failed', reason=repr(exc), finished_at=time.time())
        save()
    state.update(status='finished' if all(j['status']=='downloaded_container_validated_runtime_pending' for j in state['jobs']) else 'finished_with_failures',finished_at=time.time())
    save()


if __name__ == '__main__':
    main()
