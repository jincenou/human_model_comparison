"""Download the reviewed public-weight plan; never install or mutate shared caches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'state/modelscope_weights.json'
RESERVE = 11 * 1024**3  # Additional margin above the experiment's 10 GiB floor.


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(state):
    state['updated_at_unix'] = time.time()
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def main():
    with (ROOT / 'state/modelscope_weights.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = ROOT / 'reports/modelscope_weight_plan.json'
        plan = json.loads(plan_path.read_text())
        state = dict(pid=os.getpid(), started_at_unix=time.time(), status='running',
                     plan_sha256=digest(plan_path), disk_floor_gib=11, jobs=[])
        save(state)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for rec in plan['downloads']:
            job = dict(rec, status='checking')
            state['jobs'].append(job)
            dest = ROOT / 'weights' / rec['filename']
            dest.parent.mkdir(exist_ok=True)
            temp = dest.with_suffix(dest.suffix + '.part')
            expected = rec['sha256_or_prefix']
            try:
                if dest.exists():
                    checksum = digest(dest)
                    if dest.stat().st_size != rec['bytes'] or not checksum.startswith(expected):
                        raise RuntimeError('Existing destination failed validation; preserved for audit')
                    job.update(status='verified_reused', sha256=checksum, path=str(dest))
                    save(state)
                    continue
                if shutil.disk_usage(ROOT).free < RESERVE + rec['bytes']:
                    raise RuntimeError('Insufficient space above disk reserve')
                error = None
                for attempt in (1, 2):
                    job.update(status='downloading', attempt=attempt, bytes_downloaded=0)
                    save(state)
                    try:
                        request = urllib.request.Request(rec['url'], headers={'User-Agent': 'human-model-preparation'})
                        with opener.open(request, timeout=30) as response, temp.open('wb') as stream:
                            last_save = time.monotonic()
                            while chunk := response.read(1024**2):
                                if shutil.disk_usage(ROOT).free < RESERVE + len(chunk):
                                    raise RuntimeError('Disk reserve reached during transfer')
                                stream.write(chunk)
                                job['bytes_downloaded'] += len(chunk)
                                if job['bytes_downloaded'] > rec['bytes']:
                                    raise RuntimeError('Response exceeds expected weight size')
                                if time.monotonic() - last_save > 5:
                                    save(state)
                                    last_save = time.monotonic()
                        checksum = digest(temp)
                        if temp.stat().st_size != rec['bytes'] or not checksum.startswith(expected):
                            raise RuntimeError('Downloaded bytes failed expected size/hash validation')
                        temp.rename(dest)
                        job.update(status='verified_downloaded', sha256=checksum, path=str(dest))
                        error = None
                        break
                    except Exception as exc:
                        error = exc
                        job['last_error'] = repr(exc)
                        save(state)
                        if isinstance(exc, RuntimeError):
                            break
                if error:
                    raise error
            except Exception as exc:
                job.update(status='failed', reason=repr(exc))
            save(state)
            print(json.dumps(job), flush=True)
        state['status'] = 'verified_all' if all(j['status'].startswith('verified_') for j in state['jobs']) else 'finished_with_failures'
        state['finished_at_unix'] = time.time()
        save(state)


if __name__ == '__main__':
    main()
