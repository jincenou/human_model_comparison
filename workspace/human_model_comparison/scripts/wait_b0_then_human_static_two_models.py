"""Wait for pinned B0 process handles, then take the ordinary human GPU lease.

This supervisor never signals any generation process. The unchanged human
worker owns all three established GPU locks and the 10 GiB disk safeguard.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

H = Path(__file__).resolve().parents[1]
Q = H / 'runs/human_static_two_models_20261115_v1'
B = H.parent / 'vehicle_model_comparison/experiments/ours_b0_vehicle432_20261115_v1'
PYTHON = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python')
GENERATOR = H / 'scripts/run_human_static_two_models_20261115.py'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def process(pid):
    """Host /proc only: PID reuse and zombies are not live dependencies."""
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError):
        return None


def still_same(handle):
    now = process(handle['pid'])
    return now is not None and now['start_ticks'] == handle['start_ticks']


def wait_and_run(plan):
    gpu, method = plan['gpu'], plan['method']
    directory = Q / 'continuations'
    directory.mkdir(exist_ok=True)
    with (directory / f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / f'gpu{gpu}.json'
        state = dict(plan=plan, supervisor=process(os.getpid()), started_at=time.time())
        while True:
            live = [h for h in plan['dependencies'] if still_same(h)]
            launch = read(B / 'launchers' / f'gpu{gpu}.json')
            # Also observe a bounded restart launched by the original B0
            # supervisor, rather than trusting only the initial child PID.
            for key in ['pid', 'child_pid']:
                candidate = process(launch[key]) if launch.get(key) else None
                if candidate and 'run_ours_b0_vehicle432.py' in candidate['command']:
                    live.append(candidate)
            queue = read(B / 'queue.json')
            running = [j['id'] for j in queue['jobs']
                       if j['status'] == 'running' and j.get('gpu') == gpu]
            terminal = launch.get('status') in ['completed', 'worker_error']
            state.update(status='waiting_for_b0_process_exit', dependencies_live=live,
                         b0_running_ids=running, b0_launcher=launch, updated_at=time.time())
            save(state_path, state)
            if not live and terminal and not running:
                break
            time.sleep(15)
        pending = [j for j in read(Q / 'queue.json')['jobs']
                   if j['method'] == method and j['status'] == 'pending']
        if not pending:
            state.update(status='no_remaining_human_jobs', finished_at=time.time())
            save(state_path, state)
            return
        command = [PYTHON, '-B', '-u', str(GENERATOR), 'worker',
                   '--method', method, '--gpu', str(gpu)]
        with (Q / 'logs' / f'worker_{method}_gpu{gpu}.log').open('a') as log:
            child = subprocess.Popen(command, cwd=H, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     pass_fds=(lock.fileno(),))
            handle = None
            for _ in range(10):
                handle = process(child.pid)
                if handle:
                    break
                time.sleep(.05)
            state.update(status='human_worker_running', child=handle,
                         command=command, handed_over_at=time.time(),
                         gpu_lease='unchanged human worker takes all three GPU locks',
                         disk_guard_gib=10)
            save(state_path, state)
            code = child.wait()
        counts = collections.Counter(j['status'] for j in read(Q / 'queue.json')['jobs']
                                     if j['method'] == method)
        state.update(status='completed' if code == 0 else 'worker_error',
                     exit_code=code, method_counts=dict(counts), finished_at=time.time())
        save(state_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    wait_and_run(read(args.plan))
