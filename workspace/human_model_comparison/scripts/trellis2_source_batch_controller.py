"""Incremental TRELLIS.2 source registration, native assets only.

Two auxiliary GPU workers at most, four identities per group. Newly completed
native generation is discovered until the native controller/workers exit. A
failed source group is recorded rather than automatically retried. Existing
rendered or explicitly unavailable identities are never sent to workers again.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import trellis2_source_adapter as adapter

H = adapter.H
ROOT = adapter.ROOT
STATE = H/'state/trellis2_source_batch.json'
LOCK = H/'state/trellis2_source_batch.lock'
NATIVE_STATE = H/'runs/human_native_queue_v1/state.json'
SCRIPT = Path(__file__).resolve()
MAX_WORKERS = 2
GROUP_SIZE = 4


def load_optional(path):
    return adapter.read(path) if path.exists() else None


def process(pid, start_ticks=None, marker=None):
    """Read an actual process handle, rejecting zombies and recycled PIDs."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        proc = Path('/proc')/str(pid)
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])
        command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if fields[0] == 'Z' or (start_ticks is not None and ticks != start_ticks):
            return None
        if marker is not None and marker not in command:
            return None
        return dict(pid=pid, start_ticks=ticks, command=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def native_handles(queue, selected_ids):
    """A live controller or an existing TRELLIS.2 worker can produce more assets."""
    selected = set(selected_ids)
    outstanding = [j for j in queue.get('jobs', []) if j.get('method') == 'trellis2'
                   and j.get('sample_id') in selected and j.get('status') in ('queued', 'running')]
    candidates = [queue.get('pid')] if outstanding else []
    candidates += [j.get('child_pid') for j in outstanding if j.get('status') == 'running']
    return [p for pid in candidates
            if (p := process(pid, marker='human_native_queue_v3.py')) is not None]


def scoring_outcome(returncode, output):
    """Scorer reports per-pair failures in JSON even when its process exits 0."""
    busy = returncode != 0 and ('BlockingIOError' in output or 'Resource temporarily unavailable' in output)
    report = None
    for line in reversed(output.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and 'coverage' in candidate and 'failures' in candidate:
            report = candidate
            break
    result = dict(exit_code=returncode, busy=busy,
                  issues=not busy and (returncode != 0 or report is None or bool(report['failures'])))
    if report is not None:
        result.update(coverage=report['coverage'], failures=report['failures'])
    elif not busy:
        result['reason'] = 'Scorer emitted no recognized result JSON: '+output[-3000:]
    if returncode and not busy:
        result['reason'] = output[-3000:]
    return result


def terminal(sid, root=ROOT):
    validation = root/'renders'/sid/'validation.json'
    if validation.exists():
        return 'rendered'
    status = load_optional(root/'observations'/sid/'registration_status.json')
    if status and status.get('status') == 'unavailable':
        return 'registration_unavailable'
    return None


def native_ready(sid, home=H):
    folder = home/'runs/trellis2_human_native_v1'/sid
    state = load_optional(folder/'state.json')
    validation = load_optional(folder/'reload_validation.json')
    return bool(state and validation and state.get('sample_id') == sid
                and state.get('stage') == 'finished'
                and validation.get('status') == 'passed_pending_process_exit'
                and (folder/'native_voxel_mesh.pt').exists())


def candidates(ids, jobs, home=H, root=ROOT):
    attempted = {sid for job in jobs for sid in job['ids']}
    return [sid for sid in ids if sid not in attempted
            and terminal(sid, root) is None and native_ready(sid, home)]


def run():
    STATE.parent.mkdir(exist_ok=True)
    with LOCK.open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, selected = adapter.shared.selection()
        ids = list(selected)
        prior = load_optional(STATE)
        if prior:
            assert prior['selection_sha256'] == active['selection_sha256'], 'Cohort changed'
            assert not process(prior.get('pid'), prior.get('start_ticks'), str(SCRIPT)), 'Prior controller still alive'
        jobs = prior.get('jobs', []) if prior else []
        state = dict(status='running', pid=os.getpid(),
                     start_ticks=process(os.getpid())['start_ticks'],
                     selection_sha256=active['selection_sha256'], jobs=jobs,
                     script_sha256=adapter.sha(SCRIPT), adapter_sha256=adapter.sha(adapter.__file__),
                     gpu_pool=[4, 5, 6, 7], max_workers=MAX_WORKERS,
                     group_size=GROUP_SIZE, disk_floor_gib=10,
                     started_at=time.time(), prior_controller_pid=prior.get('pid') if prior else None)
        children = {}
        pending_score = True
        score_retry_after = 0.

        def persist():
            state['updated_at'] = time.time()
            adapter.save(STATE, state)

        persist()
        while True:
            # Actual process handles, including surviving workers from a previous
            # controller, determine whether a running job has stopped.
            for job in jobs:
                if job['status'] != 'running':
                    continue
                handle = children.get(job['group'])
                if handle:
                    child, lease, stream = handle
                    code = child.poll()
                    if code is None:
                        continue
                    stream.close(); lease.close(); del children[job['group']]
                else:
                    if process(job.get('pid'), job.get('start_ticks'), 'trellis2_source_adapter.py worker'):
                        continue
                    code = None
                outcomes = {sid: terminal(sid) for sid in job['ids']}
                complete = all(outcomes.values())
                job.update(status='completed' if complete else 'failed', exit_code=code,
                           outcomes=outcomes, finished_at=time.time())
                if not complete:
                    job['reason'] = 'worker_exited_with_incomplete_source_outputs; no automatic retry'
                pending_score = True
                persist()

            # Publishing and scoring are serialized by the shared scorer lock in
            # the adapter/entrypoint. Lock contention is a wait, not a job failure.
            if pending_score and time.time() >= score_retry_after:
                try:
                    adapter.publish_index()
                    result = subprocess.run(
                        [adapter.shared.PY_CONTROL, '-B', str(H/'scripts/run_human_metric_pipeline_v1.py'),
                         '--active-cohort', '--methods', 'trellis2'], cwd=H,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
                    with (H/'logs/trellis2_source_batch_scoring.log').open('a') as stream:
                        stream.write(result.stdout)
                    state['last_scoring'] = dict(scoring_outcome(result.returncode, result.stdout), at=time.time())
                    pending_score = state['last_scoring']['busy']
                    score_retry_after = time.time()+10
                except BlockingIOError:
                    state['last_scoring'] = dict(at=time.time(), busy=True)
                    score_retry_after = time.time()+10
                persist()

            adapter.checked(active['selection'], active['selection_sha256'])
            ready = candidates(ids, jobs)
            running = [job for job in jobs if job['status'] == 'running']
            state['ready_unstarted'] = ready
            state['disk_free_gib'] = shutil.disk_usage(H).free/(1024**3)
            if ready and len(running) < MAX_WORKERS and state['disk_free_gib'] > 10:
                slot = adapter.shared.acquire()
                if slot:
                    lease, gpu, detail = slot
                    try:
                        adapter.checked(SCRIPT, state['script_sha256'])
                        adapter.checked(adapter.__file__, state['adapter_sha256'])
                        group = max([j['group'] for j in jobs], default=-1)+1
                        batch = ready[:GROUP_SIZE]
                        log = H/'logs'/f'trellis2_source_batch_group{group:03d}.log'
                        stream = log.open('a')
                        child = subprocess.Popen(
                            [adapter.shared.PY_CONTROL, '-B', '-u', str(Path(adapter.__file__).resolve()),
                             'worker', '--gpu', detail['uuid'], '--ids', *batch], cwd=H,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                            pass_fds=(lease.fileno(),), start_new_session=True)
                        identity = process(child.pid)
                        job = dict(group=group, ids=batch, status='running', pid=child.pid,
                                   start_ticks=identity['start_ticks'] if identity else None,
                                   gpu=gpu, gpu_uuid=detail['uuid'], log=str(log), started_at=time.time())
                        jobs.append(job); children[group] = child, lease, stream
                        persist()
                    except BaseException:
                        lease.close()
                        raise

            queue = load_optional(NATIVE_STATE) or {}
            native = native_handles(queue, ids)
            state['native_live_handles'] = native
            running = [j for j in jobs if j['status'] == 'running']
            if not running and not candidates(ids, jobs) and not native and not pending_score:
                issues = any(j['status'] == 'failed' for j in jobs) or state.get('last_scoring', {}).get('issues', False)
                state.update(status='finished_with_issues' if issues else 'finished', finished_at=time.time())
                persist()
                return
            state['waiting_for'] = ('source_workers' if running else 'gpu_or_disk_slot' if ready
                                    else 'scorer' if pending_score else 'native_generation')
            persist()
            time.sleep(5)


def self_test():
    """Isolated scheduling checks: no GPU acquisition, controller or scoring."""
    import tempfile
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory); root = home/'reports'
        ids = ['ready', 'unfinished', 'unavailable', 'rendered', 'failed_before']
        for sid in ids:
            folder = home/'runs/trellis2_human_native_v1'/sid
            adapter.save(folder/'state.json', dict(sample_id=sid, stage='running' if sid == 'unfinished' else 'finished'))
            adapter.save(folder/'reload_validation.json', dict(status='passed_pending_process_exit'))
            (folder/'native_voxel_mesh.pt').touch()
        adapter.save(root/'observations/unavailable/registration_status.json', dict(status='unavailable'))
        adapter.save(root/'renders/rendered/validation.json', dict(status='rendered'))
        jobs = [dict(ids=['failed_before'], status='failed')]
        assert candidates(ids, jobs, home, root) == ['ready']
        assert candidates(ids, jobs+[dict(ids=['ready'], status='running')], home, root) == []
        assert terminal('unavailable', root) == 'registration_unavailable'
        assert terminal('rendered', root) == 'rendered'
        me = process(os.getpid()); assert me and process(os.getpid(), me['start_ticks'])
        assert process(os.getpid(), me['start_ticks']+1) is None
        assert process(os.getpid(), marker='this-command-does-not-exist') is None
        assert not native_handles(dict(pid=0, jobs=[]), ['ready'])
        with patch(__name__+'.process', side_effect=lambda pid, **kw: dict(pid=pid)):
            queue = dict(pid=10, jobs=[dict(method='ours', sample_id='ready', status='queued'),
                                      dict(method='trellis2', sample_id='removed', status='queued')])
            assert not native_handles(queue, ['ready'])
            queue['jobs'].append(dict(method='trellis2', sample_id='ready', status='queued'))
            assert native_handles(queue, ['ready']) == [dict(pid=10)]
            queue['jobs'][-1].update(status='running', child_pid=11)
            assert native_handles(queue, ['ready']) == [dict(pid=10), dict(pid=11)]
        good = json.dumps(dict(coverage={'trellis2': 1}, failures=[]))
        failed = json.dumps(dict(coverage={'trellis2': 1}, failures=[dict(id='ready', reason='bad camera')]))
        assert not scoring_outcome(0, good)['issues']
        assert scoring_outcome(0, failed)['issues'] and not scoring_outcome(0, failed)['busy']
        assert scoring_outcome(1, 'BlockingIOError')['busy']
        assert scoring_outcome(0, 'unexpected output')['issues']
    print('PASS: scheduling, PID reuse, selected TRELLIS.2 liveness, exit-0 scoring failures and busy retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.dry_run:
        active, selected = adapter.shared.selection()
        state = load_optional(STATE) or {}
        print(json.dumps(dict(ready=candidates(list(selected), state.get('jobs', [])),
                              native_live_handles=native_handles(load_optional(NATIVE_STATE) or {}, selected),
                              limits=dict(gpu_pool=[4, 5, 6, 7], max_workers=MAX_WORKERS, group_size=GROUP_SIZE)), indent=2))
    else:
        run()
