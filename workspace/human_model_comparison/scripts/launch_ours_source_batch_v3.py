"""Render existing Ours snapshots on spare GPU capacity; preserve failed cases."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from launch_replacement_HN0646_v4 import acquire_available_slot
from launch_ours_frozen_time_v2 import environment, PYTHON
from export_ours_frozen_time_v1 import sha

H = Path(__file__).resolve().parents[1]
OUT = H / 'reports/ours_source_batch_v3'
STATE = H / 'state/ours_source_batch_v3.json'
PLAN = H / 'configs/human_source_common_v1/ours_registration_batch_v3/manifest.json'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def worker(sid):
    from certify_ours_source_batch_v3 import certify
    plan = read(PLAN)
    active = read(H / 'inputs/active_human_benchmark.json')
    assert sha(active['selection']) == active['selection_sha256'] == plan['selection_sha256']
    selected = {r['id']: r for r in read(active['selection'])['objects']}
    row = next(r for r in plan['rows'] if r['id'] == sid)
    assert sha(row['registration']) == row['registration_sha256']
    reg = read(row['registration'])
    for p, digest in reg['provenance_sha256'].items():
        assert sha(p) == digest, p
    assert shutil.disk_usage(H).free > 10 * 1024**3
    cmd = [str(PYTHON), '-B', '-u', str(H / 'scripts/render_ours_source_batch_v3.py'), '--human-id', sid]
    subprocess.run(cmd, cwd=H, check=True)
    validation = H / 'exports/ours_source_camera_batch_v3' / sid / 'validation.json'
    certificate = certify(sid, selected, active, validation)
    dest = OUT / sid / 'ours.json'
    dest.parent.mkdir(exist_ok=False)
    save(dest, certificate)
    print(json.dumps(dict(id=sid, status='coordinate_trace_certified', certificate=str(dest))), flush=True)


def main():
    with (H / 'state/ours_source_batch_v3.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists(), 'Existing batch requires process audit; no automatic restart'
        plan = read(PLAN)
        active = read(H / 'inputs/active_human_benchmark.json')
        assert sha(active['selection']) == active['selection_sha256'] == plan['selection_sha256']
        old = read(H / 'reports/ours_source_certification_v3/manifest.json')
        assert old['selection_sha256'] == active['selection_sha256']
        reused = {r['id']: r for r in old['rows']}
        for r in reused.values():
            assert sha(r['registration_record']) == r['registration_sha256']
        paths = [Path(__file__), PLAN, Path(active['selection']), H/'scripts/render_ours_source_batch_v3.py',
                 H/'scripts/certify_ours_source_batch_v3.py', H/'scripts/launch_replacement_HN0646_v4.py',
                 H/'scripts/score_ours_source_batch_v3.py']
        pins = {str(p): sha(p) for p in paths}
        jobs = []
        for r in plan['rows']:
            status = 'reused_coordinate_certificate' if r['id'] in reused else 'queued' if r['status'] == 'trace_prepared' else 'trace_unavailable'
            job = dict(id=r['id'], status=status)
            if 'reason' in r:
                job['reason'] = r['reason']
            if r['id'] in reused:
                job.update(reused[r['id']]); job['status'] = status
            jobs.append(job)
        state = dict(status='running', pid=os.getpid(), started_at=time.time(), jobs=jobs, pinned_files=pins,
                     selection_sha256=active['selection_sha256'], max_concurrent_workers=2,
                     disk_floor_gib=10, launch_floor_gib=11, gpu_pool=list(range(8)),
                     resource_policy='Existing primary leases or explicit co-sharing auxiliary leases; >=32 GiB free for auxiliary starts',
                     scope='Existing frozen snapshots only; estimated source-pose trace; no generation, pose fitting or quality-based exclusion')
        running = {}
        def persist():
            state['updated_at'] = time.time()
            state['counts'] = {s: sum(j['status'] == s for j in jobs) for s in sorted({j['status'] for j in jobs})}
            save(STATE, state)
        persist()
        while any(j['status'] in ('queued', 'running') for j in jobs):
            for sid, (child, lease, stream, job) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                stream.close(); lease.close(); del running[sid]
                job.update(exit_code=code, finished_at=time.time())
                cert = OUT / sid / 'ours.json'
                if code == 0 and cert.exists():
                    job.update(status='coordinate_trace_certified_reference_gate_separate', registration_record=str(cert), registration_sha256=sha(cert))
                else:
                    job.update(status='render_or_certificate_failed', reason='See retained per-person log; no automatic pose/scale adjustment or retry')
                persist()
            queued = next((j for j in jobs if j['status'] == 'queued'), None)
            if queued and len(running) < 2 and shutil.disk_usage(H).free > 11 * 1024**3:
                slot, observation = acquire_available_slot()
                if slot:
                    lease, gpu, detail = slot
                    for p, digest in pins.items():
                        assert sha(p) == digest, p
                    sid = queued['id']; log = H / 'logs' / ('ours_source_batch_v3_' + sid + '.log')
                    stream = log.open('x')
                    env = environment(detail['uuid']); env.update(MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                    child = subprocess.Popen([str(PYTHON), '-B', '-u', str(Path(__file__).resolve()), '--worker', sid],
                                             cwd=H, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                             pass_fds=(lease.fileno(),), start_new_session=True)
                    queued.update(status='running', pid=child.pid, gpu=gpu, gpu_uuid=detail['uuid'],
                                  resource_snapshot=detail, started_at=time.time(), log=str(log))
                    running[sid] = child, lease, stream, queued
                    persist()
            if running or any(j['status'] == 'queued' for j in jobs):
                time.sleep(2)
        rows = [{k: j[k] for k in ('id','status','registration_record','registration_sha256')} for j in jobs if 'registration_record' in j]
        manifest = dict(status='batch_coordinate_traces_complete_failures_retained', selection=active['selection'],
                        selection_sha256=active['selection_sha256'], rows=rows,
                        failures=[j for j in jobs if 'registration_record' not in j],
                        reference_gate_separate=True, controller_sha256=sha(__file__))
        save(OUT / 'manifest.json', manifest)
        state.update(status='scoring_reference_eligible_pairs'); persist()
        with (H / 'logs/ours_source_batch_v3_scoring.log').open('x') as stream:
            child = subprocess.Popen([str(PYTHON), '-B', '-u', str(H/'scripts/score_ours_source_batch_v3.py')],
                                     cwd=H, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state['scoring_pid'] = child.pid; persist()
            code = child.wait()
        state.update(status='finished' if code == 0 else 'rendering_finished_scoring_failed', scoring_exit_code=code, finished_at=time.time()); persist()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--worker'); args = ap.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main()
