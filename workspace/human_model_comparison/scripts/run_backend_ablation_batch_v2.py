"""Sequential per-GPU execution of the fixed, input-selected ablation cohort.

Existing restoration processes are observed, never duplicated. Each subsequent
stage uses the established GPU lease/resource checks and stops on failure.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from common_human_orbit import H, read, save, sha, checked


def wait_for_restoration(sid, state, statepath):
    controllers = list((H/'runs/backend_ablation_cache_restore'/sid).glob('*/controller.json'))
    if not controllers:
        return False
    for path in controllers:
        while True:
            record = read(path)
            if record['status'] == 'finished':
                assert record['exit_code'] == 0
                restored = read(path.with_name('recovery.json'))
                assert restored['status'] == 'exact_original_rgba_restored'
                break
            if record['status'] != 'running':
                raise RuntimeError(f'{sid}: restoration terminal status {record["status"]}')
            # A missing handle is a failure to inspect, never authorization to retry.
            os.kill(record['pid'], 0)
            if 'worker_pid' in record:
                os.kill(record['worker_pid'], 0)
            state.update(status='waiting_on_live_restoration', current_id=sid,
                         observed_controller=str(path), observed_pid=record['pid'],
                         last_observed_live_at=time.time())
            save(statepath, state)
            time.sleep(15)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-ids', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    planpath = H/'configs/backend_ablation_batch_v2.json'; plan = read(planpath)
    checked(plan['selection'], plan['selection_sha256'])
    validation = read(H/'reports/backend_ablation_sampling_validation_v2.json')
    assert validation['status'] == 'passed'
    for p, digest in validation['code_hashes'].items():
        checked(p, digest)
    assert set(args.human_ids) <= set(plan['pending_ids'])
    assert len(args.human_ids) == len(set(args.human_ids))
    policy = read(H/'state/gpu_resource_policy.json')
    assert args.gpu in policy['allowed_gpu_indices'] and args.gpu not in policy['excluded_gpu_indices']
    path = H/'state'/f'backend_batch_v2_gpu{args.gpu}.json'
    # Claim without overwriting historical or live controllers.
    with path.open('x') as stream:
        stream.write('{}\n')
    state = dict(status='starting', pid=os.getpid(), gpu=args.gpu, ids=args.human_ids,
                 started_at=time.time(), plan_sha256=sha(planpath), script_sha256=sha(__file__),
                 stages=[], completed_ids=[])
    save(path, state)

    def run(sid, stage, script, arguments):
        checked(planpath, state['plan_sha256'])
        policy = read(H/'state/gpu_resource_policy.json')
        assert args.gpu in policy['allowed_gpu_indices'] and args.gpu not in policy['excluded_gpu_indices']
        command = [sys.executable, '-B', '-u', str(H/'scripts'/script), '--human-id', sid, *arguments]
        entry = dict(id=sid, stage=stage, status='running', command=command,
                     script_sha256=sha(H/'scripts'/script), started_at=time.time())
        state.update(status='running', current_id=sid, current_stage=stage)
        state['stages'].append(entry)
        with (H/'logs'/f'backend_batch_v2_{sid}_{stage}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=H, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(identity_lease.fileno(),))
        entry['pid'] = child.pid; save(path, state)
        code = child.wait()
        entry.update(exit_code=code, status='finished' if code == 0 else 'failed_retained', finished_at=time.time())
        save(path, state)
        if code:
            raise RuntimeError(f'{sid}: {stage} exited {code}; inspect retained output, no automatic retry')

    try:
        for sid in args.human_ids:
            with (H/'state'/f'backend_ablation_{sid}.lock').open('a') as identity_lease:
                fcntl.flock(identity_lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
                assert not (H/'runs/backend_ablation_v1'/sid).exists(), 'Existing training requires explicit inspection'
                if not wait_for_restoration(sid, state, path):
                    run(sid, 'restore', 'restore_frontend_rgba_cache.py',
                        ['--gpu', str(args.gpu), '--views', *map(str, range(24))])
                run(sid, 'prepare', 'prepare_backend_ablation.py', [])
                run(sid, 'train', 'launch_backend_ablation.py',
                    ['--gpu', str(args.gpu), '--stages', 'native', 'alpha', 'A'])
                run(sid, 'control_prepare', 'launch_backend_budget_control.py', ['--prepare'])
                run(sid, 'control_train', 'launch_backend_budget_control.py', ['--gpu', str(args.gpu)])
                run(sid, 'evaluate', 'evaluate_backend_ablation.py', ['--gpu', str(args.gpu)])
                run(sid, 'tables', 'export_backend_ablation_tables.py', [])
                state['completed_ids'].append(sid); save(path, state)
        state.update(status='scored_pending_visual_review', finished_at=time.time())
        save(path, state)
    except BaseException as exc:
        state.update(status='failed_retained', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        save(path, state)
        raise


if __name__ == '__main__':
    main()
