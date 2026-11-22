"""Finite incremental static/ED-P queue for the frozen human export re-audit.

Every GPU stage uses the existing shared lease wrapper. No export, reference,
metric definition, or published table is modified by this orchestrator.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import native_reaudit_common as common


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=common.METHODS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    ids = set(common.freeze()['ids'])
    root = common.ROOT / 'metrics' / a.method
    root.mkdir(parents=True, exist_ok=True)
    state = root / 'pipeline.json'
    script = common.H / 'scripts/score_native_reaudit_static_v1.py'
    wrapper = common.H / 'scripts/native_reaudit_gpu.py'
    started = time.time()

    def update(status, **kw):
        common.save(state, dict(method=a.method, pid=os.getpid(), gpu=a.gpu,
                               started_at=started, updated_at=time.time(), status=status, **kw))

    def run(stage, env=None, selected=None):
        cmd = [sys.executable, '-B', str(script), stage, '--method', a.method]
        if stage == 'score':
            cmd += ['--device', 'cuda', '--ids', *sorted(selected)]
        if env:
            key = f'{a.method}_pipeline_{stage}_{time.time_ns()}'
            cmd = [sys.executable, '-B', str(wrapper), '--gpu', str(a.gpu),
                   '--env', env, '--key', key, *cmd[2:]]
        update(stage, command=cmd, batch_count=len(selected) if selected else None)
        subprocess.run(cmd, check=True)

    try:
        while True:
            found = common.validations(a.method)
            terminal = {sid for sid, (_, v) in found.items()
                        if v.get('status') in {'passed', 'unavailable', 'confirmed_unavailable'}}
            scored = {x.stem for x in (root / 'static_rows').glob('*.json')}
            pending = terminal - scored
            if pending:
                run('score', 'eval', pending)
            if terminal == ids:
                break
            update('waiting_for_exports', exports_terminal=len(terminal),
                   static_rows=len(list((root / 'static_rows').glob('*.json'))),
                   remaining_ids=sorted(ids - terminal))
            time.sleep(15)
        run('edp-prepare')
        run('edp-pose', 'pose')
        run('edp-features', 'trellis2')
        run('edp-score')
        common.inventory()
        update('completed', static_rows=len(list((root / 'static_rows').glob('*.json'))),
               summary=str(root / 'static_summary.json'))
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
