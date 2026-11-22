"""Run the finite accepted static and ED-P stages on resolved native exports."""
import argparse
import os
import subprocess
import sys
import time

import native_reaudit_common as common


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['omnire', 'splatad'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    root = common.ROOT / 'resolved' / 'metrics' / a.method
    script = common.H / 'scripts/native_reaudit_resolved_v1.py'
    wrapper = common.H / 'scripts/native_reaudit_gpu.py'
    started = time.time()
    def state(status, **kw):
        common.save(root / 'pipeline.json', dict(status=status, method=a.method,
                    pid=os.getpid(), gpu=a.gpu, started_at=started, updated_at=time.time(), **kw))
    try:
        for action, env in [('score', 'eval'), ('edp-prepare', None),
                            ('edp-pose', 'pose'), ('edp-features', 'trellis2'), ('edp-score', None)]:
            cmd = [str(script), action, '--method', a.method]
            if env:
                cmd = [str(wrapper), '--gpu', str(a.gpu), '--env', env, '--key',
                       f'resolved_{a.method}_{action}_{time.time_ns()}', *cmd]
            cmd = [sys.executable, '-B', '-u', *cmd]
            state(action, command=cmd)
            subprocess.run(cmd, check=True)
        summary_path = root / 'static_summary.json'
        summary = common.read(summary_path)
        scored = [common.read(p) for p in (root / 'static_rows').glob('*.json')]
        edp = common.read(root / 'edp/all/results.json')['rows']
        expected = set(common.freeze()['ids'])
        assert {r['id'] for r in scored} == expected
        assert {r['id'] for r in edp} == {r['id'] for r in scored if r['status'] == 'scored'}
        summary.update(expected=100, unvisited=[], edp_processed=len(edp),
                       resolution=str(common.ROOT / 'resolved/decisions' / (a.method + '.json')))
        common.save(summary_path, summary)
        state('completed', static_records=len(scored), edp_processed=len(edp))
    except Exception as exc:
        state('failed', reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
