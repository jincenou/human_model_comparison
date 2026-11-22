"""Finite postprocessing scoring queue; does not regenerate 3D or geometry."""
import argparse
import os
import subprocess
import sys
import time

import native_reaudit_common as common


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=common.METHODS, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    root = common.ROOT / 'postprocess_v1'
    out = root / 'metrics' / a.method
    script = common.H / 'scripts/score_native_human_postprocess_v1.py'
    wrapper = common.H / 'scripts/native_reaudit_gpu.py'
    started = time.time()
    ids = set(common.freeze()['ids'])

    def state(status, **kw):
        common.save(out / 'pipeline.json', dict(status=status, method=a.method,
                    pid=os.getpid(), gpu=a.gpu, started_at=started, updated_at=time.time(), **kw))

    try:
        state('waiting_postprocess')
        while {p.parent.name for p in (root / 'outputs' / a.method).glob('*/postprocess.json')} != ids:
            assert time.time() - started < 1800, 'Postprocessing did not finish within 30 minutes'
            time.sleep(5)
        for action, env in [('score', 'eval'), ('edp-prepare', None),
                            ('edp-pose', 'pose'), ('edp-features', 'trellis2'), ('edp-score', None)]:
            cmd = [str(script), action, '--method', a.method]
            if env:
                cmd = [str(wrapper), '--gpu', str(a.gpu), '--env', env, '--key',
                       f'postprocess_{a.method}_{action}_{time.time_ns()}', *cmd]
            cmd = [sys.executable, '-B', '-u', *cmd]
            state(action, command=cmd)
            subprocess.run(cmd, check=True)
        rows = [common.read(p) for p in (out / 'static_rows').glob('*.json')]
        edp = common.read(out / 'edp/all/results.json')['rows']
        assert {r['id'] for r in rows} == ids and all(r['status'] == 'scored' for r in rows)
        assert {r['id'] for r in edp} == ids
        state('completed', static_records=len(rows), edp_processed=len(edp))
    except Exception as exc:
        state('failed', reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
