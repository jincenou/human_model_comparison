"""Run upstream Custom training unchanged; 30 real frames repeated 15x per epoch."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import os, sys, json, time, subprocess, shutil, fcntl, hashlib

P = Path(__file__).resolve().parents[1]
SUBJECT = 'HN0057_native_pilot_20261113'
ROOT = P / 'reports/exavatar_real_sequence_HN0057_20261113'
STATE = P / 'state/exavatar_avatar_HN0057_20261113.json'

if '--worker' in sys.argv:
    import runpy
    main = P / 'sources/exavatar/avatar/main'
    os.chdir(main)
    sys.path.insert(0, str(main))
    from config import cfg
    cfg.dataset = 'Custom'
    sys.path.insert(0, str(Path(cfg.data_dir) / 'Custom'))
    assert cfg.end_epoch == 5 and cfg.batch_size == 1
    sys.argv = ['train.py', '--subject_id', SUBJECT]
    runpy.run_path('train.py', run_name='__main__')
else:
    def save(s):
        tmp = STATE.with_suffix('.tmp')
        tmp.write_text(json.dumps(s, indent=2))
        tmp.replace(STATE)
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not STATE.exists(), 'Inspect existing run before restarting'
        assert shutil.disk_usage(P).free > 10 * 1024**3
        checks = json.loads((ROOT / 'avatar_input_checks.json').read_text())
        assert all(checks[k] == 30 for k in ['frames','masks','depthmaps','smplx_optimized/smplx_params_smoothed'])
        assert checks['background']['finite'] and checks['face_texture_mask.png']['nonzero'] > 0
        command = [_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'), '-B', '-u', str(Path(__file__).resolve()), '--worker']
        s = dict(status='training', pid=os.getpid(), gpu=6, command=command, started_at=time.time(),
                 subject=SUBJECT, real_frames=30, epochs=5, upstream_dataset_repeat=15,
                 expected_updates=2250, camera='upstream virtual camera; common-camera registration pending',
                 scope='Small real-sequence pilot, training-view reconstruction, no heldout claim',
                 protocol_sha256=hashlib.sha256((ROOT/'pilot_protocol.json').read_bytes()).hexdigest())
        save(s)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', EGL_DEVICE_ID='6', PYOPENGL_PLATFORM='egl',
                   PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
                   TORCH_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/torch_aux'), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        try:
            with (P/'logs/exavatar_avatar_HN0057_worker_20261113.log').open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                s['child_pid'] = child.pid
                save(s)
                code = child.wait()
                s['exit_code'] = code
            if code != 0:
                raise RuntimeError('Inspect training log before retry')
            s['status'] = 'trained_reload_pending'
        except Exception as e:
            s.update(status='failed', reason=repr(e))
            raise
        finally:
            s['finished_at'] = time.time()
            save(s)
