"""One native-budget ExAvatar fitting pilot; no automatic retries."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import os,json,time,subprocess,shutil,fcntl,hashlib
P=Path(__file__).resolve().parents[1];SUBJECT='HN0057_native_pilot_20261113';STATE=P/'state/exavatar_fit_HN0057_20261113.json';ROOT=P/'reports/exavatar_real_sequence_HN0057_20261113'
def save(s):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(STATE)
with STATE.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists()
 assert json.loads((ROOT/'preprocessing_checks.json').read_text())['fit_ready']
 assert shutil.disk_usage(P).free>10*1024**3
 command=[_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'),'-B','-u','fit.py','--subject_id',SUBJECT]
 s=dict(status='fitting',pid=os.getpid(),command=command,started_at=time.time(),gpu=2,subject=SUBJECT,budget='Unmodified upstream 3 epochs: 500+250+250 fitting iterations per batch',input_frames=30,protocol_sha256=hashlib.sha256((ROOT/'pilot_protocol.json').read_bytes()).hexdigest());save(s)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',EGL_DEVICE_ID='2',PYOPENGL_PLATFORM='egl',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',TORCH_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/torch_aux'),TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
 try:
  with (P/'logs/exavatar_fit_HN0057_worker_20261113.log').open('x') as log:
   child=subprocess.Popen(command,cwd=P/'sources/exavatar/fitting/main',env=env,stdout=log,stderr=subprocess.STDOUT)
   s['child_pid']=child.pid;save(s);code=child.wait();s['exit_code']=code
  assert code==0,'Inspect native fitting log before retry'
  s['status']='fitting_exited_zero_output_review_pending'
 except Exception as e:s.update(status='failed',reason=repr(e));raise
 finally:s['finished_at']=time.time();save(s)
