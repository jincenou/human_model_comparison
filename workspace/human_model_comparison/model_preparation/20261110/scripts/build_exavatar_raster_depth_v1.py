"""Compile the official depth/alpha rasterizer only into isolated ExAvatar env."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
P=Path(__file__).resolve().parents[1];ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'))
SRC=P/'sources/exavatar_raster_depth';REV='03f0b7d00383d6e96c22b37325ac9e5450947bf5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(P/'state/exavatar_raster_depth_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statefile=P/'state/exavatar_raster_depth_v1.json';assert not statefile.exists()
 assert subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()==REV
 wheelroot=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_wheels/exavatar_raster_depth_v1'));wheelroot.mkdir(parents=True,exist_ok=False)
 env={k:v for k,v in os.environ.items()if k.lower()not in('http_proxy','https_proxy','all_proxy')}
 cuda=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2'))
 env.update(CUDA_HOME=str(cuda),CUDA_VISIBLE_DEVICES='',TORCH_CUDA_ARCH_LIST='8.9',FORCE_CUDA='1',MAX_JOBS='2',
  PATH=str(cuda/'bin')+':'+str(ENV/'bin')+':'+env['PATH'],TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'),PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
 state=dict(status='building',pid=os.getpid(),source=str(SRC),revision=REV,prefix=str(ENV),cuda_compiler=str(cuda/'bin/nvcc'),arch='8.9',gpu_runtime_tested=False,stages=[],started_at=time.time())
 def save():
  t=statefile.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(statefile)
 def run(cmd):
  assert shutil.disk_usage(ENV).free>12*1024**3 and shutil.disk_usage(P).free>11*1024**3
  s=dict(command=cmd,started_at=time.time());state['stages'].append(s);save()
  child=subprocess.Popen(cmd,env=env,cwd=SRC);s['pid']=child.pid;save();code=child.wait();s['exit_code']=code;save();assert code==0,cmd
 try:
  save();run([str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheelroot),str(SRC)])
  wheels=list(wheelroot.glob('*.whl'));assert len(wheels)==1
  state.update(wheel=str(wheels[0]),wheel_sha256=sha(wheels[0]),status='installing');save()
  run([str(ENV/'bin/python'),'-m','pip','install','--no-deps','--no-index',str(wheels[0])])
  run([str(ENV/'bin/python'),'-c','import torch; import diff_gaussian_rasterization_depth as d; from diff_gaussian_rasterization_depth import _C; print(torch.__version__,d.__file__); assert hasattr(d,"GaussianRasterizer")'])
  state.update(status='extension_import_passed_gpu_runtime_pending')
 except BaseException as e:state.update(status='failed',reason=repr(e));raise
 finally:state['finished_at']=time.time();save()
if __name__=='__main__':main()
