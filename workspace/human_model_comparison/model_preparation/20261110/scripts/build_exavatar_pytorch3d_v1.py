"""Build a pinned PyTorch3D candidate only into the isolated ExAvatar environment."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time
P=Path(__file__).resolve().parents[1];ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'))
SRC=P/'sources/pytorch3d_v079';REV='33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(P/'state/exavatar_pytorch3d_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statefile=P/'state/exavatar_pytorch3d_v1.json';assert not statefile.exists()
 assert json.loads((P/'state/exavatar_raster_depth_import_verification_v2.json').read_text())['status']=='extension_import_passed_gpu_runtime_pending'
 assert subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()==REV
 wheelroot=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_wheels/exavatar_pytorch3d_v1'));wheelroot.mkdir(parents=True,exist_ok=False)
 env={k:v for k,v in os.environ.items()if k.lower()not in('http_proxy','https_proxy','all_proxy')}
 cuda=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2'))
 env.update(CUDA_HOME=str(cuda),CUDA_VISIBLE_DEVICES='',TORCH_CUDA_ARCH_LIST='8.9',FORCE_CUDA='1',MAX_JOBS='2',
  CUB_HOME=str(cuda/'include'),PATH=str(cuda/'bin')+':'+str(ENV/'bin')+':'+env['PATH'],TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'),PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
 state=dict(status='building',pid=os.getpid(),source=str(SRC),revision=REV,prefix=str(ENV),cuda_compiler=str(cuda/'bin/nvcc'),arch='8.9',gpu_runtime_tested=False,stages=[],started_at=time.time())
 def save():
  t=statefile.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(statefile)
 def run(cmd,cwd=SRC):
  assert shutil.disk_usage(ENV).free>12*1024**3 and shutil.disk_usage(P).free>11*1024**3
  s=dict(command=cmd,started_at=time.time());state['stages'].append(s);save()
  child=subprocess.Popen(cmd,env=env,cwd=cwd);s['pid']=child.pid;save();code=child.wait();s['exit_code']=code;save();assert code==0,cmd
 try:
  save();run([str(ENV/'bin/python'),'-m','pip','install','--no-cache-dir','--timeout','120','--resume-retries','10','iopath==0.1.10','portalocker==3.2.0','tqdm==4.67.1'],cwd=P)
  run([str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheelroot),str(SRC)])
  wheels=list(wheelroot.glob('*.whl'));assert len(wheels)==1
  state.update(wheel=str(wheels[0]),wheel_sha256=sha(wheels[0]),status='installing');save()
  run([str(ENV/'bin/python'),'-m','pip','install','--no-deps','--no-index',str(wheels[0])])
  run([str(ENV/'bin/python'),'-I','-c','import torch,pytorch3d; from pytorch3d import _C; from pytorch3d.structures import Meshes; from pytorch3d.ops import knn_points; print(torch.__version__,pytorch3d.__version__,_C.__file__)'],cwd=P)
  run([str(ENV/'bin/python'),'-m','pip','check'],cwd=P)
  state.update(status='extension_import_passed_gpu_runtime_pending',compatibility_note='Built against isolated torch2.6/cu124; upstream INSTALL lists tested torch only through2.4.1, so GPU/runtime/method compatibility remains to be verified.')
 except BaseException as e:state.update(status='failed',reason=repr(e));raise
 finally:state['finished_at']=time.time();save()
if __name__=='__main__':main()
