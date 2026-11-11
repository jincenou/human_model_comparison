"""Build official MMCV 2.1.0 in the isolated ExAvatar env with core pins protected."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,time

P=Path(__file__).resolve().parents[1];ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'));SRC=P/'sources/mmcv-2.1.0'
STATE=P/'state/exavatar_mmcv_v2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    lock=(P/'state/exavatar_mmcv_v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    assert sha(P/'sources/mmcv-2.1.0.tar.gz')=='d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f'
    wheelroot=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_wheels/exavatar_mmcv_v2'));wheelroot.mkdir(parents=True,exist_ok=False)
    constraints=P/'reports/exavatar_mmcv_core_constraints_v1.txt'
    constraints.write_text('torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\nnumpy==1.26.4\n')
    cuda=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2'))
    env={k:v for k,v in os.environ.items()if k.lower()not in('http_proxy','https_proxy','all_proxy')}
    env.update(CUDA_HOME=str(cuda),CUDA_VISIBLE_DEVICES='',TORCH_CUDA_ARCH_LIST='8.9',FORCE_CUDA='1',MMCV_WITH_OPS='1',MAX_JOBS='2',
        PATH=str(cuda/'bin')+':'+str(ENV/'bin')+':'+env['PATH'],TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'),PYTHONNOUSERSITE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    state=dict(status='preparing_dependencies',pid=os.getpid(),started_at=time.time(),source=str(SRC),version='2.1.0',
        environment=str(ENV),script_sha256=sha(__file__),stages=[],method_runtime_validated=False, retry_basis='v1 dependencies installed successfully; metadata failed because current setuptools removed pkg_resources. Pin setuptools60.2.0 and wheel0.45.1 from official ExAvatar environment.yml in isolated env only; preserve failed v1.')
    def save():
        t=STATE.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(STATE)
    def run(cmd,cwd=P):
        assert shutil.disk_usage(ENV).free>12*1024**3 and shutil.disk_usage(P).free>12*1024**3
        step=dict(command=cmd,started_at=time.time());state['stages'].append(step);save()
        child=subprocess.Popen(cmd,env=env,cwd=cwd);step['pid']=child.pid;save();code=child.wait();step['exit_code']=code;save();assert code==0,(cmd,code)
    try:
        save()
        run([str(ENV/'bin/python'),'-m','pip','install','--no-cache-dir','--timeout','120','--retries','5','-c',str(constraints),
            'setuptools==60.2.0','wheel==0.45.1','mmengine==0.10.7','addict==2.4.0','yapf==0.43.0','opencv-python==4.10.0.84','pyyaml==6.0.2','rich==13.4.2','termcolor==2.4.0'])
        state['status']='building';save()
        run([str(ENV/'bin/python'),'-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheelroot),str(SRC)],cwd=SRC)
        wheels=list(wheelroot.glob('*.whl'));assert len(wheels)==1
        state.update(wheel=str(wheels[0]),wheel_sha256=sha(wheels[0]),status='installing');save()
        run([str(ENV/'bin/python'),'-m','pip','install','--no-deps','--no-index',str(wheels[0])])
        run([str(ENV/'bin/python'),'-I','-c',"import torch,torchvision,numpy,mmcv,mmengine; from mmcv.ops import nms; assert torch.__version__=='2.6.0+cu124'; assert torchvision.__version__=='0.21.0+cu124'; assert numpy.__version__=='1.26.4'; assert mmcv.__version__=='2.1.0'; x=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[20.,20.,30.,30.]]); s=torch.tensor([.9,.8,.7]); _,idx=nms(x,s,.5); assert idx.tolist()==[0,2]; print('MMCV import and CPU NMS passed',mmcv.__version__,mmengine.__version__)"])
        run([str(ENV/'bin/python'),'-m','pip','check'])
        freeze=subprocess.check_output([str(ENV/'bin/python'),'-m','pip','freeze'],env=env,text=True)
        (P/'reports/exavatar_after_mmcv_freeze_v2.txt').write_text(freeze)
        state.update(status='mmcv_import_cpu_nms_passed_gpu_pending')
    except BaseException as exc:state.update(status='failed',reason=repr(exc));raise
    finally:state['finished_at']=time.time();save()

if __name__=='__main__':main()
