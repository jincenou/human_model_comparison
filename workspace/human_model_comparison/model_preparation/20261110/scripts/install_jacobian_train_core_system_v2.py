"""Install pinned JacobianAvatar core in the user-authorized system-disk prefix."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

P = Path(__file__).resolve().parents[1]
PREFIX = Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train'))
STATE = P/'state/jacobian_train_system_environment_v2.json'


def main():
    lock=(P/'state/jacobian_train_system_environment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1',TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    state=dict(status='installing',pid=os.getpid(),prefix=str(PREFIX),started_at=time.time(),
               scope='Pinned core only; CUDA extensions, body models and method demos remain pending',stages=[], retry_basis='Previous installer exited with connection reset; preserve v1 state/log. Enable bounded pip resume retries and persistent cache.')
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    save()
    try:
        assert (PREFIX/'bin/python').is_file()
        assert shutil.disk_usage(PREFIX).free>22*1024**3, 'Core install needs reserved staging headroom'
        commands=[
            [str(PREFIX/'bin/python'),'-m','pip','install','--cache-dir',_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/pip_cache'),'--retries','10','--resume-retries','20','--timeout','120','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu121'],
            [str(PREFIX/'bin/python'),'-m','pip','install','--cache-dir',_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/pip_cache'),'--retries','10','--resume-retries','20','--timeout','120','numpy==2.0.2','scipy==1.13.1','pillow==11.3.0','smplx==0.1.28','trimesh==4.11.5','ninja==1.13.0','packaging==24.2'],
            [str(PREFIX/'bin/python'),'-c','import sys,torch,torchvision,numpy,smplx; print(sys.prefix,torch.__version__,torchvision.__version__,numpy.__version__,torch.version.cuda); assert torch.__version__=="2.5.1+cu121"; assert numpy.__version__=="2.0.2"'],
            [str(PREFIX/'bin/python'),'-m','pip','check']]
        for command in commands:
            assert shutil.disk_usage(PREFIX).free>11*1024**3
            stage=dict(command=command,status='running',started_at=time.time());state['stages'].append(stage);save()
            process=subprocess.Popen(command,env=env)
            stage['pid']=process.pid;save()
            while process.poll() is None:
                if shutil.disk_usage(PREFIX).free<11*1024**3:
                    process.terminate();process.wait();raise RuntimeError('System disk reserve reached; stopped current installer, preserved evidence')
                time.sleep(5)
            stage.update(exit_code=process.returncode,status='passed' if process.returncode==0 else 'failed');save()
            assert process.returncode==0, f'Installation stage failed: {command}'
        result=subprocess.check_output([str(PREFIX/'bin/python'),'-m','pip','freeze'],env=env,text=True)
        (P/'reports/jacobian_train_system_core_freeze.txt').write_text(result)
        state.update(status='core_import_passed_method_dependencies_pending')
    except BaseException as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state.update(finished_at=time.time(),system_disk_free_bytes=shutil.disk_usage(PREFIX.parent).free);save()


if __name__=='__main__':
    main()
