"""Install pinned ExAvatar core in the user-authorized system-disk prefix."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

P = Path(__file__).resolve().parents[1]
PREFIX = Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'))
STATE = P/'state/exavatar_system_environment.json'


def main():
    lock=(P/'state/exavatar_system_environment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1',TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    state=dict(status='installing',pid=os.getpid(),prefix=str(PREFIX),started_at=time.time(),
               scope='Pinned core only; CUDA extensions, body models and method demos remain pending',stages=[])
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    save()
    try:
        assert (PREFIX/'bin/python').is_file()
        assert shutil.disk_usage(PREFIX).free>22*1024**3, 'Core install needs reserved staging headroom'
        commands=[
            [str(PREFIX/'bin/python'),'-m','pip','install','--no-cache-dir','torch==2.6.0','torchvision==0.21.0','--index-url','https://download.pytorch.org/whl/cu124'],
            [str(PREFIX/'bin/python'),'-m','pip','install','--no-cache-dir','numpy==1.26.4','scipy==1.15.2','pillow==10.4.0','smplx==0.1.28','trimesh==3.23.5','ninja','packaging'],
            [str(PREFIX/'bin/python'),'-c','import sys,torch,torchvision,numpy,smplx; print(sys.prefix,torch.__version__,torchvision.__version__,numpy.__version__,torch.version.cuda); assert torch.__version__=="2.6.0+cu124"; assert numpy.__version__=="1.26.4"'],
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
        (P/'reports/exavatar_system_core_freeze.txt').write_text(result)
        state.update(status='core_import_passed_method_dependencies_pending')
    except BaseException as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state.update(finished_at=time.time(),system_disk_free_bytes=shutil.disk_usage(PREFIX.parent).free);save()


if __name__=='__main__':
    main()
