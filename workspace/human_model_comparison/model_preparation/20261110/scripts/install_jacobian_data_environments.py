"""Prepare missing isolated environments on the user-authorized data disk.

No model execution or weight download. Native training builds stay separate.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import yaml

P = Path(__file__).resolve().parents[1]
SOURCE = P / 'sources/jacobianavatar'
parser = argparse.ArgumentParser()
parser.add_argument('kind', choices=['prep', 'sapiens2', 'flexi_fitting', 'flexi_training'])
args = parser.parse_args()
kind = args.kind
prefix = Path(_hmc_expand('${HMC_STORAGE_ROOT}/conda_envs')) / {'prep':'hcmp_jacobian_prep','sapiens2':'hcmp_sapiens2','flexi_fitting':'hcmp_flexi_fitting','flexi_training':'hcmp_flexi_training'}[kind]
temp = Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp')) / ('jacobian_' + kind)
state_path = P / 'state' / ('jacobian_data_env_' + kind + '_20261113.json')

def save(state):
    target = state_path.with_suffix('.tmp')
    target.write_text(json.dumps(state, indent=2) + '\n')
    target.replace(state_path)

with state_path.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not state_path.exists(), 'Inspect the previous attempt before restarting'
    assert not prefix.exists(), 'Never overwrite an existing environment'
    assert shutil.disk_usage('/mnt').free > 40 * 1024**3
    temp.mkdir(parents=True, exist_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for directory in ['tmp', 'conda_pkgs', 'pip_cache', 'cache']:
        (temp / directory).mkdir(exist_ok=True)
    original = (P/'sources/flexiavatar/smplx_fitting/requirements.txt') if kind=='flexi_fitting' else SOURCE / ('requirements_prep.txt' if kind == 'prep' else 'requirements_sapiens2.txt')
    if kind=='flexi_training': original=P/'sources/flexiavatar/avatar_training/environment.yml'
    contents = original.read_text()
    deferred = []
    if kind=='flexi_training':
        spec=yaml.safe_load(contents)
        packages=next(item['pip']for item in spec['dependencies']if isinstance(item,dict)and'pip'in item)
        assert 'chumpy==0.71'in packages
        contents='\n'.join(item for item in packages if item!='chumpy==0.71')+'\n'
        deferred=['PyTorch3D 0.7.6 and depth Gaussian rasterizer source builds']
    if kind == 'prep':
        deferred = [line for line in contents.splitlines() if line.startswith('libigl @ ')]
        assert len(deferred) == 1
        contents = '\n'.join(line for line in contents.splitlines() if line not in deferred) + '\n'
    requirements = temp / 'requirements.txt'
    requirements.write_text(contents)
    state = dict(status='running', pid=os.getpid(), prefix=str(prefix), started_at=time.time(),
                 original_requirements=str(original), original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                 effective_requirements=str(requirements), effective_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                 deferred=deferred, native_source_builds_pending=(kind != 'sapiens2'), full_method_validated=False,
                 authorization='User permits missing environments on data disk; weights stay on system disk', stages=[])
    save(state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(temp / 'tmp'), CONDA_PKGS_DIRS=str(temp / 'conda_pkgs'),
               PIP_CACHE_DIR=str(temp / 'pip_cache'), XDG_CACHE_HOME=str(temp / 'cache'),
               HF_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/huggingface'),
               TORCH_HOME=_hmc_expand('${HMC_WEIGHT_ROOT}/torch'),
               MODELSCOPE_CACHE=_hmc_expand('${HMC_WEIGHT_ROOT}/modelscope'))
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'PYTHONPATH']:
        env.pop(key, None)
    python = str(prefix / 'bin/python')
    commands = [
        [_hmc_expand('${CONDA_ROOT}/bin/conda'), 'create', '-y', '-p', str(prefix), '--override-channels', '-c', 'conda-forge',
         'python=' + ('3.9' if kind == 'prep' else '3.12'), 'pip'],
        [python, '-m', 'pip', 'install', '--retries', '3', '--timeout', '60'] +
        (['--no-deps'] if kind == 'prep' else []) + ['-r', str(requirements)],
        [python, '-m', 'pip', 'check'],
    ]
    if kind=='flexi_fitting':
        commands[0]=[_hmc_expand('${CONDA_ROOT}/bin/conda'),'create','-y','-p',str(prefix),'--clone',_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar')]
        constraints=temp/'torch_constraints.txt'
        constraints.write_text('torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\nnumpy==1.26.4\n')
        commands[1]=[python,'-m','pip','install','--no-build-isolation','--retries','3','--timeout','60','-c',str(constraints),'-r',str(requirements)]
        state['clone_source']=_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar')
        state['deferred']=['Official PyTorch3D 0.7.8 source build (clone contains 0.7.9)','torchaudio 2.6.0+cu124','Flexi fitting tool sources and runtime tests']
        save(state)
    if kind=='flexi_training':
        commands[0][-2]='python=3.8.20'
        constraints=temp/'torch_constraints.txt'
        constraints.write_text('torch==2.0.1+cu118\ntorchvision==0.15.2+cu118\ntorchaudio==2.0.2+cu118\nnumpy==1.23.1\n')
        torch_install=[python,'-m','pip','install','--retries','3','--timeout','60','--index-url','https://download.pytorch.org/whl/cu118','torch==2.0.1','torchvision==0.15.2','torchaudio==2.0.2']
        packages_install=[python,'-m','pip','install','--no-build-isolation','--retries','3','--timeout','60','-c',str(constraints),'-r',str(requirements)]
        chumpy_source=P/'sources/chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da'
        chumpy_install=[python,'-m','pip','install','--no-deps','--no-build-isolation',str(chumpy_source)]
        commands=[commands[0],torch_install,packages_install,chumpy_install,commands[-1]]
        state['source_package_transport']={'name':'chumpy','requested_version':'0.71','reason':'PyPI 0.71 returned 404; upstream pinned source declares 0.71', 'source':str(chumpy_source),'archive_sha256':hashlib.sha256(chumpy_source.with_suffix('.tar.gz').read_bytes()).hexdigest()}
        save(state)
    try:
        for index, command in enumerate(commands):
            stage = dict(command=command, status='running', started_at=time.time())
            state['stages'].append(stage)
            save(state)
            child = subprocess.Popen(command, cwd=SOURCE, env=env, start_new_session=True)
            stage['pid'] = child.pid
            save(state)
            while child.poll() is None:
                if shutil.disk_usage('/mnt').free < 10 * 1024**3 or time.time() - stage['started_at'] > 21600:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise RuntimeError('Installer stopped: data disk below 10 GiB or stage exceeded 6 hours')
                time.sleep(10)
            stage.update(exit_code=child.returncode, finished_at=time.time(), status='passed' if child.returncode == 0 else 'failed')
            save(state)
            if child.returncode:
                if command[-2:] == ['pip','check']:
                    state['status'] = 'installed_dependency_check_requires_review'
                    break
                raise RuntimeError('Install failed; inspect log before changing versions')
        else:
            state['status'] = 'packages_installed_native_builds_pending' if kind != 'sapiens2' else 'packages_installed_runtime_pending'
    except Exception as error:
        state.update(status='failed', reason=repr(error))
        raise
    finally:
        state.update(finished_at=time.time(), data_disk_free_bytes=shutil.disk_usage('/mnt').free)
        save(state)
