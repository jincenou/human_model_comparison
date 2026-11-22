"""Install the official pinned package list into the isolated system-disk env."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

P=Path(__file__).resolve().parents[1]
ENV=Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_jacobian_train'))
SOURCE=P/'sources/jacobianavatar'
STATE=P/'state/jacobian_training_dependencies.json'

def save(value):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)

def main():
    with (P/'state/jacobian_training_dependencies.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=None
        if STATE.exists():
            prior=json.loads(STATE.read_text());assert prior['status']=='failed','Prior installer is not a failed terminal attempt'
            archive=P/'state/installation_history';archive.mkdir(exist_ok=True)
            dest=archive/(hashlib.sha256(STATE.read_bytes()).hexdigest()+'.json')
            if not dest.exists():dest.write_bytes(STATE.read_bytes())
        assert shutil.disk_usage(ENV).free>4*1024**3,'System-disk headroom insufficient'
        data_build=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_libigl'));data_build.mkdir(parents=True,exist_ok=True)
        assert shutil.disk_usage(data_build).free>10*1024**3,'Data-disk headroom insufficient'
        temp=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_training_dependencies'));temp.mkdir(parents=True,exist_ok=True)
        requirements=SOURCE/'requirements.txt'
        state=dict(status='running',pid=os.getpid(),started_at=time.time(),prefix=str(ENV),
                   requirements=str(requirements),requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                   scope='Official Python package list only; source CUDA builds, preprocessing envs and method runs remain separate',stages=[],
                   prior_failed_attempt=prior)
        save(state)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(data_build),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',MAX_JOBS='4')
        # The inherited development proxy points at a stopped localhost service.
        # Restrict the correction to this installer; do not alter global config.
        for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(name,None)
        # URL-specific global Git proxy settings override generic http.proxy.
        # Public dependency fetches need no user Git configuration; isolate only
        # this installer instead of editing the server's shared settings.
        env['GIT_CONFIG_GLOBAL']='/dev/null'
        env['GIT_CONFIG_SYSTEM']='/dev/null'
        count=int(env.get('GIT_CONFIG_COUNT','0'))
        for offset,key in enumerate(['http.proxy','https.proxy']):
            env[f'GIT_CONFIG_KEY_{count+offset}']=key;env[f'GIT_CONFIG_VALUE_{count+offset}']=''
        env['GIT_CONFIG_COUNT']=str(count+2)
        # Git 2.25 does not implement the newer config environment overrides.
        # An installer-local executable passes the overrides on every invocation.
        shim_dir=temp/'git_proxy_override';shim_dir.mkdir(exist_ok=True)
        shim=shim_dir/'git'
        shim.write_text('#!/bin/sh\nexec /usr/bin/timeout --kill-after=10s 180s /usr/bin/git -c http.proxy= -c https.proxy= "$@"\n')
        shim.chmod(0o755)
        env['PATH']=str(shim_dir)+os.pathsep+str(ENV/'bin')+os.pathsep+env.get('PATH','/usr/bin:/bin')
        local_libigl=temp/'libigl-python-bindings-0fbf1c9a82b6986497cec1e98c717a33eaf23f83'
        install_requirements=requirements
        if local_libigl.exists():
            provenance=json.loads((local_libigl/'_source_archive_provenance.json').read_text())
            assert provenance['commit']=='0fbf1c9a82b6986497cec1e98c717a33eaf23f83'
            assert hashlib.sha256((temp/'libigl-pinned.tar.gz').read_bytes()).hexdigest()==provenance['archive_sha256']
            original='libigl @ git+https://github.com/libigl/libigl-python-bindings.git@'+provenance['commit']
            content=requirements.read_text();assert content.count(original)==1
            install_requirements=temp/'requirements-official-pinned-local-libigl.txt'
            install_requirements.write_text(content.replace(original,str(local_libigl)))
            state['libigl_source_transport']=provenance
            state['install_requirements_sha256']=hashlib.sha256(install_requirements.read_bytes()).hexdigest();save(state)
        # A C++ fetch failure must not prevent all ordinary pinned packages
        # from installing. Preserve the package cache across these stages.
        python_requirements=temp/'requirements-official-python-packages.txt'
        lines=install_requirements.read_text().splitlines()
        libigl_lines=[line for line in lines if line.startswith('libigl @ ') or line==str(local_libigl)]
        assert len(libigl_lines)==1
        python_requirements.write_text('\n'.join(line for line in lines if line not in libigl_lines)+'\n')
        env['PIP_CACHE_DIR']=str(temp/'package_cache')
        env['CMAKE_BUILD_PARALLEL_LEVEL']='4'
        cmake_sources=temp/'pinned_cmake_sources'
        cmake_args=[]
        for name in ['nanobind','libigl','eigen','tetgen','triangle','predicates','embree','cgal','boost']:
            meta_path=cmake_sources/(name+'.json')
            if meta_path.exists():
                meta=json.loads(meta_path.read_text());assert Path(meta['path']).is_dir()
                archive=Path(meta.get('archive',str(cmake_sources/(name+'.tar.gz'))))
                assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
                cmake_args.append('-DFETCHCONTENT_SOURCE_DIR_'+name.upper()+'='+meta['path'])
        env['CMAKE_ARGS']=' '.join(cmake_args)
        commands=[
            [str(ENV/'bin/python'),'-m','pip','install','--no-deps','--retries','2','--timeout','60','-r',str(python_requirements)],
            [str(ENV/'bin/python'),'-m','pip','install','--no-deps','--retries','2','--timeout','60','--config-settings=build-dir='+str(data_build/'build'),str(local_libigl)],
            [str(ENV/'bin/python'),'-m','pip','check'],
        ]
        if prior and any(stage.get('status')=='passed' and stage.get('command')==commands[0] for stage in prior.get('stages',[])):
            state['reused_python_package_stage']=prior['stages'][0]
            commands=commands[1:]
        state['cmake_args']=cmake_args
        state['native_build_directory']=str(data_build/'build')
        save(state)
        try:
            for command in commands:
                stage=dict(command=command,status='running',started_at=time.time());state['stages'].append(stage);save(state)
                result=subprocess.run(command,cwd=SOURCE,env=env,timeout=2700)
                stage.update(exit_code=result.returncode,status='passed' if result.returncode==0 else 'failed',finished_at=time.time());save(state)
                if result.returncode:raise RuntimeError('Pinned dependency step failed; inspect log before any version change')
            state['status']='python_packages_installed_cuda_builds_pending'
        except Exception as error:
            state.update(status='failed',reason=repr(error));raise
        finally:
            state.update(finished_at=time.time(),system_disk_free_bytes=shutil.disk_usage(ENV).free);save(state)

if __name__=='__main__':main()
