"""Install pinned pose dependencies in the isolated ExAvatar environment only."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl, hashlib, json, os, shutil, subprocess, time

P = Path(__file__).resolve().parents[1]
ENV = Path(_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar'))
STATE = P/'state/exavatar_pose_install_v1.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    lock = STATE.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists()
    source = P/'sources/jacobianavatar/libs/mmpose'
    dest = P/'sources/exavatar/fitting/tools/mmpose'
    provenance = json.loads((P/'state/jacobian_mmpose_source_recovery_v1.json').read_text())
    for relative, digest in provenance['files_sha256'].items():
        assert sha(source/relative) == digest, relative
    assert not dest.exists()
    shutil.copytree(source, dest, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pth', '*.egg-info', 'build'))
    chumpy = P/'sources/chumpy-580566eafc9ac68b2614b64d6f7aaa84eebb70da'
    assert sha(chumpy.with_suffix('.tar.gz')) == '51253baf90c79a14d69ed4ad911da48e194a6ccad160db80dd7c6cae5e7d14cf'
    constraints = P/'reports/exavatar_pose_constraints_v1.txt'
    constraints.write_text('torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\nnumpy==1.26.4\nmmcv==2.1.0\nmmengine==0.10.7\nsetuptools==60.2.0\n')
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', TMPDIR=_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp'),
               OMP_NUM_THREADS='4', PIP_DISABLE_PIP_VERSION_CHECK='1')
    state = dict(status='installing', pid=os.getpid(), started_at=time.time(), stages=[],
                 script_sha256=sha(__file__), environment=str(ENV), method_runtime_validated=False,
                 mmpose_revision=provenance['revision'], mmpose_source=str(dest),
                 chumpy_revision='580566eafc9ac68b2614b64d6f7aaa84eebb70da',
                 chumpy_basis='Official upstream commit pinned by Jacobian; source reports 0.71, matching ExAvatar package version. ExAvatar does not identify its own Chumpy commit.')
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    def run(args):
        assert shutil.disk_usage(ENV).free > 12*1024**3
        assert shutil.disk_usage(P).free > 12*1024**3
        step=dict(command=args, started_at=time.time());state['stages'].append(step);save()
        child=subprocess.Popen(args,env=env,cwd=dest);step['pid']=child.pid;save()
        step['exit_code']=child.wait();save();assert step['exit_code']==0, step
    py=str(ENV/'bin/python')
    try:
        save()
        run([py,'-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation',str(chumpy)])
        run([py,'-m','pip','install','--no-cache-dir','--timeout','120','--retries','5','-c',str(constraints),
             'mmdet==3.3.0','cython==3.0.12','json-tricks==3.17.3','munkres==1.1.4',
             'pycocotools==2.0.8','scikit-image==0.25.2','scikit-learn==1.6.1','shapely==2.1.0',
             'xtcocotools==1.14.3','terminaltables==3.1.10'])
        run([py,'-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation',str(dest)])
        run([py,'-I','-c',"import torch,torchvision,numpy,chumpy,mmcv,mmdet,mmpose; from mmdet.apis import init_detector,inference_detector; from mmpose.apis import init_model,inference_topdown; assert torch.__version__=='2.6.0+cu124'; assert torchvision.__version__=='0.21.0+cu124'; assert numpy.__version__=='1.26.4'; assert chumpy.__version__=='0.71'; assert mmdet.__version__=='3.3.0'; assert mmpose.__version__=='1.3.2'; x=chumpy.array([2.,3.]); numpy.testing.assert_allclose((x*x).r,[4.,9.]); print('Pose imports and Chumpy forward passed')"])
        run([py,'-m','pip','check'])
        (P/'reports/exavatar_after_pose_freeze_v1.txt').write_text(subprocess.check_output([py,'-m','pip','freeze'],env=env,text=True))
        state['status']='pose_imports_passed_weight_inference_pending'
    except BaseException as exc:
        state.update(status='failed',reason=repr(exc));raise
    finally:
        state['finished_at']=time.time();save()

if __name__=='__main__': main()
