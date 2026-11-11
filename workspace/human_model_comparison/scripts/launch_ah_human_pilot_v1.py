"""Run one fixed human pilot on GPU4, respecting existing jobs and disk floor."""
from human_asset_benchmark.paths import expand as _hmc_expand
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import importlib.util

H = Path(__file__).resolve().parents[1]
R = H.parent/'vehicle_model_comparison'
B = H.parent/'assetbench'
P = R/'model_preparation/20261109_2100_CST'
PREFIX = Path(_hmc_expand('${CONDA_ROOT}/envs/ab_ah'))
JOB = 'human_pilot_v1_HN0004'


def main():
    lock = (H/'state/ah_human_pilot_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (H/'runs/ah'/JOB).exists(), 'Existing run must be audited; no overwrite or duplicate launch'
    spec = importlib.util.spec_from_file_location('rigid_reservations', P/'scripts/dispatch_rigid111_direct.py')
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)
    assert '4' not in {c['gpu'] for c in dispatch.METHODS.values()}
    state = dict(status='preflight', gpu='4', pid=os.getpid(), sample_id='HN0004',
                 started_at=time.time(), method='AH_human_R1_AHC', scores_produced=False)
    state_path = H/'state/ah_human_pilot_v1.json'
    def save():
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(state_path)
    save()
    try:
        assert shutil.disk_usage(H).free > 11*1024**3
        assert not any(x['gpu']=='4' for x in dispatch.old_reservations())
        inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        gpu = next(line.split(',') for line in inventory.splitlines() if line.split(',')[0].strip()=='4')
        uuid = gpu[1].strip()
        occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        assert uuid not in occupied and int(gpu[2])>40000
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        import torch
        # Maintain a visible CUDA context across inference/reload CPU phases.
        lease_tensor = torch.zeros(1,device='cuda')
        state.update(status='running', gpu_uuid=uuid); save()
        env = dict(os.environ, AB_JOB_ID=JOB, AB_INPUT_ROOT=str(H/'inputs/ah/jobs'/JOB),
            AB_PHASE='fixed_human_native_pilot', CUDA_HOME=str(PREFIX),
            AB_AH_CRADIO_PATH=str(B/'weights/ah/C-RADIO'), AB_AH_VAE_PATH=str(B/'weights/ah/dc-ae'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TORCH_EXTENSIONS_DIR=str(R/'cache/torch_ext/ab_ah'),
            TORCH_HOME=str(B/'cache/torch'), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        for key, suffix in [('HF_HOME','hf'),('HF_HUB_CACHE','hf/hub'),('HF_MODULES_CACHE','hf/modules'),
                            ('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','matplotlib'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
            path=H/'cache/ah'/suffix; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
        env['PATH']=str(PREFIX/'bin')+':'+env['PATH']
        (H/'logs').mkdir(exist_ok=True)
        with (H/'logs'/f'{JOB}.log').open('x') as log:
            for stage, command in [('inference',[str(PREFIX/'bin/python'),'-u',str(H/'scripts/run_ah_human_pilot_v1.py')]),
                                   ('reload',[str(PREFIX/'bin/python'),'-u',str(H/'scripts/reload_ah_human_pilot_v1.py'),JOB])]:
                state.update(stage=stage, command=command); save()
                proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(child_pid=proc.pid); save()
                code=proc.wait(); state.update(exit_code=code); save()
                if code: raise RuntimeError(f'{stage} exited {code}; see preserved log')
        state.update(status='native_pilot_reloaded_pending_visual_common_camera')
        del lease_tensor
    except BaseException as error:
        state.update(status='failed',reason=repr(error)); raise
    finally:
        state.update(finished_at=time.time()); save()


if __name__=='__main__':
    main()
