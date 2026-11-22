from human_asset_benchmark.paths import expand as _hmc_expand
import json,subprocess,sys,time
from pathlib import Path
H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'));ROOT=H/'reports/native_human_export_reaudit_v1'
gpu=sys.argv[1];plan=json.loads((ROOT/'streetgs/neighbors/batch_plan.json').read_text())
if gpu=='0':
    for _ in range(360):
        state=json.loads((ROOT/'jobs/streetgs_bounded_neighbors_batch01.json').read_text())
        if state['status']=='completed':break
        if state['status']=='failed':raise RuntimeError('First neighbor batch needs review')
        time.sleep(10)
    else:raise RuntimeError('First neighbor batch did not finish in one hour')
command=[sys.executable,'-B','-u',str(H/'scripts/native_reaudit_gpu.py'),'--env','omnire','--gpu',gpu,
    '--key','streetgs_bounded_neighbors_remaining_gpu'+gpu,str(H/'scripts/native_reaudit_streetgs_neighbors_v1.py'),
    '--worker-key','worker_remaining_gpu'+gpu,'--ids',*plan['remaining'][gpu]]
raise SystemExit(subprocess.run(command,check=False).returncode)
