"""Read frozen OmniRe records; write separate float64 diagnostic sums only."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
REPORT=H/'reports/native_human_export_reaudit_v1/omnire'
EXPORT=H/'exports/native_human_export_reaudit_v1/omnire'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def once():
    done=[]
    for vpath in sorted(EXPORT.glob('HN*/validation.json')):
        sid=vpath.parent.name;target=REPORT/sid/'contribution_precision_audit.json'
        if target.exists():done.append(sid);continue
        v=json.loads(vpath.read_text());p=REPORT/sid/'contribution.npz'
        if v['status']!='passed' or not p.exists():continue
        with np.load(p) as a:
            c=a['visible_weight'].astype(np.float64);selected=a['selected'];background=a['background']
            retained=float(c[selected].sum());other=float(c[~selected].sum());total=retained+other
            bg=float(c[background].sum());bgr=float(c[background&selected].sum())
            rec=dict(id=sid,source_validation=str(vpath),source_validation_sha256=sha(vpath),contribution_array=str(p),contribution_array_sha256=sha(p),
                dtype='float64 reduction of original float32 native weights; selection unchanged',total_visible_weight=total,
                retained_visible_weight=retained,retained_fraction=retained/total if total else None,
                background_visible_weight=bg,background_retained_fraction=bgr/bg if bg else None,other_node_weight=other,
                independent_total_sum_delta=float(c.sum())-total,min_per_gaussian_weight=float(c.min()),
                actual_other_node_gaussians=int(np.count_nonzero((c>0)&~selected)),original_float32_diagnostics=v['contributions'],
                original_validation_modified=False,asset_or_metric_input_modified=False)
        t=target.with_suffix('.tmp');t.write_text(json.dumps(rec,indent=2)+'\n');t.replace(target);done.append(sid)
    return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        done=once();print(json.dumps({'precision_audited':len(done)}),flush=True)
        if not args.watch or len(done)>=100:break
        workers=[REPORT/f'worker_{i}.json' for i in [0,1]]
        if all(p.exists() and json.loads(p.read_text()).get('status')=='completed' for p in workers):break
        time.sleep(20)
