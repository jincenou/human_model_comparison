"""Validate all frozen OmniRe exports and summarize without modifying outputs."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
from pathlib import Path
import numpy as np

H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
REPORT=H/'reports/native_human_export_reaudit_v1/omnire'
EXPORT=H/'exports/native_human_export_reaudit_v1/omnire'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    contract=json.loads((REPORT.parent/'batch_contract.json').read_text());rows=[]
    for sid in contract['ids']:
        vp=EXPORT/sid/'validation.json';v=json.loads(vp.read_text());assert v['status']=='passed' and v['reload_verified']
        assert v['source_reference']==contract['references'][sid]
        assert v['identity_key']==contract['references'][sid]['identity_key'] and v['timestamp_us']==contract['references'][sid]['timestamp_us']
        assert v['prediction_encoding']=='white_composited_uint8' and not v['source_mask_applied_to_prediction']
        for k in ['bundle','actor_preview','actor_alpha']:assert sha(v[k])==v[k+'_sha256'],(sid,k)
        with np.load(REPORT/sid/'contribution.npz') as z:
            expected=z['actor']|(z['background']&(z['visible_weight']>0))
            assert np.array_equal(z['selected'],expected),sid
            assert np.count_nonzero(expected)==v['exported_gaussians']
            assert np.count_nonzero(z['actor'])==v['original_actor_gaussians']
        p=REPORT/sid/'contribution_precision_audit.json';d=json.loads(p.read_text());assert d['source_validation_sha256']==sha(vp)
        assert d['background_retained_fraction'] is None or abs(d['background_retained_fraction']-1.)<1e-12
        assert 0<=d['retained_fraction']<=1 and d['other_node_weight']>=0
        rows.append(dict(id=sid,validation=str(vp),validation_sha256=sha(vp),precision_audit=str(p),precision_audit_sha256=sha(p),
            actor_node_present=v['actor_node_present'],actor=v['original_actor_gaussians'],background_added=v['background_added_gaussians'],
            total=v['exported_gaussians'],other_node_fraction=d['other_node_weight']/d['total_visible_weight'],
            reload_rgb_max=v['reload_errors']['rgb_max'],reload_alpha_max=v['reload_errors']['alpha_max'],
            prior_actor_difference=v['original_actor_reload_difference'],bundle_bytes=Path(v['bundle']).stat().st_size))
    summary=dict(status='all_100_technical_exports_verified',count=len(rows),reference_contract_sha256=sha(REPORT.parent/'batch_contract.json'),
        actor_node_count=sum(r['actor_node_present'] for r in rows),background_only_ids=[r['id'] for r in rows if not r['actor_node_present']],
        added_background_count=sum(r['background_added']>0 for r in rows),
        max_reload_rgb=max(r['reload_rgb_max'] for r in rows),max_reload_alpha=max(r['reload_alpha_max'] for r in rows),
        prior_actor_nonidentical_over_1byte=[r for r in rows if r['prior_actor_difference'] and r['prior_actor_difference']['max_byte']>1],
        rows_with_other_node_support=sum(r['other_node_fraction']>0 for r in rows),
        other_node_fraction_over_5pct=sorted([r for r in rows if r['other_node_fraction']>.05],key=lambda x:-x['other_node_fraction']),
        bundle_bytes=sum(r['bundle_bytes'] for r in rows),rows=rows,
        scope='Technical export, native-parameter/reload and fixed-reference checks only; semantic/native-quality defects are not automatically removed; static snapshots do not claim background motion')
    p=REPORT/'completion_audit.json';p.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','other_node_fraction_over_5pct']},indent=2))

if __name__=='__main__':main()
