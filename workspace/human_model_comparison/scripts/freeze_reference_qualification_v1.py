"""Freeze source-only eligibility and the tested core metric arithmetic.

Does not certify method registration, ED-P, held-out generalization or orbits.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import json
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import numpy as np
from PIL import Image
import human_source_metrics_v1 as m
H=Path(__file__).resolve().parents[1]
C=H/'configs/human_source_common_v1'

def read(p):return json.loads(Path(p).read_text())
def verified(p,d):
    assert m.sha(p)==d, str(p)
def eligibility(review,mask):
    reasons=[]
    if not review['visual_passed']:reasons.append(review['visual_status'])
    if review.get('accessory_external_occlusion'):reasons.append('external_occlusion_of_carried_item')
    if (mask[0]>127).any() or (mask[-1]>127).any() or (mask[:,0]>127).any() or (mask[:,-1]>127).any():
        reasons.append('foreground_touches_raw_image_boundary')
    return reasons

def main():
    out=C/'reference_qualification_frozen_v1.json';protocol_out=C/'core_metric_protocol_frozen_v1.json'
    assert not out.exists() and not protocol_out.exists()
    selection=H/'inputs/benchmark_human100_v2/selection.json';selection_sha=m.sha(selection)
    assert selection_sha=='f6917e99f3e82d57b7ac2d141c50d8377aa953ec4ddb3e27192720d53af49f79'
    selected={r['id']:r for r in read(selection)['objects']};assert len(selected)==100
    rp=H/'reports/reference_qualification_v1/current_candidates_v1.json';refs=read(rp)
    assert refs['selection_sha256']==selection_sha and refs['model_outputs_used'] is False and not refs['pending_ids']
    for path,digest in refs['evidence_sha256'].items():verified(path,digest)
    index_path=C/'reviewed_source_v2/manifest.json';index=read(index_path);assert index['selection_sha256']==selection_sha
    cameras={r['id']:r for r in index['records']};assert set(cameras)==set(selected)
    checked_tables={};records=[]
    for r in refs['objects']:
        sid=r['id'];s=selected[sid];item=cameras[sid];cp=Path(item['path']);verified(cp,item['sha256']);c=read(cp)
        assert r['timestamp_us']==s['t_ref_us']==c['timestamp_us'] and r['identity_key']==s['identity_key']==c['identity_key']
        assert c['review_manifest_sha256']==m.sha(rp)
        for f,key in [('raw','raw_sha256'),('candidate_mask','candidate_mask_sha256'),('original_mask','original_mask_sha256')]:verified(r[f],r[key])
        assert c['source_rgb_sha256']==r['raw_sha256'] and c['source_mask_sha256']==r['candidate_mask_sha256']
        for name,digest in c['metadata_table_sha256'].items():
            path=Path(_hmc_expand('${NUSCENES_ROOT}/v1.0-trainval'))/(name+'.json')
            if name not in checked_tables:verified(path,digest);checked_tables[name]=digest
            assert checked_tables[name]==digest
        for h in r['review_history']:assert h['record'] in refs['evidence_sha256']
        rgb=np.asarray(Image.open(r['raw']).convert('RGB'));mask=np.asarray(Image.open(r['candidate_mask']).convert('L'))
        assert rgb.shape[:2]==mask.shape and list(rgb.shape[1::-1])==c['render_wh']
        prepared=m.prepare_reference(rgb,mask);assert prepared['status']=='ready' and list(prepared['roi'])==c['roi_xyxy']
        for path_key in ['reference_rgb','reference_mask']:verified(c[path_key],c[path_key+'_sha256'])
        assert np.array_equal(prepared['rgb'],np.asarray(Image.open(c['reference_rgb']).convert('RGB')))
        assert np.array_equal(prepared['mask'],np.asarray(Image.open(c['reference_mask']).convert('L')))
        reasons=eligibility(r,mask)
        records.append(dict(id=sid,identity_key=s['identity_key'],timestamp_us=s['t_ref_us'],cohort=s['cohort'],selected=True,
            reference_ready=not reasons,reference_eligible=not reasons,na_reasons=reasons,
            camera_contract=str(cp),camera_contract_sha256=item['sha256'],reference_rgb=c['source_rgb'],reference_rgb_sha256=c['source_rgb_sha256'],
            reference_mask=c['source_mask'],reference_mask_sha256=c['source_mask_sha256'],roi_xyxy=c['roi_xyxy'],reference_pixel_height=c['reference_pixel_height'],
            visual_note=r['note'],review_history=r['review_history'],mask_corrected_from_historical_input=r['corrected_candidate'],
            method_metric_eligibility='requires_independent_method_registration',scored=False))
    assert len(records)==100 and len({r['id'] for r in records})==100
    numerical_path=H/'reports/metric_numerical_validation_v1/validation.json';validation=read(numerical_path)
    assert validation['tests_failed']==0 and validation['tests_passed']==12
    for path,digest in validation['evidence'].items():verified(path,digest)
    candidate_path=C/'metric_protocol_candidate_v1.json';protocol=read(candidate_path)
    protocol.update(status='core_metrics_frozen_EDP_pending',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        production_scoring_enabled='only core metrics for independently certified method/reference pairs',
        frozen_metrics=['PSNR','SSIM','LPIPS','IoU','IoU_raw'],full_protocol_complete=False,
        code_sha256=m.sha(m.__file__),numerical_validation=str(numerical_path),numerical_validation_sha256=m.sha(numerical_path),
        candidate_protocol_sha256=m.sha(candidate_path))
    result=dict(schema='source_reference_qualification_frozen_v1',status='source_only_reference_eligibility_frozen',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),selection=str(selection),selection_sha256=selection_sha,
        denominator=100,eligible=sum(r['reference_eligible'] for r in records),na_count=sum(not r['reference_eligible'] for r in records),
        policy=dict(scope='isolated full-person visible appearance including clothes, worn bags and carried items',
            external_occlusion='exclude externally occluded body or included accessory, without severity exceptions',
            truncation='exclude raw-image truncation or foreground on the raw image boundary',
            self_occlusion='retain natural self-occlusion; do not synthesize hidden anatomy',
            pixel_size='record true raw pixel height; no score-dependent size threshold',
            selection='all 100 remain fixed; no replacements for unavailable reference or method',
            evidence='source-only review before batch scoring; predictions never choose mask, ROI or qualification'),
        original_review_manifest=str(rp),original_review_manifest_sha256=m.sha(rp),
        review_evidence_sha256=refs['evidence_sha256'],camera_index=str(index_path),camera_index_sha256=m.sha(index_path),
        metadata_table_sha256=checked_tables,model_outputs_used=False,historical_inputs_unchanged=True,
        core_protocol=str(protocol_out),objects=records,script_sha256=m.sha(__file__))
    with protocol_out.open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
    result['core_protocol_sha256']=m.sha(protocol_out)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(eligible=result['eligible'],na=result['na_count'],na_reasons=dict(Counter(reason for r in records for reason in r['na_reasons'])),output=str(out))))
if __name__=='__main__':main()
