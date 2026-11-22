"""Read actual per-person evidence and independently recompute final aggregates."""
import collections
import hashlib
import json
from pathlib import Path
import time
import numpy as np

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
O=H/'reports/streetgs_human_probe_v1'
G=H/'reports/streetgs_human_geometry_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def check(path,digest):assert sha(path)==digest,str(path)

def main():
    plan=read(O/'extraction_plan.json');check(plan['selection_path'],plan['selection_sha256'])
    ids=[r['id'] for r in read(plan['selection_path'])['objects']]
    assert len(set(ids))==100
    coverage=read(O/'coverage_audit.json')
    native=read(O/'native_adapter_check.json');assert native['status']=='passed'
    core=read(O/'static_summary.json');assert not core['unvisited'] and core['edp_processed']==100
    records=[];image_rows=[]
    for sid in ids:
        folder=H/'exports/streetgs_human_probe_v1/objects'/sid
        v=read(folder/'validation.json');assert v['status']=='passed' and v['reload_verified'] and v['selected_parameters_equal_checkpoint']
        for k in ['bundle','actor_preview','actor_alpha','reload_record']:
            check(v[k],v['reload_sha256'] if k=='reload_record' else v[k+'_sha256'])
        replay=read(v['reload_record']);assert replay['scene_checkpoint_reads']==replay['source_image_reads']==replay['source_mask_reads']==0
        assert replay['rgb_max']==replay['alpha_max']==0
        assert not v['native_dynamic_human_available']
        im=read(O/'static_rows'/(sid+'.json'));assert im['status']=='scored'
        check(folder/'validation.json',im['source_validation_sha256'])
        image_rows.append(im)
        records.append(dict(id=sid,export_sha256=sha(folder/'validation.json'),gaussians=v['gaussians'],foreground_pixels=v['foreground_pixels_gt_half']))
    expected=core['groups']['streetgs'];means={}
    for key in ['PSNR','SSIM','LPIPS','IoU','IoU_raw']:
        vals=[x['metrics'][key]['value'] for x in image_rows]
        assert all(x['metrics'][key]['status']=='ok' for x in image_rows)
        mean=float(np.mean(vals));assert abs(mean-expected[key]['value'])<1e-12
        assert expected[key]['n']==100 and set(expected[key]['ids'])==set(ids)
        means[key]=dict(value=mean,n=100)
    edp=read(O/'edp/all/results.json')['rows'];assert {r['id'] for r in edp}==set(ids)
    valid=[r for r in edp if r['score']['status']=='ok'];mean=float(np.mean([r['score']['value'] for r in valid]))
    assert expected['EDP']['n']==len(valid) and abs(mean-expected['EDP']['value'])<1e-12
    means['EDP']=dict(value=mean,n=len(valid),no_valid_parts=len(edp)-len(valid))
    poses={kind:dict(collections.Counter(read(p)['status'] for p in (O/'edp/all/poses').glob('*_'+kind+'.json'))) for kind in ['reference','streetgs']}
    geometry=read(G/'summary.json');assert geometry['status']=='completed' and not geometry['unattempted_ids']
    grows=[read(G/'results'/(sid+'.json')) for sid in ids]
    assert not any(x['status'] in ['running','technical_failure_pending_review'] for x in grows)
    gm={}
    for panel in ['annotation_vertical_extent_conditioned','native_metric']:
        rows=[x for x in grows if x['reference_panel']=='main' and panel in x['panels']]
        assert len(rows)==84
        mets=[x['panels'][panel]['metrics'] for x in rows]
        cds=[x['cd_m']*100 for x in mets if x.get('cd_m') is not None and np.isfinite(x['cd_m'])]
        fs=[x['f1_10cm']*100 for x in mets]
        assert len(fs)==84
        group=geometry['groups']['main'][panel]
        for key,vals in [('CD_cm',cds),('F10',fs)]:
            assert len(vals)==group[key]['n'] and abs(float(np.mean(vals))-group[key]['value'])<1e-10
        gm[panel]=dict(CD_cm=float(np.mean(cds)),CD_n=len(cds),F10_percent=float(np.mean(fs)),F10_n=84,valid_empty=84-len(cds))
    for r in grows:
        if r['status']=='completed':
            for k in ['certificate','returns']:check(r[k],r[k+'_sha256'])
    originals=[R/'reports/asset_experiments_goal_20261115_v1/汇总数据.json',
               R/'reports/rigid_human_selected_protocol_20261115_v3/汇总数据.json',
               R/'reports/asset_experiments_paper_20261115_v3/06_表格精确数据.json',
               H/'reports/static_reference_revision_v2/core/summary.json']
    result=dict(status='verified',checked_at=time.time(),selection_sha256=plan['selection_sha256'],
        export_count=100,independent_reload_exact_count=100,static=means,edp_detector_states=poses,
        geometry_main=gm,geometry_outcomes=dict(collections.Counter(x['status'] for x in grows)),
        geometry_unavailable=[dict(id=x['id'],reference_panel=x['reference_panel'],reason=x.get('reason')) for x in grows if x['status']!='completed'],
        temporal=dict(status='not_applicable_to_current_static_export',human_native_motion_exported=False,
                      evidence=str(O/'coverage_audit.json'),evidence_sha256=sha(O/'coverage_audit.json')),
        preserved_originals={str(p):sha(p) for p in originals},records=records)
    (O/'settlement_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','preserved_originals']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
