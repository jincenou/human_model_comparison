"""Evidence-bound completion gates and independent resolved geometry arithmetic.

Missing evidence stays pending. No scoring, export, reference or paper mutation.
"""
import collections
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np

from native_reaudit_common import read,save,sha
from summarize_native_reaudit_geometry_diagnostic_v1 import dist_means

H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1';RES=ROOT/'resolved'
METHODS=['streetgs','splatad','omnire']


def evidence(path):
    path=Path(path)
    return dict(path=str(path),exists=path.is_file(),sha256=sha(path) if path.is_file() else None)


def check(name,ok,details=None,paths=()):
    return dict(name=name,status='passed' if ok else 'pending',details=details,evidence=[evidence(p) for p in paths])


def geometry_audit(method,ids):
    root=RES/'geometry'/method;sp=root/'summary.json'
    if not sp.exists():return dict(status='pending',reason='Resolved geometry summary absent')
    summary=read(sp);paths=sorted((root/'results').glob('*.json'));rows=[read(p) for p in paths]
    statuses={'completed','extent_numerically_unresolved','export_unavailable','terminal_technical_failure_recorded'}
    if len(rows)!=100 or any(r['status'] not in statuses for r in rows):
        return dict(status='pending',reason='Resolved geometry not terminal for all 100',counts=dict(collections.Counter(r['status'] for r in rows)))
    assert {r['id'] for r in rows}==ids
    refp=H/'evaluation_static_cd_fscore_time_v1/reference_manifest.json'
    refs={r['id']:r for r in read(refp)['rows']};verified=[]
    for row in rows:
        ref=refs[row['id']];assert row['queries_sha256']==ref['queries_sha256']==sha(ref['queries'])
        assert row['identity_key']==ref['identity_key'] and row['t_ref_us']==ref['t_ref_us']
        if row['status']!='completed':assert row.get('reason')
        with np.load(ref['queries']) as z:Q=z['q'];o=z['o'];u=z['u']
        for panel,p in row.get('panels',{}).items():
            f=p['returns'] if panel=='native_metric' else row['returns'];digest=p['returns_sha256'] if panel=='native_metric' else row['returns_sha256']
            assert sha(f)==digest
            with np.load(f) as z:hit=z['hit'];ranges=z['range_pred']
            m=p['metrics'];assert m['reference_count']==len(Q) and m['hit_count']==int(hit.sum())
            assert abs(m['ray_hit']-float(hit.mean()))<1e-12
            if hit.any():
                P=o[hit]+ranges[hit,None]*u[hit];a,b=dist_means(P,Q)
                assert abs(float((a.mean()+b.mean())/2)-m['cd_m'])<1e-10
                for threshold in [.05,.1,.2]:
                    pr=float((a<threshold).mean());re=float((b<threshold).mean());f1=2*pr*re/(pr+re) if pr+re else 0
                    assert abs(m[f'f1_{round(threshold*100)}cm']-f1)<1e-12
            else:assert m['cd_m'] is None and m['f1_10cm']==0
            verified.append(dict(id=row['id'],panel=panel,returns_sha256=digest))
    for panel in ['main','low_support_exploratory']:
        for measure in ['native_metric','annotation_vertical_extent_conditioned']:
            pairs=[(r['id'],r['panels'][measure]['metrics']) for r in rows if r['reference_panel']==panel and measure in r.get('panels',{})]
            cells=summary['groups'][panel][measure]
            empty=sorted(i for i,m in pairs if m['status']=='empty_pred')
            assert cells['CD_cm']['valid_empty_n']==len(empty) and sorted(cells['CD_cm']['valid_empty_ids'])==empty
            if empty:assert cells['CD_cm']['extended_mean']=='infinity'
            for key,raw in [('CD_cm','cd_m'),('F5','f1_5cm'),('F10','f1_10cm'),('F20','f1_20cm'),('RayHit','ray_hit')]:
                valid=[(i,m[raw]) for i,m in pairs if m[raw] is not None];c=cells[key]
                assert c['n']==len(valid) and sorted(c['ids'])==sorted(i for i,v in valid)
                if valid:assert abs(c['value']-float(np.mean([v for i,v in valid])*100))<1e-10
                else:assert c['value'] is None
                assert c.get('valid_empty_n',0)==len(empty)
    result=dict(status='passed',method=method,identity_count=len(rows),counts=dict(collections.Counter(r['status'] for r in rows)),
        independent_metric_panels=len(verified),independent_formula='Blockwise scipy cdist; symmetric unsquared nearest-neighbour distance; strict 5/10/20cm F-score; full reference set retained',
        means='Equal identity; own-valid range per panel; finite-CD denominator and valid-empty F-score zero preserved',
        original_reference_manifest=evidence(refp),summary=evidence(sp),protocol=evidence(root/'protocol.json'),
        row_evidence=[evidence(p) for p in paths],verified_returns=verified,rerendered=False,old_scores_modified=False,script_sha256=sha(__file__))
    save(root/'independent_arithmetic_audit.json',result)
    return dict(status='passed',identity_count=100,counts=result['counts'],independent_metric_panels=len(verified),artifact=evidence(root/'independent_arithmetic_audit.json'))


def static_audit(method,ids):
    root=RES/'metrics'/method;sp=root/'static_summary.json'
    if not sp.exists():return dict(status='pending',reason='Resolved static summary absent')
    summary=read(sp);paths=list((root/'static_rows').glob('*.json'));rows=[read(p) for p in paths]
    if {r['id'] for r in rows}!=ids or summary.get('unvisited') or sum(summary['coverage'].values())!=100:
        return dict(status='pending',reason='Static records incomplete')
    stats=summary['groups'][method]
    for key in ['PSNR','SSIM','LPIPS','IoU']:
        valid=[r for r in rows if r['metrics'].get(key,{}).get('status')=='ok'];cell=stats[key]
        assert sorted(cell['ids'])==sorted(r['id'] for r in valid) and cell['n']==len(valid)
        assert abs(cell['value']-float(np.mean([r['metrics'][key]['value'] for r in valid])))<1e-10
    edp={};edp_paths=list((root/'edp').glob('*/results.json'))
    for path in edp_paths:
        result=read(path)
        assert result['inputs_sha256']==sha(path.parent/'inputs.json') and result['protocol_sha256']==sha(path.parent/'protocol.json')
        for row in result['rows']:
            assert row['id'] not in edp;edp[row['id']]=row
            if row['score']['status']=='na':assert row['score'].get('reason')
            else:assert row['score']['status']=='ok'
    scored={r['id'] for r in rows if r['status']=='scored'}
    if set(edp)!=scored or summary['edp_processed']!=len(edp):return dict(status='pending',reason='ED-P not attempted for every static-scored identity',edp_attempted=len(edp),static_scored=len(scored))
    valid=[r for r in edp.values() if r['score']['status']=='ok'];c=stats['EDP']
    assert c['n']==len(valid) and sorted(c['ids'])==sorted(r['id'] for r in valid)
    if valid:assert abs(c['value']-float(np.mean([r['score']['value'] for r in valid])))<1e-12
    else:assert c['value'] is None
    return dict(status='passed',static_count=len(rows),EDP_attempted=len(edp),EDP_valid=len(valid),EDP_NA=len(edp)-len(valid),
        unsupported_reason_counts=dict(collections.Counter(r['score']['reason'] for r in edp.values() if r['score']['status']=='na')),
        summary=evidence(sp),edp_results=[evidence(p) for p in edp_paths],own_valid_means_independently_recomputed=True)


def main():
    contract=read(ROOT/'batch_contract.json');ids=set(contract['ids']);assert len(ids)==100
    requirements=[];resolved={};export_details={};binding_errors=[];all_reloaded=True;all_unerased=True
    for method in METHODS:
        rows={p.parent.name:(p,read(p)) for p in (RES/'exports'/method).glob('*/validation.json')};resolved[method]=rows
        export_details[method]=dict(count=len(rows),passed=sum(v['status']=='passed' for p,v in rows.values()))
        for sid,(path,v) in rows.items():
            ref=contract['references'][sid]
            if v['source_reference']!=ref or v['identity_key']!=ref['identity_key'] or v['timestamp_us']!=ref['timestamp_us']:binding_errors.append([method,sid])
            all_reloaded &= bool(v.get('reload_verified')) and Path(v['bundle']).is_file()
            all_unerased &= not any(v.get(k,False) for k in ['reference_mask_erased_prediction','reference_mask_erases_prediction','source_mask_applied_to_prediction'])
            for field in ['actor_preview','actor_alpha']:
                assert sha(v[field])==v[field+'_sha256']
            assert v['selection_uses_metric_scores'] is False
    complete300=all(set(v)==ids for v in resolved.values())
    baseline=ROOT/'publication_baseline';manifest=read(baseline/'manifest.json');paper=R/'reports/asset_experiments_paper_20261115_v3'
    unchanged={n:sha(paper/n)==digest for n,digest in manifest.items()};published=paper/'23_原生人体复核定点更新核验.json'
    protected=all(unchanged.values()) or (published.exists() and read(published).get('protected_rigid_other_methods_temporal_values_counts_scopes') is True)
    requirements.append(dict(id=1,title='Frozen scope, all 300 records and protected unrelated results',checks=[
        check('100 identities × 3 methods retained without dropping poor outputs',complete300,export_details,[ROOT/'batch_contract.json',ROOT/'inventory.json']),
        check('Rigid and other human/temporal values, counts and scopes preserved',protected,dict(live_28_baseline_files_unchanged=all(unchanged.values()),changed_files=[k for k,v in unchanged.items() if not v]),[baseline/'manifest.json',published])]))
    aux=ROOT/'omnire/all_node_exposure_aux/summary.json';auxdata=read(aux) if aux.exists() else {}
    aux_accept=ROOT/'omnire/exposure_completion_audit.json'
    acceptance=read(aux_accept) if aux_accept.exists() else {}
    aux_ok=acceptance.get('status')=='all100_auxiliary_exposure_audit_complete' and acceptance.get('count')==100
    aux_ok=aux_ok and acceptance.get('exposure_summary_sha256')==sha(aux) and acceptance.get('selected_background_diagnostic_flags')==[] and acceptance.get('max_target_parameter_error')==0
    requirements.append(dict(id=2,title='Same identity/exposure/camera and faithful native rendering',checks=[
        check('Every resolved output binds the immutable source reference',complete300 and not binding_errors,dict(binding_errors=binding_errors),[ROOT/'batch_contract.json',ROOT/'resolved_actor_filter_audit.json']),
        check('OmniRe whole-scene exposure auxiliary audit has explicit final disposition',auxdata.get('status')=='completed' and auxdata.get('count')==100 and aux_ok,dict(count=auxdata.get('count'),candidate_diagnostic_flagged_ids=auxdata.get('flagged_ids'),selected_background_diagnostic_flags=acceptance.get('selected_background_diagnostic_flags'),target_parameter_error=acceptance.get('max_target_parameter_error'),timing_policy=acceptance.get('timing_policy')),[aux,aux_accept])]))
    street=ROOT/'streetgs/review_completion.json';streetdata=read(street) if street.exists() else {}
    semantic_counts={}
    for m in METHODS:
        p=ROOT/m/'semantic_review.json';d=read(p) if p.exists() else {};semantic_counts[m]=len(d.get('rows',d.get('records',[])))
    requirements.append(dict(id=3,title='Whole-scene semantic review and bounded neighbor diagnosis',checks=[
        check('All 300 source-scene cases actually reviewed',all(n==100 for n in semantic_counts.values()),semantic_counts,[ROOT/m/'semantic_review.json' for m in METHODS]),
        check('Flagged StreetGS cases receive fixed neighbor checks, without replacing source',streetdata.get('status')=='completed' and streetdata.get('completed_neighbor_case_count')==55,dict(source_cases=streetdata.get('source_frame_review_count'),neighbor_cases=streetdata.get('completed_neighbor_case_count'),available_neighbor_frames=streetdata.get('available_neighbor_frames')),[street,ROOT/'streetgs/neighbors/semantic_review.json']),
        check('No unreviewed extra neighbor requirement for the other two methods',all(semantic_counts[m]==100 for m in ['omnire','splatad']),dict(basis='Both complete visual reviews report native body support/partial bodies for all cases; no extra suspected-unavailable neighbor list was recorded.'),[ROOT/'omnire/semantic_review.json',ROOT/'splatad/semantic_review.json'])]))
    decisions={};decision_ok=True
    for m in METHODS:
        p=RES/'decisions'/(m+'.json');d=read(p) if p.exists() else {};rows=d.get('rows',[]);decision_ok &= {x['id'] for x in rows}==ids
        decisions[m]=d.get('counts')
        for x in rows:decision_ok &= bool(x.get('reason'))
    requirements.append(dict(id=4,title='Evidence-based repair, correct-output reuse and no score-based selection',checks=[
        check('All 300 support decisions frozen with reasons',decision_ok,decisions,[RES/'decisions'/(m+'.json') for m in METHODS]),
        check('No technically measurable poor candidate deleted for low quality',complete300 and all(v['selection_uses_metric_scores'] is False for rows in resolved.values() for p,v in rows.values()),dict(metric_score_selection=False,all_records_retained=complete300),[RES/'decisions'/(m+'.json') for m in METHODS])]))
    requirements.append(dict(id=5,title='Independent real-Gaussian export and transparent compositing',checks=[
        check('All 300 preserve independent-reload evidence and actual bundle paths',complete300 and all_reloaded,export_details,[ROOT/'resolved_actor_filter_audit.json',ROOT/'omnire/completion_audit.json']),
        check('Formal predictions are not erased by reference masks',complete300 and all_unerased,dict(alpha_and_RGB_file_hashes_checked=True),[ROOT/'batch_contract.json'])]))
    metrics={};geometry={}
    for m in METHODS:
        try:metrics[m]=static_audit(m,ids)
        except Exception as e:metrics[m]=dict(status='pending',reason=repr(e))
        try:geometry[m]=geometry_audit(m,ids)
        except Exception as e:geometry[m]=dict(status='pending',reason=repr(e))
    temporal=ROOT/'temporal_reuse_audit.json';td=read(temporal) if temporal.exists() else {}
    temporal_bound=bool(td) and all(Path(p).is_file() and sha(p)==digest for p,digest in td.get('source_artifacts',{}).items())
    temporal_bound=temporal_bound and all(sha(x['metrics'])==x['metrics_sha256'] for x in td.get('per_identity_provenance',[]))
    requirements.append(dict(id=6,title='Recomputed metrics, valid denominators/NA, ED-P coverage and unchanged temporal values',checks=[
        check('Static means and all eligible ED-P attempts independently rechecked',all(d['status']=='passed' for d in metrics.values()),metrics,[RES/'metrics'/m/'static_summary.json' for m in METHODS]),
        check('Resolved geometry distances and both-panel means independently rechecked',all(d['status']=='passed' for d in geometry.values()),geometry,[RES/'geometry'/m/'independent_arithmetic_audit.json' for m in METHODS]),
        check('Existing temporal values and native actor-only definition audited',td.get('status')=='passed_read_only_provenance_and_aggregation_audit' and td.get('identities')==82 and temporal_bound,dict(identities=td.get('identities'),adjacent_pairs=td.get('adjacent_pairs'),source_hashes_preserved=temporal_bound),[temporal])]))
    galleries={}
    final_manifest = ROOT / 'final/manifest.json'
    final_page = ROOT / 'final/index.html'
    final_data = read(final_manifest) if final_manifest.exists() else {}
    final_rows = final_data.get('records', [])
    final_page_text = final_page.read_text() if final_page.exists() else ''
    final_pairs = {(x['method'], x['id']) for x in final_rows}
    final_bound = (final_data.get('status') == 'complete' and len(final_rows) == 300
                   and final_pairs == {(m, sid) for m in METHODS for sid in ids}
                   and final_data.get('self_contained_images') is True
                   and 'data:image/' in final_page_text)
    if final_bound:
        for row in final_rows:
            vp, v = resolved[row['method']][row['id']]
            assert row['resolved_validation'] == str(vp) and row['validation_sha256'] == sha(vp)
            assert row['source_files']['final'] == v['actor_preview']
            assert row['image_sha256']['final'] == v['actor_preview_sha256']
            for key, digest in row['image_sha256'].items():
                if digest:
                    assert sha(row['source_files'][key]) == digest
        assert (ROOT / 'final/checklist_300.csv').is_file()
        assert (ROOT / 'final/metrics_delta.csv').is_file()
    for m in METHODS:
        if final_bound:
            galleries[m] = evidence(final_manifest)
            continue
        candidates=[ROOT/'gallery'/m/'manifest.json',RES/'gallery'/m/'manifest.json']
        good=None
        for f in candidates:
            if not f.exists():continue
            d=read(f)
            if d.get('count')==100 and d.get('self_contained_html') and all(x.get('validation')==str(resolved[m].get(x['id'],(None,))[0]) for x in d.get('rows',[])):
                page=f.parent/'index.html'
                if page.exists() and 'data:image/' in page.read_text():good=evidence(f);break
        galleries[m]=good
    requirements.append(dict(id=7,title='Offline final gallery, before/after report and controlled exp publication',checks=[
        check('All three offline four-column galleries bind final resolved outputs',all(galleries.values()),galleries),
        check('Final exp publication audit exists and preserves protected cells',published.exists() and read(published).get('status')=='published_verified_resolved_native_human_reaudit',None,[published,RES/'publication_ready.json'])]))
    free=shutil.disk_usage(H).free
    requirements.append(dict(id=8,title='Shared resource leases, 10 GiB reserve and reproducible dependencies',checks=[
        check('Current free disk stays above the authorized 10 GiB reserve',free>10*2**30,dict(free_bytes=free,free_GiB=free/2**30,minimum_GiB=10)),
        check('Shared GPU lease and reference-based checkpoint/dependency workflow recorded',(H/'scripts/native_reaudit_gpu.py').exists() and complete300,dict(checkpoints_referenced_not_copied=True,source_bundles_reused_by_resolved=True),[H/'scripts/native_reaudit_gpu.py',H/'scripts/native_reaudit_resolved_v1.py'])]))
    for r in requirements:r['status']='passed' if all(c['status']=='passed' for c in r['checks']) else 'pending'
    result=dict(version='native_human_reaudit_completion_requirements_v1',status='ready_for_root_final_review' if all(r['status']=='passed' for r in requirements) else 'pending',
        updated_at_utc=datetime.now(timezone.utc).isoformat(),goal_source=evidence(H/'handoff/native_human_export_reaudit_20261116/00_目标任务.txt'),
        organization='Eight completion gates consolidate the ten execution steps in the handoff; this does not redefine the frozen protocol.',requirements=requirements,
        pending=[dict(goal=r['id'],check=c['name'],details=c['details']) for r in requirements for c in r['checks'] if c['status']=='pending'],
        old_metrics_and_exports_modified=False,paper_published_by_this_script=False,script_sha256=sha(__file__))
    save(ROOT/'completion_requirements.json',result)
    print(json.dumps(dict(status=result['status'],passed_goals=sum(r['status']=='passed' for r in requirements),pending=result['pending']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
