"""Read-only arithmetic and actor-replay audit; writes a separate evidence JSON."""
from human_asset_benchmark.paths import expand as _hmc_expand
import csv,hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image

H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1/resolved'
AP=R/'reports/source_appearance_alignment_20261115_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def mean(xs):return math.fsum(xs)/len(xs) if xs else None
def pixels(a,b,mode='RGB'):
    aa=np.array(Image.open(a).convert(mode)).astype(np.int16);bb=np.array(Image.open(b).convert(mode)).astype(np.int16)
    assert aa.shape==bb.shape,(a,b,aa.shape,bb.shape)
    d=np.abs(aa-bb)
    return dict(max_byte_difference=int(d.max()),mean_byte_difference=float(d.mean()),different_channels=int(np.count_nonzero(d)))

def main():
    active=read(H/'inputs/active_results.json');oldcore_path=H/active['core_summary'];oldedp_path=H/active['edp_root']
    oldcore={(x['method'],x['id']):x for x in read(oldcore_path)['rows']}
    oldedp={(x['method'],x['id']):x for x in read(oldedp_path/'results.json')['rows']}
    oldcases={x['key']:x for x in read(oldedp_path/'inputs.json')['cases']}
    oldappearance={(x['method'],x['id'],x['metric']):float(x['aligned']) for x in csv.DictReader((AP/'逐对象重算对照.csv').open()) if x['domain']=='human' and x['aligned']}
    errors=[];aggregate={};replay={};inputs={str(p):sha(p) for p in [oldcore_path,oldedp_path/'results.json',oldedp_path/'inputs.json',oldedp_path/'protocol.json',AP/'逐对象重算对照.csv']}
    for method in ['streetgs','omnire','splatad']:
        base=ROOT/'metrics'/method;summary_path=base/'static_summary.json';summary=read(summary_path);inputs[str(summary_path)]=sha(summary_path)
        rows={x.stem:read(x) for x in (base/'static_rows').glob('*.json')}
        for x in (base/'static_rows').glob('*.json'):inputs[str(x)]=sha(x)
        ep=base/'edp/all';edprows={x['id']:x for x in read(ep/'results.json')['rows']};inputs[str(ep/'results.json')]=sha(ep/'results.json')
        assert len(rows)==len(edprows)==100,(method,len(rows),len(edprows))
        edp_checks=[];all_parts=[]
        for sid,x in edprows.items():
            values=[v['distance'] for v in x['parts'].values() if finite(v.get('distance'))]
            part_ids=[k for k,v in x['parts'].items() if finite(v.get('distance'))];recalc=mean(values);reported=x['score'].get('value')
            difference=abs(recalc-reported) if recalc is not None and reported is not None else 0. if recalc==reported else None
            ok=(difference is not None and difference<=1e-12 and set(part_ids)==set(x['valid_parts']) and len(values)==x['valid_part_count'])
            if not ok:errors.append(dict(method=method,id=sid,error='EDP per-person part aggregation',recomputed=recalc,reported=reported))
            edp_checks.append(dict(id=sid,n_parts=len(values),recomputed_person_mean=recalc,reported_person_mean=reported,absolute_error=difference,passed=ok));all_parts.extend(values)
        checks={}
        for metric,reported in summary['groups'][method].items():
            if metric=='EDP':selected={sid:x['score']['value'] for sid,x in edprows.items() if x['score'].get('status')=='ok' and finite(x['score'].get('value'))}
            else:selected={sid:x['metrics'][metric]['value'] for sid,x in rows.items() if x.get('status')=='scored' and x['metrics'].get(metric,{}).get('status')=='ok' and finite(x['metrics'][metric].get('value'))}
            recomputed=mean(list(selected.values()));diff=abs(recomputed-reported['value']) if recomputed is not None else None
            ok=(len(selected)==reported['n'] and set(selected)==set(reported['ids']) and diff is not None and diff<=1e-12)
            if not ok:errors.append(dict(method=method,error='summary aggregation',metric=metric,n_recomputed=len(selected),n_reported=reported['n'],recomputed=recomputed,reported=reported['value']))
            checks[metric]=dict(n_recomputed=len(selected),n_reported=reported['n'],ids_match=set(selected)==set(reported['ids']),recomputed_mean=recomputed,reported_mean=reported['value'],absolute_error=diff,passed=ok)
        aggregate[method]=dict(rows=100,metrics=checks,edp_person_checks=edp_checks,
            edp_identity_equal_weight_mean=checks['EDP']['recomputed_mean'],edp_incorrect_part_weighted_mean_for_diagnostic_only=mean(all_parts),
            edp_aggregation='Unweighted mean over each eligible person after averaging that person\u2019s common valid parts; no weighting by number of parts.',
            edp_protocol_exactly_matches_active_old=read(ep/'protocol.json')==read(oldedp_path/'protocol.json'))
        if method=='streetgs':continue
        decisions=read(ROOT/'decisions'/f'{method}.json');ids=[x['id'] for x in decisions['rows'] if x['selected_support']=='native_actor_only']
        newcases={x['key']:x for x in read(ep/'inputs.json')['cases']};records=[]
        for sid in ids:
            x=rows[sid];old=oldcore[method,sid];appearance=read(AP/'rows'/f'human__{method}__{sid}.json');new_edp=edprows[sid];prev_edp=oldedp[method,sid]
            validation=read(x['source_validation']);oldresult=read(old['result']);delta={}
            for metric in ['PSNR','SSIM','LPIPS']:
                prev=oldappearance[method,sid,metric];curr=x['metrics'][metric]['value'];delta[metric]=dict(previous=prev,current=curr,signed_delta=curr-prev,absolute_delta=abs(curr-prev))
            for metric in ['IoU','IoU_raw']:
                prev=oldresult['metrics'][metric]['value'];curr=x['metrics'][metric]['value'];assert old['metrics'][metric]['value']==prev
                delta[metric]=dict(previous=prev,current=curr,signed_delta=curr-prev,absolute_delta=abs(curr-prev))
            prev=prev_edp['score'].get('value');curr=new_edp['score'].get('value');delta['EDP']=dict(previous=prev,current=curr,signed_delta=curr-prev if finite(prev) and finite(curr) else None,absolute_delta=abs(curr-prev) if finite(prev) and finite(curr) else None)
            key=f'{sid}_{method}';oc,nc=oldcases[key],newcases[key];changed=oc['image_sha256']!=nc['image_sha256']
            px=pixels(oc['image'],nc['image']) if changed else dict(max_byte_difference=0,mean_byte_difference=0.,different_channels=0)
            maskdiff=pixels(oc['mask'],nc['mask'],'L') if oc['mask_sha256']!=nc['mask_sha256'] else dict(max_byte_difference=0,mean_byte_difference=0.,different_channels=0)
            unchanged={k:x['reference'].get(k)==old['reference'].get(k) for k in ['identity_key','timestamp_us','reference_rgb_sha256','reference_mask_sha256','camera_contract_sha256','roi_xyxy']}
            transform_same=x['transform']==appearance['transform'];alpha_replay=validation.get('old_actor_replay',{})
            expected=(all(unchanged.values()) and transform_same and px['max_byte_difference']<=1 and maskdiff['max_byte_difference']==0 and all(delta[k]['absolute_delta']<=1e-4 for k in ['PSNR','SSIM','LPIPS']) and all(delta[k]['absolute_delta']==0 for k in ['IoU','IoU_raw']))
            if not expected:errors.append(dict(method=method,id=sid,error='actor-only replay exceeds known quantization envelope',reference=unchanged,transform_same=transform_same,pixel=px,mask=maskdiff,deltas=delta))
            rec=dict(id=sid,selected_support='native_actor_only',source_fields_unchanged=unchanged,
                appearance_transform_unchanged=transform_same,prediction_png_pixel_difference=px,prediction_binary_mask_difference=maskdiff,
                native_replay_record=alpha_replay,metrics=delta,old_common_parts=prev_edp['valid_parts'],new_common_parts=new_edp['valid_parts'],
                common_parts_changed=prev_edp['valid_parts']!=new_edp['valid_parts'],edp_status_unchanged=prev_edp['score']['status']==new_edp['score']['status'],
                static_source=str(base/'static_rows'/f'{sid}.json'),previous_appearance_source=str(AP/'rows'/f'human__{method}__{sid}.json'),
                previous_core_result=old['result'],previous_edp_input=oc,current_edp_input=nc,
                conclusion='No export recovery benefit: reused actor. Differences, if present, are quantized PNG/detector sensitivity; no old/new score selection.' if changed else 'Reused actor and identical image/mask inputs; metrics replay matches.')
            if sid=='HN0004' and method=='omnire':
                oldpose_path=H/'reports/source_edp_common_v1/cohort_v1/poses/HN0004_omnire.json';newpose_path=ep/'poses/HN0004_omnire.json';op,np_=read(oldpose_path),read(newpose_path)
                rec['EDP_threshold_evidence']=dict(joint='COCO left knee, index13',threshold=.3,previous_confidence=op['keypoints'][13][2],current_confidence=np_['keypoints'][13][2],
                    previous_parts=prev_edp['valid_parts'],current_parts=new_edp['valid_parts'],old_pose=str(oldpose_path),new_pose=str(newpose_path),
                    explanation='A <=1-byte PNG perturbation changes detector/crop and left-knee confidence from just above0.3 to just below0.3. The left-leg part loses valid support; ED-P is averaged over2 rather than3 common parts. This is estimator threshold sensitivity, not a new asset-quality gain.')
            records.append(rec)
        replay[method]=dict(count=len(ids),max_absolute_metric_deltas={k:max((v['metrics'][k]['absolute_delta'] or 0) for v in records) for k in ['PSNR','SSIM','LPIPS','IoU','IoU_raw','EDP']},
            nonidentical_prediction_png_ids=[v['id'] for v in records if v['prediction_png_pixel_difference']['max_byte_difference']>0],
            edp_changed_ids=[v['id'] for v in records if (v['metrics']['EDP']['absolute_delta'] or 0)>0],
            edp_common_part_changed_ids=[v['id'] for v in records if v['common_parts_changed']],rows=records)
    assert all(sha(p)==expected for p,expected in inputs.items()),'Audited source changed while reading'
    out=dict(status='passed_with_documented_quantization_and_EDP_threshold_sensitivity' if not errors else 'errors_found',
        created_utc=datetime.now(timezone.utc).isoformat(),scope='All300 new static rows and three EDP result sets; actor-only before/after replay for89 OmniRe and80 SplatAD identities.',
        errors=errors,aggregate=aggregate,actor_only_replay=replay,source_sha256=inputs,
        scoring_outputs_modified=False,model_outputs_modified=False,rerender_or_rescore_performed=False,
        findings=['All comparisons use per-identity records, not aggregate means with changed denominators.',
          'PSNR/SSIM/LPIPS follow the existing source-area/centroid alignment. IoU and ED-P use unwarped common ROI; no metric-specific new pose or mask formula.',
          'No per-part weighting was used for ED-P overall means. NA identities are excluded from that metric denominator, with records retained.',
          'The5 OmniRe actor-only image perturbations are replay numerical sensitivity, not benefits from background recovery. No repeat tuning or preferred-score substitution is warranted.'])
    target=ROOT/'static_numeric_replay_audit.json';target.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=out['status'],errors=errors,replay={m:{k:v for k,v in x.items() if k!='rows'} for m,x in replay.items()},aggregation={m:{k:v for k,v in x['metrics'].items()} for m,x in aggregate.items()}),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
