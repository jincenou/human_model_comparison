"""Independent read-only checks; only an audit JSON is written."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,collections,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image

H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
P=H/'reports/native_human_export_reaudit_v1/postprocess_v1'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):return isinstance(x,(int,float)) and math.isfinite(x)
def avg(x):return math.fsum(x)/len(x) if x else None

def audit_outputs():
    protocol=read(P/'protocol.json');manifest=read(P/'inputs.json');errors=[];rows=[]
    assert sha(P/'inputs.json')==protocol['input_manifest_sha256']
    assert sha(protocol['script'])==protocol['script_sha256']
    assert len(manifest['records'])==300 and len({(r['method'],r['id']) for r in manifest['records']})==300
    for inp in manifest['records']:
        sid,method=inp['id'],inp['method'];rp=P/'outputs'/method/sid/'postprocess.json';r=read(rp)
        checked=[]
        for k in ['source_validation','original_rgb','original_alpha','static_record','prediction_unwarped','detector_record']:
            if sha(inp[k])!=inp[k+'_sha256']:errors.append(dict(method=method,id=sid,error='input hash changed',field=k))
        for k in ['mask','clean_rgb','clean_alpha']:
            if sha(r['outputs'][k])!=r['outputs'][k+'_sha256']:errors.append(dict(method=method,id=sid,error='output hash mismatch',field=k))
        maskbytes=np.asarray(Image.open(r['outputs']['mask']).convert('L'));m=maskbytes>127
        rgb=np.asarray(Image.open(inp['original_rgb']).convert('RGB'));alpha=np.asarray(Image.open(inp['original_alpha']).convert('L'))
        rgb2=np.asarray(Image.open(r['outputs']['clean_rgb']).convert('RGB'));alpha2=np.asarray(Image.open(r['outputs']['clean_alpha']).convert('L'))
        conditions=dict(binary_mask=set(np.unique(maskbytes)).issubset({0,255}),shapes_match=rgb.shape==rgb2.shape==(900,1600,3) and alpha.shape==alpha2.shape==m.shape==(900,1600),
            retained_rgb_exact=np.array_equal(rgb[m],rgb2[m]),retained_alpha_exact=np.array_equal(alpha[m],alpha2[m]),
            erased_rgb_white=bool(np.all(rgb2[~m]==255)),erased_alpha_zero=bool(np.all(alpha2[~m]==0)),
            alpha_never_increased=bool(np.all(alpha2<=alpha)))
        boxes=read(inp['detector_record'])['boxes'];idx=None;mapping_error=None
        if boxes:
            idx=min(range(len(boxes)),key=lambda i:(((boxes[i][0]+boxes[i][2])/2-256)**2+((boxes[i][1]+boxes[i][3])/2-256)**2,-boxes[i][4],i))
            x0,y0,x1,y1=inp['roi_xyxy'];b=boxes[idx];expected=[min(1600,max(0,x0+b[0]*(x1-x0)/512)),min(900,max(0,y0+b[1]*(y1-y0)/512)),min(1600,max(0,x0+b[2]*(x1-x0)/512)),min(900,max(0,y0+b[3]*(y1-y0)/512))]
            mapping_error=max(abs(x-y) for x,y in zip(expected,r['sam_box_raw_xyxy']))
            conditions['box_mapping_valid']=mapping_error<=2e-4 and r['detector_selected_index']==idx
        else:
            conditions['no_detector_empty_output']=not m.any() and not alpha2.any() and bool(np.all(rgb2==255)) and r['status']=='postprocess_no_person'
        for name,ok in conditions.items():
            if not ok:errors.append(dict(method=method,id=sid,error=name))
        if r['status']=='postprocess_exception_fallback':
            if not m.all():errors.append(dict(method=method,id=sid,error='exception fallback did not preserve input'))
        rows.append(dict(method=method,id=sid,status=r['status'],detector_boxes=len(boxes),mapping_max_error_px=mapping_error,
            foreground_nonempty=bool((alpha2>127).any()),checks=conditions,postprocess_record=str(rp),postprocess_record_sha256=sha(rp)))
    return dict(status='passed' if not errors else 'errors_found',count=len(rows),errors=errors,
        status_counts={m:dict(collections.Counter(r['status'] for r in rows if r['method']==m)) for m in protocol['methods']},
        blank_ids={m:[r['id'] for r in rows if r['method']==m and not r['foreground_nonempty']] for m in protocol['methods']},
        protocol_sha256=sha(P/'protocol.json'),input_manifest_sha256=sha(P/'inputs.json'),rows=rows,
        code_review=dict(reference_image_or_mask_pixels_used_for_erasure=False,coarse_reference_roi_used_for_mapping=True,
            fully_gt_free_claim_is_invalid=True,premultiplied_twice=False,geometry_inputs_unchanged=True,
            limitation='Only these3 native scene baselines receive this separate2D postprocess. Report it explicitly; no claim that all generation baselines or 3D geometry underwent the same cleanup.',
            no_person_interpretation='A detector/2D postprocess failure, not proof the original3D model generated no asset. White outputs must still contribute their actual image metrics, with IoU0; ED-P may be NA.'))

def audit_metrics(outputs):
    results={};errors=[]
    for method in ['streetgs','omnire','splatad']:
        base=P/'metrics'/method
        if not (base/'static_summary.json').exists():results[method]=dict(status='not_ready');continue
        s=read(base/'static_summary.json');rowfiles=list((base/'static_rows').glob('*.json'));rows=[read(x) for x in rowfiles]
        protocol_path=base/'static_protocol.json';protocol=read(protocol_path)
        code_hashes_match=all(sha(path)==digest for path,digest in protocol['code'].items())
        appearance_protocol_matches=sha(protocol['inherited_appearance_protocol'])==protocol['inherited_appearance_protocol_sha256']
        if not code_hashes_match or not appearance_protocol_matches:errors.append(dict(method=method,error='scoring code or inherited formula protocol hash mismatch'))
        edp_path=base/'edp/all/results.json';edprows=read(edp_path)['rows'] if edp_path.exists() else []
        checks={};person_checks=[]
        for r in edprows:
            vals=[p['distance'] for p in r['parts'].values() if finite(p.get('distance'))];expected=avg(vals);reported=r['score'].get('value')
            ok=(expected==reported if expected is None or reported is None else abs(expected-reported)<=1e-12)
            ok=ok and len(vals)==r['valid_part_count']
            if not ok:errors.append(dict(method=method,id=r['id'],error='EDP person mean'))
            person_checks.append(dict(id=r['id'],n_parts=len(vals),recomputed=expected,reported=reported,passed=ok))
        for metric,g in s.get('groups',{}).get(method,{}).items():
            if metric=='EDP':selected={r['id']:r['score']['value'] for r in edprows if r['score'].get('status')=='ok' and finite(r['score'].get('value'))}
            else:selected={r['id']:r['metrics'][metric]['value'] for r in rows if r.get('status')=='scored' and r['metrics'].get(metric,{}).get('status')=='ok' and finite(r['metrics'][metric].get('value'))}
            value=avg(list(selected.values()));reported=g.get('value');err=abs(value-reported) if value is not None and reported is not None else 0 if value==reported else None
            ok=len(selected)==g['n'] and set(selected)==set(g['ids']) and err is not None and err<=1e-12
            if not ok:errors.append(dict(method=method,metric=metric,error='overall mean/n/ids mismatch'))
            checks[metric]=dict(n=len(selected),reported_n=g['n'],mean=value,reported_mean=reported,absolute_error=err,passed=ok)
        blank_checks=[];rd={r['id']:r for r in rows}
        for sid in outputs['blank_ids'][method]:
            if sid not in rd:continue
            row=rd[sid];ok=row.get('status')=='scored' and row['metrics']['IoU']['value']==0 and row['metrics']['IoU_raw']['value']==0
            ok=ok and all(row['metrics'][k]['status']=='ok' and finite(row['metrics'][k]['value']) for k in ['PSNR','SSIM','LPIPS'])
            if not ok:errors.append(dict(method=method,id=sid,error='blank output was not retained/scored correctly'))
            blank_checks.append(dict(id=sid,passed=ok))
        results[method]=dict(status='complete' if len(rows)==100 and len(edprows)==100 else 'in_progress',static_rows=len(rows),edp_rows=len(edprows),metrics=checks,edp_person_checks=person_checks,blank_output_checks=blank_checks,
            scoring_code_hashes_match=code_hashes_match,inherited_appearance_protocol_matches=appearance_protocol_matches,
            scoring_protocol=str(protocol_path),scoring_protocol_sha256=sha(protocol_path))
    return dict(errors=errors,methods=results,aggregation='Arithmetic mean of eligible identity scores, after each person\u2019s own common valid parts are averaged. No zero-filling for ED-P NA, no parts-count weighting.')

def audit_publication(metrics):
    current=H.parent/'vehicle_model_comparison/reports/asset_experiments_paper_20261115_v3';baseline=P/'publication_baseline'
    filename='16_最终合并表_数值与范围.csv'
    old={(r['domain'],r['method'],r['metric']):r for r in csv.DictReader((baseline/filename).open())}
    new={(r['domain'],r['method'],r['metric']):r for r in csv.DictReader((current/filename).open())}
    allowed={('human',m,k) for m in ['SplatAD','OmniRe','StreetGS'] for k in ['PSNR','SSIM','LPIPS','IoU','ED-P']}
    errors=[];changed=[]
    if set(old)!=set(new):errors.append('Publication table row identities changed')
    for key in set(old)&set(new):
        if any(old[key][k]!=new[key][k] for k in ['value','n']):
            changed.append(dict(domain=key[0],method=key[1],metric=key[2],old_value=old[key]['value'],new_value=new[key]['value'],old_n=old[key]['n'],new_n=new[key]['n']))
            if key not in allowed:errors.append('Unapproved numeric cell changed: '+str(key))
        if key not in allowed:
            for k in ['value','n','scope','display_metric']:
                if old[key][k]!=new[key][k]:errors.append('Unapproved table field changed: '+str((key,k)))
    for domain,method,metric in sorted(allowed):
        numeric=metrics['methods'][method.lower()]['metrics']['EDP' if metric=='ED-P' else metric]
        row=new[domain,method,metric]
        if abs(float(row['value'])-numeric['mean'])>1e-12 or int(row['n'])!=numeric['n']:errors.append('Published value/n does not match independent aggregation: '+str((domain,method,metric)))
    unchanged_geometry={name:sha(current/name)==sha(baseline/name) for name in ['20_人体可观测几何补表.csv','20_人体可观测几何补表.md','20_人体可观测几何补表.tex']}
    if not all(unchanged_geometry.values()):errors.append('Geometry publication artifacts changed')
    # Strip formatting only, then compare all104 rendered table data cells.
    def tex_rows(path):
        out={}
        for line in path.read_text().splitlines():
            if ' & ' not in line:continue
            fields=line.split(' & ');method=fields[0].strip()
            if method not in ['Asset Harvester','TRELLIS','Hunyuan3D-2.1','ReconViaGen','SplatAD','OmniRe','StreetGS','Ours']:continue
            values=[]
            for cell in fields[1:]:
                cell=cell.replace('\\\\','').strip();cell=re.sub(r'\\(?:textbf|underline)\{([^{}]+)\}',r'\1',cell);values.append(cell)
            out[method]=values
        return out
    oldtex,newtex=tex_rows(baseline/'14_最终合并表.tex'),tex_rows(current/'14_最终合并表.tex');tex_changes=[]
    if set(oldtex)!=set(newtex):errors.append('Rendered TeX methods changed')
    for method in set(oldtex)&set(newtex):
        if len(oldtex[method])!=13 or len(newtex[method])!=13:errors.append('Rendered TeX column count changed');continue
        for i,(a,b) in enumerate(zip(oldtex[method],newtex[method])):
            if a!=b:
                tex_changes.append(dict(method=method,column_index_without_method=i,old=a,new=b))
                if method not in ['SplatAD','OmniRe','StreetGS'] or i not in range(6,11):errors.append('Unapproved rendered TeX numeric cell changed: '+str((method,i)))
    disclosure_text='\n'.join((current/x).read_text() for x in ['01_实验正文.tex','03_实验附录.tex','14_最终合并表.tex'])
    return dict(status='passed' if not errors else 'errors_found',errors=errors,changed_numeric_cells=changed,changed_numeric_cell_count=len(changed),
        rendered_tex_numeric_changes=tex_changes,allowed_cells=[list(x) for x in sorted(allowed)],geometry_artifacts_identical=unchanged_geometry,
        current_table=str(current/filename),current_table_sha256=sha(current/filename),baseline_table=str(baseline/filename),baseline_table_sha256=sha(baseline/filename),
        disclosure_mentions_sam2=('SAM2' in disclosure_text or 'SAM~2' in disclosure_text or 'SAM 2' in disclosure_text),
        publication_artifact_hashes={x:sha(current/x) for x in ['01_实验正文.tex','03_实验附录.tex','14_最终合并表.tex','24_人体二维前景后处理更新核验.json']},
        historical_native_edp_counts_explicitly_labeled_before_postprocessing='before image postprocessing' in (current/'03_实验附录.tex').read_text(),
        ranking_style_changes_allowed=True,other_numeric_cells_unchanged=not any('Unapproved' in x for x in errors))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publication',action='store_true');args=parser.parse_args()
    target=P/'independent_readonly_audit.json'
    if target.exists():
        previous=read(target);outputs=previous['output_contract']
        assert outputs['protocol_sha256']==sha(P/'protocol.json') and outputs['input_manifest_sha256']==sha(P/'inputs.json')
        assert all(sha(r['postprocess_record'])==r['postprocess_record_sha256'] for r in outputs['rows'])
    else:outputs=audit_outputs()
    metrics=audit_metrics(outputs)
    publication=audit_publication(metrics) if args.publication else dict(status='pending_publication_update',policy='Only3x5 PSNR/SSIM/LPIPS/IoU/ED-P value/n cells permitted; geometry/other methods/rigid/temporal unchanged.')
    scorer=H/'scripts/score_native_human_postprocess_v1.py'
    out=dict(output_contract=outputs,metrics=metrics,formal_outputs_modified=False,scores_modified=False,publication_check=publication,
        scoring_wrapper_review=dict(path=str(scorer),sha256=sha(scorer),unchanged_accepted_metric_functions_reused=True,
            two_dimensional_binding_explicitly_separate_from_native_3d_validation=True,
            white_composited_uint8_passed_without_second_alpha_multiplication=True,
            blank_prediction_identity_alignment_fallback_retained=True,
            ground_truth_used_for_existing_reference_and_alignment_only_not_prediction_pixel_erasure=True,
            prediction_pose_and_feature_cache_bound_to_new_image_and_mask_hashes=True))
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(dict(outputs=outputs['status'],output_count=outputs['count'],output_errors=outputs['errors'],metric_errors=metrics['errors'],metrics={k:v['status'] for k,v in metrics['methods'].items()}))

if __name__=='__main__':main()
