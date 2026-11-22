"""Isolated prediction-only 2D foreground publication, preserving native 3D results.

Default builds a candidate. --publish is only used after the root audit approval.
"""
import argparse
import copy
import csv
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import re
import shutil

import numpy as np

from native_reaudit_common import read,save,sha
import publish_streetgs_human_results_v1 as tabletools
from build_postprocess_human_gallery_v1 import checked_records,NOTE

H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1/postprocess_v1';BASE=ROOT/'publication_baseline'
PAPER=R/'reports/asset_experiments_paper_20261115_v3'
original=tabletools.original
METHODS={'splatad':'SplatAD','omnire':'OmniRe','streetgs':'StreetGS'}
KEYS={'PSNR':'PSNR','SSIM':'SSIM','LPIPS':'LPIPS','IoU':'IoU','ED-P':'EDP'}
ALLOW={('human',method,key) for method in METHODS.values() for key in KEYS}
FILES={'00_实验正文与合并表.html','01_实验正文.tex','02_合并主表.tex','03_实验附录.tex','05_写作依据与审校说明.txt',
       '14_最终合并表.tex','15_最终合并表.md','16_最终合并表_数值与范围.csv','17_最终合并表_来源核验.json'}
NOTE_EN=("For SplatAD, OmniRe, and StreetGS, one frozen prediction-only SAM2 foreground procedure is applied to each native RGB/alpha render "
 "before the five pedestrian image metrics (PSNR, SSIM, LPIPS, IoU, and ED-P). RTMDet prompts come from predicted renders within the existing "
 "annotated ROI; source-photo and reference-mask pixels are not used for segmentation. Kept RGB and alpha values remain unchanged, "
 "and removed pixels become white and transparent. No-person detections yield empty predictions retained in the image-metric denominator; "
 "inference exceptions use the fixed original-render fallback. This is 2D image postprocessing, not Gaussian or 3D geometry cleanup. "
 "All geometric and temporal measurements remain those of the native exports.")
TABLE_NOTE=(" For the three scene-reconstruction methods, the five pedestrian image columns include the same prediction-only SAM2 foreground "
 "postprocessing; the fixed ROI remains annotation-assisted. No-person outputs remain in the image denominators. Geometry and temporal columns retain native-export results.")
NATIVE_COVERAGE_TITLE='Native pedestrian export audit and metric coverage'
NATIVE_COVERAGE_NOTE=('The native-baseline counts in this subsection are retained for the pre-postprocessing audit only; '
 'the main table uses the postprocessed image-metric counts reported below.')


def baseline():
    manifest=read(BASE/'manifest.json');assert len(manifest)==29
    for name,digest in manifest.items():assert sha(BASE/name)==digest,('Changed frozen raw publication',name)
    proof=read(BASE/'17_最终合并表_来源核验.json');assert proof['status']=='published_native_human_reaudit_resolved_overlay'
    data={'rows':{},'display_selection_policy':copy.deepcopy(proof['display_selection_policy'])}
    for row in proof['rows']:
        data['rows'].setdefault(row['domain'],{}).setdefault(row['method'],{})[row['metric']]=dict(value=row['value'],n=row['n'],selection_scope=row['scope'])
    original.FOOTNOTE=proof['caption']
    _,md,tex,rows=tabletools.temporal_not_applicable(*original.render(data))
    assert rows==proof['rows'] and md==(BASE/'15_最终合并表.md').read_text() and tex==(BASE/'14_最终合并表.tex').read_text()
    return manifest,proof,data


def metrics(data):
    processed=checked_records();ids={r['id'] for r in processed};assert len(ids)==100
    completion=read(ROOT/'completion_audit.json');assert completion['status']=='passed' and completion['verified']==300
    assert completion['protocol_sha256']==sha(ROOT/'protocol.json')
    evidence={str(p):sha(p) for p in [ROOT/'protocol.json',ROOT/'inputs.json',ROOT/'completion_audit.json']}
    coverage={}
    for method,label in METHODS.items():
        folder=ROOT/'metrics'/method;sp=folder/'static_summary.json';summary=read(sp)
        assert summary['expected']==100 and summary['coverage']=={'scored':100} and not summary['unvisited'] and summary['edp_processed']==100
        stats=summary['groups'][method];rows={p.stem:read(p) for p in (folder/'static_rows').glob('*.json')}
        assert set(rows)==ids
        ep=folder/'edp/all/results.json';edp=read(ep);erows={r['id']:r for r in edp['rows']};assert set(erows)==ids
        assert edp['inputs_sha256']==sha(ep.parent/'inputs.json') and edp['protocol_sha256']==sha(ep.parent/'protocol.json')
        for key,raw in KEYS.items():
            c=copy.deepcopy(stats[raw]);assert len(c['ids'])==len(set(c['ids']))==c['n'] and set(c['ids'])<=ids
            if key=='ED-P':
                valid=[r for r in erows.values() if r['score']['status']=='ok'];values=[r['score']['value'] for r in valid]
                for r in erows.values():assert r['score']['status']=='ok' or (r['score']['status']=='na' and r['score'].get('reason'))
            else:
                valid=[r for r in rows.values() if r['metrics'][raw]['status']=='ok'];values=[r['metrics'][raw]['value'] for r in valid]
                assert len(valid)==100,'Blank predictions must remain in the image denominator'
            assert sorted(c['ids'])==sorted(r['id'] for r in valid)
            if values:assert abs(c['value']-float(np.mean(values)))<1e-10
            else:assert c['value'] is None
            c['selection_scope']=data['rows']['human'][label][key]['selection_scope']
            data['rows']['human'][label][key]=c
        coverage[method]=dict(image_n=100,EDP_attempted=100,EDP_valid=stats['EDP']['n'],EDP_NA=100-stats['EDP']['n'],postprocess_counts=completion['counts'][method])
        for p in [sp,folder/'static_protocol.json',ep,ep.parent/'inputs.json',ep.parent/'protocol.json']:evidence[str(p)]=sha(p)
    gallery=read(ROOT/'gallery/manifest.json');assert gallery['status']=='complete' and gallery['count']==300 and gallery['self_contained_html']
    assert gallery['protocol_sha256']==sha(ROOT/'protocol.json') and gallery['html_sha256']==sha(gallery['html'])
    evidence[str(ROOT/'gallery/manifest.json')]=sha(ROOT/'gallery/manifest.json')
    return evidence,coverage


def protect(before,after):
    key=lambda r:(r['domain'],r['method'],r['metric'])
    old={key(r):r for r in before};new={key(r):r for r in after};assert old.keys()==new.keys()
    for k,row in new.items():
        assert row['scope']==old[k]['scope'],('Scope drift',k)
        if k not in ALLOW:assert (row['value'],row['n'])==(old[k]['value'],old[k]['n']),('Protected cell changed',k)
    return [dict(key=list(k),raw=old[k],postprocessed=new[k]) for k in sorted(ALLOW)]


def comparison_report(changes,coverage):
    by={(r['key'][1],r['key'][2]):r for r in changes};rows=[];csvrows=[]
    for method,label in METHODS.items():
        cells=[]
        for key in KEYS:
            d=by[(label,key)];a,b=d['raw'],d['postprocessed'];fmt=lambda x:original.fmt(key,x) if x is not None else 'NA'
            cells.append(f"{fmt(a['value'])} → {fmt(b['value'])} ({a['n']}→{b['n']})")
            csvrows.append(dict(method=label,metric=key,raw_value=a['value'],raw_n=a['n'],postprocessed_value=b['value'],postprocessed_n=b['n'],scope=b['scope']))
        rows.append('| '+' | '.join([label,*cells])+' |')
    counts='；'.join(f"{METHODS[m]}：无人体检测 {d['postprocess_counts'].get('postprocess_no_person',0)}/100，ED-P {d['EDP_valid']}/100 有效" for m,d in coverage.items())
    report='# 人体原生图像与统一二维前景后处理对比\n\n'+NOTE+'\n\n表中格式为“原生 → 后处理（有效人数原→新）”，数值按各方法自身有效范围汇总。所有方法的前四项图像指标均保留100个对象，ED-P按实际部位支持计算。\n\n| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | ED-P ↓ |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'+counts+'。\n\n结果变化描述了图像前景分离及其评测影响，不能解释为三维形状、可驱动性或未观测表面变好。刚体、其他人体方法、全部CD/F-score和时序结果均保持原发布值。\n\n[全部300条源图／原生／后处理离线对照](gallery/index.html)\n'
    return report,tabletools.csv_text(csvrows)


def build():
    manifest,proof,data=baseline();evidence,coverage=metrics(data)
    original.FOOTNOTE=proof['caption']+TABLE_NOTE
    table,md,tex,records=tabletools.temporal_not_applicable(*original.render(data));changes=protect(proof['rows'],records)
    page=(BASE/'00_实验正文与合并表.html').read_text()
    old_heading='<h3>'+NATIVE_COVERAGE_TITLE+'</h3>'
    assert page.count(old_heading)==1
    page=page.replace(old_heading,'<h3>'+NATIVE_COVERAGE_TITLE+' (before image postprocessing)</h3><p>'+NATIVE_COVERAGE_NOTE+'</p>')
    page,n=re.subn(r'(<h2>Table 1 · Joint asset quality</h2>)<div class="tablewrap">.*?</table></div><div class="caption">.*?</div>',lambda m:m[1]+table,page,count=1,flags=re.S);assert n==1
    page,n=re.subn(r'(<h3>Exact effective sample counts</h3>).*?(</details>)',lambda m:m[1]+tabletools.count_paragraphs(records)+m[2],page,count=1,flags=re.S);assert n==1
    coverage_note=' '.join(f"{METHODS[m]}: 100 image-scored outputs, {d['postprocess_counts'].get('postprocess_no_person',0)} no-person blank outputs retained; ED-P {d['EDP_valid']}/100 supported identities." for m,d in coverage.items())
    addition='<div id="prediction-foreground-postprocess"><h3>Uniform prediction-only 2D foreground postprocessing</h3><p>'+html.escape(NOTE_EN)+'</p><p>'+html.escape(coverage_note)+'</p><p>'+html.escape(NOTE)+'</p></div>'
    assert '</main>' in page;page=page.replace('</main>',addition+'</main>')
    main=(BASE/'01_实验正文.tex').read_text();marker='\\input{02_合并主表.tex}'
    assert main.count(marker)==1;main=main.replace(marker,'\\paragraph{Prediction-image foreground postprocessing.} '+NOTE_EN+'\n\n'+marker)
    appendix=(BASE/'03_实验附录.tex').read_text()
    old_heading='\\subsection{'+NATIVE_COVERAGE_TITLE+'}'
    assert appendix.count(old_heading)==1
    appendix=appendix.replace(old_heading,'\\subsection{'+NATIVE_COVERAGE_TITLE+' (before image postprocessing)}\n'+NATIVE_COVERAGE_NOTE+'\n')
    appendix,n=re.subn(r'(\\subsection\{Exact effective sample counts\}).*?(?=\\(?:subsection|input)\{|\Z)',lambda m:m[1]+'\n\n'+tabletools.count_paragraphs(records,True)+'\n\n',appendix,count=1,flags=re.S);assert n==1
    appendix+='\n\\subsection{Uniform prediction-image foreground postprocessing}\n'+NOTE_EN+'\n\n'+coverage_note+'\n'
    writing=(BASE/'05_写作依据与审校说明.txt').read_text()+'\n\n本轮新增：'+NOTE+'\n'+coverage_note+'\n'
    updated=copy.deepcopy(proof);updated.update(status='published_prediction_2d_foreground_overlay',rows=records,caption=original.FOOTNOTE,
        prediction_foreground_baseline=str(BASE/'17_最终合并表_来源核验.json'),prediction_foreground_baseline_sha256=manifest['17_最终合并表_来源核验.json'],
        prediction_foreground_sources=evidence,prediction_foreground_changes=changes,protected_native_geometry_temporal_rigid_other_methods=True)
    artifacts={'00_实验正文与合并表.html':page,'01_实验正文.tex':main,'02_合并主表.tex':tex,'03_实验附录.tex':appendix,'05_写作依据与审校说明.txt':writing,
        '14_最终合并表.tex':tex,'15_最终合并表.md':md,'16_最终合并表_数值与范围.csv':tabletools.csv_text(records),'17_最终合并表_来源核验.json':tabletools.dumps(updated)}
    assert set(artifacts)==FILES
    report,csvreport=comparison_report(changes,coverage)
    return artifacts,report,csvreport,dict(status='candidate_verified_not_published',baseline_manifest_sha256=sha(BASE/'manifest.json'),baseline_files=manifest,
        protocol_sha256=sha(ROOT/'protocol.json'),source_hashes=evidence,changes=changes,allowed_cells=[list(x) for x in sorted(ALLOW)],coverage=coverage,
        protected_native_geometry_temporal_rigid_other_methods_values_counts=True,all_selection_scopes_unchanged=True,
        geometry_tables_byte_preserved=True,raw_native_artifacts_modified=False,prediction_only_2d_segmentation_not_3d_cleanup=True,
        source_script=str(Path(__file__).resolve()),source_script_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.selftest:
        _,proof,_=baseline();protect(proof['rows'],copy.deepcopy(proof['rows']))
        bad=copy.deepcopy(proof['rows']);target=next(x for x in bad if x['method']=='SplatAD' and x['metric']=='tOF');target['value']+=1
        try:protect(proof['rows'],bad)
        except AssertionError:pass
        else:raise AssertionError('Protected temporal mutation accepted')
        print('29-file current-native baseline exact; protected temporal mutation rejected.');return
    assert shutil.disk_usage(H).free>10*2**30
    artifacts,report,csvreport,audit=build();candidate=ROOT/'publication_candidate';candidate.mkdir(exist_ok=True)
    for name in audit['baseline_files']:
        if name in artifacts:(candidate/name).write_text(artifacts[name])
        else:shutil.copy2(BASE/name,candidate/name)
    audit['candidate_hashes']={name:sha(candidate/name) for name in audit['baseline_files']}
    assert all(audit['candidate_hashes'][name]==audit['baseline_files'][name] for name in set(audit['baseline_files'])-FILES)
    save(candidate/'24_人体二维前景后处理更新核验.json',audit)
    (ROOT/'raw_vs_clean_report.md').write_text(report);(ROOT/'raw_vs_clean.csv').write_text(csvreport)
    if a.publish:
        for name,digest in audit['baseline_files'].items():assert sha(PAPER/name)==digest,('Paper changed since current-native freeze',name)
        for path,digest in audit['source_hashes'].items():assert sha(path)==digest,('Source changed during this build',path)
        backup=PAPER/'history'/('before_prediction_foreground_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));backup.mkdir(parents=True)
        for name in FILES:shutil.copy2(PAPER/name,backup/name)
        for name in FILES:shutil.copy2(candidate/name,PAPER/name)
        audit.update(status='published_prediction_2d_foreground_overlay',backup=str(backup))
        save(PAPER/'24_人体二维前景后处理更新核验.json',audit)
    print(json.dumps(dict(candidate=str(candidate),published=a.publish,changed_image_cells=len(audit['changes']),geometry_tables_unchanged=True),ensure_ascii=False))


if __name__=='__main__':main()
