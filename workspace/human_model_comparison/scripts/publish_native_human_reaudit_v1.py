"""Build an exact-whitelist paper update from the immutable current baseline.

No historical raw-source fallback. Default writes an isolated candidate only.
Publication requires the resolved ledger, settled scores, and --publish.
"""
import argparse
import copy
import csv
from datetime import datetime,timezone
import html
import json
import math
from pathlib import Path
import re
import shutil
import sys

H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1';BASE=ROOT/'publication_baseline'
RESOLVED=ROOT/'resolved';PAPER=R/'reports/asset_experiments_paper_20261115_v3'
sys.path.insert(0,str(H/'scripts'))
import publish_streetgs_human_results_v1 as tabletools
from native_reaudit_common import read,save,sha
original=tabletools.original
METHODS={'streetgs':'StreetGS','splatad':'SplatAD','omnire':'OmniRe'}
STATIC_KEYS={'PSNR':'PSNR','SSIM':'SSIM','LPIPS':'LPIPS','IoU':'IoU','ED-P':'EDP'}
ALLOWED_MAIN={('human',m,k) for m in METHODS.values() for k in STATIC_KEYS}
ALLOWED_GEOMETRY={(m,k) for m in METHODS.values() for k in ['CD','F1']}
CHANGED_FILES={'00_实验正文与合并表.html','01_实验正文.tex','02_合并主表.tex','03_实验附录.tex','05_写作依据与审校说明.txt',
 '14_最终合并表.tex','15_最终合并表.md','16_最终合并表_数值与范围.csv','17_最终合并表_来源核验.json',
 '20_人体可观测几何补表.tex','20_人体可观测几何补表.md','20_人体可观测几何补表.csv'}

EXTRACTION=("The pedestrian comparison includes the same eight methods. For SplatAD, OmniRe, and StreetGS, "
 "a per-identity native-render audit retains verified existing target exports and supplements only confirmed omitted body support "
 "with original Background Gaussians having strictly positive visible contribution inside the frozen source mask, accounting for full-scene occlusion. "
 "The mask supplies additional target-localization information; it does not erase prediction pixels. Original Gaussian parameters, "
 "directional appearance, native decoding, and coordinate transforms are retained and independently reloaded. "
 "When no native target actor exists, a source-mask-visible Background candidate is retained with explicit unreconstructed-target or residual-appearance limitations; this is not automatically classified as recovered body support. "
 "Background texture and motion trails may remain, so technical measurability does not establish recovery of a complete person. "
 "These StreetGS checkpoints contain no native pedestrian node or learned pedestrian trajectory; its evaluation remains static-only.")
EXTRACTION_CN=('人体比较同样包含8种方法。对SplatAD、OmniRe和StreetGS逐对象核查原生场景，复用已正确的目标导出；'
 '仅对确认遗漏的人体支撑，使用冻结源mask中具有严格正可见贡献的原始背景高斯补提，依据完整场景遮挡计算可见贡献。'
 '没有原生人体节点时，保留源mask可见的背景候选，并明确记录目标未重建或残影，不能自动称为恢复了人体。'
 'mask提供额外目标定位信息，不用于擦除预测图像。高斯参数、方向外观、原生解码与坐标变换均保持原样并独立重载。'
 '仍可能混入背景和运动残影，技术可测数量不表示完整人体重建成功；StreetGS本批checkpoint没有原生人体节点和学得轨迹，只评测静态质量。')
TEMPORAL_NOTE=('The temporal values and their frozen 82-identity population are unchanged. SplatAD and OmniRe temporal rendering uses '
 'the previously audited native dynamic actor representation; static Background supplements are not assigned fabricated body motion. '
 'Thus the static extraction definition and actor-only temporal definition are explicitly distinguished.')
GEOMETRY_NOTE=('Pedestrian observable-surface geometry uses the existing 84-identity main LiDAR-support panel and each method\'s own valid outputs. '
 'The inherited annotated world-vertical-extent conditioning, query rays, alpha threshold, and strict 10 cm F-score threshold are unchanged. '
 'CD is in cm over finite distances; F-score is a percentage. Valid empty predictions contribute F-score zero and non-finite CD. '
 'For background-inclusive exports, full-package height can include unrelated structures; height conditioning also scales camera-relative '
 'translation and can therefore induce metre-scale radial displacement. Such errors reflect the interaction of extraction support and '
 'conditioning, and cannot be attributed solely to reconstructed human geometry. Native-metric results are retained as separate diagnostics '
 'and do not silently replace the conditioned comparison. Bold and underline mark the best and second distinct means; populations differ.')


def baseline():
    manifest=read(BASE/'manifest.json');assert len(manifest)==28
    for name,digest in manifest.items():assert sha(BASE/name)==digest,('Changed publication baseline',name)
    proof=read(BASE/'17_最终合并表_来源核验.json');data={'rows':{},'display_selection_policy':copy.deepcopy(proof['display_selection_policy'])}
    for row in proof['rows']:
        data['rows'].setdefault(row['domain'],{}).setdefault(row['method'],{})[row['metric']]=dict(value=row['value'],n=row['n'],selection_scope=row['scope'])
    rendered=tabletools.temporal_not_applicable(*original.render(data))
    assert rendered[3]==proof['rows']
    assert rendered[1]==(BASE/'15_最终合并表.md').read_text()
    assert rendered[2]==(BASE/'14_最终合并表.tex').read_text()
    assert original.FOOTNOTE==proof['caption']
    geom=list(csv.DictReader((BASE/'20_人体可观测几何补表.csv').open()))
    for row in geom:
        row['value']=float(row['value']) if row['value'] else None;row['n']=int(row['n'])
        data['rows']['human'][row['method']][row['metric']]=dict(value=row['value'],n=row['n'],selection_scope=row['scope'])
    return manifest,proof,data,geom


def checked_cell(cell,ids):
    assert isinstance(cell['n'],int) and 0<=cell['n']<=100
    assert len(cell['ids'])==len(set(cell['ids']))==cell['n'] and set(cell['ids'])<=ids
    assert (cell['value'] is None)==(cell['n']==0)
    if cell['value'] is not None:assert math.isfinite(cell['value'])
    return copy.deepcopy(cell)


def sources(data):
    contract=read(ROOT/'batch_contract.json');ids=set(contract['references']);assert len(ids)==100
    ready=RESOLVED/'publication_ready.json'
    assert ready.exists(),'Awaiting root-owned resolved/publication_ready.json; raw F>0 candidates cannot be published'
    settlement=read(ready);assert settlement['status']=='ready',settlement.get('status')
    assert settlement.get('frozen_id_count')==100
    evidence={str(ready):sha(ready),str(ROOT/'batch_contract.json'):sha(ROOT/'batch_contract.json')}
    geo={};coverage={}
    for method,label in METHODS.items():
        sp=RESOLVED/'metrics'/method/'static_summary.json';gp=RESOLVED/'geometry'/method/'summary.json'
        for path in [sp,gp]:
            assert path.exists(),str(path)
            assert settlement['sources'][str(path)]==sha(path),('Resolved source not pinned by readiness ledger',str(path))
            evidence[str(path)]=sha(path)
        static=read(sp);geometry=read(gp)
        assert static['expected']==100 and not static.get('unvisited',[]) and sum(static['coverage'].values())==100
        assert not any('running' in k or 'pending' in k or 'waiting' in k for k in static['coverage'])
        assert static['edp_processed']==static['coverage'].get('scored',0)
        assert geometry['status'] in ['completed','completed_with_recorded_failures'] and geometry['expected']==100
        assert sum(geometry['coverage'].values())==100
        assert not any('pending' in k or 'running' in k or 'waiting' in k for k in geometry['coverage'])
        stats=static['groups'][method]
        for key,sourcekey in STATIC_KEYS.items():
            cell=checked_cell(stats[sourcekey],ids);cell['selection_scope']='method_own_valid'
            data['rows']['human'][label][key]=cell
        groups=geometry['groups']['main'];geo[label]=copy.deepcopy(groups)
        for key,raw in [('CD','CD_cm'),('F1','F10')]:data['rows']['human'][label][key]=checked_cell(groups['annotation_vertical_extent_conditioned'][raw],ids)
        coverage[label]=dict(static=static['coverage'],edp_attempted=static['edp_processed'],edp_valid=stats['EDP']['n'],geometry=geometry['coverage'])
    for name in ['temporal_reuse_audit.json','streetgs/semantic_review.json','streetgs/neighbors/semantic_review.json']:
        path=ROOT/name;assert path.exists();evidence[str(path)]=sha(path)
    return geo,coverage,evidence,settlement


def geometry_render(data,geo):
    records=[];hr=[];mr=[];tr=[]
    ranks={key:sorted({data['rows']['human'][m][key]['value'] for m in original.ORDER if data['rows']['human'][m][key]['value'] is not None},reverse=key=='F1') for key in ['CD','F1']}
    for method in original.ORDER:
        cells=[]
        for key in ['CD','F1']:
            c=data['rows']['human'][method][key];v=c['value'];rank=ranks[key].index(v)+1 if v is not None else None;t=original.fmt(key,v) if v is not None else '◇'
            records.append(dict(method=method,metric=key,value=v,n=c['n'],rank=rank,scope='method_own_valid_main_reference_support'))
            cells.append((t,rank))
        n='/'.join(str(data['rows']['human'][method][key]['n']) for key in ['CD','F1']);label=original.LABEL.get(method,method)
        ht=['<strong>'+t+'</strong>' if k==1 else '<u>'+t+'</u>' if k==2 else t for t,k in cells]
        mt=['**'+t+'**' if k==1 else '<u>'+t+'</u>' if k==2 else t for t,k in cells]
        tt=[r'\textbf{'+t+'}' if k==1 else r'\underline{'+t+'}' if k==2 else r'$\diamond$' if t=='◇' else t for t,k in cells]
        hr.append('<tr><th>'+label+'</th>'+''.join('<td>'+x+'</td>' for x in [*ht,n])+'</tr>')
        mr.append('| '+' | '.join([label,*mt,n])+' |');tr.append(' & '.join([label,*tt,n])+r' \\')
    supplements=[]
    for method in METHODS.values():
        d=geo[method]['native_metric'];c=d['CD_cm'];f=d['F10']
        supplements.append(f"{method}: native-metric CD {original.fmt('CD',c['value']) if c['value'] is not None else 'NA'} cm (n={c['n']}), F-score {original.fmt('F1',f['value']) if f['value'] is not None else 'NA'}% (n={f['n']}).")
    caption=GEOMETRY_NOTE+' '+' '.join(supplements)
    ht='<div id="streetgs-human-geometry"><h3>Supplementary pedestrian observable geometry</h3><div class="tablewrap"><table style="min-width:620px"><thead><tr><th>Method</th><th>CD ↓</th><th>F-score ↑</th><th>n (CD/F-score)</th></tr></thead><tbody>'+''.join(hr)+'</tbody></table></div><p class="caption">'+html.escape(caption)+'</p></div>'
    md='| Method | CD ↓ | F-score ↑ | n (CD/F-score) |\n|---|---:|---:|---:|\n'+'\n'.join(mr)+'\n\n'+caption+'\n'
    tex='\n'.join([r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Method & CD$\downarrow$ & F-score$\uparrow$ & $n$ (CD/F-score) \\',r'\midrule',*tr,r'\bottomrule',r'\end{tabular}',r'\caption{'+caption.replace('%',r'\%')+'}',r'\label{tab:human_observable_geometry}',r'\end{table}'])+'\n'
    return ht,md,tex,records


def whitelist(before,after,allowed,keys):
    old={tuple(r[k] for k in keys):r for r in before};new={tuple(r[k] for k in keys):r for r in after};assert set(old)==set(new)
    for key,row in new.items():
        if key not in allowed:assert all(row[x]==old[key][x] for x in ['value','n','scope']),('Protected metric changed',key)
    return [dict(key=list(k),before=old[k],after=new[k]) for k in sorted(allowed) if any(old[k][x]!=new[k][x] for x in ['value','n','scope'])]


def amend(text):
    assert tabletools.BASELINE_SENTENCE in text or tabletools.CHINESE_SENTENCE in text
    return text.replace(tabletools.BASELINE_SENTENCE,EXTRACTION).replace(tabletools.CHINESE_SENTENCE,EXTRACTION_CN)


def build():
    manifest,proof,data,oldgeo=baseline();policy=copy.deepcopy(data['display_selection_policy'])
    geo,coverage,evidence,settlement=sources(data)
    table,md,tex,records=tabletools.temporal_not_applicable(*original.render(data))
    changes=whitelist(proof['rows'],records,ALLOWED_MAIN,['domain','method','metric'])
    gh,gm,gt,gr=geometry_render(data,geo)
    geometry_changes=whitelist(oldgeo,gr,ALLOWED_GEOMETRY,['method','metric'])
    assert data['display_selection_policy']==policy and original.FOOTNOTE==proof['caption']
    notes=[]
    for method,c in coverage.items():notes.append(f"{method}: {c['static'].get('scored',0)} static-scored identities; ED-P attempted for {c['edp_attempted']}, valid common-part support for {c['edp_valid']}. Unsupported ED-P records remain NA, not zero.")
    coverage_note=' '.join(notes)+' '+TEMPORAL_NOTE+' A technically reloadable candidate is not evidence that the complete target person was recovered.'
    coverage_html='<div id="streetgs-human-coverage"><h3>Native pedestrian export audit and metric coverage</h3><p>'+html.escape(coverage_note)+'</p></div>'
    webpage=amend((BASE/'00_实验正文与合并表.html').read_text())
    webpage,n=re.subn(r'(<h2>Table 1 · Joint asset quality</h2>)<div class="tablewrap">.*?</table></div><div class="caption">.*?</div>',lambda m:m[1]+table,webpage,count=1,flags=re.S);assert n==1
    webpage,n=re.subn(r'(<h3>Exact effective sample counts</h3>).*?(</details>)',lambda m:m[1]+tabletools.count_paragraphs(records)+m[2],webpage,count=1,flags=re.S);assert n==1
    webpage,n=re.subn(r'<div id="streetgs-human-geometry">.*?</p></div>',lambda m:gh,webpage,count=1,flags=re.S);assert n==1
    webpage,n=re.subn(r'<div id="streetgs-human-coverage">.*?</p></div>',lambda m:coverage_html,webpage,count=1,flags=re.S);assert n==1
    main=amend((BASE/'01_实验正文.tex').read_text());appendix=amend((BASE/'03_实验附录.tex').read_text())
    writing=(BASE/'05_写作依据与审校说明.txt').read_text()
    oldnote='StreetGS已增加GT人体框及各半轴0.25 m扩展辅助的静态背景高斯子集提取，提供了额外定位信息；通过独立重载后计分，但不声称提取的是完整人体。无原生人体轨迹，因此时序不适用。'
    assert writing.count(oldnote)==1
    writing=writing.replace(oldnote,EXTRACTION_CN)
    appendix,n=re.subn(r'(\\subsection\{Exact effective sample counts\}).*?(?=\\(?:subsection|input)\{|\Z)',lambda m:m[1]+'\n\n'+tabletools.count_paragraphs(records,True)+'\n\n',appendix,count=1,flags=re.S);assert n==1
    replacement=r'\subsection{Native pedestrian export audit and metric coverage}'+'\n'+coverage_note+'\n\n'
    appendix,n=re.subn(r'\\subsection\{StreetGS pedestrian extraction and metric coverage\}.*?(?=\\(?:subsection|input)\{|\Z)',lambda m:replacement,appendix,count=1,flags=re.S);assert n==1
    updated_proof=copy.deepcopy(proof);updated_proof.update(status='published_native_human_reaudit_resolved_overlay',rows=records,
        native_human_reaudit_baseline=str(BASE/'17_最终合并表_来源核验.json'),native_human_reaudit_baseline_sha256=manifest['17_最终合并表_来源核验.json'],
        native_human_reaudit_sources=evidence,native_human_reaudit_changes=changes,unrelated_values_counts_scopes_preserved=True)
    artifacts={'00_实验正文与合并表.html':webpage,'01_实验正文.tex':main,'02_合并主表.tex':tex,'03_实验附录.tex':appendix,'05_写作依据与审校说明.txt':writing,
       '14_最终合并表.tex':tex,'15_最终合并表.md':md,'16_最终合并表_数值与范围.csv':tabletools.csv_text(records),
       '17_最终合并表_来源核验.json':tabletools.dumps(updated_proof),'20_人体可观测几何补表.tex':gt,'20_人体可观测几何补表.md':gm,'20_人体可观测几何补表.csv':tabletools.csv_text(gr)}
    assert set(artifacts)==CHANGED_FILES
    return artifacts,dict(status='passed_candidate_whitelist',baseline_manifest_sha256=sha(BASE/'manifest.json'),baseline_files=manifest,
        main_cell_whitelist=[list(x) for x in sorted(ALLOWED_MAIN)],geometry_cell_whitelist=[list(x) for x in sorted(ALLOWED_GEOMETRY)],
        changes=changes,geometry_changes=geometry_changes,source_hashes=evidence,coverage=coverage,
        protected_rigid_other_methods_temporal_values_counts_scopes=True,table_caption_unchanged=True,
        display_selection_policy_unchanged=True,source_script=str(Path(__file__).resolve()),source_script_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.selftest:
        _,proof,data,geom=baseline();assert whitelist(proof['rows'],copy.deepcopy(proof['rows']),ALLOWED_MAIN,['domain','method','metric'])==[]
        changed=copy.deepcopy(proof['rows']);changed[0]['value']+=1
        try:whitelist(proof['rows'],changed,ALLOWED_MAIN,['domain','method','metric'])
        except AssertionError:pass
        else:raise AssertionError('Whitelist did not reject protected rigid mutation')
        print('Baseline exact reconstruction and protected-cell rejection passed; no paper writes.');return
    artifacts,audit=build();candidate=ROOT/'publication_candidate';candidate.mkdir(exist_ok=True)
    for name in audit['baseline_files']:
        target=candidate/name
        if name in artifacts:target.write_text(artifacts[name])
        else:shutil.copy2(BASE/name,target)
    audit['candidate_hashes']={name:sha(candidate/name) for name in audit['baseline_files']}
    for name in set(audit['baseline_files'])-CHANGED_FILES:assert audit['candidate_hashes'][name]==audit['baseline_files'][name]
    save(candidate/'23_原生人体复核定点更新核验.json',audit)
    if a.publish:
        for name,digest in audit['baseline_files'].items():assert sha(PAPER/name)==digest,('Live paper changed since frozen baseline',name)
        for path,digest in audit['source_hashes'].items():assert sha(path)==digest,('Resolved source changed',path)
        backup=PAPER/'history'/('before_native_human_reaudit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));backup.mkdir(parents=True)
        for name in CHANGED_FILES:shutil.copy2(PAPER/name,backup/name)
        for name in CHANGED_FILES:shutil.copy2(candidate/name,PAPER/name)
        audit.update(status='published_verified_resolved_native_human_reaudit',backup=str(backup))
        save(PAPER/'23_原生人体复核定点更新核验.json',audit)
    print(json.dumps(dict(candidate=str(candidate),published=a.publish,main_changed=len(audit['changes']),geometry_changed=len(audit['geometry_changes'])),ensure_ascii=False))


if __name__=='__main__':main()
