"""Requirement-by-requirement experimental completion audit, before packaging."""
import csv,math,statistics,time
from html.parser import HTMLParser
from urllib.parse import unquote
from collections import Counter
from common import *
PANEL='annotation_vertical_extent_conditioned'
def csvread(p):return list(csv.DictReader(Path(p).open()))
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[];self.ids=set();self.anchors=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.add(d['id'])
        value=d.get('href') or d.get('src')
        if value:
            if value.startswith('#'):self.anchors.append(value[1:])
            elif '://' not in value:self.targets.append(unquote(value))
def main():
    active,selection_rows=selection();assert len({r['scene'] for r in selection_rows.values()})==67
    refs=read(E/'reference_manifest.json')['rows'];ref={r['id']:r for r in refs};ledger=read(E/'geometry_ledger.json')['rows'];assert len(ledger)==600 and {r['id'] for r in ledger}==set(selection_rows)
    assert all(Counter(r['method'] for r in ledger if r['id']==sid)==Counter(METHODS) for sid in selection_rows)
    protocol=read(E/'protocol.json');assert sha(E/'reference_manifest.json')==protocol['reference_manifest_sha256'];assert all(r['reference_qualified'] for r in refs)
    assert Counter(r['candidate_panel'] for r in refs)==Counter(main=84,low_support_exploratory=16)
    for r in refs:
        assert r['identity_key']==selection_rows[r['id']]['identity_key'] and r['t_ref_us']==selection_rows[r['id']]['t_ref_us'] and abs(r['delta_ms'])<=50
        checked(r['reference_rgb'],r['reference_rgb_sha256'])
        assert (r['candidate_panel']=='main')==(r['query_count']>=20)
    visual=read(E/'references/visual_review.json')['rows'];assert len(visual)==100
    for r in visual:assert r['model_scores_consulted'] is False and r['independent_human_blind_review'] is False;checked(r['evidence'],r['evidence_sha256'])
    audit=read(E/'validation/geometry_independent_audit.json');assert audit['status']=='passed' and audit['scored_rows_verified']==528 and audit['panel_results_verified']==711 and not audit['errors']
    for r in audit['rows']:checked(E/'results'/r['method']/(r['id']+'.json'),r['result_sha256'])
    evidence=read(E/'validation/geometry_evidence_audit.json');assert evidence['status']=='passed' and evidence['terminal_records']==600 and evidence['ledger_sha256']==sha(E/'geometry_ledger.json')
    results={};counts=Counter(r['geometry_status'] for r in ledger)
    assert counts==Counter(scored=527,empty_pred=1,NA_global_correspondence_unresolved=52,NA_native_stationary_filter=16,NA_native_actor_filtered_by_original_init=1,NA_native_generation_failure=3)
    for r in ledger:
        if r['geometry_status'] in ['scored','empty_pred']:
            checked(r['result'],r['result_sha256']);d=read(r['result']);assert d['started_at']>=protocol['frozen_at'];results[r['method'],r['id']]=d
        else:checked(r['evidence'],r['evidence_sha256'])
    recovery=csvread(E/'tables/registration_recovery.csv');assert len(recovery)==54 and sum(r['accepted']=='True' for r in recovery)==2
    supplement=read(E/'registration_recovery/protocol_supplement.json');assert supplement['score_values_used'] is False and supplement['all54_gaps_evaluated'];checked(supplement['artifact'],supplement['sha256'])
    scenes=read(E/'validation/missing_scene_audit_final.json')['records'];assert len(scenes)==17 and read(E/'validation/omnire_missing_reextraction.json')['status']=='NA_native_actor_filtered_by_original_init'
    for filename in ['cpu.json','voxel_analytic.json','pilot_vertical_reassessment.json']:assert read(E/'validation'/filename)['status']=='passed'
    assert len(protocol['pilots'])==6
    sets=read(E/'tables/common_and_paired_sets.json');stats=csvread(E/'tables/geometry_summary.csv')
    for population in ['main','low_support_exploratory']:
        valid={r['id'] for r in refs if r['candidate_panel']==population}
        for condition,methods in [(PANEL,METHODS),('native_metric',['omnire','splatad'])]:
            own={m:{sid for mm,sid in results if mm==m and sid in valid} for m in methods};prefix=population+'/'+condition;common=set.intersection(*own.values());finite={sid for sid in common if all(results[m,sid]['panels'][condition]['metrics']['cd_m'] is not None for m in methods)}
            assert set(sets[prefix+'/common_admission'])==common and set(sets[prefix+'/common_finite_cd'])==finite
            for m in methods:assert set(sets[prefix+'/own/'+m])==own[m]
            if condition==PANEL:
                for m in METHODS[1:]:
                    pair=own['ours']&own[m];fin={sid for sid in pair if all(results[mm,sid]['panels'][condition]['metrics']['cd_m'] is not None for mm in ['ours',m])}
                    assert set(sets[prefix+'/ours_vs_'+m+'/admission'])==pair and set(sets[prefix+'/ours_vs_'+m+'/finite_cd'])==fin
    for r in stats:
        prefix=r['reference_population']+'/'+r['condition'];key=prefix+'/own/'+r['method'] if r['set']=='own' else prefix+'/'+r['set'];ids=sets[key];metrics=[results[r['method'],sid]['panels'][r['condition']]['metrics'] for sid in ids];finite=[d['cd_m'] for d in metrics if d['cd_m'] is not None];empty=len(ids)-len(finite)
        assert int(r['n'])==len(ids) and int(r['n_finite'])==len(finite) and int(r['n_empty'])==empty
        if finite:near(r['cd_finite_mean_cm'],statistics.mean(finite)*100);near(r['cd_finite_median_cm'],statistics.median(finite)*100)
        else:assert r['cd_finite_mean_cm']==''
        if empty:assert r['cd_extended_mean_cm']=='inf'
        elif finite:near(r['cd_extended_mean_cm'],statistics.mean(finite)*100)
        for k in ['f1_5cm','f1_10cm','f1_20cm','precision_10cm','recall_10cm','ray_hit','ray_acc_5cm','ray_acc_10cm','ray_acc_20cm']:
            if metrics:near(r[k],statistics.mean(d[k] for d in metrics))
            else:assert r[k]==''
    timeaudit=read(E/'validation/time_audit.json');assert timeaudit['status']=='passed' and timeaudit['terminal_verified']==20 and all(r['status']=='passed' for r in timeaudit['rows'])
    for r in timeaudit['rows']:checked(r['record'],r['record_sha256'])
    tr=csvread(E/'tables/time_per_run.csv');assert len(tr)==20 and all(r['status']=='success' and r['independent_reload_status']=='passed' for r in tr)
    for s in csvread(E/'tables/time_summary.csv'):
        rows=[r for r in tr if r['method']==s['method']]
        if not rows:assert s['scope']=='not_applicable_scene_training_not_measured' and s['person_equal_weight_mean_seconds']=='';continue
        assert len(rows)==5 and len({r['id'] for r in rows})==4
        pm=[statistics.mean(float(r['seconds']) for r in rows if r['id']==sid) for sid in sorted({r['id'] for r in rows})]
        near(s['person_equal_weight_mean_seconds'],statistics.mean(pm));near(s['person_equal_weight_median_seconds'],statistics.median(pm))
    for r in tr:
        if r['method']=='ours':
            proof=read(r['independent_reload_record']);assert proof['status']=='passed' and proof['updates']==[32999,15000,5000] and proof['stage_intervals_verified']==17
    resource=read(E/'validation/time_resource_review.json');assert resource['minimum_free_gib']>10
    for p in (E/'state').glob('*.json'):
        d=read(p)
        if d.get('status')=='running':
            proc=Path('/proc')/str(d['pid'])/'stat'
            assert not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[19]!=str(d.get('start_ticks')),('owned worker still running',str(p))
    report=E/'report/详细实验报告.html';parser=Links();parser.feed(report.read_text());assert all(a in parser.ids for a in parser.anchors)
    for path in parser.targets:assert (report.parent/path).exists(),path
    for name in ['详细实验报告.txt','论文正文与附录.txt','复现说明.txt']:assert (E/'report'/name).stat().st_size>1000
    assert read(E/'report/report_status.json')['complete']
    text=(E/'report/详细实验报告.txt').read_text()
    for term in ['HN0035','HN0646','34.8415','44.879','52,999' if '52,999' in text else '52999','独立人工盲审','场景','低支持','native_metric','分阶段预加载','25次','72条']:assert term in text,term
    latex=list((E/'tables/latex').glob('*.tex'));assert len(latex)>=20
    for p in latex:
        t=p.read_text();assert t.count('\\begin{tabular}')==1 and t.count('\\end{tabular}')==1
    tasks=[('A01','精确100人/67场景/600行冻结入口','intake.json'),('A02','100组真实参考、视觉审查、84主/16低支持','reference_manifest.json'),('A03','全部原生来源与可用性，600行最终去向','tables/asset_inventory.csv'),('B01','原生Gaussian ED与T2多层alpha、解析验证','validation/voxel_analytic.json'),('B02','528条高度尺度/变换证书与6360支持图','validation/geometry_evidence_audit.json'),('B03','54条统一修正，2条恢复，52条有证据NA','tables/registration_recovery.csv'),('B04','17条场景缺项原生审计与有界重提取','validation/missing_scene_audit_final.json'),('C01','4个固定身份、6法pilot后冻结，补充源关系规则有记录','protocol.json'),('C02','528条可测及72条NA闭合，711个面板独立复算','validation/geometry_independent_audit.json'),('D01','元数据分层4人加锚点重复，20计划','time_plan.json'),('D02','20完整成功端点和独立重载，Ours完整52999更新','validation/time_audit.json'),('E01','全部共同/有限/配对/低支持/native与人物等权Time独立汇总复核','tables/geometry_summary.csv'),('E02','完整中文HTML/TXT、正文附录、CSV/LaTeX/图、复现；压缩包由package_results完成','report/详细实验报告.html')]
    rows=[dict(task_id=i,status='passed',verified_requirement=desc,evidence=str(E/p),evidence_sha256=sha(E/p)) for i,desc,p in tasks]
    csv_write(E/'tables/task_completion.csv',rows)
    proof=dict(status='passed',scope='all experimental requirements; final archive integrity separately in completion_audit.json',tasks=rows,geometry_terminal=600,measured_rows=528,finite_cd_rows=527,empty_prediction_rows=1,NA_rows=72,metric_panels=711,reference_main=84,reference_low=16,common_main_admission=46,common_main_finite=45,time_success=20,time_planned=20,independent_summary_rows_checked=len(stats),latex_tables=len(latex),report_local_links_checked=len(parser.targets),owned_background_workers_alive=False,script_sha256=sha(__file__),finished_at=time.time())
    save(E/'validation/experiment_completion_audit.json',proof);print({k:v for k,v in proof.items() if k!='tasks'})
if __name__=='__main__':main()
