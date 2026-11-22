"""Export paper tables from active results; recheck means against per-person rows."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import re
import shutil
import statistics

H = Path(__file__).resolve().parents[1]
OUT = H/'reports/paper_experiment_report_20261113'
NAMES = {'ours':'Ours','ah':'AH','trellis_original':'TRELLIS','trellis2':'TRELLIS.2','omnire':'OmniRe','splatad':'SplatAD'}
ORDER = list(NAMES)
CORE = ['PSNR','SSIM','LPIPS','IoU','IoU_raw']
SOURCES = {}


def content(path):
    path = Path(path)
    if not path.is_absolute(): path = H/path
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(H))] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def read(path):
    return json.loads(content(path))


def rows(path):
    return list(csv.DictReader(content(path).splitlines()))


def near(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(a,b)


def table(headers, records, best=()):
    extremes={i:(min if direction=='min' else max)(float(r[i])for r in records if r[i]!='')for i,direction in best}
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for r in records:
        cells=[]
        for i,value in enumerate(r):
            v=f'{value:.4f}' if isinstance(value,float) else str(value)
            if i in extremes and value!='' and float(value)==extremes[i]: v='**'+v+'**'
            cells.append(v)
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


active=read('inputs/active_results.json')
summary=read(active['core_summary'])
selection=read('inputs/active_human_benchmark.json')
assert summary['selection_sha256']==selection['selection_sha256']
core_rows={(r['method'],r['id']):r for r in summary['rows']}
for pairing in summary['pairings'].values():
    for metric,item in pairing.items():
        for method,score in item['paired'].items():
            values=[core_rows[method,i]['metrics'][metric]['value']for i in item['common_ids']]
            assert len(values)==score['n']
            if values and all(v is not None for v in values):near(statistics.mean(values),score['mean']['value'])
edp_root=Path(active['edp_root'])/'tables'
edp=rows(edp_root/'paired_means.csv');ep=rows(edp_root/'paired_per_identity.csv')
for r in edp:
    values=[float(p['value'])for p in ep if p['group']==r['group']and p['method']==r['method']]
    assert len(values)==int(r['n']);near(statistics.mean(values),r['mean'])
dynamic=rows('reports/dynamic_source_clip_v1/tables/paired_means.csv')
dp=rows('reports/dynamic_source_clip_v1/tables/per_identity.csv')
for r in dynamic:
    values=[float(p['value'])for p in dp if p['method']==r['method']and p['metric']==r['metric']]
    assert len(values)==int(r['n'])==82;near(statistics.mean(values),r['mean'])
ablation=rows('reports/backend_ablation_batch_v2/paired_means.csv')
ap=rows('reports/backend_ablation_batch_v2/per_identity.csv')
for r in ablation:
    members=[p for p in ap if p['variant']==r['variant']and(r['cohort']=='all'or p['cohort']==r['cohort'])]
    assert len(members)==int(r['n'])
    for metric in CORE:near(statistics.mean(float(p[metric])for p in members),r[metric])

group6=next(g for g in summary['pairings']if len(g.split('+'))==6)
group5='ours+ah+trellis_original+omnire+splatad'
advisor=Path(active['advisor'])
combined=rows(advisor/'总表.csv')
for r in combined:
    method=next(k for k,v in NAMES.items()if v==r['模型'])
    for metric in CORE[:4]:near(r[metric],summary['pairings'][group6][metric]['paired'][method]['mean']['value'])
    near(r['ED-P'],next(x['mean']for x in edp if x['group']==group6 and x['method']==method))
    for metric in ['tOF','tLP100']:
        if r[metric]:near(r[metric],next(x['mean']for x in dynamic if x['method']==method and x['metric']==metric))
coverage=rows(edp_root/'coverage.csv')
parts=['### 12.1 总表','静态核心46人，ED-P 40人，动态82人；每列共同集合内比较。',
       table(['模型','PSNR ↑','SSIM ↑','LPIPS ↓','IoU ↑','ED-P_common ↓','tOF ↓','tLP100 ↓'],
             [[r['模型']]+[float(r[k])if r[k]else ''for k in ['PSNR','SSIM','LPIPS','IoU','ED-P','tOF','tLP100']]for r in combined],
             [(1,'max'),(2,'max'),(3,'min'),(4,'max'),(5,'min'),(6,'min'),(7,'min')])]
for group,label in [(group6,'六方法共同46人'),(group5,'五方法共同77人')]:
    rec=[]
    for method in ORDER:
        if method not in group.split('+'):continue
        rec.append([NAMES[method]]+[float(summary['pairings'][group][m]['paired'][method]['mean']['value'])for m in CORE])
    parts+=['### 12.2 '+label if group==group6 else '### 12.3 '+label,
            table(['模型','PSNR ↑','SSIM ↑','LPIPS ↓','IoU ↑','IoU_raw ↑'],rec,[(1,'max'),(2,'max'),(3,'min'),(4,'max'),(5,'max')])]
parts+=['### 12.4 方法覆盖率', table(['模型','静态核心有效/100','ED-P有效/100'],
        [[NAMES[m],summary['coverage'][m],next(int(r['valid_people'])for r in coverage if r['method']==m)]for m in ORDER])]
parts+=['### 12.5 ED-P共同集合',table(['比较集合','模型','共同人物','共同部位','ED-P_common ↓'],
        [['六方法'if len(r['group'].split('+'))==6 else '五方法',NAMES[r['method']],r['n'],r['shared_part_slots'],float(r['mean'])]for r in edp if r['group']in(group5,group6)])]
parts+=['### 12.6 后端消融', table(['组别','变体','n','PSNR ↑','SSIM ↑','LPIPS ↓','IoU ↑','IoU_raw ↑'],
       [[r['cohort'],r['variant'],r['n']]+[float(r[k])for k in CORE]for r in ablation])]
resources=[]
for variant in ['native_33k','complete_A_53k','rgb_continuation_53k']:
    subset=[r for r in ap if r['variant']==variant]
    resources.append([variant,len(subset)]+[statistics.mean(float(r[k])for r in subset)for k in ['training_minutes','peak_allocated_gib','peak_reserved_gib']])
parts+=['### 12.7 后端资源实测均值',table(['变体','n','训练分钟','峰值allocated GiB','峰值reserved GiB'],resources),
        '续训时间包含共同原生训练阶段；GPU共享条件下实测，不作跨模型速度排名。']
protocol=read('configs/human_source_common_v1/core_metric_protocol_frozen_v1.json')
parts+=['### 12.8 核心指标环境与权重',table(['包','版本'],list(protocol['packages'].items())),
        'AlexNet骨干SHA256：`'+protocol['lpips']['backbone_sha256']+'`。',
        'LPIPS线性权重SHA256：`'+protocol['lpips']['linear_sha256']+'`。',
        '逐文件来源见本目录 `sources.json`。这些是评分环境，不等于各模型的训练环境。',
        '### 12.9 安全复查入口',
        '在人体实验根目录执行以下只读状态查询与报表重建；不会启动训练。',
        _hmc_expand('```bash\n${CONDA_ROOT}/envs/ab_eval/bin/python scripts/run_human_metric_pipeline_v1.py --active-cohort --status\n${CONDA_ROOT}/envs/ab_eval/bin/python scripts/build_paper_experiment_report.py\n```'),
        '环境与方法接入版本见 `model_preparation/20261110/reports/新模型环境与复现核查.md`；本报告各指标细节以冻结配置和保存的逐人结果为准。']
report=OUT/'实验与附录说明.md'
prefix=report.read_text().split('<!-- GENERATED_TABLES -->')[0]
report.write_text(prefix+'<!-- GENERATED_TABLES -->\n\n'+'\n\n'.join(parts)+'\n')
import markdown
document = markdown.markdown(report.read_text(), extensions=['tables','fenced_code','toc'])
page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>nuScenes人体资产实验与附录说明</title><style>body{font:16px/1.8 system-ui,sans-serif;color:#1c2430;margin:36px auto;padding:0 24px;max-width:1180px}h1{font-size:30px}h2{margin-top:42px;border-bottom:1px solid #dce2e8;padding-bottom:8px}h3{margin-top:28px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;display:block;overflow:auto}th,td{border:1px solid #dce2e8;padding:8px 12px}th{background:#f1f4f8}code{font-size:13px;overflow-wrap:anywhere}pre{background:#f5f7f9;padding:16px;overflow:auto}a{color:#155fa0}strong{color:#111}@media print{body{max-width:none;margin:0;font-size:11px}table{font-size:9px}h2{break-after:avoid}tr{break-inside:avoid}}</style>
<p><a href="实验与附录说明.md">Markdown源文件</a> · <a href="sources.json">数值核查与来源</a></p>'''+document+'</html>'
(OUT/'index.html').write_text(page)
tables=OUT/'tables';tables.mkdir(exist_ok=True)
for folder,names in [(advisor,['总表.csv','覆盖率.csv','静态5模型_77人.csv','静态6模型_46人.csv','EDP5模型.csv','EDP6模型.csv']),
                     (Path('reports/backend_ablation_batch_v2'),['per_identity.csv','paired_means.csv'])]:
    for name in names:
        content(folder/name)
        target=('ablation_'if 'backend' in str(folder)else '')+name
        shutil.copy2(H/folder/name,tables/target)
missing=[]
for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if '://' not in link and not (OUT/link).is_file():missing.append(link)
assert not missing,missing
(OUT/'sources.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selection_sha256=selection['selection_sha256'],sources=SOURCES,checks=['core means recomputed from per-person rows',
    'ED-P means recomputed from paired per-person rows','dynamic means recomputed from per-person rows',
    'ablation group means recomputed from per-person rows','advisor total matched to current method pairings','local report links exist'],
    verification_scope='Numeric aggregation and local references; not a fresh audit of all rendered pixels',
    report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
print(report)
print('Paired means and source tables verified; local links valid')
