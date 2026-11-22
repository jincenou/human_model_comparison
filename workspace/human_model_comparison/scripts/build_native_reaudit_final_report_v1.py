"""Offline, source-bound report of resolved native human exports; no metric mutation.

Run in ab_eval. Only resolved/exports/*/*/validation.json selects final images.
Historical files and conservative candidates remain explicitly distinct.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import base64
import collections
import csv
import datetime
import hashlib
import html
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from human_source_metrics_v1 import crop

H = Path(__file__).resolve().parents[1]
ROOT = H / 'reports/native_human_export_reaudit_v1'
RES = ROOT / 'resolved'
OUT = ROOT / 'final'
METHODS = ('splatad', 'omnire', 'streetgs')
NAMES = dict(splatad='SplatAD', omnire='OmniRe', streetgs='StreetGS')
FONT = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'IoU', 'ED-P', 'CD', 'F-score')
LABELS = {'PSNR': 'PSNR ↑', 'SSIM': 'SSIM ↑', 'LPIPS': 'LPIPS ↓', 'IoU': 'IoU ↑',
          'ED-P': 'ED-P ↓', 'CD': 'CD (cm) ↓', 'F-score': 'F-score@10cm (%) ↑'}


def read(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(x):
    if x is None or x == '':
        return None
    return float(x)


def value_text(value, digits=3):
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}' if math.isfinite(value) else '∞'


def historical_metrics():
    out = {m: {} for m in METHODS}
    sources = [ROOT / 'publication_baseline/16_最终合并表_数值与范围.csv',
               ROOT / 'publication_baseline/20_人体可观测几何补表.csv']
    for path in sources:
        for row in csv.DictReader(path.open()):
            method = row['method'].lower()
            if method not in METHODS or row.get('domain', 'human') != 'human':
                continue
            metric = {'F1': 'F-score'}.get(row['metric'], row['metric'])
            out[method][metric] = dict(value=number(row['value']), n=int(row['n']),
                scope=row['scope'], source=str(path), source_sha256=sha(path))
    return out


def current_metrics(method):
    sp = RES / 'metrics' / method / 'static_summary.json'
    gp = RES / 'geometry' / method / 'summary.json'
    static = read(sp, {})
    geom = read(gp, {})
    values = {}
    for key, val in static.get('groups', {}).get(method, {}).items():
        key = 'ED-P' if key == 'EDP' else key
        if key in METRICS:
            values[key] = {k: v for k, v in val.items() if k != 'ids'}
            values[key].update(source=str(sp), source_sha256=sha(sp))
    for key, metric in [('CD_cm', 'CD'), ('F10', 'F-score')]:
        val = geom.get('groups', {}).get('main', {}).get('annotation_vertical_extent_conditioned', {}).get(key)
        if val is not None:
            values[metric] = {k: v for k, v in val.items() if k not in ['ids', 'valid_empty_ids']}
            values[metric].update(source=str(gp), source_sha256=sha(gp), panel='main/annotation_vertical_extent_conditioned')
    static_rows = {p.stem: read(p) for p in (RES / 'metrics' / method / 'static_rows').glob('*.json')}
    geometry_rows = {p.stem: read(p) for p in (RES / 'geometry' / method / 'results').glob('*.json')}
    edp_rows = {}
    for ep in sorted((RES / 'metrics' / method / 'edp').glob('*/results.json')):
        for row in read(ep)['rows']:
            assert row['id'] not in edp_rows, (method, row['id'], 'duplicate ED-P')
            edp_rows[row['id']] = row
    complete = (len(static_rows) == 100 and len(geometry_rows) == 100 and len(edp_rows) == 100
                and not static.get('unvisited', ['pending']) and not geom.get('unattempted_ids', ['pending']))
    return dict(values=values, static=static, geometry=geom, static_rows=static_rows,
                geometry_rows=geometry_rows, edp_rows=edp_rows, complete=complete)


def evidence_paths(method, sid, validation):
    """Read final image strictly from the resolved validation; never infer from candidate."""
    v = validation
    candidate_path = Path(v['candidate_validation'])
    assert sha(candidate_path) == v['candidate_validation_sha256']
    candidate = read(candidate_path)
    final = Path(v['actor_preview'])
    assert final.is_file() and sha(final) == v['actor_preview_sha256']
    if method == 'splatad':
        full = v['comparison_images']['full_scene']
        old = (v.get('old_output') or {}).get('actor_preview')
        time_note = '完整场景与目标使用冻结源曝光；遵循既定源相机协议。'
    elif method == 'omnire':
        preferred = ROOT / 'omnire/all_node_exposure_aux' / sid / 'all_node_time_full.png'
        assert preferred.is_file(), ('OmniRe exposure-corrected full scene missing', sid)
        full = str(preferred)
        prior = v.get('prior_registration')
        if isinstance(prior, str):
            prior = read(prior)
        old = (prior or {}).get('source_rgb')
        time_note = '完整场景中有时间包围帧的动态节点查询源曝光；无包围帧节点保留原生 anchor。目标采用冻结源曝光，不将该诊断整场景当作独立资产。'
    else:
        full = v['fullscene_source']
        old = read(v['old_validation']).get('actor_preview')
        time_note = '完整场景采用冻结源相机；本配置没有原生动态人体节点，最终是静态背景支持提取。'
    assert Path(full).is_file()
    if old:
        assert Path(old).is_file(), (method, sid, 'old image missing')
    return dict(source=v['source_reference']['reference_rgb'], full_scene=full, old=old,
                final=str(final), candidate=candidate['actor_preview'], time_note=time_note), candidate


def gaussian_counts(method, candidate, selected_support):
    if method == 'splatad':
        actor = int(candidate['actor_gaussians'])
        total = int(candidate['selected_gaussians'])
    elif method == 'omnire':
        actor = int(candidate['original_actor_gaussians'])
        total = int(candidate['exported_gaussians'])
    else:
        actor = 0
        total = int(candidate['selected_count'])
    selected_bg = 0 if selected_support == 'native_actor_only' else total - actor
    return dict(candidate_gaussian_count=total, selected_actor_gaussian_count=actor,
                selected_background_gaussian_count=selected_bg,
                selected_total_gaussian_count=actor + selected_bg,
                count_basis='native actor original count; Background included only by frozen resolved decision')


def montage(paths, roi, labels, size=240):
    im = Image.new('RGB', (size * len(paths), size + 29), 'white')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 17)
    for i, (path, label) in enumerate(zip(paths, labels)):
        draw.text((i * size + 7, 4), label, fill='#1b2d42', font=font)
        if path:
            src = np.asarray(Image.open(path).convert('RGB'))
            assert src.shape[:2] == (900, 1600), (path, src.shape)
            tile = Image.fromarray(crop(src, roi)).resize((size, size), Image.Resampling.LANCZOS)
        else:
            tile = Image.new('RGB', (size, size), '#eeeeee')
            ImageDraw.Draw(tile).text((18, size // 2), 'No previous export', fill='#555555', font=font)
        im.paste(tile, (i * size, 29))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=89, subsampling=0)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def row_status(current, sid):
    s = current['static_rows'].get(sid, {})
    e = current['edp_rows'].get(sid, {})
    g = current['geometry_rows'].get(sid, {})
    score = e.get('score', {})
    metric_status = {k: v.get('status') for k, v in s.get('metrics', {}).items()}
    reasons = [str(v.get('reason')) for v in s.get('metrics', {}).values() if v.get('reason')]
    if s.get('reason'):
        reasons.append(str(s['reason']))
    ep_reason = score.get('reason')
    if ep_reason:
        reasons.append('ED-P: ' + ep_reason)
    panel = g.get('reference_panel')
    if panel == 'low_support_exploratory':
        reasons.append('几何主表：真实可观测参考支持不足，仅列探索面板')
    if g.get('reason'):
        reasons.append('几何: ' + str(g['reason']))
    cond = g.get('panels', {}).get('annotation_vertical_extent_conditioned', {}).get('metrics', {})
    if cond.get('status') == 'empty_pred':
        reasons.append('几何空预测：F-score=0；CD无有限值，扩展均值为∞')
    return dict(static_status=s.get('status', 'pending'), edp_status=score.get('status', 'pending'),
                edp_valid_parts=e.get('valid_part_count'), geometry_status=g.get('status', 'pending'),
                geometry_reference_panel=panel, geometry_surface_status=cond.get('status'),
                metric_status=metric_status, na_reasons=reasons,
                score_values={k: v.get('value') for k, v in s.get('metrics', {}).items()},
                edp_value=score.get('value'))


def delta_records(old, current):
    rows = []
    for method in METHODS:
        for metric in METRICS:
            a = old[method].get(metric, {})
            b = current[method]['values'].get(metric, {})
            av, bv = a.get('value'), b.get('value')
            rows.append(dict(method=NAMES[method], metric=metric, old_value=av, old_n=a.get('n'),
                final_value=bv, final_n=b.get('n'), final_minus_old=(bv-av) if av is not None and bv is not None else None,
                final_valid_empty_n=b.get('valid_empty_n', 0), final_extended_mean=b.get('extended_mean'),
                old_source=a.get('source'), final_source=b.get('source'),
                scope='各方法每指标自身有效对象；前后有效对象可能不同，非配对因果效果'))
    return rows


def tables(old, current):
    parts = ['<div class="table-wrap"><table><thead><tr><th>方法</th>']
    parts.extend('<th>' + esc(LABELS[k]) + '</th>' for k in METRICS)
    parts.append('</tr></thead><tbody>')
    for method in METHODS:
        parts.append('<tr><th>' + NAMES[method] + '</th>')
        for metric in METRICS:
            a = old[method].get(metric, {})
            b = current[method]['values'].get(metric, {})
            cell = f'<span class="old">{value_text(a.get("value"))} (n={a.get("n",0)})</span><br>→ <strong>{value_text(b.get("value"))}</strong> (n={b.get("n",0)})'
            if metric == 'CD' and b.get('valid_empty_n', 0):
                cell += f'<br><small>空预测 {b["valid_empty_n"]}；扩展均值 ∞</small>'
            if not b:
                cell = '<span class="old">' + value_text(a.get('value')) + '</span><br>→ 待汇总'
            parts.append('<td>' + cell + '</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return ''.join(parts)


def build(allow_pending=False):
    contract = read(RES / 'batch_contract.json')
    assert len(contract['ids']) == 100
    OUT.mkdir(parents=True, exist_ok=True)
    current = {m: current_metrics(m) for m in METHODS}
    pending = [m for m in METHODS if not current[m]['complete']]
    if pending and not allow_pending:
        raise RuntimeError('Metrics unfinished; use --allow-pending-metrics for explicitly marked snapshot: ' + str(pending))
    old = historical_metrics()
    decisions = {}
    for m in METHODS:
        p = RES / 'decisions' / (m + '.json')
        d = read(p)
        assert len(d['rows']) == 100 and {x['id'] for x in d['rows']} == set(contract['ids'])
        decisions[m] = {x['id']: x for x in d['rows']}
    cards, checklist, manifest_rows = [], [], []
    for method in METHODS:
        for sid in contract['ids']:
            vp = RES / 'exports' / method / sid / 'validation.json'
            v = read(vp)
            assert v and v['status'] == 'passed' and v['source_reference'] == contract['references'][sid]
            assert v['reload_verified'] and not v['selection_uses_metric_scores']
            assert v['resolution_sha256'] == sha(RES / 'decisions' / (method + '.json'))
            d = decisions[method][sid]
            assert v['selected_support'] == d['selected_support']
            paths, candidate = evidence_paths(method, sid, v)
            counts = gaussian_counts(method, candidate, d['selected_support'])
            status = row_status(current[method], sid)
            sem = d.get('semantic_review_row', {})
            note = sem.get('visual_note', sem.get('note', sem.get('native_person_status', '')))
            if isinstance(note, dict):
                note = json.dumps(note, ensure_ascii=False)
            selected = '原生目标 actor-only' if d['selected_support'] == 'native_actor_only' else '经审计选用的可见支持提取'
            ref = contract['references'][sid]
            row_image = montage([paths[k] for k in ['source', 'full_scene', 'old', 'final']], ref['roi_xyxy'],
                                ['Source', 'Full scene', 'Previous export', 'Final selected'])
            candidate_same = paths['candidate'] == paths['final'] or sha(paths['candidate']) == sha(paths['final'])
            candidate_image = '' if candidate_same else montage([paths['final'], paths['candidate']], ref['roi_xyxy'],
                                                               ['Final selected', 'Audit candidate'])
            detail = {**counts, 'identity_key': ref['identity_key'], 'timestamp_us': ref['timestamp_us'],
                'camera_contract_sha256': ref['camera_contract_sha256'], 'source_mask_sha256': ref['reference_mask_sha256'],
                'selected_support': d['selected_support'], 'resolved_validation': str(vp), 'validation_sha256': sha(vp),
                'full_scene_exposure_note': paths['time_note'], 'scores': status, 'source_files': paths,
                'same_final_as_candidate': candidate_same}
            cards.append(f'<article data-method="{method}" data-id="{sid}" data-na="{int(bool(status["na_reasons"]))}">'
                f'<h3>{NAMES[method]} · {sid}</h3><p class="tag">{selected} · 独立重载通过 · 最终高斯 {counts["selected_total_gaussian_count"]:,}</p>'
                f'<img loading="lazy" src="{row_image}" alt="{method} {sid} source, full scene, previous, final">'
                f'<p>{esc(note)}</p><p class="small">{esc(paths["time_note"])}</p>'
                f'<p class="small">图像：{esc(status["static_status"])}；ED-P：{esc(status["edp_status"])} / 有效部位 {esc(status["edp_valid_parts"])}；几何：{esc(status["geometry_status"])} / {esc(status["geometry_reference_panel"])}。'
                f'{esc("；".join(status["na_reasons"]))}</p>'
                '<details><summary>审计候选与正式选择的区别</summary><p>候选只是按正可见贡献保留的支持集合。正式选用依据原生层和目标身份核查，未使用指标分数；不能把候选总高斯数当作正式目标高斯数。</p>'
                + (f'<img loading="lazy" src="{candidate_image}" alt="final versus conservative candidate">' if candidate_image else '<p>本对象正式选用该候选，保留其背景污染与残影限制。</p>')
                + f'<p>候选总数 {counts["candidate_gaussian_count"]:,}；正式目标节点 {counts["selected_actor_gaussian_count"]:,}；正式背景支持 {counts["selected_background_gaussian_count"]:,}；正式总数 {counts["selected_total_gaussian_count"]:,}。</p></details>'
                f'<details><summary>原始路径、哈希、指标与缺测原因</summary><pre>{esc(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details></article>')
            checklist.append(dict(method=NAMES[method], id=sid, actor_node_present=v.get('actor_node_present'),
                previous_export_present=bool(paths['old']), selected_support=d['selected_support'],
                technical_status=v['status'], independent_reload_verified=v['reload_verified'],
                candidate_gaussian_count=counts['candidate_gaussian_count'],
                selected_actor_gaussian_count=counts['selected_actor_gaussian_count'],
                selected_background_gaussian_count=counts['selected_background_gaussian_count'],
                selected_total_gaussian_count=counts['selected_total_gaussian_count'],
                static_status=status['static_status'], edp_status=status['edp_status'], edp_valid_parts=status['edp_valid_parts'],
                geometry_status=status['geometry_status'], geometry_reference_panel=status['geometry_reference_panel'],
                geometry_surface_status=status['geometry_surface_status'], na_reasons='; '.join(status['na_reasons']),
                semantic_note=note, formal_image=paths['final'], formal_image_sha256=v['actor_preview_sha256'],
                source_image=paths['source'], source_full_scene=paths['full_scene'], previous_image=paths['old'],
                conservative_candidate_image=paths['candidate'], validation=str(vp), validation_sha256=sha(vp)))
            manifest_rows.append(dict(method=method, id=sid, **detail,
                image_sha256={k:sha(paths[k]) if paths[k] else None for k in ['source','full_scene','old','final','candidate']}))
    assert len(checklist) == 300 and len({(x['method'],x['id']) for x in checklist}) == 300
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    banner = ('<p class="pending">正式图像 300/300 已齐；指标仍在汇总：' + ', '.join(NAMES[m] for m in pending) + '。这是明确标记的进度快照，待批次结束重建此页。</p>') if pending else '<p class="done">300/300 正式导出完成；全部静态、ED-P 与几何记录已结算。技术通过不等于完整人体重建成功。</p>'
    summary_lines=[]
    for m in METHODS:
        ms=[x for x in checklist if x['method']==NAMES[m]]
        num_actor=sum(x['selected_support']=='native_actor_only' for x in ms)
        summary_lines.append(f'<li><b>{NAMES[m]}</b>：{num_actor} 项采用原生 actor-only，{100-num_actor} 项采用经审计的可见支持提取；100 项均保留技术有效输出，未按指标挑选或删除低质量对象。</li>')
    metric_html=tables(old,current)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>100 人 × 3 方法：正式人体导出复核</title>
<style>body{margin:0;background:#eef2f6;color:#192a3d;font:16px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1350px;margin:auto;padding:24px}section,article{background:white;margin:18px 0;padding:20px;border-radius:12px}h1{font-size:29px}h2{font-size:22px}h3{margin:0;font-size:20px}p{margin:10px 0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.small,small{font-size:13px;color:#53637a}.tag{font-size:13px;color:#355278}.done{background:#e6f3ed;padding:14px}.pending{background:#fff1cf;padding:14px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #dbe2eb;text-align:left;vertical-align:top;min-width:88px}.old{color:#778599}details{margin-top:12px}summary{cursor:pointer;color:#305c8b}.controls{position:sticky;top:0;z-index:2;background:#e9eef5f5;padding:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}select,input{font:inherit;padding:7px;border:1px solid #a8b7ca;border-radius:5px}a{color:#185a9d}button{padding:7px 13px;cursor:pointer}article[hidden]{display:none}.links{display:flex;gap:18px;flex-wrap:wrap}</style>
<main><h1>100 人 × 3 方法：正式人体导出复核与指标结算</h1>'''+banner+f'<p class="small">生成时间 UTC：{esc(now)}。冻结清单 SHA-256：{esc(contract["selection_sha256"])}。</p>'
    page+='<section><h2>本次完成了什么</h2><ul>'+''.join(summary_lines)+'</ul><p>展示列为源照片、完整原生场景、旧独立导出、最终正式选用的独立导出。四列使用同一冻结 ROI、同一显示比例，没有逐列移动、配准或适配大小。原照片的模糊与遮挡保持原样；指标的二维外观对齐另按既定统一协议执行。</p><p><b>局部背景可能露出非目标信息。mask 只用于选择高斯，不擦除预测像素；技术可测不代表完整人体。</b>原生缺腿、透明、残影和环境污染均保留，未针对个别人体调参。静态背景支持不能当作源时间驱动的动作序列。</p><p>OmniRe 整场景使用已核查的曝光版本：有时间包围帧的动态节点查询源时间，其他节点保留原生 anchor；最终目标输出仍严格读取正式 resolved validation，整场景图不参与冒充独立输出。</p></section>'
    page+='<section><h2>旧版 → 最终版指标</h2><p>旧版来自本轮开始前的论文数据快照；新版来自正式 resolved 指标结果。每个数值都使用本方法、本指标自身有效对象，不取三方法交集。前后对象数可能不同，因此差值是报告汇总变化，并非同对象配对的因果效果。</p>'+metric_html
    page+='<p class="small">PSNR / SSIM / LPIPS：统一二维 mask 质心与面积对齐后的源视角外观。IoU：对应统一轮廓协议。ED-P：可检测且具有有效特征的共同身体部位距离，缺少有效部位记 NA。CD / F-score：同一真实可观测 LiDAR 参考主面板和标注竖直尺度标定，CD 为未平方双向最近距离均值，F-score 阈值 10 cm；不代表不可见背面完整度。CD 有限均值与空预测数、扩展均值同时保留，空预测不补零成完美几何。16 人的低参考支持探索面板不混入主几何均值。</p><p class="small">本次静态导出复核没有重训时序模型。原生动作分支的 tOF / tLP 不因加入静态背景而增加有效人数；旧时序记录保留在结果来源中，不制造新的动态分数。</p></section>'
    na_counts=collections.Counter(reason for row in checklist for reason in row['na_reasons'].split('; ') if reason)
    page+='<section><h2>300 项检查与缺测原因</h2><p>每个方法都检查冻结的 100 人。下方每项保留独立重载、支持来源、指标状态及 NA 原因；画廊中的隐藏/筛选只影响浏览，不改变任何汇总。</p><ul>'+''.join(f'<li>{esc(k)}：{v} 项</li>' for k,v in sorted(na_counts.items()))+'</ul><div class="links"><a href="checklist_300.csv" download>下载 300 项检查清单 CSV</a><a href="metrics_delta.csv" download>下载指标前后差值 CSV</a><a href="offline_report.zip" download>下载离线报告 ZIP</a></div></section>'
    page+='<div class="controls"><label>方法 <select id="method"><option value="all">全部方法</option>'+''.join(f'<option value="{m}">{NAMES[m]}</option>' for m in METHODS)+'</select></label><label>对象 <input id="person" placeholder="HN0273"></label><label><input id="onlyna" type="checkbox">仅看缺测/探索面板记录</label><button id="reset">重置</button><span id="count"></span></div>'+''.join(cards)
    page+='''</main><script>const method=document.getElementById('method'),person=document.getElementById('person'),onlyna=document.getElementById('onlyna');function filter(){let n=0;document.querySelectorAll('article').forEach(x=>{let show=(method.value==='all'||x.dataset.method===method.value)&&x.dataset.id.toLowerCase().includes(person.value.trim().toLowerCase())&&(!onlyna.checked||x.dataset.na==='1');x.hidden=!show;if(show)n++});document.getElementById('count').textContent=n+' / 300 项';}method.onchange=filter;person.oninput=filter;onlyna.onchange=filter;document.getElementById('reset').onclick=()=>{method.value='all';person.value='';onlyna.checked=false;filter()};filter();</script></html>'''
    (OUT/'index.html').write_text(page)
    csv_write(OUT/'checklist_300.csv',checklist)
    delta=delta_records(old,current);csv_write(OUT/'metrics_delta.csv',delta)
    manifest=dict(schema='native_human_export_reaudit_final_report_v1',status='complete' if not pending else 'metrics_in_progress',
        updated_utc=now,expected_records=300,records=manifest_rows,metrics_delta=delta,
        snapshot_methods_pending_metrics=pending,source_contract=str(RES/'batch_contract.json'),
        source_contract_sha256=sha(RES/'batch_contract.json'),script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        no_prediction_erasing=True,no_independent_display_fit=True,self_contained_images=True,
        final_images_selected_only_by_resolved_validation=True,weights_or_full_source_data_copied=False)
    save(OUT/'manifest.json',manifest)
    save(OUT/'summary.json',dict(status=manifest['status'],updated_utc=now,records=300,
        metrics={m:current[m]['values'] for m in METHODS},old_metrics=old,
        decision_counts={m:dict(collections.Counter(x['selected_support'] for x in decisions[m].values())) for m in METHODS},
        na_reason_counts=dict(na_counts),checklist=str(OUT/'checklist_300.csv'),html=str(OUT/'index.html')))
    with zipfile.ZipFile(OUT/'offline_report.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['index.html','checklist_300.csv','metrics_delta.csv','summary.json','manifest.json']:
            z.write(OUT/name,arcname=name)
    print(json.dumps(dict(status=manifest['status'],records=300,html=str(OUT/'index.html'),
        zip=str(OUT/'offline_report.zip'),html_mib=(OUT/'index.html').stat().st_size/2**20,
        zip_mib=(OUT/'offline_report.zip').stat().st_size/2**20),ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-pending-metrics',action='store_true');a=p.parse_args()
    build(a.allow_pending_metrics)
