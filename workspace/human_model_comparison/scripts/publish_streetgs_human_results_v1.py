"""Add audited StreetGS human results without changing previous raw score sources.

Default is a read-only dry run. --publish first backs up every existing artifact
that will change, then adds a source overlay and updates the manuscript products.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
P = R / 'reports/asset_experiments_paper_20261115_v3'
O = H / 'reports/streetgs_human_probe_v1'
G = H / 'reports/streetgs_human_geometry_v1'
E = H / 'exports/streetgs_human_probe_v1/objects'
sys.path.insert(0, str(R / 'scripts'))
import publish_settled_joint_asset_table as original


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(p, digest):
    assert sha(p) == digest, 'Source changed: ' + str(p)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collected_rows(folder, ids, label):
    rows = {r['id']: r for p in folder.glob('*.json') for r in [read(p)]}
    require(set(rows) == set(ids), label + ': not all 100 frozen identities have explicit records')
    return rows


def observed_cell(rows, key, value_key='value'):
    valid = [r for r in rows if r['metrics'].get(key, {}).get('status') == 'ok']
    return dict(value=sum(r['metrics'][key][value_key] for r in valid) / len(valid) if valid else None,
                n=len(valid), ids=sorted(r['id'] for r in valid))


def verify_cell(actual, expected, label):
    require(actual['n'] == expected['n'] and sorted(actual.get('ids', [])) == expected['ids'], label + ': count/ID mismatch')
    if expected['value'] is None:
        require(actual['value'] is None, label + ': missing mean mismatch')
    else:
        require(actual['value'] is not None and math.isclose(actual['value'], expected['value'], rel_tol=1e-12, abs_tol=1e-12), label + ': recomputed mean mismatch')


def completed_sources():
    active = read(H / 'inputs/active_human_benchmark.json')
    checked(active['selection'], active['selection_sha256'])
    ids = sorted(r['id'] for r in read(active['selection'])['objects'])
    require(len(ids) == len(set(ids)) == 100, 'Expected the exact frozen100')
    source = {}
    for sid in ids:
        p = E / sid / 'validation.json'
        require(p.exists(), sid + ': source export absent')
        row = read(p)
        require(row['status'] in ['passed', 'unavailable', 'confirmed_unavailable'], sid + ': source export is not terminal')
        if row['status'] == 'passed':
            require(row.get('reload_verified') is True, sid + ': independent reload not verified')
            if row.get('reload_record'):
                checked(row['reload_record'], row['reload_sha256'])
                reload = read(row['reload_record'])
                require(reload['status'] == 'passed' and reload['independent_process'], sid + ': reload failed')
        else:
            require(bool(row.get('reason')), sid + ': missing unavailability explanation')
        source[sid] = row
    static = collected_rows(O / 'static_rows', ids, 'Static scoring')
    for sid, row in static.items():
        require(row['status'] in ['scored', 'reference_ineligible', 'native_export_unavailable', 'confirmed_unavailable'], sid + ': static stage not settled')
        if row['status'] == 'scored':
            checked(row['source_validation'], row['source_validation_sha256'])
            require(source[sid]['status'] == 'passed', sid + ': scored output has no accepted export')
        else:
            require(bool(row.get('reason')), sid + ': missing static exclusion reason')
    stats = read(O / 'static_summary.json')['groups']['streetgs']
    for key in ['PSNR', 'SSIM', 'LPIPS', 'IoU']:
        verify_cell(stats[key], observed_cell(list(static.values()), key), 'StreetGS ' + key)
    edp = {}
    edp_paths = []
    for path in sorted((O / 'edp').glob('*/results.json')):
        data = read(path)
        checked(path.parent / 'inputs.json', data['inputs_sha256'])
        checked(path.parent / 'protocol.json', data['protocol_sha256'])
        edp_paths.append(path)
        for row in data['rows']:
            require(row['method'] == 'streetgs' and row['id'] not in edp, 'Duplicate/non-StreetGS ED-P row')
            require(row['score']['status'] in ['ok', 'na'], 'ED-P has nonterminal status')
            if row['score']['status'] == 'na':
                require(bool(row['score'].get('reason')), 'ED-P NA has no reason')
            edp[row['id']] = row
    scored = {sid for sid, row in static.items() if row['status'] == 'scored'}
    require(set(edp) == scored, 'ED-P must be attempted for every static-scored identity; unavailable source rows stay NA')
    normalized = [dict(id=sid, metrics={'EDP': row['score']}) for sid, row in edp.items()]
    verify_cell(stats['EDP'], observed_cell(normalized, 'EDP'), 'StreetGS ED-P')
    geometry = collected_rows(G / 'results', ids, 'Geometry scoring')
    allowed = {'completed', 'extent_numerically_unresolved', 'source_unavailable', 'unavailable', 'confirmed_unavailable'}
    for sid, row in geometry.items():
        require(row['status'] in allowed, sid + ': geometry still running or failure not resolved')
        if row['status'] != 'completed':
            require(bool(row.get('reason')), sid + ': geometry exclusion has no reason')
    geom_groups = read(G / 'summary.json')['groups']['main']
    for measurement in ['annotation_vertical_extent_conditioned', 'native_metric']:
        measured = [(r, r['panels'][measurement]['metrics']) for r in geometry.values()
                    if r.get('reference_panel') == 'main' and measurement in r.get('panels', {})]
        for source_key, raw_key, mult in [('CD_cm', 'cd_m', 100), ('F10', 'f1_10cm', 100)]:
            pairs = [(r, m) for r, m in measured if m[raw_key] is not None]
            expected = dict(value=sum(m[raw_key] * mult for r, m in pairs) / len(pairs) if pairs else None,
                            n=len(pairs), ids=sorted(r['id'] for r, m in pairs))
            verify_cell(geom_groups[measurement][source_key], expected, 'StreetGS geometry ' + measurement + ' ' + source_key)
    geo = copy.deepcopy(geom_groups['annotation_vertical_extent_conditioned'])
    geo['native_metric_supplement'] = copy.deepcopy(geom_groups['native_metric'])
    evidence = [H / 'inputs/active_human_benchmark.json', Path(active['selection']),
                O / 'native_adapter_check.json', O / 'static_protocol.json', O / 'static_summary.json',
                G / 'protocol.json', G / 'summary.json', *edp_paths]
    evidence += [E / sid / 'validation.json' for sid in ids]
    evidence += list((O / 'static_rows').glob('*.json')) + list((G / 'results').glob('*.json'))
    require(read(O / 'native_adapter_check.json')['status'] == 'passed', 'Native adapter numerical check failed')
    return stats, geo, {str(p): sha(p) for p in evidence}, dict(
        frozen_id_count=100, source_terminal=len(source), static_terminal=len(static),
        edp_attempted=len(edp), edp_source_unavailable=100-len(edp), geometry_terminal=len(geometry))


def temporal_not_applicable(table, md, tex, records):
    # Historical rendering code special-cases StreetGS as not-yet-evaluated.
    # This addition establishes static-only extraction, so correct this row only.
    table, n = re.subn(r'<tr><th>StreetGS</th>.*?</tr>',
        lambda m: m[0].replace('◇', '∅').replace('not_evaluated', 'not_applicable')
                    .replace('Not evaluated in this configuration', 'Static background subset has no native pedestrian motion'), table)
    require(n == 1, 'Expected one StreetGS HTML row')
    md = '\n'.join(line.replace('◇', '∅') if line.startswith('| StreetGS |') else line for line in md.splitlines()) + '\n'
    tex = '\n'.join(line.replace(r'$\diamond$', r'$\varnothing$') if line.startswith('StreetGS &') else line for line in tex.splitlines()) + '\n'
    for row in records:
        if row['domain'] == 'human' and row['method'] == 'StreetGS' and row['metric'] in ['tOF', 'tLP100']:
            require(row['value'] is None, 'Unexpected fabricated StreetGS temporal value')
            row['scope'] = 'not_applicable'
    return table, md, tex, records


def geometry_table(data):
    records, html_rows, md_rows, tex_rows = [], [], [], []
    ranks = {key: sorted({data['rows']['human'][name][key]['value'] for name in original.ORDER
                         if data['rows']['human'][name][key]['value'] is not None}, reverse=key == 'F1') for key in ['CD', 'F1']}
    for name in original.ORDER:
        values = []
        for key in ['CD', 'F1']:
            cell = data['rows']['human'][name][key]
            value = cell['value']
            rank = ranks[key].index(value) + 1 if value is not None else None
            text = original.fmt(key, value) if value is not None else '◇'
            records.append(dict(method=name, metric=key, value=value, n=cell.get('n', 0), rank=rank, scope='method_own_valid_main_reference_support'))
            values.append((text, rank))
        n = '/'.join(str(data['rows']['human'][name][key].get('n', 0)) for key in ['CD', 'F1'])
        label = original.LABEL.get(name, name)
        hc = ['<strong>'+t+'</strong>' if k == 1 else '<u>'+t+'</u>' if k == 2 else t for t, k in values]
        mc = ['**'+t+'**' if k == 1 else '<u>'+t+'</u>' if k == 2 else t for t, k in values]
        tc = [r'\textbf{'+t+'}' if k == 1 else r'\underline{'+t+'}' if k == 2 else r'$\diamond$' if t == '◇' else t for t, k in values]
        html_rows.append('<tr><th>'+label+'</th>'+''.join('<td>'+t+'</td>' for t in [*hc,n])+'</tr>')
        md_rows.append('| '+ ' | '.join([label,*mc,n])+' |')
        tex_rows.append(' & '.join([label,*tc,n])+r' \\')
    native = data['rows']['human']['StreetGS']['native_metric_geometry']
    street = data['rows']['human']['StreetGS']
    caption = ('Pedestrian observable-surface geometry on the existing 84-person main reference-support panel. '
               'Each method uses its own valid outputs after the existing annotated world-vertical-extent calibration. '
               'CD is in cm over finite distances; F-score is a percentage at 10 cm. n lists CD/F-score counts. '
               'Valid empty predictions have F-score zero and non-finite CD; failure records are not filled with zero. '
               'Bold and underline mark the best and second distinct full-precision means; populations differ. '
               'StreetGS uses the GT-box-assisted static-background extraction described in the text; its count denotes technical measurability, not successful recovery of complete people. '
               f"Its height-conditioned result has {street['CD'].get('valid_empty_n', 0)} valid empty predictions, excluded only from the finite-CD mean. "
               f"At native metric scale, the supplementary StreetGS CD is {original.fmt('CD', native['CD_cm']['value'])} cm (n={native['CD_cm']['n']}) "
               f"and F-score is {original.fmt('F1', native['F10']['value'])}% (n={native['F10']['n']}), with {native['CD_cm'].get('valid_empty_n', 0)} valid empty predictions. "
               'These native-metric diagnostics do not replace the height-conditioned comparison. The shared height-calibration rule maps the extent of the extracted subset, including background trails, to annotated pedestrian height while scaling camera-relative translation; consequently, its large conditioned error cannot be attributed solely to the original metric geometry.')
    ht = '<div id="streetgs-human-geometry"><h3>Supplementary pedestrian observable geometry</h3><div class="tablewrap"><table style="min-width:620px"><thead><tr><th>Method</th><th>CD ↓</th><th>F-score ↑</th><th>n (CD/F-score)</th></tr></thead><tbody>'+''.join(html_rows)+'</tbody></table></div><p class="caption">'+html.escape(caption)+'</p></div>'
    md = '| Method | CD ↓ | F-score ↑ | n (CD/F-score) |\n|---|---:|---:|---:|\n'+'\n'.join(md_rows)+'\n\n'+caption+'\n'
    tex = '\n'.join([r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrr}',r'\toprule',r'Method & CD$\downarrow$ & F-score$\uparrow$ & $n$ (CD/F-score) \\',r'\midrule',*tex_rows,r'\bottomrule',r'\end{tabular}',r'\caption{'+caption.replace('%',r'\%')+r'}',r'\label{tab:human_observable_geometry}',r'\end{table}'])+'\n'
    return ht, md, tex, records


BASELINE_SENTENCE = ('The pedestrian comparison includes the same eight methods. For StreetGS, we independently export '
    'and reload native static-background Gaussian subsets selected by the GT pedestrian box of the source image\'s training frame, expanded by '
    '0.25 m along each half-axis, providing additional target-localization information. These subsets can contain '
    'background texture or motion trails; their sample count denotes technical measurability rather than recovery '
    'of complete people. '
    'The checkpoints contain no native pedestrian node or learned pedestrian trajectory for these identities. '
    'Accordingly, StreetGS is evaluated for static quality only and its temporal metrics are inapplicable.')
CHINESE_SENTENCE = ('人体比较同样包含8种方法。StreetGS借助源图对应训练帧的GT人体框及各半轴0.25 m扩展，'
    '从原生静态背景高斯中提取对应子集并独立重载；该过程向它提供了额外目标定位信息。'
    '子集可能包含背景纹理和运动残影，数量仅表示技术上可计分，不表示完整人体重建成功。'
    '现有checkpoint未包含这些身份的原生人体节点及学得的人体轨迹，因此只评测静态质量，不适用时序指标。')


def amend_prose(text):
    old = ('The pedestrian comparison includes the same methods except StreetGS, for which this configured implementation '
           'did not provide a reliably verified independent pedestrian export.')
    text = text.replace(old, BASELINE_SENTENCE)
    oldcn = '人体比较其中7种方法；StreetGS的本次实现未获得通过可靠性核验的独立人体导出。'
    text = text.replace(oldcn, CHINESE_SENTENCE)
    text = text.replace('8种刚体方法、7种人体方法', '8种刚体方法、8种人体方法')
    text = text.replace('8种刚体方法与7种人体方法', '8种刚体方法与8种人体方法')
    text = text.replace('8种刚体方法和7种人体方法', '8种刚体方法和8种人体方法')
    text = text.replace('StreetGS未评测人体的原因限定于本配置未获得可靠的独立导出，不扩大为模型普遍能力结论。',
        'StreetGS已增加GT人体框及各半轴0.25 m扩展辅助的静态背景高斯子集提取，提供了额外定位信息；通过独立重载后计分，但不声称提取的是完整人体。无原生人体轨迹，因此时序不适用。')
    return text


def count_paragraphs(records, tex=False):
    lines = []
    for name in original.ORDER:
        rows = [r for r in records if r['method'] == name]
        parts = [r['domain']+' '+original.metric_label(r['metric'])+(' $n='+str(r['n'])+'$' if tex else ' n='+str(r['n'])) for r in rows]
        line = original.LABEL.get(name, name)+': '+'; '.join(parts)+'.'
        lines.append(line if tex else '<p>'+line+'</p>')
    return '\n\n'.join(lines) if tex else ''.join(lines)


def build():
    stats, geo, sources, gate = completed_sources()
    proofpath = P / '17_最终合并表_来源核验.json'
    proof = read(proofpath)
    if proof.get('streetgs_human_overlay'):
        proofpath = Path(proof['baseline_proof'])
        checked(proofpath, proof['baseline_proof_sha256'])
        proof = read(proofpath)
    rawpath = Path(proof['source'])
    rawhash = sha(rawpath)
    data = copy.deepcopy(read(rawpath))
    for row in proof['rows']:
        cell = data['rows'].get(row['domain'], {}).get(row['method'], {}).get(row['metric'], {})
        require((row['value'],row['n']) == (cell.get('value'),cell.get('n',0)), 'Historical raw/proof mismatch')
    original_selection = copy.deepcopy(data['display_selection_policy'])
    cells = {}
    for key, source_key in [('PSNR','PSNR'),('SSIM','SSIM'),('LPIPS','LPIPS'),('IoU','IoU'),('ED-P','EDP')]:
        cells[key] = dict(copy.deepcopy(stats[source_key]), selection_scope='method_own_valid')
    for key in ['tOF','tLP100']:
        cells[key] = dict(value=None,n=0,ids=[],selection_scope='not_applicable',reason='Static-background extraction has no native learned pedestrian trajectory')
    cells['CD'] = copy.deepcopy(geo['CD_cm'])
    cells['F1'] = copy.deepcopy(geo['F10'])
    cells['native_metric_geometry'] = copy.deepcopy(geo['native_metric_supplement'])
    data['rows']['human']['StreetGS'] = cells
    table, md, tex, records = temporal_not_applicable(*original.render(data))
    require(original.FOOTNOTE == proof['caption'], 'The existing table caption must remain unchanged')
    require(data['display_selection_policy'] == original_selection, 'Original display selection changed')
    old = {(r['domain'],r['method'],r['metric']):r for r in proof['rows']}
    for row in records:
        if row['domain'] == 'human' and row['method'] == 'StreetGS':
            continue
        before = old[(row['domain'],row['method'],row['metric'])]
        require((row['value'],row['n'],row['scope']) == (before['value'],before['n'],before['scope']), 'Unrelated metric value, population or scope changed')
    gh, gm, gt, gr = geometry_table(data)
    coverage_note = (f"All {gate['edp_attempted']} technically scored StreetGS subsets were processed for ED-P. "
        f"Only {cells['ED-P']['n']} had valid body parts supported by both the reference and the subset's own opacity; "
        f"the remaining {gate['edp_attempted']-cells['ED-P']['n']} were recorded as NA because no valid common part was available. "
        'These are metric-support exclusions, not execution failures, and the ED-P mean is conditional on the reported supported subset. '
        'A count of 100 exported source-frame subsets does not establish 100 successfully reconstructed complete humans.')
    coverage_html = '<div id="streetgs-human-coverage"><h3>StreetGS pedestrian extraction and metric coverage</h3><p>'+html.escape(coverage_note)+'</p></div>'
    webpage = amend_prose((P / '00_实验正文与合并表.html').read_text())
    webpage, n = re.subn(r'(<h2>Table 1 · Joint asset quality</h2>)<div class="tablewrap">.*?</table></div><div class="caption">.*?</div>', lambda m:m[1]+table, webpage, count=1, flags=re.S)
    require(n == 1, 'Missing unique main table in HTML')
    webpage, n = re.subn(r'(<h3>Exact effective sample counts</h3>).*?(</details>)', lambda m:m[1]+count_paragraphs(records)+m[2], webpage, count=1, flags=re.S)
    require(n == 1, 'Missing exact count section')
    if '<div id="streetgs-human-geometry">' in webpage:
        webpage = re.sub(r'<div id="streetgs-human-geometry">.*?</p></div>', lambda m:gh, webpage, count=1, flags=re.S)
    else:
        require('</main>' in webpage, 'Missing main document end')
        webpage = webpage.replace('</main>',gh+'</main>')
    if '<div id="streetgs-human-coverage">' in webpage:
        webpage = re.sub(r'<div id="streetgs-human-coverage">.*?</p></div>',lambda m:coverage_html,webpage,count=1,flags=re.S)
    else:
        webpage = webpage.replace('<div id="streetgs-human-geometry">',coverage_html+'<div id="streetgs-human-geometry">')
    main = amend_prose((P / '01_实验正文.tex').read_text())
    appendix = amend_prose((P / '03_实验附录.tex').read_text())
    appendix, n = re.subn(r'(\\subsection\{Exact effective sample counts\}).*?(?=\\(?:subsection|input)\{|\Z)',lambda m:m[1]+'\n\n'+count_paragraphs(records,True)+'\n\n', appendix, count=1, flags=re.S)
    require(n == 1, 'Missing appendix count section')
    coverage_tex = r'\subsection{StreetGS pedestrian extraction and metric coverage}'+'\n'+coverage_note+'\n\n'
    if r'\subsection{StreetGS pedestrian extraction and metric coverage}' in appendix:
        appendix = re.sub(r'\\subsection\{StreetGS pedestrian extraction and metric coverage\}.*?(?=\\(?:subsection|input)\{|\Z)',lambda m:coverage_tex,appendix,count=1,flags=re.S)
    else:
        appendix += '\n'+coverage_tex
    if r'\input{20_人体可观测几何补表.tex}' not in appendix:
        appendix += '\n'+r'\input{20_人体可观测几何补表.tex}'+'\n'
    artifacts = {'00_实验正文与合并表.html':webpage,'01_实验正文.tex':main,'02_合并主表.tex':tex,
                 '03_实验附录.tex':appendix,'14_最终合并表.tex':tex,'15_最终合并表.md':md,
                 '16_最终合并表_数值与范围.csv':csv_text(records),'20_人体可观测几何补表.tex':gt,
                 '20_人体可观测几何补表.md':gm,'20_人体可观测几何补表.csv':csv_text(gr)}
    if (P / '05_写作依据与审校说明.txt').exists():
        artifacts['05_写作依据与审校说明.txt'] = amend_prose((P / '05_写作依据与审校说明.txt').read_text())
    require('reliably verified independent pedestrian export' not in webpage and old is not None, 'Stale exclusion remains')
    overlay = dict(status='settled_streetgs_human_addition',base_source=str(rawpath),base_source_sha256=rawhash,
                   baseline_proof=str(proofpath),baseline_proof_sha256=sha(proofpath),
                   updates={'human':{'StreetGS':cells}},sources=sources,completion=gate,
                   protocol='Same accepted source-camera, 2D appearance, ED-P, and observable-surface geometry definitions',
                   extraction='Native StreetGS static-background subset selected by the source image training-frame GT pedestrian box plus 0.25 m per half-axis; additional target-localization information; real Gaussian parameters and independent reload; technical count is not complete-human reconstruction success',
                   temporal='Not applicable: no native pedestrian node/trajectory; no synthesized motion',
                   historical_raw_sources_modified=False,display_selection_policy_unchanged=True)
    return artifacts, overlay, proof, records, gr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    try:
        artifacts, overlay, baseline, records, geometry = build()
    except (ValueError, AssertionError, KeyError, FileNotFoundError) as error:
        print(dumps(dict(status='not_ready',reason=str(error),published=False)))
        raise SystemExit(2)
    if not args.publish:
        print(dumps(dict(status='ready_dry_run',published=False,completion=overlay['completion'],
                        streetgs_main=[r for r in records if r['method']=='StreetGS' and r['domain']=='human'],
                        streetgs_geometry=[r for r in geometry if r['method']=='StreetGS'],
                        artifacts=list(artifacts),old_caption_retained=True)))
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = P / 'history' / ('before_streetgs_human_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    names = [*artifacts, '17_最终合并表_来源核验.json', '19_展示与正文更新核验.json',
             '21_StreetGS人体补测来源覆盖.json', '22_StreetGS人体补测更新核验.json']
    prior = {}
    for name in names:
        path = P / name
        if path.exists():
            prior[name] = sha(path)
            shutil.copy2(path, backup / name)
    # Keep the original baseline independently addressable on repeated publishing.
    baseline_path = backup / 'baseline_17_最终合并表_来源核验.json'
    baseline_path.write_text(dumps(baseline))
    overlay['baseline_proof'] = str(baseline_path)
    overlay['baseline_proof_sha256'] = sha(baseline_path)
    overlaypath = P / '21_StreetGS人体补测来源覆盖.json'
    artifacts[overlaypath.name] = dumps(overlay)
    proof = dict(baseline, status='published_after_streetgs_human_settlement',rows=records,
                 streetgs_human_overlay=str(overlaypath),source_overlay_sha256=hashlib.sha256(dumps(overlay).encode()).hexdigest(),
                 baseline_proof=str(baseline_path),baseline_proof_sha256=sha(baseline_path),
                 old_source_unchanged=True,completion=overlay['completion'])
    artifacts['17_最终合并表_来源核验.json'] = dumps(proof)
    for name,text in artifacts.items():
        (P / name).write_text(text)
    checked(overlay['base_source'],overlay['base_source_sha256'])
    audit = dict(status='passed',backup=str(backup),before=prior,after={name:sha(P/name) for name in artifacts},
                 static_completion=overlay['completion'],main_cells=len(records),
                 numeric_cells=sum(r['value'] is not None for r in records),
                 best_cells=sum(r['rank']==1 for r in records),second_cells=sum(r['rank']==2 for r in records),
                 unrelated_values_and_denominators_unchanged=True,top50_policy_and_caption_unchanged=True,
                 temporal_values_not_fabricated=True,geometry_values_recomputed_from_records=True,
                 manuscript_latex_compiled=False,source_script=str(Path(__file__)),source_script_sha256=sha(__file__))
    (P / '22_StreetGS人体补测更新核验.json').write_text(dumps(audit))
    # Retain the previous audit in history and point readers to the new evidence.
    (P / '19_展示与正文更新核验.json').write_text(dumps(dict(status='superseded_by_streetgs_addition',
        current_audit=str(P/'22_StreetGS人体补测更新核验.json'),current_audit_sha256=sha(P/'22_StreetGS人体补测更新核验.json'),
        previous_audit=str(backup/'19_展示与正文更新核验.json'))))
    print(dumps(dict(status='published',backup=str(backup),audit=str(P/'22_StreetGS人体补测更新核验.json'))))


if __name__ == '__main__':
    main()
