"""Retain failure examples and causal evidence without retrying generation."""
import base64
import html
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from common_human_orbit import H, read, save, sha, checked

ROOT = H/'reports/trellis2_failure_root_cause_v1'
LABELS = {'native_generation_failed':'原生生成失败',
          'source_registration_unavailable':'有原生输出，但共同相机配准不可用',
          'registration_pending':'尚未完成，不能归入失败', 'scored':'已计分'}


def picture(path, label, rgba=False):
    image = Image.open(path)
    if rgba:
        layer = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', layer.size, 'white'), layer).convert('RGB')
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    data = base64.b64encode(buffer.getvalue()).decode()
    return '<figure><figcaption>'+html.escape(label)+'</figcaption><img loading="lazy" alt="'+html.escape(label)+'" src="data:image/png;base64,'+data+'"></figure>'


def main():
    summary_path = H/'reports/source_camera_metric_cohort_v3/summary.json'
    raw = summary_path.read_bytes(); summary = json.loads(raw)
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    audit = read(ROOT/'audit.json')
    inputs = {r['id']:r for r in audit['rows']}
    replay_path = ROOT/'controlled_replay/replay.json'; replay = read(replay_path)
    assert replay['status'] == 'finished' and len(replay['tasks']) == 4
    assert replay['production_assets_modified'] is False
    for r in replay['tasks']:
        folder = ROOT/'controlled_replay'/(r['id']+'_'+r['variant'])
        checked(folder/'input.png', r['input_sha256'])
        checked(folder/'native_voxel_mesh.pt', r['native_checkpoint_sha256'])
        if r['variant'] == 'original':
            assert all(v['exact_tensor_equal'] for v in r['comparison_to_production'].values())
    rows = sorted((r for r in summary['rows'] if r['method'] == 'trellis2'), key=lambda r:r['id'])
    counts = Counter(r['status'] for r in rows)
    out = ROOT/'failure_cases'; out.mkdir(exist_ok=True)
    doc = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>TRELLIS.2失败案例与原因记录</title><style>body{max-width:1150px;margin:24px auto;padding:0 18px;font:16px/1.65 system-ui;background:#f4f5f7;color:#20252b}section{background:white;padding:18px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;display:block}figcaption{min-height:52px}code{overflow-wrap:anywhere}.wide{width:100%}</style>',
        '<h1>TRELLIS.2失败案例与原因记录</h1><p>固定100人；失败保留，不替换人物，不修改种子反复生成。可计分输出无论视觉好坏都保留；原有指标不变。</p>',
        '<p>'+html.escape('；'.join(LABELS.get(k,k)+f'：{v}人' for k,v in sorted(counts.items())))+'</p>',
        '<p><b>以下图片是失败诊断，不是统一源视角对比。</b>输入以白底显示；原生预览保留官方黑底和原生相机，不能用图中人物大小或角度作跨模型比较。尚未完成项单列，不算失败。</p>',
        '<p><a href="../../source_camera_metric_cohort_v3/tables/RESULTS.md">返回指标及正文说明</a></p>',
        '<section><h2>已经完成的有限对照复现</h2><p>相同官方代码、权重、seed=42和1024_cascade设置：HN0046的薄片输出、HN0004的完整人体对照、HN0214旧mask输出均被复现，原生顶点/面/稀疏坐标/属性与原资产逐张量完全一致。缺陷并非GLB导出或统一相机渲染才产生。</p>',
        '<p>HN0214旧mask包含大块背景。只修正同一真实图像的mask后，输出从背景板状结构变成人体形状；该结果只用于归因，未替换正式资产、未新增正式分数。不能把全部失败归因于模型本身。</p>',
        '<p>对已有43例配准失败的详细审计发现36例原生几何接近薄片；52例已计分输出中也有5例满足同一几何诊断条件。因此薄片诊断不是筛除规则，也不是模型成功的定义。</p>']
    comparison = ROOT/'controlled_replay/comparison.jpg'
    if comparison.exists():
        doc.append(picture(comparison,'四行依次为HN0046原输入、HN0004原输入、HN0214旧mask、HN0214修正mask；黑底列为官方原生诊断。'))
    doc.append('</section>')
    evidence = []
    for r in rows:
        if r['status'] == 'scored': continue
        sid = r['id']; item = inputs[sid]
        checked(item['input_path'], item['input_sha256'])
        doc += ['<section><h2>'+sid+' · '+html.escape(LABELS.get(r['status'], r['status']))+'</h2>',
            '<p><code>'+html.escape(str(r.get('reason') or r['status']))+'</code></p><div class="grid">',
            picture(item['input_path'],'实际生成输入（RGBA显示在白底）',rgba=True)]
        record = dict(id=sid,status=r['status'],reason=r.get('reason'),
            input_path=item['input_path'],input_sha256=item['input_sha256'],previews=[])
        for index in range(2):
            p = H/'runs/trellis2_human_native_v1'/sid/f'native_base_color_{index}.png'
            if p.exists():
                doc.append(picture(p,f'原生预览{index}（官方黑底与相机）'))
                record['previews'].append(dict(path=str(p),sha256=sha(p)))
            else:
                doc.append('<p>没有此原生预览；不以空白图片冒充生成结果。</p>')
        doc.append('</div></section>'); evidence.append(record)
    doc.append('</html>'); (out/'index.html').write_text(''.join(doc))
    save(out/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        summary_sha256=digest,counts=dict(counts),rows=evidence,html_sha256=sha(out/'index.html'),
        script_sha256=sha(__file__),replay=str(replay_path),replay_sha256=sha(replay_path),
        original_scores_changed=False,generation_retried=False,scope='Native failure diagnostics; not common-camera render comparison'))
    save(H/'configs/trellis2_failure_handling_v1.json',dict(
        user_instruction='失败案例保留；按能够正常生成并可统一评测的实际数量计分，正文披露失败情况',
        scope='Existing fixed100 TRELLIS.2 run',retry_failed_generation=False,
        retry_failed_registration=False,finish_existing_running_jobs=True,
        replace_failed_identities=False,quality_based_exclusion=False,
        corrected_mask_replay_replaces_production=False,denominator=100,
        comparison='same common valid identities across compared methods; method-only means labeled separately',
        disclosure='Separate native generation failures, registration-unavailable outputs, pending jobs, and input-mask limitations'))
    print(json.dumps(dict(report=str(out/'index.html'),counts=dict(counts)),ensure_ascii=False))


if __name__ == '__main__':
    main()
