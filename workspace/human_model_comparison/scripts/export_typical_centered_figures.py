"""Presentation-only centering of existing six-method source renders.

All source pixels are kept on white padded canvases. Translation is per panel;
scale is shared across each identity. No metric inputs or assets are modified.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from active_human_results import resolve

H = Path(__file__).resolve().parents[1]
OUT = H / 'reports/human_typical_comparison_20261114'
METHODS = ['reference', 'ours', 'ah', 'trellis_original', 'trellis2', 'omnire', 'splatad']
LABELS = ['Reference', 'Ours', 'AH', 'TRELLIS', 'TRELLIS.2', 'OmniRe', 'SplatAD']
# Visual coverage selection, not a representative quantitative sample.
SELECTED = {
    'HN0004': '背面、红色背包、短裤',
    'HN0040': '侧面、大步行走、短裤',
    'HN0051': '侧面、长发、连衣裙',
    'HN0116': '花纹连衣裙、肩包',
    'HN0126': '浅色上衣、手提袋',
    'HN0137': '站姿、长袖衬衫',
    'HN0141': '浅色衬衫、长裤、行走',
    'HN0144': '背面、连帽上衣、背包',
    'HN0150': '深色衣着、背包、屈臂',
    'HN0161': '浅色头发、短裤、背包',
    'HN0163': '侧背面、格纹长袖',
    'HN0166': '红色上衣、肩包、抬手',
    'HN0179': '侧面、短裤、双肩包',
    'HN0211': '背面、红色短袖',
    'HN0216': '侧背面、短裤、背包',
    'HN0228': '正侧面、长发、粉色鞋',
    'HN0262': '侧背面、抬腿步态',
    'HN0266': '正侧面、长发、手臂动作',
    'HN0273': '低头、蓝色上衣、手持物',
    'HN0281': '背面、裙装、花纹肩包',
    'HN0291': '连帽上衣、低头、手持物',
    'HN0309': '长发、深色长袖、屈臂',
    'HN0310': '侧面、大步行走、背包',
    'HN0321': '低头、手持牵引绳',
}
FONT = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labeled_grid(rows, labels, panel_size=512):
    header = round(panel_size / 8)
    im = Image.new('RGB', (panel_size * len(labels), header + panel_size * len(rows)), 'white')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, round(panel_size * .066))
    for col, label in enumerate(labels):
        draw.text((col * panel_size + panel_size // 2, header // 2), label, font=font, anchor='mm', fill='black')
    for row, panels in enumerate(rows):
        for col, panel in enumerate(panels):
            im.paste(panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS),
                     (col * panel_size, header + row * panel_size))
    return im


def main():
    gallery = resolve('gallery')
    source_manifest = json.loads((gallery / 'manifest.json').read_text())
    eligible = {r['id'] for r in source_manifest['rows'] if r['source_count'] == 6}
    assert set(SELECTED) <= eligible
    for folder in ['rows', 'models_only', 'sheets', 'previews']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for sid, reason in SELECTED.items():
        images, evidence, shifts = [], [], []
        for method in METHODS:
            path = gallery / 'source' / sid / (method + '.png')
            digest = sha(path)
            assert source_manifest['image_files'][str(path)] == digest
            im = Image.open(path).convert('RGB')
            assert im.size == (512, 512)
            arr = np.asarray(im)
            # Threshold only estimates display center; it never masks pixels.
            support = np.max(255 - arr.astype(np.int16), axis=2) > 12
            yy, xx = np.where(support)
            assert len(xx) > 100, (sid, method, 'empty image')
            bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            dx = round(256 - (bbox[0] + bbox[2]) / 2)
            dy = round(256 - (bbox[1] + bbox[3]) / 2)
            shifts.append((dx, dy))
            images.append(im)
            evidence.append(dict(method=method, source=str(path), sha256=digest,
                                 visible_bbox_xyxy=bbox, translation_xy=[dx, dy]))
        # Same padding and resize for all seven panels; include ENTIRE original.
        margin = max(24, max(abs(v) for pair in shifts for v in pair) + 16)
        size = 512 + 2 * margin
        panels = []
        for im, (dx, dy), item in zip(images, shifts, evidence):
            canvas = Image.new('RGB', (size, size), 'white')
            xy = (margin + dx, margin + dy)
            assert min(xy) >= 0 and max(xy) + 512 <= size
            canvas.paste(im, xy)
            assert np.array_equal(np.asarray(canvas.crop((*xy, xy[0] + 512, xy[1] + 512))), np.asarray(im))
            box = item['visible_bbox_xyxy']
            center = [(box[i] + box[i+2]) / 2 + xy[i] for i in range(2)]
            assert max(abs(v - size / 2) for v in center) <= .5
            item['center_error_output_px'] = [abs(v-size/2)*512/size for v in center]
            panels.append(canvas.resize((512, 512), Image.Resampling.LANCZOS))
        labeled_grid([panels], LABELS).save(OUT / 'rows' / (sid + '.png'))
        labeled_grid([panels[1:]], LABELS[1:]).save(OUT / 'models_only' / (sid + '.png'))
        labeled_grid([panels], LABELS, 240).save(OUT / 'previews' / (sid + '.jpg'), quality=92)
        rows.append(panels)
        records.append(dict(id=sid, selection_reason=reason, common_canvas_size=size,
                            common_scale=512/size, panels=evidence))
    for i in range(3):
        group = rows[i*8:(i+1)*8]
        labeled_grid(group, LABELS).save(OUT / 'sheets' / f'comparison_{i+1:02}.png')
        labeled_grid([r[1:] for r in group], LABELS[1:]).save(OUT / 'sheets' / f'models_only_{i+1:02}.png')
        labeled_grid(group, LABELS, 240).save(OUT / 'previews' / f'comparison_{i+1:02}.jpg', quality=94)
    labeled_grid(rows, LABELS, 320).save(OUT / 'all_24.png')
    previous = json.loads((resolve('advisor') / 'figures_manifest.json').read_text())['selected_ids']
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), selected_count=len(rows),
        previous_selected_ids=previous, added_ids=sorted(set(SELECTED)-set(previous)),
        six_method_source_available_count=len(eligible), columns=LABELS,
        selection_policy='Manual visual coverage of clothing, accessories, headings and gait; qualitative showcase, not quantitative sample.',
        observed_severe_examples_not_selected={'HN0108':'TRELLIS.2 duplicated/displaced body', 'HN0181':'TRELLIS.2 line-like degeneration',
            'HN0497':'TRELLIS.2 large black occluding surface', 'HN0233':'TRELLIS.2 severely fragmented body'},
        centering_policy='Visible nonwhite bounding-box midpoint, threshold 12/255, translation only; whole input retained with white padding; identical isotropic scale within identity.',
        original_artifacts_preserved=True, modifies_metric_inputs=False, changes_camera_or_pose=False,
        source_manifest=str(gallery/'manifest.json'), source_manifest_sha256=sha(gallery/'manifest.json'),
        script_sha256=sha(__file__), rows=records)
    manifest['output_images'] = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.suffix in ['.png','.jpg']}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    doc = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>24个人体资产 · 六模型对比</title><style>body{font:16px/1.6 system-ui;max-width:1800px;margin:28px auto;padding:0 20px;background:#f5f6f8;color:#243042}a{color:#175ba6}article{background:white;padding:16px;margin:20px 0}img{width:100%;display:block}h1{font-size:26px}nav{display:flex;gap:20px;flex-wrap:wrap}small{color:#596575}</style>',
        '<h1>24个人体资产 · 六模型对比</h1><p>Ours · AH · TRELLIS · TRELLIS.2 · OmniRe · SplatAD</p>',
        '<p><small>同一对象源视角，白底居中；同一行共用缩放比例，保留模型原有大小与质量差异。仅调整展示排版，指标仍使用原图。</small></p>',
        '<nav><a href="all_24.png">24人总图</a>']
    for i in range(3):
        doc.append(f'<a href="sheets/comparison_{i+1:02}.png">第{i+1}组高清图（8人）</a>')
    doc.append('<a href="manifest.json">来源与选样记录</a></nav>')
    for sid, reason in SELECTED.items():
        doc.append(f'<article><b>{sid}</b> · {html.escape(reason)} · <a href="rows/{sid}.png">高清图</a> · <a href="models_only/{sid}.png">仅六模型</a><a href="rows/{sid}.png"><img loading="lazy" src="previews/{sid}.jpg" alt="{sid}六模型对比"></a></article>')
    doc.append('</html>')
    (OUT/'index.html').write_text(''.join(doc))
    print(json.dumps(dict(output=str(OUT/'index.html'), objects=len(rows), new_objects=len(manifest['added_ids']), panels=len(rows)*7), ensure_ascii=False))


if __name__ == '__main__':
    main()
