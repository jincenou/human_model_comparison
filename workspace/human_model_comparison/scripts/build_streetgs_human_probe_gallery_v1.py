"""Fixed first-eight source-camera gallery; no score or common-set selection."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

H = Path(__file__).resolve().parents[1]
METHODS = [('streetgs', 'StreetGS'), ('splatad', 'SplatAD'), ('omnire', 'OmniRe'), ('ours', 'Ours')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(path, digest):
    assert sha(path) == digest, f'Hash mismatch: {path}'


def tile(rgb, roi):
    """One reference-defined ROI for every method; no prediction alignment."""
    left, top, right, bottom = map(int, roi)
    canvas = Image.new('RGB', (right-left, bottom-top), 'white')
    box = (max(0, left), max(0, top), min(rgb.width, right), min(rgb.height, bottom))
    if box[2] > box[0] and box[3] > box[1]:
        canvas.paste(rgb.crop(box), (box[0]-left, box[1]-top))
    return canvas.resize((512, 512), Image.Resampling.BICUBIC)


def absent(reason):
    im = Image.new('RGB', (512, 512), '#eeeeee')
    ImageDraw.Draw(im).text((160, 245), 'Not available', fill='#555555')
    return im, dict(status='not_available', reason=reason)


def prediction(sid, method, ref, camera, old_rows):
    if method == 'streetgs':
        path = H/'exports/streetgs_human_probe_v1/objects'/sid/'validation.json'
        if not path.exists():
            return absent('Export has not finished')
        record = read(path)
        if record['status'] != 'passed' or not record.get('reload_verified'):
            return absent(record.get('reason', record['status']))
        assert record['identity_key'] == ref['identity_key']
        assert record['timestamp_us'] == ref['timestamp_us']
        assert record['camera_contract_sha256'] == ref['camera_contract_sha256']
        rgbpath, alphapath = record['actor_preview'], record['actor_alpha']
        rgbhash, alphahash = record['actor_preview_sha256'], record['actor_alpha_sha256']
        encoding = record['prediction_encoding']
    else:
        row = old_rows.get((sid, method))
        if not row or not row.get('result'):
            return absent((row or {}).get('reason', 'No existing source-camera result'))
        verified(row['result'], row['result_sha256'])
        result = read(row['result'])
        path = Path(result['registration_record'])
        verified(path, result['registration_sha256'])
        record = read(path)
        assert record['identity'] == sid and record['identity_key'] == ref['identity_key']
        assert record['timestamp_us'] == ref['timestamp_us']
        assert record['camera_sha256'] == ref['camera_contract_sha256']
        rgbpath, alphapath = record['source_rgb'], record['source_alpha']
        rgbhash, alphahash = record['source_rgb_sha256'], record['source_alpha_sha256']
        encoding = record['prediction_encoding']
    assert encoding == 'white_composited_uint8', encoding
    verified(rgbpath, rgbhash)
    verified(alphapath, alphahash)
    rgb, alpha = Image.open(rgbpath).convert('RGB'), Image.open(alphapath).convert('L')
    assert list(rgb.size) == camera['render_wh'] and alpha.size == rgb.size
    image = tile(rgb, ref['roi_xyxy'])
    return image, dict(status='render_available', record=str(path), record_sha256=sha(path),
                      source_rgb=rgbpath, source_rgb_sha256=rgbhash,
                      source_alpha=alphapath, source_alpha_sha256=alphahash,
                      prediction_encoding=encoding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=H/'reports/streetgs_human_probe_v1/gallery')
    args = parser.parse_args()
    selection = H/'inputs/benchmark_human100_v3/selection.json'
    frozen = read(selection)['objects']
    ids = [row['id'] for row in frozen[:8]]
    qualification = H/'configs/human_source_common_v1/reference_qualification_frozen_v5.json'
    refs = {row['id']: row for row in read(qualification)['objects']}
    active = H/'inputs/active_results.json'
    summary_path = H/read(active)['core_summary']
    summary = read(summary_path)
    old_rows = {(row['id'], row['method']): row for row in summary['rows']}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    labels = ['Reference'] + [label for _, label in METHODS]
    sheet = Image.new('RGB', (256*len(labels), 286*len(ids)), 'white')
    draw = ImageDraw.Draw(sheet)
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>StreetGS 人体静态提取核查</title>',
             '<style>body{font:15px system-ui;margin:24px;color:#222;background:#f7f7f7}',
             'section{padding:16px;background:white;margin:18px 0}.scroll{overflow:auto}',
             '.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;min-width:900px}',
             'img{width:100%}figure{margin:0;text-align:center}small{display:block;color:#555}',
             'p{line-height:1.65}</style><h1>StreetGS 人体静态提取核查</h1>',
             '<p>固定展示冻结名单的前 8 人，不按分数或共同成功范围挑选。每行使用同一真实曝光、',
             '源相机和参考确定的裁剪范围；预测图保留自身背景混入、姿态和尺度误差，未做二维对齐。</p>',
             '<p>StreetGS 为原场景静态背景高斯的固定三维框提取，参数未修改；它不是恢复的人体动态节点。',
             '其他列复用已验证的源相机渲染。灰格表示本次没有对应可用输出，不代表模型的预测图。</p>']
    evidence = []
    for row_index, sid in enumerate(ids):
        ref = refs[sid]
        verified(ref['camera_contract'], ref['camera_contract_sha256'])
        camera = read(ref['camera_contract'])
        for name in ['reference_rgb', 'reference_mask']:
            verified(ref[name], ref[name+'_sha256'])
        rgb = np.asarray(Image.open(ref['reference_rgb']).convert('RGB'))
        mask = np.asarray(Image.open(ref['reference_mask']).convert('L')) > 127
        assert rgb.shape[:2] == mask.shape and list(rgb.shape[1::-1]) == camera['render_wh']
        target = tile(Image.fromarray(np.where(mask[..., None], rgb, 255).astype('uint8')), ref['roi_xyxy'])
        images, records = [target], []
        for method, _ in METHODS:
            im, record = prediction(sid, method, ref, camera, old_rows)
            images.append(im)
            records.append(dict(method=method, **record))
        parts.append('<section><h2>'+html.escape(sid)+'</h2><div class="scroll"><div class="grid">')
        for col, (label, im) in enumerate(zip(labels, images)):
            data = io.BytesIO()
            im.save(data, format='PNG')
            uri = 'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode('ascii')
            parts.append('<figure><b>'+label+'</b><img src="'+uri+'">')
            if col and records[col-1]['status'] == 'not_available':
                parts.append('<small>'+html.escape(records[col-1]['reason'])+'</small>')
            parts.append('</figure>')
            sheet.paste(im.resize((256, 256), Image.Resampling.LANCZOS), (256*col, 286*row_index+28))
            draw.text((256*col+6, 286*row_index+7), sid+' '+label, fill='black')
        parts.append('</div></div></section>')
        evidence.append(dict(id=sid, identity_key=ref['identity_key'], timestamp_us=ref['timestamp_us'],
                             reference=ref, outputs=records))
    parts.append('</html>')
    (out/'index.html').write_text(''.join(parts))
    sheet.save(out/'comparison.jpg', quality=95)
    manifest = dict(schema='streetgs_human_fixed_first8_gallery_v1', ids=ids,
                    selection_rule='first eight identities in immutable frozen order; no success or score filtering',
                    selection=str(selection), selection_sha256=sha(selection),
                    reference_qualification=str(qualification), reference_qualification_sha256=sha(qualification),
                    existing_summary=str(summary_path), existing_summary_sha256=sha(summary_path),
                    fixed_source_camera=True, prediction_based_2d_alignment=False,
                    prediction_erased_by_reference_mask=False, scores_produced=False,
                    methods=[key for key, _ in METHODS], rows=evidence,
                    script_sha256=sha(__file__), html_sha256=sha(out/'index.html'),
                    contact_sheet_sha256=sha(out/'comparison.jpg'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print(out/'index.html')
    print(out/'comparison.jpg')


if __name__ == '__main__':
    main()
