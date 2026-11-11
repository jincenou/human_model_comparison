"""Create a portable, explicitly preliminary review of unexcluded nuScenes candidates."""
from human_asset_benchmark.paths import expand as _hmc_expand
import base64
import hashlib
import html
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/nuscenes_cohort_audit_v2/audit.json'
OUT = ROOT / 'reports/nuscenes_candidate_review_v2'
cv2.setNumThreads(1)
FONT = ImageFont.truetype(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'), 16)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def first_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open ' + str(path))
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Cannot decode first frame: ' + str(path))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def panel(image, width=240, height=340):
    return ImageOps.pad(Image.fromarray(image), (width, height), method=Image.Resampling.LANCZOS, color=(235,235,235))


def main():
    audit = json.loads(AUDIT.read_text())
    rows = sorted((r for r in audit['records'] if r['cohort'] == 'candidate44' and not r['blocked_by_prior_exclusion']), key=lambda r:r['id'])
    assert len(rows) == 37
    OUT.mkdir(parents=True, exist_ok=True)
    media = OUT / 'media'
    media.mkdir(exist_ok=True)
    records, cards = [], []
    for n, row in enumerate(rows, 1):
        ident, paths = row['id'], row['paths']
        native = json.loads(Path(paths['native_metadata']).read_text())
        mapping = json.loads(Path(paths['frame_mapping']).read_text())
        first = native['frames'][0]
        slot = mapping['frames'][0]
        assert not slot['interpolated'] and slot['left_native_index'] == slot['right_native_index'] == 0
        raw0 = np.asarray(Image.open(first['path']).convert('RGB'))
        rawref = np.asarray(Image.open(row['reference_frame']['path']).convert('RGB'))
        with np.load(paths['mask_npz'], allow_pickle=False) as z:
            mask0 = z['masks'][0]
        if mask0.dtype != np.uint8:
            raise ValueError((ident, 'Unexpected mask dtype', str(mask0.dtype)))
        mask0 = mask0 > 127
        assert mask0.shape == raw0.shape[:2], (ident, 'mask geometry')
        overlay = raw0.copy()
        overlay[mask0] = (overlay[mask0].astype(float) * .55 + np.array([0,210,120]) * .45).astype(np.uint8)
        preview = first_video(paths['asset_preview'])
        assert preview.shape[1] % 5 == 0 and preview.shape[0] > 32, (ident,'unexpected stored preview layout')
        w = preview.shape[1] // 5
        reference_folder = ROOT / 'references/exact_exposure_v2' / ident
        reference_meta = json.loads((reference_folder/'metadata.json').read_text())
        assert reference_meta['timestamp_us'] == row['t_ref_us']
        assert reference_meta['source_sha256'] == row['reference_raw_sha256']
        assert reference_meta['mask_sha256'] == sha(reference_folder/'mask.png')
        exact_mask = np.asarray(Image.open(reference_folder/'mask_crop.png')) > 127
        assert exact_mask.shape == rawref.shape[:2]
        exact_overlay = rawref.copy()
        exact_overlay[exact_mask] = (rawref[exact_mask].astype(float)*.55 + np.array([0,210,120])*.45).astype(np.uint8)
        white = np.where(exact_mask[:,:,None],rawref,255).astype(np.uint8)
        panels = [rawref, exact_overlay, white, first_video(paths['input_video']), preview[32:,3*w:4*w], preview[32:,4*w:5*w]]
        labels = ['Real: proposed t_ref', 'SAM2 mask: t_ref', 'Masked reference: t_ref', 'Actual input: slot 0', 'Stored Ours: view 00', 'Stored Ours: view 08']
        canvas = Image.new('RGB', (1440,390), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((6,3),f'C{n:02d} {ident} | {row["scene"]} | t_ref={row["t_ref_us"]} us | real={row["real_exposures"]}, model slots={row["model_input_slots"]}',fill='black',font=FONT)
        for j,(im,label) in enumerate(zip(panels,labels)):
            draw.text((j*240+3,28),label,fill='black',font=FONT)
            canvas.paste(panel(im),(j*240,50))
        name = f'{ident}.jpg'
        canvas.save(media/name, quality=90)
        hashes = {k:sha(paths[k]) for k in ['mask_npz','input_video','asset_preview']}
        rec = {'id':ident,'review_number':f'C{n:02d}','scene':row['scene'],'identity_key':row['identity_key'],
            't_ref_proposed_us':row['t_ref_us'],'first_exposure_us':first['frame_timestamp_micros'],
            'native_exposures':row['real_exposures'],'model_slots':row['model_input_slots'],
            'first_mask_area':int(mask0.sum()),'paths':paths,'hashes':hashes,
            'exact_input_mask_slots_at_t_ref':row['exact_input_slots_for_reference'],
            'new_exact_reference_mask':reference_meta,
            'runtime_reload_valid':False,'reference_visual_review':'pending','selected':False,
            'preview':'Stored first-frame Grid4D preview; not a new native reload or t_ref rendering.',
            'image':f'media/{name}'}
        records.append(rec)
        image='data:image/jpeg;base64,'+base64.b64encode((media/name).read_bytes()).decode()
        masknote = '前三列均为拟用参考时刻；新SAM2 mask仍待逐项审核。后三列属于原首帧输入与已有资产预览'
        cards.append(f'<article><h2>{rec["review_number"]} · {ident} · {html.escape(row["scene"])}</h2><img src="{image}" alt="{ident}真实参考与已有输入资产" loading="lazy"><p>拟用真实参考时刻 {rec["t_ref_proposed_us"]} μs。<b>{masknote}</b>。</p><details><summary>身份与来源</summary><pre>{html.escape(json.dumps(rec,ensure_ascii=False,indent=2))}</pre></details></article>')
        print(ident, 'media_ready', flush=True)
    manifest={'status':'preliminary_inventory_not_25person_approval','audit':str(AUDIT),'audit_sha256':sha(AUDIT),
        'scope':'37 unexcluded candidates within frozen100 scenes, no Waymo, original75 unchanged',
        'excluded_prior_ids':[r['id'] for r in audit['records'] if r['cohort']=='candidate44' and r['blocked_by_prior_exclusion']],
        'limitations':['Not a proposed or frozen25 list.','Runtime render reload and exact reference-mask review are pending.',
            'Stored preview is first-frame context and is not the proposed t_ref output.','Images are scaled for human review only, not metric inputs.'],
        'records':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nuScenes新增人物：37候选初审</title><style>body{max-width:1500px;margin:auto;padding:20px;background:#f4f5f7;font:16px system-ui;color:#17202c}article{background:white;padding:16px;margin:18px 0;border:1px solid #ccd3dc;border-radius:8px}img{width:100%;height:auto}p{line-height:1.6}h2{font-size:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style><h1>nuScenes 新增人物：37候选初审</h1><p>原75人保留。44个库存候选中7个有历史排除记录，本页只展示其余37人。尚未提出或冻结新增25人名单；仍需逐人参考审核和runtime重载渲染验证。本页供核查进度，不是最终批准请求。</p><p>列顺序：拟用参考时刻真实图 / 同时刻新SAM2 mask叠加 / 同时刻白底参考 / 实际首帧模型输入 / 已有资产首帧视角00 / 已有资产首帧视角08。后三列属于首帧上下文，不冒充参考时刻结果。新mask只是自动候选，尚未完成全部目视验收。图片已内嵌，可离线打开。</p>'''+''.join(cards)+'</html>'
    (OUT/'index.html').write_text(page)
    for start in range(0,len(records),6):
        subset=records[start:start+6]
        sheet=Image.new('RGB',(1440,390*len(subset)),'white')
        for j,rec in enumerate(subset):
            sheet.paste(Image.open(OUT/rec['image']),(0,390*j))
        sheet.save(OUT/f'contact_{start//6+1:02d}.jpg',quality=88)
    with zipfile.ZipFile(OUT/'nuscenes_candidate_review_v2.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'index.html','index.html')
        z.write(OUT/'manifest.json','manifest.json')
    print(json.dumps({'candidates':len(records),'output':str(OUT),'html_bytes':(OUT/'index.html').stat().st_size}))


if __name__ == '__main__':
    main()
