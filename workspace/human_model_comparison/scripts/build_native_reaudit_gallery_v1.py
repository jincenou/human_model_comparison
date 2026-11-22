"""Source/full-scene/old/new visual evidence, without score-based selection."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import native_reaudit_common as common
from human_source_metrics_v1 import crop

ROOT,H=common.ROOT,common.H
FONT=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')


def images(method,sid,v):
    if method=='streetgs':
        old=common.read(v['old_validation'])['actor_preview']
        return v['fullscene_source'],old
    if method=='omnire':
        old=v.get('prior_registration')
        if isinstance(old,str):old=common.read(old)
        return v['full_scene_preview'],old.get('source_rgb') if old else None
    d=v['comparison_images']
    # Batch audit retains native target-only rendered before augmentation.
    return d.get('fullscene',d.get('full_scene')),d.get('native_target_only',d.get('target_only',d.get('old_actor')))


def tile(path,roi,size):
    if path and Path(path).is_file():
        a=np.asarray(Image.open(path).convert('RGB'))
        return Image.fromarray(crop(a,roi)).resize((size,size),Image.Resampling.LANCZOS)
    im=Image.new('RGB',(size,size),'#eeeeee')
    ImageDraw.Draw(im).text((12,size//2),'No native actor output',font=ImageFont.truetype(FONT,15),fill='#555555')
    return im


def build(method):
    contract=common.freeze();found=common.validations(method)
    out=ROOT/'gallery'/method;out.mkdir(parents=True,exist_ok=True)
    cols=['Source','Native scene','Old export','New export'];size=256;height=296
    cards=[];rows=[];records=[]
    for sid in contract['ids']:
        if sid not in found:continue
        vp,v=found[sid]
        if v['status']!='passed':continue
        ref=contract['references'][sid];assert v['source_reference']==ref
        full,old=images(method,sid,v)
        assert full and Path(full).is_file(),(method,sid,'missing native scene evidence')
        paths=[ref['reference_rgb'],full,old,v['actor_preview']]
        row=Image.new('RGB',(size*4,height),'white');draw=ImageDraw.Draw(row)
        for n,(label,p) in enumerate(zip(cols,paths)):
            draw.text((size*n+8,6),(sid+' | ' if n==0 else '')+label,font=ImageFont.truetype(FONT,18),fill='#172535')
            row.paste(tile(p,ref['roi_xyxy'],size),(n*size,36))
        dest=out/(sid+'.jpg');row.save(dest,quality=94,subsampling=0);rows.append(row)
        uri='data:image/jpeg;base64,'+base64.b64encode(dest.read_bytes()).decode()
        cards.append(f'<section id="{sid}"><h2>{sid}</h2><a href="{uri}" download="{method}_{sid}.jpg"><img src="{uri}" alt="{method} {sid} 同源视角对照"></a></section>')
        records.append(dict(id=sid,validation=str(vp),validation_sha256=common.sha(vp),
                            paths=paths,path_sha256=[common.sha(p) if p else None for p in paths],row=str(dest),row_sha256=common.sha(dest)))
    sheets=[]
    for start in range(0,len(rows),10):
        chosen=rows[start:start+10];sheet=Image.new('RGB',(size*4,height*len(chosen)),'#eeeeee')
        for i,row in enumerate(chosen):sheet.paste(row,(0,i*height))
        p=out/f'review_{start//10+1:02d}.jpg';sheet.save(p,quality=94,subsampling=0);sheets.append(str(p))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{method} 人体导出复核</title><style>body{{font:16px/1.65 system-ui;background:#edf2f7;max-width:1400px;margin:24px auto;padding:16px}}section{{background:white;padding:12px;margin:18px 0}}img{{width:100%;height:auto}}h1{{font-size:26px}}h2{{font-size:18px}}input{{padding:10px}}</style><h1>{html.escape(method)}：人体原场景与导出前后对照</h1>
<p>共 {len(rows)}/100 个技术通过对象，按冻结编号排列。四列依次为原照片、同曝光同相机场景、旧独立导出、新独立导出；共用冻结ROI，不单独缩放或移动人体，不对预测应用GT mask。</p><p>新导出补入源mask内正可见贡献的背景高斯，保留原生参数，可能同时保留背景与残影。技术通过不等于成功恢复完整人体；原场景缺失人体的情况仍然保留。这里是诊断展示，正式PSNR/SSIM/LPIPS另按既定二维对齐协议计算。</p><p><input placeholder="按编号查找" oninput="document.querySelectorAll('section').forEach(s=>s.hidden=!s.id.toLowerCase().includes(this.value.trim().toLowerCase()))"></p>'''+''.join(cards)+'</html>'
    (out/'index.html').write_text(page)
    common.save(out/'manifest.json',dict(method=method,count=len(rows),rows=records,contact_sheets=sheets,
        frozen_order=True,prediction_erased_by_reference_mask=False,independent_per_column_alignment=False,
        self_contained_html=True,script_sha256=common.sha(__file__)))
    print(method,len(rows),str(out/'index.html'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=common.METHODS,required=True);a=p.parse_args();build(a.method)
