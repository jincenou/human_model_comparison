"""Portable frozen identity gallery; stored previews are explicitly not t_ref renders."""
import base64
import html
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from build_candidate_review_v2 import first_video, panel, sha, FONT

H=Path(__file__).resolve().parents[1]
OUT=H/'reports/human100_frozen_v2'


def main():
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    data=json.loads(selection.read_text());rows=data['objects']
    assert len(rows)==100 and len({r['identity_key'] for r in rows})==100
    OUT.mkdir(exist_ok=False);(OUT/'media').mkdir()
    cards=[];records=[]
    labels=['Real: frozen t_ref','SAM2 candidate: t_ref','White reference: t_ref',
            'Actual input: slot 0','Stored Ours: view 00','Stored Ours: view 08']
    for row in rows:
        ident=row['id'];ref=row['reference_frame'];paths=row['paths']
        assert sha(ref['path'])==ref['sha256']
        mask_path=Path(row['reference_mask_candidate']['path'])
        assert sha(mask_path)==row['reference_mask_candidate']['sha256']
        real=np.array(Image.open(ref['path']).convert('RGB'))
        mask=np.array(Image.open(mask_path).crop(ref['crop_xyxy']))>127
        assert mask.shape==real.shape[:2]
        overlay=real.copy();overlay[mask]=(real[mask]*.55+np.array([0,210,120])*.45).astype('uint8')
        white=np.where(mask[...,None],real,255).astype('uint8')
        stored=first_video(paths['asset_preview']);assert stored.shape[1]%5==0
        width=stored.shape[1]//5
        ims=[real,overlay,white,first_video(paths['input_video']),stored[32:,3*width:4*width],stored[32:,4*width:5*width]]
        canvas=Image.new('RGB',(1440,390),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,3),f'{row["benchmark_number"]:03d} {ident} | {row["cohort"]} | {row["scene"]} | t_ref={row["t_ref_us"]}',fill='black',font=FONT)
        for i,(im,label) in enumerate(zip(ims,labels)):
            draw.text((i*240+3,28),label,fill='black',font=FONT);canvas.paste(panel(im),(i*240,50))
        dest=OUT/'media'/f'{ident}.jpg';canvas.save(dest,quality=88)
        rec=dict(id=ident,benchmark_number=row['benchmark_number'],cohort=row['cohort'],scene=row['scene'],
                 identity_key=row['identity_key'],t_ref_us=row['t_ref_us'],candidate_review_number=row['candidate_review_number'],
                 image=f'media/{ident}.jpg',sha256=sha(dest),reference_mask_status='candidate_pending_final_qualification',
                 ours_preview_status='stored_first_frame_context_not_frozen_t_ref_reload',metric_eligible=False)
        records.append(rec)
        uri='data:image/jpeg;base64,'+base64.b64encode(dest.read_bytes()).decode()
        cohort='原75人' if row['cohort']=='original75' else '新增25人'
        cards.append(f'<article data-search="{ident.lower()} {row["scene"]} {row["cohort"]}"><h2>{row["benchmark_number"]:03d} · {ident} · {cohort} · {row["scene"]}</h2><img src="{uri}" loading="lazy" alt="{ident}冻结身份资料"><p>冻结真实时刻：{row["t_ref_us"]} μs。前三列为该时刻参考；后三列为已有首帧输入和资产预览。参考资格、指定时刻重载和公共相机尚待验收。</p><details><summary>身份与来源记录</summary><pre>{html.escape(json.dumps(row,ensure_ascii=False,indent=2))}</pre></details></article>')
        print(ident,flush=True)
    excluded='、'.join(r['review_number']+' '+r['id'] for r in data['exclusions_this_review'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nuScenes 冻结100人体</title><style>body{max-width:1500px;margin:auto;padding:20px;background:#f4f5f7;font:16px system-ui;color:#17202c}article{background:white;padding:16px;margin:18px 0;border:1px solid #ccd3dc;border-radius:8px}img{width:100%;height:auto}p{line-height:1.6}h2{font-size:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}input{padding:12px;min-width:300px}header{background:#fff;padding:18px}</style><header><h1>nuScenes 冻结100人体：原75人＋新增25人</h1><p>身份名单已按用户审核冻结，覆盖冻结100场景中的66个场景。失败不会换人。此页是名单及参考审核资料，不是正式成绩或同相机结果。</p><p>列顺序：冻结时刻真实图 / 同时刻SAM2候选mask / 同时刻白底参考 / 实际首帧模型输入 / 已有Ours首帧视角00 / 已有Ours首帧视角08。mask仍需最终资格验收，存量预览不能冒充冻结时刻的新重载。</p><p>本次排除：'''+html.escape(excluded)+'''。</p><input id="search" placeholder="筛选 HN编号、场景或 original75/new25"><span id="count">100 人</span></header>'''+''.join(cards)+'''<script>document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.toLowerCase().trim();let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!a.dataset.search.includes(q);if(!a.hidden)n++});document.querySelector('#count').textContent=n+' 人'})</script></html>'''
    (OUT/'index.html').write_text(page)
    manifest=dict(status='identity_list_frozen_reference_qualification_pending',selection_path=str(selection),
                  selection_sha256=sha(selection),count=100,original75=75,new25=25,records=records)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for start in range(0,100,6):
        group=records[start:start+6];sheet=Image.new('RGB',(1440,390*len(group)),'white')
        for i,rec in enumerate(group):sheet.paste(Image.open(OUT/rec['image']),(0,390*i))
        sheet.save(OUT/f'contact_{start//6+1:02d}.jpg',quality=85)
    with zipfile.ZipFile(OUT/'human100_frozen_v2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['index.html','manifest.json']:z.write(OUT/name,name)
        z.write(selection,'selection.json')
        z.write(selection.parent/'identities.csv','identities.csv')
    assert page.count('<article ')==100 and page.count('data:image/jpeg;base64,')==100
    with zipfile.ZipFile(OUT/'human100_frozen_v2.zip') as z:assert z.testzip() is None
    print(json.dumps(dict(count=100,selection_sha256=sha(selection),output=str(OUT))))


if __name__=='__main__':main()
