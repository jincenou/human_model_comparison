"""Show exactly the 17 StreetGS ED-P-valid people, in frozen cohort order."""
from human_asset_benchmark.paths import expand as _hmc_expand
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_streetgs_human_probe_gallery_v1 as old

H=old.H
ROOT=H/'reports/streetgs_human_probe_v1'
OUT=ROOT/'gallery_edp_valid17_v1'
PART_NAMES=dict(head='头部',torso='躯干',left_arm='左臂',right_arm='右臂',left_leg='左腿',right_leg='右腿')


def font(size):
    p=Path(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'))
    assert p.is_file()
    return ImageFont.truetype(str(p),size)


def main():
    score_path=ROOT/'edp/all/results.json';scores=old.read(score_path)
    valid={r['id']:r for r in scores['rows'] if r['method']=='streetgs' and r['score']['status']=='ok'}
    assert len(valid)==17
    selection=H/'inputs/benchmark_human100_v3/selection.json'
    ids=[r['id'] for r in old.read(selection)['objects'] if r['id'] in valid]
    assert len(ids)==len(set(ids))==17 and set(ids)==set(valid)
    summary_path=H/old.read(H/'inputs/active_results.json')['core_summary']
    summary=old.read(summary_path)
    assert summary['selection_sha256']==old.sha(selection)
    old_rows={(r['id'],r['method']):r for r in summary['rows']}
    refs={}
    for r in summary['rows']:
        if not r.get('reference'):continue
        if r['id'] in refs:assert refs[r['id']]==r['reference']
        refs[r['id']]=r['reference']
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'rows').mkdir(exist_ok=True)
    labels=['Reference']+[label for _,label in old.METHODS]
    records=[];row_images=[];csv_rows=[];sections=[]
    for ordinal,sid in enumerate(ids,1):
        ref=refs[sid];entry=valid[sid]
        for name in ['camera_contract','reference_rgb','reference_mask']:
            old.verified(ref[name],ref[name+'_sha256'])
        camera=old.read(ref['camera_contract'])
        rgb=np.asarray(Image.open(ref['reference_rgb']).convert('RGB'))
        mask=np.asarray(Image.open(ref['reference_mask']).convert('L'))>127
        assert rgb.shape[:2]==mask.shape and list(rgb.shape[1::-1])==camera['render_wh']
        target=old.tile(Image.fromarray(np.where(mask[...,None],rgb,255).astype('uint8')),ref['roi_xyxy'])
        images=[target];predictions=[]
        for method,_ in old.METHODS:
            tile,evidence=old.prediction(sid,method,ref,camera,old_rows)
            if method=='streetgs':assert evidence['status']=='render_available'
            if evidence['status']!='render_available':
                tile=Image.new('RGB',(512,512),'#eeeeee')
                ImageDraw.Draw(tile).text((256,256),'No available output',font=font(28),anchor='mm',fill='#666666')
            images.append(tile);predictions.append(dict(method=method,**evidence))
        # Check that these are the exact unaligned pixels supplied to ED-P.
        score_row=old.read(ROOT/'static_rows'/(sid+'.json'))
        for tile,key in [(images[0],'reference'),(images[1],'prediction_unwarped')]:
            f=score_row['files'];old.verified(f[key],f[key+'_sha256'])
            assert np.array_equal(np.asarray(tile),np.asarray(Image.open(f[key]).convert('RGB'))),(sid,key)
        row=Image.new('RGB',(512*5,554),'white');draw=ImageDraw.Draw(row)
        for col,(label,tile) in enumerate(zip(labels,images)):
            draw.text((512*col+12,9),label,font=font(24),fill='#182534')
            row.paste(tile,(512*col,42))
        path=OUT/'rows'/(sid+'.png');row.save(path)
        row_images.append(row)
        parts='、'.join(PART_NAMES[k] for k in entry['valid_parts'])
        evidence=dict(id=sid,ordinal=ordinal,identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],
                      edp=entry['score']['value'],valid_parts=entry['valid_parts'],valid_part_count=entry['valid_part_count'],
                      reference=ref,outputs=predictions,row_png=str(path),row_png_sha256=old.sha(path),
                      exact_edp_input_pixels_verified=True)
        records.append(evidence)
        csv_rows.append(dict(id=sid,ordinal=ordinal,EDP=entry['score']['value'],valid_part_count=entry['valid_part_count'],
                             valid_parts=';'.join(entry['valid_parts']),
                             unavailable_methods=';'.join(r['method'] for r in predictions if r['status']!='render_available')))
        uri='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')
        missing=[label for (key,label),r in zip(old.METHODS,predictions) if r['status']!='render_available']
        sections.append('<section data-id="'+sid+'"><div class="heading"><h2>'+str(ordinal)+'. '+sid+'</h2>'
            '<span>StreetGS ED-P ↓：'+f"{entry['score']['value']:.3f}"+' ｜ 有效部位：'+str(entry['valid_part_count'])+'/6</span>'
            '<button onclick="saveRow(this)">下载此行 PNG</button></div><p class="parts">计分部位：'+parts+'</p>'
            +('<p class="missing">缺少对应历史输出：'+html.escape('、'.join(missing))+'；灰格表示缺失。</p>' if missing else '')
            +'<div class="scroll"><img src="'+uri+'" alt="'+sid+'：原图与四种方法的源视角对比"></div></section>')
    sheets=[]
    for start in range(0,len(ids),6):
        subset=row_images[start:start+6];sheet=Image.new('RGB',(1600,384*len(subset)),'#f5f7fa');draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(subset):
            entry=valid[ids[start+n]]
            draw.text((10,384*n+5),f"{start+n+1:02d}  {ids[start+n]}   StreetGS ED-P: {entry['score']['value']:.3f}   Parts: {entry['valid_part_count']}/6",font=font(20),fill='#182534')
            sheet.paste(row.resize((1600,346),Image.Resampling.LANCZOS),(0,384*n+34))
        path=OUT/f'comparison_{start//6+1:02d}.jpg';sheet.save(path,quality=95)
        sheets.append(dict(path=str(path),sha256=old.sha(path),ids=ids[start:start+6]))
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetGS ED-P 可评分的17个人体对比</title><style>
body{font:16px system-ui,sans-serif;background:#f3f5f8;color:#172535;margin:0}main{max-width:1500px;margin:auto;padding:24px}
h1{font-size:26px}p{line-height:1.7}section{background:white;border:1px solid #dce2e9;border-radius:10px;margin:22px 0;padding:16px}
.heading{display:flex;align-items:center;gap:22px;flex-wrap:wrap}h2{font-size:19px;margin:0}.parts{color:#536477;margin:12px 0}.missing{color:#79562c;font-size:14px}
.scroll{overflow-x:auto}img{display:block;width:100%;min-width:1050px;height:auto}button{padding:8px 14px;background:#eaf0f7;border:0;border-radius:6px;cursor:pointer}
input{padding:10px;width:260px;border:1px solid #bfcbd8;border-radius:6px}.lead{padding:16px;background:#e8eef6;border-radius:8px}
</style></head><body><main><h1>StreetGS ED-P 可评分的17个人体对比</h1>
<div class="lead"><p>完整展示实际纳入 StreetGS ED-P 均值的17人，按原冻结名单顺序排列，不按得分高低排序，不要求其他模型也成功。</p>
<p>列顺序：Reference、StreetGS、SplatAD、OmniRe、Ours。每行使用相同源曝光、相机和参考 ROI；不额外调整预测朝向或大小，不用 GT mask 擦除预测。Reference 与 StreetGS 的图片已逐像素核对为实际 ED-P 输入。</p>
<p>“可评分”表示至少一个身体部位满足既有计算条件，不表示人物完整或重建质量通过。StreetGS 为原场景背景高斯的三维框提取，保留残影和背景混入。没有可用历史输出的模型保留灰格。</p>
<p>本 HTML 已内嵌全部图片，可单独下载后离线打开；每个人的整行 PNG 可在对应按钮下载。</p></div>
<p><input placeholder="输入对象编号，例如 HN0646" aria-label="按对象编号查找" oninput="filterRows(this.value)"> <span id="count">17 / 17 人</span></p>
'''+''.join(sections)+'''</main><script>
function filterRows(v){let n=0;document.querySelectorAll('section').forEach(s=>{let show=s.dataset.id.toLowerCase().includes(v.trim().toLowerCase());s.hidden=!show;if(show)n++});document.getElementById('count').textContent=n+' / 17 人'}
function saveRow(b){let s=b.closest('section'),a=document.createElement('a');a.href=s.querySelector('img').src;a.download=s.dataset.id+'_comparison.png';a.click()}
</script></body></html>'''
    (OUT/'index.html').write_text(page)
    with (OUT/'scores_and_parts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    manifest=dict(schema='streetgs_edp_valid17_source_gallery_v1',ids=ids,
                  selection_rule='All and only StreetGS ED-P status=ok; immutable cohort order; no ranking or cross-method intersection',
                  selection=str(selection),selection_sha256=old.sha(selection),edp_results=str(score_path),edp_results_sha256=old.sha(score_path),
                  active_summary=str(summary_path),active_summary_sha256=old.sha(summary_path),rows=records,contact_sheets=sheets,
                  self_contained_html=True,fixed_source_camera=True,prediction_based_2d_alignment=False,prediction_erased_by_reference_mask=False,
                  script_sha256=old.sha(__file__),helper_sha256=old.sha(old.__file__),html_sha256=old.sha(OUT/'index.html'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(count=len(ids),html=str(OUT/'index.html'),contact_sheets=[x['path'] for x in sheets],
                         missing=[dict(id=r['id'],method=x['method']) for r in records for x in r['outputs'] if x['status']!='render_available']),ensure_ascii=False))


if __name__=='__main__':main()
