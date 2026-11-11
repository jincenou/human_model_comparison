"""Freeze 100 visually checked people and make a self-contained offline review page."""
import base64,collections,html,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
C=R/'inputs/candidates.json';MEDIA=R/'reports/candidates';OUT=R/'reports/human100_review_v1'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
data=json.loads(C.read_text());rows=data['candidates']
manual=R/'inputs/manual_decisions.json'
decisions=json.loads(manual.read_text()) if manual.exists() else {}
selected=[];count=collections.Counter()
for r in rows:
    if decisions.get(r['id'])!='keep':continue
    if count[r['dataset']]>=({'nuscenes':80,'waymo':20}[r['dataset']]):continue
    ev=MEDIA/r['id']/'evidence.json'
    assert ev.exists(),r['id'];v=json.loads(ev.read_text())
    row=dict(r,number=len(selected)+1,candidate_number=r['candidate_number'],evidence=str(ev),
             source_video_sha256=v['hashes']['source_video'],mask_sha256=v['hashes']['mask_npz'],
             actual_input_sha256=v['hashes']['input_video'],asset_preview_sha256=v['hashes']['asset_preview'],
             sampled_slots=v['sampled_slots'],review_status='proposed_after_direct_visual_review')
    selected.append(row);count[r['dataset']]+=1
assert count=={'nuscenes':80,'waymo':20},count
OUT.mkdir(parents=True,exist_ok=True)
manifest={'status':'proposed_for_user_review','total':len(selected),'dataset_counts':dict(count),
          'selection_basis':'Direct first-frame visual review of native observed crop, target mask, actual model input and Grid4D view00/view08. Three-timepoint comparisons were generated for user review. Existing quality flags and production provenance also used.',
          'limitations':['Curated from existing assets, not random population.','Native source crop and interpolated model slots are distinguished.','Asset preview reflects preserved older Grid4D render; no new renderer or quality certification.','Middle and final frames require further individual visual review.','Long-sequence motion, grounding and strict heldout metrics not certified by static review.'],
          'excluded_after_visual_review':{k:v for k,v in decisions.items() if v!='keep'},'items':selected}
write(OUT/'selection.json',manifest)
def datauri(p):return 'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()
cards=[]
for r in selected:
    id=r['id'];first=datauri(MEDIA/id/'first.jpg');contact=datauri(MEDIA/id/'contact.jpg')
    label=f"{r['number']:03d} · {id} · {r['dataset']} · {r['scene']} · {r['motion_type']}"
    cards.append(f'''<article class="card" data-dataset="{r['dataset']}" data-id="{id}">
      <header><b>{html.escape(label)}</b><span>候选 {r['candidate_number']:03d}　输入人物中位高度 {r['height_median_px']:.0f}px</span></header>
      <img class="first" src="{first}" alt="{html.escape(label)} 首帧：真实观测、mask、模型输入、资产正侧视" loading="lazy">
      <details><summary>查看前／中／后三个时刻与溯源</summary><img class="detail" src="{contact}" alt="{html.escape(label)} 三时刻完整对照" loading="lazy"><p>原生视频：{html.escape(r['paths']['source_video'])}<br>实际输入：{html.escape(r['paths']['input_video'])}<br>分割：{html.escape(r['paths']['mask_npz'])}<br>资产：{html.escape(r['paths']['runtime'])}</p></details>
      <label><input type="checkbox" class="reject"> 标记排除这个编号</label>
    </article>''')
body='\n'.join(cards)
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人体 100 对照审核</title>
<style>body{{font:16px system-ui,sans-serif;background:#f4f5f7;color:#17202c;margin:0 auto;padding:24px;max-width:1180px}}h1{{margin:0 0 8px}}.toolbar{{position:sticky;top:0;background:#f4f5f7ee;padding:12px 0;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}button,select{{padding:8px 12px;border:1px solid #adb8c5;border-radius:6px;background:white;cursor:pointer}}.card{{background:white;border:1px solid #d8dfe7;border-radius:10px;padding:14px;margin:16px 0;box-shadow:0 1px 3px #aaa4}}.card header{{display:flex;gap:10px;justify-content:space-between;flex-wrap:wrap;margin-bottom:8px}}.card header span{{color:#626f80;font-size:14px}}img{{display:block;width:100%;height:auto}}.first{{max-height:370px;object-fit:contain;object-position:left;background:#eee}}summary{{cursor:pointer;margin:10px 0;color:#0759a2}}details p{{font-size:11px;word-break:break-all;color:#667}}.card:has(.reject:checked){{outline:3px solid #dc8b46}}.note{{font-size:14px;color:#526075;line-height:1.5}}#picked{{font-variant-numeric:tabular-nums}}@media(max-width:720px){{body{{padding:12px}}.card header{{display:block}}}}</style></head><body>
<h1>人体候选 100 个</h1><p>nuScenes 80 个，Waymo 20 个。编号属于本轮人体，与车辆编号独立。逐项显示原始观测裁剪、mask、实际模型输入，以及已有资产 view 00 / 08。展开后可看三个时刻。原图与插帧时刻的区别已在图中标记。</p>
<p class="note">这是供你排除的候选名单。我们按输入与资产的可辨识对应关系筛选，尚未把动作、接地、真实留出视角或几何精度判为合格。真实观测、mask 和资产均未在此重新生成。</p>
<div class="toolbar"><select id="filter"><option value="all">全部 100</option><option value="nuscenes">nuScenes 80</option><option value="waymo">Waymo 20</option></select><span id="picked">已标记排除 0 个</span><button id="copy">复制排除编号</button><button id="download">下载排除清单</button></div>
{body}<script>const key='human100_exclusions_v1';let saved=JSON.parse(localStorage.getItem(key)||'[]');document.querySelectorAll('.card').forEach(c=>{{let box=c.querySelector('.reject');box.checked=saved.includes(c.dataset.id);box.onchange=update}});function update(){{let ids=[...document.querySelectorAll('.card')].filter(c=>c.querySelector('.reject').checked).map(c=>c.dataset.id);localStorage.setItem(key,JSON.stringify(ids));document.querySelector('#picked').textContent='已标记排除 '+ids.length+' 个';return ids}}update();document.querySelector('#filter').onchange=e=>document.querySelectorAll('.card').forEach(c=>c.hidden=e.target.value!=='all'&&c.dataset.dataset!==e.target.value);document.querySelector('#copy').onclick=()=>navigator.clipboard.writeText(update().join('、')).catch(()=>alert('浏览器不允许复制时，请使用下载排除清单'));document.querySelector('#download').onclick=()=>{{let ids=update();let blob=new Blob([JSON.stringify({{human_review_exclusions:ids}},null,2)],{{type:'application/json'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human100_exclusions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)}};</script></body></html>'''
(OUT/'index.html').write_text(page)
with zipfile.ZipFile(OUT/'human100_review_v1.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    z.write(OUT/'index.html','index.html');z.write(OUT/'selection.json','selection.json')
print(json.dumps({'selected':len(selected),'counts':dict(count),'html_bytes':(OUT/'index.html').stat().st_size,'zip_bytes':(OUT/'human100_review_v1.zip').stat().st_size,'output':str(OUT)}))
