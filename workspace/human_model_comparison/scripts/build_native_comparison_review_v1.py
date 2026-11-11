"""Faithful review of completed native renders; no inferred views or score claims."""
from pathlib import Path
import datetime,hashlib,html,json,shutil,sys
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from human_source_metrics_v1 import prepare_reference
H=Path(__file__).resolve().parents[1];O=H/'reports/human_native_comparison_20261111_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if O.exists() and '--refresh' not in sys.argv:
  assert '--finish-panels' in sys.argv, 'Existing report must not be overwritten'
  manifest=read(O/'manifest.json');records=manifest['objects'];columns=manifest['columns'];counts=manifest['display_counts']
  finish_panels(records,columns)
  print(json.dumps(dict(report=str(O/'index.html'),shown=counts,complete=[r['id']for r in records if r['all_six']])))
  return
 if O.exists():
  assert '--refresh' in sys.argv
  old=O/'manifest.json';history=O/'manifest_history';history.mkdir(exist_ok=True)
  archive=history/(sha(old)+'.json')
  if not archive.exists():shutil.copyfile(old,archive)
 O.mkdir(exist_ok=True);(O/'media').mkdir(exist_ok=True)
 selection=H/'inputs/benchmark_human100_v2/selection.json';rows=read(selection)['objects']
 q=read(H/'runs/human_native_queue_v1/state.json');jobs={j['key']:j for j in q['jobs']}
 continuation=H/'runs/ours_reload_continuation_v3/state.json'
 recovered=read(continuation)if continuation.exists()else{'jobs':[]}
 recovered_jobs={j['id']:j for j in recovered['jobs']}
 recovered_ids={j['id']for j in recovered['jobs']if j['status']=='passed'}
 omni_continuation=H/'runs/omnire_original75_time_v1/state.json'
 omni_old=read(omni_continuation)if omni_continuation.exists()else{'jobs':[]}
 omni_old_jobs={j['id']:j for j in omni_old['jobs']}
 splat_state=H/'runs/splatad_original75_time_v1/state.json'
 splat_old=read(splat_state)if splat_state.exists()else{'jobs':[]}
 splat_old_jobs={j['id']:j for j in splat_old['jobs']}
 unavailable_splat={j['id']:j for p in [H/'inputs/scene_extract_v2/splatad_plan.json',H/'inputs/scene_extract_original75_v1/splatad_plan.json'] for j in read(p)['objects'] if j['status']=='unavailable'}
 reference_manifest=H/'reports/reference_qualification_v1/current_candidates_v1.json'
 reference_candidates={j['id']:j for j in read(reference_manifest)['objects']}
 columns=[('reference','真实参考','冻结曝光，候选可见mask白底'),('ours','Ours · 视角00','固定时刻，原生身体坐标'),
  ('ours_back','Ours · 视角08','同一姿态，固定环绕另一侧'),('ah','AH','AHC预测源相机'),
  ('trellis_original','TRELLIS 原版','原生相机0，未配准'),('trellis2','TRELLIS.2','原生PBR相机0，未配准'),
  ('omnire','OmniRe 非SMPL','真实源相机＋曝光时间适配'),('splatad','SplatAD','真实源相机；原75人已通过项含时间核查')]
 records=[];counts={m:0 for m,_,_ in columns}
 for row in rows:
  sid=row['id'];camera=read(H/'configs/human_source_common_v1/candidates'/sid/'camera.json')
  reference=reference_candidates[sid]
  assert sha(reference['raw'])==reference['raw_sha256'] and sha(reference['candidate_mask'])==reference['candidate_mask_sha256']
  candidate_mask=np.asarray(Image.open(reference['candidate_mask']).convert('L'))
  prepared=prepare_reference(np.asarray(Image.open(reference['raw']).convert('RGB')),candidate_mask)
  assert prepared['status']=='ready'
  display_roi=prepared['roi']
  ah=H/'runs/ah'/('human_pilot_v1_HN0004'if sid=='HN0004'else f'human_native_v1_{sid}')
  paths=dict(reference=Path(reference['raw']),ours=H/'exports/ours_frozen_time_v1'/sid/'reload_review_v2/view_00.png',
   ours_back=H/'exports/ours_frozen_time_v1'/sid/'reload_review_v2/view_08.png',ah=ah/'reload_validation/ahc_source.png',
   trellis_original=H/'runs/trellis_original_human_native_v1'/sid/'reload_0.png',trellis2=H/'runs/trellis2_human_native_v1'/sid/'reload_shaded_0.png',
   omnire=H/'exports/common_camera_human_time_v2/omnire'/sid/'native_actor_source.png',
   splatad=H/'exports/common_camera_human_v1/splatad'/sid/'native_actor_source.png')
  ys=np.where(candidate_mask>127)[0]
  record=dict(id=sid,scene=row['scene'],cohort=row['cohort'],timestamp_us=row['t_ref_us'],reference_pixel_height=int(ys.max()-ys.min()+1),views={})
  record.update(reference_candidate_mask=reference['candidate_mask'],reference_candidate_mask_sha256=reference['candidate_mask_sha256'],display_roi_xyxy=display_roi,reference_visual_status=reference['visual_status'],accessory_external_occlusion=reference.get('accessory_external_occlusion',False),historical_generation_inputs_unchanged=True)
  for method,label,note in columns:
   job=jobs.get(f"{'ours'if method=='ours_back'else method}_{sid}",{})
   if method in ('ours','ours_back') and recovered_jobs.get(sid,{}).get('status')=='failed':
    job=recovered_jobs[sid]
   valid=method=='reference' or job.get('status')in('native_passed','prior_pilot_passed')
   source=paths[method]
   if method=='omnire' and sid in omni_old_jobs:
    old_job=omni_old_jobs[sid]
    if old_job['status']=='passed':
     validation_path=Path(old_job['validation']);assert sha(validation_path)==old_job['validation_sha256']
     validation=read(validation_path)
     assert validation['state_time_query_valid'] and validation['common_camera_projection_valid'] and validation['stored_canonical_parameters_equal']
     valid=True;source=Path(validation['actor_preview'])
    else:
     job=old_job
   if method=='splatad':
    if sid in unavailable_splat:
     job=unavailable_splat[sid];valid=False
    elif sid in splat_old_jobs:
     old_job=splat_old_jobs[sid]
     if old_job['status']=='passed':
      validation_path=Path(old_job['validation']);assert sha(validation_path)==old_job['validation_sha256']
      validation=read(validation_path)
      assert validation['state_time_query_valid'] and validation['common_camera_projection_valid'] and validation['stored_canonical_parameters_equal']
      valid=True;source=Path(validation['actor_preview'])
     else:job=old_job
   if method in ('ours','ours_back') and sid in recovered_ids:
    folder=H/'exports/ours_frozen_time_v1'/sid/'reload_review_v3'
    validation=read(folder/'validation.json');assert validation['reload_valid'] and validation['analytic_coordinate_check']['passed']
    valid=True;source=folder/('view_00.png'if method=='ours'else'view_08.png')
   if not(valid and source.exists()):
    record['views'][method]=dict(status=job.get('status','not_reloaded_this_round'),note=job.get('reason'));continue
   dest=O/'media'/f'{sid}_{method}.png'
   if method=='reference':
    Image.fromarray(prepared['rgb']).save(dest)
    transform='Source-only reviewed candidate visible mask and reference-derived ROI; qualification not frozen; historical model inputs unchanged'
   elif method in ('omnire','splatad'):
    im=Image.open(source).convert('RGB');assert list(im.size)==camera['render_wh']
    l,t,r,b=display_roi;roi=Image.new('RGB',(r-l,b-t),'white')
    box=(max(0,l),max(0,t),min(im.width,r),min(im.height,b));roi.paste(im.crop(box),(box[0]-l,box[1]-t))
    roi.resize((512,512),Image.Resampling.BICUBIC).save(dest)
    transform='reference-defined ROI only; no GT masking of prediction'
   else:shutil.copyfile(source,dest);transform='byte-identical copy of saved render'
   record['views'][method]=dict(status='shown',image=str(dest.relative_to(O)),source=str(source),source_sha256=sha(source),image_sha256=sha(dest),display_transform=transform)
   counts[method]+=1
  record['all_six']=all(record['views'][m]['status']=='shown' for m in ['ours','ah','trellis_original','trellis2','omnire','splatad'])
  records.append(record)
 records.sort(key=lambda r:(not r['all_six'],r['id']))
 title='100人 · 原生资产预览（视角尚未统一）'
 css='''body{margin:0;background:#eef1f5;color:#182332;font:15px system-ui}header{padding:24px 3vw;background:#142132;color:white}h1{margin:0 0 12px}p{line-height:1.7;margin:8px 0}nav{position:sticky;top:0;background:#fff;padding:12px 3vw;z-index:2;border-bottom:1px solid #ccc;display:flex;gap:16px;align-items:center;flex-wrap:wrap}input,select,button{padding:9px;border:1px solid #bac4d1;border-radius:5px;font:inherit}main{padding:10px 2vw}article{padding:18px;background:white;margin:16px 0;border-radius:10px}h2{margin:0 0 8px}.grid{display:grid;grid-template-columns:repeat(8,minmax(155px,1fr));gap:10px;overflow-x:auto}.cell{min-width:155px;background:#f5f6f8;border:1px solid #dce2e8;border-radius:6px;overflow:hidden}.cell img{width:100%;display:block;cursor:zoom-in;background:white}.label{font-weight:650;padding:8px 8px 3px}.note{font-size:12px;color:#5c6574;padding:0 8px 8px;min-height:34px}.missing{height:190px;display:flex;align-items:center;justify-content:center;color:#6d7683;padding:12px}.meta{color:#687181;font-size:13px;margin-bottom:14px}.badge{font-size:12px;padding:4px 8px;background:#dff0e8;color:#186046;border-radius:4px}dialog{border:0;border-radius:10px;max-width:90vw}dialog::backdrop{background:#000b}dialog img{max-height:80vh;max-width:85vw}small{color:#687181}a{color:#235fb5}'''
 parts=[f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style>',
 f'<header><h1>{title}</h1><p><a style="color:#b9ddff" href="../source_camera_metric_pilot_v1/index.html">查看统一源相机的小样例与指标诊断（正在接入其余方法）</a></p><p>同一身份、同一冻结时刻的当前资产结果。生成方法展示原生相机；OmniRe / SplatAD 展示真实源相机的参考ROI。</p><p><strong>视角、尺度和光照尚未跨方法统一，不用于排名或正式打分。</strong> Ours两个方向属于同一固定姿态；本页没有生成新的AI示意图，也没有修饰模型缺陷。</p><p>AH / TRELLIS 为生成；OmniRe / SplatAD 为已有场景提取；Ours 为已有资产指定时刻重载。未完成栏保留，不用其他人物代替。</p></header>',
 '<nav><input id="search" placeholder="搜索 HN编号 / 场景"><label><input id="complete" type="checkbox" checked>仅看六种方法均有本轮结果</label><span id="count"></span><small>点击图片放大；取消勾选可看全部100人</small></nav><main>']
 status={'queued':'等待处理','running':'正在处理','failed':'检查未通过，待修复','not_reloaded_this_round':'本轮尚无独立重载结果','unavailable':'现有场景无对应独立人体节点'}
 for row in records:
  sid=row['id'];parts.append(f'<article data-search="{sid.lower()} {row["scene"]}" data-complete="{str(row["all_six"]).lower()}"><h2>{sid} <span class="badge">{row["cohort"]}</span></h2><div class="meta">{row["scene"]} · 冻结曝光 {row["timestamp_us"]} μs · 原图人体高度 {row["reference_pixel_height"]} px</div><div class="grid">')
  for method,label,note in columns:
   view=row['views'][method]
   if method=='reference':
    note='有外部人物或物体遮挡；资格待冻结' if row['reference_visual_status']=='ineligible_external_occlusion' else '携带物有局部遮挡；资格待冻结' if row['accessory_external_occlusion'] else '修订可见mask已审核；资格待冻结'
   parts.append(f'<div class="cell"><div class="label">{label}</div><div class="note">{note}</div>')
   if view['status']=='shown':parts.append(f'<img loading="lazy" src="{view["image"]}" alt="{sid} {label}" onclick="zoom(this)">')
   else:parts.append(f'<div class="missing">{status.get(view["status"],html.escape(view["status"]))}</div>')
   parts.append('</div>')
  parts.append('</div></article>')
 parts.append('''</main><dialog id="viewer" onclick="this.close()"><div id="caption"></div><img id="large"></dialog><script>const cards=[...document.querySelectorAll('article')],s=document.getElementById('search'),c=document.getElementById('complete');function filter(){let n=0;for(const a of cards){let ok=a.dataset.search.includes(s.value.toLowerCase().trim())&&(!c.checked||a.dataset.complete==='true');a.hidden=!ok;if(ok)n++}document.getElementById('count').textContent=n+' / 100 人'}s.oninput=c.onchange=filter;filter();function zoom(im){document.getElementById('large').src=im.src;document.getElementById('caption').textContent=im.alt;document.getElementById('viewer').showModal()}</script></html>''')
 temp=O/'index.html.tmp';temp.write_text(''.join(parts));temp.replace(O/'index.html')
 manifest=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(selection),queue_snapshot_counts=q['counts'],display_counts=counts,score_eligible=False,columns=columns,objects=records)
 manifest['ours_continuation_counts']=recovered.get('counts',{})
 manifest['omnire_original75_counts']=omni_old.get('counts',{})
 manifest['splatad_original75_counts']=splat_old.get('counts',{})
 manifest['reference_visual_manifest']=str(reference_manifest)
 manifest['reference_visual_manifest_sha256']=sha(reference_manifest)
 temp=O/'manifest.json.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');temp.replace(O/'manifest.json')
 finish_panels(records,columns)
 print(json.dumps(dict(report=str(O/'index.html'),shown=counts,complete=[r['id']for r in records if r['all_six']])))

def finish_panels(records,columns):
 # Fixed-order scientific contact panels, using the already saved renders.
 font=ImageFont.load_default(size=21)
 labels=['Real reference','Ours view00','Ours view08','AH / estimated camera','TRELLIS original','TRELLIS.2 native PBR','OmniRe / source ROI','SplatAD / source ROI']
 for row in [x for x in records if x['all_six']][:3]:
  sheet=Image.new('RGB',(1536,912),'#edf0f4');draw=ImageDraw.Draw(sheet)
  draw.text((16,10),row['id']+' | Native-view preview; cameras NOT yet aligned',font=font,fill='#152338')
  for i,((m,_,_),label)in enumerate(zip(columns,labels)):
   x=(i%4)*384;y=48+(i//4)*432
   draw.text((x+10,y+6),label,font=font,fill='#152338')
   im=Image.open(O/row['views'][m]['image']).convert('RGB').resize((384,384),Image.Resampling.LANCZOS)
   sheet.paste(im,(x,y+38))
  sheet.save(O/f'{row["id"]}_comparison.jpg',quality=95)

if __name__=='__main__':main()
