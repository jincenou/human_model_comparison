"""Resumable source-view preparation and core scoring for present/future assets.

The producer/registration adapter owns geometric validation. This runner accepts
only hashed passed certification records and never estimates a pose or resizes
individual predictions. Immutable per-pair keys allow incremental future batches.
"""
import argparse,fcntl,hashlib,html,json,os,shutil,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import human_source_metrics_v1 as m
H=Path(__file__).resolve().parents[1]
C=H/'configs/human_source_common_v1'
DEFAULT_METHODS=['ours','ah','trellis_original','omnire','splatad']
REGISTRY={
 'ours':['exports/ours_frozen_time_v1/{id}/reload_review_v3/validation.json','exports/ours_frozen_time_v1/{id}/reload_review_v2/validation.json'],
 'ah':['runs/ah/human_native_v1_{id}/reload_validation/validation.json','runs/ah/human_pilot_v1_{id}/reload_validation/validation.json'],
 'trellis_original':['runs/trellis_original_human_native_v1/{id}/reload_validation.json'],
 'trellis2':['runs/trellis2_human_native_v1/{id}/reload_validation.json'],
 'omnire':['exports/common_camera_human_original75_time_v1/omnire/{id}/validation.json','exports/common_camera_human_time_v2/omnire/{id}/validation.json'],
 'splatad':['exports/common_camera_human_original75_time_v1/splatad/{id}/validation.json','exports/common_camera_human_v1/splatad/{id}/validation.json'],
 'exavatar':[], 'jacobianavatar':[], 'flexiavatar':[], 'hunyuan3d21':[], 'reconviagen':[]}

def read(p):return json.loads(Path(p).read_text())
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_name(p.name+'.tmp.'+str(os.getpid()));temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def verified(path,expected):
 if m.sha(path)!=expected:raise ValueError('Changed input: '+str(path))
def check_registration(ref,method,entry):
 path=Path(entry['registration_record']);verified(path,entry['registration_sha256']);r=read(path)
 for key,value in [('status','passed'),('registration_certified',True),('identity',ref['id']),('identity_key',ref['identity_key']),('timestamp_us',ref['timestamp_us']),('camera_sha256',ref['camera_contract_sha256']),('method',method)]:
  if r.get(key)!=value:raise ValueError('Registration mismatch: '+key)
 if r.get('posthoc_pose_or_scale_fit') is not False or r.get('reference_metric_values_used') is not False:
  raise ValueError('Registration must explicitly exclude score-driven pose/scale selection')
 if r['prediction_encoding']!='white_composited_uint8':raise ValueError('Unaccepted alpha encoding')
 for key in ['source_rgb','source_alpha']:verified(r[key],r[key+'_sha256'])
 verified(r['render_validation'],r['render_validation_sha256']);verified(r['render_script'],r['render_script_sha256'])
 return r

def prepare(run,methods,index_paths):
 unknown=set(methods)-set(REGISTRY)
 if unknown:raise ValueError('Add the method adapter/registry before preparing: '+','.join(sorted(unknown)))
 if len(methods)!=len(set(methods)):raise ValueError('Duplicate methods')
 qpath=C/'reference_qualification_frozen_v1.json';q=read(qpath);ppath=C/'core_metric_protocol_frozen_v1.json';p=read(ppath)
 verified(ppath,q['core_protocol_sha256']);verified(m.__file__,p['code_sha256'])
 verified(q['selection'],q['selection_sha256']);assert q['denominator']==100
 assert len(q['objects'])==100 and len({r['id'] for r in q['objects']})==100
 assert {r['id'] for r in read(q['selection'])['objects']}=={r['id'] for r in q['objects']}
 certs={};indexes=[]
 for ip in index_paths:
  index=read(ip);verified(index['qualification'],index['qualification_sha256']);assert index['qualification_sha256']==m.sha(qpath)
  indexes.append(dict(path=str(ip.resolve()),sha256=m.sha(ip)))
  for row in index['rows']:
   if row.get('status')!='certified_for_core_metrics':continue
   key=(row['id'],row['method'])
   if key in certs and row['registration_sha256']!=certs[key]['registration_sha256']:raise ValueError('Conflicting registrations: '+str(key))
   certs[key]=row
 rows=[]
 for ref in q['objects']:
  verified(ref['camera_contract'],ref['camera_contract_sha256'])
  verified(ref['reference_rgb'],ref['reference_rgb_sha256']);verified(ref['reference_mask'],ref['reference_mask_sha256'])
  for method in methods:
   native=next((H/p.format(id=ref['id']) for p in REGISTRY[method] if (H/p.format(id=ref['id'])).exists()),None)
   row=dict(id=ref['id'],method=method,reference=ref,native_record_present=native is not None,native_record=str(native) if native else None)
   if not ref['reference_eligible']:row.update(status='reference_na',reason='; '.join(ref['na_reasons']))
   elif (ref['id'],method) not in certs:
    row.update(status='awaiting_registration' if native else 'awaiting_asset',reason='No accepted source-camera certificate; native previews cannot be scored')
   else:
    entry=certs[(ref['id'],method)]
    try:
     reg=check_registration(ref,method,entry)
     row.update(status='ready',registration_record=entry['registration_record'],registration_sha256=entry['registration_sha256'])
     row['input_key']=digest(dict(reference=ref,registration_sha256=entry['registration_sha256'],protocol_sha256=m.sha(ppath),metric_code_sha256=m.sha(m.__file__),runner_sha256=m.sha(__file__)))
    except (KeyError,ValueError,FileNotFoundError) as e:row.update(status='validation_failed',reason=str(e))
   rows.append(row)
 plan=dict(schema='incremental_human_source_metric_plan_v1',selected_denominator=100,methods=methods,
  qualification=str(qpath),qualification_sha256=m.sha(qpath),protocol=str(ppath),protocol_sha256=m.sha(ppath),
  metric_code_sha256=m.sha(m.__file__),runner_sha256=m.sha(__file__),registration_indexes=indexes,rows=rows,
  summary={method:dict(Counter(r['status'] for r in rows if r['method']==method)) for method in methods})
 key=digest(plan);dest=run/'plans'/(key+'.json')
 if not dest.exists():save(dest,plan)
 else:assert read(dest)==plan
 save(run/'latest_plan.json',dict(plan=str(dest.resolve()),sha256=m.sha(dest),key=key,prepared_at=time.time()))
 print(json.dumps(dict(plan=str(dest),summary=plan['summary'])),flush=True)
 return plan

def load_plan(run):
 pointer=read(run/'latest_plan.json');verified(pointer['plan'],pointer['sha256']);p=read(pointer['plan'])
 for key in ['qualification','protocol']:verified(p[key],p[key+'_sha256'])
 verified(m.__file__,p['metric_code_sha256']);verified(__file__,p['runner_sha256'])
 return p

def result_path(run,row):return run/'results'/row['method']/row['id']/(row['input_key']+'.json')
def score_pair(row,lpips):
 ref=row['reference'];verified(ref['camera_contract'],ref['camera_contract_sha256'])
 verified(ref['reference_rgb'],ref['reference_rgb_sha256']);verified(ref['reference_mask'],ref['reference_mask_sha256'])
 r=check_registration(ref,row['method'],row);c=read(ref['camera_contract'])
 raw=np.asarray(Image.open(ref['reference_rgb']).convert('RGB'));mask=np.asarray(Image.open(ref['reference_mask']).convert('L'))
 rgb=np.asarray(Image.open(r['source_rgb']).convert('RGB'));alpha=np.asarray(Image.open(r['source_alpha']).convert('L'))
 assert raw.shape==rgb.shape and mask.shape==alpha.shape==rgb.shape[:2]
 assert list(rgb.shape[1::-1])==c['render_wh']
 a=m.prepare_reference(raw,mask);assert a['status']=='ready' and list(a['roi'])==ref['roi_xyxy']==c['roi_xyxy']
 b=m.prepare_prediction(rgb,alpha,a['roi'],r['prediction_encoding'])
 scores=dict(PSNR=m.psnr(a['rgb'],b['rgb']),SSIM=m.ssim(a['rgb'],b['rgb']),LPIPS=lpips(a['rgb'],b['rgb']),IoU=m.iou(a['mask'],b['mask']),IoU_raw=m.iou(mask,alpha))
 scores['ED-P']=m.na('feature_extractor_and_visibility_protocol_pending')
 return dict(status='scored',id=row['id'],method=row['method'],input_key=row['input_key'],metric_eligible=True,
  metrics=scores,reference_pixel_height=ref['reference_pixel_height'],roi_xyxy=ref['roi_xyxy'],
  registration_record=row['registration_record'],registration_sha256=row['registration_sha256'],
  reference_rgb_sha256=ref['reference_rgb_sha256'],reference_mask_sha256=ref['reference_mask_sha256'],
  prediction_rgb_sha256=r['source_rgb_sha256'],prediction_alpha_sha256=r['source_alpha_sha256'],
  real_heldout_eligible=False,scope='source reconstruction core metrics; five-method final comparison and ED-P remain incomplete'),a,b

def run_scores(run):
 import torch
 torch.set_num_threads(4)
 with (run/'scorer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=load_plan(run);cfg=read(plan['protocol'])['lpips'];metric=None
  state=dict(status='running',pid=os.getpid(),started_at=time.time(),scored_new=0,reused=0,failed=0,device='cpu',threads=4)
  save(run/'state.json',state)
  try:
   for row in plan['rows']:
    if row['status']!='ready':continue
    path=result_path(run,row)
    if path.exists():
     saved=read(path)
     if saved.get('status')=='scored' and saved.get('input_key')==row['input_key']:
      # Reuse only while the same source/registration bytes remain current.
      ref=row['reference'];verified(ref['camera_contract'],ref['camera_contract_sha256']);verified(ref['reference_rgb'],ref['reference_rgb_sha256']);verified(ref['reference_mask'],ref['reference_mask_sha256']);check_registration(ref,row['method'],row)
      state['reused']+=1;continue
     raise ValueError('Existing result is not a verified success: '+str(path))
    if shutil.disk_usage(run).free<10*1024**3:raise RuntimeError('Data-disk 10 GiB reserve reached')
    if metric is None:metric=m.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'],'cpu')
    state.update(current=dict(id=row['id'],method=row['method']),updated_at=time.time());save(run/'state.json',state)
    try:
     result,a,b=score_pair(row,metric)
     media=run/'media'/row['id'];media.mkdir(parents=True,exist_ok=True)
     rp=media/'reference.png'
     if not rp.exists():Image.fromarray(a['rgb']).save(rp)
     pp=media/(row['method']+'_'+row['input_key'][:12]+'.png');Image.fromarray(b['rgb']).save(pp)
     ap=media/(row['method']+'_'+row['input_key'][:12]+'_alpha.png');Image.fromarray(b['mask']).save(ap)
     result.update(reference_display=str(rp),prediction_display=str(pp),alpha_display=str(ap),lpips=metric.provenance)
     save(path,result);state['scored_new']+=1
     print(row['id'],row['method'],'scored',flush=True)
    except Exception as e:
     state['failed']+=1
     save(run/'failures'/row['method']/(row['id']+'_'+row['input_key']+'.json'),dict(status='failed',reason=str(e),traceback=traceback.format_exc(),input_key=row['input_key']))
   state['status']='finished_with_failures' if state['failed'] else 'finished'
  except BaseException as e:state.update(status='stopped_with_error',error=str(e));raise
  finally:state.update(finished_at=time.time());save(run/'state.json',state)
 summarize(run)

def summarize(run):
 plan=load_plan(run);methods=plan['methods'];ids=list(dict.fromkeys(row['id'] for row in plan['rows']));records=[];by_method={x:{} for x in methods}
 for row in plan['rows']:
  result=read(result_path(run,row)) if row['status']=='ready' and result_path(run,row).exists() else None
  if result and result.get('status')=='scored':
   assert result['input_key']==row['input_key'];by_method[row['method']][row['id']]=result
   records.append(dict(id=row['id'],method=row['method'],status='scored',result=str(result_path(run,row)),input_key=row['input_key'],metrics=result['metrics']))
  else:
   failure=run/'failures'/row['method']/(row['id']+'_'+row.get('input_key','')+'.json')
   error=read(failure) if failure.exists() else None
   records.append(dict(id=row['id'],method=row['method'],status='failed' if error else row['status'],reason=error['reason'] if error else row.get('reason','prepared_not_scored')))
 names=['PSNR','SSIM','LPIPS','IoU','IoU_raw'];pairings={}
 for chosen in [methods]+[['omnire','splatad']] if all(x in methods for x in ['omnire','splatad']) else [methods]:
  key='+'.join(chosen)
  pairings[key]={name:m.paired_summary({x:by_method[x] for x in chosen},name,ids) for name in names}
 summary=dict(status='core_metric_batch_summary',denominator=100,scope='source reconstruction; not held-out or final five-method ranking',
  coverage={x:len(by_method[x]) for x in methods},pairings=pairings,rows=records,plan_summary=plan['summary'],
  qualification_sha256=plan['qualification_sha256'],protocol_sha256=plan['protocol_sha256'])
 save(run/'summary.json',summary)
 # Direct comparison only at the common reference ROI; pending methods stay empty.
 parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>人体统一源视角评测准备与结果</title><style>body{font:15px sans-serif;margin:24px;background:#f3f5f8}section{background:white;margin:18px 0;padding:18px}.grid{display:grid;grid-template-columns:repeat('+str(len(methods)+1)+',minmax(140px,1fr));gap:8px}.cell img{width:100%;background:white}.note{font-size:12px;color:#555}input{padding:8px}small{display:block}</style>',
 '<h1>人体统一源视角：评测准备与已通过项</h1><p>同一人物、冻结曝光、原始相机及公共参考ROI。保留原始预测缺陷；未配准的方法不放入评分对比。</p>',
 '<p>冻结100人；参考合格91人，9人因外部遮挡记NA。当前为源视角重建指标，非真实留出评测；ED-P和完整五方法对比尚未完成。</p>',
 '<p>覆盖率：'+html.escape(json.dumps(summary['coverage'],ensure_ascii=False))+' / 每方法100人</p>',
 '<p><a href="summary.json">逐对象成绩、同对象配对与缺测原因</a></p><input id="q" placeholder="搜索HN编号" oninput="for(const s of document.querySelectorAll(\'section\'))s.hidden=!s.id.toLowerCase().includes(this.value.toLowerCase())">']
 status_labels={'reference_na':'参考不合格（NA）','awaiting_registration':'资产已见，待源相机配准','awaiting_asset':'等待资产/验证记录','validation_failed':'输入验证失败','ready':'待计算'}
 for sid in ids:
  subset=[r for r in plan['rows'] if r['id']==sid];ref=subset[0]['reference'];display=run/'media'/sid/'reference.png'
  parts.append('<section id="'+sid+'"><h2>'+sid+'</h2><p class="note">原图人物高度 '+str(ref['reference_pixel_height'])+' px · 曝光 '+str(ref['timestamp_us'])+' μs</p><div class="grid"><div class="cell">真实参考')
  if display.exists():parts.append('<img loading="lazy" src="'+str(display.relative_to(run))+'">')
  else:parts.append('<p>'+html.escape('; '.join(ref['na_reasons']) if not ref['reference_eligible'] else '参考已冻结')+'</p>')
  parts.append('</div>')
  for row in subset:
   result=by_method[row['method']].get(sid);parts.append('<div class="cell"><b>'+html.escape(row['method'])+'</b>')
   if result:
    parts.append('<img loading="lazy" src="'+str(Path(result['prediction_display']).relative_to(run))+'">')
    for key in ['PSNR','SSIM','LPIPS','IoU']:
     val=result['metrics'][key];text='∞' if val['status']=='positive_infinity' else f"{val['value']:.4f}" if val['status']=='ok' else 'NA'
     parts.append('<small>'+key+': '+text+'</small>')
   else:parts.append('<p>'+html.escape(status_labels.get(row['status'],row['status']))+'</p>')
   parts.append('</div>')
  parts.append('</div></section>')
 parts.append('</html>');(run/'index.html').write_text(''.join(parts))
 print(json.dumps(dict(report=str(run/'index.html'),coverage=summary['coverage'])),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','report']);ap.add_argument('--run-dir',type=Path,default=H/'reports/source_camera_metric_batch_v1')
 ap.add_argument('--methods',nargs='+',default=DEFAULT_METHODS);ap.add_argument('--registration-index',action='append',type=Path)
 a=ap.parse_args();run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
 if a.action=='prepare':prepare(run,a.methods,a.registration_index or [H/'reports/scene_source_registration_v1/manifest.json'])
 elif a.action=='run':run_scores(run)
 else:summarize(run)
if __name__=='__main__':main()
