"""One command for source registration scan, incremental scoring and exports."""
import argparse,csv,fcntl,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import human_metric_batch_v1 as batch
H=batch.H

def expand_active_methods(summary,active,ids,certs):
 """Append a full current cohort for newly certified registered methods.

 Existing rows, including their result keys and historical missing reasons, are
 left intact. A method with one ready asset still has a denominator of 100.
 """
 discovered=list(dict.fromkeys(method for _,method in certs))
 unknown=set(discovered)-set(batch.REGISTRY)
 if unknown:raise ValueError('Unregistered active methods: '+','.join(sorted(unknown)))
 added=[method for method in discovered if method not in summary['coverage']]
 if not added:return []
 batch.verified(active['reference_qualification'],active['reference_qualification_sha256'])
 qualification=batch.read(active['reference_qualification'])
 if qualification['selection_sha256']!=active['selection_sha256']:raise ValueError('Qualification is not for the active cohort')
 refs={r['id']:r for r in qualification['objects']}
 if len(qualification['objects'])!=len(ids) or set(refs)!=set(ids):raise ValueError('Qualification must contain the complete active cohort')
 for method in added:
  summary['coverage'][method]=0
  for sid in ids:
   ref=refs[sid]
   summary['rows'].append(dict(id=sid,method=method,reference=ref,metric_eligible=False,
    status='registration_pending' if ref['reference_eligible'] else 'reference_na',
    reason='method_source_registration_pending' if ref['reference_eligible'] else '; '.join(ref.get('na_reasons',[]))))
 assert len(summary['rows'])==len(ids)*len(summary['coverage'])
 assert {(r['id'],r['method']) for r in summary['rows']}=={(sid,m) for sid in ids for m in summary['coverage']}
 return added

def active_pairing_groups(methods):
 """Keep the original comparison when a partially covered method is added."""
 groups=[batch.DEFAULT_METHODS,['omnire','splatad'],['ours','omnire','splatad'],list(methods)]
 groups.extend([[method,'trellis2'] for method in ['ours','trellis_original']])
 result=[]
 for group in groups:
  if all(method in methods for method in group) and group not in result:result.append(list(group))
 return result

def scoring_methods(requested,available):
 methods=list(available) if requested is None else list(requested)
 if len(methods)!=len(set(methods)):raise ValueError('Duplicate requested methods')
 unknown=set(methods)-set(available)
 if unknown:raise ValueError('Methods have no active cohort: '+','.join(sorted(unknown)))
 return methods

def apply_registration_missingness(summary,records,certs):
 """A failed method registration is distinct from pending work or bad reference."""
 unavailable={(r['id'],r['method']):r for r in records if r.get('status')=='unavailable'}
 for row in summary['rows']:
  key=(row['id'],row['method'])
  if key not in unavailable or key in certs or row['status']=='scored':continue
  item=unavailable[key]
  row.update(status='source_registration_unavailable',metric_eligible=False,
             reason=item.get('reason') or 'native_source_registration_unavailable')

def apply_native_generation_failures(summary,queue,certs):
 """Report terminal TRELLIS.2 generation failures instead of pending alignment.

 Only authoritative matching worker completions qualify. Existing scores and
 source certificates take precedence; no reference eligibility is changed.
 """
 failed={}
 for job in queue.get('jobs',[]):
  if job.get('method')!='trellis2' or job.get('status')!='failed':continue
  path=job.get('worker_record')
  if not path or not Path(path).exists():continue
  record=batch.read(path)
  if record.get('status')!='failed' or record.get('key')!=job.get('key'):continue
  if record.get('finished_at') is None or record.get('finished_at')!=job.get('finished_at'):continue
  failed[(job['sample_id'],job['method'])]=dict(path=path,sha256=batch.m.sha(path),reason=record.get('reason','native generation failed'))
 for row in summary['rows']:
  key=(row['id'],row['method'])
  if key not in failed or key in certs or row['status']not in ['registration_pending','native_generation_failed']:continue
  evidence=failed[key]
  row.update(status='native_generation_failed',metric_eligible=False,reason=evidence['reason'],native_failure_evidence=evidence)

def check_active_registration(ref,method,entry):
 """Accept declared source-keypoint SE(3), while rejecting metric-guided fits."""
 batch.verified(entry['registration_record'],entry['registration_sha256'])
 r=batch.read(entry['registration_record'])
 if r.get('posthoc_pose_or_scale_fit') is False:
  return batch.check_registration(ref,method,entry)
 expected=dict(status='passed',registration_certified=True,identity=ref['id'],identity_key=ref['identity_key'],
   timestamp_us=ref['timestamp_us'],camera_sha256=ref['camera_contract_sha256'],method=method,
   registration_kind='source_keypoint_rigid_pnp_v1',posthoc_pose_or_scale_fit=True,
   reference_metric_values_used=False,source_keypoints_used_for_registration=True,
   predicted_joint_pose_unchanged=True,prediction_bbox_used=False,real_heldout_eligible=False,
   prediction_encoding='white_composited_uint8')
 for key,value in expected.items():
  if r.get(key)!=value:raise ValueError('Declared source registration mismatch: '+key)
 protocol=H/'configs/human_source_common_v1/source_keypoint_rigid_pnp_v1.json'
 if Path(r['registration_protocol']).resolve()!=protocol:raise ValueError('Unrecognized registration protocol')
 batch.verified(protocol,r['registration_protocol_sha256'])
 for key in ['source_rgb','source_alpha','render_validation','render_script','trace_record']:
  batch.verified(r[key],r[key+'_sha256'])
 trace=batch.read(r['trace_record']);T=np.array(trace['T_source_camera_from_export'])
 if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid rigid source transform')
 np.testing.assert_allclose(T[3],[0,0,0,1],atol=1e-8)
 np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
 if np.linalg.det(T[:3,:3])<=0:raise ValueError('Reflected source transform')
 for key in ['native_payload','source_keypoints','camera_contract']:batch.verified(trace[key],trace[key+'_sha256'])
 if trace['camera_contract_sha256']!=ref['camera_contract_sha256']:raise ValueError('Trace uses another camera')
 if trace.get('reference_metric_values_used') is not False or trace.get('asset_modified') is not False:raise ValueError('Unaccepted fit scope')
 return r

def score_active_pair(row,metric):
 """The extra registration route uses the same frozen image operations/formulas."""
 r=check_active_registration(row['reference'],row['method'],row)
 if r.get('posthoc_pose_or_scale_fit') is False:return batch.score_pair(row,metric)
 ref=row['reference']
 for key in ['camera_contract','reference_rgb','reference_mask']:batch.verified(ref[key],ref[key+'_sha256'])
 c=batch.read(ref['camera_contract'])
 raw=np.asarray(Image.open(ref['reference_rgb']).convert('RGB'));mask=np.asarray(Image.open(ref['reference_mask']).convert('L'))
 rgb=np.asarray(Image.open(r['source_rgb']).convert('RGB'));alpha=np.asarray(Image.open(r['source_alpha']).convert('L'))
 assert raw.shape==rgb.shape and mask.shape==alpha.shape==rgb.shape[:2]
 assert list(rgb.shape[1::-1])==c['render_wh']
 reference=batch.m.prepare_reference(raw,mask);assert reference['status']=='ready' and list(reference['roi'])==ref['roi_xyxy']==c['roi_xyxy']
 prediction=batch.m.prepare_prediction(rgb,alpha,reference['roi'],r['prediction_encoding'])
 scores=dict(PSNR=batch.m.psnr(reference['rgb'],prediction['rgb']),SSIM=batch.m.ssim(reference['rgb'],prediction['rgb']),
  LPIPS=metric(reference['rgb'],prediction['rgb']),IoU=batch.m.iou(reference['mask'],prediction['mask']),IoU_raw=batch.m.iou(mask,alpha),
  **{'ED-P':batch.m.na('feature_extractor_and_visibility_protocol_pending')})
 return dict(status='scored',id=row['id'],method=row['method'],input_key=row['input_key'],metric_eligible=True,
  metrics=scores,reference_pixel_height=ref['reference_pixel_height'],roi_xyxy=ref['roi_xyxy'],
  registration_record=row['registration_record'],registration_sha256=row['registration_sha256'],
  reference_rgb_sha256=ref['reference_rgb_sha256'],reference_mask_sha256=ref['reference_mask_sha256'],
  prediction_rgb_sha256=r['source_rgb_sha256'],prediction_alpha_sha256=r['source_alpha_sha256'],
  registration_kind=r['registration_kind'],registration_protocol_sha256=r['registration_protocol_sha256'],
  posthoc_pose_or_scale_fit=True,source_keypoints_used_for_registration=True,real_heldout_eligible=False,
  scope='Source reconstruction after declared fixed-body-keypoint rigid registration; no world metric-scale claim'),reference,prediction

def active_pipeline(a):
 """Incrementally score any certified method against the current cohort.

 The frozen metric implementation and its cache keys are unchanged. This entry
 replaces the per-method/per-replacement scoring launchers for subsequent work.
 """
 import time
 from collections import Counter
 run=H/'reports/source_camera_metric_cohort_v3'
 active=batch.read(H/'inputs/active_human_benchmark.json')
 batch.verified(active['selection'],active['selection_sha256'])
 summary=batch.read(run/'summary.json')
 assert summary['selection_sha256']==active['selection_sha256']
 if a.status:
  print(json.dumps(dict(selection=active['selection'],coverage=summary['coverage'],
       states={m:dict(Counter(r['status'] for r in summary['rows'] if r['method']==m)) for m in summary['coverage']},
       results=str(run/'metrics.csv')),ensure_ascii=False,indent=2));return
 with (run/'scorer.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Read after acquiring the same lock as the ongoing legacy scorer.
  summary=batch.read(run/'summary.json')
  batch.verified(summary['protocol'],summary['protocol_sha256'])
  protocol=batch.read(summary['protocol']);batch.verified(batch.m.__file__,protocol['code_sha256'])
  selected=batch.read(active['selection'])['objects'];ids=[r['id'] for r in selected]
  assert len(ids)==len(set(ids))==100
  assert {(r['id'],r['method']) for r in summary['rows']}=={(sid,m) for sid in ids for m in summary['coverage']}
  history=run/'history';history.mkdir(exist_ok=True)
  backup=history/(batch.m.sha(run/'summary.json')+'.json')
  if not backup.exists():backup.write_bytes((run/'summary.json').read_bytes())
  for row in summary['rows']:
   if row['status']=='scored':
    batch.verified(row['result'],row['result_sha256']);assert batch.read(row['result'])['metrics']==row['metrics']
  configured=active.get('registration_indexes',[])
  if not a.registration_index:
   for entry in configured:batch.verified(entry['path'],entry['sha256'])
  indexes=a.registration_index or [Path(entry['path']) for entry in configured] or [H/'reports/ours_source_certification_v3/manifest.json',H/'reports/ours_source_batch_v3/manifest.json']
  certs={};index_records=[];registration_missingness=[]
  for ip in indexes:
   if not ip.exists():continue
   index=batch.read(ip)
   if index.get('selection_sha256')!=active['selection_sha256']:raise ValueError('Registration index is not for the active cohort: '+str(ip))
   index_records.append(dict(path=str(ip),sha256=batch.m.sha(ip)))
   registration_missingness.extend(index.get('unavailable',[]))
   for entry in index['rows']:
    if 'registration_record' not in entry:continue
    batch.verified(entry['registration_record'],entry['registration_sha256'])
    cert=batch.read(entry['registration_record']);key=(entry['id'],cert['method'])
    if cert.get('identity')!=key[0] or cert.get('status')!='passed' or cert.get('registration_certified') is not True:raise ValueError('Index entry is not a certified identity')
    if entry.get('method',key[1])!=key[1]:raise ValueError('Index method differs from certificate')
    if key[0] not in ids:raise ValueError('Nonselected identity in current registration index')
    if key in certs and certs[key]['registration_sha256']!=entry['registration_sha256']:raise ValueError('Conflicting registrations: '+str(key))
    certs[key]=entry
  added_methods=expand_active_methods(summary,active,ids,certs)
  apply_registration_missingness(summary,registration_missingness,certs)
  native_state=H/'runs/human_native_queue_v1/state.json'
  if native_state.exists():apply_native_generation_failures(summary,batch.read(native_state),certs)
  requested_methods=scoring_methods(a.methods,summary['coverage'])
  metric=None;computed=0;reused=0;failures=[]
  for row in summary['rows']:
   if row['method'] not in requested_methods:continue
   ref=row['reference'];entry=certs.get((row['id'],row['method']))
   if ref is None or not ref['reference_eligible'] or entry is None:continue
   prepared=dict(id=row['id'],method=row['method'],reference=ref,registration_record=entry['registration_record'],registration_sha256=entry['registration_sha256'])
   prepared['input_key']=batch.digest(dict(reference=ref,registration_sha256=entry['registration_sha256'],protocol_sha256=summary['protocol_sha256'],metric_code_sha256=batch.m.sha(batch.m.__file__),runner_sha256=batch.m.sha(batch.__file__)))
   try:
    check_active_registration(ref,row['method'],prepared)
    if row['status']=='scored' and row['input_key']!=prepared['input_key']:raise ValueError('Existing score has a different registration; explicit versioned comparison required')
    path=batch.result_path(run,prepared)
    if path.exists():
     result=batch.read(path);assert result['status']=='scored' and result['input_key']==prepared['input_key'];reused+=1
    elif a.prepare_only:
     row.update(status='ready',reason=None,registration_record=entry['registration_record'],registration_sha256=entry['registration_sha256'],input_key=prepared['input_key']);continue
    else:
     if shutil.disk_usage(H).free<10*1024**3:raise RuntimeError('Data disk below 10 GiB; retain pair for later scoring')
     if metric is None:
      import torch
      torch.set_num_threads(4);cfg=protocol['lpips']
      metric=batch.m.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'],'cpu')
     result,reference,pred=score_active_pair(prepared,metric)
     media=run/'media'/row['id'];media.mkdir(parents=True,exist_ok=True)
     for name,array in [('reference',reference['rgb']),(row['method'],pred['rgb']),(row['method']+'_alpha',pred['mask'])]:Image.fromarray(array).save(media/(name+'.png'))
     result.update(lpips=metric.provenance,active_selection_sha256=active['selection_sha256'],execution_script_sha256=batch.m.sha(__file__))
     batch.save(path,result);computed+=1
    row.update(status='scored',reason=None,metric_eligible=result['metric_eligible'],result=str(path),result_sha256=batch.m.sha(path),input_key=prepared['input_key'],metrics=result['metrics'])
   except Exception as error:
    failure=dict(id=row['id'],method=row['method'],reason=str(error));failures.append(failure)
    if row['status']!='scored':row.update(status='validation_or_scoring_failed',reason=str(error))
  methods=list(summary['coverage']);by_method={m:{} for m in methods}
  for row in summary['rows']:
   if row['status']=='scored':by_method[row['method']][row['id']]=batch.read(row['result'])
  names=['PSNR','SSIM','LPIPS','IoU','IoU_raw'];pairings={}
  for chosen in active_pairing_groups(methods):
   pairings['+'.join(chosen)]={n:batch.m.paired_summary({m:by_method[m] for m in chosen},n,ids) for n in names}
  summary.update(coverage={m:len(rows) for m,rows in by_method.items()},pairings=pairings,updated_at=time.time(),
                 last_incremental_run=dict(entrypoint=str(Path(__file__).resolve()),script_sha256=batch.m.sha(__file__),
                  new_scores=computed,reused_scores=reused,failures=failures,registration_indexes=index_records,
                  added_methods=added_methods,requested_methods=requested_methods))
  batch.save(run/'summary.json',summary)
  fields=['id','method','status','reason','result','result_sha256']+[k for n in names+['ED-P'] for k in [n,n+'_status']]
  with (run/'metrics.csv.tmp').open('w',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
   for row in summary['rows']:
    item={k:row.get(k) for k in fields[:6]}
    for name in names+['ED-P']:
     v=row.get('metrics',{}).get(name,dict(status='na',value=None));item[name+'_status']=v['status'];item[name]='inf' if v['status']=='positive_infinity' else v['value'] if v['status']=='ok' else 'NA'
    writer.writerow(item)
  (run/'metrics.csv.tmp').replace(run/'metrics.csv')
  from export_active_metric_tables import main as export_tables
  export_tables(quiet=True)
  print(json.dumps(dict(coverage=summary['coverage'],new_scores=computed,reused=reused,failures=failures),ensure_ascii=False))

def finalize(run):
 plan=batch.load_plan(run);summary=batch.read(run/'summary.json')
 assert summary['qualification_sha256']==plan['qualification_sha256'] and summary['protocol_sha256']==plan['protocol_sha256']
 refs={row['id']:row['reference'] for row in plan['rows']}
 for ref in refs.values():
  c=batch.read(ref['camera_contract']);src=Path(c['reference_rgb']);batch.verified(src,c['reference_rgb_sha256'])
  dst=run/'media'/ref['id']/'reference.png';dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():np.testing.assert_array_equal(np.asarray(Image.open(src)),np.asarray(Image.open(dst)))
  else:shutil.copyfile(src,dst)
 names=['PSNR','SSIM','LPIPS','IoU','IoU_raw','ED-P']
 fields=['id','method','timestamp_us','reference_pixel_height','reference_eligible','status','na_reason']
 fields += [field for n in names for field in [n,n+'_status']]
 fields += ['camera_sha256','result_json','input_key']
 tmp=run/'metrics.csv.tmp'
 with tmp.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for row in summary['rows']:
   ref=refs[row['id']];out=dict(id=row['id'],method=row['method'],timestamp_us=ref['timestamp_us'],reference_pixel_height=ref['reference_pixel_height'],reference_eligible=ref['reference_eligible'],status=row['status'],na_reason=row.get('reason',''),camera_sha256=ref['camera_contract_sha256'],result_json=row.get('result',''),input_key=row.get('input_key',''))
   for name in names:
    value=row.get('metrics',{}).get(name,{'status':'na','value':None});out[name+'_status']=value['status'];out[name]='inf' if value['status']=='positive_infinity' else value['value'] if value['status']=='ok' else 'NA'
   writer.writerow(out)
 tmp.replace(run/'metrics.csv')
 batch.summarize(run)
 print('CSV and all 100 frozen reference displays prepared:',run,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=H/'reports/source_camera_metric_batch_v1');p.add_argument('--methods',nargs='+',default=None)
 p.add_argument('--registration-index',action='append',type=Path,default=[]);p.add_argument('--prepare-only',action='store_true');p.add_argument('--finalize-only',action='store_true')
 p.add_argument('--active-cohort',action='store_true',help='Use the current cohort, cached scores and incremental method certificates')
 p.add_argument('--status',action='store_true',help='Read current active-cohort coverage without launching work');a=p.parse_args()
 if a.active_cohort or a.status:active_pipeline(a);return
 if a.methods is None:a.methods=list(batch.DEFAULT_METHODS)
 run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
 with (run/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if a.finalize_only:finalize(run);return
  subprocess.run([sys.executable,'-B',str(H/'scripts/prepare_scene_registration_v2.py')],check=True)
  indexes=[H/'reports/scene_source_registration_incremental_v2/latest_manifest.json']+a.registration_index
  batch.prepare(run,a.methods,indexes)
  if a.prepare_only:batch.summarize(run)
  else:batch.run_scores(run)
  finalize(run)
if __name__=='__main__':main()
