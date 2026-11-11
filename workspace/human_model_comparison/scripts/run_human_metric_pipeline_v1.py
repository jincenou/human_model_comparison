"""One command for source registration scan, incremental scoring and exports."""
import argparse,csv,fcntl,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import human_metric_batch_v1 as batch
H=batch.H

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
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=H/'reports/source_camera_metric_batch_v1');p.add_argument('--methods',nargs='+',default=batch.DEFAULT_METHODS)
 p.add_argument('--registration-index',action='append',type=Path,default=[]);p.add_argument('--prepare-only',action='store_true');p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
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
