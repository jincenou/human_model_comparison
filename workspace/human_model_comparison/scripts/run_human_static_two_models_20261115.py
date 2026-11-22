"""Frozen100 single-pose human Hunyuan/Recon production; no metric or quality filtering."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse, contextlib, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
P=R/'model_preparation/20261109_2100_CST'
Q=H/'runs/human_static_two_models_20261115_v1'
D=H/'scripts/human_static_two_models_20261115'
ENVS=Path(_hmc_expand('${CONDA_ROOT}/envs'))
METHODS={'hunyuan3d21':('hunyuan21','ab_cmp_hunyuan21','generate_hunyuan21.py'),
         'reconviagen':('recon02','ab_cmp_recon02','generate_recon02.py')}
PILOT='HN0004'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
@contextlib.contextmanager
def transaction():
 with (Q/'queue.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);q=read(Q/'queue.json')
  yield q
  q.update(updated_at=time.time(),counts=dict(Counter(x['status'] for x in q['jobs'])))
  save(Q/'queue.json',q)

def prepare():
 assert not (Q/'manifest.json').exists(),'Immutable manifest already exists; use check or worker'
 selection=H/'inputs/benchmark_human100_v3/selection.json';selected=read(selection)['objects']
 sources=[H/'inputs/human_static_native_v1/manifest.json',H/'inputs/human_static_replacement_v3/manifest.json']
 all_inputs={r['sample_id']:dict(r,origin_manifest=str(p),origin_manifest_sha256=sha(p)) for p in sources for r in read(p)['objects']}
 rows=[]
 for s in selected:
  r=all_inputs[s['id']];assert r['identity_key']==s['identity_key']
  assert r['timestamp_us']==s['t_ref_us'];assert r['real_input_count']==1 and r['generated_input_count']==0
  for k,h in [('image_rgba','image_rgba_sha256'),('raw_source_path','raw_source_sha256'),('exact_mask_path','exact_mask_sha256')]:assert sha(r[k])==r[h],(s['id'],k)
  r['views']=[dict(image=r['image_rgba'],sha256=r['image_rgba_sha256'],split='train',raw_image=r['raw_source_path'],raw_sha256=r['raw_source_sha256'],timestamp_us=r['timestamp_us'])]
  r.update(input_protocol='single real RGBA at frozen source pose; no FLUX, duplicate views or cross-time frames',selection=str(selection),selection_sha256=sha(selection))
  rows.append(r)
 assert len(rows)==len({r['sample_id'] for r in rows})==100
 assert PILOT in {r['sample_id'] for r in rows} and 'HN0646' in {r['sample_id'] for r in rows} and 'HN0035' not in {r['sample_id'] for r in rows}
 pinned=[Path(__file__),*sorted(D.glob('*.py')),P/'scripts/recon02_serialization.py',P/'scripts/dispatch_rigid111_direct.py']
 upstream=P/'sources/recon02-f6720923260927333e22761fa722144d214733d9/trellis/pipelines/trellis_image_to_3d.py'
 manifest=dict(schema='frozen100_human_static_two_methods_v1',selection=str(selection),selection_sha256=sha(selection),population=100,objects=rows,
  input_files_copied=False,scope='Static source-conditioned generation and native reload; no dynamic pose fitting',pilot=PILOT,
  methods=list(METHODS),code_hashes={str(p):sha(p) for p in pinned},
  recon_single_image_support=dict(source=str(upstream),sha256=sha(upstream),lines='975-1060',evidence='vggt_feat stacks a nonempty list; run derives view count n from tokens, with no minimum two-view guard; one-view inference must pass pilot'),
  settings=dict(seed=42,hunyuan_texture_seed=0,reference_pose='unchanged real anchor',disk_guard_gib=10),
  exclusions='No automatic fallback, input augmentation, synthetic views, pose editing or appearance-based result rejection')
 Q.mkdir(parents=True,exist_ok=True);save(Q/'manifest.json',manifest)
 ids=[PILOT]+sorted(r['sample_id'] for r in rows if r['sample_id']!=PILOT)
 save(Q/'queue.json',dict(status='prepared',manifest_sha256=sha(Q/'manifest.json'),jobs=[dict(method=m,sample_id=s,status='pending') for s in ids for m in METHODS]))
 save(H/'state/experiment_scope_20261115_hunyuan_recon_addendum.json',dict(user_instruction='不是，如果可以去生成那就直接尽快挂着生成',supersedes_for_methods=['Hunyuan3D-2.1','ReconViaGen'],historical_scope=str(H/'state/experiment_scope_20261114.json'),active_new_methods=list(METHODS),other_cancelled_methods_remain_cancelled=True,selection=str(selection),manifest=str(Q/'manifest.json'),created_at=time.time()))
 print(json.dumps(dict(manifest=str(Q/'manifest.json'),objects=100,jobs=200)))

def check():
 m=read(Q/'manifest.json');assert sha(Q/'manifest.json')==read(Q/'queue.json')['manifest_sha256']
 assert sha(m['selection'])==m['selection_sha256']
 for p,s in m['code_hashes'].items():assert sha(p)==s,p
 for r in m['objects']:
  assert sha(r['image_rgba'])==r['image_rgba_sha256']
 print('Frozen100 input hashes and adapter hashes verified',flush=True)

def claim(method,gpu):
 with transaction() as q:
  gate=Q/f'{method}_pilot_gate.json'
  if not gate.exists():
   pilot=next(x for x in q['jobs'] if x['method']==method and x['sample_id']==PILOT)
   if pilot['status']=='failed':return 'pilot_failed'
   if pilot['status']!='pending':return None
   job=pilot
  else:
   g=read(gate);assert g['status']=='passed' and g['manifest_sha256']==q['manifest_sha256']
   job=next((x for x in q['jobs'] if x['method']==method and x['status']=='pending'),None)
  if job is None:return None
  job.update(status='running',gpu=gpu,pid=os.getpid(),started_at=time.time());return dict(job)

def finish(job,result):
 with transaction() as q:
  row=next(x for x in q['jobs'] if x['method']==job['method'] and x['sample_id']==job['sample_id'])
  row.update(status=result['status'],finished_at=time.time(),result=result['result'],reason=result.get('reason'))

def validate(method,folder):
 from PIL import Image
 import numpy as np
 state=read(folder/'state.json');assert state['sample_id']==folder.name
 assert state['status'] in ['generated_and_mesh_reload_passed_pending_render','generated_pending_independent_reload']
 record=read(folder/'reload_render_validation.json');assert record['output_sha256']==sha(state['output'])
 assert record['mesh_objects']>0 and record['texture_images']>0
 views=[]
 for p in record['views']:
  a=np.asarray(Image.open(p).convert('RGBA'));assert a.shape==(512,512,4)
  views.append(dict(path=p,sha256=sha(p),foreground_pixels=int((a[:,:,3]>127).sum())))
 if method=='reconviagen':
  gs=read(folder/'reload_validation.json');assert gs['gaussians']>0
  assert gs['checkpoint_sha256']==sha(folder/'native_gaussians.pt')
  assert all(x['max_pixel_difference']<=1 and x['mean_pixel_difference']<=.01 for x in gs['views'])
 return dict(status='passed',sample_id=folder.name,method=method,views=views,geometry_positive=True,
  all_preview_views_empty=not any(x['foreground_pixels'] for x in views),quality_filter_applied=False,
  common_camera_qualified=False,metric_eligible=False,scores_produced=False,
  scope='Native output integrity and reload only; independent source camera registration pending')

def execute(job,gpu,lease_fds):
 sys.path.insert(0,str(P/'scripts'));import dispatch_rigid111_direct as resources
 method=job['method'];native,env_name,script=METHODS[method]
 folder=Q/'objects'/method/job['sample_id'];folder.parent.mkdir(parents=True,exist_ok=True)
 env=resources.environment(native);env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONPATH=str(P/'scripts'))
 py=ENVS/env_name/'bin/python'
 commands=[('generate',[str(py),'-B','-u',str(D/script),'--sample-id',job['sample_id'],'--run-dir',str(folder)],env)]
 if method=='reconviagen':commands.append(('native_reload',[str(py),'-B','-u',str(D/'reload_recon02.py'),'--run-dir',str(folder)],env))
 commands.append(('glb_reload',[str(ENVS/'ab_cmp_hunyuan21/bin/python'),'-B','-u',str(D/'reload_mesh.py'),'--run-dir',str(folder)],dict(env,LD_LIBRARY_PATH=str(ENVS/'ab_cmp_hunyuan21/lib'),CUDA_VISIBLE_DEVICES='')))
 result=dict(status='running',method=method,sample_id=job['sample_id'],real_input_count=1,generated_input_count=0,
  manifest_sha256=sha(Q/'manifest.json'),started_at=time.time(),gpu=gpu,stages=[],result=str(folder/'result.json'))
 try:
  for stage,cmd,e in commands:
   log=Q/'logs'/f'{method}_{job["sample_id"]}_{stage}.log';log.parent.mkdir(exist_ok=True)
   started=time.time()
   with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=H,env=e,stdout=f,stderr=subprocess.STDOUT,pass_fds=lease_fds)
    save(Q/'workers'/f'gpu{gpu}.json',dict(status='running',method=method,sample_id=job['sample_id'],stage=stage,pid=os.getpid(),child_pid=child.pid,command=cmd,time=time.time()))
    code=child.wait()
   result['stages'].append(dict(stage=stage,returncode=code,elapsed_seconds=time.time()-started,log=str(log)))
   if code:raise RuntimeError(f'{stage} exited {code}; see {log}')
  validation=validate(method,folder);save(folder/'native_validation.json',validation)
  result.update(status='completed',validation=str(folder/'native_validation.json'),validation_sha256=sha(folder/'native_validation.json'),output=read(folder/'state.json')['output'])
  if job['sample_id']==PILOT:save(Q/f'{method}_pilot_gate.json',dict(status='passed',sample_id=PILOT,manifest_sha256=sha(Q/'manifest.json'),validation=result['validation'],validation_sha256=result['validation_sha256'],passed_at=time.time(),quality_selection=False))
 except Exception as e:
  result.update(status='failed',reason=repr(e),traceback=traceback.format_exc())
 finally:
  result['finished_at']=time.time();save(folder/'result.json',result);finish(job,result)
 print(json.dumps(dict(method=method,sample_id=job['sample_id'],status=result['status'],reason=result.get('reason'))),flush=True)
 return result

def worker(method,gpu):
 check();sys.path.insert(0,str(P/'scripts'));import dispatch_rigid111_direct as resources
 slot_lock=Q/f'worker_gpu{gpu}.lock';slot_lock.parent.mkdir(exist_ok=True)
 with slot_lock.open('a') as s:
  fcntl.flock(s,fcntl.LOCK_EX|fcntl.LOCK_NB)
  locks=[P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock',H/f'state/gpu_auxiliary_slot_{gpu}.lock',Path(f'/tmp/drivelab-gpu-locks/gpu{gpu}.lock')]
  while True:
   with contextlib.ExitStack() as stack:
    try:
     files=[]
     for p in locks:
      p.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
    except BlockingIOError:
     save(Q/'workers'/f'gpu{gpu}.json',dict(status='waiting_shared_gpu_lease',pid=os.getpid(),method=method,time=time.time()));time.sleep(10);continue
    ready,detail=resources.available(str(gpu))
    if not ready:
     save(Q/'workers'/f'gpu{gpu}.json',dict(status='waiting_gpu_idle',pid=os.getpid(),method=method,observation=detail,time=time.time()));time.sleep(10);continue
    while True:
     if shutil.disk_usage(H).free<10*1024**3:
      save(Q/'workers'/f'gpu{gpu}.json',dict(status='waiting_disk_10gib',pid=os.getpid(),method=method,time=time.time()));time.sleep(30);continue
     job=claim(method,gpu)
     if job is None or job=='pilot_failed':
      save(Q/'workers'/f'gpu{gpu}.json',dict(status='pilot_failed' if job else 'no_eligible_pending_jobs',method=method,pid=os.getpid(),time=time.time()));return
     result=execute(job,gpu,tuple(f.fileno() for f in files))
     if result['status']=='failed' and job['sample_id']==PILOT:return

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','worker']);p.add_argument('--method',choices=list(METHODS));p.add_argument('--gpu',type=int)
 a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='check':check()
 else:
  assert a.method and a.gpu in range(8);worker(a.method,a.gpu)
