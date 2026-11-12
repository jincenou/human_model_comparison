"""Recover retired RGBA views from original, hash-verified videos.

Use the original segmentation function without changing its numerical body.
Only the view loop and completed temporal admission call are scoped here.
Output is isolated; exact original PNG hashes determine recovery, not quality.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

H=Path(__file__).resolve().parents[1]
CODE=H.parent/'human'
SOURCE=H.parent/'asset_work/human_nuscenes100_20261026'
ROOT=H/'runs/backend_ablation_cache_restore'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');t.replace(p)


def worker(sid,views,folder):
    sys.path.insert(0,str(CODE));os.environ['HUMAN_RUN_ROOT']=str(SOURCE)
    import dataset
    manifest=SOURCE/'data'/sid/'fdanyone.json';m=read(manifest);config=read(SOURCE/'config'/f'{sid}.json')
    assert sha(manifest)==config['provenance']['manifest_sha256']
    source=SOURCE/'outputs'/sid/'generation';videos=[]
    validation=read(SOURCE/'reports'/f'{sid}_frontend_validation.json')
    assert validation['state']=='all_videos_decoded_cameras_matched'
    for view in views:
        video=source/'videos'/f'{view:02d}.mp4';expected=next(v for v in validation['videos']if v['view']==view)
        assert expected['sha256']==sha(video) and expected['frames']==m['frame_count']
        videos.append(dict(view=view,path=str(video),sha256=expected['sha256']))
        assert not (folder/'rgba/images'/f'cam{view:02d}').exists()
    assert read(source/'metadata.json')['output']['frames_per_video']==m['frame_count']
    sys.path.insert(0,str(CODE/'vendor/4danyone'))
    from fdanyone.assets import resolve_foreground_model,BIREFNET_FILES
    weights=resolve_foreground_model(CODE/'vendor/4danyone/models')
    pins=[CODE/'dataset.py',CODE/'vendor/sam_transfer_guard.py',CODE/'vendor/4danyone/fdanyone/foreground.py',Path(dataset.RESOURCES['sam2_weights']),*[weights/f for f in BIREFNET_FILES]]
    provenance={str(p):sha(p)for p in pins}
    tree=ast.parse((CODE/'dataset.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='masks')
    changed=[];body=[]
    for node in fn.body:
        if isinstance(node,ast.ImportFrom)and node.module=='frontend':
            assert [n.name for n in node.names]==['apply_temporal_policy'];changed.append('omit_temporal_admission_import');continue
        if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='apply_temporal_policy':
            changed.append('completed_frozen_video_temporal_policy_verified_read_only');continue
        if isinstance(node,ast.For)and isinstance(node.target,ast.Name)and node.target.id=='view':
            assert ast.unparse(node.iter)=='range(group, 24, 4)'
            node.iter=ast.List(elts=[ast.Constant(value=v)for v in views],ctx=ast.Load());changed.append('requested_existing_views_only')
        body.append(node)
    assert len(changed)==3
    fn.body=body;module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    replay=ast.unparse(module);(folder/'mask_replay.py').write_text(replay+'\n')
    namespace=dict(dataset.__dict__,ROOT=folder,locations=lambda identity:(source,folder/'rgba',m['frame_count']))
    record=dict(status='replaying_original_segmentation',id=sid,views=views,pid=os.getpid(),started_at=time.time(),manifest=str(manifest),manifest_sha256=sha(manifest),videos=videos,original_dependency_hashes=provenance,scoped_changes=changed,compiled_replay_sha256=sha(folder/'mask_replay.py'),diffusion_generation_rerun=False,old_assets_modified=False)
    save(folder/'recovery.json',record)
    exec(compile(module,str(folder/'mask_replay.py'),'exec'),namespace)
    namespace['masks'](sid,views[0])
    rows=[]
    for expected in [r for r in m['input_hashes']if r['view']in views]:
        p=folder/'rgba/images'/f'cam{expected["view"]:02d}'/f'{expected["frame"]:06d}.png';digest=sha(p)
        rows.append(dict(view=expected['view'],frame=expected['frame'],path=str(p),sha256=digest,original_sha256=expected['sha256'],exact_match=digest==expected['sha256']))
    for p,digest in provenance.items():assert sha(p)==digest
    assert sha(manifest)==record['manifest_sha256'] and all(sha(v['path'])==v['sha256']for v in videos)
    exact=sum(r['exact_match']for r in rows);record.update(status='exact_original_rgba_restored'if exact==len(rows)else'replay_differs_from_original_cache',frames=rows,exact_matches=exact,expected_frames=m['frame_count']*len(views),finished_at=time.time(),training_started=False)
    save(folder/'recovery.json',record);print(record['status'],exact,'/',len(rows),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);group=p.add_mutually_exclusive_group();group.add_argument('--view',type=int);group.add_argument('--views',nargs='+',type=int);p.add_argument('--gpu',type=int,default=2);p.add_argument('--worker',action='store_true');a=p.parse_args()
    active=read(H/'inputs/active_human_benchmark.json');assert a.human_id in {r['id']for r in read(active['selection'])['objects']}
    views=a.views if a.views is not None else[0 if a.view is None else a.view]
    assert len(views)==len(set(views))and all(0<=v<24 for v in views)
    folder=ROOT/a.human_id/(('view_'if len(views)==1 else'views_')+'_'.join(f'{v:02d}'for v in views))
    if a.worker:worker(a.human_id,views,folder);return
    policy=read(H/'state/gpu_resource_policy.json');assert a.gpu in policy['allowed_gpu_indices']and a.gpu not in policy['excluded_gpu_indices']
    with (H/'state'/f'gpu_auxiliary_slot_{a.gpu}.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        uuid,free=data[0].strip(),int(data[1]);assert free>=32768 and shutil.disk_usage(H).free>10*1024**3
        folder.mkdir(parents=True,exist_ok=False)
        sys.path.insert(0,str(CODE));import common
        common.ROOT=folder;env=common.environment(a.gpu,'4danyone-sam2')
        env.update(HUMAN_RUN_ROOT=str(SOURCE),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        command=[str(common.PYTHONS['4danyone-sam2']),'-B','-u',str(Path(__file__).resolve()),'--worker','--human-id',a.human_id,'--views',*[str(v)for v in views]]
        state=dict(status='running',id=a.human_id,views=views,pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,command=command,started_at=time.time(),script_sha256=sha(__file__))
        save(folder/'controller.json',state)
        with (folder/'worker.log').open('x')as log:child=subprocess.Popen(command,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
        state['worker_pid']=child.pid;save(folder/'controller.json',state);code=child.wait()
        state.update(status='finished'if code==0 else'failed_retained',exit_code=code,finished_at=time.time());save(folder/'controller.json',state)
        if code:raise RuntimeError('RGBA recovery failed; inspect retained worker.log')


if __name__=='__main__':main()
