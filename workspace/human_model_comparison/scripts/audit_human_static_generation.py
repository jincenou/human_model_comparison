"""CPU-only integrity settlement for the frozen two-method human generation.

Consume existing generation/reload evidence; never generate, re-render, or reject
an output by appearance. Cache hashes by inode/size/mtime/ctime so immutable large
payloads are hashed once rather than on every watcher iteration.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

H=Path(__file__).resolve().parents[1]
N=H/'runs/human_static_two_models_20261115_v1'
O=N/'generation_audit'
METHODS=['hunyuan3d21','reconviagen']


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Hashes:
    def __init__(self):
        p=O/'file_hash_cache.json';self.cache=read(p) if p.exists() else {};self.fresh=0;self.reused=0
    def sha(self,path,expected=None):
        p=Path(path);st=p.stat();signature=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        old=self.cache.get(str(p))
        if old and old['signature']==signature:
            result=old['sha256'];self.reused+=1
        else:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            result=h.hexdigest();self.fresh+=1
            self.cache[str(p)]=dict(signature=signature,sha256=result)
        if expected is not None:assert result==expected,(str(p),'SHA256 mismatch')
        return result
    def bind(self,path):return dict(path=str(path),sha256=self.sha(path))


def object_audit(job,source,manifest_sha,hashes):
    import numpy as np
    from PIL import Image
    method,sid=job['method'],job['sample_id'];folder=N/'objects'/method/sid
    rp=Path(job['result']);assert rp==folder/'result.json'
    result=read(rp)
    assert (result['method'],result['sample_id'],result['status'])==(method,sid,job['status'])
    assert result['manifest_sha256']==manifest_sha and result['real_input_count']==1 and result['generated_input_count']==0
    assert result['finished_at']>=result['started_at']
    if job['status']=='failed':
        assert result.get('reason') and job.get('reason')==result['reason']
        return dict(method=method,id=sid,status='recorded_generation_failure',reason=result['reason'],result=hashes.bind(rp),quality_retry=False)
    assert job['status']=='completed'
    state_path=folder/'state.json';state=read(state_path)
    assert state['sample_id']==sid and state['stage']=='finished'
    assert state['real_input_count']==1 and state['generated_input_count']==0
    if method=='hunyuan3d21':
        assert state['input_manifest_sha256']==manifest_sha and state['input']==source['image_rgba'] and state['input_sha256']==source['image_rgba_sha256']
        expected_stages=['generate','glb_reload']
    else:
        assert state['manifest_sha256']==manifest_sha and state['inputs']==[dict(path=source['image_rgba'],sha256=source['image_rgba_sha256'])]
        expected_stages=['generate','native_reload','glb_reload']
    assert [s['stage'] for s in result['stages']]==expected_stages
    assert all(s['returncode']==0 for s in result['stages'])
    for stage in result['stages']:assert Path(stage['log']).is_file()
    vp=Path(result['validation']);assert vp==folder/'native_validation.json'
    hashes.sha(vp,result['validation_sha256']);validation=read(vp)
    assert (validation['status'],validation['method'],validation['sample_id'])==('passed',method,sid)
    assert validation['geometry_positive'] and not validation['quality_filter_applied']
    assert not validation['scores_produced']
    output=Path(result['output']);assert output==Path(state['output']) and output.parent==folder
    output_sha=hashes.sha(output,state['output_sha256'])
    with output.open('rb') as f:
        header=f.read(12)
    assert header[:4]==b'glTF' and int.from_bytes(header[4:8],'little')==2 and int.from_bytes(header[8:12],'little')==output.stat().st_size
    reload_path=folder/'reload_render_validation.json';reload=read(reload_path)
    assert reload['sample_id']==sid and Path(reload['run_dir'])==folder and reload['output_sha256']==output_sha
    assert reload['mesh_objects']>0 and reload['texture_images']>0
    assert [v['path'] for v in validation['views']]==reload['views']
    for view in validation['views']:
        hashes.sha(view['path'],view['sha256']);pixels=np.asarray(Image.open(view['path']).convert('RGBA'))
        assert pixels.shape==(512,512,4) and int((pixels[:,:,3]>127).sum())==view['foreground_pixels']
    assert validation['all_preview_views_empty']==(not any(v['foreground_pixels'] for v in validation['views']))
    payloads=[dict(path=str(output),sha256=output_sha)]
    native_reload=None
    if method=='reconviagen':
        for name,sha in state['output_hashes'].items():
            assert Path(name).name==name
            hashes.sha(folder/name,sha);payloads.append(dict(path=str(folder/name),sha256=sha))
        native_reload=folder/'reload_validation.json';g=read(native_reload)
        assert g['gaussians']==state['gaussians'] and g['gaussians']>0
        hashes.sha(folder/'native_gaussians.pt',g['checkpoint_sha256'])
        for i,view in enumerate(g['views']):
            hashes.sha(view['path'],view['sha256'])
            pred=np.asarray(Image.open(view['path']).convert('RGB')).astype(np.int16)
            original=np.asarray(Image.open(folder/f'native_preview_{i}.png').convert('RGB')).astype(np.int16)
            assert pred.shape==original.shape
            difference=np.abs(pred-original)
            assert int(difference.max())==view['max_pixel_difference']<=1
            assert abs(float(difference.mean())-view['mean_pixel_difference'])<1e-12 and view['mean_pixel_difference']<=.01
    return dict(method=method,id=sid,status='passed_native_integrity',result=hashes.bind(rp),state=hashes.bind(state_path),
                validation=hashes.bind(vp),reload_evidence=hashes.bind(reload_path),
                gaussian_reload_evidence=hashes.bind(native_reload) if native_reload else None,payloads=payloads,
                identity_key=source['identity_key'],timestamp_us=source['timestamp_us'],input=source['image_rgba'],input_sha256=source['image_rgba_sha256'],
                real_input_count=1,generated_input_count=0,native_single_view_reloaded=True,quality_filter_applied=False)


def run():
    O.mkdir(parents=True,exist_ok=True);hashes=Hashes()
    manifest=read(N/'manifest.json');queue=read(N/'queue.json');manifest_sha=hashes.sha(N/'manifest.json',queue['manifest_sha256'])
    hashes.sha(manifest['selection'],manifest['selection_sha256']);selected={x['id']:x for x in read(manifest['selection'])['objects']}
    sources={r['sample_id']:r for r in manifest['objects']};ids=set(sources)
    assert len(manifest['objects'])==len(ids)==100 and ids==set(selected) and 'HN0646' in ids and 'HN0035' not in ids
    assert manifest['methods']==METHODS and manifest['settings']['disk_guard_gib']==10
    for path,sha in manifest['code_hashes'].items():hashes.sha(path,sha)
    hashes.sha(manifest['recon_single_image_support']['source'],manifest['recon_single_image_support']['sha256'])
    for sid,row in sources.items():
        assert row['identity_key']==selected[sid]['identity_key'] and row['timestamp_us']==selected[sid]['t_ref_us']
        assert row['real_input_count']==1 and row['generated_input_count']==0 and len(row['views'])==1
        view=row['views'][0];assert view['image']==row['image_rgba'] and view['timestamp_us']==row['timestamp_us'] and view['sha256']==row['image_rgba_sha256']
        for path_key,hash_key in [('image_rgba','image_rgba_sha256'),('raw_source_path','raw_source_sha256'),('exact_mask_path','exact_mask_sha256')]:hashes.sha(row[path_key],row[hash_key])
    jobs=queue['jobs'];assert len(jobs)==len({(j['method'],j['sample_id']) for j in jobs})==200
    for method in METHODS:assert {j['sample_id'] for j in jobs if j['method']==method}==ids
    counts={m:dict(collections.Counter(j['status'] for j in jobs if j['method']==m)) for m in METHODS}
    rows=[];errors=[]
    for job in jobs:
        if job['status'] not in ['completed','failed']:continue
        try:rows.append(object_audit(job,sources[job['sample_id']],manifest_sha,hashes))
        except Exception as exc:errors.append(dict(method=job['method'],id=job['sample_id'],reason=repr(exc),traceback=traceback.format_exc()))
    terminal=all(j['status'] in ['completed','failed'] for j in jobs)
    result=dict(status='failed' if errors else 'passed_complete' if terminal else 'passed_subset',
                terminal=terminal,expected_tasks=200,objects_per_method=100,counts=counts,
                manifest=hashes.bind(N/'manifest.json'),selection=hashes.bind(manifest['selection']),queue=hashes.bind(N/'queue.json'),
                completed_native_records=sum(r['status']=='passed_native_integrity' for r in rows),
                recorded_failures=sum(r['status']=='recorded_generation_failure' for r in rows),
                errors=errors,rows=rows,real_single_view_input_verified=True,no_flux_or_duplicate_pose_views=True,
                old_intermediate_statuses_not_used_as_terminal=True,quality_filter_applied=False,quality_retries=0,
                hash_policy='SHA256 cached only while device/inode/size/mtime/ctime unchanged; changed files rehashed',
                files_hashed_this_pass=hashes.fresh,cached_hash_checks_this_pass=hashes.reused,
                scope='Native generation/export/reload and frozen input identity only; no GPU rerun or source-view metric eligibility claimed',
                script=hashes.bind(__file__),updated_at=time.time())
    save(O/'file_hash_cache.json',hashes.cache);save(O/'latest.json',result)
    if terminal:save(O/'final_certificate.json',result)
    print(json.dumps({k:result[k] for k in ['status','terminal','counts','completed_native_records','recorded_failures','files_hashed_this_pass','cached_hash_checks_this_pass']}),flush=True)
    return result


def main(watch,interval):
    O.mkdir(parents=True,exist_ok=True)
    with (O/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            q=N/'queue.json';st=q.stat();signature=(st.st_size,st.st_mtime_ns)
            if signature!=previous:
                try:
                    result=run();previous=signature
                    state=dict(status=result['status'],pid=os.getpid(),terminal=result['terminal'],updated_at=time.time())
                    save(O/'watcher.json',state)
                    if result['terminal'] or not watch:return
                except Exception as exc:
                    save(O/'watcher.json',dict(status='audit_execution_error',pid=os.getpid(),reason=repr(exc),traceback=traceback.format_exc(),updated_at=time.time()))
                    if not watch:raise
            if not watch:return
            time.sleep(interval)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args();main(a.watch,a.interval)
