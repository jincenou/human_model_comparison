"""CPU-only, independent arithmetic and artifact audit for 200 new human tasks.

A watcher audits completed subsets as they appear, then forces a complete hash
and arithmetic replay after both source and geometry queues reach terminal state.
No metrics, native assets or existing benchmark results are modified.
"""
import argparse,collections,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist
import evaluate_human_static_two_models as src
H=src.H;O=H/'reports/human_static_two_models_final_audit_20261115_v1';G=H/'reports/human_static_two_models_geometry_20261115_v1';HG=H/'evaluation_static_cd_fscore_time_v1'
METHODS=['hunyuan3d21','reconviagen'];PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']
S=Path(__file__).resolve();read=src.read;save=src.save

def close(a,b,tol=1e-10):
    assert a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
    assert abs(a-b)<=tol,(a,b,abs(a-b),tol)
    return abs(a-b)

class HashAudit:
    def __init__(self,force=False):
        self.path=O/'hash_cache.json';self.cache=read(self.path) if self.path.exists() else {};self.force=force;self.checked={}
    def digest(self,path):
        p=Path(path);st=p.stat();key=str(p);sig=[st.st_size,st.st_mtime_ns];old=self.cache.get(key)
        digest=old['sha256'] if not self.force and old and old['stat']==sig else src.sha(p)
        self.cache[key]=dict(stat=sig,sha256=digest);self.checked[key]=digest;return digest
    def verify(self,path,expected):
        actual=self.digest(path);assert actual==expected,('Artifact hash differs',str(path),actual,expected);return actual
    def pairs(self,d,keys):
        for key in keys:self.verify(d[key],d[key+'_sha256'])
    def finish(self):save(self.path,self.cache)

def independent_ssim(a,b):
    a=a.astype(np.float64)/255.;b=b.astype(np.float64)/255.
    x=np.arange(-5,6,dtype=float);g=np.exp(-x*x/(2*1.5**2));g/=g.sum();kernel=np.outer(g,g)[...,None]
    def smooth(v):return fftconvolve(v,kernel,mode='valid',axes=(0,1))
    ma,mb=smooth(a),smooth(b);va=smooth(a*a)-ma*ma;vb=smooth(b*b)-mb*mb;cov=smooth(a*b)-ma*mb
    return float((((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean())

def independent_iou(a,b):
    a=a>127;b=b>127;assert a.any();return float(np.logical_and(a,b).sum()/np.logical_or(a,b).sum())

def image_audit(path,h,identities,refs,protocols,cache,force):
    row=read(path);key=(row['method'],row['id']);assert row['id'] in identities
    q=refs[row['id']];identity=identities[row['id']]
    assert q['identity_key']==identity['identity_key'] and q['timestamp_us']==identity['timestamp_us']
    if row['status']=='registration_unavailable':
        p=src.OUT/row['method']/'observations'/row['id']/'registration_status.json';s=read(p)
        assert s['status']=='unavailable' and s['reason']==row['reason'];return dict(id=row['id'],method=row['method'],status='expected_registration_NA',reason=row['reason'])
    assert row['status']=='scored';assert row['protocol_sha256'] in protocols
    assert row['reference']==q;assert row['selection_sha256']==read(src.OUT/'protocol.json')['selection_sha256']
    h.pairs(q,['reference_rgb','reference_mask','camera_contract'])
    render=row['source_render'];h.pairs(render,['source_rgb','source_alpha','registration','payload','camera_contract'])
    reg=read(render['registration']);assert reg['identity_key']==identity['identity_key'] and reg['timestamp_us']==identity['timestamp_us']
    assert reg['id']==row['id'] and reg['method']==row['method'];h.verify(reg['payload'],reg['payload_sha256']);h.verify(reg['camera_contract'],q['camera_contract_sha256'])
    f=row['files'];h.pairs(f,['reference','reference_mask','prediction_unwarped','prediction_mask','prediction_aligned'])
    signature=h.digest(path);ck=row['method']+'/'+row['id'];prior=cache.get(ck)
    if prior and prior['row_sha256']==signature and not force:return prior
    a=np.asarray(Image.open(f['reference']).convert('RGB'));b=np.asarray(Image.open(f['prediction_aligned']).convert('RGB'))
    assert a.shape==b.shape==(512,512,3)
    mse=float(np.mean((a.astype(float)/255.-b.astype(float)/255.)**2));psnr=-10*math.log10(mse) if mse else math.inf
    assert math.isfinite(psnr),'Exact image match requires positive-infinity summary handling'
    ssim=independent_ssim(a,b);gt=np.asarray(Image.open(f['reference_mask']).convert('L'));pred=np.asarray(Image.open(f['prediction_mask']).convert('L'));iou=independent_iou(gt,pred)
    gt_raw=np.asarray(Image.open(q['reference_mask']).convert('L'));pred_raw=np.asarray(Image.open(render['source_alpha']).convert('L'));raw_iou=independent_iou(gt_raw,pred_raw)
    expected=dict(PSNR=psnr,SSIM=ssim,IoU=iou,IoU_raw=raw_iou);errors={}
    for metric,v in expected.items():assert row['metrics'][metric]['status']=='ok';errors[metric]=close(v,row['metrics'][metric]['value'],2e-10)
    lp=row['metrics']['LPIPS'];assert lp['status']=='ok' and math.isfinite(lp['value']) and lp['value']>=0
    report=dict(id=row['id'],method=row['method'],status='passed',row_sha256=signature,recomputed=expected,absolute_errors=errors,LPIPS='record/weights/provenance verified; neural inference not repeated in CPU arithmetic audit',identity_verified=True,hashes_verified=True)
    cache[ck]=report;return report

def edp_audit(root,h):
    result=read(root/'results.json');h.verify(root/'inputs.json',result['inputs_sha256']);h.verify(root/'protocol.json',result['protocol_sha256'])
    manifest=read(root/'inputs.json');cases={x['key']:x for x in manifest['cases']};rows=[]
    for row in result['rows']:
        sid,method=row['id'],row['method'];assert method in METHODS;arrays=[]
        for key in [sid+'_reference',sid+'_'+method]:
            case=cases[key];h.pairs(case,['image','mask']);p=root/'features'/(key+'.npz');meta=read(p.with_suffix('.json'));h.verify(p,meta['features_sha256']);h.verify(root/'poses'/(key+'.json'),meta['pose_sha256']);assert meta['image_sha256']==case['image_sha256'] and meta['mask_sha256']==case['mask_sha256']
            data=dict(np.load(p));assert data['embedding'].shape==(6,1024) and np.isfinite(data['embedding']).all();assert np.array_equal(data['counts'],[(data['labels']==i).sum() for i in range(6)]);arrays.append(data)
        ref,pred=arrays;distances=[];parts=[]
        for i,name in enumerate(PARTS):
            nr,nn=int(ref['counts'][i]),int(pred['counts'][i]);rr=ref['embedding'][i].astype(float);pp=pred['embedding'][i].astype(float);norm=float(np.sqrt(np.square(rr).sum())*np.sqrt(np.square(pp).sum()));detail=row['parts'][name]
            assert detail['reference_patches']==nr and detail['prediction_patches']==nn
            if nr and nn and math.isfinite(norm) and norm>1e-12:
                distance=1-float(np.clip(np.sum(rr*pp)/norm,-1,1));close(distance,detail['distance'],2e-12);distances.append(distance);parts.append(name)
            else:assert detail['distance'] is None
        assert parts==row['valid_parts'] and len(parts)==row['valid_part_count']
        if distances:assert row['score']['status']=='ok';close(float(np.mean(distances)),row['score']['value'],2e-12)
        else:assert row['score']['status']=='na' and row['score']['value'] is None
        rows.append(dict(id=sid,method=method,status='passed',value=float(np.mean(distances)) if distances else None,valid_parts=parts,valid_part_count=len(parts),feature_hashes_verified=True))
    return rows

def geometry_audit(path,h,identities,refs):
    row=read(path);sid,method=row['id'],row['method'];assert sid in identities
    if row['status']!='completed':return dict(id=sid,method=method,status=row['status'],reason=row.get('reason'))
    ref=refs[sid];assert ref['identity_key']==row['identity_key']==identities[sid]['identity_key'];assert ref['t_ref_us']==row['t_ref_us']==identities[sid]['timestamp_us']
    assert row['protocol_sha256']==h.digest(G/'protocol.json');h.verify(ref['queries'],row['queries_sha256']);h.pairs(row,['certificate','returns','source_registration'])
    cert=read(row['certificate']);h.pairs(cert,['asset','extent','camera_contract']);h.verify(cert['source_registration'],cert['source_registration_sha256'])
    assert cert['reference_manifest_sha256']==h.digest(HG/'reference_manifest.json');assert cert['queries_sha256']==row['queries_sha256'];A=np.asarray(cert['T_world_from_native']);s=cert['scale_m_per_native_unit'];np.testing.assert_allclose(A[:3,:3]@A[:3,:3].T,np.eye(3)*s*s,atol=1e-8);close(s*cert['native_vertical_extent'],cert['annotation_vertical_extent_m'],1e-9)
    q=dict(np.load(ref['queries']));ret=dict(np.load(row['returns']));hit=ret['hit'];p=q['o'][hit]+ret['range_pred'][hit,None]*q['u'][hit];np.testing.assert_allclose(p,ret['predicted_endpoints'],atol=1e-12,rtol=0)
    assert hit.dtype==np.bool_ and len(hit)==len(q['q']);np.testing.assert_allclose(q['q'],q['o']+q['r'][:,None]*q['u'],atol=1e-7,rtol=0)
    met=row['panels']['annotation_vertical_extent_conditioned']['metrics'];assert met['reference_count']==len(q['q']) and met['hit_count']==len(p);close(float(hit.mean()),met['ray_hit'])
    if len(p):
        distances=cdist(p,q['q'],metric='euclidean');pg=distances.min(1);gp=distances.min(0);cd=.5*(float(pg.mean())+float(gp.mean()));close(cd,met['cd_m'],1e-10)
        np.testing.assert_allclose(pg,ret['distances_pg'],atol=1e-10,rtol=0);np.testing.assert_allclose(gp,ret['distances_gp'],atol=1e-10,rtol=0)
    else:cd=None;pg=np.empty(0);gp=np.full(len(q['q']),np.inf);assert met['cd_m'] is None and met['status']=='empty_pred'
    out=dict(id=sid,method=method,status='passed',reference_panel=ref['candidate_panel'],CD_cm=None if cd is None else cd*100,valid_empty=not len(p),RayHit=float(hit.mean())*100)
    for cm in [5,10,20]:
        precision=float((pg<cm/100).mean()) if len(pg) else 0.;recall=float((gp<cm/100).mean());f=2*precision*recall/(precision+recall) if precision+recall else 0.;close(f,met[f'f1_{cm}cm']);close(precision,met[f'precision_{cm}cm']);close(recall,met[f'recall_{cm}cm']);out[f'F{cm}']=f*100
    return out

def aggregate(rows,key,valid):
    selected=[r for r in rows if valid(r)];ids=sorted(r['id'] for r in selected);assert len(ids)==len(set(ids));vals=[r[key] for r in selected];return dict(value=float(np.mean(vals)) if vals else None,n=len(vals),ids=ids)

def settle_upstream_failures():
    """A failed source job is terminal execution-NA, never a missing GPU job.

    Preserve existing geometry results. A later infrastructure repair can
    explicitly reopen this recorded task; the watcher does not retry models or
    turn an unavailable score into zero.
    """
    changes=[];protocol=src.sha(G/'protocol.json')
    for job in read(src.OUT/'queue.json')['jobs']:
        if job['status']!='evaluation_stage_failed':continue
        path=G/'results'/job['method']/(job['id']+'.json')
        if path.exists():continue
        record=dict(id=job['id'],method=job['method'],status='execution_failure_pending_review',
            reason='Upstream source evaluation failed; geometry was not attempted',
            upstream_source_failure=True,upstream_failure=job.get('failure'),
            source_task_finished_at=job.get('finished_at'),protocol_sha256=protocol,
            finished_at=time.time(),metric_zero_imputation=False)
        save(path,record);changes.append(dict(id=job['id'],method=job['method']))
    return changes

def run(force=False):
    O.mkdir(parents=True,exist_ok=True);propagated=settle_upstream_failures();h=HashAudit(force);failures=[];image_rows=[];edp_rows=[];geom_rows=[];cachepath=O/'image_replay_cache.json';cache=read(cachepath) if cachepath.exists() else {}
    protocol=read(src.OUT/'protocol.json');gen=read(src.GEN/'manifest.json');h.verify(src.GEN/'manifest.json',protocol['generation_manifest_sha256']);identities={r['sample_id']:r for r in gen['objects']};ids=set(protocol['ids']);assert ids==set(identities) and len(ids)==100
    h.verify(protocol['selection'],protocol['selection_sha256']);h.verify(protocol['scoring_reference_summary'],protocol['scoring_reference_summary_sha256']);references={r['id']:r['reference'] for r in read(protocol['scoring_reference_summary'])['rows'] if r.get('reference')};refgeo=read(HG/'reference_manifest.json');geo_refs={r['id']:r for r in refgeo['rows']};assert set(geo_refs)==ids
    phashes={h.digest(p) for p in [src.OUT/'protocol.json',*(src.OUT/'preflight_history').glob('*.json')]}
    for path,digest in protocol['code'].items():h.verify(path,digest)
    for path,digest in read(G/'protocol.json')['code'].items():h.verify(path,digest)
    lpips=read(src.R/'reports/source_appearance_alignment_20261115_v1/protocol.json')['lpips']
    for name in ['backbone','linear']:h.verify(lpips[name],lpips[name+'_sha256'])
    ep=read(H/'reports/source_edp_common_v1/cohort_reference_revision_v2/protocol.json')
    for mapping in ['weights','configurations','code']:
        for path,digest in ep[mapping].items():h.verify(path,digest)
    queue=read(src.OUT/'queue.json')['jobs'];pairs={(j['method'],j['id']) for j in queue};assert pairs=={(m,s) for m in METHODS for s in ids} and len(queue)==200
    generated=[]
    for job in read(src.GEN/'queue.json')['jobs']:
        if job['status']!='completed':continue
        method,sid=job['method'],job['sample_id'];folder=src.GEN/'objects'/method/sid
        try:
            result=read(folder/'result.json');state=read(folder/'state.json');identity=identities[sid];assert result['method']==method and result['sample_id']==sid and result['manifest_sha256']==h.digest(src.GEN/'manifest.json');h.verify(result['validation'],result['validation_sha256']);assert read(result['validation'])['status']=='passed'
            if method=='hunyuan3d21':assert state['input_sha256']==identity['image_rgba_sha256'];h.verify(state['input'],state['input_sha256']);payload=folder/'demo_textured.glb';expected=state['output_sha256']
            else:assert len(state['inputs'])==1 and state['inputs'][0]['sha256']==identity['image_rgba_sha256'];h.verify(state['inputs'][0]['path'],state['inputs'][0]['sha256']);payload=folder/'native_gaussians.pt';expected=state['output_hashes'][payload.name]
            h.verify(payload,expected);generated.append(dict(method=method,id=sid,status='passed'))
        except Exception as e:failures.append(dict(stage='generation_binding',method=method,id=sid,reason=repr(e),traceback=traceback.format_exc()))
    for method in METHODS:
        for p in (src.OUT/'rows'/method).glob('*.json'):
            try:image_rows.append(image_audit(p,h,identities,references,phashes,cache,force))
            except Exception as e:failures.append(dict(stage='image',path=str(p),reason=repr(e),traceback=traceback.format_exc()))
        for p in (src.OUT/method/'edp').glob('*/results.json'):
            try:edp_rows+=edp_audit(p.parent,h)
            except Exception as e:failures.append(dict(stage='EDP',path=str(p),reason=repr(e),traceback=traceback.format_exc()))
        for p in (G/'results'/method).glob('*.json'):
            try:geom_rows.append(geometry_audit(p,h,identities,geo_refs))
            except Exception as e:failures.append(dict(stage='geometry',path=str(p),reason=repr(e),traceback=traceback.format_exc()))
    for label,rows in [('generation',generated),('image',image_rows),('EDP',edp_rows),('geometry',geom_rows)]:
        pairs=[(r['method'],r['id']) for r in rows];assert len(pairs)==len(set(pairs)),('Duplicate',label)
    counts={m:dict(collections.Counter(j['status'] for j in queue if j['method']==m)) for m in METHODS};source_terminal=all(j['status'] not in ['running','awaiting_generation'] for j in queue);geometry_terminal=len(geom_rows)==200 and all(r['status']!='running' for r in geom_rows)
    summary=dict(source={},geometry={})
    for method in METHODS:
        images=[dict(r,**r.get('recomputed',{})) for r in image_rows if r['method']==method];ep=[r for r in edp_rows if r['method']==method]
        cells={k:aggregate(images,k,lambda r:r['status']=='passed') for k in ['PSNR','SSIM','IoU','IoU_raw']};cells['EDP']=aggregate(ep,'value',lambda r:r['value'] is not None);cells['EDP']['parts']=sum(r['valid_part_count'] for r in ep if r['value'] is not None)
        rawrows=[read(p) for p in (src.OUT/'rows'/method).glob('*.json')];lp=[dict(id=r['id'],value=r['metrics']['LPIPS']['value']) for r in rawrows if r.get('metrics',{}).get('LPIPS',{}).get('status')=='ok'];cells['LPIPS']=aggregate(lp,'value',lambda r:True);summary['source'][method]=cells
        panels={}
        for panel in ['main','low_support_exploratory']:
            gr=[r for r in geom_rows if r['method']==method and r['status']=='passed' and r['reference_panel']==panel];gc={k:aggregate(gr,k,lambda r,k=k:r[k] is not None) for k in ['CD_cm','F5','F10','F20','RayHit']};gc['CD_cm']['valid_empty_n']=sum(r['valid_empty'] for r in gr);panels[panel]=gc
        summary['geometry'][method]=panels
    # Final summaries are compared only once producers are quiescent; a partial
    # producer's summary may legitimately lag a newly published per-object row.
    final=source_terminal and geometry_terminal;summary_checked=False
    if final and not failures:
        try:
            ss=read(src.OUT/'summary.json')['groups'];gs=read(G/'summary.json')['groups']
            for m in METHODS:
                for k,cell in summary['source'][m].items():
                    old=ss[m][k];assert old['n']==cell['n'] and old['ids']==cell['ids']
                    if cell['n']:close(old['value'],cell['value'],2e-10)
                for panel,cells in summary['geometry'][m].items():
                    for k,cell in cells.items():
                        old=gs[m][panel][k];assert old['n']==cell['n'] and old['ids']==cell['ids']
                        if cell['n']:close(old['value'],cell['value'],2e-8)
                        if k=='CD_cm':assert old['valid_empty_n']==cell['valid_empty_n']
            summary_checked=True
        except Exception as e:failures.append(dict(stage='final_summary',reason=repr(e),traceback=traceback.format_exc()))
    execution=sum(j['status']=='evaluation_stage_failed' for j in queue)+sum(r['status']=='execution_failure_pending_review' for r in geom_rows)
    status='failed' if failures else 'needs_execution_review' if final and execution else 'passed_complete' if final and force else 'passed_subset'
    audit=dict(status=status,created_at=time.time(),script=str(S),script_sha256=h.digest(S),force_full_hash_and_replay=force,scope=dict(expected_source_tasks=200,expected_geometry_tasks=200,source_terminal=source_terminal,geometry_terminal=geometry_terminal,source_counts=counts,execution_failures=execution,generated_bindings=len(generated),image_records=len(image_rows),EDP_records=len(edp_rows),geometry_records=len(geom_rows)),summary_verified=summary_checked,checks=dict(identity_and_hash='Frozen exact100 IDs, source identity/timestamp/camera, original payload and PNG/features/return hashes',image='Independent PSNR, FFT Gaussian valid-window SSIM, binary own-mask IoU from saved PNGs',EDP='Independent cosine distances of cached six-part embeddings, own-mask labels/counts and valid-part mean',geometry='Independent brute-force pairwise Euclidean nearest neighbors, both directions CD/2, strict5/10/20cm F-score, valid-empty policy, cm and percent conversion',aggregation='Own-valid ids and n, duplicate rejection; full producer summaries compared at terminal state',LPIPS='Existing score record + native PNG/code/weight protocol provenance; no neural forward rerun'),generated=generated,images=image_rows,EDP=edp_rows,geometry=geom_rows,recomputed_summary=summary,failures=failures)
    audit['upstream_failures_settled_this_pass']=propagated
    save(O/'latest.json',audit);save(cachepath,cache);h.finish()
    if final and force:save(O/'final_certificate.json',audit)
    print(json.dumps(dict(status=status,source_terminal=source_terminal,geometry_terminal=geometry_terminal,image_records=len(image_rows),geometry_records=len(geom_rows),failures=len(failures))),flush=True);return audit

def watch(interval):
    previous=None
    while True:
        paths=[src.OUT/'queue.json',*list((src.OUT/'rows').glob('*/*.json')),*list((src.OUT).glob('*/edp/*/results.json')),*list((G/'results').glob('*/*.json'))]
        signature=[(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths]
        if signature!=previous:
            try:
                audit=run();previous=signature
                if audit['scope']['source_terminal'] and audit['scope']['geometry_terminal']:
                    # Allow the last worker to publish its final summary.
                    time.sleep(2);audit=run(force=True)
                    save(O/'watcher.json',dict(status=audit['status'],pid=os.getpid(),finished_at=time.time()));break
            except Exception as e:save(O/'watcher.json',dict(status='audit_execution_error',pid=os.getpid(),reason=repr(e),traceback=traceback.format_exc(),updated_at=time.time()))
        save(O/'watcher.json',dict(status='watching_incremental_results',pid=os.getpid(),updated_at=time.time(),interval_seconds=interval));time.sleep(interval)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=60);p.add_argument('--force-full',action='store_true');a=p.parse_args()
    if a.watch:watch(a.interval)
    else:run(a.force_full)
