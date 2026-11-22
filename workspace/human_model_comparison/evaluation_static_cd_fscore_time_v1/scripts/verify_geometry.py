"""Independent dense-distance and artifact/transform audit of every scored row."""
import time
import numpy as np
from common import *
hashes={}
def check(p,h):
    p=str(p)
    if p not in hashes:hashes[p]=sha(p)
    assert hashes[p]==h,('hash mismatch',p)
def equal(a,b):
    if a is None or b is None:assert a is b,(a,b)
    elif isinstance(a,str):assert a==b
    else:np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-10)
def independent_metrics(q,o,u,r,rp,hit):
    p=o[hit]+rp[hit,None]*u[hit];n=len(q);count=len(p);result=dict(reference_count=n,hit_count=count,ray_hit=count/n,status='ok' if count else 'empty_pred')
    # Dense pairwise distances, independently of the KD-tree implementation.
    if count:
        distance=np.sqrt(((p[:,None,:]-q[None,:,:])**2).sum(2));dpg=distance.min(1);dgp=distance.min(0);result.update(cd_pg_m=float(dpg.mean()),cd_gp_m=float(dgp.mean()),cd_m=float((dpg.mean()+dgp.mean())*.5))
    else:dpg=np.empty(0);dgp=np.full(n,np.inf);result.update(cd_pg_m=None,cd_gp_m=None,cd_m=None)
    for tau,k in [(.05,'5'),(.1,'10'),(.2,'20')]:
        precision=float((dpg<tau).mean()) if count else 0.;recall=float((dgp<tau).mean());f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        result.update({f'precision_{k}cm':precision,f'recall_{k}cm':recall,f'f1_{k}cm':f1,f'ray_acc_{k}cm':float((hit&(abs(rp-r)<tau)).sum()/n)})
    return result,p,dpg,dgp

def main():
    active,sel=selection();protocol=read(E/'protocol.json');ph=sha(E/'protocol.json');refs={r['id']:r for r in read(E/'reference_manifest.json')['rows']};check(E/'reference_manifest.json',protocol['reference_manifest_sha256'])
    for p,h in protocol['code'].items():check(p,h)
    for ref in refs.values():
        check(ref['queries'],ref['queries_sha256']);check(ref['scan_path'],ref['scan_sha256']);check(ref['label_path'],ref['label_sha256']);check(ref['camera_contract'],ref['camera_contract_sha256'])
        z=dict(np.load(ref['queries']));assert len(z['q'])==ref['query_count'];np.testing.assert_allclose(z['q'],z['o']+z['r'][:,None]*z['u'],atol=1e-10,rtol=0)
        raw=np.fromfile(ref['scan_path'],dtype=np.float32).reshape(-1,5)[:,:3];T=z['T_world_from_sensor'];actual=raw[z['raw_index']].astype(float)@T[:3,:3].T+T[:3,3];np.testing.assert_allclose(actual,z['q'],atol=1e-9,rtol=0)
        assert int(z['t_ref_us'])==ref['t_ref_us']==sel[ref['id']]['t_ref_us']
    checked_rows=[];errors=[]
    for path in sorted((E/'results').glob('*/*.json')):
        d=read(path)
        if d['status']!='completed':continue
        try:
            ref=refs[d['id']];assert d['protocol_sha256']==ph and d['identity_key']==ref['identity_key'] and d['t_ref_us']==ref['t_ref_us'];check(ref['queries'],d['queries_sha256']);check(d['certificate'],d['certificate_sha256']);check(d['returns'],d['returns_sha256'])
            c=read(d['certificate']);assert c['protocol_sha256']==ph and c['queries_sha256']==d['queries_sha256'];check(c['extent'],c['extent_sha256']);extent=read(c['extent']);assert extent['status']=='passed';precision=extent['precision'];assert precision['passed'] and precision['boundary_fraction_of_height']<=.01 and precision['fine_pixel_fraction_of_height']<=.005
            V=np.array(c['T_source_camera_from_native']);gauge=np.linalg.det(V[:3,:3])**(1/3);V[:3]/=gauge;s=c['scale_m_per_native_unit'];camera=read(ref['camera_contract']);A=np.asarray(camera['T_world_from_camera_opencv'])@np.diag([s,s,s,1])@V
            np.testing.assert_allclose(A,c['T_world_from_native'],atol=1e-10,rtol=0);equal(s*c['native_vertical_extent'],ref['annotation_vertical_extent_m']);equal(c['native_vertical_extent'],extent['measurements'][str(precision['fine_resolution'])]['vertical_extent'])
            ev=c['evidence']
            if ev.get('asset'):check(ev['asset'],ev['asset_sha256'])
            if ev.get('preparation'):
                prep=ev['preparation'];assert prep['identity_key']==d['identity_key'] and prep['t_ref_us']==d['t_ref_us'];check(prep['prepared'],prep['prepared_sha256'])
                for p,h in prep['provenance'].items():
                    if p.startswith('/') and isinstance(h,str) and len(h)==64:check(p,h)
            q=dict(np.load(ref['queries']));panel_checks=[]
            for condition,panel in d['panels'].items():
                target=d['returns'] if condition=='annotation_vertical_extent_conditioned' else panel['returns'];digest=d['returns_sha256'] if condition=='annotation_vertical_extent_conditioned' else panel['returns_sha256'];check(target,digest);z=dict(np.load(target));hit=z['hit'];assert hit.dtype==bool
                np.testing.assert_array_equal(hit,(z['alpha']>=.5)&(z['range_pred']>=.1)&(z['range_pred']<=200.))
                fresh,p,dpg,dgp=independent_metrics(q['q'],q['o'],q['u'],q['r'],z['range_pred'],hit)
                for k,v in panel['metrics'].items():equal(v,fresh[k])
                for k,v in [('predicted_endpoints',p),('distances_pg',dpg),('distances_gp',dgp)]:
                    if k in z:np.testing.assert_allclose(z[k],v,atol=1e-11,rtol=1e-10)
                panel_checks.append(dict(condition=condition,query_count=len(q['q']),hit_count=int(hit.sum()),status='passed'))
            checked_rows.append(dict(id=d['id'],method=d['method'],result_sha256=sha(path),panels=panel_checks,status='passed'))
        except Exception as error:errors.append(dict(id=d['id'],method=d['method'],error=repr(error)))
    report=dict(status='passed' if not errors else 'failed',all100_references_verified=True,scored_rows_verified=len(checked_rows),panel_results_verified=sum(len(r['panels']) for r in checked_rows),rows=checked_rows,errors=errors,hash_bound_files=len(hashes),script_sha256=sha(__file__),metric_recalculation='all rows: independent dense pairwise Euclidean distances, no geometry_core import',finished_at=time.time())
    save(E/'validation/geometry_independent_audit.json',report);save(E/'validation/verified_input_hashes.json',hashes);print({k:v for k,v in report.items() if k not in ['rows']},flush=True)
    assert not errors

if __name__=='__main__':main()
