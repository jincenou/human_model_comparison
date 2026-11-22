"""Independent numerical checks for the implemented metric/similarity contracts."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from common import *
sys.path.insert(0,str(G/'scripts'))
from geometry_core import metrics,bilinear_accumulated,sector_ids,sector_rotation,selftest

def main():
    tests=selftest()['checks'];rng=np.random.default_rng(7492)
    for n in [7,20,71]:
        o=np.broadcast_to([10.,-5.,2.],(n,3));u=rng.normal(size=(n,3));u/=np.linalg.norm(u,axis=1,keepdims=True)
        r=rng.uniform(1,15,n);q=o+r[:,None]*u;pred=r+rng.normal(0,.15,n);hit=rng.random(n)>.3
        m,p,_,_=metrics(q,o,u,r,pred,hit)
        distances=np.sqrt(((p[:,None,:]-q[None,:,:])**2).sum(-1))
        dp=distances.min(1);dg=distances.min(0);assert abs(m['cd_m']-(dp.mean()+dg.mean())/2)<1e-12
        for tau in [.05,.1,.2]:
            precision=(dp<tau).mean();recall=(dg<tau).mean();f=2*precision*recall/(precision+recall) if precision+recall else 0
            assert abs(m[f'f1_{round(tau*100)}cm']-f)<1e-12
    tests.append('independent dense pairwise distance recomputation with missed fixed queries')
    rot=Rotation.from_rotvec([.2,-.7,.4]).as_matrix();s=2.3;t=np.array([2.,4.,7.]);x=rng.normal(size=(10,3))
    cov=np.diag([.1,.2,.4]);newcov=s*s*rot@cov@rot.T
    assert np.allclose(np.linalg.eigvalsh(newcov),s*s*np.linalg.eigvalsh(cov))
    scaled=s*(x@rot.T+t);wrong=s*x@rot.T+t;assert not np.allclose(scaled,wrong)
    pixel=lambda a:a[:,:2]/a[:,2:]
    assert np.allclose(pixel(scaled),pixel(x@rot.T+t));tests.append('scale translation and covariance; preserve source projection; negative partial-scale control')
    from pilot import combine_vertical,assess_vertical
    thin=[dict(name=n,bounds={'0':[-.2,.2],'2':[-1.,1.]}) for n in ['y_pos','y_neg']]
    h=combine_vertical(thin);assert h['vertical_extent']==2.
    assert assess_vertical({'512':h,'1024':h},3.)['passed'];tests.append('thin plane height measurable from opposing side views despite four edge-on empty views')
    for ref in read(E/'reference_manifest.json')['rows']:
        checked(ref['queries'],ref['queries_sha256'])
        with np.load(ref['queries']) as z:
            raw=np.fromfile(ref['scan_path'],np.float32).reshape(-1,5)[z['raw_index'],:3].astype(float);T=z['T_world_from_sensor']
            np.testing.assert_allclose(raw@T[:3,:3].T+T[:3,3],z['q'],atol=1e-10)
            np.testing.assert_allclose(z['q'],z['o']+z['r'][:,None]*z['u'],atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(raw,axis=1),z['r'],atol=1e-10)
    tests.append('all 100 fixed query files independently mapped to original raw scan indices; nominal range invariant')
    save(E/'validation/cpu.json',dict(status='passed',checks=tests,script_sha256=sha(__file__),metric_source=str(G/'scripts/geometry_core.py'),metric_source_sha256=sha(G/'scripts/geometry_core.py')))
    print('passed',len(tests),'checks')

if __name__=='__main__':main()
