"""Fully decode each unexcluded candidate NPZ; this does not certify rendering."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/nuscenes_cohort_audit_v2/audit.json'
OUT = ROOT / 'reports/nuscenes_cohort_audit_v2/runtime_decode.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    rows = [r for r in json.loads(AUDIT.read_text())['records']
            if r['cohort']=='candidate44' and not r['blocked_by_prior_exclusion']]
    report = {'status':'running','audit_sha256':sha(AUDIT),'started_at':time.time(),
        'scope':'Full NPZ decompression, CRC, numeric integrity and time support; no render validation.',
        'records':[]}
    for row in rows:
        rec = {'id':row['id'],'path':row['paths']['runtime'],'render_reload_valid':False}
        try:
            path = Path(rec['path'])
            before = path.stat()
            with np.load(path,allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            rec['sha256'] = sha(path)
            after = path.stat()
            assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            assert all(np.issubdtype(a.dtype,np.number) and np.isfinite(a).all() for a in arrays.values())
            xyz = arrays['frame_xyz']
            f,n,_ = xyz.shape
            assert xyz.shape==(f,n,3) and f>1 and n>0
            for k,s in [('frame_quaternions',(f,n,4)),('frame_scales',(f,n,3)),
                        ('timestamps',(f,)),('root_xy',(f,2)),('root_yaw',(f,)),('asset_to_standard',(4,4))]:
                assert arrays[k].shape==s,(k,arrays[k].shape,s)
            assert arrays['opacities'].shape in [(n,),(n,1)]
            assert np.all((arrays['opacities']>=0)&(arrays['opacities']<=1))
            assert (arrays['frame_scales']>0).all()
            normerror = float(np.max(np.abs(np.linalg.norm(arrays['frame_quaternions'],axis=-1)-1)))
            assert normerror<1e-4,('quaternion_norm_error',normerror)
            assert arrays['sh'].shape==(n,16,3),'Full SH degree3 expected for this batch'
            timestamps=arrays['timestamps']
            assert (np.diff(timestamps)>0).all()
            assert np.allclose(arrays['asset_to_standard'][3],[0,0,0,1])
            assert abs(np.linalg.det(arrays['asset_to_standard'][:3,:3]))>1e-10
            query=row['reference_seconds_from_first_native']
            assert timestamps[0]<=query<=timestamps[-1]
            right=int(np.searchsorted(timestamps,query,side='left'))
            right=min(right,f-1)
            left=max(right-1,0)
            if timestamps[right]==query:left=right
            fraction=0. if left==right else float((query-timestamps[left])/(timestamps[right]-timestamps[left]))
            rec.update(status='decoded_numeric_valid',frames=f,gaussians=n,sh_degree=3,
                quaternion_norm_max_error=normerror,time_start_s=float(timestamps[0]),time_end_s=float(timestamps[-1]),
                reference_seconds=query,bracket=[left,right],interpolation_fraction=fraction,
                nonlooping_time_supported=True,
                root_motion_restoration='arrays available; native/world transform validation pending')
            del arrays, xyz
        except Exception as exc:
            rec.update(status='failed',reason=repr(exc))
        report['records'].append(rec)
        OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(row['id'],rec['status'],rec.get('reason',''),flush=True)
    report.update(status='finished',finished_at=time.time())
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
