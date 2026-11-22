"""Fresh-process native weights vs exported arrays, independently recomputed covariance.

This runs outside every measured generation interval. The raw network replay and
PLY loader are distinct from the baked NPZ and snapshot readers.
"""
import argparse,json,os,sys,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
E=Path(__file__).resolve().parents[1];CODE=E.parent.parent/'human'
sys.path.insert(0,str(CODE))
import torch
from reconstruction import load_asset
from utils.general_utils import quaternion_to_matrix
from common import sha,save

def cov(q,s,raw=False):
    q=np.asarray(q,np.float64);s=np.asarray(s,np.float64)
    if raw:
        # Native rasterizer deliberately uses nonunit quaternion components.
        w,x,y,z=q.T
        R=np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                    2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                    2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)
    else:R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    return (R*s[:,None,:]**2)@R.transpose(0,2,1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();root=Path(a.run)
    record=json.loads((root/'timing.json').read_text());sid=record['id'];out=root/'native_assets'/sid
    endpoint=json.loads((out/'time_endpoint.json').read_text());assert record['status']=='success'
    assert record['output_sha256']==sha(out/'time_endpoint.json')
    for name,h in endpoint['files'].items():assert sha(out/name)==h,name
    for tag,count in [('native',32999),('alpha',47999),('asset',52999)]:
        d=json.loads((root/'outputs'/sid/tag/'complete.json').read_text())
        actual=d.get('actual_updates',d.get('actual_optimizer_updates'))
        if actual is None:actual=json.loads((root/'outputs'/sid/tag/'snapshots'/('33000.json' if tag=='native' else '48000.json' if tag=='alpha' else '53000.json')).read_text())['actual_optimizer_updates']
        assert actual==count,(tag,d)
    torch.set_num_threads(2)
    with np.load(out/'asset_4dgs.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    for k,v in arrays.items():
        if np.issubdtype(v.dtype,np.number):assert np.isfinite(v).all(),k
    assert arrays['shs'].shape[1:]==(16,3) and arrays['native_shs'].shape[1:]==(16,3)
    assert arrays['frame_xyz'].shape[0]==61 and int(arrays['active_sh_degree'][0])==3
    assert np.array_equal(arrays['frame_timestamps'],np.arange(61)/25)
    model,deform,pipe,config=load_asset(out/'runtime',53000)
    assert np.array_equal(model.get_xyz.detach().cpu().numpy(),arrays['native_canonical_xyz'])
    assert np.array_equal(model.get_features.detach().cpu().numpy(),arrays['native_shs'])
    assert np.array_equal(model.get_opacity.detach().cpu().numpy(),arrays['opacity'])
    T=arrays['asset_from_native'];replay=[]
    with torch.no_grad():
        for frame in [0,30,60]:
            n=len(model.get_xyz);t=model.get_xyz.new_tensor([[frame/60]]).expand(n,-1)
            warp=deform.step(model.get_xyz.detach(),t)
            xyz=((quaternion_to_matrix(warp['d_xyz'][:,:4])@model.get_xyz[...,None]).squeeze(-1)+warp['d_xyz'][:,4:]).cpu().numpy()
            q=(model.get_rotation+warp['d_rotation']).cpu().numpy();s=(model.get_scaling+warp['d_scaling']).cpu().numpy()
            diff=max(float(np.max(np.abs(x-y))) for x,y in [(xyz,arrays['native_frame_xyz'][frame]),(q,arrays['native_frame_rotation'][frame]),(s,arrays['native_frame_scaling'][frame])])
            assert diff<2e-5,diff
            expected=xyz.astype(np.float64)@T[:3,:3].T+T[:3,3]
            position_error=float(np.max(np.abs(expected-arrays['frame_xyz'][frame])))
            assert position_error<2e-5
            expected_cov=T[:3,:3][None]@cov(q,s,True)@T[:3,:3].T[None]
            got=cov(arrays['frame_rotation'][frame],arrays['frame_scaling'][frame])
            error=float(np.max(np.linalg.norm(expected_cov-got,axis=(1,2))/np.maximum(np.linalg.norm(expected_cov,axis=(1,2)),1e-20)))
            assert error<2e-5,error
            replay.append(dict(frame=frame,native_array_absolute_max=diff,export_position_absolute_max=position_error,covariance_relative_max=error))
    with np.load(out/'frozen_time_native_export_coordinates.npz',allow_pickle=False) as z:snap={k:z[k] for k in z.files}
    elapsed=endpoint['source_seconds'];times=arrays['frame_timestamps'];assert 0<=elapsed<=times[-1]
    hi=min(int(np.searchsorted(times,elapsed,side='right')),60);lo=max(0,hi-1);f=0 if hi==lo else (elapsed-times[lo])/(times[hi]-times[lo])
    expected=(1-f)*arrays['frame_xyz'][lo]+f*arrays['frame_xyz'][hi]
    assert np.allclose(expected,snap['means'],atol=1e-6,rtol=1e-6)
    expected_cov=(1-f)*cov(arrays['frame_rotation'][lo],arrays['frame_scaling'][lo])+f*cov(arrays['frame_rotation'][hi],arrays['frame_scaling'][hi])
    got=cov(snap['quaternions_wxyz'],snap['scales']);rel=float(np.max(np.linalg.norm(expected_cov-got,axis=(1,2))/np.maximum(np.linalg.norm(expected_cov,axis=(1,2)),1e-20)))
    assert rel<2e-5,rel
    assert np.array_equal(snap['sh'],arrays['shs']) and np.array_equal(snap['opacities'],arrays['opacity'].reshape(-1))
    assert np.allclose(snap['local_from_appearance'],np.eye(3))
    with np.load(out/'root_motion.npz',allow_pickle=False) as z:
        for k in z.files:
            if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all()
        root_fields={k:list(z[k].shape) for k in z.files}
    stages=record['stages'];assert len(stages)==17 and all(s['status']=='completed' and s['complete_interval_accounting'] for s in stages)
    assert abs(sum(s['seconds'] for s in stages)-record['seconds'])<1e-8
    proof=dict(status='passed',id=sid,repeat=record['repeat'],endpoint_sha256=sha(out/'time_endpoint.json'),timing_record_sha256=sha(root/'timing.json'),script_sha256=sha(__file__),native_fresh_process_replay=replay,snapshot_covariance_relative_max=rel,snapshot_fraction=float(f),snapshot_frames=[lo,hi],root_fields=root_fields,all_endpoint_files_hash_verified=True,complete_SH3=True,updates=[32999,15000,5000],stage_intervals_verified=17,finished_at=time.time())
    save(root/'independent_reload.json',proof);print(proof,flush=True)

if __name__=='__main__':main()
