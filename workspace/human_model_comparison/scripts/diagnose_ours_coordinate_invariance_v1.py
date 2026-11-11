"""Separate analytic frame-transform errors from float32 raster discontinuities."""
from pathlib import Path
import json,sys,os,fcntl,hashlib
H=Path(__file__).resolve().parents[1];P=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
sys.path.insert(0,str(P/'scripts'))
import dispatch_rigid111_direct as resources
def main():
 lease=None;gpu=None
 for index in ['4','3','2','1','0','5']:
  ready,detail=resources.available(index)
  if not ready:continue
  handle=(P/f'runs/rigid111_gpu_locks/gpu_{index}.lock').open('a')
  print('Waiting for shared lease on idle GPU',index,flush=True)
  fcntl.flock(handle,fcntl.LOCK_EX)
  ready,detail=resources.available(index)
  if ready:lease=handle;gpu=index;os.environ['CUDA_VISIBLE_DEVICES']=detail['uuid'];break
  handle.close()
 assert lease is not None,'No shared GPU lease available; retry only after inspecting resource state'
 import numpy as np
 import torch
 from scipy.spatial.transform import Rotation
 from render_ours_frozen_time_v2 import render,GaussianAssetFrame
 visible=torch.zeros(1,device='cuda')
 out=H/'reports/ours_coordinate_diagnostic_v1';out.mkdir(exist_ok=False)
 def covariance(cloud):
  q=cloud.quaternions.astype('float64');q/=np.linalg.norm(q,axis=-1,keepdims=True)
  axes=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
  return (axes*cloud.scales[:,None,:].astype('float64')**2)@axes.transpose(0,2,1)
 records=[]
 for sid in ['HN0004','HN0145','HN0161']:
  path=H/'exports/ours_frozen_time_v1'/sid/'inplace.npz'
  with np.load(path,allow_pickle=False)as z:cloud=GaussianAssetFrame(z['means'],z['quaternions_wxyz'],z['scales'],z['opacities'],z['sh'],z['local_from_appearance'])
  eye=np.array([3.2,0,1.25]);target=np.array([0,0,.95]);f=target-eye;f/=np.linalg.norm(f);r=np.cross(f,[0,0,1]);r/=np.linalg.norm(r);d=np.cross(f,r)
  cam=np.eye(4);cam[:3,:3]=np.stack([r,d,f],axis=1);cam[:3,3]=eye
  transform=np.eye(4);angle=.73;transform[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
  moved=cloud.transformed(transform);mcam=transform@cam;k=np.array([[600,0,256],[0,600,256],[0,0,1.]])
  xyz=(cloud.means.astype('float64')-cam[:3,3])@cam[:3,:3];xyz2=(moved.means.astype('float64')-mcam[:3,3])@mcam[:3,:3]
  cov=cam[:3,:3].T@covariance(cloud)@cam[:3,:3];cov2=mcam[:3,:3].T@covariance(moved)@mcam[:3,:3]
  dirs=cloud.means.astype('float64')-eye;dirs/=np.linalg.norm(dirs,axis=-1,keepdims=True)
  dirs2=moved.means.astype('float64')-mcam[:3,3];dirs2/=np.linalg.norm(dirs2,axis=-1,keepdims=True)
  with torch.inference_mode():
   rgb,alpha=render(cloud,cam,k);repeat,ra=render(cloud,cam,k);other,oa=render(moved,mcam,k)
  def stats(a,b):
   diff=np.abs(a-b)
   return dict(max=float(diff.max()),mean=float(diff.mean()),rms=float(np.sqrt(np.mean(diff**2))),count_gt_002=int((diff>.002).sum()),count_gt_one_byte=int((diff>1/255+1e-6).sum()),elements=int(diff.size))
  record=dict(id=sid,gpu=gpu,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
   camera_space_position_max_error=float(np.abs(xyz-xyz2).max()),
   covariance_relative_frobenius_max=float((np.linalg.norm(cov-cov2,axis=(1,2))/np.maximum(np.linalg.norm(cov,axis=(1,2)),1e-20)).max()),
   appearance_direction_max_error=float(np.abs(dirs@cloud.local_from_appearance-dirs2@moved.local_from_appearance).max()),
   sh_exact_equal=bool(np.array_equal(cloud.sh,moved.sh)),opacity_exact_equal=bool(np.array_equal(cloud.opacities,moved.opacities)),
   repeat_rgb=stats(rgb,repeat),repeat_alpha=stats(alpha,ra),rotated_rgb=stats(rgb,other),rotated_alpha=stats(alpha,oa),
   diagnostic_only=True,old_acceptance_threshold_unchanged=True)
  (out/f'{sid}.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record);print(json.dumps(record),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(records=records,renderer_alpha_cutoff='Installed gsplat ALPHA_THRESHOLD = 1/255; source Common.h and RasterizeToPixels3DGSFwd.cu. This is a candidate numerical discontinuity explanation, not proof of per-pixel cause.',no_failed_result_promoted=True),indent=2)+'\n')
 del visible;lease.close()
if __name__=='__main__':main()
