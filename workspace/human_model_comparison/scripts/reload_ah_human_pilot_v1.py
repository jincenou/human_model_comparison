"""Validate exported native PLY in a fresh process; not a scored evaluator."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
from PIL import Image
R=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'));B=Path(_hmc_expand('${HMC_WORKSPACE}/assetbench'));repo=B/'repos/ah';job=R/'runs/ah'/sys.argv[1];record=json.loads((job/'job.json').read_text());assert record['status']=='inference_completed_pending_reload'
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'benchmark'))
from utils import load_ply,render_from_cameras
ply=Path(record['output_ply']);out=job/'reload_validation';out.mkdir(exist_ok=False);cam=json.loads((ply.parent/'ahc_source_prediction.json').read_text());gaussians=load_ply(str(ply)).cuda()
assert gaussians.shape[-1]==14 and gaussians.shape[1]>0 and torch.isfinite(gaussians).all()
assert (gaussians[...,4:7]>0).all();quat_norm=gaussians[...,7:11].norm(dim=-1);assert (quat_norm>0.9).all() and (quat_norm<1.1).all()
direction=np.array(cam['cam_directions'][0]);assert abs(np.linalg.norm(direction)-1)<1e-3
angle=np.deg2rad(17);rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
# 17 degrees is absent from the 16 construction-view azimuths and 80 preview-view azimuths.
directions=np.stack([direction,rot@direction]);dist=float(np.array(cam['dists']).flat[0]);fov=float(np.array(cam['fov']).flat[0]);lwh=np.array(cam['lwh']).reshape(-1)
with torch.inference_mode():rgb,alpha=render_from_cameras(gaussians,directions,[dist,dist],[fov,fov],lwh,output_size=512,bg_color=(0.5,0.5,0.5))
assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
for i,name in enumerate(['ahc_source','unseen_virtual_17deg']):
 assert int((alpha[i]>0.5).sum())>100,'Empty foreground render'
 Image.fromarray((rgb[i].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(out/(name+'.png'))
 Image.fromarray((alpha[i].squeeze().clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(out/(name+'_alpha.png'))
result={'status':'passed','gaussian_count':gaussians.shape[1],'ply_sha256':hashlib.sha256(ply.read_bytes()).hexdigest(),'camera_source':'AHC predictions only','directions':directions.tolist(),'dist':dist,'fov_degrees':fov,'lwh_prediction':lwh.tolist(),'renderer':'official AH benchmark.utils.render_from_cameras without alignment or metric calls','views':['ahc_source','unseen_virtual_17deg'],'real_holdout_projection':'not_validated','scores_produced':False}
(out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False));record['status']='human_pilot_reloaded_pending_identity_pose_common_camera';record['reload_validation']=str(out/'validation.json');(job/'job.json').write_text(json.dumps(record,indent=2,allow_nan=False));print(json.dumps(result))
