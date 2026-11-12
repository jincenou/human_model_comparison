"""Render a prepared real-exposure clip from existing native scene checkpoints."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prepare_dynamic_source_clip import H,read,save,sha,selected
from export_dynamic_source_assets import plan
R=H.parent/'vehicle_model_comparison'
sys.path.insert(0,str(R/'scripts'));sys.path.insert(0,str(R/'adapters'))

def output_frame(folder,f,rgb,alpha):
 row=dict(index=f['index'],timestamp_us=f['timestamp_us'])
 for name,x in [('rgb',rgb),('alpha',alpha)]:
  x=x.detach().cpu().numpy()
  if name=='alpha' and x.ndim==3:x=x[...,0]
  assert np.isfinite(x).all();p=folder/f"{f['index']:05d}_{name}.png";Image.fromarray(np.floor(np.clip(x,0,1)*255+.5).astype(np.uint8)).save(p);row[name]=str(p);row[name+'_sha256']=sha(p)
 return row

@torch.no_grad()
def splatad(sid,clip,folder):
 from train_splatad_reviewed_scene import MonitoredTrainer
 from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
 # YAML stores these names in __main__.
 globals()['MonitoredTrainer']=MonitoredTrainer;globals()['TrainOnlyMonitoredTrainer']=TrainOnlyMonitoredTrainer
 from nerfstudio.utils.eval_utils import eval_setup
 _,item=plan('splatad',sid)
 assert sha(item['config'])==item['config_sha256'] and sha(item['bundle'])==item['bundle_sha256']
 config,pipeline,checkpoint,step=eval_setup(Path(item['config']),test_mode='val');assert step==30000
 model=pipeline.model;manager=pipeline.datamanager;pipeline.eval();parsed=manager.train_dataparser_outputs
 filenames=[Path(p).resolve() for p in manager.train_dataset.image_filenames];anchor=filenames.index(Path(item['source_image']).resolve())
 camera=manager.train_dataset.cameras[anchor:anchor+1].to(model.device);assert float(parsed.dataparser_scale)==1
 M=np.eye(4);M[:3]=parsed.dataparser_transform.detach().cpu().numpy()[:3];cv_gl=np.diag([1.,-1.,-1.,1.])
 actor=next(i for i,r in enumerate(parsed.metadata['trajectories']) if r['uuid']==item['native_id'].split(':')[-1]);mask=model.id.detach().reshape(-1)==actor
 bundle=torch.load(item['bundle'],map_location='cpu',weights_only=False)
 for name,param in model.gauss_params.items():assert torch.equal(param.detach()[mask].cpu(),bundle['gauss_params'][name])
 model.gauss_params=torch.nn.ParameterDict({n:torch.nn.Parameter(p.detach()[mask].clone(),requires_grad=False) for n,p in model.gauss_params.items()})
 model.config.background_color='white';model.config.use_camopt_in_eval=False;model.config.compensate_rs_camera=False
 for k in ('lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','rs_time'):model.rs_editing[k]=0.
 rows=[]
 for f in clip['frames']:
  c=copy.deepcopy(camera);T=M@np.array(f['T_world_from_camera_opencv'])@cv_gl;K=np.array(f['K_raw'])
  c.camera_to_worlds[0]=torch.tensor(T[:3],dtype=c.camera_to_worlds.dtype,device=model.device);c.width.fill_(f['render_wh'][0]);c.height.fill_(f['render_wh'][1]);c.times.fill_(f['timestamp_us']/1e6-float(parsed.time_offset))
  for name,value in [('fx',K[0,0]),('fy',K[1,1]),('cx',K[0,2]),('cy',K[1,2])]:getattr(c,name).fill_(float(value))
  result=model.get_outputs_for_camera(camera=c);rows.append(output_frame(folder,f,result['rgb'],result['accumulation']));print('splatad',f['index'],flush=True)
 return rows,dict(native_decoder_preserved=True,sensor_appearance='same original camera sensor embedding for all real exposures',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),root_motion='native learned SE3 queried at physical exposure; internal shape rigid')

@torch.no_grad()
def omnire(sid,clip,folder):
 repo=Path(_hmc_expand('${DRIVESTUDIO_ROOT}'));sys.path.insert(0,str(repo))
 from omegaconf import OmegaConf
 from datasets.driving_dataset import DrivingDataset
 from utils.misc import import_str
 from models.gaussians.basics import dataclass_gs
 from omnire_time_query_v1 import at_physical_exposure
 _,item=plan('omnire',sid);assert sha(item['config'])==item['config_sha256']
 cfg=OmegaConf.load(item['config']);data=Path(item['processed_data']);dataset=DrivingDataset(data_cfg=cfg.data)
 trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,model_config=cfg.model,num_train_images=len(dataset.train_image_set),num_full_images=len(dataset.full_image_set),test_set_indices=dataset.test_timesteps,scene_aabb=dataset.get_aabb().reshape(2,3),device=torch.device('cuda'))
 trainer.resume_from_checkpoint(ckpt_path=item['checkpoint'],load_only_model=True);trainer.set_eval();model=trainer.models['DeformableNodes']
 prior_path=H/'exports/common_camera_human_time_v2/omnire'/sid/'validation.json'
 if not prior_path.exists():prior_path=H/'exports/common_camera_human_original75_time_v1/omnire'/sid/'validation.json'
 prior=read(prior_path);instance=prior['actor_index'];mask=model.point_ids.reshape(-1)==instance
 native=torch.load(H/'exports/dynamic_human_feasibility_v1'/f'omnire_{sid}_native_dynamic.pt',weights_only=True,map_location='cpu')
 for k,v in native['canonical'].items():assert torch.equal(model.state_dict()[k][mask].detach().cpu(),v)
 time_audit=next(r for r in read(H/'reports/model_time_audit_v1/omnire.json')['records'] if r['id']==sid);cam_id=time_audit['camera_index'];cam_unique=dataset.pixel_source.camera_data[cam_id].unique_cam_idx
 M=np.linalg.inv(np.loadtxt(data/'extrinsics'/f'{cfg.data.start_timestep:03d}_0.txt'))
 rows=[]
 for f in clip['frames']:
  bracket=f['omnire'];anchor=bracket['lo'] if bracket['fraction']<.5 else bracket['hi'];idx=anchor*dataset.pixel_source.num_cams+cam_unique
  info,camera_info=dataset.pixel_source.get_image(idx)
  info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in info.items()};camera_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in camera_info.items()}
  T=M@np.array(f['T_world_from_camera_opencv']);camera_info.update(camera_to_world=torch.tensor(T,dtype=torch.float32,device='cuda'),intrinsics=torch.tensor(f['K_raw'],dtype=torch.float32,device='cuda'),width=torch.tensor(f['render_wh'][0],device='cuda'),height=torch.tensor(f['render_wh'][1],device='cuda'))
  width,height=f['render_wh']
  info={k:info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
  yy,xx=torch.meshgrid(torch.arange(height,device='cuda'),torch.arange(width,device='cuda'),indexing='ij')
  info['pixel_coords']=torch.stack([yy/height,xx/width],dim=-1)
  trainer.cur_frame=torch.tensor(anchor,device='cuda');assert not trainer.in_test_set
  for node in trainer.models.values():
   if hasattr(node,'in_test_set'):node.in_test_set=False
   if hasattr(node,'set_cur_frame'):node.set_cur_frame(trainer.cur_frame)
  camera=trainer.process_camera(camera_infos=camera_info,image_ids=info['img_idx'].flatten()[0],novel_view=True)
  temporal=dict(status='mapped',source_frame=anchor,timestamp_us=f['timestamp_us'],grid_timestamp_us=None,grid_minus_exposure_us=None,continuous_bracket=bracket)
  with at_physical_exposure(model,instance,temporal):gaussians=model.get_gaussians(camera)
  actor=dataclass_gs(**{k:v[mask] for k,v in gaussians.items()},detach_keys=[],extras=None)
  render,_=trainer.render_gaussians(gs=actor,cam=camera,near_plane=trainer.render_cfg.near_plane,far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
  alpha=render['opacity'];raw=render['rgb_gaussians'];bias=trainer.affine_transformation(torch.zeros_like(raw),info)
  rgb=(trainer.affine_transformation(raw,info)-bias*(1-alpha)+(1-alpha)).clamp(0,1)
  row=output_frame(folder,f,rgb,alpha);row['native_appearance_frame']=anchor;rows.append(row);print('omnire',f['index'],flush=True)
 return rows,dict(checkpoint=item['checkpoint'],checkpoint_sha256=sha(item['checkpoint']),canonical_parameters_exact=True,
  appearance='Native same-sensor affine at nearest learned temporal grid, no new exposure fitting/interpolation',root_motion='Native continuous deformation plus learned-root interpolation at physical exposure')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method',choices=['omnire','splatad']);p.add_argument('--human-id',default='HN0004');a=p.parse_args();torch.set_num_threads(4)
 active,row=selected(a.human_id);out=H/'reports/dynamic_source_clip_v1'/a.human_id;clip=read(out/'clip.json');assert clip['selection_sha256']==active['selection_sha256'];folder=out/a.method;folder.mkdir(exist_ok=False)
 frames,provenance=globals()[a.method](a.human_id,clip,folder)
 save(folder/'validation.json',dict(status='real_exposure_dynamic_source_rendered_pending_comparison',id=a.human_id,method=a.method,frames=frames,provenance=provenance,
  clip_sha256=sha(out/'clip.json'),script_sha256=sha(__file__),prediction_encoding='white_composited_uint8',render_wh=[1600,900],temporal_metric_eligible=False))
