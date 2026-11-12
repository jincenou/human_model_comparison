"""Orbit a frozen native scene actor, preserving deformation, SH and RGB decoder."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common_human_orbit import H,OUT,read,save,sha,checked,inputs,native_views,output_frame

R=H.parent/'vehicle_model_comparison'
sys.path.insert(0,str(R/'scripts'));sys.path.insert(0,str(R/'adapters'))


def scene_context(sid,method,cert,timestamp):
    """Reuse the certified source actor, time and appearance from any cohort."""
    checked(cert['render_validation'],cert['render_validation_sha256'])
    prior=read(cert['render_validation']);prior_path=Path(cert['render_validation'])
    if method=='omnire' and 'actor_index'not in prior:
        # HN0004 source certification uses its dynamic anchor. Actor identity
        # comes from the same checkpoint's earlier canonical extraction.
        prior_path=H/'exports/common_camera_human_time_v2/omnire'/sid/'validation.json'
        prior=read(prior_path)
    candidates=[]
    for group in ['scene_extract_v2','scene_extract_original75_v1','scene_extract_replacement_v3']:
        path=H/'inputs'/group/(method+'_plan.json')
        if not path.exists():continue
        for item in read(path)['objects']:
            if item['id']==sid and item.get('status')=='ready_for_native_decode':
                if item['checkpoint']==prior['checkpoint']:candidates.append((item,path))
    if not candidates:raise ValueError('No certified native scene input for '+sid+' '+method)
    item,plan=candidates[0]
    assert all(x['native_id']==item['native_id'] and x['config']==item['config']for x,_ in candidates)
    assert item['native_id']==cert['identity_key'] and item['raw_timestamp_us']==timestamp
    checked(item['source_image'],item['source_sha256'])
    clip_path=H/'reports/dynamic_source_clip_v1'/sid/'clip.json'
    if clip_path.exists():
        clip=read(clip_path);assert clip['identity_key']==item['native_id']
        frame=next(f for f in clip['frames']if f['timestamp_us']==timestamp)
        mapping=clip_path
    else:
        frame=dict(timestamp_us=timestamp);mapping=prior_path
        if method=='omnire':
            temporal=prior['temporal_query'];assert temporal['query_timestamp_us']==timestamp
            frame.update(source_frame=prior['source_frame'],omnire=dict(lo=temporal['lo_frame'],hi=temporal['hi_frame'],fraction=temporal['fraction'],normalized_query=temporal['normalized_query']))
    return item,prior,frame,dict(path=str(mapping),sha256=sha(mapping),plan=str(plan),plan_sha256=sha(plan),source_validation=str(prior_path),source_validation_sha256=sha(prior_path))


def splatad(sid,frame,camera,protocol,item,prior):
    from train_splatad_reviewed_scene import MonitoredTrainer
    from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
    globals()['MonitoredTrainer']=MonitoredTrainer;globals()['TrainOnlyMonitoredTrainer']=TrainOnlyMonitoredTrainer
    from nerfstudio.utils.eval_utils import eval_setup
    checked(item['config'],item['config_sha256']);checked(item['bundle'],item['bundle_sha256'])
    config,pipeline,checkpoint,step=eval_setup(Path(item['config']),test_mode='val');assert step==30000
    assert str(checkpoint)==item['checkpoint']
    pipeline.eval();model=pipeline.model;manager=pipeline.datamanager;parsed=manager.train_dataparser_outputs
    anchor=[Path(p).resolve()for p in manager.train_dataset.image_filenames].index(Path(item['source_image']).resolve())
    c=manager.train_dataset.cameras[anchor:anchor+1].to(model.device);assert float(parsed.dataparser_scale)==1
    M=np.eye(4);M[:3]=parsed.dataparser_transform.cpu().numpy()[:3];cv_gl=np.diag([1.,-1.,-1.,1.])
    actor=next(i for i,r in enumerate(parsed.metadata['trajectories'])if r['uuid']==item['native_id'].split(':')[-1])
    mask=model.id.detach().reshape(-1)==actor;bundle=torch.load(item['bundle'],map_location='cpu',weights_only=False)
    for name,param in model.gauss_params.items():assert torch.equal(param.detach()[mask].cpu(),bundle['gauss_params'][name])
    model.gauss_params=torch.nn.ParameterDict({n:torch.nn.Parameter(p.detach()[mask].clone(),requires_grad=False)for n,p in model.gauss_params.items()})
    model.config.background_color='white';model.config.use_camopt_in_eval=False;model.config.compensate_rs_camera=False
    for k in ('lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','rs_time'):model.rs_editing[k]=0.
    K=np.array(camera['K_raw']);c.width.fill_(camera['render_wh'][0]);c.height.fill_(camera['render_wh'][1]);c.times.fill_(frame['timestamp_us']/1e6-float(parsed.time_offset))
    for name,value in [('fx',K[0,0]),('fy',K[1,1]),('cx',K[0,2]),('cy',K[1,2])]:getattr(c,name).fill_(float(value))
    points,_=model._get_actor_adjusted_means(model.means,c.times,model.id,calc_vels=False)
    source_from_model=np.linalg.inv(M@np.array(camera['T_world_from_camera_opencv']))
    views,gauge=native_views(points.cpu().numpy(),source_from_model,protocol['frames'])
    def render(view):
        cam=copy.deepcopy(c);pose=np.linalg.inv(view)@cv_gl
        cam.camera_to_worlds[0]=torch.tensor(pose[:3],device=model.device,dtype=cam.camera_to_worlds.dtype)
        result=model.get_outputs_for_camera(camera=cam)
        return result['rgb'],result['accumulation']
    return views,gauge,render,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),bundle=item['bundle'],bundle_sha256=item['bundle_sha256'],native_decoder_preserved=True,geometry_and_pose='Native actor held at one physical exposure; no trajectory or shape edits',appearance='Original camera sensor embedding and native decoder')


def omnire(sid,frame,camera,protocol,item,prior):
    repo=Path(_hmc_expand('${DRIVESTUDIO_ROOT}'));sys.path.insert(0,str(repo))
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str
    from models.gaussians.basics import dataclass_gs
    from omnire_time_query_v1 import at_physical_exposure
    checked(item['config'],item['config_sha256']);cfg=OmegaConf.load(item['config']);data=Path(item['processed_data'])
    checked(item['checkpoint'],prior['checkpoint_sha256'])
    dataset=DrivingDataset(data_cfg=cfg.data)
    trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,model_config=cfg.model,num_train_images=len(dataset.train_image_set),num_full_images=len(dataset.full_image_set),test_set_indices=dataset.test_timesteps,scene_aabb=dataset.get_aabb().reshape(2,3),device=torch.device('cuda'))
    trainer.resume_from_checkpoint(ckpt_path=item['checkpoint'],load_only_model=True);trainer.set_eval();model=trainer.models['DeformableNodes']
    instance=prior['actor_index'];mask=model.point_ids.reshape(-1)==instance
    checked(prior['prior_snapshot'],prior['prior_snapshot_sha256'])
    native=torch.load(prior['prior_snapshot'],weights_only=False,map_location='cpu')
    assert native['uuid']==item['native_id'].split(':')[-1] and native['checkpoint']==item['checkpoint']
    for k,v in native['canonical_parameters'].items():assert torch.equal(model.state_dict()[k][mask].detach().cpu(),v)
    camera_index=int(Path(prior['processed_source_image']).stem.split('_')[1])
    cam_unique=dataset.pixel_source.camera_data[camera_index].unique_cam_idx
    M=np.linalg.inv(np.loadtxt(data/'extrinsics'/f'{cfg.data.start_timestep:03d}_0.txt'))
    bracket=frame['omnire'];anchor=frame.get('source_frame',bracket['lo']if bracket['fraction']<.5 else bracket['hi'])
    info,caminfo=dataset.pixel_source.get_image(anchor*dataset.pixel_source.num_cams+cam_unique)
    info={k:v.cuda()if isinstance(v,torch.Tensor)else v for k,v in info.items()};caminfo={k:v.cuda()if isinstance(v,torch.Tensor)else v for k,v in caminfo.items()}
    width,height=camera['render_wh'];info={k:info[k].flatten()[0].expand(height,width)for k in ('img_idx','normed_time','frame_idx')}
    yy,xx=torch.meshgrid(torch.arange(height,device='cuda'),torch.arange(width,device='cuda'),indexing='ij');info['pixel_coords']=torch.stack([yy/height,xx/width],dim=-1)
    trainer.cur_frame=torch.tensor(anchor,device='cuda');assert not trainer.in_test_set
    for node in trainer.models.values():
        if hasattr(node,'in_test_set'):node.in_test_set=False
        if hasattr(node,'set_cur_frame'):node.set_cur_frame(trainer.cur_frame)
    temporal=dict(status='mapped',source_frame=anchor,timestamp_us=frame['timestamp_us'],grid_timestamp_us=None,grid_minus_exposure_us=None,continuous_bracket=bracket)
    def native_camera(view):
        ci=dict(caminfo,camera_to_world=torch.tensor(np.linalg.inv(view),dtype=torch.float32,device='cuda'),intrinsics=torch.tensor(camera['K_raw'],dtype=torch.float32,device='cuda'),width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
        return trainer.process_camera(camera_infos=ci,image_ids=info['img_idx'].flatten()[0],novel_view=True)
    source_from_model=np.linalg.inv(M@np.array(camera['T_world_from_camera_opencv']));nativecam=native_camera(source_from_model)
    with at_physical_exposure(model,instance,temporal):initial=model.get_gaussians(nativecam)
    points=initial['_means'][mask].cpu().numpy();views,gauge=native_views(points,source_from_model,protocol['frames'])
    def render(view):
        cam=native_camera(view)
        with at_physical_exposure(model,instance,temporal):gaussians=model.get_gaussians(cam)
        actor=dataclass_gs(**{k:v[mask]for k,v in gaussians.items()},detach_keys=[],extras=None)
        result,_=trainer.render_gaussians(gs=actor,cam=cam,near_plane=trainer.render_cfg.near_plane,far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
        alpha=result['opacity'];raw=result['rgb_gaussians'];bias=trainer.affine_transformation(torch.zeros_like(raw),info)
        rgb=(trainer.affine_transformation(raw,info)-bias*(1-alpha)+(1-alpha)).clamp(0,1)
        return rgb,alpha
    return views,gauge,render,dict(checkpoint=item['checkpoint'],checkpoint_sha256=sha(item['checkpoint']),native_appearance_frame=anchor,full_native_directional_appearance_preserved=True,geometry_and_pose='Native continuous non-SMPL deformation and root held at the fixed physical exposure',appearance='Same-sensor nearest learned affine; no exposure fitting')


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);p.add_argument('--method',choices=['omnire','splatad'],required=True);args=p.parse_args();sid,method=args.human_id,args.method
    torch.set_num_threads(4);active,selected,reference,camera,cert,certificate=inputs(sid,method)
    root=OUT/sid;protocol=read(root/'protocol.json');assert protocol['reference_camera_sha256']==reference['camera_contract_sha256'] and protocol['selection_sha256']==active['selection_sha256']
    item,prior,frame,mapping=scene_context(sid,method,cert,selected['t_ref_us'])
    folder=root/method;folder.mkdir(exist_ok=False)
    with torch.inference_mode():
        views,gauge,render,provenance=globals()[method](sid,frame,camera,protocol,item,prior);rows=[]
        for i,view in enumerate(views):
            rgb,alpha=render(view);row=output_frame(folder,i,rgb,alpha,protocol);row['native_world_to_camera']=view.tolist();rows.append(row);print(method,sid,'orbit',i,flush=True)
    diagnostics={}
    for name in ['rgb','alpha']:
        checked(cert['source_'+name],cert['source_'+name+'_sha256'])
        a=np.asarray(Image.open(rows[0][name])).astype(np.float64);b=np.asarray(Image.open(cert['source_'+name])).astype(np.float64);d=np.abs(a-b)
        diagnostics[name]=dict(max_pixel_difference=float(d.max()),mean_pixel_difference=float(d.mean()),used_as_quality_gate=False)
    archive=root/'source_archive';archive.mkdir(exist_ok=True);(archive/(sha(__file__)+'.py')).write_bytes(Path(__file__).read_bytes())
    save(folder/'manifest.json',dict(status='rendered_pending_joint_visual_review',id=sid,method=method,timestamp_us=selected['t_ref_us'],protocol_sha256=sha(root/'protocol.json'),registration=certificate,frames=rows,gauge=gauge,provenance=provenance,source_view_diagnostics=diagnostics,physical_time_mapping=mapping['path'],physical_time_mapping_sha256=mapping['sha256'],scene_input=mapping,metric_scores_produced=False,source_metric_inputs_modified=False,script_sha256=sha(__file__)))


if __name__=='__main__':main()
