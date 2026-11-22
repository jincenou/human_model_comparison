"""Original75 physical-time revalidation, same native renderer/time contract as new25 v2."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
sys.path.insert(0,str(ROOT/'scripts'))
REPO = Path(_hmc_expand('${DRIVESTUDIO_ROOT}'))
sys.path.insert(0,str(REPO))
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.gaussians.basics import dataclass_gs
from datasets.base.pixel_source import get_rays
sys.path.insert(0,str(ROOT.parent/'vehicle_model_comparison/scripts'))
import common_camera_contract
from common_camera_contract import load_contract,projection_audit
from omnire_time_query_v1 import at_physical_exposure

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def image(t,p):
    a=t.detach().cpu().numpy()
    if a.ndim==3 and a.shape[-1]==1:a=a[...,0]
    Image.fromarray((np.clip(a,0,1)*255).round().astype('uint8')).save(p)

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--human-id',required=True);a=parser.parse_args()
    logging.basicConfig(level=logging.INFO);torch.set_num_threads(4)
    plan_path=ROOT/'inputs/scene_extract_original75_v1/omnire_plan.json'
    plan=json.loads(plan_path.read_text())
    assert sha(plan['selection'])==plan['selection_sha256']
    item=next(x for x in plan['objects'] if x['id']==a.human_id)
    assert item['status']=='ready_for_native_decode' and item['dataset']=='nuscenes'
    a.sample_id=a.human_id;scene=item['scene'];uuid=item['native_id'].split(':')[-1]
    source=Path(item['source_image']).resolve();assert sha(source)==item['source_sha256']
    checkpoint=Path(item['checkpoint']);assert checkpoint.stat().st_size==item['checkpoint_bytes']
    checkpoint_sha=sha(checkpoint)
    assert sha(item['config'])==item['config_sha256']
    assert sha(item['instances_info'])==item['instances_info_sha256']
    cfg=OmegaConf.load(item['config']);data=Path(item['processed_data'])
    assert cfg.data.pixel_source.load_smpl is False
    node_type='DeformableNodes';a.method='omnire'
    common_camera_contract.CONTRACT=ROOT/'configs/human_source_common_v1/candidates'/a.human_id/'camera.json'
    contract=json.loads(common_camera_contract.CONTRACT.read_text());assert contract['selection_sha256']==plan['selection_sha256']
    assert contract['timestamp_us']==item['raw_timestamp_us']
    prior_path=Path(item['prior_snapshot'])
    assert sha(prior_path)==item['prior_snapshot_sha256']
    prior=torch.load(prior_path,map_location='cpu',weights_only=False)
    assert prior['uuid']==uuid and prior['checkpoint']==str(checkpoint)
    out=ROOT/'reports/omnire_scene_actor_audit_20261116_v1'/a.human_id;out.mkdir(parents=True,exist_ok=False)
    def unavailable(reason):
        (out/'validation.json').write_text(json.dumps(dict(status='unavailable',sample_id=a.sample_id,scene=scene,reason=reason,uuid=uuid,checkpoint=str(checkpoint),selection_sha256=plan['selection_sha256'],scores_produced=False),indent=2)+'\n')
    infos=json.loads((data/'instances/instances_info.json').read_text())
    true_ids=[int(k) for k,x in infos.items() if x['id']==uuid]
    if not true_ids:
        unavailable('native_identity_absent_from_training_instances');return
    assert len(true_ids)==1, 'Native identity ambiguous'
    source_sha=sha(source)
    source_matches=[p for p in (data/'images').glob('*.jpg') if p.stat().st_size==source.stat().st_size and sha(p)==source_sha]
    if not source_matches:
        unavailable('exact_frozen_source_image_absent_from_training_input');return
    assert len(source_matches)==1, 'Exact source image ambiguous in native input'
    frame,cam=map(int,source_matches[0].stem.split('_'))
    dataset=DrivingDataset(data_cfg=cfg.data)
    init_cfg=OmegaConf.to_container(cfg.model[node_type].init,resolve=True)
    native=dataset.get_init_objects(cur_node_type=node_type,**init_cfg)
    native_keys=list(native)
    dataset_ids=[i for i in native_keys if int(dataset.pixel_source.instances_true_id[i])==true_ids[0]]
    if not dataset_ids:
        unavailable('native_actor_filtered_by_original_init')
        return
    assert len(dataset_ids)==1
    instance_index=native_keys.index(dataset_ids[0])
    trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2,3),device=torch.device('cuda'))
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint),load_only_model=True)
    trainer.set_eval();model=trainer.models[node_type]
    assert model.num_instances==len(native_keys)
    # Native insertion order is reproduced from the exact training data/config.
    # Sizes and visibility are unchanged by optimization and verify the ID map.
    for index,key in enumerate(native_keys):
        assert torch.allclose(model.instances_size[index].cpu(),native[key]['size'].cpu(),atol=1e-5)
        assert torch.equal(model.instances_fv[:,index].cpu(),native[key]['frame_info'].cpu())
    for key,value in prior['canonical_parameters'].items():
        current=model.state_dict()[key][model.point_ids.reshape(-1)==instance_index].detach().cpu()
        assert torch.equal(current,value), ('Stored canonical parameter changed',key)
    assert any('features_rest' in key for key in prior['canonical_parameters'])
    del native
    unique_cam=dataset.pixel_source.camera_data[cam].unique_cam_idx
    idx=frame*dataset.pixel_source.num_cams+unique_cam
    image_info,camera_info=dataset.pixel_source.get_image(idx)
    image_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in image_info.items()}
    camera_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in camera_info.items()}
    contract=load_contract(a.sample_id)
    world_from_model=np.loadtxt(data/'extrinsics'/f'{cfg.data.start_timestep:03d}_0.txt')
    model_from_world=np.linalg.inv(world_from_model)
    model_from_camera=model_from_world @ np.asarray(contract['T_world_from_camera_opencv'])
    assert np.max(np.abs(camera_info['camera_to_world'].cpu().numpy()-model_from_camera))<1e-4
    width,height=contract['render_wh']
    camera_info['camera_to_world']=torch.tensor(model_from_camera,dtype=torch.float32,device='cuda')
    camera_info['intrinsics']=torch.tensor(contract['K_raw'],dtype=torch.float32,device='cuda')
    camera_info['width']=torch.tensor(width,device='cuda')
    camera_info['height']=torch.tensor(height,device='cuda')
    x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
    origins,viewdirs,direction_norm=get_rays(x.flatten(),y.flatten(),camera_info['camera_to_world'],camera_info['intrinsics'])
    image_info={k:image_info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
    image_info.update(origins=origins.reshape(height,width,3),viewdirs=viewdirs.reshape(height,width,3),
        direction_norm=direction_norm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
    # novel_view disables camera refinement only; the native source timestamp and appearance ID are retained.
    full=trainer(image_info,camera_info,novel_view=True)
    camera=trainer.process_camera(camera_infos=camera_info,image_ids=image_info['img_idx'].flatten()[0],novel_view=True)
    camera_audit=projection_audit(contract,model_from_world,camera.camtoworlds.cpu().numpy(),camera.Ks.cpu().numpy())
    assert bool(model.instances_fv[model.cur_frame,instance_index])
    time_path=ROOT/'reports/model_time_audit_v1/omnire.json'
    time_audit=json.loads(time_path.read_text());assert time_audit['selection_sha256']==plan['selection_sha256']
    time_record=next(r for r in time_audit['records'] if r['id']==a.human_id)
    mask=model.point_ids.reshape(-1)==instance_index
    baseline=model.get_gaussians(camera)
    # Endpoint check: querying the native grid time must reproduce the original renderer.
    endpoint=dict(time_record,timestamp_us=time_record['grid_timestamp_us'],continuous_bracket=dict(lo=int(model.cur_frame),hi=int(model.cur_frame),fraction=0.,normalized_query=float(model.normalized_timestamps[model.cur_frame])))
    with at_physical_exposure(model,instance_index,endpoint):
        native_again=model.get_gaussians(camera)
    endpoint_errors={k:float((native_again[k][mask]-baseline[k][mask]).abs().max()) for k in baseline}
    assert max(endpoint_errors.values())<1e-5,endpoint_errors
    with at_physical_exposure(model,instance_index,time_record) as temporal_query:
        gs=model.get_gaussians(camera)
    temporal_query.update(endpoint_max_errors=endpoint_errors,time_audit_sha256=sha(time_path),
        gaussian_center_change_rms=float(((gs['_means'][mask]-baseline['_means'][mask])**2).sum(-1).mean().sqrt()))
    # Keep SH, covariance, learned deformation and root pose; no image-driven alignment.

    # All comparisons reuse one physical-exposure context and unchanged checkpoint.
    # Other scene actors retain the existing scene anchor, exactly as the stored actor protocol.
    with at_physical_exposure(model,instance_index,time_record) as temporal_query:
        full_matched=trainer(image_info,camera_info,novel_view=True)
        all_gs=trainer.collect_gaussians(camera,image_info['img_idx'].flatten()[0])
        all_target=[]
        class_counts={}
        for cname in trainer.gaussian_classes:
            candidate=trainer.models[cname].get_gaussians(camera)
            if candidate is None:continue
            n=len(candidate['_means']);class_counts[cname]=n
            keep=mask if cname==node_type else torch.zeros(n,dtype=torch.bool,device='cuda')
            assert len(keep)==n
            all_target.append(keep)
        target=torch.cat(all_target)
        assert len(target)==len(all_gs.means) and int(target.sum())==int(mask.sum())
        combined,render_fn=trainer.render_gaussians(gs=all_gs,cam=camera,near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
        sky=full_matched['rgb_sky'];bias=trainer.affine_transformation(torch.zeros_like(combined['rgb_gaussians']),image_info)
        def outputs(keep):
            raw,depth,alpha=render_fn(keep.float())
            return dict(raw=raw,alpha=alpha,
                scene=trainer.affine_transformation(raw+sky*(1-alpha),image_info).clamp(0,1),
                white=(trainer.affine_transformation(raw,image_info)-bias*(1-alpha)+(1-alpha)).clamp(0,1))
        target_output=outputs(target);without=outputs(~target)
        bg=outputs(trainer.pts_labels==trainer.gaussian_classes['Background'])
        full_rgb=trainer.affine_transformation(combined['rgb_gaussians']+sky*(1-combined['opacity']),image_info).clamp(0,1)
        assert float((full_rgb-full_matched['rgb'].clamp(0,1)).abs().max())<1e-5
        isolated=dataclass_gs(**{k:v[mask] for k,v in gs.items()},detach_keys=[],extras=None)
        isolated_render,_=trainer.render_gaussians(gs=isolated,cam=camera,near_plane=trainer.render_cfg.near_plane,
            far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
        ia=isolated_render['opacity'];ir=isolated_render['rgb_gaussians']
        isolated_white=(trainer.affine_transformation(ir,image_info)-bias*(1-ia)+(1-ia)).clamp(0,1)
    # Audit artifacts only. Existing outputs and metric inputs are never overwritten.
    arrays={'full_scene_matched.png':full_rgb,'without_target_scene.png':without['scene'],
        'target_only_white.png':target_output['white'],'target_only_alpha.png':target_output['alpha'],
        'background_only_white.png':bg['white'],'background_only_alpha.png':bg['alpha'],
        'isolated_reload_white.png':isolated_white,'full_scene_original_grid.png':full['rgb']}
    for name,tensor in arrays.items():image(tensor,out/name)
    summary=json.loads((ROOT/'reports/source_camera_metric_cohort_v3/summary.json').read_text())
    row=next(r for r in summary['rows'] if r['id']==a.human_id and r['method']=='omnire')
    ref=row['reference'];ref_mask=np.asarray(Image.open(ref['reference_mask']).convert('L'))>127
    roi=ref['roi_xyxy'];m=torch.as_tensor(ref_mask,device='cuda')
    oldfolder=ROOT/'exports/common_camera_human_original75_time_v1/omnire'/a.human_id
    old=np.asarray(Image.open(oldfolder/'native_actor_source.png').convert('RGB')).astype(float)
    re=np.asarray(Image.open(out/'isolated_reload_white.png').convert('RGB')).astype(float)
    def summary_delta(a,b):
        d=(a-b).abs();return {'mask_mae_rgb01':float(d[m].mean()),'mask_max_rgb01':float(d[m].max())}
    diagnostics={'full_renderer_vs_collected_max_rgb01':float((full_rgb-full_matched['rgb'].clamp(0,1)).abs().max()),
        'isolated_reload_vs_saved_u8_max':float(abs(old-re).max()),'isolated_reload_vs_saved_u8_mean':float(abs(old-re).mean()),
        'allscene_zero_other_opacity_vs_true_subset':summary_delta(target_output['white'],isolated_white),
        'target_removal_effect_in_reference_mask':summary_delta(full_rgb,without['scene']),
        'grid_vs_physical_target_context':summary_delta(full['rgb'].clamp(0,1),full_rgb),
        'target_alpha_mean_in_reference_mask':float(target_output['alpha'][m].mean()),
        'target_alpha_over_half_reference_fraction':float((target_output['alpha'][m]>.5).float().mean()),
        'background_alpha_mean_in_reference_mask':float(bg['alpha'][m].mean())}
    from PIL import ImageDraw
    tiles=[('GT',Path(ref['reference_rgb'])),('Full scene',out/'full_scene_matched.png'),
        ('Without target actor',out/'without_target_scene.png'),('Target actor',out/'target_only_white.png'),
        ('Background only',out/'background_only_white.png')]
    sheet=Image.new('RGB',(320*len(tiles),356),'white');draw=ImageDraw.Draw(sheet)
    for col,(label,path) in enumerate(tiles):
        tile=Image.open(path).convert('RGB').crop(tuple(roi)).resize((320,320))
        sheet.paste(tile,(col*320,36));draw.text((col*320+8,10),label,fill='black')
    sheet.save(out/'comparison.png')
    record=dict(id=a.human_id,method='omnire',scene=scene,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        actor_index=instance_index,actor_gaussians=int(mask.sum()),class_counts=class_counts,
        camera_contract=common_camera_contract.CONTRACT.as_posix(),camera_projection_audit=camera_audit,
        physical_exposure=temporal_query,reference=ref,diagnostics=diagnostics,
        script_sha256=sha(__file__),source_script_sha256=sha(ROOT/'scripts/render_omnire_original75_time_v1.py'),
        old_actor_rgb_sha256=sha(oldfolder/'native_actor_source.png'),output_dir=str(out),
        scope='read-only checkpoint audit; target physical time and all scene Gaussian classes; other actors retain original scene anchor; no formal metrics changed')
    (out/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'id':a.human_id,'diagnostics':diagnostics,'output_dir':str(out)}),flush=True)

if __name__=='__main__':main()
