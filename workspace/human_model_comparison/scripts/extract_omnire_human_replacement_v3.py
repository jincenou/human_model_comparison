"""Recover native instance identity and render one fixed-time OmniRe actor."""
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

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(_hmc_expand('${DRIVESTUDIO_ROOT}'))
sys.path.insert(0,str(REPO))
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.gaussians.basics import dataclass_gs

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
    plan_path=ROOT/'inputs/scene_extract_replacement_v3/omnire_plan.json'
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
    node_type='DeformableNodes'
    out=ROOT/'exports/omnire_human100_v2/objects'/a.human_id;out.mkdir(parents=True,exist_ok=False)
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
    del native
    unique_cam=dataset.pixel_source.camera_data[cam].unique_cam_idx
    idx=frame*dataset.pixel_source.num_cams+unique_cam
    image_info,camera_info=dataset.pixel_source.get_image(idx)
    image_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in image_info.items()}
    camera_info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in camera_info.items()}
    full=trainer(image_info,camera_info)
    camera=trainer.process_camera(camera_infos=camera_info,image_ids=image_info['img_idx'].flatten()[0],novel_view=False)
    assert bool(model.instances_fv[model.cur_frame,instance_index])
    gs=model.get_gaussians(camera);mask=model.point_ids.reshape(-1)==instance_index
    if int(mask.sum())==0:
        unavailable('native_actor_has_zero_gaussians_after_training');return
    subset={k:v[mask] for k,v in gs.items()}
    actor=dataclass_gs(**subset,detach_keys=[],extras=None)
    render,_=trainer.render_gaussians(gs=actor,cam=camera,near_plane=trainer.render_cfg.near_plane,
        far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
    alpha=render['opacity'];raw=render['rgb_gaussians']
    # Apply the native affine to foreground color; the affine bias is alpha weighted.
    bias=trainer.affine_transformation(torch.zeros_like(raw),image_info)
    rgb=(trainer.affine_transformation(raw,image_info)-bias*(1-alpha)+(1-alpha)).clamp(0,1)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert float(alpha.max())>0 and float(rgb.std())>1e-5
    image(full['rgb'],out/'native_full_scene_source.png');image(rgb,out/'native_actor_source.png');image(alpha,out/'native_actor_alpha.png')
    bundle={'sample_id':a.sample_id,'uuid':uuid,'native_instance_index':instance_index,
        'native_dataset_id':dataset_ids[0],'source_frame':frame,'native_cur_frame':int(model.cur_frame),
        'checkpoint':str(checkpoint),'posed_gaussians':{k:v.cpu() for k,v in subset.items()},
        'canonical_parameters':{k:v[mask].detach().cpu() for k,v in model.state_dict().items()
                                if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==mask.numel()},
        'camera_K':camera.Ks.cpu(),'camera_to_world':camera.camtoworlds.cpu(),
        'appearance_contract':'Posed RGB is source-view SH evaluation; retain checkpoint/SH and native renderer for other views.'}
    torch.save(bundle,out/'native_actor_snapshot.pt')
    result={'status':'passed','sample_id':a.sample_id,'scene':scene,'method':'OmniRe legacy non-SMPL',
        'source_image':str(source),'source_image_sha256':source_sha,'processed_source_image':str(source_matches[0]),
        'native_id':item['native_id'],'actor_uuid':uuid,'actor_index':instance_index,'gaussians':int(mask.sum()),
        'source_frame':frame,'source_camera_index':int(idx),'native_cur_frame':int(model.cur_frame),
        'native_node_type':node_type,'frozen_raw_timestamp_us':item.get('raw_timestamp_us'),
        'checkpoint':str(checkpoint),'step':int(trainer.step),'bundle':str(out/'native_actor_snapshot.pt'),
        'real_heldout_eligible':False,'holdout_policy':'Pending full human input audit; prior scene flag not inherited',
        'selection_sha256':plan['selection_sha256'],'checkpoint_sha256':checkpoint_sha,'config_sha256':item['config_sha256'],
        'extraction_script_sha256':sha(__file__),'common_camera_valid':False,'scores_produced':False,
        'identity_validation':'reproduced native initialization order; all instance sizes and frame visibility exactly checked',
        'renderer':'native OmniRe renderer, SH and alpha-weighted native affine, white background',
        'actor_preview':str(out/'native_actor_source.png'),'actor_alpha':str(out/'native_actor_alpha.png'),
        'full_scene_preview':str(out/'native_full_scene_source.png'),'render_wh':[int(camera.W),int(camera.H)],
        'camera_K':camera.Ks.cpu().tolist(),'camera_to_world':camera.camtoworlds.cpu().tolist()}
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
