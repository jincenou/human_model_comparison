"""Reload and render a selected native SplatAD actor with its own RGB decoder."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'vehicle_model_comparison/adapters'))
sys.path.insert(0, str(ROOT.parent / 'vehicle_model_comparison/scripts'))
from train_splatad_reviewed_scene import MonitoredTrainer  # Legacy YAML target alias.
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer  # Strict YAML target alias.
from nerfstudio.utils.eval_utils import eval_setup
import common_camera_contract
from common_camera_contract import load_contract,projection_audit


def save_image(array, path):
    Image.fromarray((array.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--human-id',required=True);args=parser.parse_args()
    args.sample_id=args.human_id;args.variant='existing_scene_reuse'
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plan=json.loads((ROOT/'inputs/scene_extract_original75_v1/splatad_plan.json').read_text())
    assert sha(plan['selection'])==plan['selection_sha256']
    item=next(x for x in plan['objects'] if x['id']==args.human_id)
    assert item['status']=='ready_for_native_decode' and item['dataset']=='nuscenes'
    assert sha(item['formal_validation'])==item['formal_validation_sha256']
    assert sha(item['config'])==item['config_sha256']
    assert sha(item['bundle'])==item['bundle_sha256']
    assert sha(item['prior_validation'])==item['prior_validation_sha256']
    assert sha(item['camera_contract'])==item['camera_contract_sha256']
    source_image=Path(item['source_image']).resolve();assert sha(source_image)==item['source_sha256']
    checkpoint_sha=sha(item['checkpoint'])
    config_path=Path(item['config']);actor_bundle=item['bundle'];actor_gaussians=item['gaussians']
    scene=item['scene'];reconstruction_only=True
    item['splatad_actor_uuid']=item['native_id'].split(':')[-1]
    common_camera_contract.CONTRACT=ROOT/'configs/human_source_common_v1/candidates'/args.human_id/'camera.json'
    contract=json.loads(common_camera_contract.CONTRACT.read_text());assert contract['selection_sha256']==plan['selection_sha256']
    assert contract['timestamp_us']==item['raw_timestamp_us']
    out=ROOT/'exports/common_camera_human_original75_time_v1/splatad'/args.sample_id
    assert not out.exists(), 'Do not overwrite existing native extraction'
    config, pipeline, checkpoint, step = eval_setup(config_path, test_mode='val')
    assert step == 30000
    model = pipeline.model
    manager = pipeline.datamanager
    filenames = [Path(p).resolve() for p in manager.train_dataset.image_filenames]
    matches = [i for i, path in enumerate(filenames) if path == source_image]
    assert len(matches) == 1, (source_image, len(matches))
    camera_index = matches[0]
    camera = manager.train_dataset.cameras[camera_index:camera_index + 1].to(model.device)
    contract=load_contract(args.sample_id)
    parsed=manager.train_dataparser_outputs
    assert float(parsed.dataparser_scale)==1.0
    model_from_world=np.eye(4)
    model_from_world[:3]=parsed.dataparser_transform.detach().cpu().numpy()[:3]
    cv_to_gl=np.diag([1.,-1.,-1.,1.])
    model_from_camera_cv=model_from_world @ np.asarray(contract['T_world_from_camera_opencv'])
    model_from_camera_gl=model_from_camera_cv @ cv_to_gl
    native_pose_error=float(np.max(np.abs(camera.camera_to_worlds[0].cpu().numpy()-model_from_camera_gl[:3])))
    assert native_pose_error<1e-3, native_pose_error
    native_render_wh=[int(camera.width.item()),int(camera.height.item())]
    camera.camera_to_worlds[0]=torch.tensor(model_from_camera_gl[:3],device=model.device,dtype=camera.camera_to_worlds.dtype)
    width,height=contract['render_wh']
    camera.width.fill_(width);camera.height.fill_(height)
    raw_k=np.asarray(contract['K_raw'])
    for name,value in [('fx',raw_k[0,0]),('fy',raw_k[1,1]),('cx',raw_k[0,2]),('cy',raw_k[1,2])]:
        getattr(camera,name).fill_(float(value))
    assert abs(float(camera.times.item())-(contract['timestamp_us']/1e6-float(parsed.time_offset)))<1e-4
    model.config.use_camopt_in_eval=False
    model.config.compensate_rs_camera=False
    for key in ('lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','rs_time'):
        model.rs_editing[key]=0.
    camera_audit=projection_audit(contract,model_from_world,
        np.vstack([camera.camera_to_worlds[0].cpu().numpy(),[0,0,0,1]]) @ cv_to_gl,
        camera.get_intrinsics_matrices()[0].cpu().numpy())
    camera_audit.update(native_training_render_wh=native_render_wh,native_pose_max_error=native_pose_error,
                        timestamp_us=contract['timestamp_us'],native_time_offset_seconds=float(parsed.time_offset))
    actor_index = next(i for i, actor in enumerate(manager.train_dataparser_outputs.metadata['trajectories'])
                       if actor['uuid'] == item['splatad_actor_uuid'])
    mask = model.id.detach().reshape(-1) == actor_index
    assert int(mask.sum()) == actor_gaussians
    bundle = torch.load(actor_bundle, map_location='cpu', weights_only=False)
    assert bundle['uuid'] == item['splatad_actor_uuid'] and bundle['actor_index'] == actor_index
    for name, parameter in model.gauss_params.items():
        assert torch.equal(parameter.detach()[mask].cpu(), bundle['gauss_params'][name])
    pipeline.eval()
    actors=model.dynamic_actors
    assert all(actors.actor_editing[k]==0. for k in ('lateral','longitudinal','rotation','height'))
    with torch.no_grad():
        query=camera.times.reshape(-1,1)
        times=actors.unique_timestamps
        assert torch.isfinite(times).all() and bool((times[1:]>times[:-1]).all())
        assert float(times[0])<=float(query.item())<=float(times[-1]), 'Query outside native timeline'
        right=min(int(torch.searchsorted(times,query.reshape(-1))[0]),len(times)-1)
        left=max(right-1,0)
        pose_matrix,valid=actors.get_boxes2world(query,flatten=False)
        assert bool(valid[0,actor_index]), 'Actor absent at both native interpolation brackets'
        pose_matrix=pose_matrix[0,actor_index]
        assert torch.isfinite(pose_matrix).all()
        points=model.gauss_params['means'].detach()[mask]
        ids=model.gauss_params['id'].detach()[mask]
        native_points,_=model._get_actor_adjusted_means(points,query,ids,calc_vels=False)
        expected=points @ pose_matrix[:3,:3].T+pose_matrix[:3,3]
        error=float((native_points-expected).abs().max())
        assert error<1e-4, error
        time_audit=dict(timestamp_us=contract['timestamp_us'],time_offset_seconds=float(parsed.time_offset),
            physical_query_seconds=contract['timestamp_us']/1e6-float(parsed.time_offset),native_query_seconds=float(query.item()),
            query_error_seconds=abs(float(query.item())-(contract['timestamp_us']/1e6-float(parsed.time_offset))),
            native_bracket_indices=[left,right],native_bracket_times=[float(times[left]),float(times[right])],
            actor_present_at_brackets=[bool(actors.actor_present_at_time[left,actor_index]),bool(actors.actor_present_at_time[right,actor_index])],
            box_to_model_at_query=pose_matrix.cpu().tolist(),native_means_transform_max_error=error,
            pose_query='Native learned 6D rotation and position interpolation, including native epsilon/clamping semantics; no edit or re-pose.',
            appearance='Unchanged native SplatAD quaternion/decoder behavior; no substitute SH renderer.',
            frozen_time_inside_native_timeline=True)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        scene_result = model.get_outputs_for_camera(camera=copy.deepcopy(camera))
        original_parameters = model.gauss_params
        subset = torch.nn.ParameterDict({name: torch.nn.Parameter(parameter.detach()[mask].clone(), requires_grad=False)
                                         for name, parameter in original_parameters.items()})
        model.gauss_params = subset
        original_background = model.config.background_color
        try:
            model.config.background_color = 'white'
            actor_result = model.get_outputs_for_camera(camera=copy.deepcopy(camera))
        finally:
            model.gauss_params = original_parameters
            model.config.background_color = original_background
    rgb = actor_result['rgb'].detach()
    alpha = actor_result['accumulation'].detach()
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    assert float(alpha.max()) > 0 and float(rgb.std()) > 1e-5
    save_image(scene_result['rgb'], out / 'native_full_scene_source.png')
    save_image(rgb, out / 'native_actor_source.png')
    save_image(alpha.squeeze(-1), out / 'native_actor_alpha.png')
    result = {'status': 'passed', 'sample_id': args.sample_id, 'scene': scene,
              'source_image': str(source_image), 'source_camera_index': camera_index,
              'source_relation': 'exact frame traced from frozen retained input provenance; no pixel metric asserted',
              'actor_uuid': item['splatad_actor_uuid'], 'gaussians': int(mask.sum()),
              'checkpoint': str(checkpoint), 'step': step, 'bundle': actor_bundle,
              'variant':'existing_scene_reuse','scene_validation_source':item['formal_validation'],
              'real_heldout_eligible':False,'holdout_policy':'Pending complete human input audit',
              'selection_sha256':plan['selection_sha256'],'checkpoint_sha256':checkpoint_sha,'bundle_sha256':item['bundle_sha256'],
              'config_sha256':item['config_sha256'],'extraction_script_sha256':sha(__file__),
              'stored_canonical_parameters_equal':True,'state_time_query_valid':True,'state_time_audit':time_audit,
              'native_checkpoint_reloaded':True,'native_variant':item['native_variant'],
              'common_camera_projection_valid':True,'common_camera_valid':False,'metric_eligible':False,'scores_produced':False,
              'frozen_raw_timestamp_us':item.get('raw_timestamp_us'),
              'renderer': 'native SplatAD rasterizer and native RGB decoder',
              'background': 'white',
              'common_camera_audit':camera_audit,
              'render_wh': [int(rgb.shape[1]), int(rgb.shape[0])],
              'source_camera_K': camera.get_intrinsics_matrices()[0].detach().cpu().tolist(),
              'source_camera_to_world': camera.camera_to_worlds[0].detach().cpu().tolist(),
              'source_camera_time': camera.times.detach().cpu().reshape(-1).tolist(),
              'full_scene_preview': str(out / 'native_full_scene_source.png'),
              'actor_preview': str(out / 'native_actor_source.png'),
              'actor_alpha': str(out / 'native_actor_alpha.png'),
              'actor_alpha_pixels_gt_0_5': int((alpha > 0.5).sum()),
              'rgb_std': float(rgb.std())}
    (out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
