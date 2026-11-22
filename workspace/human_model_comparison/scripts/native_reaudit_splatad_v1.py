"""Scene-batched native SplatAD audit and independent source-exposure exports.

Selection is fixed before scoring: all target-ID rows plus Background rows with
strictly positive native compositing contribution in the frozen source mask.
Other actor IDs are never merged. The mask never erases prediction pixels.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
REPORT = H / 'reports/native_human_export_reaudit_v1/splatad'
EXPORT = H / 'exports/native_human_export_reaudit_v1/splatad'
LEASES = R / 'model_preparation/20261109_2100_CST/runs/rigid111_gpu_locks'
sys.path[:0] = [str(H/'scripts'), str(R/'adapters'), str(R/'scripts')]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def disk_guard():
    if shutil.disk_usage(H).free < 10*1024**3:
        raise RuntimeError('disk_free_below_10_GiB')


def scene_progress(scene, stage, sid=None):
    """Child heartbeat distinguishes model loading from actually waiting for a GPU lock."""
    statefile=REPORT/'state.json'
    if not statefile.exists():return
    with (REPORT/'state.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX);state=read(statefile)
        job=state['jobs'].get(scene)
        if job and job.get('gpu'):
            state['workers'][job['gpu']].update(status=stage,scene=scene,object_id=sid,child_pid=os.getpid(),heartbeat=time.time())
            job['phase']=stage;job['object_id']=sid;state['updated_at']=time.time();save(statefile,state)


def prepare():
    from score_streetgs_human_static_v1 import references
    active, summary_path, refs = references()
    selection = {x['id']: x for x in read(active['selection'])['objects']}
    old = {x['id']: x for x in read(summary_path)['rows'] if x['method']=='splatad'}
    plans = {}
    for p in (H/'inputs/scene_extract_v2/splatad_plan.json', H/'inputs/scene_extract_original75_v1/splatad_plan.json'):
        for x in read(p)['objects']:
            if x.get('status') == 'ready_for_native_decode': plans[x['id']] = x
    groups = {}
    for sid, obj in selection.items():
        ref = refs[sid]; prior = None
        if old[sid].get('result'):
            result = read(old[sid]['result']); reg = read(result['registration_record'])
            prior = read(reg['render_validation']); config = Path(prior['checkpoint']).parent.parent/'config.yml'
        elif sid in plans:
            config = Path(plans[sid]['config'])
        else:
            candidates = sorted((R/'runs').glob(f'splatad*/**/{obj["scene"]}/**/config.yml'))
            assert len(candidates)==1, (sid, candidates)
            config = candidates[0]
        checkpoint = config.parent/'nerfstudio_models/step-000030000.ckpt'
        assert checkpoint.exists()
        key = str(config)
        group = groups.setdefault(key, dict(scene=obj['scene'], config=str(config), checkpoint=str(checkpoint), objects=[]))
        group['objects'].append(dict(id=sid, identity_key=obj['identity_key'], source_reference=ref,
                                     old_record=old[sid], prior_validation=prior))
    for config,group in groups.items():
        duplicates=sum(g['scene']==group['scene'] for g in groups.values())
        group['job_id']=group['scene'] if duplicates==1 else group['scene']+'__'+hashlib.sha256(config.encode()).hexdigest()[:8]
    manifest = dict(schema='splatad_native_human_export_reaudit_v1', selection=active['selection'],
        selection_sha256=active['selection_sha256'], reference_summary=str(summary_path),
        reference_summary_sha256=sha(summary_path), prepared_script_sha256=sha(__file__),
        rule='all target actor-ID rows UNION native Background rows with source-mask compositing contribution F > 0; no other actor rows',
        foreground_definition='Frozen current reference mask >127; no prediction pixel erasure',
        groups=list(groups.values()), objects=len(selection), scenes=len({g['scene'] for g in groups.values()}), checkpoint_groups=len(groups))
    path = REPORT/'manifest.json'
    if path.exists():
        previous = read(path)
        assert previous['selection_sha256']==manifest['selection_sha256'] and previous['reference_summary_sha256']==manifest['reference_summary_sha256']
        return previous
    save(path, manifest)
    save(REPORT/'state.json', dict(status='prepared', updated_at=time.time(), workers={},
        jobs={g['job_id']:dict(status='pending', scene=g['scene'], ids=[x['id'] for x in g['objects']], attempts=0) for g in manifest['groups']}))
    return manifest


def native_imports():
    global np, torch, Image, native, eval_setup
    import numpy as np
    import torch
    from PIL import Image
    from train_splatad_reviewed_scene import MonitoredTrainer
    from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
    # Legacy config YAML tags resolve these aliases through __main__.
    globals()['MonitoredTrainer'] = MonitoredTrainer
    globals()['TrainOnlyMonitoredTrainer'] = TrainOnlyMonitoredTrainer
    import nerfstudio.models.splatad as native
    from nerfstudio.utils.eval_utils import eval_setup
    torch.set_num_threads(2)


def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k:device_tree(v,device) for k,v in value.items()}
    return value


def png(value, path):
    a = value.detach().cpu().numpy()
    if a.ndim==3 and a.shape[-1]==1: a=a[...,0]
    assert np.isfinite(a).all()
    a=(a.clip(0,1)*255).round().astype(np.uint8)
    Image.fromarray(a).save(path)
    return a


def isolated_dynamic(model, trajectories, actor_index):
    if actor_index is None: return None
    temporal={'actor_positions','actor_rotations_6d','actor_vel_linear','actor_vel_angular','actor_poses_at_time',
              'actor_present_at_time','initial_positions','initial_rotations_6d'}
    per_actor={'actor_sizes','actor_symmetric','actor_deformable','actor_to_id'}
    state={}
    for k,v in model.dynamic_actors.state_dict().items():
        state[k]=cpu(v[:,actor_index:actor_index+1] if k in temporal else v[actor_index:actor_index+1] if k in per_actor else v)
    state['actor_to_id'].zero_()
    return dict(state=state, trajectory_metadata=cpu(trajectories[actor_index]),
                original_actor_index=actor_index, exported_actor_index=0,
                native_behavior='Original optimized native actor trajectory; no background articulation')


def load_posed_gaussians(bundle_path, device='cpu'):
    """Geometry interface: native-metric posed rows and world transform; no scene checkpoint.

    Returns means, raw native quats, log_scales/scales, opacity_logits/opacities,
    target_actor_rows and T_world_from_native. Quaternions follow native gsplat wxyz.
    Background supplements are a static frozen-exposure snapshot.
    """
    import torch
    pack=torch.load(bundle_path,map_location='cpu',weights_only=False)
    g=pack['gauss_params']; put=lambda x:x.to(device)
    return dict(means=put(pack['posed_means']),quats=put(g['quats']),
        log_scales=put(g['scales']),scales=put(g['scales'].exp()),
        opacity_logits=put(g['opacities']),opacities=put(g['opacities'].sigmoid()),
        target_actor_rows=put(pack['target_actor_rows']),
        T_world_from_native=put(torch.as_tensor(pack['T_world_from_native'],dtype=torch.float64)),
        source_timestamp_us=pack['source_reference']['timestamp_us'],
        canonical_native_means=put(g['means']),velocities=put(pack['posed_velocities']),
        render_cfg={k:pack['raster_settings'][k] for k in ['radius_clip','rasterize_mode','absgrad','near_plane','far_plane','eps2d','tile_size']},
        raster_settings=pack['raster_settings'])


def render_bundle(bundle_path, device='cuda', native_actor_only=False, query_time_seconds=None):
    """Independent native raster/features/CNN renderer, loading only bundle + shared decoder.

    A non-reference time requires native_actor_only=True: static Background support
    must never be assigned fabricated human motion. Times use native offset seconds.
    """
    if 'native' not in globals(): native_imports()
    pack=torch.load(bundle_path,map_location='cpu',weights_only=False)
    dependency=pack['shared_dependency']; assert sha(dependency['path'])==dependency['sha256']
    shared=torch.load(dependency['path'],map_location='cpu',weights_only=False)
    g=device_tree(pack['gauss_params'],device)
    chosen=pack['target_actor_rows'].to(device) if native_actor_only else torch.ones(len(g['means']),dtype=torch.bool,device=device)
    means=pack['posed_means'].to(device); velocities=pack['posed_velocities'].to(device)
    if query_time_seconds is not None:
        assert native_actor_only and pack['native_dynamic'] is not None
        from nerfstudio.model_components.dynamic_actors import DynamicActors,DynamicActorsConfig
        d=pack['native_dynamic']; actors=DynamicActors(DynamicActorsConfig(),[d['trajectory_metadata']]).to(device)
        actors.load_state_dict(device_tree(d['state'],device),strict=True)
        actors.n_times,actors.n_actors=d['state']['actor_positions'].shape[:2];actors.eval()
        proxy=SimpleNamespace(dynamic_actors=actors,training=False)
        means_t,vel_t=native.SplatADModel._get_actor_adjusted_means(proxy,g['means'][chosen],
            torch.tensor([[query_time_seconds]],device=device),torch.zeros((int(chosen.sum()),1),device=device),calc_vels=True)
        means=means.clone();velocities=velocities.clone();means[chosen]=means_t;velocities[chosen]=vel_t
    kwargs=device_tree(pack['raster_settings'],device)
    kwargs.update(means=means[chosen],quats=g['quats'][chosen],scales=g['scales'][chosen].exp(),
        opacities=g['opacities'][chosen].sigmoid().squeeze(-1),
        colors=torch.cat([g['features_dc'][chosen],g['features_rest'][chosen]],dim=-1),
        velocities=velocities[chosen],render_mode='RGB+ED')
    decoder=native.RGBDecoderCNN(**shared['decoder_config']).to(device)
    decoder.load_state_dict(device_tree(shared['decoder_state'],device),strict=True);decoder.eval()
    with torch.no_grad():
        rendered,alpha,_=native.rasterization(**kwargs)
        feature=rendered[...,:-1]
        embedding=shared['appearance_embedding'][pack['sensor_index']].to(device).expand(*feature.shape[:-1],-1)
        feature=torch.cat([feature,embedding],dim=-1)
        from nerfstudio.cameras.cameras import Cameras
        c=pack['camera'];K=c['K'];camera=Cameras(camera_to_worlds=torch.as_tensor(c['camera_to_world_gl'],dtype=torch.float32)[None],
            fx=K[0][0],fy=K[1][1],cx=K[0][2],cy=K[1][2],width=c['render_wh'][0],height=c['render_wh'][1]).to(device)
        rays=native.get_ray_dirs_pinhole(camera,c['render_wh'][0],c['render_wh'][1],camera.camera_to_worlds)
        rgb=(decoder(feature,rays[None])+(1-alpha)).clamp(0,1)
    return rgb[0].cpu(),alpha[0].cpu()


def make_camera(manager,model,ref):
    source=Path(ref['reference_rgb']).resolve()
    paths=[Path(p).resolve() for p in manager.train_dataset.image_filenames]
    matches=[i for i,p in enumerate(paths) if p==source]
    if matches: index=matches[0]
    else:
        same=[i for i,p in enumerate(paths) if p.parent.name==source.parent.name]
        assert same,'No matching sensor in native dataset'
        native_t=ref['timestamp_us']/1e6-float(manager.train_dataparser_outputs.time_offset)
        index=min(same,key=lambda i:abs(float(manager.train_dataset.cameras.times[i].item())-native_t))
    camera=manager.train_dataset.cameras[index:index+1].to(model.device)
    parsed=manager.train_dataparser_outputs;assert float(parsed.dataparser_scale)==1.0
    contract=read(ref['camera_contract']);assert sha(ref['camera_contract'])==ref['camera_contract_sha256']
    model_from_world=np.eye(4);model_from_world[:3]=parsed.dataparser_transform.detach().cpu().numpy()[:3]
    pose=model_from_world@np.asarray(contract['T_world_from_camera_opencv'])@np.diag([1.,-1.,-1.,1.])
    native_pose_error=float(np.abs(camera.camera_to_worlds[0].cpu().numpy()-pose[:3]).max())
    if matches: assert native_pose_error<1e-3
    camera.camera_to_worlds[0]=torch.tensor(pose[:3],device=model.device,dtype=camera.camera_to_worlds.dtype)
    width,height=contract['render_wh'];camera.width.fill_(width);camera.height.fill_(height)
    K=np.asarray(contract['K_raw'])
    for name,value in [('fx',K[0,0]),('fy',K[1,1]),('cx',K[0,2]),('cy',K[1,2])]:getattr(camera,name).fill_(float(value))
    t=ref['timestamp_us']/1e6-float(parsed.time_offset);camera.times.fill_(t)
    return camera,dict(render_wh=[width,height],K=K.tolist(),camera_to_world_gl=pose[:3].tolist(),
        source_camera_index=index,exact_training_image_match=bool(matches),source_native_pose_error=native_pose_error,
        timestamp_us=ref['timestamp_us'],native_time_seconds=t,time_offset_seconds=float(parsed.time_offset)),np.linalg.inv(model_from_world)


def shared_decoder(model,group,pilot):
    path=(REPORT/'pilots/shared' if pilot else EXPORT/'shared')/(group['job_id']+'.pt')
    core=getattr(model.rgb_decoder,'_orig_mod',model.rgb_decoder);state=core.state_dict()
    first=state['net.0.main_branch.0.weight'];blocks=max(int(k.split('.')[1]) for k in state)
    pack=dict(schema='splatad_native_decoder_v1',scene=group['scene'],checkpoint=group['checkpoint'],
        decoder_state=cpu(state),appearance_embedding=cpu(model.appearance_embedding.weight),
        decoder_config=dict(in_dim=first.shape[1],hidden_dim=first.shape[0],kernel_size=first.shape[-1],
            num_hidden_blocks=blocks-1,out_dim=state[f'net.{blocks}.weight'].shape[0]),
        model_source=str(Path(native.__file__)),model_source_sha256=sha(native.__file__),
        contains_scene_gaussians=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():torch.save(pack,path)
    else:
        old=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.equal(old['decoder_state'][k],v) for k,v in pack['decoder_state'].items())
    return dict(path=str(path),sha256=sha(path),contains_scene_gaussians=False)


def process_object(row,group,pipeline,checkpoint_sha,dependency,pilot=False):
    started=time.time();sid=row['id'];ref=row['source_reference'];disk_guard()
    if row.get('prior_validation'):
        assert checkpoint_sha==row['prior_validation']['checkpoint_sha256'],'Original checkpoint changed'
    out=(REPORT/'pilots' if pilot else EXPORT/'objects')/sid
    report=(REPORT/'pilots' if pilot else REPORT/'objects')/sid
    if (out/'validation.json').exists():
        old=read(out/'validation.json')
        if old.get('status')=='passed':return old
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    assert sha(ref['reference_rgb'])==ref['reference_rgb_sha256']
    assert sha(ref['reference_mask'])==ref['reference_mask_sha256']
    model=pipeline.model;manager=pipeline.datamanager
    camera,camera_info,world_from_native=make_camera(manager,model,ref)
    trajectories=manager.train_dataparser_outputs.metadata['trajectories']
    uuid=row['identity_key'].split(':')[-1]
    actor_index=next((i for i,t in enumerate(trajectories) if t['uuid']==uuid),None)
    original=model.gauss_params;ids=model.id.detach().reshape(-1);nactors=len(trajectories)
    target=ids==actor_index if actor_index is not None else torch.zeros(len(ids),dtype=torch.bool,device=ids.device)
    background=ids>=nactors
    assert bool(((ids>=0)&(ids<=nactors)).all())
    native_bg=model.config.background_color
    captured={};raster=native.rasterization
    def capture(**kwargs):
        result=raster(**kwargs)
        if not captured:
            captured.update(kwargs={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()},alpha=result[1].detach())
        return result
    native.rasterization=capture
    try:
        with torch.no_grad():full=model.get_outputs_for_camera(copy.deepcopy(camera))
    finally:native.rasterization=raster
    full_rgb=png(full['rgb'],report/'full_scene.png');png(full['accumulation'],report/'full_alpha.png')
    del full
    truth=np.asarray(Image.open(ref['reference_mask']).convert('L'))>127
    assert truth.shape==(camera_info['render_wh'][1],camera_info['render_wh'][0]) and truth.any()
    mask=torch.from_numpy(truth).to(model.device)
    kwargs=captured['kwargs'];gradient_kwargs={k:v for k,v in kwargs.items() if k!='colors'}
    colors=torch.ones((len(ids),3),device=model.device,dtype=kwargs['means'].dtype,requires_grad=True)
    gradient_kwargs.update(colors=colors,render_mode='RGB',sh_degree=None,absgrad=False)
    rgb,alpha,_=raster(**gradient_kwargs)
    alpha_error=float((alpha-captured['alpha']).abs().max());constant_error=float((rgb-alpha).abs().max())
    assert alpha_error<1e-5 and constant_error<1e-5
    loss=rgb[0,:,:,0][mask].sum()+rgb[0,:,:,1][~mask].sum()
    weight=torch.autograd.grad(loss,colors,only_inputs=True)[0][:,:2].detach().cpu().numpy()
    assert np.isfinite(weight).all() and (weight>=0).all()
    expected=np.array([float(captured['alpha'][0,:,:,0][mask].double().sum()),float(captured['alpha'][0,:,:,0][~mask].double().sum())])
    conservation=np.abs(weight.astype(np.float64).sum(0)-expected)/np.maximum(expected,1.)
    assert conservation.max()<2e-4,conservation
    fg,bg=weight.T;target_cpu=target.cpu().numpy();bg_cpu=background.cpu().numpy()
    added=bg_cpu&(fg>0);selected=target_cpu|added;indices=np.flatnonzero(selected).astype(np.int64)
    np.savez_compressed(report/'native_contributions.npz',foreground_weight=fg,background_weight=bg,
        target_rows=target_cpu,background_rows=bg_cpu,selected_indices=indices)
    contribution=dict(full_foreground_mass=float(expected[0]),target_foreground_mass=float(fg[target_cpu].sum(dtype=np.float64)),
        background_foreground_mass=float(fg[bg_cpu].sum(dtype=np.float64)),
        excluded_other_actor_foreground_mass=float(fg[~(target_cpu|bg_cpu)].sum(dtype=np.float64)),
        retained_eligible_foreground_mass=float(fg[selected].sum(dtype=np.float64)),
        selected_outside_mask_mass=float(bg[selected].sum(dtype=np.float64)),
        eligible_retained_fraction=1.0 if fg[target_cpu|bg_cpu].sum()>0 else None,
        alpha_max_error=alpha_error,constant_feature_max_error=constant_error,
        conservation_relative_error=conservation.tolist(),all_positive_background_rows_retained=True,
        other_actor_policy='Excluded all non-target dynamic actor IDs even when they contribute inside mask')
    per_actor=[];ids_cpu=ids.cpu().numpy()
    for i,t in enumerate(trajectories):
        m=ids_cpu==i;mass=float(fg[m].sum(dtype=np.float64))
        if mass>0:per_actor.append(dict(index=i,uuid=t['uuid'],label=t.get('label'),foreground_mass=mass,is_target=i==actor_index))
    contribution['contributing_actor_nodes']=per_actor
    del colors,loss,rgb,alpha,gradient_kwargs,weight;gc.collect()
    def render_selection(name,keep,white):
        model.gauss_params=original if keep is None else torch.nn.ParameterDict({k:torch.nn.Parameter(v.detach()[keep].clone(),requires_grad=False) for k,v in original.items()})
        model.config.background_color='white' if white else native_bg
        try:
            with torch.no_grad():result=model.get_outputs_for_camera(copy.deepcopy(camera))
            image=png(result['rgb'],report/(name+'.png'));a=png(result['accumulation'],report/(name+'_alpha.png'))
            return image,a
        finally:model.gauss_params=original;model.config.background_color=native_bg
    if target.any():old_rgb,old_alpha=render_selection('native_target_only',target,True)
    else:
        old_rgb=np.full(full_rgb.shape,255,np.uint8);old_alpha=np.zeros(truth.shape,np.uint8)
        Image.fromarray(old_rgb).save(report/'native_target_only.png');Image.fromarray(old_alpha).save(report/'native_target_only_alpha.png')
    if target.any():render_selection('without_target',~target,False)
    else:
        shutil.copyfile(report/'full_scene.png',report/'without_target.png')
        shutil.copyfile(report/'full_alpha.png',report/'without_target_alpha.png')
    render_selection('background_only',background,True)
    prior=row.get('prior_validation');replay={}
    if prior:
        for key,new,old_path in [('full_scene',full_rgb,prior['full_scene_preview']),('native_target_only',old_rgb,prior['actor_preview'])]:
            if Path(old_path).exists():
                delta=np.abs(new.astype(int)-np.asarray(Image.open(old_path).convert('RGB')).astype(int))
                replay[key]=dict(max_byte_difference=int(delta.max()),mean_byte_difference=float(delta.mean()))
        assert all(x['max_byte_difference']<=1 for x in replay.values()),replay
    assert len(indices)>0,'No target rows and no positive Background contribution'
    index=torch.from_numpy(indices).to(model.device)
    candidate_rgb,candidate_alpha=render_selection('candidate',index,True)
    selected_params={k:cpu(v[index]) for k,v in original.items()}
    native_dynamic=isolated_dynamic(model,trajectories,actor_index)
    posed_means=cpu(kwargs['means'][index]);posed_velocities=cpu(kwargs['velocities'][index])
    snapshot={k:cpu(v) for k,v in kwargs.items() if k not in ['means','quats','scales','opacities','colors','velocities']}
    metadata=camera.metadata or {};sensor=metadata.get('sensor_idxs',torch.tensor([model.fallback_sensor_idx.value],device=model.device))
    sensor_index=int(sensor.reshape(-1)[0])
    pack=dict(schema='splatad_human_native_reaudit_v1',id=sid,method='splatad',scene=group['scene'],
        checkpoint=group['checkpoint'],checkpoint_sha256=checkpoint_sha,
        source_reference=ref,camera=camera_info,T_world_from_native=world_from_native.tolist(),
        gauss_params=selected_params,posed_means=posed_means,posed_velocities=posed_velocities,
        original_indices=torch.from_numpy(indices),target_actor_rows=torch.from_numpy(target_cpu[indices]),
        background_rows=torch.from_numpy(bg_cpu[indices]),native_dynamic=native_dynamic,
        shared_dependency=dependency,raster_settings=snapshot,sensor_index=sensor_index,
        coordinate_frame='Original native scene metric frame; canonical target rows retained with exact frozen-time posed means',
        original_actor_index=actor_index,selection_rule='All target-ID rows + Background F>0; other actor nodes excluded',
        annotation_conditioning='Frozen current source mask selects rows by native compositing contribution; never erases prediction pixels',
        static_background_support=True,dynamic_policy='Only native actor rows retain native temporal motion; Background supplements are a frozen-time static snapshot',
        contains_full_scene_gaussians=False,requires_scene_checkpoint_for_render=False)
    bundle=out/'native_asset.pt';torch.save(pack,bundle)
    del pack,captured,kwargs,index,selected_params;gc.collect();torch.cuda.empty_cache()
    reload_rgb,reload_alpha=render_bundle(bundle,device=str(model.device))
    final_rgb=png(reload_rgb,out/'native_actor_source.png');final_alpha=png(reload_alpha,out/'native_actor_alpha.png')
    reload_error=dict(rgb_max_byte_difference=int(np.abs(final_rgb.astype(int)-candidate_rgb.astype(int)).max()),
        alpha_max_byte_difference=int(np.abs(final_alpha.astype(int)-candidate_alpha.astype(int)).max()))
    assert max(reload_error.values())<=1,reload_error
    changed=bool(added.any()) or prior is None
    record=dict(id=sid,method='splatad',status='passed',source_reference=ref,identity_key=row['identity_key'],
        timestamp_us=ref['timestamp_us'],camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],
        checkpoint=group['checkpoint'],checkpoint_sha256=checkpoint_sha,bundle=str(bundle),bundle_sha256=sha(bundle),
        actor_preview=str(out/'native_actor_source.png'),actor_preview_sha256=sha(out/'native_actor_source.png'),
        actor_alpha=str(out/'native_actor_alpha.png'),actor_alpha_sha256=sha(out/'native_actor_alpha.png'),
        prediction_encoding='white_composited_uint8',render_wh=camera_info['render_wh'],T_world_from_native=world_from_native.tolist(),
        scene=group['scene'],source_camera=camera_info,actor_node_present=actor_index is not None,
        actor_gaussians=int(target.sum()),added_background_gaussians=int(added.sum()),selected_gaussians=len(indices),
        shared_dependency=dependency,source_node_policy='All target node retained, positive visible-contribution Background support added, all unrelated actors excluded',
        source_support_status='native_actor_with_background_support_candidate' if target.any() else 'background_only_candidate_target_presence_requires_visual_review',
        semantic_review_required=True,semantic_reconstruction_success_claimed=False,source_contributions=contribution,
        old_output=prior,reused=not changed,changed=changed,
        reason='Fixed native contribution rule adds Background support; mixed background/tails remain explicitly documented' if changed else 'Complete native target node reused; no additional positive Background support',
        reload_verified=True,reload_independent_of_scene_gaussians=True,reload_error=reload_error,old_image_replay=replay,
        gaussian_parameters_modified=False,trained=False,reference_mask_erases_prediction=False,
        dynamic_eligibility='Use preserved original native actor-only branch; static supplements are not a new human motion sequence' if target.any() else 'No native actor motion; static snapshot only',
        comparison_images={k:str(report/(k+'.png')) for k in ['full_scene','without_target','native_target_only','background_only','candidate']},
        contribution_arrays=str(report/'native_contributions.npz'),contribution_arrays_sha256=sha(report/'native_contributions.npz'),
        script_sha256=sha(__file__),elapsed_seconds=time.time()-started)
    save(out/'validation.json',record);save(report/'audit.json',record)
    print(json.dumps(dict(id=sid,status='passed',actor=int(target.sum()),added=int(added.sum()),selected=len(indices),reload=reload_error)),flush=True)
    return record


def process_scene(scene,ids=None,pilot=False):
    if not pilot:scene_progress(scene,'loading_scene')
    native_imports();disk_guard();manifest=read(REPORT/'manifest.json')
    matches=[g for g in manifest['groups'] if g['job_id']==scene or (g['scene']==scene and ids is not None and any(o['id'] in ids for o in g['objects']))]
    assert len(matches)==1,(scene,[g['job_id'] for g in matches])
    group=matches[0]
    objects=[o for o in group['objects'] if ids is None or o['id'] in ids]
    if not pilot:objects=[o for o in objects if not (EXPORT/'objects'/o['id']/'validation.json').exists() or read(EXPORT/'objects'/o['id']/'validation.json').get('status')!='passed']
    if not objects:return
    checkpoint_sha=sha(group['checkpoint'])
    def inference_initialization(config):
        config.pipeline.datamanager.paint_points=False
        return config
    _,pipeline,checkpoint,step=eval_setup(Path(group['config']),test_mode='val',update_config_callback=inference_initialization)
    assert step==30000 and str(checkpoint)==group['checkpoint']
    pipeline.eval();model=pipeline.model
    model.config.use_camopt_in_eval=False;model.config.compensate_rs_camera=False
    for k in ('lin_vel_x','lin_vel_y','lin_vel_z','ang_vel_x','ang_vel_y','ang_vel_z','rs_time'):model.rs_editing[k]=0.
    dependency=shared_decoder(model,group,pilot)
    failures=[]
    for row in objects:
        if not pilot:scene_progress(scene,'rendering_and_exporting',row['id'])
        try:process_object(row,group,pipeline,checkpoint_sha,dependency,pilot)
        except Exception as error:
            detail=dict(id=row['id'],method='splatad',status='audit_failed',source_reference=row['source_reference'],
                checkpoint=group['checkpoint'],checkpoint_sha256=checkpoint_sha,reason=repr(error),traceback=traceback.format_exc(),
                semantic_generation_failure_claimed=False,reload_verified=False,script_sha256=sha(__file__))
            dest=(REPORT/'pilots' if pilot else EXPORT/'objects')/row['id']/'validation.json';save(dest,detail)
            failures.append(row['id']);print(detail['traceback'],flush=True)
            if pilot:raise
    del pipeline,model;gc.collect();torch.cuda.empty_cache()
    if failures:raise RuntimeError('Objects require failed-audit review: '+','.join(failures))


def worker(gpu):
    from run_splatad_human_train_only_queue import environment,PYTHON
    REPORT.mkdir(parents=True,exist_ok=True);statefile=REPORT/'state.json'
    def state_update(fn):
        with (REPORT/'state.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);s=read(statefile);result=fn(s);s['updated_at']=time.time();save(statefile,s);return result
    state_update(lambda s:s['workers'].update({gpu:dict(pid=os.getpid(),status='running',started_at=time.time())}))
    while True:
        if shutil.disk_usage(H).free<10*1024**3:
            state_update(lambda s:s['workers'][gpu].update(status='waiting_disk_10GiB'));time.sleep(15);continue
        def claim(s):
            pending=[k for k,v in s['jobs'].items() if v['status']=='pending']
            if not pending:return None
            scene=pending[0];s['jobs'][scene].update(status='running',gpu=gpu,started_at=time.time(),attempts=s['jobs'][scene]['attempts']+1)
            s['workers'][gpu].update(status='waiting_gpu_lease',scene=scene);s['status']='running';return scene
        scene=state_update(claim)
        if scene is None:break
        log=REPORT/'logs'/f'{scene}.log';log.parent.mkdir(exist_ok=True)
        with (LEASES/f'gpu_{gpu}.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            command=[str(PYTHON),'-B','-u',str(Path(__file__)),'--scene',scene]
            with log.open('a') as output:
                child=subprocess.Popen(command,cwd=R,env=environment(gpu),stdout=output,stderr=subprocess.STDOUT)
                state_update(lambda s:s['jobs'][scene].update(pid=child.pid,command=command,log=str(log)))
                code=child.wait()
        state_update(lambda s:s['jobs'][scene].update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time()))
    def finish(s):
        s['workers'][gpu].update(status='finished',finished_at=time.time())
        if all(v['status'] in ('completed','failed') for v in s['jobs'].values()):s['status']='finished_with_failures' if any(v['status']=='failed' for v in s['jobs'].values()) else 'finished'
    state_update(finish)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');p.add_argument('--scene');p.add_argument('--ids',nargs='+');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.prepare:
        m=prepare();print(json.dumps(dict(objects=m['objects'],scenes=m['scenes'],manifest=str(REPORT/'manifest.json'))))
    elif a.worker:worker(a.worker)
    elif a.scene:process_scene(a.scene,a.ids,a.pilot)
    else:p.error('Choose --prepare, --worker GPU, or --scene SCENE')


if __name__=='__main__':main()
