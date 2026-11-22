"""Export unchanged StreetGS background rows for the frozen human cohort.

One native annotated box + fixed .25m per half axis, no mask selection or fit.
Full Gaussian parameters and source affine are retained in small independent
bundles. Reload is a separate process with no scene checkpoint reads.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
import yaml

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
D = Path(_hmc_expand('${DRIVESTUDIO_ROOT}'))
OUT = H / 'exports/streetgs_human_probe_v1'
REPORT = H / 'reports/streetgs_human_probe_v1'
KEYS = ['_means', '_quats', '_scales', '_opacities', '_features_dc', '_features_rest']
sys.path.insert(0, str(R / 'scripts'))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    t.replace(p)


def prepare():
    path = REPORT / 'extraction_plan.json'
    if path.exists():
        return read(path)
    sel = H / 'inputs/benchmark_human100_v3/selection.json'
    selected = read(sel)['objects']; assert len(selected) == 100
    q = read(H / 'configs/human_source_common_v1/reference_qualification_frozen_v5.json')
    refs = {x['id']: x for x in q['objects']}
    queues = [R / 'state/nuscenes100_expansion_queue.json', R / 'state/streetgs_drivestudio_nusc_v1.json']
    checkpoints = {}
    for scene in sorted({x['scene'] for x in selected}):
        choices = list((R / 'runs/streetgs_nuscenes100_expansion/formal/nuScenes' / scene).glob('checkpoint_final.pth'))
        choices += list((R / 'runs/streetgs_drivestudio_nusc_v1/formal/nuScenes' / scene).glob('checkpoint_final.pth'))
        if not choices:
            # Only recorded StreetGS jobs, never another model's checkpoint.
            def walk(x):
                if isinstance(x, dict):
                    if x.get('checkpoint') and scene in str(x['checkpoint']) and 'streetgs' in str(x['checkpoint']).lower():
                        cp = Path(x['checkpoint'])
                        if cp.is_file() and cp.name == 'checkpoint_final.pth': choices.append(cp)
                    for y in x.values(): walk(y)
                elif isinstance(x, list):
                    for y in x: walk(y)
            for p in queues:
                if p.exists(): walk(read(p))
        choices = sorted(set(choices))
        assert choices, ('No recorded StreetGS checkpoint', scene)
        cp = choices[0]; cfgp = cp.with_name('config.yaml'); cfg = yaml.safe_load(cfgp.read_text())
        assert 'RigidNodes' in cfg['model'] and 'DeformableNodes' not in cfg['model'] and 'SMPLNodes' not in cfg['model']
        checkpoints[scene] = dict(checkpoint=str(cp), config=str(cfgp), data=str(Path(cfg['data']['data_root']) / '000'))
    rows = []
    for obj in selected:
        ref = refs[obj['id']]; cp = checkpoints[obj['scene']]
        assert ref['identity_key'] == obj['identity_key'] and ref['timestamp_us'] == obj['t_ref_us']
        rows.append(dict(id=obj['id'], scene=obj['scene'], identity_key=obj['identity_key'], timestamp_us=obj['t_ref_us'],
                         camera_contract=ref['camera_contract'], camera_contract_sha256=ref['camera_contract_sha256'], **cp))
    d = dict(schema='streetgs_human_background_extraction_v1', selection_sha256=sha(sel), rows=rows,
             padding_per_axis_m=.25, selection='native_background_centres_inside_source_native_annotation_box_plus_fixed_padding',
             raw_parameters_modified=False, score_based_selection=False, mask_used_for_row_selection=False,
             temporal_policy='No native human nodes: no inferred or GT-driven human motion added',
             script_sha256=sha(__file__), created_at=time.time())
    # Retain both selection path and selection rule explicitly.
    d['selection_path'] = str(sel); d['selection_rule'] = d.pop('selection')
    save(path, d); return d


def source_binding(row, cfg):
    camera = read(row['camera_contract']); assert sha(row['camera_contract']) == row['camera_contract_sha256']
    assert camera['identity_key'] == row['identity_key'] and camera['timestamp_us'] == row['timestamp_us']
    source = Path(camera['source_rgb']); assert sha(source) == camera['source_rgb_sha256']
    data = Path(row['data'])
    matches = [p for p in (data / 'images').glob('*.jpg') if p.stat().st_size == source.stat().st_size and sha(p) == camera['source_rgb_sha256']]
    if len(matches) != 1:
        raise ValueError(f'Frozen source not uniquely present in native training images: {len(matches)}')
    frame, camid = map(int, matches[0].stem.split('_'))
    info = read(data / 'instances/instances_info.json')
    objects = [v for v in info.values() if v['id'] == row['identity_key'].split(':')[-1]]
    if len(objects) != 1: raise ValueError('Frozen identity absent or ambiguous in native annotations')
    obj = objects[0]; ann = obj['frame_annotations']; idx = ann['frame_idx'].index(frame)
    Twobj = np.asarray(ann['obj_to_world'][idx]); dims = np.asarray(ann['box_size'][idx])
    assert Twobj.shape == (4,4) and dims.shape == (3,) and np.all(dims > 0)
    assert cfg['data']['start_timestep'] == 0 and cfg['data']['pixel_source']['test_image_stride'] == 0
    M = np.linalg.inv(np.loadtxt(data / 'extrinsics/000_0.txt'))
    native_cam = M @ np.loadtxt(data / 'extrinsics' / f'{frame:03d}_{camid}.txt')
    target_cam = M @ np.asarray(camera['T_world_from_camera_opencv'])
    error = float(np.abs(native_cam - target_cam).max()); assert error < 1e-5, error
    cameras = cfg['data']['pixel_source']['cameras']; image_index = frame * len(cameras) + cameras.index(camid)
    return camera, dict(source_frame=frame, source_camera_index=image_index, native_camera_id=camid,
                       native_annotation_category=obj.get('class_name'), processed_source_image=str(matches[0]),
                       T_world_from_object=Twobj.tolist(), size_length_width_height_m=dims.tolist(),
                       T_model_from_world=M.tolist(), camera_transform_error=error,
                       box_time_policy='Original training frame annotation; used only for selecting a static subset, not animating it')


def asset(pack):
    c = pack['canonical']; cfg = pack['config']; ctrl = {**cfg['trainer']['gaussian_ctrl_general_cfg'], **cfg['model']['Background'].get('ctrl', {})}
    scales = c['_scales'].exp()
    if scales.shape[1] == 1: scales = scales.repeat(1,3)
    assert scales.shape[1] == 3
    a = dict(arrays=dict(means=c['_means'], quats=torch.nn.functional.normalize(c['_quats'], dim=-1),
                        scales=scales, opacity=c['_opacities'].sigmoid().reshape(-1)), canonical=c,
             affine_state=pack['affine_state'], config=cfg, degree=min(pack['step']//ctrl['sh_degree_interval'],ctrl['sh_degree']),
             validation=dict(source_camera_index=pack['binding']['source_camera_index']))
    return a


def render(pack):
    from register_drivestudio_actor import render_source
    camera = pack['camera']; pose = np.asarray(pack['binding']['T_model_from_world']) @ np.asarray(camera['T_world_from_camera_opencv'])
    pose = torch.as_tensor(pose, dtype=torch.float32, device='cuda')
    V = torch.linalg.inv(pose).cpu().numpy(); r = pack['config']['trainer']['render']
    rgb, alpha, _ = render_source(asset(pack), V, camera['K_raw'], camera['render_wh'], r['near_plane'], r['far_plane'], pixel_edges=False)
    return rgb, alpha


def export_scene(scene, ids=None):
    plan = prepare(); rows = [r for r in plan['rows'] if r['scene'] == scene and (not ids or r['id'] in ids)]
    assert rows; assert shutil.disk_usage(H).free > 10 * 1024**3
    cfg = yaml.safe_load(Path(rows[0]['config']).read_text()); cp = Path(rows[0]['checkpoint'])
    ck = torch.load(cp, map_location='cpu', weights_only=False, mmap=True)
    assert ck['step'] == 30000 and 'DeformableNodes' not in ck['models'] and 'SMPLNodes' not in ck['models']
    state = ck['models']['Background']; checkpoint_sha = sha(cp); config_sha = sha(rows[0]['config'])
    for row in rows:
        folder = OUT / 'objects' / row['id']; folder.mkdir(parents=True, exist_ok=True)
        p = folder / 'validation.json'
        if p.exists() and read(p)['status'] in ['passed','exported_pending_reload']: continue
        rec = dict(**row, method='streetgs', status='running', extraction='native_background_fixed_box_subset',
                   checkpoint_sha256=checkpoint_sha, config_sha256=config_sha, plan_sha256=sha(REPORT/'extraction_plan.json'),
                   script_sha256=sha(__file__), started_at=time.time(), native_dynamic_human_available=False)
        try:
            camera, bind = source_binding(row, cfg)
            from render_static_vehicle_box_pilot import select_rows
            box = dict(T_world_from_object=bind['T_world_from_object'], size_length_width_height_m=bind['size_length_width_height_m'], padding_per_axis_m=.25)
            indices = select_rows(state['_means'], torch.ones(len(state['_means']),dtype=torch.bool), np.asarray(bind['T_model_from_world']), box)
            ix = torch.as_tensor(indices); canonical = {k:state[k][ix].clone().contiguous() for k in KEYS}
            assert all(torch.isfinite(v).all() for v in canonical.values())
            assert all(torch.equal(canonical[k],state[k][ix]) for k in KEYS)
            pack = dict(schema='streetgs_independent_static_background_gaussians_v1', id=row['id'], method='streetgs',
                        identity_key=row['identity_key'], canonical=canonical, config=cfg, step=ck['step'],
                        affine_state={k:v.clone() for k,v in ck['models']['Affine'].items()} if 'Affine' in ck['models'] else None,
                        binding=bind, camera=camera, T_world_from_native=np.linalg.inv(bind['T_model_from_world']).tolist(),
                        original_indices=torch.as_tensor(indices), checkpoint_sha256=checkpoint_sha, config_sha256=config_sha)
            bp = folder / 'native_background_subset.pt'; torch.save(pack,bp)
            rgb, alpha = render(pack)
            rp, ap = folder/'native_actor_source.png', folder/'native_actor_alpha.png'
            Image.fromarray(rgb).save(rp); Image.fromarray(alpha).save(ap)
            rec.update(status='exported_pending_reload', bundle=str(bp), bundle_sha256=sha(bp), gaussians=len(indices),
                       sample_id=row['id'], actor_uuid=row['identity_key'].split(':')[-1],
                       actor_preview=str(rp), actor_preview_sha256=sha(rp), actor_alpha=str(ap), actor_alpha_sha256=sha(ap),
                       prediction_encoding='white_composited_uint8', render_wh=camera['render_wh'], binding=bind,
                       selected_parameters_equal_checkpoint=True, raw_parameters_modified=False,
                       foreground_pixels_gt_half=int((alpha>127).sum()), reload_verified=False,
                       checkpoint_models=list(ck['models']), frozen_raw_timestamp_us=row['timestamp_us'])
        except Exception as exc:
            import traceback
            rec.update(status='unavailable' if isinstance(exc,(ValueError,AssertionError)) else 'execution_failed', reason=repr(exc), traceback=traceback.format_exc())
        rec['finished_at'] = time.time(); save(p,rec); print(row['id'],rec['status'],rec.get('gaussians'),rec.get('reason'),flush=True)


def reload(sid):
    folder = OUT/'objects'/sid; p=folder/'validation.json'; rec=read(p)
    if rec['status']=='passed': return rec
    assert rec['status']=='exported_pending_reload'
    assert sha(rec['bundle']) == rec['bundle_sha256']
    pack = torch.load(rec['bundle'], map_location='cpu', weights_only=False)
    assert pack['id']==sid and pack['identity_key']==rec['identity_key']
    rgb,alpha=render(pack); diffs={}
    for name,arr,path in [('rgb',rgb,rec['actor_preview']),('alpha',alpha,rec['actor_alpha'])]:
        old=np.asarray(Image.open(path)); diff=abs(old.astype(int)-arr.astype(int))
        diffs[name+'_max']=int(diff.max()); diffs[name+'_mean']=float(diff.mean())
    assert diffs['rgb_max']==0 and diffs['alpha_max']==0,diffs
    evidence=dict(status='passed',independent_process=True,scene_checkpoint_reads=0,source_image_reads=0,
                  source_mask_reads=0,exact_replay=True,**diffs,bundle_sha256=rec['bundle_sha256'],checked_at=time.time())
    save(folder/'reload.json',evidence)
    rec.update(status='passed',reload_verified=True,reload_record=str(folder/'reload.json'),reload_sha256=sha(folder/'reload.json'))
    save(p,rec);print(sid,'reload passed',diffs,flush=True);return rec


def native_check(sid):
    """Independent upstream dataset/trainer validation of the first pilot only."""
    from run_background_vehicle_scene import DriveStudio
    rec=read(OUT/'objects'/sid/'validation.json');pack=torch.load(rec['bundle'],map_location='cpu',weights_only=False)
    trainer=DriveStudio(rec);cam=pack['camera'];bind=pack['binding']
    frame=dict(source_rgb=cam['source_rgb'],source_rgb_sha256=cam['source_rgb_sha256'],
               T_world_from_object=bind['T_world_from_object'],size_length_width_height_m=bind['size_length_width_height_m'])
    import common_camera_contract
    common_camera_contract.CONTRACT=REPORT/'extraction_plan.json'
    rgb,alpha,audit,shape=trainer.render(frame,cam,pack['original_indices'].numpy())
    values={'rgb':rgb,'alpha':alpha};checks={}
    for name,t in values.items():
        ar=(t.detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
        if name=='alpha':ar=ar[...,0]
        old=np.asarray(Image.open(rec['actor_preview'] if name=='rgb' else rec['actor_alpha']))
        diff=abs(ar.astype(int)-old.astype(int));checks[name+'_max']=int(diff.max());checks[name+'_mean']=float(diff.mean())
        Image.fromarray(ar).save(OUT/'objects'/sid/('upstream_'+name+'.png'))
    assert checks['rgb_max']<=1 and checks['alpha_max']<=1,checks
    save(REPORT/'native_adapter_check.json',dict(status='passed',id=sid,checks=checks,camera_audit=audit,
         upstream_renderer='Actual DrivingDataset + MultiTrainer + Background.get_gaussians',checkpoint=rec['checkpoint'],
         checkpoint_sha256=rec['checkpoint_sha256'],extraction_script_sha256=sha(__file__),subset_shape=shape))
    print('native check passed',checks,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','export','reload','native-check','summary']);p.add_argument('--scene');p.add_argument('--ids',nargs='*');a=p.parse_args();torch.set_num_threads(2)
    if a.stage=='prepare':print(len(prepare()['rows']))
    elif a.stage=='export':export_scene(a.scene,a.ids)
    elif a.stage=='reload':
        for sid in a.ids:reload(sid)
    elif a.stage=='native-check':native_check(a.ids[0])
    else:
        rows=[read(p) for p in sorted((OUT/'objects').glob('*/validation.json'))]
        d=dict(counts=dict(collections.Counter(r['status'] for r in rows)),frozen_denominator=100,rows=rows)
        save(REPORT/'export_summary.json',d);print(d['counts'])


if __name__=='__main__':main()
