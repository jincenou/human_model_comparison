"""Bounded CPU geometry audit; native network evaluation, not video acceptance.

Retains native SH/decoder features. Tests the same actor at visible native time
samples, without altering poses or fitting to reference images.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
D = Path(_hmc_expand('${DRIVESTUDIO_ROOT}'))
S = Path(_hmc_expand('${SPLATAD_ROOT}'))
OUT = H / 'reports/dynamic_human_feasibility_20261112_v1'
EXP = H / 'exports/dynamic_human_feasibility_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def native_defs(path, names, parent=None):
    """Evaluate exact selected upstream CPU definitions without CUDA imports."""
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body
    if parent: nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent).body
    nodes = [copy.deepcopy(n) for n in nodes if getattr(n, 'name', '') in names]
    assert len(nodes) == len(names)
    space = dict(torch=torch, nn=nn, F=F, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), space)
    return space

def rigid_residual(a, b):
    a, b = a.double(), b.double()
    a, b = a-a.mean(0), b-b.mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    fix = torch.eye(3, dtype=torch.float64)
    fix[-1, -1] = torch.linalg.det(u @ vh)
    error = a @ (u @ fix @ vh) - b
    return float(error.square().sum(1).mean().sqrt())

def probe_omni():
    validation = H/'exports/omnire_human100_v2/objects/HN0004/validation.json'
    v = json.loads(validation.read_text()); ck = Path(v['checkpoint'])
    assert sha(ck) == v['checkpoint_sha256']
    c = torch.load(ck, map_location='cpu', weights_only=False)
    state = c['models']['DeformableNodes']; actor = v['actor_index']
    cfg = yaml.safe_load(ck.with_name('config.yaml').read_text())['model']['DeformableNodes']
    assert cfg['ctrl']['use_deformgs_for_nonrigid'] and c['step'] > cfg['ctrl']['use_deformgs_after']
    network_cls = native_defs(D/'models/modules.py', ['get_embedder','Embedder','ConditionalDeformNetwork'])['ConditionalDeformNetwork']
    net = network_cls(input_ch=3, **cfg['networks']).eval()
    weights = {k.removeprefix('deform_network.'): x for k,x in state.items() if k.startswith('deform_network.')}
    net.load_state_dict(weights, strict=True)
    mask = state['points_ids'].flatten() == actor
    assert int(mask.sum()) == v['gaussians']
    means = state['_means'][mask].clone()
    frames = state['instances_fv'][:,actor].nonzero().flatten()
    # Deterministic visible sequence, no selection by appearance/score.
    frames = frames[torch.linspace(0,len(frames)-1,min(9,len(frames))).round().long()]
    native = native_defs(D/'models/nodes/deformable.py',['get_deformation'], 'DeformableNodes')['get_deformation']
    model = SimpleNamespace(point_ids=torch.zeros((len(means),1),dtype=torch.long),
        instances_embedding=state['instances_embedding'][actor:actor+1],
        instances_size=state['instances_size'][actor:actor+1], deform_network=net,
        normalized_timestamps=torch.linspace(0,1,len(state['instances_fv'])))
    outputs=[]
    with torch.inference_mode():
        for frame in frames.tolist():
            model.cur_frame=frame
            xyz, quat, scale=native(model,means)
            outputs.append((means+xyz,quat,scale))
    rms=[rigid_residual(outputs[0][0],x[0]) for x in outputs]
    bundle=dict(method='OmniRe non-SMPL DeformableNodes', sample_id='HN0004',
        actor_uuid=v['actor_uuid'], source_checkpoint_sha256=v['checkpoint_sha256'],
        canonical={k:state[k][mask].clone() for k in ['_means','_scales','_quats','_features_dc','_features_rest','_opacities']},
        embedding=model.instances_embedding.clone(), size=model.instances_size.clone(), network_config=cfg['networks'],
        network_state=weights, normalized_timestamps=model.normalized_timestamps,
        root_quaternions=state['instances_quats'][:,actor].clone(),
        root_translations=state['instances_trans'][:,actor].clone(),
        visibility=state['instances_fv'][:,actor].clone(),
        scope='Geometry and native state export; appearance renderer and physical-time video acceptance pending')
    path=EXP/'omnire_HN0004_native_dynamic.pt'; torch.save(bundle,path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    reloaded=network_cls(input_ch=3,**loaded['network_config']).eval();reloaded.load_state_dict(loaded['network_state'],strict=True)
    model.deform_network=reloaded;model.instances_embedding=loaded['embedding'];model.instances_size=loaded['size']
    maxerr=0.
    with torch.inference_mode():
        for frame,expected in zip(frames.tolist(),outputs):
            model.cur_frame=frame;delta=native(model,loaded['canonical']['_means'])[0]
            maxerr=max(maxerr,float((loaded['canonical']['_means']+delta-expected[0]).abs().max()))
    assert maxerr == 0 and max(rms)>1e-6
    return dict(status='native_dynamic_geometry_and_reload_passed_video_pending',sample_id='HN0004',gaussians=len(means),
        native_frames=frames.tolist(),geometry_sha256=[hashlib.sha256(x[0].numpy().tobytes()).hexdigest() for x in outputs],nonrigid_best_rigid_fit_rms_m=rms, reload_max_error=maxerr,
        root_endpoint_distance_m=float((bundle['root_translations'][frames[-1]]-bundle['root_translations'][frames[0]]).norm()),
        bundle=str(path),bundle_sha256=sha(path),source_checkpoint=str(ck),source_checkpoint_sha256=sha(ck),
        native_modules_sha256=sha(D/'models/modules.py'),native_deformation_sha256=sha(D/'models/nodes/deformable.py'),
        video_accepted=False,physical_time_synchronization_accepted=False)

def probe_splat():
    path=R/'exports/splatad_human_reviewed95/formal/scene-0917/33c5b1d0f97d4da7a603a570033bb0fb.pt'
    bundle=torch.load(path,map_location='cpu',weights_only=False); actor=bundle['actor_index']
    ck=Path(bundle['native_checkpoint']); c=torch.load(ck,map_location='cpu',weights_only=False)['pipeline']
    mask=c['_model.gauss_params.id'].flatten()==actor
    for key,val in bundle['gauss_params'].items(): assert torch.equal(val,c['_model.gauss_params.'+key][mask]),key
    prefix='_model.dynamic_actors.'
    state={k.removeprefix(prefix):v for k,v in c.items() if k.startswith(prefix)}
    valid=state['actor_present_at_time'][:,actor].nonzero().flatten()
    frames=valid[torch.linspace(0,len(valid)-1,min(9,len(valid))).round().long()]
    rotation=native_defs(S/'nerfstudio/cameras/camera_utils.py',['rotation_6d_to_matrix'])['rotation_6d_to_matrix']
    rot=rotation(state['actor_rotations_6d'][frames,actor]);trans=state['actor_positions'][frames,actor]
    means=bundle['gauss_params']['means']; clouds=[means @ r.T+t for r,t in zip(rot,trans)]
    rms=[rigid_residual(clouds[0],x) for x in clouds]
    assert max(rms)<1e-4
    # Learned actor trajectory, not the parser's unoptimized annotation poses.
    dynamic=dict(positions=state['actor_positions'][:,actor].clone(),
        rotations_6d=state['actor_rotations_6d'][:,actor].clone(),
        timestamps=state['unique_timestamps'].clone(),visibility=state['actor_present_at_time'][:,actor].clone(),
        actor_velocity_linear=state['actor_vel_linear'][:,actor].clone(),
        actor_velocity_angular=state['actor_vel_angular'][:,actor].clone())
    packed=dict(method='SplatAD',sample_id='HN0004',actor_uuid=bundle['uuid'],gauss_params=bundle['gauss_params'],
        dynamic=dynamic,source_checkpoint_sha256=sha(ck),
        appearance_contract=bundle['appearance_contract'],
        scope='Geometry/trajectory export; native learned appearance decoder required for RGB rendering')
    out=EXP/'splatad_HN0004_native_dynamic.pt';torch.save(packed,out)
    loaded=torch.load(out,map_location='cpu',weights_only=True)
    rr=rotation(loaded['dynamic']['rotations_6d'][frames]);tt=loaded['dynamic']['positions'][frames]
    errors=[float((loaded['gauss_params']['means']@r.T+t-x).abs().max()) for r,t,x in zip(rr,tt,clouds)]
    assert max(errors)==0
    return dict(status='native_rigid_trajectory_and_reload_passed_video_pending',sample_id='HN0004',gaussians=len(means),
        native_frames=frames.tolist(),geometry_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in clouds],native_timestamps=state['unique_timestamps'][frames].tolist(),
        best_rigid_fit_rms_m=rms, root_endpoint_distance_m=float((trans[-1]-trans[0]).norm()),reload_max_error=max(errors),
        dataset_deformable_flag=bool(state['actor_deformable'][actor]),
        deformable_flag_is_not_a_deformation_network=True,bundle=str(out),bundle_sha256=sha(out),
        source_checkpoint=str(ck),source_checkpoint_sha256=sha(ck),source_actor_bundle_sha256=sha(path),
        native_model_sha256=sha(S/'nerfstudio/models/splatad.py'),video_accepted=False)

def probe_street():
    ck=R/'runs/streetgs_nuscenes100_expansion/formal/nuScenes/scene-0019/checkpoint_final.pth'
    c=torch.load(ck,map_location='cpu',weights_only=False)
    models=list(c['models']);assert 'DeformableNodes' not in models and 'SMPLNodes' not in models
    return dict(status='stop_existing_checkpoint_has_no_human_node_branch',checkpoint=str(ck),
        checkpoint_sha256=sha(ck),models=models,
        category_mapping=str(D/'datasets/nuscenes/nuscenes_sourceloader.py'),
        category_mapping_sha256=sha(D/'datasets/nuscenes/nuscenes_sourceloader.py'),
        filter_source=str(D/'datasets/driving_dataset.py'),filter_source_sha256=sha(D/'datasets/driving_dataset.py'),
        reason='Pedestrian classes map to SMPL/Deformable; RigidNodes initialization excludes them. Background image crop is not an identity-indexed dynamic Gaussian asset.',
        scope='Bounded checkpoint and active loader audit, not a claim that every StreetGS variant is impossible')

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Bounded CPU native geometry feasibility, no new scene training and no final videos',script_sha256=sha(__file__))
    for name,fn in [('omnire',probe_omni),('splatad',probe_splat),('streetgs',probe_street)]:
        report[name]=fn();print(name,json.dumps(report[name]),flush=True)
        (OUT/'native_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
