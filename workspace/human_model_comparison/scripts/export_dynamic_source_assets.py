"""Export native human deformation/trajectories from existing scene checkpoints.

No scene training, actor reposing, geometry simplification or appearance selection.
Uses the same native CPU definitions as the accepted HN0004 dynamic pilot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from audit_dynamic_human_feasibility_v1 import H,D,S,EXP,sha,native_defs,rigid_residual
from trellis_source_batch import selection,checked,read,save

REPORT=H/'reports/dynamic_source_clip_v1/native_exports'

def plan(method,sid):
    active,selected=selection();assert sid in selected
    p=H/'inputs/scene_extract_v2'/(method+'_plan.json')
    item=next((r for r in read(p)['objects'] if r['id']==sid),None)
    if item is None:
        p=H/'inputs/scene_extract_original75_v1'/(method+'_plan.json')
        item=next(r for r in read(p)['objects'] if r['id']==sid)
    assert item['native_id']==selected[sid]['identity_key'] and item['raw_timestamp_us']==selected[sid]['t_ref_us']
    checked(item['source_image'],item['source_sha256']);checked(item['config'],item['config_sha256'])
    return active,item

def geometry_hashes(clouds):
    return [hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in clouds]

def save_bundle(method,sid,payload):
    path=EXP/f'{method}_{sid}_native_dynamic.pt';assert not path.exists(),'Existing dynamic export must be reused or explicitly audited'
    assert not path.with_suffix('.partial').exists()
    torch.save(payload,path.with_suffix('.partial'));path.with_suffix('.partial').replace(path)
    return path,torch.load(path,map_location='cpu',weights_only=True)

def omnire(sid):
    active,item=plan('omnire',sid)
    vp=H/'exports/omnire_human100_v2/objects'/sid/'validation.json'
    if not vp.exists():vp=H/'exports/common_camera_human_original75_time_v1/omnire'/sid/'validation.json'
    v=read(vp)
    checkpoint=Path(v['checkpoint']);assert checkpoint==Path(item['checkpoint']);checked(checkpoint,v['checkpoint_sha256'])
    assert v['actor_uuid']==item['native_id'].split(':')[-1]
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=source['models']['DeformableNodes'];actor=v['actor_index'];mask=state['points_ids'].flatten()==actor
    assert int(mask.sum())==v['gaussians']>0
    cfg=yaml.safe_load(Path(item['config']).read_text())['model']['DeformableNodes']
    assert cfg['ctrl']['use_deformgs_for_nonrigid'] and source['step']>cfg['ctrl']['use_deformgs_after']
    cls=native_defs(D/'models/modules.py',['get_embedder','Embedder','ConditionalDeformNetwork'])['ConditionalDeformNetwork']
    deform=native_defs(D/'models/nodes/deformable.py',['get_deformation'],'DeformableNodes')['get_deformation']
    weights={k.removeprefix('deform_network.'):x for k,x in state.items() if k.startswith('deform_network.')}
    net=cls(input_ch=3,**cfg['networks']).eval();net.load_state_dict(weights,strict=True)
    frames=state['instances_fv'][:,actor].nonzero().flatten();assert len(frames)>1
    frames=frames[torch.linspace(0,len(frames)-1,min(9,len(frames))).round().long()]
    packed=dict(method='OmniRe non-SMPL DeformableNodes',sample_id=sid,actor_uuid=v['actor_uuid'],
        source_checkpoint_sha256=v['checkpoint_sha256'],canonical={k:state[k][mask].clone() for k in ['_means','_scales','_quats','_features_dc','_features_rest','_opacities']},
        embedding=state['instances_embedding'][actor:actor+1].clone(),size=state['instances_size'][actor:actor+1].clone(),
        network_config=cfg['networks'],network_state=weights,normalized_timestamps=torch.linspace(0,1,len(state['instances_fv'])),
        root_quaternions=state['instances_quats'][:,actor].clone(),root_translations=state['instances_trans'][:,actor].clone(),visibility=state['instances_fv'][:,actor].clone(),
        scope='Native geometry and learned-root state; physical exposure and RGB checked by downstream clip renderer')
    def evaluate(p):
        model=cls(input_ch=3,**p['network_config']).eval();model.load_state_dict(p['network_state'],strict=True)
        means=p['canonical']['_means'];obj=SimpleNamespace(point_ids=torch.zeros((len(means),1),dtype=torch.long),instances_embedding=p['embedding'],instances_size=p['size'],deform_network=model,normalized_timestamps=p['normalized_timestamps'])
        clouds=[]
        with torch.inference_mode():
            for frame in frames.tolist():obj.cur_frame=frame;clouds.append(means+deform(obj,means)[0])
        return clouds
    expected=evaluate(packed);path,loaded=save_bundle('omnire',sid,packed);actual=evaluate(loaded)
    assert geometry_hashes(actual)==geometry_hashes(expected)
    return dict(status='native_geometry_and_reload_passed_rgb_pending',id=sid,method='omnire',identity_key=item['native_id'],selection_sha256=active['selection_sha256'],
        bundle=str(path),bundle_sha256=sha(path),source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),source_extraction=str(vp),source_extraction_sha256=sha(vp),
        native_frames=frames.tolist(),geometry_sha256=geometry_hashes(expected),nonrigid_best_rigid_fit_rms_m=[rigid_residual(expected[0],x) for x in expected],
        native_modules_sha256=sha(D/'models/modules.py'),native_deformation_sha256=sha(D/'models/nodes/deformable.py'),
        native_shape_quality_not_a_gate=True,video_accepted=False)

def splatad(sid):
    active,item=plan('splatad',sid);checked(item['bundle'],item['bundle_sha256'])
    bundle=torch.load(item['bundle'],map_location='cpu',weights_only=False);checkpoint=Path(item['checkpoint'])
    assert checkpoint==Path(bundle['native_checkpoint']) and bundle['uuid']==item['native_id'].split(':')[-1]
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)['pipeline'];actor=bundle['actor_index'];mask=c['_model.gauss_params.id'].flatten()==actor
    for key,value in bundle['gauss_params'].items():assert torch.equal(value,c['_model.gauss_params.'+key][mask]),key
    prefix='_model.dynamic_actors.';state={k.removeprefix(prefix):v for k,v in c.items() if k.startswith(prefix)}
    visible=state['actor_present_at_time'][:,actor].nonzero().flatten();assert len(visible)>1
    frames=visible[torch.linspace(0,len(visible)-1,min(9,len(visible))).round().long()]
    dynamic=dict(positions=state['actor_positions'][:,actor].clone(),rotations_6d=state['actor_rotations_6d'][:,actor].clone(),timestamps=state['unique_timestamps'].clone(),visibility=state['actor_present_at_time'][:,actor].clone(),actor_velocity_linear=state['actor_vel_linear'][:,actor].clone(),actor_velocity_angular=state['actor_vel_angular'][:,actor].clone())
    packed=dict(method='SplatAD',sample_id=sid,actor_uuid=bundle['uuid'],gauss_params=bundle['gauss_params'],dynamic=dynamic,source_checkpoint_sha256=sha(checkpoint),appearance_contract=bundle['appearance_contract'],scope='Native rigid learned actor trajectory and all Gaussian features; original appearance decoder required for RGB')
    rotate=native_defs(S/'nerfstudio/cameras/camera_utils.py',['rotation_6d_to_matrix'])['rotation_6d_to_matrix']
    def evaluate(p):
        rr=rotate(p['dynamic']['rotations_6d'][frames]);tt=p['dynamic']['positions'][frames]
        return [p['gauss_params']['means']@r.T+t for r,t in zip(rr,tt)]
    expected=evaluate(packed);path,loaded=save_bundle('splatad',sid,packed);actual=evaluate(loaded)
    assert geometry_hashes(actual)==geometry_hashes(expected)
    return dict(status='native_rigid_trajectory_and_reload_passed_rgb_pending',id=sid,method='splatad',identity_key=item['native_id'],selection_sha256=active['selection_sha256'],
        bundle=str(path),bundle_sha256=sha(path),source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),source_actor_bundle=item['bundle'],source_actor_bundle_sha256=item['bundle_sha256'],
        native_frames=frames.tolist(),geometry_sha256=geometry_hashes(expected),best_rigid_fit_rms_m=[rigid_residual(expected[0],x) for x in expected],
        native_model_sha256=sha(S/'nerfstudio/models/splatad.py'),video_accepted=False,rigid_motion_is_valid_baseline=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',nargs='+',required=True);args=parser.parse_args();torch.set_num_threads(4)
    EXP.mkdir(exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    for sid in args.ids:
        for method,fn in [('omnire',omnire),('splatad',splatad)]:
            report=REPORT/f'{sid}_{method}.json'
            if report.exists():
                prior=read(report);checked(prior['bundle'],prior['bundle_sha256']);print(sid,method,'existing verified export',flush=True);continue
            result=fn(sid);result.update(export_script_sha256=sha(__file__),cpu_only=True);save(report,result)
            print(sid,method,result['status'],flush=True)

if __name__=='__main__':main()
