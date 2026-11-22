"""Map contributing non-target nodes using immutable instance size/visibility."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
from types import SimpleNamespace
import sys,json,gc
H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
sys.path.insert(0,_hmc_expand('${DRIVESTUDIO_ROOT}'))
import numpy as np
import torch
from omegaconf import OmegaConf
from datasets.nuscenes.nuscenes_sourceloader import NuScenesPixelSource
from datasets.base.scene_dataset import ModelType
R=H/'reports/native_human_export_reaudit_v1/omnire'
E=H/'exports/native_human_export_reaudit_v1/omnire'
torch.set_num_threads(4)
plan=json.loads((R/'plan.json').read_text())
records=[]
for group in plan['groups']:
    objects=[o for o in group['objects'] if json.loads((E/o['id']/'validation.json').read_text())['other_node_contributions']]
    if not objects:continue
    first=objects[0]['plan'];cfg=OmegaConf.load(first['config']);checkpoint=torch.load(group['checkpoint'],map_location='cpu',weights_only=False)
    n=checkpoint['models']['DeformableNodes']['instances_fv'].shape[0]
    pix=SimpleNamespace(data_path=first['processed_data'],start_timestep=int(cfg.data.start_timestep),end_timestep=int(cfg.data.start_timestep)+n,data_cfg=SimpleNamespace(load_smpl=False))
    NuScenesPixelSource.load_objects(pix)
    infos=json.loads(Path(first['instances_info']).read_text());mapping={}
    for kind in ['RigidNodes','DeformableNodes']:
        state=checkpoint['models'][kind]
        for i,size in enumerate(state['instances_size']):
            fv=state['instances_fv'][:,i];matches=[]
            for j,true in enumerate(pix.instances_true_id.tolist()):
                typ=int(pix.instances_model_types[j]);allowed=typ==int(ModelType.RigidNodes) if kind=='RigidNodes' else typ in [int(ModelType.DeformableNodes),int(ModelType.SMPLNodes)]
                if allowed and torch.equal(fv,pix.per_frame_instance_mask[:,j]) and torch.allclose(size,pix.instances_size[j],atol=1e-5,rtol=0):
                    matches.append(dict(true_id=true,uuid=infos[str(true)]['id'],category=infos[str(true)]['class_name'],dataset_index=j))
            mapping[(kind,i)]=matches
    for obj in objects:
        sid=obj['id'];v=json.loads((E/sid/'validation.json').read_text());camera=json.loads(Path(v['camera_contract']).read_text());world_from_native=np.asarray(v['T_world_from_native']);world_from_camera=np.asarray(camera['T_world_from_camera_opencv']);native_to_camera=np.linalg.inv(world_from_camera)@world_from_native;K=np.asarray(camera['K_raw']);frame=v['source_frame'];tm=obj['time'];br=tm['continuous_bracket'];details=[]
        for row in v['other_node_contributions']:
            node=row['node'];i=row['instance_index'];state=checkpoint['models'][node];matches=mapping[(node,i)];anchor=state['instances_trans'][frame,i].numpy();xyz=native_to_camera[:3,:3]@anchor+native_to_camera[:3,3];uv=K@xyz;uv=uv[:2]/uv[2];motion=None
            lo,hi=br['lo'],br['hi'];fraction=br['fraction']
            if state['instances_fv'][lo,i] and state['instances_fv'][hi,i]:
                interp=(1-fraction)*state['instances_trans'][lo,i].numpy()+fraction*state['instances_trans'][hi,i].numpy();p=native_to_camera[:3,:3]@interp+native_to_camera[:3,3];p=K@p;uv2=p[:2]/p[2]
                motion=dict(root_translation_shift_m=float(np.linalg.norm(interp-anchor)),projected_root_shift_px=float(np.linalg.norm(uv2-uv)),source_grid_offset_us=tm['grid_minus_exposure_us'])
            details.append(dict(**row,identity_candidates=matches,identity_status='unique_immutable_size_visibility_match' if len(matches)==1 else 'ambiguous_needs_original_init',same_uuid_as_target=len(matches)==1 and matches[0]['uuid']==obj['plan']['native_id'].split(':')[-1],source_camera_root_uv=uv.tolist(),source_camera_root_depth_m=float(xyz[2]),source_exposure_motion=motion))
        records.append(dict(id=sid,target_uuid=obj['plan']['native_id'].split(':')[-1],source_reference=v['source_reference'],native_checkpoint=group['checkpoint'],nodes=details))
    del checkpoint;gc.collect()
    (R/'other_node_identity_progress.json').write_text(json.dumps({'objects':len(records),'last_scene':first['scene']})+'\n')
result=dict(status='identity_audit_complete',records=records,policy='CPU native load_objects and exact instances_size/instances_fv matching; ambiguous matches retained as ambiguous, never guessed; pixel shift is learned root translation diagnostic, not rendered-image proof')
(R/'other_node_identity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'objects':len(records),'nodes':sum(len(r['nodes']) for r in records),'ambiguous':sum(n['identity_status']!='unique_immutable_size_visibility_match' for r in records for n in r['nodes'])}),flush=True)
