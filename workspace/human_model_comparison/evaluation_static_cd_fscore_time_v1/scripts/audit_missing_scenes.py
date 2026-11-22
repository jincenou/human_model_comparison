"""One bounded native parser/checkpoint/export pass for every absent scene actor."""
import sys,time,gc
import numpy as np
from common import *
sys.path[:0]=[str(R/'adapters'),str(R/'scripts')]
import torch,yaml
from train_splatad_reviewed_scene import MonitoredTrainer
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDatabase,ALLOWED_DEFORMABLE_CLASSES,ALLOWED_RIGID_CLASSES,_is_label_allowed
import nerfstudio.data.dataparsers.nuscenes_dataparser as parser_source
torch.set_num_threads(2)
_,selection_rows=selection();inventory=[]
for p in sorted((R/'reports').glob('splatad*validation.json')):
    d=read(p)
    if d.get('status')=='passed' and d.get('stage')=='formal' and d.get('actors') is not None and Path(d.get('checkpoint','/no')).is_file():inventory.append((p,d))
records=[]
for job in read(E/'intake.json')['rows']:
    if job['asset_available'] or job['method'] not in ['splatad','omnire']:continue
    sid=job['id'];row=selection_rows[sid];uuid=job['identity_key'].split(':')[-1];scene=row.get('scene') or row.get('scene_name')
    if not scene:
        for p in H.glob('inputs/scene_extract*/*plan.json'):
            match=[r for r in read(p)['objects'] if r['id']==sid]
            if match:scene=match[0]['scene'];break
    result=dict(id=sid,method=job['method'],identity_key=job['identity_key'],t_ref_us=job['t_ref_us'],scene=scene,prior_reason=job['prior_2d_reason'],no_scene_training=True,no_filter_override=True)
    if job['method']=='splatad':
        choices=[(p,d) for p,d in inventory if d.get('scene')==scene];assert choices,(sid,scene)
        matches=[dict(validation=str(p),actor=a) for p,d in choices for a in d['actors'] if a['uuid']==uuid]
        result['existing_export_inventory']=[dict(path=str(p),sha256=sha(p),checkpoint=d['checkpoint'],actor_count=len(d['actors'])) for p,d in choices]
        result['matching_exported_actors']=matches
        # First unchanged formal checkpoint in deterministic inventory order.
        p,d=choices[0];config=Path(d['checkpoint']).parent.parent/'config.yml';cfg=yaml.load(config.read_text(),Loader=yaml.Loader).pipeline.datamanager.dataparser
        parser=cfg.setup();parser.nusc=NuScenesDatabase(version=cfg.version,dataroot=str(cfg.data),verbose=False);parser.scene=parser.nusc.get('scene',parser.nusc.field2token('scene','name',cfg.sequence)[0])
        raw={};convert=parser._traj_dict_to_list
        def capture(trajectories):raw.update(trajectories);return convert(trajectories)
        parser._traj_dict_to_list=capture;trajectories=parser._get_actor_trajectories();indices=[i for i,t in enumerate(trajectories) if t['uuid']==uuid]
        ck=torch.load(d['checkpoint'],map_location='cpu',weights_only=False);cp=ck['pipeline'];gauss_ids=cp['_model.gauss_params.id'].reshape(-1);native_actor_count=int(cp['_model.dynamic_actors.actor_sizes'].shape[0])
        result.update(config=str(config),config_sha256=sha(config),checkpoint_sha256=sha(d['checkpoint']),native_parser_sha256=sha(parser_source.__file__),native_initial_target_indices=indices,native_checkpoint_actor_count=native_actor_count,background_has_identity_labels=False,include_deformable_actors=cfg.include_deformable_actors,initial_parser_actor_count=len(trajectories),native_annotation_instance_present=uuid in raw)
        result['annotation_tables_sha256']={n:sha(Path(cfg.data)/cfg.version/(n+'.json')) for n in ['sample','sample_annotation','instance','category','scene']}
        if uuid in raw:
            entries=raw[uuid];positions=torch.tensor(np.stack([e['pose'] for e in entries]).astype(np.float32))[:,:2,3];std=positions.std(0);allowed=set(ALLOWED_RIGID_CLASSES)|(set(ALLOWED_DEFORMABLE_CLASSES) if cfg.include_deformable_actors else set())
            result.update(xy_std_m=std.tolist(),native_stationary_filter=not bool((std>.5).any()),label=entries[0]['label'],label_allowed=bool(_is_label_allowed(entries[0]['label'],allowed)),annotation_samples=len(entries))
        if matches or indices:result.update(status='potential_native_recovery_needs_mapping',reason='Target found; do not finalize as NA until mapping/extraction checked')
        elif result.get('native_stationary_filter'):result.update(status='NA_native_stationary_filter',reason='Unchanged parser excludes this target before independent actor initialization; no matching identity export in any existing audited formal checkpoint.')
        elif not result.get('label_allowed',True):result.update(status='NA_native_label_or_deformable_filter',reason='Unchanged scene parser excludes target category; no initialized independent native actor.')
        else:result.update(status='NA_missing_native_identity',reason='Exact target not initialized by retained native parser; no matching export.')
        result['annotation_assisted_region_extraction']='Not admitted: shared background Gaussian IDs do not establish person identity or temporal instance isolation; an OBB crop alone cannot distinguish target surface from background/floor/neighbours.'
        del parser,raw,trajectories,ck,cp,gauss_ids;gc.collect()
    else:
        p=H/'inputs/scene_extract_v2/omnire_plan.json';item=next(i for i in read(p)['objects'] if i['id']==sid);info=read(item['instances_info']);ck=torch.load(item['checkpoint'],map_location='cpu',weights_only=False);models=ck['models'];found={}
        for name,state in models.items():
            if isinstance(state,dict):
                vals={k:v.tolist() for k,v in state.items() if isinstance(v,torch.Tensor) and ('true' in k or 'instance' in k) and v.numel()<10000}
                if vals:found[name]=vals
        result.update(plan_sha256=sha(p),checkpoint=item['checkpoint'],checkpoint_sha256=sha(item['checkpoint']),instance_info_sha256=sha(item['instances_info']),native_true_ids=item['native_true_ids'],checkpoint_instance_mappings=found,instance_annotations={k:v for k,v in info.items() if str(k) in set(map(str,item['native_true_ids']))},status='pending_checkpoint_mapping_review')
        del ck,models;gc.collect()
    records.append(result);save(E/'validation/missing_scene_audit.json',dict(status='running',script_sha256=sha(__file__),records=records));print(sid,job['method'],result['status'],flush=True)
save(E/'validation/missing_scene_audit.json',dict(status='complete_pending_review',script_sha256=sha(__file__),records=records,finished_at=time.time()))
