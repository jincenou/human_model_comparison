"""Audit original missing actors through the unchanged native trajectory filter, CPU only."""
from pathlib import Path
import hashlib, json, sys, time
H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
sys.path[:0]=[str(R/'adapters'),str(R/'scripts')]
import numpy as np
import torch, yaml
from train_splatad_reviewed_scene import MonitoredTrainer
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDatabase, ALLOWED_DEFORMABLE_CLASSES, ALLOWED_RIGID_CLASSES, _is_label_allowed
import nerfstudio.data.dataparsers.nuscenes_dataparser as native

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    out=H/'reports/splatad_missing_original9_v1';out.mkdir(exist_ok=False)
    audit=H/'reports/scene_human_reuse_v2/audit.json'
    rows=json.loads(audit.read_text())['records']
    ids=['HN0038','HN0131','HN0136','HN0199','HN0263','HN0290','HN0311','HN0317','HN0668']
    result=dict(status='running',started_at=time.time(),script_sha256=sha(__file__),audit_sha256=sha(audit),
                native_parser=str(Path(native.__file__)),native_parser_sha256=sha(native.__file__),
                no_scene_training=True,records=[])
    def save():
        p=out/'audit.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2)+'\n');t.replace(p)
    save()
    for ident in ids:
        row=next(r for r in rows if r['id']==ident);uuid=row['identity_key'].split(':')[-1]
        formal=R/f"reports/splatad_human_reviewed95_formal_{row['scene']}_validation.json"
        val=json.loads(formal.read_text());assert val['status']=='passed'
        assert all(a['uuid']!=uuid for a in val['actors'])
        config=Path(val['checkpoint']).parent.parent/'config.yml'
        cfg=yaml.load(config.read_text(),Loader=yaml.Loader).pipeline.datamanager.dataparser
        assert cfg.include_deformable_actors and cfg.sequence==row['scene']
        parser=cfg.setup()
        parser.nusc=NuScenesDatabase(version=cfg.version,dataroot=str(cfg.data),verbose=False)
        parser.scene=parser.nusc.get('scene',parser.nusc.field2token('scene','name',cfg.sequence)[0])
        raw={};original=parser._traj_dict_to_list
        def capture(trajectories):
            raw.update(trajectories)
            return original(trajectories)
        parser._traj_dict_to_list=capture
        trajectories=parser._get_actor_trajectories()
        record=dict(id=ident,scene=row['scene'],identity_key=row['identity_key'],config=str(config),config_sha256=sha(config),
                    formal_validation=str(formal),formal_validation_sha256=sha(formal),checkpoint=val['checkpoint'],
                    native_annotation_instance_present=uuid in raw,native_initial_trajectory_present=any(t['uuid']==uuid for t in trajectories),
                    exported_nonzero_actor_present=False,independent_actor_recoverable=False,metric_eligible=False)
        tables=Path(cfg.data)/cfg.version
        record['native_table_sha256']={n:sha(tables/(n+'.json')) for n in ('sample','sample_annotation','instance','category','scene')}
        if uuid in raw:
            entries=raw[uuid]
            positions=torch.from_numpy(np.stack([e['pose'] for e in entries]).astype(np.float32))[:,:2,3]
            std=positions.std(dim=0)
            allowed=set(ALLOWED_RIGID_CLASSES)|set(ALLOWED_DEFORMABLE_CLASSES)
            stationary=not bool((std>.50).any())
            label_allowed=_is_label_allowed(entries[0]['label'],allowed)
            record.update(native_annotation_samples=len(entries),label=entries[0]['label'],xy_std_m=std.tolist(),
                          native_stationary_filter=stationary,native_label_allowed=bool(label_allowed))
            if stationary and not record['native_initial_trajectory_present']:
                record.update(status='unavailable_native_stationary_filter',reason='Native parser excludes trajectories whose float32 XY position std does not exceed 0.50m in either axis. No independent native actor was initialized; do not retrain or substitute another identity.')
            elif not label_allowed and not record['native_initial_trajectory_present']:
                record.update(status='unavailable_native_label_filter',reason='Native parser label filter excludes target.')
            else:
                record.update(status='later_native_filter_or_zero_gaussians_pending',reason='Initial trajectory is present; full parser/checkpoint inspection remains necessary.')
                record['independent_actor_recoverable']=None
        else:
            record.update(status='unavailable_absent_native_annotation',reason='Target instance absent from exact training annotation tables.')
        result['records'].append(record);save();print(ident,record['status'],flush=True)
        del parser,raw,trajectories
    from collections import Counter
    result.update(status='finished',finished_at=time.time(),counts=dict(Counter(r['status'] for r in result['records'])));save()

if __name__=='__main__': main()
