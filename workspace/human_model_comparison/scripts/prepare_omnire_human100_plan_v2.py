"""Resolve frozen identity and exact exposure against existing scenes, without training."""
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf

H=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    rows=json.loads(selection.read_text())['objects']
    reuse=H/'reports/scene_human_reuse_v2/audit.json'
    inventory={x['id']:x for x in json.loads(reuse.read_text())['records']}
    records=[]
    for row in rows:
        if row['cohort']!='new25':continue
        candidates=inventory[row['id']]['methods']['omnire']['scene_candidates']
        valid=[c for c in candidates if c['checkpoint_exists'] and c['config_exists']]
        result=dict(id=row['id'],scene=row['scene'],native_id=row['identity_key'],dataset='nuscenes',
                    raw_timestamp_us=row['t_ref_us'],source_image=row['reference_frame']['raw_source_path'],
                    source_sha256=row['reference_frame']['raw_source_sha256'])
        assert valid, (row['id'],'No existing scene checkpoint')
        # Audit order prioritizes established human/non-SMPL scenes, then expansion.
        chosen=valid[0];config_path=Path(chosen['config']);cfg=OmegaConf.load(config_path)
        assert cfg.data.pixel_source.load_smpl is False
        assert 'DeformableNodes' in cfg.model
        data=Path(cfg.data.data_root)/f'{int(cfg.data.scene_idx):03d}'
        infos=json.loads((data/'instances/instances_info.json').read_text())
        true_ids=[int(k) for k,v in infos.items() if v['id']==row['identity_key'].split(':')[-1]]
        source=Path(result['source_image']);assert sha(source)==result['source_sha256']
        matches=[p for p in (data/'images').glob('*.jpg') if p.stat().st_size==source.stat().st_size and sha(p)==result['source_sha256']]
        assert len(true_ids)<=1 and len(matches)<=1
        result.update(checkpoint=chosen['checkpoint'],checkpoint_bytes=Path(chosen['checkpoint']).stat().st_size,
            config=str(config_path),config_sha256=sha(config_path),queue_source=chosen['source'],
            instances_info=str(data/'instances/instances_info.json'),instances_info_sha256=sha(data/'instances/instances_info.json'),
            processed_data=str(data),native_true_ids=true_ids,source_matches=[str(p) for p in matches],
            selected_checkpoint_policy='First existing human non-SMPL checkpoint in audit order; no result quality selection',
            status='ready_for_native_decode' if true_ids and matches else 'unavailable',
            reason=None if true_ids and matches else 'native_identity_absent' if not true_ids else 'frozen_real_exposure_absent',
            common_camera_valid=False,scored=False)
        records.append(result)
    out=H/'inputs/scene_extract_v2';out.mkdir(exist_ok=True)
    target=out/'omnire_plan.json';assert not target.exists()
    target.write_text(json.dumps(dict(selection=str(selection),selection_sha256=sha(selection),reuse_audit_sha256=sha(reuse),
                                     no_scene_training=True,objects=records),indent=2)+'\n')
    print(json.dumps(dict(ready=sum(r['status']=='ready_for_native_decode' for r in records),
                          unavailable=[(r['id'],r['reason']) for r in records if r['status']=='unavailable'],plan=str(target))))


if __name__=='__main__':main()
