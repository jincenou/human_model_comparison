"""Pin the existing original75 native SplatAD assets and exact exposures."""
from pathlib import Path
import hashlib,json,time
H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    assert sha(selection)=='f6917e99f3e82d57b7ac2d141c50d8377aa953ec4ddb3e27192720d53af49f79'
    audit=json.loads((H/'reports/scene_human_reuse_v2/audit.json').read_text())
    missing_path=H/'reports/splatad_missing_original9_v1/audit.json'
    missing={r['id']:r for r in json.loads(missing_path.read_text())['records']}
    items=[]
    for row in audit['records']:
        if row['cohort']!='original75': continue
        ident=row['id'];camera_path=H/f'configs/human_source_common_v1/candidates/{ident}/camera.json'
        camera=json.loads(camera_path.read_text())
        assert camera['selection_sha256']==sha(selection)
        assert camera['timestamp_us']==row['t_ref_us'] and camera['identity_key']==row['identity_key']
        assert sha(camera['source_rgb'])==camera['source_rgb_sha256']
        item=dict(id=ident,scene=row['scene'],native_id=row['identity_key'],dataset='nuscenes',
            raw_timestamp_us=row['t_ref_us'],source_image=camera['source_rgb'],source_sha256=camera['source_rgb_sha256'],
            camera_contract=str(camera_path),camera_contract_sha256=sha(camera_path),real_heldout_eligible=False,metric_eligible=False)
        if ident in missing:
            proof=missing[ident];assert proof['status']=='unavailable_native_stationary_filter'
            item.update(status='unavailable',reason=proof['status'],evidence=str(missing_path),evidence_sha256=sha(missing_path))
        else:
            prior_path=R/f'exports/splatad_human95/objects/{ident}/validation.json'
            prior=json.loads(prior_path.read_text());assert prior['status']=='passed'
            assert prior['actor_uuid']==row['identity_key'].split(':')[-1]
            assert prior['frozen_raw_timestamp_us']==row['t_ref_us']
            assert Path(prior['source_image']).resolve()==Path(camera['source_rgb']).resolve()
            native_tag=Path(prior['bundle']).parents[2].name
            assert native_tag in ('splatad_human_reviewed95','splatad_reconstruction_only','splatad_reviewed288_train_only'),native_tag
            formal=R/f"reports/{native_tag}_formal_{row['scene']}_validation.json"
            validation=json.loads(formal.read_text());assert validation['status']=='passed' and validation['step']==30000
            assert validation['checkpoint']==prior['checkpoint']
            actor=next(a for a in validation['actors'] if a['uuid']==prior['actor_uuid'])
            assert actor['bundle']==prior['bundle'] and actor['gaussians']==prior['gaussians']
            checkpoint=Path(prior['checkpoint']);assert checkpoint.is_file()
            config=checkpoint.parent.parent/'config.yml'
            item.update(status='ready_for_native_decode',reason=None,formal_validation=str(formal),formal_validation_sha256=sha(formal),
                checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,config=str(config),config_sha256=sha(config),
                bundle=prior['bundle'],bundle_sha256=sha(prior['bundle']),gaussians=actor['gaussians'],native_variant=prior['variant'],
                prior_validation=str(prior_path),prior_validation_sha256=sha(prior_path),
                selection_policy='Exact existing original-person checkpoint and bundle. No selection by quality. Historical holdout flag not inherited.')
        items.append(item)
    assert len(items)==75 and sum(i['status']=='ready_for_native_decode' for i in items)==66
    out=H/'inputs/scene_extract_original75_v1/splatad_plan.json';assert not out.exists();out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(selection=str(selection),selection_sha256=sha(selection),created_at=time.time(),
        no_scene_training=True,objects=items,counts={'ready_for_native_decode':66,'unavailable_native_stationary_filter':9}),indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
