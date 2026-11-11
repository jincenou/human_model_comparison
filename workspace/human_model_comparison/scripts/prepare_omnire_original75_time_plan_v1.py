"""Reconcile original75 snapshots with exact exposures; never pick new checkpoints."""
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parents[1]
R = H.parent/'vehicle_model_comparison'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    selection = H/'inputs/benchmark_human100_v2/selection.json'
    time_path = H/'reports/model_time_audit_v1/omnire.json'
    times = read(time_path)
    assert times['selection_sha256'] == sha(selection)
    mapped = {x['id']: x for x in times['records']}
    rows = []
    for selected in read(selection)['objects']:
        if selected['cohort'] != 'original75':
            continue
        sid = selected['id']
        old_path = R/'exports/omnire_human95/objects'/sid/'validation.json'
        old = read(old_path)
        timing = mapped[sid]
        assert old['status'] == 'passed' and old['native_id'] == selected['identity_key']
        assert old['frozen_raw_timestamp_us'] == selected['t_ref_us']
        assert timing['status'] == 'mapped' and timing['continuous_bracket'] is not None
        config = Path(timing['config_path'])
        checkpoint = Path(old['checkpoint'])
        assert config.parent == checkpoint.parent
        assert sha(config) == timing['config_sha256']
        source = Path(old['source_image'])
        source_sha = sha(source)
        assert source_sha == selected['reference_frame']['raw_source_sha256'] == old['source_image_sha256']
        processed = Path(timing['source_image'])
        assert sha(processed) == source_sha
        data = processed.parent.parent
        infos = data/'instances/instances_info.json'
        ids = [int(k) for k, v in read(infos).items() if v['id'] == old['actor_uuid']]
        assert len(ids) == 1
        bundle = Path(old['bundle'])
        assert bundle.is_file() and checkpoint.is_file()
        rows.append(dict(id=sid, scene=selected['scene'], cohort='original75', dataset='nuscenes',
            native_id=selected['identity_key'], raw_timestamp_us=selected['t_ref_us'],
            source_image=str(source), source_sha256=source_sha,
            checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            config=str(config), config_sha256=sha(config),
            processed_data=str(data), instances_info=str(infos), instances_info_sha256=sha(infos),
            native_true_ids=ids, source_matches=[str(processed)],
            prior_snapshot=str(bundle), prior_snapshot_sha256=sha(bundle),
            prior_validation=str(old_path), prior_validation_sha256=sha(old_path),
            status='ready_for_native_decode', common_camera_valid=False, scored=False,
            selected_checkpoint_policy='Exact checkpoint of retained original75 native snapshot; no quality-based reselection.'))
    assert len(rows) == 75 and len({x['id'] for x in rows}) == 75
    out = H/'inputs/scene_extract_original75_v1'
    out.mkdir(exist_ok=True)
    plan = out/'omnire_plan.json'
    assert not plan.exists()
    plan.write_text(json.dumps(dict(selection=str(selection), selection_sha256=sha(selection),
        time_audit=str(time_path), time_audit_sha256=sha(time_path), no_scene_training=True,
        scope='Original75 only; revalidate physical exposure and raw K/T using existing scene/snapshot.',
        objects=rows), indent=2)+'\n')
    print(json.dumps(dict(plan=str(plan), ready=len(rows))))


if __name__ == '__main__':
    main()
