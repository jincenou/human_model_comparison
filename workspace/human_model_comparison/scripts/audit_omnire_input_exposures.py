"""Recover actual OmniRe input exposures from the native processed sample chain.

Metadata can retain held-out samples. Only the scene's first/next/last chain,
verified against every processed camera image, defines the observed inputs.
"""
from pathlib import Path
from common_human_orbit import H, read, save, sha, checked

CHANNELS = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_BACK']


def main():
    source = H/'reports/input_budget_audit_v1/audit.json'
    previous = read(source); directories = {}
    for row in previous['rows']:
        if row['method'] != 'omnire' or not row.get('plan'):
            continue
        plan = read(row['plan']); item = next(r for r in plan['objects'] if r['id']==row['id'])
        if item.get('processed_data'):
            key = item['processed_data']
            directories.setdefault(key, dict(scene=row['scene'], ids=[], plan=row['plan']))['ids'].append(row['id'])
    out = H/'reports/omnire_input_exposure_audit_v1'; out.mkdir(exist_ok=False)
    records = []
    for processed, item in sorted(directories.items()):
        processed = Path(processed); base = processed.parents[2]
        data = base/'train_only_nuscenes'
        evidence = {}
        if not data.exists():
            audit_path = base/'input_audit.json'; audit = read(audit_path)
            assert Path(audit['processed_path']) == processed and audit['interpolate_N']==0
            assert audit['scene'] == item['scene']
            data = Path(audit['data_root']); evidence[str(audit_path)] = sha(audit_path)
        tables = {}
        for table in ['scene','sample','sample_data','sensor','calibrated_sensor']:
            path = data/'v1.0-trainval'/(table+'.json')
            tables[table] = read(path); evidence[str(path)] = sha(path)
        scene = next(r for r in tables['scene'] if r['name']==item['scene'])
        samples = {r['token']:r for r in tables['sample']}
        chain = []; seen = set(); token = scene['first_sample_token']
        while True:
            assert token not in seen, 'Cyclic native sample chain'
            seen.add(token); row = samples[token]
            assert row['scene_token'] == scene['token']
            chain.append(row)
            if token == scene['last_sample_token']:
                break
            assert row['next'], 'Sample chain ended before declared last sample'
            token = row['next']
        assert len(chain) == scene['nbr_samples']
        assert [r['timestamp'] for r in chain] == sorted(r['timestamp'] for r in chain)
        sensors = {r['token']:r['channel'] for r in tables['sensor']}
        channels = {r['token']:sensors[r['sensor_token']] for r in tables['calibrated_sensor']}
        exposures = {}
        for row in tables['sample_data']:
            channel = channels[row['calibrated_sensor_token']]
            if row['is_key_frame'] and row['sample_token'] in seen and channel in CHANNELS:
                key = row['sample_token'], channel
                assert key not in exposures
                exposures[key] = row
        actual = set((processed/'images').glob('*.jpg')); expected = set(); images = []
        for index, sample in enumerate(chain):
            for camera, channel in enumerate(CHANNELS):
                row = exposures[sample['token'], channel]
                target = processed/'images'/f'{index:03d}_{camera}.jpg'
                original = data/row['filename']; digest = sha(target)
                checked(original, digest)  # Native processor copies JPEGs byte-for-byte.
                expected.add(target)
                images.append(dict(frame=index, camera=camera, channel=channel,
                                   sample_token=sample['token'], sample_timestamp_us=sample['timestamp'],
                                   timestamp_us=row['timestamp'], processed_image=str(target),
                                   original_image=str(original), sha256=digest))
        assert actual == expected
        ts = [r['timestamp_us'] for r in images]
        record = dict(status='all_processed_images_matched_native_chain', scene=item['scene'],
                      ids=item['ids'], processed_data=str(processed), metadata_root=str(data),
                      cameras=CHANNELS, real_images=len(images), sample_chain_count=len(chain),
                      retained_metadata_samples=len(tables['sample']),
                      real_duration_seconds=(max(ts)-min(ts))/1e6,
                      first_image_timestamp_us=min(ts), last_image_timestamp_us=max(ts),
                      sample_chain_duration_seconds=(chain[-1]['timestamp']-chain[0]['timestamp'])/1e6,
                      duration_definition='Last minus first actual camera exposure; gaps are retained, not summed continuous observation time',
                      mapping_validation='Every processed JPEG SHA256 equals the corresponding native keyframe JPEG from the traversed first/next/last sample chain',
                      metadata_hashes=evidence, images=images)
        path = out/(item['scene']+'.json'); save(path, record)
        records.append({k:v for k,v in record.items() if k not in ['images','metadata_hashes']})
        records[-1].update(record=str(path), record_sha256=sha(path))
        print(item['scene'],len(images),record['real_duration_seconds'],flush=True)
    save(out/'manifest.json',dict(status='complete', scenes=len(records),
         identities=sum(len(r['ids']) for r in records), source_inventory=str(source),
         source_inventory_sha256=sha(source), records=records, script_sha256=sha(__file__),
         additional_scene_training=False, target_visibility_count=False))


if __name__ == '__main__':
    main()
