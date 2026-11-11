"""Audit frozen75 and candidate44 without changing selection, inputs or assets."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VEHICLE = ROOT.parent / 'vehicle_model_comparison'
BATCH = ROOT.parent / 'asset_work/human_nuscenes100_20261026'
LIBRARY = ROOT.parent / 'asset_library/human'
OUT = ROOT / 'reports/nuscenes_cohort_audit_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                magic = stream.read(8)
                if magic[:6] != b'\x93NUMPY':
                    raise ValueError('Non-NPY member: ' + info.filename)
                width = 2 if magic[6] == 1 else 4
                length = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
                if length > 65536:
                    raise ValueError('Unexpectedly large NPY header')
                result[info.filename.removesuffix('.npy')] = ast.literal_eval(stream.read(length).decode())
    return result


def candidate_paths(ident):
    clip = BATCH / 'inputs' / ident / 'clip0'
    return {k: str(v) for k, v in {
        'native_metadata': clip / 'native.json',
        'frame_mapping': clip / 'inputs/frame_mapping.json',
        'source_video': clip / 'observed_native.mp4',
        'input_video': clip / 'inputs/main.mp4',
        'mask_npz': clip / 'segmentation/target_masks.npz',
        'asset_preview': LIBRARY / 'previews' / f'{ident}.mp4',
        'runtime': LIBRARY / ident / 'runtime.npz',
        'asset_metadata': LIBRARY / ident / 'asset.json',
        'production_config': BATCH / 'config' / f'{ident}.json',
    }.items()}


def main():
    sources = {
        'frozen100': VEHICLE / 'inputs/benchmark_nuscenes100_v1/selection.json',
        'prior_selection': VEHICLE / 'inputs/benchmark_nuscenes_20261109/selection.json',
        'frozen_times': VEHICLE / 'inputs/benchmark_20261106/human_snapshot_manifest_v1.json',
        'candidate44': VEHICLE / 'handoff/human_experiment_research_20261110/candidate_inventory_readonly.json',
        'historical_decisions': ROOT / 'inputs/manual_decisions.json',
    }
    source_hashes = {k: sha(v) for k, v in sources.items()}
    selection = read(sources['frozen100'])
    prior = read(sources['prior_selection'])
    times = {r['id']: r for r in read(sources['frozen_times'])['objects'] if r['dataset'] == 'nuscenes'}
    decisions = read(sources['historical_decisions'])
    frozen = selection['existing_humans']
    candidates = read(sources['candidate44'])['records']
    scenes = {r['scene']: r for r in selection['scenes']}
    frozen_keys = {r['identity_key'] for r in frozen}
    explicit_keys = {r['identity_key'] for r in prior['human_excluded']}
    assert len(frozen) == len(frozen_keys) == 75
    assert len(scenes) == 100
    assert frozen_keys == {r['identity_key'] for r in prior['humans']}
    assert not frozen_keys & explicit_keys
    assert len(candidates) == len({r['identity_key'] for r in candidates}) == 44
    assert not frozen_keys & {r['identity_key'] for r in candidates}
    split_cache = {}
    rows = []
    for cohort, records in [('original75', frozen), ('candidate44', candidates)]:
        for row in records:
            ident = row['id']
            out = {k: row[k] for k in ('id', 'scene', 'identity_key')}
            out.update(cohort=cohort, dataset='nuscenes', issues=[], selected=cohort == 'original75',
                       visual_reference_review='pending', runtime_reload_valid=False,
                       candidate_approval='not_requested' if cohort == 'candidate44' else 'previously_frozen')
            paths = row.get('paths', candidate_paths(ident))
            out['paths'] = paths
            out['file_presence'] = {k: Path(v).is_file() for k, v in paths.items()}
            out['historical_decision'] = decisions.get(ident)
            out['blocked_by_prior_exclusion'] = (row['identity_key'] in explicit_keys or
                str(decisions.get(ident, '')).startswith('exclude:'))
            if out['blocked_by_prior_exclusion']:
                out['issues'].append('prior_exclusion_preserved')
            if row['scene'] not in scenes:
                out['issues'].append('scene_outside_frozen100')
            out['issues'] += ['missing_' + k for k, ok in out['file_presence'].items() if not ok]
            try:
                native = read(paths['native_metadata'])
                frames = native['frames']
                mapping = read(paths['frame_mapping'])
                out['provenance_hashes'] = {k: sha(paths[k]) for k in (
                    'native_metadata', 'frame_mapping', 'asset_metadata', 'production_config')}
                out['native_identity_matches'] = (native['scene'] == row['scene'] and
                    'nuscenes:' + native['track_id'] == row['identity_key'] and native['dataset'] == 'nuscenes')
                if not out['native_identity_matches']:
                    out['issues'].append('native_identity_mismatch')
                out['real_exposures'] = len(frames)
                out['model_input_slots'] = len(mapping['frames'])
                out['interpolated_slots'] = sum(bool(f.get('interpolated')) for f in mapping['frames'])
                out['native_duration_s'] = (frames[-1]['frame_timestamp_micros'] - frames[0]['frame_timestamp_micros']) / 1e6
                out['input_duration_s'] = mapping['frames'][-1]['video_seconds']
                out['native_source_paths_present'] = sum(Path(f['raw_source_path']).is_file() for f in frames)
                if out['native_source_paths_present'] != len(frames):
                    out['issues'].append('missing_native_source_frames')
                if row['scene'] not in split_cache:
                    s = scenes[row['scene']]
                    split_cache[row['scene']] = (s, read(s['split']), sha(s['split']))
                scene, split, split_hash = split_cache[row['scene']]
                out['split'] = {'path': scene['split'], 'sha256': split_hash,
                                'matches_frozen_hash': split_hash == scene['split_sha256']}
                if not out['split']['matches_frozen_hash']:
                    out['issues'].append('frozen_split_hash_mismatch')
                train_keys = {f['sample_data_token'] for f in split['files']
                              if f.get('is_key_frame') and f.get('split') == 'train'}
                if cohort == 'original75':
                    ref = times[ident]
                    target = ref['native_frame']
                    out['t_ref_status'] = 'original_frozen_unchanged'
                    out['t_ref_us'] = ref['raw_timestamp_us']
                    out['original_time_record'] = ref
                    if target not in frames:
                        out['issues'].append('frozen_native_record_differs')
                else:
                    eligible = [f for f in frames if f.get('sample_data_token') in train_keys
                                and not f.get('interpolated') and not f.get('enhanced')]
                    middle = statistics.median(f['frame_timestamp_micros'] for f in frames)
                    target = min(eligible, key=lambda f: (abs(f['frame_timestamp_micros'] - middle),
                                                        f['frame_timestamp_micros'])) if eligible else None
                    out['t_ref_status'] = 'proposed_not_frozen' if target else 'no_retained_train_keyframe'
                    out['t_ref_us'] = target['frame_timestamp_micros'] if target else None
                    if target is None:
                        out['issues'].append('no_retained_train_keyframe')
                if target:
                    out['reference_frame'] = target
                    out['reference_raw_sha256'] = sha(target['raw_source_path'])
                    out['reference_crop_sha256'] = sha(target['path'])
                    out['reference_bytes_match_record'] = (
                        out['reference_raw_sha256'] == target['raw_source_sha256'] and
                        out['reference_crop_sha256'] == target['sha256'])
                    if not out['reference_bytes_match_record']:
                        out['issues'].append('reference_bytes_changed')
                    out['reference_is_train_keyframe'] = target.get('sample_data_token') in train_keys
                    out['reference_seconds_from_first_native'] = (out['t_ref_us'] - frames[0]['frame_timestamp_micros']) / 1e6
                    out['exact_input_slots_for_reference'] = [f['slot'] for f in mapping['frames']
                        if not f.get('interpolated') and f.get('left_native_index') == target['native_index']
                        and f.get('right_native_index') == target['native_index']]
                headers = npz_headers(paths['runtime'])
                out['runtime_headers'] = headers
                out['runtime_bytes'] = Path(paths['runtime']).stat().st_size
                needed = {'frame_xyz','frame_quaternions','frame_scales','opacities','sh',
                          'timestamps','root_xy','root_yaw','asset_to_standard','ground_offset'}
                out['runtime_required_keys_present'] = needed <= headers.keys()
                if not out['runtime_required_keys_present']:
                    out['issues'].append('runtime_required_keys_missing')
                asset = read(paths['asset_metadata'])
                config = read(paths['production_config'])
                out['asset_provenance'] = {'metadata': asset,
                    'upstream_commit': config.get('upstream_commit'),
                    'configured_iterations': config.get('optimization', {}).get('iterations'),
                    'invariants': config.get('invariants'), 'config_provenance': config.get('provenance')}
            except Exception as exc:
                out['issues'].append('audit_error:' + repr(exc))
            rows.append(out)
            print(ident, cohort, out['issues'], flush=True)
    # Read-only audit must not change authoritative source manifests.
    assert source_hashes == {k: sha(v) for k, v in sources.items()}
    report = {'schema':'human_nuscenes_cohort_audit_v2', 'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'status':'audit_only_no_new_selection_frozen',
        'source_files':{k:{'path':str(v),'sha256':source_hashes[k]} for k,v in sources.items()},
        'time_rule':'Nearest retained train keyframe to median native exposure; earlier timestamp breaks ties.',
        'scope':{'frozen_scenes':100,'original_people':75,'additional_candidates':44,'target_additions':25},
        'limitations':['NPZ headers verify structural presence only, not full decode or rendering.',
            'No new visual approval, camera validation or metric eligibility is claimed.',
            'Prior exclusions are preserved even when historical reasons mention asset appearance.',
            'Proposed times are not frozen; source identity and input review must complete first.'],
        'counts':{'original75':sum(r['cohort']=='original75' for r in rows),
            'candidate44':sum(r['cohort']=='candidate44' for r in rows),
            'candidates_blocked_by_prior_exclusion':sum(r['cohort']=='candidate44' and r['blocked_by_prior_exclusion'] for r in rows),
            'issues':dict(collections.Counter(i for r in rows for i in r['issues']))}, 'records':rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['counts'],ensure_ascii=False))


if __name__ == '__main__':
    main()
