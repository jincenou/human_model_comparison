"""CPU-only replay of existing unavailable TRELLIS registrations; never fit anew.

Replays the frozen SQPnP/LM calculation from cached keypoint observations. Writes
only a separate audit JSON; native assets, registrations and scores are untouched.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_registration_keypoints_v1 import triangulate

H = Path(__file__).resolve().parents[1]
ROOT = H / 'reports/trellis_source_batch'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(path, digest):
    assert sha(path) == digest, str(path)


def pose_diagnostics(xyz, uv, K, r, t):
    R = cv2.Rodrigues(r)[0]
    depths = (xyz @ R.T + t[:, 0])[:, 2]
    errors = np.linalg.norm(cv2.projectPoints(xyz, r, t, K, None)[0][:, 0] - uv, axis=-1)
    return dict(finite_transform=bool(np.isfinite(R).all() and np.isfinite(t).all()),
                rotation_determinant=float(np.linalg.det(R)),
                depths_native_units=depths.tolist(),
                nonpositive_depth_indices=np.flatnonzero(depths <= 0).tolist(),
                mean_reprojection_error_px=float(errors.mean()),
                reprojection_error_is_diagnostic_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'remaining_registration_audit.json')
    args = parser.parse_args()
    protocol_path = H / 'configs/human_source_common_v1/source_keypoint_rigid_pnp_v1.json'
    protocol = read(protocol_path)
    assert protocol['body_joint_indices'] == list(range(5, 17))
    assert protocol['confidence_threshold'] == .5 and protocol['minimum_correspondences'] == 6
    assert protocol['solver'] == 'SQPnP followed by LM'
    assert protocol['lm_max_iterations'] == 30 and protocol['lm_epsilon'] == 1e-8
    active_path = H / 'inputs/active_human_benchmark.json'
    active = read(active_path)
    checked(active['selection'], active['selection_sha256'])
    selected = {r['id']: r for r in read(active['selection'])['objects']}
    manifest_path = ROOT / 'manifest.json'
    manifest = read(manifest_path)
    assert manifest['selection_sha256'] == active['selection_sha256']
    rows = []
    for missing in manifest['unavailable']:
        sid = missing['id']
        assert sid in selected
        out = ROOT / 'observations' / sid
        keypoint_path = out / 'keypoints.json'
        keypoints = read(keypoint_path)
        records = []
        for entry in keypoints['records']:
            checked(entry['record'], entry['sha256'])
            records.append(read(entry['record']))
        reference = next(r for r in records if r['case']['role'] == 'reference')
        native = [r for r in records if r['case']['role'] == 'native_asset_view']
        joints, unavailable = [], []
        for joint in protocol['body_joint_indices']:
            observations = [r for r in native if r['keypoints'] is not None and r['keypoint_scores'][joint] > .5]
            source_score = None if reference['keypoints'] is None else reference['keypoint_scores'][joint]
            if source_score is None or source_score <= .5 or len(observations) < 2:
                unavailable.append(dict(joint_index=joint, native_confident_observations=len(observations),
                                        source_confidence=source_score, reason='insufficient_confident_observations'))
                continue
            try:
                geometry = triangulate([r['case']['camera'] for r in observations],
                                       [r['keypoints'][joint] for r in observations],
                                       [r['keypoint_scores'][joint] for r in observations])
                joints.append(dict(joint_index=joint, xyz=geometry['xyz']))
            except ValueError as error:
                unavailable.append(dict(joint_index=joint, reason=str(error)))
        row = dict(id=sid, existing_registration_status=read(out / 'registration_status.json'),
                   registration_status_file=record(out / 'registration_status.json'),
                   keypoint_manifest=record(keypoint_path),
                   keypoint_records=keypoints['records'],
                   body_correspondence_count=len(joints), joint_indices=[r['joint_index'] for r in joints],
                   unavailable_joints=unavailable,
                   native_detection_counts=dict(Counter(r.get('reason') or r['status'] for r in native)))
        if len(joints) < protocol['minimum_correspondences']:
            row.update(reason='fewer_than_six_body_correspondences', recovery='No retry under unchanged fixed protocol')
        else:
            cp = reference['case']['camera_contract']
            checked(cp, reference['case']['camera_contract_sha256'])
            cam = read(cp)
            assert cam['timestamp_us'] == selected[sid]['t_ref_us']
            assert cam['identity_key'] == selected[sid]['identity_key']
            xyz = np.asarray([r['xyz'] for r in joints], np.float64)
            uv = np.asarray([reference['keypoints_raw_pixels'][r['joint_index']] for r in joints], np.float64)
            K = np.asarray(cam['K_raw'], np.float64)
            success, r, t = cv2.solvePnP(xyz, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
            assert success, 'SQPnP returned no solution'
            row['sqpnp'] = pose_diagnostics(xyz, uv, K, r, t)
            r, t = cv2.solvePnPRefineLM(xyz, uv, K, None, r, t,
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-8))
            row['lm'] = pose_diagnostics(xyz, uv, K, r, t)
            row['nonpositive_depth_joints'] = [joints[i]['joint_index'] for i in row['lm']['nonpositive_depth_indices']]
            assert row['lm']['finite_transform'] and row['lm']['rotation_determinant'] > .999
            assert row['nonpositive_depth_joints'], 'Previously unavailable case now needs separate review'
            row.update(reason='source_pnp_nonpositive_depth_at_joints_' + '_'.join(map(str, row['nonpositive_depth_joints'])),
                       failed_check='trellis_source_batch.py:141 positive camera-space depth',
                       camera_contract=record(cp),
                       recovery='No solver change, joint removal, or nonpositive-depth bypass; retain method-level missing result')
        rows.append(row)
    output = dict(status='cpu_replay_complete_no_registration_or_score_changes',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  selection=record(active['selection']), input_manifest=record(manifest_path),
                  protocol=record(protocol_path), opencv_version=cv2.__version__, numpy_version=np.__version__,
                  scripts=[record(__file__), record(H / 'scripts/trellis_source_batch.py'),
                           record(H / 'scripts/analyze_registration_keypoints_v1.py'),
                           record(H / 'scripts/audit_static_source_cameras_v1.py')],
                  command=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python -B ') + str(Path(__file__).resolve()) + ' --output /tmp/trellis_remaining_registration_replay.json',
                  rows=rows, unavailable_count=len(rows),
                  scope='Cached keypoint observations only; no GPU, rendering, generation, alternative solver or metric-guided tuning')
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), rows=[{k: r[k] for k in ['id', 'body_correspondence_count', 'reason']} for r in rows]), ensure_ascii=False))


if __name__ == '__main__':
    main()
