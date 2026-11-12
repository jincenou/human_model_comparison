"""Explain cached registration failures without rerendering or changing fits."""
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from trellis2_source_adapter import H, ROOT, read, save, sha, checked
from analyze_registration_keypoints_v1 import triangulate


def audit():
    rows = []
    for path in sorted((ROOT / 'observations').glob('*/registration_status.json')):
        status = read(path)
        if status['status'] != 'unavailable':
            continue
        kp_path = path.parent / 'keypoints.json'
        kp = read(kp_path)
        records = []
        for item in kp['records']:
            checked(item['record'], item['sha256'])
            records.append(read(item['record']))
        ref = next(r for r in records if r['case']['role'] == 'reference')
        native = [r for r in records if r['case']['role'] == 'native_asset_view']
        joints, valid = [], []
        for i in range(5, 17):
            obs = [r for r in native if r['keypoints'] is not None and r['keypoint_scores'][i] > .5]
            ref_score = float(ref['keypoint_scores'][i]) if ref['keypoints'] is not None else None
            row = dict(joint_index=i, native_confident_views=len(obs), reference_confidence=ref_score)
            if len(obs) < 2 or ref_score is None or ref_score <= .5:
                row.update(status='insufficient_observations', native_insufficient=len(obs) < 2,
                           reference_insufficient=ref_score is None or ref_score <= .5)
            else:
                try:
                    geometry = triangulate([r['case']['camera'] for r in obs],
                                           [r['keypoints'][i] for r in obs],
                                           [r['keypoint_scores'][i] for r in obs])
                    row.update(status='usable', geometry=geometry)
                    valid.append(row)
                except ValueError as error:
                    row.update(status='triangulation_failed', reason=str(error))
            joints.append(row)
        row = dict(id=status['id'], recorded_reason=status.get('reason'),
                   evidence=[dict(path=str(p), sha256=sha(p)) for p in [path, kp_path]],
                   native_views=len(native), native_views_with_person=sum(r['keypoints'] is not None for r in native),
                   native_detection_statuses=dict(Counter(r.get('reason') or r.get('status') for r in native)),
                   reference_confident_body_joints=sum(j['reference_confidence'] is not None and j['reference_confidence'] > .5 for j in joints),
                   native_multiview_body_joints=sum(j['native_confident_views'] >= 2 for j in joints),
                   usable_body_correspondences=len(valid), joints=joints)
        if len(valid) < 6:
            row['verified_failure'] = 'fewer_than_six_body_correspondences'
        else:
            xyz = np.array([j['geometry']['xyz'] for j in valid], np.float64)
            uv = np.array([ref['keypoints_raw_pixels'][j['joint_index']] for j in valid], np.float64)
            K = np.array(ref['case']['raw_K'], np.float64)
            ok, r, t = cv2.solvePnP(xyz, uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                row['verified_failure'] = 'pnp_no_solution'
            else:
                r, t = cv2.solvePnPRefineLM(xyz, uv, K, None, r, t,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-8))
                rotation = cv2.Rodrigues(r)[0]
                depths = (xyz @ rotation.T + t[:, 0])[:, 2]
                row['pnp'] = dict(finite=bool(np.isfinite(rotation).all() and np.isfinite(t).all()),
                                 rotation_determinant=float(np.linalg.det(rotation)), depths=depths.tolist(),
                                 nonpositive_depth_joints=[j['joint_index'] for j, d in zip(valid, depths) if d <= 0])
                row['verified_failure'] = 'pnp_nonpositive_depth' if (depths <= 0).any() else 'needs_review'
        rows.append(row)
    report = dict(created_at=time.time(), script_sha256=sha(__file__), rows=rows,
                  counts=dict(Counter(r['verified_failure'] for r in rows)),
                  reference_fewer_than_six=sum(r['reference_confident_body_joints'] < 6 for r in rows),
                  native_fewer_than_six=sum(r['native_multiview_body_joints'] < 6 for r in rows),
                  both_fewer_than_six=sum(r['reference_confident_body_joints'] < 6 and r['native_multiview_body_joints'] < 6 for r in rows),
                  native_zero_person_views=sum(r['native_views_with_person'] == 0 for r in rows),
                  scope='Read-only replay of frozen observations and solver; no asset, camera, threshold or score changed',
                  detector_failure_is_not_proof_of_bad_asset=True)
    save(ROOT / 'failure_audit.json', report)
    print({k: v for k, v in report.items() if k != 'rows'})
    for r in rows:
        print(r['id'], r['usable_body_correspondences'], r['native_views_with_person'],
              r['reference_confident_body_joints'], r['native_multiview_body_joints'], r['verified_failure'])


if __name__ == '__main__':
    audit()
