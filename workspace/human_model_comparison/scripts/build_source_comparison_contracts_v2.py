"""One source camera and reviewed-reference ROI per identity, independent of method."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from human_source_metrics_v1 import sha, prepare_reference

H = Path(__file__).resolve().parents[1]
OUT = H / 'configs/human_source_common_v1/reviewed_source_v2'

def main():
    selection = H / 'inputs/benchmark_human100_v2/selection.json'
    review = H / 'reports/reference_qualification_v1/current_candidates_v1.json'
    selected = json.loads(selection.read_text())['objects']
    refs = json.loads(review.read_text())
    assert refs['selection_sha256'] == sha(selection)
    by_id = {r['id']: r for r in refs['objects']}
    assert set(by_id) == {r['id'] for r in selected} and len(by_id) == 100
    OUT.mkdir(parents=True, exist_ok=False)
    records = []
    for row in selected:
        sid = row['id']
        ref = by_id[sid]
        prior = H / 'configs/human_source_common_v1/candidates' / sid / 'camera.json'
        camera = json.loads(prior.read_text())
        assert camera['selection_sha256'] == sha(selection)
        assert camera['timestamp_us'] == row['t_ref_us'] == ref['timestamp_us']
        assert camera['identity_key'] == row['identity_key'] == ref['identity_key']
        assert sha(ref['raw']) == ref['raw_sha256'] == camera['source_rgb_sha256']
        assert sha(ref['candidate_mask']) == ref['candidate_mask_sha256']
        rgb = np.asarray(Image.open(ref['raw']).convert('RGB'))
        mask = np.asarray(Image.open(ref['candidate_mask']).convert('L'))
        assert list(rgb.shape[1::-1]) == camera['render_wh'] and mask.shape == rgb.shape[:2]
        prepared = prepare_reference(rgb, mask)
        assert prepared['status'] == 'ready'
        target = OUT / sid
        target.mkdir()
        Image.fromarray(prepared['rgb']).save(target / 'reference_rgb.png')
        Image.fromarray(prepared['mask']).save(target / 'reference_mask.png')
        reasons = []
        if not ref['visual_passed']:
            reasons.append(ref['visual_status'])
        if sid in refs['accessory_occlusion_requires_eligibility_review_ids']:
            reasons.append('carried_object_external_occlusion_pending_eligibility')
        c = {key: camera[key] for key in ['sample_id', 'identity_key', 'scene', 'timestamp_us',
             'selection_sha256', 'render_wh', 'K_raw', 'T_world_from_camera_opencv',
             'sample_data_token', 'ego_pose_token', 'calibrated_sensor_token', 'metadata_table_sha256']}
        ys = np.nonzero(mask > 127)[0]
        c.update(schema='reviewed_source_camera_contract_v2',
                 status='source_camera_and_reference_roi_fixed_registration_pending',
                 source_rgb=ref['raw'], source_rgb_sha256=ref['raw_sha256'],
                 source_mask=ref['candidate_mask'], source_mask_sha256=ref['candidate_mask_sha256'],
                 prior_camera_contract=str(prior), prior_camera_contract_sha256=sha(prior),
                 review_manifest=str(review), review_manifest_sha256=sha(review),
                 roi_xyxy=list(prepared['roi']), roi_size=512, roi_expansion=1.10,
                 reference_pixel_height=int(ys.max()-ys.min()+1),
                 reference_visual_passed=ref['visual_passed'], reference_pending_reasons=reasons,
                 reference_rgb=str(target/'reference_rgb.png'), reference_rgb_sha256=sha(target/'reference_rgb.png'),
                 reference_mask=str(target/'reference_mask.png'), reference_mask_sha256=sha(target/'reference_mask.png'),
                 prediction_encoding_required=True, prediction_mask_policy='own alpha only; never reference mask',
                 render_policy='render on raw canvas with this K/T, then apply this common reference ROI',
                 rgb_interpolation='PIL bicubic', mask_interpolation='PIL nearest', background='white',
                 per_method_bbox_resize_allowed=False, score_based_camera_selection_allowed=False,
                 pose_correction_allowed=False, final_metric_eligible=False,
                 orbit_settings_frozen=False)
        f=target/'camera.json';f.write_text(json.dumps(c,indent=2)+'\n')
        records.append(dict(id=sid,path=str(f),sha256=sha(f),reference_pending_reasons=reasons))
    manifest=dict(schema='reviewed_source_contract_index_v2', selection_sha256=sha(selection),
                  count=100, records=records, common_camera_registration_required=True,
                  final_metric_protocol_frozen=False, predictions_used_to_choose_roi=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(contracts=100,reference_without_pending_flags=sum(not x['reference_pending_reasons'] for x in records),output=str(OUT))))

if __name__ == '__main__':
    main()
