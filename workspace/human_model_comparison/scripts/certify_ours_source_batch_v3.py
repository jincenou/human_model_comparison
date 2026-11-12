"""Certify traced source coordinates, not correctness of the predicted human pose.

Model pose, stature and appearance errors remain in the output. No image-based
angle, scale, bbox or joint optimization is performed by this certification.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from export_ours_frozen_time_v1 import H,sha


def certify(sid,selected,active,validation):
    row=selected[sid];validation=Path(validation)
    v=json.loads(validation.read_text());rp=Path(v['registration']);assert sha(rp)==v['registration_sha256'];reg=json.loads(rp.read_text())
    cp=Path(reg['camera_contract']);assert sha(cp)==reg['camera_contract_sha256']==v['camera_contract_sha256'];c=json.loads(cp.read_text())
    assert c['identity_key']==row['identity_key'] and c['timestamp_us']==row['t_ref_us']==reg['timestamp_us']
    assert sha(c['source_rgb'])==c['source_rgb_sha256']==row['reference_frame']['raw_source_sha256']
    assert reg['source_time_seconds']==row['reference_seconds_from_first_native']
    assert 0<=reg['fraction']<=1 and reg['max_endpoint_projection_error_px']<.02
    assert reg['source_pose_estimated'] is True
    for path,digest in reg['provenance_sha256'].items():assert sha(path)==digest,path
    assert sha(reg['snapshot'])==reg['snapshot_sha256']
    provenance=Path(reg['snapshot']).parent/'provenance.json';p=json.loads(provenance.read_text())
    assert p['id']==sid and p['timestamp_us']==row['t_ref_us'] and p['sh_degree']==3
    assert p['root_roundtrip_position_max_error']<2e-5
    assert sha(p['runtime_path'])==p['runtime_sha256']
    for path,digest in p['code_sha256'].items():assert sha(path)==digest,path
    T=np.array(reg['T_source_camera_from_export']);world=np.array(c['T_world_from_camera_opencv'])
    np.testing.assert_allclose(reg['T_world_from_export'],world@T,atol=1e-8,rtol=0)
    singular=np.linalg.svd(T[:3,:3],compute_uv=False);assert np.allclose(singular,singular[0],rtol=1e-5) and np.linalg.det(T[:3,:3])>0
    np.testing.assert_allclose(v['K_raw'],c['K_raw'],rtol=0,atol=1e-9);assert v['render_wh']==c['render_wh']
    script=H/'scripts/render_ours_source_batch_v3.py';assert sha(script)==v['script_sha256']
    e=v['coordinate_equivalence']
    # User decision 2026-11-12: equivalent-render image differences are
    # diagnostic, not a per-person scoring gate. Identity/time/camera/source
    # geometry checks above remain required. Neither rendered output is changed.
    assert all(np.isfinite(e[k]) and e[k]>=0 for k in ['rgb_max','rgb_mean','alpha_max','alpha_mean'])
    assert v['preserved_sh_degree']==3 and v['covariance_and_appearance_transformed_together']
    for key in ['source_rgb','source_alpha']:assert sha(v[key])==v[key+'_sha256']
    rgb=np.array(Image.open(v['source_rgb']).convert('RGB'));alpha=np.array(Image.open(v['source_alpha']).convert('L'))
    assert list(rgb.shape[1::-1])==c['render_wh'] and rgb.shape[:2]==alpha.shape
    error=int((255-rgb[alpha==0]).max()) if np.any(alpha==0) else 0;assert error<=2
    rec=dict(status='passed',registration_certified=True,identity=sid,identity_key=row['identity_key'],timestamp_us=row['t_ref_us'],
             camera_sha256=sha(cp),method='ours',camera_contract=str(cp),render_validation=str(validation),render_validation_sha256=sha(validation),
             render_script=str(script),render_script_sha256=sha(script),source_rgb=v['source_rgb'],source_rgb_sha256=v['source_rgb_sha256'],
             source_alpha=v['source_alpha'],source_alpha_sha256=v['source_alpha_sha256'],prediction_encoding='white_composited_uint8',
             zero_alpha_background_max_byte_error=error,posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,
             registration_kind='existing_conditioning_root_pose_and_export_camera_trace',source_pose_estimated=True,
             independent_world_metric_geometry_certified=False,predicted_pose_and_size_errors_preserved=True,
             full_sh_degree=3,coordinate_equivalence=e,coordinate_equivalence_role='diagnostic_only_not_a_scoring_gate',
             pixel_gate_removed_by_user_instruction=True,reference_source_rgb_used_in_original_pose_estimation=True,
             real_heldout_eligible=False,scope='Source reconstruction in the real camera using the method original estimated pose/scale; no claim of accurate ground-truth pose or body dimensions',
             trace_record=str(rp),trace_record_sha256=sha(rp),active_selection_sha256=active['selection_sha256'],certification_script_sha256=sha(__file__))
    return rec
