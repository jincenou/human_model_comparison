"""Freeze common choices before any formal geometric scores are produced."""
import time
from common import *
from pilot import combine_vertical,assess_vertical
assert not list((E/'results').glob('*/*.json'))
assert read(E/'validation/cpu.json')['status']=='passed'
assert read(E/'validation/voxel_analytic.json')['status']=='passed'
pilots=[]
for method,sid in [('ours','HN0004'),('ah','HN0004'),('trellis_original','HN0038'),('trellis2','HN0646'),('omnire','HN0038'),('splatad','HN0172')]:
    p=E/'pilot'/method/sid/'measurement.json';d=read(p);assert d['status']=='passed'
    # Previous successful six-view pilot records are preserved, not overwritten.
    mm={str(n):combine_vertical([v['measurement'] for v in d['views'] if v['view']['resolution']==n and 'measurement' in v]) for n in []}
    for n in d['measurements']:
        views=[v['measurement'] for v in d['views'] if str(v['view'].get('resolution',v['view'].get('width',0)))==n and 'measurement' in v]
        if not views:views=[v['measurement'] for v in d['views'] if Path(v['path']).name.startswith(n+'_') and 'measurement' in v]
        mm[n]=combine_vertical(views)
    check=assess_vertical(mm,d['frame']['span']);assert check['passed']
    pilots.append(dict(method=method,id=sid,measurement=str(p),sha256=sha(p),vertical_reassessment=check))
save(E/'validation/pilot_vertical_reassessment.json',dict(status='passed',rows=pilots))
protocol=dict(version='human_static_geometry_v1',frozen_at=time.time(),selection_sha256=SELECTION_SHA,
 reference_manifest_sha256=sha(E/'reference_manifest.json'),reference='same-sample labeled single LiDAR keyframe; category + unique same-category OBB; first real raw index per 5cm voxel',
 reference_qualification='Codex direct visual review; nominal scan rays; 84 main >=20 voxels,16 low support separately; no instantaneous full-surface GT claim',
 main_panel='annotation_vertical_extent_conditioned',scale_rule='s=annotation_world_vertical_extent/full_native_alpha_world_gravity_height; A=T_world_camera @ diag(s,s,s,1) @ rigid_source_T; translation scales too',
 scene_native_metric='separate panel with original physical-time world mapping, no annotation extent',
 annotation_information=['identity OBB and category for reference filtering','gravity and OBB vertical extent for global scale only'],
 prohibited=['ICP','LiDAR endpoint fitting','joint pose correction','nonuniform scaling','per-example alpha pruning','score-selected registration'],
 native_extent=dict(views='six orthographic cardinal views aligned to world gravity; all attempted',alpha_uint8_threshold='>127 after round(255*alpha)',support='union of nonempty side-view vertical endpoints; at least one opposing side pair; edge-on empty views retained as evidence; clipping/technical failures do not pass',resolutions=[512,1024,2048,4096],boundary_tolerance_fraction=.01,pixel_pitch_fraction=.005,framing='complete native primitive conservative alpha framing; no primitive removed'),
 depth=dict(sectors=6,fov_degrees=90,resolution=1024,alpha_threshold=.5,range_m=[.1,200.],gaussian='native alpha weights; ED=sum(w*z)/sum(w); interpolate Dz and A before dividing; range=z/u_cam_z',trellis_metrology='validated gsplat antialiased eps=.1 only for orthographic extent; original kernel for perspective queries',splatad_metrology='native-radius/AA validated shared alpha only for orthographic extent; original kernel for perspective queries',trellis2='native MeshWithVoxel trilinear alpha, SSAA2, accumulate A/Dz; start 8 layers, uniformly double to16/32/64 if next-hit remaining transmission >1e-4; fail if unresolved; retains native8 when converged',trellis2_glb_transform_used=False),
 metrics=dict(cd='0.5*(mean nearest P-to-G + mean nearest G-to-P), unsquared L2 metres',thresholds_m=[.05,.1,.2],comparison='strict <',fixed_reference='all G retained regardless of missed queries',empty_pred='F1 and RayHit/RayAcc=0; CD null; extended mean infinity when any empty',aggregation='person equal weight; own/common-admission/common-finite/Ours-paired sets; low-support separate'),
 registration_recovery=dict(scope='all54 existing direct assets with missing source relation',score_blind=True,min_body_correspondences=6,confidence_threshold=.5,one_round='fixed photometric-content zoom for pose detection at512; original camera mapping retained; pair-consensus triangulation >=3 views <=8px, >=10degree; single positive-depth SQPnP and LM only if remains positive',visual_review='Codex semantic correspondence review, not independent human blind review; no score-based acceptance'),
 pilots=pilots,time_plan_sha256=sha(E/'time_plan.json'),code={str(p):sha(p) for p in [E/'scripts/depth.py',E/'scripts/scene_depth.py',E/'scripts/pilot.py',E/'scripts/recover_registration.py',G/'scripts/geometry_core.py']})
immutable(E/'protocol.json',protocol)
print('protocol frozen before scores',flush=True)
