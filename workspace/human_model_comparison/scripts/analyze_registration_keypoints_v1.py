"""Triangulation diagnostics for native views; no automatic source registration."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from audit_static_source_cameras_v1 import sha,read,checked


def triangulate(cameras, pixels, weights=None):
    if len(cameras)<2 or len(cameras)!=len(pixels):raise ValueError('At least two matching cameras required')
    weights=np.ones(len(cameras)) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=(len(cameras),) or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Finite positive weights required')
    equations=[]
    for cam,uv,weight in zip(cameras,pixels,weights):
        K=np.asarray(cam['effective_pixel_K'],dtype=float);T=np.asarray(cam['world_to_camera'],dtype=float)
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
            raise ValueError('Invalid camera')
        xy=np.linalg.solve(K,[*uv,1.]);xy=xy[:2]/xy[2]
        for axis in range(2):
            row=xy[axis]*T[2]-T[axis]
            equations.append(row/max(np.linalg.norm(row[:3]),1e-12)*np.sqrt(weight))
    matrix=np.stack(equations)
    if not np.isfinite(matrix).all():raise ValueError('Non-finite observations')
    _,singular,V=np.linalg.svd(matrix)
    if singular[-2]<1e-8 or abs(V[-1,3])<1e-10:raise ValueError('Degenerate/parallel rays')
    xyz=V[-1,:3]/V[-1,3]
    depths=[];residuals=[];rays=[]
    for camera,uv in zip(cameras,pixels):
        T=np.asarray(camera['world_to_camera']);K=np.asarray(camera['effective_pixel_K'])
        point=T[:3,:3] @ xyz+T[:3,3];depths.append(float(point[2]))
        if point[2]<=0:raise ValueError('Triangulated point behind a camera')
        px=K @ point;residuals.append(float(np.linalg.norm(px[:2]/px[2]-uv)))
        center=-T[:3,:3].T @ T[:3,3]
        ray=xyz-center;rays.append(ray/np.linalg.norm(ray))
    cross=max(np.linalg.norm(np.cross(a,b)) for a,b in itertools.combinations(rays,2))
    if cross<1e-6:raise ValueError('Degenerate collinear rays')
    return dict(xyz=xyz.tolist(),depths=depths,reprojection_errors_pixels=residuals,
                max_reprojection_error_pixels=max(residuals),mean_reprojection_error_pixels=float(np.mean(residuals)),
                max_effective_ray_angle_degrees=float(np.rad2deg(np.arcsin(np.clip(cross,0,1)))),
                singular_values=singular.tolist())


def analyze_joint(observations,joint):
    selected=[r for r in observations if r.get('keypoints') is not None and r['keypoint_scores'][joint]>.5]
    result=dict(joint_index=joint,observation_case_ids=[r['case']['case_id'] for r in selected],
                selected_observations=len(selected),confidence_threshold=.5,semantic_validated=False)
    if len(selected)<2:return dict(result,status='unavailable',reason='fewer_than_two_confident_observations')
    cameras=[r['case']['camera'] for r in selected]
    pixels=np.array([r['keypoints'][joint] for r in selected]);weights=np.array([r['keypoint_scores'][joint] for r in selected])
    try:
        geometry=triangulate(cameras,pixels,weights)
        movement=[]
        # Sensitivity to one pixel in each observation; diagnostic, not a pose fit.
        for i in range(len(pixels)):
            for axis in range(2):
                altered=pixels.copy();altered[i,axis]+=1
                try:
                    changed=triangulate(cameras,altered,weights)
                    movement.append(float(np.linalg.norm(np.array(changed['xyz'])-geometry['xyz'])))
                except ValueError:
                    return dict(result,status='unstable',reason='one_pixel_perturbation_degenerate',geometry=geometry)
        return dict(result,status='triangulated_pending_semantic_review',geometry=geometry,
                    max_native_displacement_per_one_pixel=max(movement))
    except ValueError as error:return dict(result,status='unavailable',reason=str(error))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    state=read(args.state)
    if state['status']!='cpu_keypoint_observations_complete_pending_review':raise ValueError('Keypoint extraction incomplete')
    groups={};source_records={}
    for item in state['cases']:
        checked(item['record'],item['sha256']);record=read(item['record']);case=record['case']
        if case['role']=='native_asset_view':groups.setdefault(case['id'],[]).append(record)
        else:source_records[case['id']]=dict(path=item['record'],sha256=item['sha256'])
    records=[]
    for sid,observations in groups.items():
        payload_hashes={r['case']['native_payload_sha256'] for r in observations}
        if len(payload_hashes)!=1:raise ValueError('Views use different assets')
        joints=[analyze_joint(observations,i) for i in range(5,17)]
        records.append(dict(id=sid,native_views=len(observations),source_reference=source_records.get(sid),
                            native_payload_sha256=next(iter(payload_hashes)),joints=joints,
                            triangulated_joints=sum(j['status']=='triangulated_pending_semantic_review' for j in joints),
                            source_camera_registered=False,common_camera_valid=False,metric_eligible=False))
    report=dict(status='native_semantic_geometry_diagnostics_only',input_state=str(args.state.resolve()),
                input_state_sha256=sha(args.state),script_sha256=sha(__file__),objects=records,
                head_face_joints_excluded=True,confidence_is_visibility=False,
                jointly_reposed=False,source_camera_optimized=False,scores_produced=False)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps([{k:r[k] for k in ['id','native_views','triangulated_joints']} for r in records]))


if __name__=='__main__':main()
