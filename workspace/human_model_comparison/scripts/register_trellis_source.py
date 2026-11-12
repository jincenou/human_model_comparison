"""Source-keypoint rigid registration of existing TRELLIS Gaussians.

Fits only one SE(3) transform with fixed real intrinsics. No image metric,
mask score, predicted bbox, per-joint pose, or output asset is optimized.
This is source reconstruction after keypoint registration, not held-out scoring.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from audit_static_source_cameras_v1 import H,TRELLIS,read,sha,checked

OUT=H/'reports/trellis_source_registration'
REG=H/'configs/human_source_common_v1/trellis_source_registration'

def prepare(ids):
    import cv2
    active=read(H/'inputs/active_human_benchmark.json');checked(active['selection'],active['selection_sha256'])
    selected={r['id']:r for r in read(active['selection'])['objects']}
    path=H/'reports/trellis_registration_keypoints_v1/triangulation_v1.json';tri=read(path)
    records={r['id']:r for r in tri['objects']};REG.mkdir(exist_ok=True)
    for sid in ids:
        assert sid in selected
        row=records[sid];rp=Path(row['source_reference']['path']);checked(rp,row['source_reference']['sha256']);ref=read(rp)
        cp=Path(ref['case']['camera_contract']);checked(cp,ref['case']['camera_contract_sha256']);cam=read(cp)
        assert cam['timestamp_us']==selected[sid]['t_ref_us'] and cam['identity_key']==selected[sid]['identity_key']
        joints=[j for j in row['joints'] if j['status']=='triangulated_pending_semantic_review' and ref['keypoint_scores'][j['joint_index']]>.5]
        xyz=np.array([j['geometry']['xyz'] for j in joints],np.float64)
        uv=np.array([ref['keypoints_raw_pixels'][j['joint_index']] for j in joints],np.float64)
        K=np.array(cam['K_raw'],np.float64);assert len(joints)>=6
        success,r,t=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);assert success
        r,t=cv2.solvePnPRefineLM(xyz,uv,K,None,r,t,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,30,1e-8))
        R=cv2.Rodrigues(r)[0];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t[:,0]
        points=xyz@R.T+t[:,0];assert np.isfinite(T).all() and np.linalg.det(R)>.999 and np.all(points[:,2]>0)
        projected=cv2.projectPoints(xyz,r,t,K,None)[0][:,0];error=np.linalg.norm(projected-uv,axis=-1)
        payload=H/'runs/trellis_original_human_native_v1'/sid/'native_gaussians.pt';checked(payload,row['native_payload_sha256'])
        record=dict(id=sid,identity_key=cam['identity_key'],timestamp_us=cam['timestamp_us'],method='trellis_original',
            active_selection_sha256=active['selection_sha256'],camera_contract=str(cp),camera_contract_sha256=sha(cp),
            native_payload=str(payload),native_payload_sha256=sha(payload),T_source_camera_from_export=T.tolist(),
            working_units='native TRELLIS units; PnP determines translation in those same units',
            metric_world_scale_certified=False,scale_gauge=1.,source_orientation_known_from_generation=False,
            protocol='Body joints 5..16 with confidence>0.5; fixed source K; one SQPnP solution then LM(30,1e-8); all accepted correspondences retained',
            native_joint_triangulation=str(path),native_joint_triangulation_sha256=sha(path),
            source_keypoints=str(rp),source_keypoints_sha256=sha(rp),joint_indices=[j['joint_index'] for j in joints],
            xyz_native=xyz.tolist(),target_pixels_raw=uv.tolist(),projected_pixels_raw=projected.tolist(),
            reprojection_error_px=error.tolist(),mean_reprojection_error_px=float(error.mean()),max_reprojection_error_px=float(error.max()),
            pose_errors_preserved=True,asset_modified=False,source_keypoints_used_for_registration=True,
            posthoc_pose_or_scale_fit=True,fit_scope='single rigid asset-to-camera transform, not human joint pose or scale',
            reference_metric_values_used=False,prediction_bbox_used=False,real_heldout_eligible=False,
            registration_certified=False,metric_eligible=False,script_sha256=sha(__file__),opencv_version=cv2.__version__)
        dest=REG/(sid+'.json');assert not dest.exists();dest.write_text(json.dumps(record,indent=2)+'\n')
        print(sid,'PnP mean/max pixels',error.mean(),error.max(),flush=True)


def render(sid):
    import torch
    from PIL import Image
    from types import SimpleNamespace
    sys.path.insert(0,str(TRELLIS))
    from trellis.representations.gaussian import Gaussian
    from trellis.renderers.gaussian_render import render as native_render,intrinsics_to_projection
    rp=REG/(sid+'.json');reg=read(rp);checked(reg['native_payload'],reg['native_payload_sha256']);checked(reg['camera_contract'],reg['camera_contract_sha256'])
    cam=read(reg['camera_contract']);K=np.array(cam['K_raw']);W,Hh=cam['render_wh'];T=np.array(reg['T_source_camera_from_export'])
    packed=torch.load(reg['native_payload'],map_location='cpu',weights_only=True);assert packed['init_params']['sh_degree']==0
    gs=Gaussian(**packed['init_params'])
    for name,value in packed['params'].items():
        assert torch.isfinite(value).all();setattr(gs,name,value.cuda())
    # Native C++ ndc2Pix samples integer pixel centres. Its wrapper flips cy;
    # cancel that sign and the half-pixel offset to recover the raw OpenCV K.
    kn=torch.tensor([[K[0,0]/W,0,(K[0,2]+.5)/W],[0,K[1,1]/Hh,1-(K[1,2]+.5)/Hh],[0,0,1]],device='cuda',dtype=torch.float32)
    view=torch.tensor(T,device='cuda',dtype=torch.float32);proj=intrinsics_to_projection(kn,.01,1000)
    camera=SimpleNamespace(image_width=W,image_height=Hh,FoVx=2*np.arctan(W/(2*K[0,0])),FoVy=2*np.arctan(Hh/(2*K[1,1])),
        world_view_transform=view.T.contiguous(),full_proj_transform=(proj@view).T.contiguous(),camera_center=torch.linalg.inv(view)[:3,3])
    pipe=SimpleNamespace(kernel_size=.1,debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    with torch.inference_mode():
        rgb=native_render(camera,gs,pipe,torch.ones(3,device='cuda'))['render']
        a=native_render(camera,gs,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(gs.get_xyz))['render']
        assert torch.isfinite(rgb).all() and torch.isfinite(a).all()
        assert float((a-a[:1]).abs().max())<1e-6
        # Three isolated Gaussians verify this actual off-centre raw-K renderer.
        probe=Gaussian(**packed['init_params']);probe.from_scaling(torch.full((1,3),max(.002,2*float(probe.mininum_kernel_size)),device='cuda'))
        probe.from_rotation(torch.tensor([[1.,0.,0.,0.]],device='cuda'));probe.from_opacity(torch.tensor([[.9]],device='cuda'));probe.from_features(torch.zeros((1,3,1),device='cuda'))
        checks=[];yy,xx=torch.meshgrid(torch.arange(Hh,device='cuda'),torch.arange(W,device='cuda'),indexing='ij')
        depth=float(np.mean((np.array(reg['xyz_native'])@T[:3,:3].T+T[:3,3])[:,2]))
        for offset in [[0,0],[.05,.03],[-.04,-.04]]:
            point=np.array([*offset,depth]);native=(point-T[:3,3])@T[:3,:3]
            probe.from_xyz(torch.tensor([native],device='cuda',dtype=torch.float32))
            v=native_render(camera,probe,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(probe.get_xyz))['render'][0]
            centroid=torch.stack([(v*xx).sum(),(v*yy).sum()])/v.sum();expected=K@point;expected=expected[:2]/expected[2]
            checks.append(dict(expected=expected.tolist(),centroid=centroid.cpu().tolist(),max_error_px=float(np.abs(centroid.cpu().numpy()-expected).max())))
        assert max(r['max_error_px'] for r in checks)<.1
    dest=OUT/sid;dest.mkdir(exist_ok=False)
    for name,arr in [('source_rgb',rgb.permute(1,2,0)),('source_alpha',a[0])]:Image.fromarray(np.floor(arr.clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)).save(dest/(name+'.png'))
    for name,value in packed['params'].items():assert torch.equal(value,getattr(gs,name).cpu())
    result=dict(status='source_keypoint_registered_rendered_pending_review',id=sid,method='trellis_original',
        registration=str(rp),registration_sha256=sha(rp),camera_contract=reg['camera_contract'],camera_contract_sha256=reg['camera_contract_sha256'],
        render_wh=[W,Hh],K_raw=cam['K_raw'],source_rgb=str(dest/'source_rgb.png'),source_alpha=str(dest/'source_alpha.png'),
        source_rgb_sha256=sha(dest/'source_rgb.png'),source_alpha_sha256=sha(dest/'source_alpha.png'),prediction_encoding='white_composited_uint8',
        raw_camera_projection_checks=checks,gaussian_parameters_unchanged=True,sh_degree=0,full_covariance_preserved=True,
        source_keypoints_used_for_registration=True,metric_world_scale_certified=False,posthoc_pose_or_scale_fit=True,
        reference_metric_values_used=False,real_heldout_eligible=False,metric_eligible=False,script_sha256=sha(__file__))
    (dest/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','render']);p.add_argument('--ids',nargs='+',default=['HN0004','HN0039','HN0040']);args=p.parse_args()
    if args.action=='prepare':prepare(args.ids)
    else:
        for sid in args.ids:render(sid)
