"""One common, score-blind correction: view consensus and detector input zoom.

No asset or body joint is moved; only one source-camera rigid transform is fit.
"""
import argparse,itertools,sys,copy,time,os
import numpy as np
from PIL import Image,ImageDraw
from common import *
sys.path.insert(0,str(H/'scripts'))
from analyze_registration_keypoints_v1 import triangulate

def root(job):
    return H/'reports'/('trellis_source_batch' if job['method']=='trellis_original' else 'trellis2_source_registration')/'observations'/job['id']

def consensus(obs,joint):
    obs=[r for r in obs if r.get('keypoints') is not None and r['keypoint_scores'][joint]>.5]
    if len(obs)<3:raise ValueError('less_than_three_native_observations')
    def errors(x):
        out=[]
        for r in obs:
            c=r['case']['camera'];v=np.asarray(c['world_to_camera'])@np.r_[x,1];p=np.asarray(c['effective_pixel_K'])@v[:3]
            out.append(np.linalg.norm(p[:2]/p[2]-r['keypoints'][joint]) if v[2]>0 else 1e9)
        return np.array(out)
    candidates=[]
    for pair in itertools.combinations(obs,2):
        try:
            g=triangulate([r['case']['camera'] for r in pair],[r['keypoints'][joint] for r in pair])
            if g['max_effective_ray_angle_degrees']<10:continue
            e=errors(g['xyz']);ix=np.flatnonzero(e<=8.)
            if len(ix)>=3:candidates.append((-len(ix),float(e[ix].mean()),ix))
        except ValueError:pass
    if not candidates:raise ValueError('no_three_view_consensus')
    ix=sorted(candidates,key=lambda x:(x[0],x[1]))[0][2]
    chosen=[obs[i] for i in ix]
    g=triangulate([r['case']['camera'] for r in chosen],[r['keypoints'][joint] for r in chosen],[r['keypoint_scores'][joint] for r in chosen])
    if g['max_reprojection_error_pixels']>8 or g['max_effective_ray_angle_degrees']<10:raise ValueError('unstable_consensus')
    return dict(joint_index=joint,geometry=g,inlier_cases=[r['case']['case_id'] for r in chosen],available_views=len(obs))

def register(job,records,out):
    import cv2
    ref=next(r for r in records if r['case']['role']=='reference');native=[r for r in records if r['case']['role']=='native_asset_view'];joints=[]
    if ref.get('keypoints') is not None:
        for j in range(5,17):
            if ref['keypoint_scores'][j]<=.5:continue
            try:joints.append(consensus(native,j))
            except ValueError:pass
    rec=dict(id=job['id'],method=job['method'],status='unresolved',joints=joints,script_sha256=sha(__file__),asset_sha256=job['asset_sha256'],source_observations_sha256=sha(root(job)/'keypoints.json'),LiDAR_endpoints_used=False,source_pixels_only=True)
    try:
        assert len(joints)>=6,'fewer_than_six_reliable_body_correspondences'
        xyz=np.array([j['geometry']['xyz'] for j in joints],np.float64);uv=np.array([ref['keypoints_raw_pixels'][j['joint_index']] for j in joints],np.float64)
        K=np.asarray(read(job['camera_contract'])['K_raw'],np.float64)
        ok,r,t=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP);assert ok
        # Refinement must not turn a positive SQPnP solution behind the camera.
        before=(r.copy(),t.copy());rr,tt=cv2.solvePnPRefineLM(xyz,uv,K,None,r.copy(),t.copy())
        if np.all((xyz@cv2.Rodrigues(rr)[0].T+tt[:,0])[:,2]>0):r,t=rr,tt
        else:r,t=before
        Rm=cv2.Rodrigues(r)[0];assert np.all((xyz@Rm.T+t[:,0])[:,2]>0),'nonpositive_camera_depth'
        T=np.eye(4);T[:3,:3]=Rm;T[:3,3]=t[:,0]
        px=cv2.projectPoints(xyz,r,t,K,None)[0][:,0]
        rec.update(status='candidate_pending_visual_review',T_source_camera_from_native=T.tolist(),source_reprojection_error_px=np.linalg.norm(px-uv,axis=1).tolist(),source_pixels_raw=uv.tolist(),projected_pixels_raw=px.tolist(),source_record=ref['case'])
    except Exception as error:rec['reason']=str(error)
    save(out/'registration.json',rec);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--zoom',action='store_true');a=p.parse_args()
    jobs=[j for j in read(E/'intake.json')['rows'] if j['method'] in ['trellis_original','trellis2'] and j['asset_available'] and not j.get('T_source_camera_from_native')]
    detector=pose=None
    if a.zoom:
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        import torch
        from mmdet.apis import init_detector,inference_detector
        from mmpose.apis import init_model,inference_topdown
        from mmpose.evaluation.functional import nms
        from mmpose.utils import adapt_mmdet_pipeline
        torch.set_num_threads(4);prep=H/'model_preparation/20261110';source=prep/'sources/exavatar/fitting/tools/mmpose'
        configs=[source/'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',source/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py']
        weights=[prep/'weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',prep/'weights/dw-ll_ucoco_384.pth']
        for path,digest in zip(weights,['235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9','0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07']):checked(path,digest)
        detector=init_detector(str(configs[0]),str(weights[0]),device='cuda');detector.cfg=adapt_mmdet_pipeline(detector.cfg)
        pose=init_model(str(configs[1]),str(weights[1]),device='cuda')
        for model,w in zip([detector,pose],weights):
            ck=torch.load(w,map_location='cpu',weights_only=False);model.load_state_dict({k.removeprefix('module.'):v for k,v in ck.get('state_dict',ck).items()},strict=True);model.eval()
    summary=[]
    for job in jobs:
        out=E/'registration_recovery'/('zoom' if a.zoom else 'existing')/job['method']/job['id'];out.mkdir(parents=True,exist_ok=True)
        index=read(root(job)/'keypoints.json');records=[]
        for item in index['records']:
            checked(item['record'],item['sha256']);records.append(read(item['record']))
        if a.zoom:
            changed=[]
            for r in records:
                if r['case']['role']=='reference':changed.append(r);continue
                case=r['case'];checked(case['image'],case['image_sha256']);im=Image.open(case['image']).convert('RGB');rgb=np.array(im)
                # Fixed photometric bbox includes every nonwhite pixel beyond 5/255.
                # This modifies only pose-detector resolution, never the asset.
                ys,xs=np.nonzero((255-rgb.astype(float)).max(2)>5)
                if not len(xs):changed.append(r);continue
                lo=np.array([xs.min(),ys.min()]);hi=np.array([xs.max()+1,ys.max()+1]);side=int(np.ceil(1.2*max(hi-lo)));side=max(side,16)
                lo=np.floor((lo+hi-side)/2).astype(int);image=im.crop((*lo,*(lo+side))).resize((512,512),Image.Resampling.LANCZOS)
                ip=out/(case['case_id']+'_zoom.png');image.save(ip);zoom=copy.deepcopy(r)
                with torch.inference_mode():det=inference_detector(detector,str(ip)).pred_instances.cpu().numpy()
                boxes=np.c_[det.bboxes,det.scores];boxes=boxes[(det.labels==0)&(det.scores>.3)]
                if len(boxes):boxes=boxes[nms(boxes,.3)]
                zoom.update(keypoints=None,keypoint_scores=None,zoom_image=str(ip),zoom_sha256=sha(ip),zoom_original_xy=lo.tolist(),zoom_side=side,detected_people=len(boxes))
                if len(boxes)==1:
                    with torch.inference_mode():pred=inference_topdown(pose,str(ip),boxes[:,:4])[0].pred_instances
                    xy=pred.keypoints[0];xy=(xy+.5)*(side/512)-.5+lo
                    zoom.update(keypoints=xy.tolist(),keypoint_scores=pred.keypoint_scores[0].tolist())
                changed.append(zoom)
            records=changed;save(out/'observations.json',dict(records=records,configs={str(p):sha(p) for p in configs},weights={str(p):sha(p) for p in weights}))
        result=register(job,records,out);summary.append(result);print(job['id'],job['method'],result['status'],len(result['joints']),flush=True)
        save(E/'registration_recovery'/('zoom_summary.json' if a.zoom else 'existing_summary.json'),dict(rows=summary))

if __name__=='__main__':main()
