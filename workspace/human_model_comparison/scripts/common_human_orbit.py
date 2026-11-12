"""Shared display cameras with a source-projection-preserving depth gauge.

This is a nonmetric visualization protocol, separate from source scoring.
Only an isotropic change of units about the source camera is made. No asset
bbox centering, image resizing, pose edits, or image-score fitting is used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parents[1]
OUT=H/'reports/common_human_orbit_v1'
METHODS=['ours','ah','trellis_original','omnire','splatad','trellis2']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()
def save(p,d):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def checked(p,digest):
    if sha(p)!=digest:raise ValueError('Changed input: '+str(p))

def camera_views(camera):
    K=np.asarray(camera['K_raw'],np.float64)
    box=np.asarray(camera['roi_xyxy'],np.float64)
    center=np.linalg.solve(K,np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1.]))
    up=np.asarray(camera['T_world_from_camera_opencv'])[:3,:3].T@np.array([0.,0.,1.])
    up/=np.linalg.norm(up)
    skew=np.array([[0,-up[2],up[1]],[up[2],0,-up[0]],[-up[1],up[0],0]])
    frames=[]
    for i in range(16):
        angle=i*2*np.pi/16
        R=np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
        pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=center-R@center
        frames.append(dict(index=i,angle_degrees=i*22.5,camera_to_display=pose.tolist()))
    return center,up,frames

def native_views(points,source_from_native,cameras):
    points=np.asarray(points,np.float64);T=np.asarray(source_from_native,np.float64)
    assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
    source=points@T[:3,:3].T+T[:3,3]
    depth=source[:,2];positive=depth[depth>0]
    if not len(positive):raise ValueError('Native geometry has no positive source-camera depth')
    median=float(np.median(positive));S=np.eye(4);S[:3,:3]/=median
    display_from_native=S@T;views=[]
    for frame in cameras:
        view=np.linalg.inv(np.asarray(frame['camera_to_display']))@display_from_native
        scale=float(np.cbrt(np.linalg.det(view[:3,:3])))
        if not np.isfinite(scale) or scale<=0:raise ValueError('Proper similarity required')
        view[:3]/=scale
        np.testing.assert_allclose(view[:3,:3].T@view[:3,:3],np.eye(3),atol=1e-5,rtol=0)
        views.append(view)
    return views,dict(source_positive_depth_median=median,positive_depth_points=len(positive),
        all_native_points=len(points),display_from_native=display_from_native.tolist(),
        rule='Median positive source-camera depth of all native centers/vertices; no opacity, bbox or quality filtering',
        physical_scale_certified=False,asset_parameters_modified=False)

def inputs(sid,method):
    active=read(H/'inputs/active_human_benchmark.json');checked(active['selection'],active['selection_sha256'])
    selected=next(r for r in read(active['selection'])['objects'] if r['id']==sid)
    summary=read(H/'reports/source_camera_metric_cohort_v3/summary.json')
    row=next(r for r in summary['rows'] if r['id']==sid and r['method']==method)
    assert row['status']=='scored';checked(row['result'],row['result_sha256']);result=read(row['result'])
    cp=Path(result['registration_record']);checked(cp,result['registration_sha256']);cert=read(cp)
    reference=row['reference'];checked(reference['camera_contract'],reference['camera_contract_sha256'])
    camera=read(reference['camera_contract'])
    assert camera['timestamp_us']==selected['t_ref_us']==cert['timestamp_us']
    assert camera['identity_key']==selected['identity_key']==cert['identity_key']
    return active,selected,reference,camera,cert,dict(path=str(cp),sha256=sha(cp))

def prepare(sid):
    active,selected,reference,camera,_,_=inputs(sid,'ours')
    center,up,frames=camera_views(camera)
    record=dict(id=sid,identity_key=selected['identity_key'],timestamp_us=selected['t_ref_us'],
        selection_sha256=active['selection_sha256'],status='pilot_protocol_pending_visual_review',
        purpose='Fixed-pose relative source-view orbit; no physical-scale, heldout or back-view metric claim',
        reference_camera=reference['camera_contract'],reference_camera_sha256=reference['camera_contract_sha256'],
        K_raw=camera['K_raw'],render_wh=camera['render_wh'],roi_xyxy=reference['roi_xyxy'],output_wh=[512,512],
        center_display=center.tolist(),up_display=up.tolist(),camera_distance_display=float(np.linalg.norm(center)),
        background='white',angles='0,22.5,...337.5 degrees relative to source camera; not anatomical front/back labels',
        fixed_pose_and_timestamp=True,prediction_bbox_used=False,image_metric_used=False,
        depth_gauge='Each complete native geometry uses median positive source-camera depth=1; only source-origin isotropic units change',
        source_projection_preserved=True,physical_scale_certified=False,
        limitations=['Point/vertex sampling density affects the depth gauge; display is not a metric reconstruction test',
                    'Model pose, proportions, orientation errors, fragments and absent surfaces remain visible',
                    'No real side/back reference is assumed; orbit images are not added to source metric scores'],
        frames=frames,script_sha256=sha(__file__))
    folder=OUT/sid;folder.mkdir(parents=True,exist_ok=True);path=folder/'protocol.json'
    if path.exists():assert read(path)==record,'Existing orbit protocol differs; inspect before replacing'
    else:save(path,record)
    archive=folder/'source_archive';archive.mkdir(exist_ok=True);(archive/(sha(__file__)+'.py')).write_bytes(Path(__file__).read_bytes())
    return record

def output_frame(folder,index,rgb,alpha,protocol):
    from PIL import Image
    from human_source_metrics_v1 import crop
    def array(x):
        if hasattr(x,'detach'):x=x.detach().cpu().numpy()
        x=np.asarray(x);assert np.isfinite(x).all()
        return np.floor(np.clip(x,0,1)*255+.5).astype(np.uint8)
    rgb,alpha=array(rgb),array(alpha)
    if alpha.ndim==3:alpha=alpha[...,0]
    assert rgb.shape[:2]==alpha.shape and list(rgb.shape[1::-1])==protocol['render_wh']
    row=dict(index=index,angle_degrees=protocol['frames'][index]['angle_degrees'])
    for name,data in [('rgb',rgb),('alpha',alpha)]:
        p=folder/f'{index:02d}_{name}.png';Image.fromarray(data).save(p);row[name]=str(p);row[name+'_sha256']=sha(p)
        p=folder/f'{index:02d}_{name}_roi.png';Image.fromarray(crop(data,protocol['roi_xyxy'],name=='alpha')).save(p)
        row[name+'_roi']=str(p);row[name+'_roi_sha256']=sha(p)
    return row
