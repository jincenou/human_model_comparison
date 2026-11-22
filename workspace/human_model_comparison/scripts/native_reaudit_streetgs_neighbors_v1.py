"""Bounded native StreetGS neighbor inspection; never replace frozen exposures."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import gc
import hashlib
import itertools
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

H=Path(__file__).resolve().parents[1];R=H.parent/'vehicle_model_comparison'
ROOT=H/'reports/native_human_export_reaudit_v1'
OUT=ROOT/'streetgs/neighbors'
sys.path[:0]=[str(H/'scripts'),str(R/'scripts')]
from native_reaudit_common import read,save,sha,validations
from run_background_vehicle_scene import DriveStudio
from build_streetgs_scene_audit_gallery_v1 import fitted

OFFSETS=(-2,-1,1,2)
FONT=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')


def render(native,frame,camid):
    from datasets.base.pixel_source import get_rays
    path=native.data/'images'/f'{frame:03d}_{camid}.jpg'
    source=Image.open(path).convert('RGB');width,height=source.size
    dataset=native.dataset
    index=frame*dataset.pixel_source.num_cams+dataset.pixel_source.camera_data[camid].unique_cam_idx
    info,ci=dataset.pixel_source.get_image(index)
    info={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in info.items()}
    ci={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in ci.items()}
    raw_camera=np.loadtxt(native.data/'extrinsics'/f'{frame:03d}_{camid}.txt')
    pose=native.T@raw_camera
    assert np.abs(ci['camera_to_world'].cpu().numpy()-pose).max()<1e-4
    raw=np.loadtxt(native.data/'intrinsics'/f'{camid}.txt')
    K=np.array([[raw[0],0,raw[2]],[0,raw[1],raw[3]],[0,0,1]])
    ci.update(camera_to_world=torch.tensor(pose,dtype=torch.float32,device='cuda'),
              intrinsics=torch.tensor(K,dtype=torch.float32,device='cuda'),width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
    x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
    origins,directions,norm=get_rays(x.flatten(),y.flatten(),ci['camera_to_world'],ci['intrinsics'])
    info={k:info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
    info.update(origins=origins.reshape(height,width,3),viewdirs=directions.reshape(height,width,3),
                direction_norm=norm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
    with torch.inference_mode():result=native.trainer(info,ci,novel_view=False)
    pixels=(result['rgb'].cpu().numpy().clip(0,1)*255).round().astype('uint8')
    return source,Image.fromarray(pixels),raw_camera,K,dict(source_rgb=str(path),source_sha256=sha(path),
        native_full_rgb_uint8_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),render_wh=[width,height])


def crop_box(ann,frame,raw_camera,K,size):
    if frame not in ann['frame_idx']:return None,'identity_not_annotated_in_frame'
    j=ann['frame_idx'].index(frame);dims=np.asarray(ann['box_size'][j]);T=np.asarray(ann['obj_to_world'][j])
    corners=np.asarray(list(itertools.product([-1,1],repeat=3)))*dims[None]/2
    pts=np.c_[corners,np.ones(8)]@(np.linalg.inv(raw_camera)@T).T
    if not (pts[:,2]>.1).all():return None,'identity_box_crosses_camera_near_plane'
    uv=pts[:,:3]@K.T;uv=uv[:,:2]/uv[:,2:3]
    lo,hi=uv.min(0),uv.max(0);extent=(hi-lo)*1.2;mid=(hi+lo)/2
    low=np.floor(np.maximum(mid-extent/2,[0,0])).astype(int)
    high=np.ceil(np.minimum(mid+extent/2,size)).astype(int)
    if np.any(high-low<=15):return None,'identity_box_outside_or_too_small'
    return tuple(int(x) for x in [*low,*high]),None


def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',required=True);p.add_argument('--worker-key',default='worker');a=p.parse_args()
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype(FONT,18)
    records=validations('streetgs');groups={}
    for sid in a.ids:
        if (OUT/sid/'audit.json').exists():continue
        vp,v=records[sid];assert v['status']=='passed'
        groups.setdefault(v['checkpoint'],[]).append((sid,vp,v))
    assert a.worker_key.replace('_','').isalnum()
    statepath=OUT/(a.worker_key+'.json');state=dict(status='running',pid=__import__('os').getpid(),ids=a.ids,completed=[],started_at=time.time(),script_sha256=sha(__file__))
    save(statepath,state)
    for checkpoint,items in groups.items():
        assert shutil.disk_usage(H).free>10*1024**3
        assert sha(checkpoint)==items[0][2]['checkpoint_sha256']
        native=DriveStudio(items[0][2]);cache={}
        annotations=read(native.data/'instances/instances_info.json')
        for sid,vp,v in items:
            state.update(current_id=sid,updated_at=time.time());save(statepath,state)
            bundle=torch.load(v['bundle'],map_location='cpu',weights_only=False)
            assert sha(v['bundle'])==v['bundle_sha256']
            binding=bundle['binding'];camid=binding['native_camera_id'];center=binding['source_frame']
            ann=next(x for x in annotations.values() if x['id']==v['identity_key'].split(':')[-1])['frame_annotations']
            dest=OUT/sid;dest.mkdir(exist_ok=True);tiles=[];frames=[]
            for offset in OFFSETS:
                frame=center+offset;path=native.data/'images'/f'{frame:03d}_{camid}.jpg'
                tile=Image.new('RGB',(512,300),'#edf1f6');draw=ImageDraw.Draw(tile)
                draw.text((8,4),f'{sid} | frame {frame:03d} ({offset:+d})',font=font,fill='black')
                draw.text((8,28),'Source',font=font,fill='black');draw.text((264,28),'StreetGS native scene',font=font,fill='black')
                rec=dict(frame=frame,offset=offset)
                if not path.exists():rec.update(status='outside_native_frame_range');draw.text((8,90),'Frame unavailable',font=font,fill='red')
                else:
                    key=(frame,camid)
                    if key not in cache:cache[key]=render(native,frame,camid)
                    source,pred,raw_camera,K,proof=cache[key]
                    roi,reason=crop_box(ann,frame,raw_camera,K,source.size);rec.update(proof,roi_xyxy=roi)
                    if roi:
                        tile.paste(fitted(source.crop(roi),(256,248)),(0,52));tile.paste(fitted(pred.crop(roi),(256,248)),(256,52))
                        rec.update(status='rendered_identity_crop',visual_target_presence='pending_manual_review')
                    else:rec.update(status=reason);draw.text((8,90),'No usable same-identity box',font=font,fill='red')
                image=dest/f'offset_{offset:+d}.jpg';tile.save(image,quality=95)
                rec.update(comparison=str(image),comparison_sha256=sha(image));frames.append(rec);tiles.append(tile)
            overview=Image.new('RGB',(512*4,300),'white')
            for i,tile in enumerate(tiles):overview.paste(tile,(512*i,0))
            overview.save(dest/'comparison.jpg',quality=95)
            save(dest/'audit.json',dict(id=sid,scene=v['scene'],identity_key=v['identity_key'],center_frame=center,native_camera_id=camid,
                status='bounded_neighbors_rendered_pending_visual_review',offsets=list(OFFSETS),checkpoint=checkpoint,checkpoint_sha256=v['checkpoint_sha256'],
                export_validation=str(vp),export_validation_sha256=sha(vp),source_reference=v['source_reference'],
                native_CamPose_enabled=True,crop_rule='Same instance per-frame annotated 3D box projected to the original camera, fixed 1.2 expansion',
                full_images_saved=False,frozen_source_replaced=False,formal_metrics_changed=False,parameters_changed=False,
                overview=str(dest/'comparison.jpg'),overview_sha256=sha(dest/'comparison.jpg'),frames=frames,script_sha256=sha(__file__)))
            state['completed'].append(sid);save(statepath,state);print(sid,'neighbors saved',flush=True)
        del native,cache;gc.collect();torch.cuda.empty_cache()
    state.update(status='completed',finished_at=time.time(),current_id=None);save(statepath,state)


if __name__=='__main__':main()
