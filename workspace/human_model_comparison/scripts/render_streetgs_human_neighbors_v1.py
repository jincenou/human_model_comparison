"""Inspect all native camera frames around an apparently missing human."""
from human_asset_benchmark.paths import expand as _hmc_expand
import base64
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
sys.path[:0] = [str(H/'scripts'), str(R/'scripts')]
from extract_streetgs_human_background_v1 import read, save, sha
from run_background_vehicle_scene import DriveStudio
from build_streetgs_scene_audit_gallery_v1 import fitted


def main():
    torch.set_num_threads(2)
    sid = 'HN0108'
    root = H/'reports/streetgs_human_scene_audit_v1'
    out = root/'neighbors_HN0108';out.mkdir(parents=True, exist_ok=True)
    rec = read(H/'exports/streetgs_human_probe_v1/objects'/sid/'validation.json')
    pack = torch.load(rec['bundle'], map_location='cpu', weights_only=False)
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    camid = pack['binding']['native_camera_id']; center = pack['binding']['source_frame']
    native = DriveStudio(rec); trainer, dataset = native.trainer, native.dataset
    from datasets.base.pixel_source import get_rays
    info_all = read(native.data/'instances/instances_info.json')
    ann = next(v for v in info_all.values() if v['id'] == rec['identity_key'].split(':')[-1])['frame_annotations']
    font_path = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, 18)
    tiles, records, sections = [], [], []
    paths = sorted((native.data/'images').glob(f'*_{camid}.jpg'))
    with torch.no_grad():
        for path in paths:
            frame = int(path.stem.split('_')[0])
            source = Image.open(path).convert('RGB'); width, height = source.size
            index = frame * dataset.pixel_source.num_cams + dataset.pixel_source.camera_data[camid].unique_cam_idx
            info, ci = dataset.pixel_source.get_image(index)
            info = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in info.items()}
            ci = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in ci.items()}
            raw_camera = np.loadtxt(native.data/'extrinsics'/f'{frame:03d}_{camid}.txt')
            pose = native.T @ raw_camera
            assert float(np.abs(ci['camera_to_world'].cpu().numpy()-pose).max()) < 1e-4
            raw = np.loadtxt(native.data/'intrinsics'/f'{camid}.txt')
            K = np.array([[raw[0],0,raw[2]],[0,raw[1],raw[3]],[0,0,1]])
            ci.update(camera_to_world=torch.tensor(pose,dtype=torch.float32,device='cuda'),
                      intrinsics=torch.tensor(K,dtype=torch.float32,device='cuda'),
                      width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
            x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
            origins,directions,norm=get_rays(x.flatten(),y.flatten(),ci['camera_to_world'],ci['intrinsics'])
            info={k:info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
            info.update(origins=origins.reshape(height,width,3),viewdirs=directions.reshape(height,width,3),
                        direction_norm=norm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
            result=trainer(info,ci,novel_view=False)
            pred=Image.fromarray((result['rgb'].cpu().numpy().clip(0,1)*255).round().astype('uint8'))
            folder=out/f'{frame:03d}';folder.mkdir(exist_ok=True)
            pred.save(folder/'scene_native.png')
            roi=None
            if frame in ann['frame_idx']:
                j=ann['frame_idx'].index(frame)
                dims=np.asarray(ann['box_size'][j]);Twobj=np.asarray(ann['obj_to_world'][j])
                corners=np.asarray(list(itertools.product([-1,1],repeat=3)))*dims[None]/2
                pts=np.c_[corners,np.ones(8)] @ (np.linalg.inv(raw_camera)@Twobj).T
                if (pts[:,2]>.1).all():
                    uv=pts[:,:3]@K.T;uv=uv[:,:2]/uv[:,2:3]
                    lo,hi=uv.min(0),uv.max(0);extent=(hi-lo)*1.2;mid=(hi+lo)/2
                    x0,y0=np.floor(np.maximum(mid-extent/2,[0,0])).astype(int)
                    x1,y1=np.ceil(np.minimum(mid+extent/2,[width,height])).astype(int)
                    if x1-x0>15 and y1-y0>15:roi=(int(x0),int(y0),int(x1),int(y1))
            tile=Image.new('RGB',(512,288),'#edf1f6');draw=ImageDraw.Draw(tile)
            draw.text((8,5),f'Frame {frame:03d} ({frame-center:+d})'+('  FROZEN' if frame==center else ''),font=font,fill='black')
            if roi:
                tile.paste(fitted(source.crop(roi),(256,256)),(0,32));tile.paste(fitted(pred.crop(roi),(256,256)),(256,32))
            else:
                tile.paste(fitted(source,(256,256)),(0,32));tile.paste(fitted(pred,(256,256)),(256,32))
                draw.text((8,270),'No in-frame identity box',font=font,fill='red')
            tile.save(folder/'comparison.jpg',quality=95)
            tiles.append(tile)
            full=Image.new('RGB',(1600,480),'white');d=ImageDraw.Draw(full)
            for col,(name,img) in enumerate([('Source',source),('StreetGS native scene',pred)]):
                im=img.copy()
                if roi:ImageDraw.Draw(im).rectangle(roi,outline='red',width=4)
                d.text((col*800+8,4),name,font=font,fill='black');full.paste(im.resize((800,450)),(col*800,30))
            full.save(folder/'full_comparison.jpg',quality=92)
            data='data:image/jpeg;base64,'+base64.b64encode((folder/'comparison.jpg').read_bytes()).decode()
            full_data='data:image/jpeg;base64,'+base64.b64encode((folder/'full_comparison.jpg').read_bytes()).decode()
            sections.append(f'<section><h2>Frame {frame:03d}（相对冻结帧 {frame-center:+d}）</h2><img class="crop" src="{data}"><details><summary>完整源图与场景</summary><img src="{full_data}"></details></section>')
            records.append(dict(frame=frame,offset=frame-center,source_rgb=str(path),source_sha256=sha(path),
                render=str(folder/'scene_native.png'),render_sha256=sha(folder/'scene_native.png'),
                roi_xyxy=roi,identity_annotation_available=frame in ann['frame_idx']))
            print(frame,roi,flush=True)
    sheet=Image.new('RGB',(512*4,288*((len(tiles)+3)//4)),'#ccd3df')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*512,(i//4)*288))
    sheet.save(out/'overview.jpg',quality=95)
    html='''<!doctype html><meta charset="utf-8"><title>HN0108 StreetGS 前后帧核查</title><style>body{font:16px/1.6 system-ui;background:#edf1f6;max-width:1600px;margin:24px auto}section{background:white;padding:20px;margin:20px 0}img{max-width:100%}.crop{width:768px}summary{cursor:pointer}</style><h1>HN0108：StreetGS 同一相机全部23帧</h1><p>每幅局部图左为原照片，右为原生完整场景渲染。Frame 007 是冻结源帧；序号为处理后的帧编号，不是秒数。按同一实例的每帧标注框定位，各列同坐标裁切。未更换冻结帧、没有修改训练模型或正式指标；目标是否在原图中可见需结合照片判断。</p>'''+''.join(sections)
    (out/'index.html').write_text(html)
    save(out/'audit.json',dict(id=sid,scene=rec['scene'],identity_key=rec['identity_key'],center_frame=center,
        native_camera_id=camid,count=len(records),checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],
        native_CamPose_enabled=True,trained=False,formal_metrics_updated=False,frames=records,script_sha256=sha(__file__)))


if __name__=='__main__':main()
