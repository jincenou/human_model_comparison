"""Render opposite camera azimuths from the existing AH PLY, without regeneration."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

H=Path(__file__).resolve().parents[1]
repo=H.parent/'assetbench/repos/ah'
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'benchmark'))
from utils import load_ply,render_from_cameras

job=H/'runs/ah/human_pilot_v1_HN0004'
ply=job/'outputs/HN0004/gaussians.ply'
cam=json.loads((ply.parent/'ahc_source_prediction.json').read_text())
direction=np.array(cam['cam_directions'][0]);directions=[]
angles=[180,197]
for degrees in angles:
    a=np.deg2rad(degrees)
    directions.append(np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@direction)
out=job/'opposite_view';out.mkdir(exist_ok=False)
with torch.inference_mode():
    gs=load_ply(str(ply)).cuda()
    rgb,alpha=render_from_cameras(gs,np.stack(directions),[cam['dists'][0]]*2,[cam['fov'][0]]*2,cam['lwh'],512,(.5,.5,.5))
assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
for i,angle in enumerate(angles):
    Image.fromarray((rgb[i].permute(1,2,0).clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(out/f'azimuth_{angle}.png')
    Image.fromarray((alpha[i].squeeze().clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(out/f'azimuth_{angle}_alpha.png')
(out/'provenance.json').write_text(json.dumps(dict(method='Asset Harvester',sample='HN0004',ply=str(ply),
    ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),azimuth_offset_from_ahc_source_degrees=angles,
    directions=np.stack(directions).tolist(),distance=cam['dists'][0],fov=cam['fov'][0],pose_modified=False,
    regenerated=False,scores_produced=False,renderer='Official AH benchmark.utils.render_from_cameras'),indent=2)+'\n')
print(out)
