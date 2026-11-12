"""Render Ours frozen assets using the traced estimated source registration."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from export_ours_frozen_time_v1 import H,sha
from hasim.shared.assets.library_cloud import GaussianAssetFrame,similarity

def tensor(a):return torch.as_tensor(np.array(a),device='cuda',dtype=torch.float32).contiguous()
def render(cloud,view,K,wh):
    xyz=tensor(cloud.means);eye=tensor(np.linalg.inv(view)[:3,3]);directions=xyz-eye
    directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
    colors=(spherical_harmonics(cloud.sh_degree,(directions@tensor(cloud.local_from_appearance)).contiguous(),tensor(cloud.sh))+.5).clamp_min(0)
    rgb,alpha,_=rasterization(means=xyz,quats=tensor(cloud.quaternions),scales=tensor(cloud.scales),opacities=tensor(cloud.opacities),colors=colors,
        viewmats=tensor(view)[None],Ks=tensor(K)[None],width=wh[0],height=wh[1],backgrounds=torch.ones((1,3),device='cuda'),
        packed=False,render_mode='RGB',near_plane=.01,far_plane=1000,eps2d=.3)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    return rgb[0].clamp(0,1).cpu().numpy(),alpha[0,...,0].clamp(0,1).cpu().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--human-id',required=True);sid=ap.parse_args().human_id
    path=H/'configs/human_source_common_v1/ours_registration_batch_v3'/(sid+'.json');reg=json.loads(path.read_text())
    cp=Path(reg['camera_contract']);assert sha(cp)==reg['camera_contract_sha256'];c=json.loads(cp.read_text())
    snap=Path(reg['snapshot']);assert sha(snap)==reg['snapshot_sha256']
    with np.load(snap,allow_pickle=False) as z:cloud=GaussianAssetFrame(z['means'],z['quaternions_wxyz'],z['scales'],z['opacities'],z['sh'],z['local_from_appearance'])
    assert cloud.sh_degree==3
    T=np.array(reg['T_source_camera_from_export']);K=np.array(c['K_raw']);wh=c['render_wh']
    mapped=cloud.transformed(T)
    scale,rot=similarity(T)
    # Equivalent rigid camera in original coordinates retains native appearance and covariance.
    view=np.eye(4);view[:3,:3]=rot;view[:3,3]=T[:3,3]/scale
    with torch.inference_mode():
        rgb,alpha=render(mapped,np.eye(4),K,wh)
        native_rgb,native_alpha=render(cloud,view,K,wh)
    error=np.abs(rgb-native_rgb);ae=np.abs(alpha-native_alpha)
    out=H/'exports/ours_source_camera_batch_v3'/sid;out.mkdir(parents=True,exist_ok=False)
    for name,a in [('source_rgb',rgb),('source_alpha',alpha)]:Image.fromarray(np.floor(a*255+.5).astype(np.uint8)).save(out/(name+'.png'))
    report=dict(status='source_camera_rendered_pending_visual_and_registration_review',id=sid,
        registration=str(path),registration_sha256=sha(path),camera_contract=str(cp),camera_contract_sha256=sha(cp),
        source_rgb=str(out/'source_rgb.png'),source_alpha=str(out/'source_alpha.png'),
        source_rgb_sha256=sha(out/'source_rgb.png'),source_alpha_sha256=sha(out/'source_alpha.png'),
        preserved_sh_degree=3,prediction_encoding='white_composited_uint8',render_wh=wh,K_raw=c['K_raw'],
        covariance_and_appearance_transformed_together=True,
        coordinate_equivalence=dict(rgb_max=float(error.max()),rgb_mean=float(error.mean()),alpha_max=float(ae.max()),alpha_mean=float(ae.mean())),
        source_pose_estimated=True,metric_eligible=False,scored=False,script_sha256=sha(__file__))
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
