"""Fresh-process 16-view fixed-pose review, preserving SH appearance coordinates."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from gsplat import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from export_ours_frozen_time_v1 import H,sha
from ours_coordinate_contract_v1 import evaluate,raster_difference,validate_raster
from hasim.shared.assets.library_cloud import GaussianAssetFrame


def tensor(x):return torch.as_tensor(np.array(x),device='cuda',dtype=torch.float32).contiguous()


def render(cloud,camera,k):
    means=tensor(cloud.means);position=tensor(camera[:3,3])
    directions=means-position;directions=directions/torch.linalg.vector_norm(directions,dim=-1,keepdim=True)
    appearance=(directions@tensor(cloud.local_from_appearance)).contiguous()
    colors=(spherical_harmonics(cloud.sh_degree,appearance,tensor(cloud.sh))+.5).clamp_min(0)
    rgb,alpha,_=rasterization(means=means,quats=tensor(cloud.quaternions),scales=tensor(cloud.scales),
        opacities=tensor(cloud.opacities),colors=colors,viewmats=tensor(np.linalg.inv(camera))[None],
        Ks=tensor(k)[None],width=512,height=512,backgrounds=torch.ones((1,3),device='cuda'),
        packed=False,render_mode='RGB',near_plane=.01,far_plane=100,eps2d=.3)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    return rgb[0].clamp(0,1).cpu().numpy(),alpha[0,...,0].clamp(0,1).cpu().numpy()


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--human-id',required=True);args=parser.parse_args()
    folder=H/'exports/ours_frozen_time_v1'/args.human_id
    provenance=json.loads((folder/'provenance.json').read_text());snapshot=folder/'inplace.npz'
    assert sha(snapshot)==provenance['output_sha256']['inplace.npz']
    with np.load(snapshot,allow_pickle=False) as z:
        cloud=GaussianAssetFrame(z['means'],z['quaternions_wxyz'],z['scales'],z['opacities'],z['sh'],z['local_from_appearance'])
    assert cloud.sh_degree==3
    out=folder/'reload_review_v3';out.mkdir(exist_ok=False)
    # Fixed pilot settings, independent of predicted bbox; not a source-camera score.
    target=np.array([0.,0.,.95]);radius=3.2;camera_height=1.25
    k=np.array([[600.,0,256],[0,600.,256],[0,0,1.]])
    cameras=[];images=[];areas=[]
    with torch.inference_mode():
        for index in range(16):
            angle=index*2*np.pi/16;eye=np.array([radius*np.cos(angle),radius*np.sin(angle),camera_height])
            forward=target-eye;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            camera=np.eye(4);camera[:3,:3]=np.stack([right,down,forward],axis=1);camera[:3,3]=eye
            rgb,alpha=render(cloud,camera,k);area=int((alpha>.5).sum())
            im=Image.fromarray((rgb*255).round().astype('uint8'));im.save(out/f'view_{index:02d}.png');images.append(im)
            Image.fromarray((alpha*255).round().astype('uint8')).save(out/f'alpha_{index:02d}.png')
            areas.append(area);cameras.append(camera.tolist())
        # Rotate covariance, geometry, appearance frame and camera together. RGB must agree.
        angle=.73;rotation=np.eye(4);rotation[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        original_rgb,original_alpha=render(cloud,np.array(cameras[0]),k)
        moved=cloud.transformed(rotation)
        analytic=evaluate(cloud,np.array(cameras[0]),moved,rotation@np.array(cameras[0]));assert analytic['passed'],analytic
        rotated_rgb,rotated_alpha=render(moved,rotation@np.array(cameras[0]),k)
        rgb_error=float(np.abs(original_rgb-rotated_rgb).max());alpha_error=float(np.abs(original_alpha-rotated_alpha).max())
        rgb_stats=raster_difference(original_rgb,rotated_rgb);alpha_stats=raster_difference(original_alpha,rotated_alpha)
        assert validate_raster(rgb_stats) and validate_raster(alpha_stats),(rgb_stats,alpha_stats)
        repeat_rgb,repeat_alpha=render(cloud,np.array(cameras[0]),k)
        assert np.array_equal(original_rgb,repeat_rgb) and np.array_equal(original_alpha,repeat_alpha)
        previous=[]
        for index in range(16):
            old=folder/'reload_review_v2'/f'view_{index:02d}.png'
            if old.exists():
                delta=np.abs(np.asarray(Image.open(old)).astype('int16')-np.asarray(Image.open(out/old.name)).astype('int16'))
                assert delta.max()==0,('Original saved render changed',old,int(delta.max()))
                previous.append(dict(view=index,old_sha256=sha(old),new_sha256=sha(out/old.name),max_byte_difference=int(delta.max())))
    sheet=Image.new('RGB',(512*4,512*4),'white')
    for index,im in enumerate(images):sheet.paste(im,((index%4)*512,(index//4)*512))
    sheet.save(out/'contact.jpg',quality=90)
    result=dict(status='fixed_time_native_snapshot_reload_passed',id=args.human_id,snapshot_sha256=sha(snapshot),
        source_time_seconds=provenance['source_time_seconds'],gaussians=len(cloud.means),sh_degree=cloud.sh_degree,
        coordinate_variant='inplace body coordinates; root restoration separately preserved, world registration pending',
        render_script_sha256=sha(__file__),renderer='gsplat unpacked projection, full covariance and all SH bands evaluated in retained appearance frame',
        adapter_change='v3 separates analytic frame invariance from float32 alpha-cutoff raster tolerance; model, input, orbit, SH and native render unchanged',
        analytic_coordinate_check=analytic,rigid_rotation_rgb_statistics=rgb_stats,rigid_rotation_alpha_statistics=alpha_stats,
        previous_native_render_equality=previous,repeat_render_exact_equal=True,
        invariance_contract_sha256=sha(H/'scripts/ours_coordinate_contract_v1.py'),
        raster_acceptance='Analytic geometry/covariance/appearance required; rotated max<=1/255+1e-5, mean<1e-6, RMS<2e-5; same-input repeat exact',
        old_v2_pixel_threshold_would_pass=bool(rgb_error<.002 and alpha_error<.002),
        K=k.tolist(),camera_to_world=cameras,background='white',alpha_pixels_gt_half=areas,
        rigid_coordinate_invariance_max_rgb_error=rgb_error,rigid_coordinate_invariance_max_alpha_error=alpha_error,
        camera_settings_status='fixed pilot values, not final cross-method orbit freeze',
        reload_valid=True,common_camera_valid=False,metric_eligible=False,scored=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
