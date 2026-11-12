"""Fixed native views for semantic triangulation; not the final common-camera orbit."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

os.environ.setdefault('SPCONV_ALGO','native')
os.environ.setdefault('ATTN_BACKEND','flash-attn')
from audit_static_source_cameras_v1 import H,P,TRELLIS,sha,read,checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--human-id',required=True,
        choices=['HN0004','HN0035','HN0039','HN0040']);sid=parser.parse_args().human_id
    sys.path.insert(0,str(TRELLIS))
    import torch
    from trellis.utils import render_utils
    from trellis.representations.gaussian import Gaussian
    audit=read(H/'reports/static_source_camera_audit_v1'/(sid+'.json'))
    payload=Path(audit['trellis_original']['native_payload'])
    checked(payload,audit['trellis_original']['native_payload_sha256'])
    packed=torch.load(payload,map_location='cpu',weights_only=True)
    if packed['init_params']['sh_degree']!=0:raise ValueError('Unexpected SH order; do not truncate')
    gs=Gaussian(**packed['init_params'])
    for name,value in packed['params'].items():
        if not torch.isfinite(value).all():raise ValueError('Non-finite Gaussian parameters')
        setattr(gs,name,value.cuda())
    yaws=[2*np.pi*i/16 for i in range(16)]
    views,intrinsics=render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws,[.2]*16,2.,40.)
    white=render_utils.get_renderer(gs,resolution=512,bg_color=(1,1,1))
    alpha_renderer=render_utils.get_renderer(gs,resolution=512,bg_color=(0,0,0))
    # Check the actual compiled rasterizer's pixel-center convention. A source
    # code equation alone is insufficient evidence for the loaded binary.
    probe=Gaussian(**packed['init_params'])
    probe.from_scaling(torch.full((1,3),max(.02,2*float(probe.mininum_kernel_size)),device='cuda'))
    probe.from_rotation(torch.tensor([[1.,0.,0.,0.]],device='cuda'))
    probe.from_opacity(torch.tensor([[.9]],device='cuda'))
    probe.from_features(torch.zeros((1,3,1),device='cuda'))
    projection_checks=[]
    with torch.inference_mode():
        for index,point in [(0,[.1,-.15,.12]),(4,[-.13,.08,-.07]),(8,[.05,.1,.2])]:
            xyz=torch.tensor([point],dtype=torch.float32,device='cuda');probe.from_xyz(xyz)
            response=alpha_renderer.render(probe,views[index],intrinsics[index],colors_overwrite=torch.ones_like(xyz))['color'][0]
            yy,xx=torch.meshgrid(torch.arange(512,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
            centroid=torch.stack([(response*xx).sum(),(response*yy).sum()])/response.sum()
            native=(views[index][:3,:3] @ xyz[0])+views[index][:3,3]
            projected=intrinsics[index] @ native;expected=projected[:2]/projected[2]*512-.5
            error=float(torch.abs(centroid-expected).max())
            if not np.isfinite(error) or error>.1:raise ValueError(f'Compiled rasterizer pixel convention failed: {error}')
            projection_checks.append(dict(view_index=index,point=point,centroid=centroid.cpu().tolist(),
                                          expected_pixel=expected.cpu().tolist(),max_error_pixels=error))
    out=H/'exports/trellis_registration_views_v1'/sid;out.mkdir(parents=True,exist_ok=False)
    cases=[]
    with torch.inference_mode():
        for i,(view,K) in enumerate(zip(views,intrinsics)):
            rgb=white.render(gs,view,K)['color']
            opacity_rgb=alpha_renderer.render(gs,view,K,colors_overwrite=torch.ones_like(gs.get_xyz))['color']
            if not torch.isfinite(rgb).all() or not torch.isfinite(opacity_rgb).all():
                raise ValueError('Non-finite native rendering')
            if float((opacity_rgb-opacity_rgb[:1]).abs().max())>1e-6:
                raise ValueError('Opacity color override inconsistent across channels')
            alpha=opacity_rgb[0]
            if float(alpha.min()) < -1e-6 or float(alpha.max())>1+1e-5:raise ValueError('Invalid alpha')
            # Keep upstream preview truncation for this unscored native probe.
            image=np.clip(rgb.permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8)
            alpha_image=np.floor(alpha.clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)
            image_path=out/f'view_{i:02d}.png';alpha_path=out/f'view_{i:02d}_alpha.png'
            Image.fromarray(image).save(image_path);Image.fromarray(alpha_image).save(alpha_path)
            if i==0:
                original=payload.parent/'reload_0.png'
                native_validation=read(payload.parent/'reload_validation.json')
                checked(original,native_validation['views'][0]['sha256'])
                difference=np.abs(image.astype(np.int16)-np.asarray(Image.open(original)).astype(np.int16))
                reload_check=dict(max_pixel_difference=int(difference.max()),mean_pixel_difference=float(difference.mean()))
                if difference.max()>1 or difference.mean()>.01:
                    raise ValueError('Fixed first camera no longer reproduces native reload')
            kn=K.cpu().numpy().astype(np.float64);extr=view.cpu().numpy().astype(np.float64)
            # Native wrapper + C++ ndc2Pix: the y principal-point sign differs
            # from ordinary K. Here every fixed view has exactly centered K.
            effective=np.array([[512*kn[0,0],0,512*kn[0,2]-.5],
                                [0,512*kn[1,1],512*(1-kn[1,2])-.5],[0,0,1.]])
            if not np.allclose(kn[:2,2],[.5,.5]):raise ValueError('Only centered native cameras are permitted in this probe')
            cases.append(dict(case_id=f'{sid}_orbit_{i:02d}',id=sid,role='native_asset_view',view_index=i,
                image=str(image_path),image_sha256=sha(image_path),own_alpha=str(alpha_path),own_alpha_sha256=sha(alpha_path),
                identity_key=audit['identity_key'],timestamp_us=audit['timestamp_us'],
                native_payload=str(payload),native_payload_sha256=audit['trellis_original']['native_payload_sha256'],
                camera=dict(yaw_radians=yaws[i],pitch_radians=.2,radius=2.,fov_degrees=40.,
                    normalized_intrinsics=kn.tolist(),world_to_camera=extr.tolist(),effective_pixel_K=effective.tolist(),
                    projection_matrix=(effective @ extr[:3]).tolist(),image_wh=[512,512],
                    pixel_convention='native C++ ndc2Pix; integer pixel sampling; centered intrinsics',source_orientation_known=False)))
    for name,value in packed['params'].items():
        if not torch.equal(value,getattr(gs,name).cpu()):raise ValueError('Rendering changed native Gaussian parameters')
    sources=[Path(__file__),TRELLIS/'trellis/utils/render_utils.py',TRELLIS/'trellis/renderers/gaussian_render.py',
             TRELLIS/'trellis/representations/gaussian/gaussian_model.py',
             P/'sources/recon02_diff_gaussian_rasterization_local/cuda_rasterizer/forward.cu',
             P/'sources/recon02_diff_gaussian_rasterization_local/cuda_rasterizer/auxiliary.h']
    import diff_gaussian_rasterization as rasterizer_module
    sources.extend([Path(rasterizer_module.__file__),Path(rasterizer_module._C.__file__)])
    report=dict(status='native_fixed_views_rendered_pending_semantic_registration',id=sid,cases=cases,
                native_first_view_equivalence=reload_check,parameter_equality_after_render=True,
                compiled_rasterizer_projection_checks=projection_checks,
                source_sha256={str(p):sha(p) for p in sources},
                own_alpha_method='unit RGB override and black background with unchanged native opacity/covariance',
                background='white',rgb_quantization='native clip then truncate; unscored diagnostic only',
                common_orbit_accepted=False,common_camera_valid=False,metric_eligible=False,scores_produced=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(id=sid,views=len(cases),status=report['status'],native_equivalence=reload_check)),flush=True)


if __name__=='__main__':main()
