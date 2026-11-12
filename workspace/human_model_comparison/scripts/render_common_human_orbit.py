"""Render existing static/frozen assets on shared source-relative orbit cameras."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common_human_orbit import H,OUT,read,save,sha,checked,inputs,native_views,output_frame


def static_renderer(sid,method,camera,cert):
    checked(cert['trace_record'],cert['trace_record_sha256']);reg=read(cert['trace_record'])
    T=np.asarray(reg['T_source_camera_from_export']);K=np.asarray(camera['K_raw']);wh=camera['render_wh']
    if method=='ours':
        from render_ours_source_pilot_v1 import render
        from hasim.shared.assets.library_cloud import GaussianAssetFrame
        checked(reg['snapshot'],reg['snapshot_sha256'])
        with np.load(reg['snapshot']) as z:
            cloud=GaussianAssetFrame(z['means'],z['quaternions_wxyz'],z['scales'],z['opacities'],z['sh'],z['local_from_appearance'])
        assert cloud.sh_degree==3
        return cloud.means,T,lambda view:render(cloud,view,K,wh),dict(native_payload=reg['snapshot'],native_payload_sha256=reg['snapshot_sha256'],appearance='full native SH degree3',renderer='native-coordinate gsplat; unchanged Gaussian parameters')
    if method=='ah':
        from render_ah_source_pilot_v2 import render
        sys.path.insert(0,str(H.parent/'assetbench/repos/ah/benchmark'));from utils import load_ply
        checked(reg['native_ply'],reg['native_ply_sha256']);gs=load_ply(reg['native_ply']).cuda()[0]
        scale=float(np.cbrt(np.linalg.det(T[:3,:3])))
        return gs[:,:3].cpu().numpy(),T,lambda view:render(gs,view,K,wh,near=.1/scale,far=500/scale),dict(native_payload=reg['native_ply'],native_payload_sha256=reg['native_ply_sha256'],appearance='native AH RGB',renderer='same source-camera gsplat path; parameters unchanged')
    checked(reg['native_payload'],reg['native_payload_sha256'])
    if method=='trellis2':
        from trellis2_source_adapter import mesh_load,basecolor_render
        mesh=mesh_load(reg['native_payload']);W,Hh=wh
        kn=torch.tensor([[K[0,0]/W,0,(K[0,2]+.5)/W],[0,K[1,1]/Hh,1-(K[1,2]+.5)/Hh],[0,0,1]],device='cuda',dtype=torch.float32)
        def render(view):
            rgb,a=basecolor_render(mesh,torch.tensor(view,device='cuda',dtype=torch.float32),kn,W,Hh)
            return rgb.permute(1,2,0),a
        return mesh.vertices.cpu().numpy(),T,render,dict(native_payload=reg['native_payload'],native_payload_sha256=reg['native_payload_sha256'],appearance='native sRGB base color unlit + own alpha; depthpeel8 SSAA2',renderer='same registered native voxel mesh path; geometry and material attributes unchanged')
    assert method=='trellis_original'
    from types import SimpleNamespace
    from audit_static_source_cameras_v1 import TRELLIS
    sys.path.insert(0,str(TRELLIS))
    from trellis.representations.gaussian import Gaussian
    from trellis.renderers.gaussian_render import render as native_render,intrinsics_to_projection
    packed=torch.load(reg['native_payload'],map_location='cpu',weights_only=True)
    assert packed['init_params']['sh_degree']==0
    gs=Gaussian(**packed['init_params'])
    for name,value in packed['params'].items():setattr(gs,name,value.cuda())
    W,Hh=wh
    kn=torch.tensor([[K[0,0]/W,0,(K[0,2]+.5)/W],[0,K[1,1]/Hh,1-(K[1,2]+.5)/Hh],[0,0,1]],device='cuda',dtype=torch.float32)
    projection=intrinsics_to_projection(kn,.01,1000.)
    pipe=SimpleNamespace(kernel_size=.1,debug=False,compute_cov3D_python=False,convert_SHs_python=False)
    def render(view):
        v=torch.tensor(view,device='cuda',dtype=torch.float32)
        c=SimpleNamespace(image_width=W,image_height=Hh,FoVx=2*np.arctan(W/(2*K[0,0])),FoVy=2*np.arctan(Hh/(2*K[1,1])),world_view_transform=v.T.contiguous(),full_proj_transform=(projection@v).T.contiguous(),camera_center=torch.linalg.inv(v)[:3,3])
        rgb=native_render(c,gs,pipe,torch.ones(3,device='cuda'))['render']
        alpha=native_render(c,gs,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(gs.get_xyz))['render'][0]
        return rgb.permute(1,2,0),alpha
    return gs.get_xyz.cpu().numpy(),T,render,dict(native_payload=reg['native_payload'],native_payload_sha256=reg['native_payload_sha256'],appearance='native Gaussian SH degree0',renderer='same source registered native rasterizer; unchanged parameters')


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);p.add_argument('--method',required=True,choices=['ours','ah','trellis_original','trellis2']);args=p.parse_args()
    sid,method=args.human_id,args.method;torch.set_num_threads(4)
    active,selected,reference,camera,cert,certificate=inputs(sid,method)
    root=OUT/sid;protocol=read(root/'protocol.json')
    assert protocol['reference_camera_sha256']==reference['camera_contract_sha256']
    assert protocol['selection_sha256']==active['selection_sha256'] and protocol['timestamp_us']==selected['t_ref_us']
    folder=root/method;folder.mkdir(exist_ok=False)
    with torch.inference_mode():
        points,T,render,provenance=static_renderer(sid,method,camera,cert)
        views,gauge=native_views(points,T,protocol['frames']);rows=[]
        for i,view in enumerate(views):
            rgb,alpha=render(view);row=output_frame(folder,i,rgb,alpha,protocol);row['native_world_to_camera']=view.tolist();rows.append(row)
            print(method,sid,'orbit',i,flush=True)
    checked(cert['source_rgb'],cert['source_rgb_sha256']);checked(cert['source_alpha'],cert['source_alpha_sha256'])
    diagnostics={}
    for name in ['rgb','alpha']:
        a=np.asarray(Image.open(rows[0][name])).astype(np.float64);b=np.asarray(Image.open(cert['source_'+name])).astype(np.float64)
        d=np.abs(a-b);diagnostics[name]=dict(max_pixel_difference=float(d.max()),mean_pixel_difference=float(d.mean()),used_as_quality_gate=False)
    archive=root/'source_archive';archive.mkdir(exist_ok=True);(archive/(sha(__file__)+'.py')).write_bytes(Path(__file__).read_bytes())
    save(folder/'manifest.json',dict(status='rendered_pending_joint_visual_review',id=sid,method=method,timestamp_us=selected['t_ref_us'],protocol_sha256=sha(root/'protocol.json'),registration=certificate,frames=rows,gauge=gauge,provenance=provenance,source_view_diagnostics=diagnostics,metric_scores_produced=False,source_metric_inputs_modified=False,script_sha256=sha(__file__)))


if __name__=='__main__':main()
