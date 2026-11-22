"""Actual native geometry/alpha adapters. No Gaussian-centre surface substitute."""
import sys
from types import SimpleNamespace
import numpy as np
from common import *
sys.path[:0]=[str(H/'scripts'),str(R/'scripts'),str(R/'evaluation_nuscenes_v1/scripts'),str(G/'scripts')]

class GaussianDepth:
    def __init__(self, job):
        import torch
        self.torch=torch;self.job=job;self.method=job['method'];self.gs=None
        torch.set_num_threads(4)
        checked(job['asset'],job['asset_sha256'])
        self.options=dict(packed=False,rasterize_mode='classic',eps2d=.3,radius_clip=0.)
        if job.get('decoded_arrays'):
            with np.load(job['decoded_arrays']) as z:arrays=[z[k] for k in ['means','quats','scales','opacity']]
            self.options.update(rasterize_mode='antialiased',eps2d=.1)
        elif self.method=='ours':
            with np.load(job['asset']) as z:
                assert z['sh'].shape[1]==16
                arrays=[z[k] for k in ['means','quaternions_wxyz','scales','opacities']]
        elif self.method=='ah':
            sys.path.insert(0,str(H.parent/'assetbench/repos/ah/benchmark'))
            from utils import load_ply
            gs=load_ply(job['asset'])[0]
            arrays=[gs[:,:3],gs[:,7:11],gs[:,4:7],gs[:,3]]
        elif self.method=='trellis_original':
            from audit_static_source_cameras_v1 import TRELLIS
            sys.path.insert(0,str(TRELLIS))
            from trellis.representations.gaussian import Gaussian
            packed=torch.load(job['asset'],map_location='cpu',weights_only=True)
            self.gs=Gaussian(**packed['init_params'])
            for key,v in packed['params'].items():setattr(self.gs,key,v.cuda())
            arrays=[self.gs.get_xyz,self.gs.get_rotation,self.gs.get_scaling,self.gs.get_opacity.reshape(-1)]
            self.options.update(rasterize_mode='antialiased',eps2d=.1)
        else:raise ValueError('Use certified scene decode for '+self.method)
        self.arrays=[self.tensor(a) for a in arrays]
        self.arrays[3]=self.arrays[3].reshape(-1)
        assert all(torch.isfinite(a).all() for a in self.arrays)
        if self.method!='trellis_original' or job.get('decoded_arrays'):
            from gsplat.rendering import rasterization
            self.rasterization=rasterization

    def tensor(self,a):return self.torch.as_tensor(a,device='cuda',dtype=self.torch.float32).contiguous()

    def render(self,V,K,n,near,far,ortho=False,force_gsplat=False):
        torch=self.torch
        if self.method=='trellis_original' and self.gs is not None and not ortho and not force_gsplat:
            from trellis.renderers.gaussian_render import render,intrinsics_to_projection
            v=self.tensor(V);kn=self.tensor([[K[0,0]/n,0,K[0,2]/n],[0,K[1,1]/n,1-K[1,2]/n],[0,0,1]])
            proj=intrinsics_to_projection(kn,near,far)
            camera=SimpleNamespace(image_width=n,image_height=n,FoVx=2*np.arctan(n/(2*K[0,0])),FoVy=2*np.arctan(n/(2*K[1,1])),world_view_transform=v.T.contiguous(),full_proj_transform=(proj@v).T.contiguous(),camera_center=torch.linalg.inv(v)[:3,3])
            pipe=SimpleNamespace(kernel_size=.1,debug=False,compute_cov3D_python=False,convert_SHs_python=False)
            z=(self.arrays[0]@v[:3,:3].T+v[:3,3])[:,2]
            colors=torch.stack([torch.ones_like(z),z,torch.zeros_like(z)],dim=1)
            with torch.inference_mode():im=render(camera,self.gs,pipe,torch.zeros(3,device='cuda'),override_color=colors)['render']
            a=im[0];ed=torch.where(a>0,im[1]/a.clamp_min(1e-20),0)
        else:
            options=dict(self.options)
            if ortho:options['camera_model']='ortho'
            with torch.inference_mode():
                im,alpha,_=self.rasterization(*self.arrays,torch.zeros_like(self.arrays[0]),self.tensor(V)[None],self.tensor(K)[None],n,n,near_plane=near,far_plane=far,render_mode='RGB+ED',**options)
            ed=im[0,...,-1];a=alpha[0,...,0]
        assert torch.isfinite(ed).all() and torch.isfinite(a).all()
        return ed.cpu().numpy(),a.cpu().numpy()

class VoxelDepth:
    def __init__(self,job,layers=8):
        import torch
        import nvdiffrast.torch as dr
        from trellis2_source_adapter import mesh_load
        checked(job['asset'],job['asset_sha256'])
        self.torch=torch;self.dr=dr;self.job=job;self.mesh=mesh_load(job['asset']);self.layers=layers
        self.context=dr.RasterizeCudaContext(device='cuda');self.arrays=None;self.diagnostics={}

    def tensor(self,a):return self.torch.as_tensor(a,device='cuda',dtype=self.torch.float32).contiguous()

    def render(self,V,K,n,near,far,ortho=False,layers=None):
        if layers is None:
            for count in [8,16,32,64]:
                result=self.render(V,K,n,near,far,ortho,layers=count)
                if self.diagnostics['next_layer_max_remaining']<=1e-4:return result
            raise ValueError('Voxel depth unresolved after 64 layers; remaining transmission exceeds 1e-4')
        torch=self.torch;dr=self.dr;mesh=self.mesh
        import torch.nn.functional as F
        from flex_gemm.ops.grid_sample import grid_sample_3d
        from trellis2.renderers.pbr_mesh_renderer import intrinsics_to_projection
        layers=layers or self.layers;v=self.tensor(V)
        if ortho:
            proj=self.tensor([[2*K[0,0]/n,0,0,2*K[0,2]/n-1],[0,2*K[1,1]/n,0,2*K[1,2]/n-1],[0,0,2/(far-near),-(far+near)/(far-near)],[0,0,0,1]])
        else:
            kn=self.tensor([[K[0,0]/n,0,K[0,2]/n],[0,K[1,1]/n,1-K[1,2]/n],[0,0,1]])
            proj=intrinsics_to_projection(kn,near,far)
        vertices=mesh.vertices[None];vh=torch.cat([vertices,torch.ones_like(vertices[...,:1])],-1)
        clip=vh@(proj@v).T;nn=n*2;wh=(nn,nn)
        alpha=torch.zeros((*wh,1),device='cuda');dz=torch.zeros_like(alpha)
        coords=torch.cat([torch.zeros_like(mesh.coords[...,:1]),mesh.coords],-1)
        last_added=0.;tail_max=0.;tail_remaining=0.
        with torch.inference_mode(),dr.DepthPeeler(self.context,clip,mesh.faces,wh) as peeler:
            for layer in range(layers+1):
                rast,_=peeler.rasterize_next_layer();hit=rast[0,...,-1]>0
                if not bool(hit.any()):break
                xyz=dr.interpolate(vertices,rast,mesh.faces)[0][0]
                position=xyz[hit];query=((position-mesh.origin)/mesh.voxel_size).reshape(1,-1,3)
                sampled=[]
                for start in range(0,query.shape[1],250000):
                    sampled.append(grid_sample_3d(mesh.attrs,coords,mesh.voxel_shape,query[:,start:start+250000],mode='trilinear')[0])
                attrs=torch.cat(sampled,0);a=attrs[:,mesh.layout['alpha']].clamp(0,1)
                remaining=(1-alpha[hit]);weight=remaining*a
                if layer==layers:
                    tail_max=float(weight.max());tail_remaining=float(remaining.max());break
                z=position@v[2,:3]+v[2,3]
                alpha[hit]+=weight;dz[hit]+=weight*z[:,None]
                last_added=float(weight.max())
        def down(x):return F.interpolate(x.permute(2,0,1)[None],(n,n),mode='bilinear',align_corners=False,antialias=True)[0,0]
        a=down(alpha);d=down(dz);ed=torch.where(a>0,d/a.clamp_min(1e-20),0)
        self.diagnostics=dict(layers_requested=layers,last_layer_max_weight=last_added,next_layer_max_weight=tail_max,next_layer_max_remaining=tail_remaining,ssaa=2)
        assert torch.isfinite(ed).all() and torch.isfinite(a).all()
        return ed.cpu().numpy(),a.cpu().numpy()

def query(adapter,arrays,A,resolution=1024):
    from geometry_core import sector_ids,sector_rotation,bilinear_accumulated
    _,scale=rigid_part(A);n=resolution;K=np.array([[n/2,0,n/2],[0,n/2,n/2],[0,0,1.]])
    pred=np.zeros(len(arrays['q']));alpha=np.zeros_like(pred);sectors=sector_ids(arrays['u']);origin=arrays['o'][0]
    assert np.allclose(arrays['o'],origin,atol=1e-8)
    diagnostics=[]
    for sector in np.unique(sectors):
        ix=np.flatnonzero(sectors==sector);C=sector_rotation(sector)
        V=np.eye(4);V[:3,:3]=C;V[:3,3]=-C@origin;V=V@A;V[:3]/=scale
        ed,a=adapter.render(V,K,n,.1/scale,200/scale)
        uc=arrays['u'][ix]@C.T;pixel=uc@K.T;xy=pixel[:,:2]/pixel[:,2:]-.5
        z,al=bilinear_accumulated(ed,a,xy);pred[ix]=z*scale/uc[:,2];alpha[ix]=al
        diagnostics.append(dict(sector=int(sector),**getattr(adapter,'diagnostics',{})))
    hit=(alpha>=.5)&(pred>=.1)&(pred<=200)
    assert np.isfinite(pred).all() and np.isfinite(alpha).all()
    return pred,alpha,hit,diagnostics
