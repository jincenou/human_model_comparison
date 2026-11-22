"""Frozen physical-time scene Gaussians, decoded through the retained native code."""
from human_asset_benchmark.paths import expand as _hmc_expand
import sys
from types import SimpleNamespace
import numpy as np
from common import *
sys.path[:0]=[str(H/'scripts'),str(R/'scripts'),str(R/'adapters'),str(R/'evaluation_nuscenes_v1/scripts')]

def prepare(job,out):
    import torch,yaml
    from render_common_scene_orbit import scene_context
    cert=read(job['source_certificate']);cam=read(job['camera_contract'])
    item,prior,frame,mapping=scene_context(job['id'],job['method'],cert,job['t_ref_us'])
    cp=Path(item['checkpoint']);checked(cp,prior['checkpoint_sha256']);checked(item['config'],item['config_sha256'])
    evidence={str(cp):sha(cp),item['config']:sha(item['config']),mapping['source_validation']:mapping['source_validation_sha256']}
    if job['method']=='omnire':
        cfg=yaml.safe_load(Path(item['config']).read_text())
        if job['id']=='HN0004':
            from render_common_scene_orbit import omnire
            from omnire_time_query_v1 import at_physical_exposure
            protocol=dict(frames=[dict(camera_to_display=np.eye(4).tolist())])
            with torch.inference_mode():
                views,gauge,render,provenance=omnire(job['id'],frame,cam,protocol,item,prior)
                closure=dict(zip(render.__code__.co_freevars,[c.cell_contents for c in render.__closure__]))
                with at_physical_exposure(closure['model'],closure['instance'],closure['temporal']):
                    gs=closure['model'].get_gaussians(closure['native_camera'](views[0]))
                posed={k:v[closure['mask']].cpu() for k,v in gs.items()}
            Tworld=np.asarray(cam['T_world_from_camera_opencv'])@views[0]
            temporal=frame['omnire'];evidence.update(provenance)
            evidence={k:v for k,v in evidence.items() if isinstance(v,str)}
        else:
            bp=Path(prior['bundle']);bundle=torch.load(bp,map_location='cpu',weights_only=False)
            assert bundle['sample_id']==job['id'] and 'nuscenes:'+bundle['uuid']==job['identity_key']
            assert bundle['temporal_query']['query_timestamp_us']==job['t_ref_us'] and prior['state_time_query_valid']
            assert bundle['temporal_query']==prior['temporal_query']
            state=torch.load(cp,map_location='cpu',weights_only=False)['models']['DeformableNodes']
            mask=state['points_ids'].reshape(-1)==bundle['native_instance_index']
            for k,v in bundle['canonical_parameters'].items():torch.testing.assert_close(state[k][mask],v,rtol=0,atol=0)
            posed=bundle['posed_gaussians'];Tworld=np.linalg.inv(np.asarray(prior['common_camera_audit']['T_model_from_world']))
            temporal=bundle['temporal_query'];evidence[str(bp)]=sha(bp)
        arrays={k:posed[v].float().contiguous() for k,v in [('means','_means'),('quats','_quats'),('scales','_scales'),('opacity','_opacities')]}
        arrays['opacity']=arrays['opacity'].reshape(-1)
        render_cfg=cfg['trainer']['render'];extra={}
    else:
        from test_splatad_dynamic_export import dynamic_module
        from nerfstudio.models.splatad import SplatADModel
        from register_splatad_actor import config_node
        bp=Path(item['bundle']);checked(bp,item['bundle_sha256']);bundle=torch.load(bp,map_location='cpu',weights_only=False)
        checkpoint=torch.load(cp,map_location='cpu',weights_only=False);state=checkpoint['pipeline'];index=bundle['actor_index'];gp=bundle['gauss_params']
        assert 'nuscenes:'+bundle['uuid']==job['identity_key']
        mask=state['_model.gauss_params.id'].reshape(-1)==index
        for name,v in gp.items():torch.testing.assert_close(v,state['_model.gauss_params.'+name][mask],rtol=0,atol=0)
        ds={k.removeprefix('_model.dynamic_actors.'):v for k,v in state.items() if k.startswith('_model.dynamic_actors.')}
        dynamic=dynamic_module(ds,bundle['trajectory'])
        # Use the independently audited actual physical camera exposure query.
        if 'source_camera_time' in prior:seconds=float(np.asarray(prior['source_camera_time']).reshape(-1)[0])
        else:raise ValueError('Missing physical source time')
        offset=prior['common_camera_audit'].get('native_time_offset_seconds')
        if offset is not None:assert abs(seconds+offset-job['t_ref_us']/1e6)<1e-5
        query=torch.tensor([[seconds]],dtype=torch.float32)
        with torch.inference_mode():means,velocities=SplatADModel._get_actor_adjusted_means(SimpleNamespace(dynamic_actors=dynamic,training=False),gp['means'],query,gp['id'])
        arrays=dict(means=means.cpu(),quats=gp['quats'],scales=gp['scales'].exp(),opacity=gp['opacities'].sigmoid().reshape(-1))
        cfg=yaml.compose(Path(item['config']).read_text());mc=config_node(cfg,'pipeline','model')
        render_cfg=dict(rasterize_mode=config_node(mc,'rasterize_mode').value,radius_clip=float(config_node(mc,'radius_clip_pix').value),absgrad=config_node(mc,'use_absgrad').value=='true')
        Tworld=np.linalg.inv(np.asarray(prior['common_camera_audit']['T_model_from_world']));temporal=dict(query_seconds=seconds,t_ref_us=job['t_ref_us'],offset_seconds=offset)
        extra=dict(velocities=velocities.cpu());evidence[str(bp)]=sha(bp)
    # Raw world mapping is retained; SO(3) drift is checked, not silently fitted.
    rigid_part(Tworld)
    assert all(torch.isfinite(v).all() for v in arrays.values())
    out.mkdir(parents=True,exist_ok=True)
    pack=dict(arrays=arrays,method=job['method'],render_cfg=render_cfg,T_world_from_native=Tworld,temporal=temporal,evidence=evidence,**extra)
    torch.save(pack,out/'prepared.pt')
    rec=dict(id=job['id'],method=job['method'],identity_key=job['identity_key'],t_ref_us=job['t_ref_us'],status='prepared',
      prepared=str(out/'prepared.pt'),prepared_sha256=sha(out/'prepared.pt'),T_world_from_native=Tworld.tolist(),temporal=temporal,render_cfg=render_cfg,
      provenance=evidence,scene_mapping=mapping,script_sha256=sha(__file__),original_asset_modified=False,arrays_at_exact_physical_query=True)
    save(out/'preparation.json',rec)
    return pack,rec

class SceneDepth:
    def __init__(self,pack):
        import torch
        self.torch=torch;self.pack=pack;self.method=pack['method']
        self.arrays=[self.tensor(pack['arrays'][k]) for k in ['means','quats','scales','opacity']]
        cfg=pack['render_cfg']
        if self.method=='omnire':
            sys.path.insert(0,_hmc_expand('${DRIVESTUDIO_ROOT}'))
            from models.gaussians.basics import rasterization
            self.options=dict(packed=cfg['packed'],absgrad=cfg['absgrad'],sparse_grad=cfg['sparse_grad'],rasterize_mode='antialiased' if cfg['antialiased'] else 'classic',radius_clip=cfg.get('radius_clip',0.))
        else:
            from nerfstudio.models.splatad import rasterization
            self.options=dict(velocities=self.tensor(pack['velocities']),linear_velocity=torch.zeros((1,3),device='cuda'),angular_velocity=torch.zeros((1,3),device='cuda'),rolling_shutter_time=torch.zeros(1,device='cuda'),tile_size=16,packed=False,radius_clip=cfg['radius_clip'],sh_degree=None,sparse_grad=False,absgrad=cfg['absgrad'],rasterize_mode=cfg['rasterize_mode'],channel_chunk=128,eps2d=.3)
        self.rasterization=rasterization
    def tensor(self,v):return self.torch.as_tensor(v,device='cuda',dtype=self.torch.float32).contiguous()
    def render(self,V,K,n,near,far,ortho=False):
        options=dict(self.options)
        if ortho:
            assert self.method=='omnire','SplatAD ortho uses the validated native-radius shared-alpha adapter'
            options['ortho']=True
        with self.torch.inference_mode():
            im,alpha,_=self.rasterization(*self.arrays,self.torch.zeros_like(self.arrays[0]),viewmats=self.tensor(V)[None],Ks=self.tensor(K)[None],width=n,height=n,near_plane=near,far_plane=far,render_mode='RGB+ED',**options)
        assert self.torch.isfinite(im).all() and self.torch.isfinite(alpha).all()
        return im[0,...,-1].cpu().numpy(),alpha[0,...,0].cpu().numpy()
