import sys
import numpy as np
import torch
from PIL import Image
from common import *
from scene_depth import prepare,SceneDepth
from types import SimpleNamespace
from render_common_scene_orbit import scene_context
from test_splatad_dynamic_export import dynamic_module
from nerfstudio.models.splatad import SplatADModel
job=next(j for j in read(E/'intake.json')['rows'] if j['method']=='splatad' and j['id']=='HN0211')
out=E/'validation/splat_numeric';pack,rec=prepare(job,out);cert=read(job['source_certificate']);cam=read(job['camera_contract']);item,prior,frame,mapping=scene_context(job['id'],job['method'],cert,job['t_ref_us'])
cp=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)['pipeline'];bundle=torch.load(item['bundle'],map_location='cpu',weights_only=False);ds={k.removeprefix('_model.dynamic_actors.'):v for k,v in cp.items() if k.startswith('_model.dynamic_actors.')};dyn=dynamic_module(ds,bundle['trajectory']).cuda();gp=bundle['gauss_params']
query=torch.tensor([[pack['temporal']['query_seconds']]],device='cuda',dtype=torch.float32)
with torch.inference_mode():means,vel=SplatADModel._get_actor_adjusted_means(SimpleNamespace(dynamic_actors=dyn,training=False),gp['means'].cuda(),query,gp['id'].cuda())
old=pack['arrays']['means'];change=float((old-means.cpu()).abs().max());original=np.array(Image.open(cert['source_alpha']).convert('L'))
result=dict(posed_means_CPU_GPU_max=change,checks=[])
for mode in ['CPU_pose','GPU_pose']:
    if mode=='GPU_pose':pack['arrays']['means']=means.cpu();pack['velocities']=vel.cpu()
    adapter=SceneDepth(pack);T=np.eye(4);T[:3]=np.asarray(prior['source_camera_to_world']).reshape(-1,3,4)[0];V=np.linalg.inv(T@np.diag([1,-1,-1,1]))
    for camera in ['world_contract','native_float_camera']:
        if camera=='world_contract':V=np.linalg.inv(cam['T_world_from_camera_opencv'])@pack['T_world_from_native'];K=cam['K_raw']
        else:V=np.linalg.inv(T@np.diag([1,-1,-1,1]));K=prior['source_camera_K']
        ed,a=adapter.render(V,np.asarray(K),1600,.01,1000.);alpha=np.rint(a[:900]*255);diff=abs(alpha-original)
        result['checks'].append(dict(mode=mode,camera=camera,max=float(diff.max()),mean=float(diff.mean()),pixels_over2=int((diff>2).sum()),threshold_disagreements=int(((alpha>127)!=(original>127)).sum()),support_pixels=int((original>127).sum())))
        Image.fromarray(diff.clip(0,255).astype('uint8')).save(out/(mode+'_'+camera+'.png'))
save(out/'comparison.json',result);print(result,flush=True)
