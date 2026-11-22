"""Independent analytic transparent planes through the actual voxel sampler."""
from types import SimpleNamespace
import itertools
import numpy as np
from common import *
from depth import VoxelDepth
import torch
job=next(j for j in read(E/'intake.json')['rows'] if j['method']=='trellis2' and j['id']=='HN0646')
a=VoxelDepth(job);checks=[]
def planes(zs,opacity):
    vs=[];faces=[]
    for z in zs:
        k=len(vs);vs.extend([[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]]);faces.extend([[k,k+1,k+2],[k,k+2,k+3]])
    coords=torch.tensor(list(itertools.product(range(7),repeat=3)),device='cuda',dtype=torch.int32)
    a.mesh=SimpleNamespace(vertices=a.tensor(vs),faces=torch.tensor(faces,device='cuda',dtype=torch.int32),coords=coords,attrs=torch.full((len(coords),1),opacity,device='cuda'),origin=a.tensor([-3,-3,0]),voxel_size=1.,voxel_shape=[1,1,7,7,7],layout={'alpha':slice(0,1)})
V=np.eye(4);K=np.array([[128.,0,128],[0,128,128],[0,0,1.]])
planes([2.,3.],.5);ed,al=a.render(V,K,256,.1,200.)
assert abs(float(al[128,128])-.75)<1e-6 and abs(float(ed[128,128])-7/3)<1e-5
checks.append(dict(name='two_transparent_planes',alpha=float(al[128,128]),depth=float(ed[128,128]),expected_alpha=.75,expected_depth=7/3))
zs=np.linspace(2,3,12);planes(zs,.5);ed,al=a.render(V,K,256,.1,200.)
w=.5**np.arange(1,13);expected=float(w@zs/w.sum())
assert a.diagnostics['layers_requested']==16 and abs(float(ed[128,128])-expected)<1e-5
checks.append(dict(name='deep_peeling_automatic_uniform_refinement',layers=a.diagnostics['layers_requested'],expected_depth=expected,depth=float(ed[128,128]),remaining=a.diagnostics['next_layer_max_remaining']))
planes([2.],1.);ed,al=a.render(V,K,256,.1,200.);assert abs(float(ed[128,150])-2)<1e-6
u=np.array([(150.5-128)/128,0,1.]);u/=np.linalg.norm(u);assert abs(2/u[2]-np.linalg.norm([2*(150.5-128)/128,0,2]))<1e-10
checks.append(dict(name='opaque_plane_and_oblique_z_to_range',status='passed'))
save(E/'validation/voxel_analytic.json',dict(status='passed',checks=checks,script_sha256=sha(__file__),depth_sha256=sha(E/'scripts/depth.py')))
print('analytic_voxel_passed',flush=True)
