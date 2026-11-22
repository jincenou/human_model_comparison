from human_asset_benchmark.paths import expand as _hmc_expand
import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
sys.path.insert(0,str(H/'scripts'))
import score_native_reaudit_geometry_v1 as geom
rows=geom.read(geom.HG/'reference_manifest.json')['rows']
results=[]
for sid in ['HN0252','HN0312']:
 ref=next(x for x in rows if x['id']==sid)
 b,v,pack,e=geom.load_export('splatad',sid,ref);a=geom.SceneDepth(pack)
 c=geom.read(ref['camera_contract']);mat=np.linalg.inv(np.asarray(c['T_world_from_camera_opencv']))@np.asarray(pack['T_world_from_native'])
 expected=np.asarray(Image.open(v['actor_alpha']).convert('L')).astype(float)
 for name,M in [('world_reconstructed',mat),('native_saved',b['raster_settings']['viewmats'][0])]:
  with torch.inference_mode():
   _,al,_=a.rasterization(*a.arrays,torch.zeros_like(a.arrays[0]),viewmats=a.tensor(M)[None],Ks=a.tensor(c['K_raw'])[None],width=c['render_wh'][0],height=c['render_wh'][1],near_plane=pack['render_cfg']['near_plane'],far_plane=pack['render_cfg']['far_plane'],render_mode='RGB+ED',**a.options)
  diff=np.abs((al[0,...,0].cpu().numpy()*255).round().clip(0,255)-expected)
  results.append(dict(id=sid,view_matrix=name,max=float(diff.max()),mean=float(diff.mean()),view_difference=float(np.max(np.abs(mat-np.asarray(b['raster_settings']['viewmats'][0]))))))
geom.save(Path(__file__).with_suffix('.json'),results)
print(json.dumps(results))
