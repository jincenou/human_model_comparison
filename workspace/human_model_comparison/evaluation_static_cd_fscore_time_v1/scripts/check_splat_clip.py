from common import *
from scene_depth import SceneDepth
from PIL import Image
import torch,numpy as np
job=next(j for j in read(E/'intake.json')['rows'] if j['method']=='splatad' and j['id']=='HN0211');pack=torch.load(E/'validation/splat_numeric/prepared.pt',map_location='cpu',weights_only=False);a=SceneDepth(pack);cam=read(job['camera_contract']);V=np.linalg.inv(cam['T_world_from_camera_opencv'])@pack['T_world_from_native'];K=np.array(cam['K_raw']);cert=read(job['source_certificate']);old=np.array(Image.open(cert['source_alpha']).convert('L'));rows=[]
for near in [.01,.5]:
    ed,alpha=a.render(V,K,1600,near,1e10);dif=abs(np.rint(alpha[:900]*255)-old);rows.append(dict(near=near,max=float(dif.max()),mean=float(dif.mean())))
save(E/'validation/splat_numeric/native_clip_check.json',dict(rows=rows,script_sha256=sha(__file__)));print(rows,flush=True)
