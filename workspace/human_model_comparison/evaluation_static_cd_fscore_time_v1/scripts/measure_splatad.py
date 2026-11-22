import sys
import numpy as np
import torch
from common import *
from pilot import measure
from register_splatad_actor import shared_alpha
out=Path(sys.argv[1]);rec=read(out/'preparation.json');checked(rec['prepared'],rec['prepared_sha256']);pack=torch.load(rec['prepared'],map_location='cpu',weights_only=False)
class Alpha:
    def __init__(self):self.arrays=[pack['arrays'][k] for k in ['means','quats','scales','opacity']]
    def render(self,V,K,n,near,far,ortho=False):
        a=shared_alpha(pack,V,K,(n,n),near,far,'ortho' if ortho else 'pinhole')/255.
        return np.zeros_like(a),a
measure(Alpha(),rigid_part(rec['T_world_from_native'])[0][:3,:3],out)
