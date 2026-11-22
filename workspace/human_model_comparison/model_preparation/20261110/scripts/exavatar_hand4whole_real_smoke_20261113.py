"""Real-image component test, not a fitted or trained ExAvatar result."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
from pathlib import Path
import sys
import os
import numpy as np
import torch
import cv2

P=Path(__file__).resolve().parents[1]
demo=P/'sources/exavatar/fitting/tools/Hand4Whole_RELEASE/demo'
os.chdir(demo)
sys.path[:0]=[str(demo/'../main'),str(demo/'../common'),str(demo/'../data')]
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox,generate_patch_image
from utils.human_models import smpl_x

outdir=P/'reports/exavatar_hand4whole_HN0057_smoke_20261113'
outdir.mkdir(exist_ok=True)
candidate=Path(_hmc_expand('${HMC_WORKSPACE}/asset_work/human_nuscenes100_20261026/inputs/HN0057/candidate.json'))
record=json.loads(candidate.read_text())['keyframes'][0]
src=Path(record['image_path']);image=cv2.cvtColor(cv2.imread(str(src)),cv2.COLOR_BGR2RGB)
x1,y1,x2,y2=record['bbox_xyxy']
bbox=process_bbox(np.array([x1,y1,x2-x1,y2-y1]),image.shape[1],image.shape[0])
patch,trans,inverse=generate_patch_image(image,bbox,1.,0.,False,cfg.input_img_shape)
cv2.imwrite(str(outdir/'input_patch.png'),cv2.cvtColor(patch.astype(np.uint8),cv2.COLOR_RGB2BGR))
model=get_model('test').cuda().eval()
checkpoint=torch.load(demo/'snapshot_6.pth.tar',map_location='cpu',weights_only=False)
state={k.removeprefix('module.'):v for k,v in checkpoint['network'].items()}
incompatible=model.load_state_dict(state,strict=False)
# Upstream checkpoints omit the separately distributed body-model buffers.
# All learned network parameters must still be present.
assert not incompatible.unexpected_keys, str(incompatible)
assert all(k.startswith('smplx_layer.') for k in incompatible.missing_keys),str(incompatible)
assert not set(incompatible.missing_keys).intersection(dict(model.named_parameters())),str(incompatible)
inputs={'img':torch.from_numpy(patch.copy()).float().permute(2,0,1)[None].cuda()/255.}
with torch.no_grad(): output=model(inputs,{}, {},'test')
arrays={k:v.cpu().numpy() for k,v in output.items() if isinstance(v,torch.Tensor)}
for k,v in arrays.items():assert np.isfinite(v).all(),k
assert arrays['smplx_mesh_cam'].shape==(1,10475,3)
np.savez_compressed(outdir/'prediction.npz',**arrays)
result=dict(status='real_image_hand4whole_component_passed',identity='HN0057',input=record,
            source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            bbox_source='Existing nuScenes identity annotation; used to isolate known person for component test, not full native detector pipeline.',
            checkpoint_sha256=hashlib.sha256((demo/'snapshot_6.pth.tar').read_bytes()).hexdigest(),
            separately_loaded_body_buffers=incompatible.missing_keys,
            tensor_shapes={k:list(v.shape) for k,v in arrays.items()},
            full_fitting_validated=False,avatar_trained=False)
(outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],output=str(outdir))))
