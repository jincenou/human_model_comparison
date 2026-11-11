"""Generate a small synthetic contract fixture, not a real benchmark result."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from human_asset_benchmark.metrics import sha,reference_roi

def make_example(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    rgb=np.full((96,128,3),255,dtype=np.uint8);mask=np.zeros((96,128),dtype=np.uint8)
    mask[16:80,44:84]=255;rgb[mask>0]=[80,120,170]
    pred=rgb.copy();alpha=mask.copy()
    for name,x in [('reference_rgb',rgb),('reference_mask',mask),('prediction_rgb',pred),('prediction_alpha',alpha)]:Image.fromarray(x).save(out/(name+'.png'))
    camera={'identity':'synthetic_fixture','timestamp_us':1,'render_wh':[128,96],
            'roi_xyxy':list(reference_roi(mask)),'K_raw':[[100,0,64],[0,100,48],[0,0,1]],
            'T_world_from_camera_opencv':np.eye(4).tolist()}
    cp=out/'camera.json';cp.write_text(json.dumps(camera,indent=2)+'\n')
    identity=dict(identity=camera['identity'],timestamp_us=1,camera_sha256=sha(cp))
    reg=dict(identity,status='passed',scope='synthetic identity construction only; not evidence for a real model')
    rp=out/'registration.json';rp.write_text(json.dumps(reg,indent=2)+'\n')
    ref=dict(identity,rgb=str(out/'reference_rgb.png'),mask=str(out/'reference_mask.png'))
    pred=dict(identity,rgb=str(out/'prediction_rgb.png'),alpha=str(out/'prediction_alpha.png'),encoding='white_composited_uint8',
              registration_validated=True,registration_path=str(rp),registration_sha256=sha(rp))
    for spec,keys in [(ref,['rgb','mask']),(pred,['rgb','alpha'])]:
        for key in keys:spec[key+'_sha256']=sha(spec[key])
    manifest=dict(reference_eligible=True,reference=ref,prediction=pred,camera={'path':str(cp)},synthetic=True)
    (out/'pair.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();make_example(args.output)

