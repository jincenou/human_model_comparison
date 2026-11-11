"""Score a registered pair on one reference-defined ROI; never align images here."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from . import metrics as m

def evaluate(manifest):
    ref=manifest['reference'];pred=manifest['prediction'];camera=manifest['camera']
    if not manifest.get('reference_eligible'):
        raise ValueError('Reference eligibility must be established before scoring')
    if not pred.get('registration_validated'):
        raise ValueError('Prediction registration has not been validated')
    for key in ('identity','timestamp_us','camera_sha256'):
        if ref[key]!=pred[key]:raise ValueError('Reference/prediction mismatch: '+key)
    if m.sha(camera['path'])!=ref['camera_sha256']:
        raise ValueError('Camera hash mismatch')
    c=json.loads(Path(camera['path']).read_text())
    if c['identity']!=ref['identity'] or c['timestamp_us']!=ref['timestamp_us']:
        raise ValueError('Camera identity/time mismatch')
    registration=Path(pred['registration_path'])
    if m.sha(registration)!=pred['registration_sha256']:raise ValueError('Registration hash mismatch')
    reg=json.loads(registration.read_text())
    for key in ('identity','timestamp_us','camera_sha256'):
        if reg[key]!=pred[key]:raise ValueError('Registration evidence mismatch: '+key)
    if reg.get('status')!='passed':raise ValueError('Registration evidence must pass')
    def load(spec,key,mode):
        p=Path(spec[key])
        if m.sha(p)!=spec[key+'_sha256']:raise ValueError('Image hash mismatch: '+key)
        return np.asarray(Image.open(p).convert(mode))
    rgb=load(ref,'rgb','RGB');mask=load(ref,'mask','L')
    prgb=load(pred,'rgb','RGB');alpha=load(pred,'alpha','L')
    if rgb.shape!=prgb.shape or mask.shape!=alpha.shape or rgb.shape[:2]!=mask.shape:
        raise ValueError('All inputs must share the raw source canvas')
    if list(rgb.shape[1::-1])!=c['render_wh']:raise ValueError('Camera canvas mismatch')
    prepared=m.prepare_reference(rgb,mask)
    if prepared['status']!='ready':return dict(status='na',reason=prepared['reason'])
    if list(prepared['roi'])!=c['roi_xyxy']:raise ValueError('ROI must be derived from this reference mask')
    if pred['encoding']=='white_composited_uint8':p=m.prepare_prediction(prgb,alpha,prepared['roi'],pred['encoding'])
    elif pred['encoding']=='straight_uint8':p=m.prepare_prediction(prgb/255.,alpha/255.,prepared['roi'],'straight_float')
    else:raise ValueError('Unsupported encoding; make alpha representation explicit')
    scores={'PSNR':m.psnr(prepared['rgb'],p['rgb']),'SSIM':m.ssim(prepared['rgb'],p['rgb']),
            'IoU':m.iou(prepared['mask'],p['mask']),'IoU_raw':m.iou(mask,alpha)}
    if manifest.get('lpips'):
        cfg=manifest['lpips'];metric=m.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'],cfg.get('device','cpu'))
        scores['LPIPS']=metric(prepared['rgb'],p['rgb'])
    else:scores['LPIPS']=m.na('explicit_local_weights_not_supplied')
    scores['ED-P']=m.na('feature_and_visibility_protocol_not_yet_released')
    return dict(status='scored_core_metrics',identity=ref['identity'],timestamp_us=ref['timestamp_us'],
                camera_sha256=ref['camera_sha256'],roi_xyxy=list(prepared['roi']),metrics=scores,
                scope='source reconstruction; this does not establish held-out generalization or backside accuracy')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=evaluate(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


if __name__ == "__main__":
    main()
