"""Candidate source metric protocol. Production scoring still needs qualification.

All arrays are explicit sRGB bytes or finite float RGB/alpha in [0,1]. Reference
mask determines one common ROI; it is never applied to a prediction.
"""
from pathlib import Path
import hashlib,math
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

PARTS=('head','torso','left_arm','right_arm','left_leg','right_leg')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def score(value):
    if math.isinf(value) and value>0:return dict(status='positive_infinity',value=None)
    assert math.isfinite(value)
    return dict(status='ok',value=float(value))
def na(reason):return dict(status='na',value=None,reason=reason)
def bytes_rgb(x):
    x=np.asarray(x);assert x.dtype==np.uint8 and x.ndim==3 and x.shape[2]==3
    return x
def bytes_mask(x):
    x=np.asarray(x);assert x.dtype==np.uint8 and x.ndim==2
    return x
def quantize(x):
    x=np.asarray(x);assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1
    return np.floor(x*255.+.5).astype(np.uint8)
def reference_roi(mask):
    mask=bytes_mask(mask);ys,xs=np.nonzero(mask>127)
    if len(xs)==0:return None
    bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    side=math.ceil(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.10)
    left=math.floor((bbox[0]+bbox[2]-side)/2);top=math.floor((bbox[1]+bbox[3]-side)/2)
    return (left,top,left+side,top+side)
def crop(array,roi,is_mask=False):
    array=bytes_mask(array)if is_mask else bytes_rgb(array)
    l,t,r,b=roi;assert r-l==b-t and r>l
    image=Image.fromarray(array);out=Image.new('L'if is_mask else'RGB',(r-l,b-t),0 if is_mask else(255,255,255))
    box=(max(0,l),max(0,t),min(image.width,r),min(image.height,b))
    if box[2]>box[0] and box[3]>box[1]:out.paste(image.crop(box),(box[0]-l,box[1]-t))
    return np.asarray(out.resize((512,512),Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC))
def prepare_reference(rgb,mask):
    rgb=bytes_rgb(rgb);mask=bytes_mask(mask);assert rgb.shape[:2]==mask.shape
    roi=reference_roi(mask)
    if roi is None:return dict(status='na',reason='empty_reference_mask')
    visible=mask>127;white=np.where(visible[...,None],rgb,255).astype(np.uint8)
    return dict(status='ready',roi=roi,rgb=crop(white,roi),mask=crop(np.where(visible,255,0).astype(np.uint8),roi,True))
def prepare_prediction(rgb,alpha,roi,encoding):
    if encoding=='white_composited_uint8':
        rgb=bytes_rgb(rgb);alpha=bytes_mask(alpha)
    else:
        rgb=np.asarray(rgb,dtype=np.float64);alpha=np.asarray(alpha,dtype=np.float64)
        assert rgb.ndim==3 and rgb.shape[-1]==3 and rgb.shape[:2]==alpha.shape
        assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
        assert rgb.min()>=0 and rgb.max()<=1 and alpha.min()>=0 and alpha.max()<=1
        if encoding=='straight_float':white=rgb*alpha[...,None]+1-alpha[...,None]
        elif encoding=='premultiplied_float':
            assert np.all(rgb<=alpha[...,None]+1e-6),'Premultiplied RGB exceeds own alpha'
            white=rgb+1-alpha[...,None]
        else:raise ValueError('Explicit prediction RGB/alpha encoding required')
        rgb=quantize(np.clip(white,0,1));alpha=quantize(alpha)
    assert rgb.shape[:2]==alpha.shape
    return dict(rgb=crop(rgb,roi),mask=crop(alpha,roi,True),raw_mask=alpha)
def psnr(a,b):
    a=bytes_rgb(a).astype(np.float64)/255;b=bytes_rgb(b).astype(np.float64)/255;assert a.shape==b.shape
    mse=float(np.mean((a-b)**2,dtype=np.float64))
    return score(math.inf if mse==0 else -10*math.log10(mse))
def ssim(a,b):
    a=bytes_rgb(a).astype(np.float64)/255;b=bytes_rgb(b).astype(np.float64)/255
    assert a.shape==b.shape and min(a.shape[:2])>=11
    grid=np.arange(-5,6,dtype=np.float64);w=np.exp(-grid**2/(2*1.5**2));w/=w.sum()
    def smooth(x):return convolve1d(convolve1d(x,w,axis=0,mode='reflect'),w,axis=1,mode='reflect')
    ma,mb=smooth(a),smooth(b)
    va=smooth(a*a)-ma*ma;vb=smooth(b*b)-mb*mb;cab=smooth(a*b)-ma*mb
    c1=.01**2;c2=.03**2
    values=((2*ma*mb+c1)*(2*cab+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
    return score(float(values[5:-5,5:-5].mean(dtype=np.float64)))
def iou(a,b):
    a=bytes_mask(a)>127;b=bytes_mask(b)>127;assert a.shape==b.shape
    if not a.any():return na('empty_reference_mask')
    return score(np.count_nonzero(a&b)/np.count_nonzero(a|b))
def aggregate_visible_parts(values):
    assert set(values).issubset(PARTS)
    valid={k:float(v)for k,v in values.items()if v is not None and math.isfinite(v)}
    return dict(score=score(float(np.mean(list(valid.values()))))if valid else na('no_valid_visible_parts'),
        valid_parts=list(valid),valid_part_count=len(valid),total_part_count=6,coverage=len(valid)/6)

class AlexLPIPS:
    """Load both local weight files explicitly; constructor cannot download."""
    def __init__(self,backbone,linear,expected_backbone_sha,expected_linear_sha,device='cpu'):
        import torch,lpips
        assert sha(backbone)==expected_backbone_sha and sha(linear)==expected_linear_sha
        self.torch=torch;self.device=device
        self.model=lpips.LPIPS(net='alex',version='0.1',pnet_rand=True,pretrained=False,verbose=False)
        source=torch.load(backbone,map_location='cpu',weights_only=True)
        mapped={key:source['features.'+key.split('.',1)[1]]for key in self.model.net.state_dict()}
        self.model.net.load_state_dict(mapped,strict=True)
        linear_state=torch.load(linear,map_location='cpu',weights_only=True)
        assert set(linear_state)=={f'lin{i}.model.1.weight'for i in range(5)}
        with torch.no_grad():
            for name,tensor in linear_state.items():self.model.get_parameter(name).copy_(tensor)
        self.model.pnet_rand=False
        self.model.to(device=device,dtype=torch.float32).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.provenance=dict(backbone=str(backbone),backbone_sha256=expected_backbone_sha,linear=str(linear),linear_sha256=expected_linear_sha,
            network='alex',version='0.1',dtype='float32',mode='eval',input='RGB NCHW sRGB bytes /127.5 -1',downloads=False)
    def __call__(self,a,b):
        torch=self.torch
        def tensor(x):return torch.from_numpy(bytes_rgb(x).astype(np.float32)/127.5-1).permute(2,0,1)[None].to(self.device)
        with torch.no_grad():value=float(self.model(tensor(a),tensor(b)).item())
        return score(value)

def paired_summary(methods,metric,selected_ids):
    selected=set(selected_ids);assert len(selected)==len(selected_ids)
    valid={};coverage={}
    for method,rows in methods.items():
        assert set(rows).issubset(selected)
        valid[method]={sid:row['metrics'][metric]for sid,row in rows.items()
            if row.get('metric_eligible') is True and metric in row.get('metrics',{})
            and row['metrics'][metric]['status']in('ok','positive_infinity')}
        coverage[method]=dict(n=len(valid[method]),denominator=len(selected))
    common=sorted(set.intersection(*(set(v)for v in valid.values())))if valid else[]
    summaries={}
    for method,rows in valid.items():
        values=[rows[sid]for sid in common];infinities=sum(v['status']=='positive_infinity'for v in values)
        summaries[method]=dict(n=len(common),denominator=len(selected),positive_infinity_count=infinities,
            mean=na('empty_common_identity_set')if not values else score(math.inf if infinities else float(np.mean([v['value']for v in values]))))
    return dict(metric=metric,common_ids=common,paired=summaries,individual_coverage=coverage)
