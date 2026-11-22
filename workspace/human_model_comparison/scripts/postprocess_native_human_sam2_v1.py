"""Frozen prediction-only SAM2 image postprocess, separate from 3D validation."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
ROOT = H / 'reports/native_human_export_reaudit_v1'
OUT = ROOT / 'postprocess_v1'
SOURCE = H.parent / 'human/vendor/segmentation'
WEIGHT = Path(_hmc_expand('${HMC_MODEL_CACHE}/grounded_sam2/sam2.1_hiera_base_plus.pt'))
WEIGHT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
METHODS = ('splatad', 'omnire', 'streetgs')
PILOTS = [('splatad', 'HN0004'), ('omnire', 'HN0317'), ('streetgs', 'HN0317')]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.writing')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n');q.replace(p)


def disk_guard():
    assert shutil.disk_usage(H).free > 10*1024**3, 'Less than 10 GiB free; stop without deleting outputs'


def prepare():
    disk_guard(); contract = read(ROOT/'resolved/batch_contract.json')
    assert len(contract['ids']) == 100 and sha(WEIGHT) == WEIGHT_SHA
    code = {str(p.relative_to(SOURCE)): sha(p) for p in sorted((SOURCE/'sam2').rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.yaml')}
    code_sha = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    records=[]
    for method in METHODS:
        for sid in contract['ids']:
            vp=ROOT/'resolved/exports'/method/sid/'validation.json';v=read(vp)
            sp=ROOT/'resolved/metrics'/method/'static_rows'/(sid+'.json');s=read(sp)
            dp=ROOT/'resolved/metrics'/method/'edp/all/poses'/(sid+'_'+method+'.json');d=read(dp)
            pp=Path(s['files']['prediction_unwarped'])
            assert v['status']=='passed' and v['prediction_encoding']=='white_composited_uint8'
            assert s['reference']==v['source_reference'] and s['status']=='scored'
            assert sha(pp)==s['files']['prediction_unwarped_sha256']==d['image_sha256']
            assert Image.open(pp).size==(512,512)
            assert sha(v['actor_preview'])==v['actor_preview_sha256']
            assert sha(v['actor_alpha'])==v['actor_alpha_sha256']
            assert Image.open(v['actor_preview']).size==Image.open(v['actor_alpha']).size==(1600,900)
            records.append(dict(id=sid,method=method,source_validation=str(vp),source_validation_sha256=sha(vp),
                original_rgb=v['actor_preview'],original_rgb_sha256=v['actor_preview_sha256'],
                original_alpha=v['actor_alpha'],original_alpha_sha256=v['actor_alpha_sha256'],
                static_record=str(sp),static_record_sha256=sha(sp),prediction_unwarped=str(pp),
                prediction_unwarped_sha256=sha(pp),detector_record=str(dp),detector_record_sha256=sha(dp),
                detector_status=d['status'],roi_xyxy=v['source_reference']['roi_xyxy'],
                identity_key=v['identity_key'],timestamp_us=v['timestamp_us']))
    manifest=dict(schema='prediction_sam2_postprocess_inputs_v1',selection_sha256=contract['selection_sha256'],
        ids=contract['ids'],methods=list(METHODS),expected=300,records=records,
        source_contract=str(ROOT/'resolved/batch_contract.json'),source_contract_sha256=sha(ROOT/'resolved/batch_contract.json'))
    mp=OUT/'inputs.json'
    if mp.exists():assert read(mp)==manifest,'Frozen inputs changed'
    else:save(mp,manifest)
    protocol=dict(schema='prediction_sam2_postprocess_v1',version=1,expected=300,selection_sha256=contract['selection_sha256'],
        input_manifest=str(mp),input_manifest_sha256=sha(mp),methods=list(METHODS),ids=contract['ids'],
        model=dict(name='SAM2.1 Hiera base-plus',weight=str(WEIGHT),weight_sha256=WEIGHT_SHA,
                   source=str(SOURCE),source_tree_sha256=code_sha,source_files=code,config=CONFIG,
                   build_apply_postprocessing=False,precision='float32_no_autocast',predictor_mask_threshold=0.0,
                   max_hole_area=0.0,max_sprinkle_area=0.0,multimask_output=False),
        detector=dict(cache='resolved metric prediction_unwarped RTMDet-m person cache',image_hash_must_match=True,
                      inherited_detector_score=0.3,inherited_detector_nms=0.3,coordinate_wh=[512,512],
                      selection='minimum squared distance of predicted person-box centre to fixed ROI centre (256,256); ties highest detector score, then existing box index',
                      mapping='edge coordinates: x=x0+u*(x1-x0)/512; y=y0+v*(y1-y0)/512; clip to full-image edges',
                      no_new_detection_threshold=True,no_reference_mask_pixel_access=True),
        operation=dict(input='resolved native white-composited RGB and native own alpha at 1600x900',
                       prompt='prediction detector box only; frozen ROI is coarse coordinate localization, not GT silhouette',
                       binary_mask='single SAM2 mask logits > 0, no dilation/erosion/hole-fill/body repair',
                       rgb='np.where(S[...,None],original_rgb,255)',alpha='original_alpha*S',
                       no_person='empty S; white RGB; zero alpha; status postprocess_no_person; retained in image metric denominator',
                       inference_exception='S=ones; original RGB and alpha retained; status postprocess_exception_fallback; no object-specific retry or method selection',
                       empty_sam_mask='white RGB and zero alpha; retained; no manual prompt correction'),
        scope=dict(two_dimensional_only=True,gaussian_parameters_modified=False,geometry_not_changed=True,
                   annotation_assisted_roi=True,fully_ground_truth_free_claimed=False,reference_pixels_used=False,
                   original_source_rgb_used_for_segmentation=False,reference_mask_erases_prediction=False,
                   metric_values_used_for_selection=False,raw_results_modified=False),
        disk_reserve_gib=10,script=str(Path(__file__).resolve()),script_sha256=sha(__file__))
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==protocol,'Frozen protocol changed'
    else:save(p,protocol)
    return manifest,protocol


def verified():
    p=read(OUT/'protocol.json');assert sha(__file__)==p['script_sha256'];assert sha(OUT/'inputs.json')==p['input_manifest_sha256']
    assert sha(WEIGHT)==p['model']['weight_sha256']
    for rel,digest in p['model']['source_files'].items():assert sha(SOURCE/rel)==digest
    return read(OUT/'inputs.json'),p


def choose_box(boxes, roi):
    b=np.asarray(boxes,dtype=np.float64)
    if b.size==0:return None,None,None
    assert b.ndim==2 and b.shape[1]==5 and np.isfinite(b).all()
    assert np.all(b[:,2]>b[:,0]) and np.all(b[:,3]>b[:,1])
    dist=((b[:,:2]+b[:,2:4])*.5-256.)**2
    idx=min(range(len(b)),key=lambda i:(float(dist[i].sum()),-float(b[i,4]),i))
    box=b[idx,:4].copy();x0,y0,x1,y1=roi
    box[[0,2]]=x0+box[[0,2]]*(x1-x0)/512.
    box[[1,3]]=y0+box[[1,3]]*(y1-y0)/512.
    box[[0,2]]=np.clip(box[[0,2]],0,1600);box[[1,3]]=np.clip(box[[1,3]],0,900)
    assert box[2]>box[0] and box[3]>box[1]
    return idx,b[idx].tolist(),box.astype(np.float32)


def process(row, predictor, protocol, torch):
    disk_guard();folder=OUT/'outputs'/row['method']/row['id'];folder.mkdir(parents=True,exist_ok=True)
    recpath=folder/'postprocess.json'
    if recpath.exists():
        previous=read(recpath)
        assert previous['protocol_sha256']==sha(OUT/'protocol.json')
        assert previous['source_validation_sha256']==row['source_validation_sha256']
        for key in ['mask','clean_rgb','clean_alpha']:assert sha(previous['outputs'][key])==previous['outputs'][key+'_sha256']
        return previous
    started=time.time()
    for field in ['source_validation','original_rgb','original_alpha','static_record','prediction_unwarped','detector_record']:
        assert sha(row[field])==row[field+'_sha256'],('Input changed',row['id'],field)
    rgb=np.asarray(Image.open(row['original_rgb']).convert('RGB'))
    alpha=np.asarray(Image.open(row['original_alpha']).convert('L'))
    assert rgb.shape==(900,1600,3) and alpha.shape==(900,1600)
    detector=read(row['detector_record']);assert detector['image_sha256']==row['prediction_unwarped_sha256']
    idx,small,box=choose_box(detector['boxes'],row['roi_xyxy'])
    status='postprocess_no_person';error=None;predicted_iou=None
    mask=np.zeros((900,1600),dtype=bool)
    if box is not None:
        try:
            with torch.inference_mode():
                predictor.set_image(rgb)
                masks,scores,_=predictor.predict(box=box,multimask_output=False)
            assert masks.shape==(1,900,1600) and np.isfinite(scores).all()
            mask=masks[0].astype(bool);predicted_iou=float(scores[0])
            status='postprocess_segmented' if mask.any() else 'postprocess_empty_mask'
        except Exception as exc:
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            mask=np.ones((900,1600),dtype=bool);status='postprocess_exception_fallback'
            torch.cuda.empty_cache()
        finally:predictor.reset_predictor()
    cleaned=np.where(mask[...,None],rgb,255).astype(np.uint8)
    clean_alpha=np.where(mask,alpha,0).astype(np.uint8)
    assert np.array_equal(cleaned[mask],rgb[mask]) and np.array_equal(clean_alpha[mask],alpha[mask])
    assert np.all(cleaned[~mask]==255) and np.all(clean_alpha[~mask]==0)
    for name,array in [('mask',mask.astype(np.uint8)*255),('clean_rgb',cleaned),('clean_alpha',clean_alpha)]:
        Image.fromarray(array).save(folder/(name+'.png'))
    outputs={}
    for name in ['mask','clean_rgb','clean_alpha']:
        outputs[name]=str(folder/(name+'.png'));outputs[name+'_sha256']=sha(outputs[name])
    result=dict(schema='prediction_image_postprocess_record_v1',id=row['id'],method=row['method'],status=status,
        **{k:v for k,v in row.items() if k not in ['id','method']},
        original_prediction_encoding='white_composited_uint8',output_prediction_encoding='white_composited_uint8',
        render_wh=[1600,900],protocol=str(OUT/'protocol.json'),protocol_sha256=sha(OUT/'protocol.json'),
        predictor_weight=str(WEIGHT),predictor_weight_sha256=WEIGHT_SHA,predictor_code_tree_sha256=protocol['model']['source_tree_sha256'],
        predictor_config=CONFIG,script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        box_policy=protocol['detector']['selection'],box_mapping=protocol['detector']['mapping'],
        detector_candidate_count=len(detector['boxes']),detector_selected_index=idx,
        selected_detector_box_512_xyxy_score=small,sam_box_raw_xyxy=None if box is None else box.tolist(),
        sam_predicted_iou=predicted_iou,outputs=outputs,mask_pixels=int(mask.sum()),
        original_alpha_nonzero_pixels=int((alpha>0).sum()),clean_alpha_nonzero_pixels=int((clean_alpha>0).sum()),
        clean_alpha_gt_127_pixels=int((clean_alpha>127).sum()),
        kept_pixels_rgb_exact=True,kept_pixels_alpha_exact=True,reference_mask_pixels_read=False,
        original_source_photo_pixels_read=False,annotation_assisted_roi=True,
        geometry_changed=False,gaussian_export_validation_claimed=False,all_entries_retained_for_scoring=True,
        error=error,started_at=started,finished_at=time.time(),elapsed_seconds=time.time()-started)
    save(recpath,result)
    print(json.dumps(dict(method=row['method'],id=row['id'],status=status,mask_pixels=int(mask.sum())),ensure_ascii=False),flush=True)
    return result


def run(method, gpu, pilot=False):
    manifest,protocol=verified();rows=[r for r in manifest['records'] if (r['method'],r['id']) in PILOTS] if pilot else [r for r in manifest['records'] if r['method']==method]
    key='pilot' if pilot else method;state=OUT/'jobs'/(key+'.json');leases=[]
    try:
        for p in [OUT/'jobs'/(key+'.lock'),R/'model_preparation/20261109_2100_CST/runs/rigid111_gpu_locks'/f'gpu_{gpu}.lock',H/'state'/f'gpu_auxiliary_slot_{gpu}.lock']:
            p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(f)
        gpuinfo=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        uuid=gpuinfo[0].strip();assert int(gpuinfo[1])>4096
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        sys.path.insert(0,str(SOURCE))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        torch.set_num_threads(4);torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available()
        save(state,dict(status='loading_model',pid=os.getpid(),gpu=gpu,gpu_uuid=uuid,expected=len(rows),protocol_sha256=sha(OUT/'protocol.json')))
        model=build_sam2(CONFIG,str(WEIGHT),device='cuda',apply_postprocessing=False).eval()
        predictor=SAM2ImagePredictor(model,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0)
        results=[]
        for row in rows:
            results.append(process(row,predictor,protocol,torch))
            save(state,dict(status='running',pid=os.getpid(),gpu=gpu,gpu_uuid=uuid,expected=len(rows),done=len(results),last=row['id'],
                            counts=dict(collections.Counter(r['status'] for r in results)),updated_at=time.time(),protocol_sha256=sha(OUT/'protocol.json')))
        save(state,dict(status='completed',pid=os.getpid(),gpu=gpu,gpu_uuid=uuid,expected=len(rows),done=len(results),
                        counts=dict(collections.Counter(r['status'] for r in results)),finished_at=time.time(),protocol_sha256=sha(OUT/'protocol.json'),torch=torch.__version__))
        if pilot:pilot_gallery()
    except Exception as exc:
        save(state,dict(status='batch_exception',pid=os.getpid(),gpu=gpu,reason=repr(exc),traceback=traceback.format_exc(),updated_at=time.time()))
        raise
    finally:
        for f in reversed(leases):f.close()


def pilot_gallery():
    font=ImageFont.truetype(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'),17)
    out=Image.new('RGB',(1024,300*len(PILOTS)),'white');draw=ImageDraw.Draw(out)
    for j,(method,sid) in enumerate(PILOTS):
        r=read(OUT/'outputs'/method/sid/'postprocess.json');roi=r['roi_xyxy']
        rgb=Image.open(r['original_rgb']).convert('RGB');clean=Image.open(r['outputs']['clean_rgb']).convert('RGB');mask=Image.open(r['outputs']['mask']).convert('RGB');a=Image.open(r['outputs']['clean_alpha']).convert('RGB')
        for i,(im,label) in enumerate(zip([rgb,mask,clean,a],['Original prediction','Predicted SAM mask','Clean prediction','Clean own alpha'])):
            draw.text((i*256+6,j*300+4),label,fill='#182a3b',font=font);out.paste(im.crop(roi).resize((256,256)),(i*256,j*300+29))
        draw.text((7,j*300+283),method+' '+sid+' | '+r['status'],fill='#182a3b',font=ImageFont.truetype(font.path,12))
    out.save(OUT/'pilot_comparison.jpg',quality=95,subsampling=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--method',choices=METHODS);p.add_argument('--gpu',type=int);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.prepare:
        m,_=prepare();print('Frozen postprocess protocol and',len(m['records']),'inputs')
    else:
        assert a.gpu is not None and (a.method or a.pilot);run(a.method,a.gpu,a.pilot)
