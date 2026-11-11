"""Produce versioned point-prompt mask corrections without replacing originals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_exact_reference_masks_v2 import ROOT, OUT, SOURCE, WEIGHT, CONFIG, EXPECTED_WEIGHT_SHA, sha, save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args()
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    assert args.gpu_uuid not in [s.strip() for s in occupied]
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    plan_path=ROOT/'inputs/reference_mask_corrections_v1.json'
    plan=json.loads(plan_path.read_text())
    assert sha(WEIGHT)==EXPECTED_WEIGHT_SHA
    predictor=SAM2ImagePredictor(build_sam2(CONFIG,str(WEIGHT),device='cuda'))
    results=[]
    for row in plan['records']:
        ident=row['id'];base=OUT/ident;dest=base/'revisions/points_v1'
        assert not dest.exists(), 'Correction revision already exists; inspect it before choosing another revision'
        dest.mkdir(parents=True)
        old=json.loads((base/'metadata.json').read_text())
        assert sha(old['source_path'])==old['source_sha256']
        image=np.array(Image.open(old['source_path']).convert('RGB'),copy=True)
        x0,y0,x1,y1=map(int,old['crop_xyxy'])
        crop_points=np.array(row['positive']+row['negative'],dtype=np.float32)
        assert (crop_points[:,0]>=0).all() and (crop_points[:,0]<x1-x0).all()
        assert (crop_points[:,1]>=0).all() and (crop_points[:,1]<y1-y0).all()
        points=crop_points+[x0,y0]
        labels=np.array([1]*len(row['positive'])+[0]*len(row['negative']),dtype=np.int32)
        with torch.inference_mode():
            predictor.set_image(image)
            masks,scores,_=predictor.predict(point_coords=points,point_labels=labels,
                box=np.asarray(old['bbox_prompt_raw_xyxy'],dtype=np.float32),multimask_output=False)
        mask=masks[0].astype(np.uint8)*255
        Image.fromarray(mask).save(dest/'mask.png')
        Image.fromarray(mask[y0:y1,x0:x1]).save(dest/'mask_crop.png')
        overlay=image.copy();foreground=mask>127
        overlay[foreground]=(image[foreground]*.55+np.array([0,210,120])*.45).astype(np.uint8)
        Image.fromarray(overlay[y0:y1,x0:x1]).save(dest/'overlay_crop.jpg',quality=94)
        new=dict(old,mask_path=str(dest/'mask.png'),mask_sha256=sha(dest/'mask.png'),
            foreground_pixels=int(foreground.sum()),sam_predicted_iou=float(scores[0]),
            mask_origin='exact_exposure_sam2_box_and_reviewed_points_candidate',
            correction_plan=str(plan_path),correction_plan_sha256=sha(plan_path),correction=row,
            base_mask_sha256=old['mask_sha256'],correction_script_sha256=sha(__file__),
            visual_review='pending',reference_ready=False,metric_eligible=False)
        save(dest/'metadata.json',new)
        results.append({'id':ident,'path':str(dest),'foreground_pixels':new['foreground_pixels'],
                        'sam_predicted_iou':new['sam_predicted_iou'],'status':'correction_candidate_pending_review'})
        print(json.dumps(results[-1]),flush=True)
    save(OUT/'point_correction_run_v1.json',{'finished_at':time.time(),'results':results})


if __name__=='__main__':
    main()
