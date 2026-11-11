"""Reference-only SAM2 correction on CPU; immutable plan, source and mask hashes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from prepare_exact_reference_masks_v2 import ROOT, SOURCE, WEIGHT, CONFIG, EXPECTED_WEIGHT_SHA, sha, save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());revision=plan['version']
    assert revision.replace('_','').isalnum()
    statefile=ROOT/'state'/f'{revision}.json'
    lock=statefile.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not statefile.exists()
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    assert sha(WEIGHT)==EXPECTED_WEIGHT_SHA
    predictor=SAM2ImagePredictor(build_sam2(CONFIG,str(WEIGHT),device='cpu'))
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),plan_sha256=sha(plan_path),
        device='cpu',precision='float32_no_autocast',script_sha256=sha(__file__),jobs=[])
    save(statefile,state)
    for row in plan['records']:
        job=dict(id=row['id'],status='running',started_at=time.time());state['jobs'].append(job);save(statefile,state)
        try:
            assert sha(row['source_path'])==row['source_sha256']
            assert sha(row['base_mask_path'])==row['base_mask_sha256']
            base=Path(row['base_mask_path']).parent;meta=json.loads((base/'metadata.json').read_text())
            image=np.array(Image.open(row['source_path']).convert('RGB'))
            positives=row['positive_raw_xy'];negatives=row['negative_raw_xy']
            points=np.array(positives+negatives,dtype=np.float32);labels=np.array([1]*len(positives)+[0]*len(negatives))
            assert np.isfinite(points).all() and (points>=0).all()
            assert (points[:,0]<image.shape[1]).all() and (points[:,1]<image.shape[0]).all()
            box=np.array(row.get('bbox_raw_xyxy',meta['bbox_prompt_raw_xyxy']),dtype=np.float32)
            with torch.inference_mode():
                predictor.set_image(image)
                masks,scores,_=predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=False)
            mask=masks[0].astype(np.uint8)*255
            assert mask.shape==image.shape[:2] and mask.any()
            dest=base/'revisions'/revision;dest.mkdir(parents=True,exist_ok=False)
            Image.fromarray(mask).save(dest/'mask.png')
            overlay=image.copy();foreground=mask>127;overlay[foreground]=(image[foreground]*.6+np.array([230,30,30])*.4).astype(np.uint8)
            crop=meta['crop_xyxy'];Image.fromarray(overlay).crop(crop).save(dest/'overlay_crop.png')
            new=dict(meta,mask_path=str(dest/'mask.png'),mask_sha256=sha(dest/'mask.png'),base_mask_sha256=row['base_mask_sha256'],
                correction=row,correction_plan=str(plan_path),correction_plan_sha256=state['plan_sha256'],
                correction_script_sha256=state['script_sha256'],weight_sha256=EXPECTED_WEIGHT_SHA,device='cpu',
                foreground_pixels=int(foreground.sum()),sam_predicted_iou=float(scores[0]),
                original_mask_unchanged=sha(row['base_mask_path'])==row['base_mask_sha256'],
                generation_inputs_unchanged=True,reference_ready=False,metric_eligible=False,visual_review='pending')
            save(dest/'metadata.json',new)
            job.update(status='candidate_pending_review',path=str(dest),foreground_pixels=new['foreground_pixels'],elapsed_seconds=time.time()-job['started_at'])
        except Exception as exc:job.update(status='failed',reason=repr(exc))
        save(statefile,state);print(json.dumps(job),flush=True)
    state.update(status='finished_candidates_pending_review',finished_at=time.time());save(statefile,state)


if __name__=='__main__':main()
