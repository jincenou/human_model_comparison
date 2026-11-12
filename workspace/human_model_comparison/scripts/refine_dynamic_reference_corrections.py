"""Versioned source-only corrections, propagated without touching model renders.

Manual prompts come from inspected raw RGB coordinates. Original masks remain
immutable; corrected candidates still require all-frame visual review.
"""
import argparse
import os
from pathlib import Path
import sys
import time

from review_dynamic_references import H, ROOT, read, save, sha
from prepare_exact_reference_masks_v2 import SOURCE, WEIGHT, CONFIG, EXPECTED_WEIGHT_SHA


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--ids',nargs='+',required=True)
    args=parser.parse_args();plan_path=Path(args.plan);plan=read(plan_path);version=plan['version']
    assert version.replace('_','').isalnum()
    from launch_dynamic_full_cohort import acquire
    slot=acquire()
    while slot is None:time.sleep(5);slot=acquire()
    lease,gpu,detail=slot;os.environ['CUDA_VISIBLE_DEVICES']=detail['uuid'];os.environ['OMP_NUM_THREADS']='4'
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4);assert sha(WEIGHT)==EXPECTED_WEIGHT_SHA
    image_predictor=SAM2ImagePredictor(build_sam2(CONFIG,str(WEIGHT),device='cuda'))
    video=build_sam2_video_predictor(CONFIG,str(WEIGHT),device='cuda')
    try:
        for sid in args.ids:
            folder=ROOT/sid;assert not (folder/'reference_qualification.json').exists()
            clip=read(folder/'clip.json');base_path=folder/'mask_candidates_video_seed_v1.json';base=read(base_path)
            assert base['clip_sha256']==sha(folder/'clip.json')
            config=plan['people'][sid];assert config['clip_sha256']==sha(folder/'clip.json')
            dest=folder/('reference_masks_'+version);dest.mkdir(exist_ok=False)
            keys={};edits=[]
            with torch.inference_mode():
                for spec in config['seeds']:
                    i=spec['index'];frame=clip['frames'][i];old=base['frames'][i]
                    assert sha(frame['source_rgb'])==frame['source_rgb_sha256'] and sha(old['path'])==old['sha256']
                    rgb=np.asarray(Image.open(frame['source_rgb']).convert('RGB'))
                    prior=np.asarray(Image.open(old['path']).convert('L'))>127;mask=prior.copy()
                    for action in spec['operations']:
                        if action['kind']=='add_accessory':
                            image_predictor.set_image(rgb)
                            box=action['bbox'];positives=action['positive'];negatives=action.get('negative',[])
                            assert all(0<=x<rgb.shape[1] and 0<=y<rgb.shape[0] for x,y in positives+negatives)
                            masks,scores,_=image_predictor.predict(point_coords=np.array(positives+negatives,np.float32),
                                point_labels=np.array([1]*len(positives)+[0]*len(negatives)),box=np.array(box,np.float32),multimask_output=False)
                            region=np.zeros_like(mask);x0,y0,x1,y1=box;region[y0:y1,x0:x1]=True
                            mask |= masks[0].astype(bool)&region
                        elif action['kind']=='remove_polygon':
                            region=Image.new('L',(rgb.shape[1],rgb.shape[0]),0);ImageDraw.Draw(region).polygon([tuple(p) for p in action['points']],fill=255)
                            mask &= np.asarray(region)==0
                        else:raise ValueError(action['kind'])
                    path=dest/f'seed_{i:05d}.png';Image.fromarray(mask.astype('uint8')*255).save(path)
                    keys[i]=(mask&~prior) if config.get('propagate_additions_only') else mask
                    assert keys[i].any(), 'Empty correction seed'
                    edits.append(dict(index=i,path=str(path),sha256=sha(path),base_mask=old['path'],base_mask_sha256=old['sha256'],
                        source_rgb_sha256=frame['source_rgb_sha256'],added_pixels=int((mask&~prior).sum()),removed_pixels=int((prior&~mask).sum()),operations=spec['operations']))
                state=video.init_state(video_path=str(folder/'reference_masks_video_seed_v1/source_links'),offload_video_to_cpu=True)
                for i,mask in sorted(keys.items()):video.add_new_mask(state,frame_idx=i,obj_id=1,mask=mask)
                predictions={};start=min(keys)
                for reverse in (False,True):
                    for i,ids,logits in video.propagate_in_video(state,start_frame_idx=start,reverse=reverse):
                        assert ids==[1];predictions[i]=(logits[0,0]>0).cpu().numpy()
                predictions.update(keys)
            assert sorted(predictions)==list(range(len(clip['frames'])))
            rows=[]
            for frame in clip['frames']:
                i=frame['index']
                if config.get('propagate_additions_only'):
                    old=base['frames'][i];assert sha(old['path'])==old['sha256']
                    predictions[i] |= np.asarray(Image.open(old['path']).convert('L'))>127
                p=dest/f'{i:05d}.png';Image.fromarray(predictions[i].astype('uint8')*255).save(p)
                rows.append(dict(index=i,timestamp_us=frame['timestamp_us'],path=str(p),sha256=sha(p)))
            save(folder/('mask_candidates_'+version+'.json'),dict(id=sid,status='corrected_candidates_pending_all_frame_visual_review',
                frames=rows,clip_sha256=sha(folder/'clip.json'),base_candidate_manifest=str(base_path),base_candidate_sha256=sha(base_path),
                correction_plan=str(plan_path),correction_plan_sha256=sha(plan_path),seed_edits=edits,
                prediction_inputs_used=False,source_rgb_interpolation=False,reference_eligible=False,static_reference_unchanged=True,
                propagate_additions_only=config.get('propagate_additions_only',False),
                weight_sha256=sha(WEIGHT),script_sha256=sha(__file__),gpu=gpu,precision='float32_no_autocast'))
            print(sid,'candidate correction ready',edits,flush=True)
            del state,predictions;torch.cuda.empty_cache()
    finally:lease.close()


if __name__=='__main__':main()
