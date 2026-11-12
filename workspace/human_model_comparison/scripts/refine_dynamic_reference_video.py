"""SAM2 video mask candidates seeded only by the accepted real-source mask.

Uses all original exposures in order and preserves the exact seed mask. Native
mask candidates and model renders are untouched. Outputs still require review.
"""
import argparse
import os
from pathlib import Path
import sys
import time

from prepare_exact_reference_masks_v2 import SOURCE, WEIGHT, CONFIG, EXPECTED_WEIGHT_SHA
from review_dynamic_references import H, ROOT, read, save, sha


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--ids',nargs='+',required=True)
    args = ap.parse_args()
    from launch_dynamic_full_cohort import acquire
    slot = acquire()
    while slot is None: time.sleep(5);slot=acquire()
    lease, gpu, detail = slot
    os.environ['CUDA_VISIBLE_DEVICES']=detail['uuid']
    os.environ['OMP_NUM_THREADS']='4'
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(4)
    assert sha(WEIGHT)==EXPECTED_WEIGHT_SHA
    predictor=build_sam2_video_predictor(CONFIG,str(WEIGHT),device='cuda')
    try:
        for sid in args.ids:
            folder=ROOT/sid
            assert not (folder/'reference_qualification.json').exists()
            clip=read(folder/'clip.json')
            camera=read(H/'configs/human_source_common_v1/reviewed_source_v2'/sid/'camera.json')
            seed=next(f for f in clip['frames'] if f['timestamp_us']==camera['timestamp_us'])
            assert seed['source_rgb_sha256']==camera['source_rgb_sha256']
            assert sha(camera['source_mask'])==camera['source_mask_sha256']
            accepted=np.asarray(Image.open(camera['source_mask']).convert('L'))>127
            assert accepted.shape==tuple(reversed(seed['render_wh']))
            dest=folder/'reference_masks_video_seed_v1';dest.mkdir(exist_ok=False)
            frames=dest/'source_links';frames.mkdir()
            for f in clip['frames']:
                assert sha(f['source_rgb'])==f['source_rgb_sha256']
                (frames/f"{f['index']:05d}.jpg").symlink_to(f['source_rgb'])
            print(sid,'GPU',gpu,'seed frame',seed['index'],flush=True)
            predictions={}
            with torch.inference_mode():
                state=predictor.init_state(video_path=str(frames),offload_video_to_cpu=True)
                predictor.add_new_mask(state,frame_idx=seed['index'],obj_id=1,mask=accepted)
                for reverse in (False,True):
                    for index, ids, logits in predictor.propagate_in_video(state,start_frame_idx=seed['index'],reverse=reverse):
                        assert ids==[1]
                        predictions[index]=(logits[0,0]>0).cpu().numpy()
                # Decoder interpolation must never change the accepted seed.
                predictions[seed['index']]=accepted
            assert sorted(predictions)==list(range(len(clip['frames'])))
            rows=[]
            for f in clip['frames']:
                p=dest/f"{f['index']:05d}.png"
                Image.fromarray(predictions[f['index']].astype('uint8')*255).save(p)
                rows.append(dict(index=f['index'],timestamp_us=f['timestamp_us'],path=str(p),sha256=sha(p)))
            save(folder/'mask_candidates_video_seed_v1.json',dict(id=sid,
                status='exact_real_exposure_masks_pending_visual_review',frames=rows,
                clip_sha256=sha(folder/'clip.json'),weight_sha256=sha(WEIGHT),script_sha256=sha(__file__),
                method='SAM2.1 base-plus bidirectional video propagation from accepted exact-exposure mask',
                seed_camera=str(H/'configs/human_source_common_v1/reviewed_source_v2'/sid/'camera.json'),
                seed_frame=seed['index'],seed_timestamp_us=seed['timestamp_us'],
                seed_mask=camera['source_mask'],seed_mask_sha256=camera['source_mask_sha256'],
                seed_mask_preserved=True,reference_eligible=False,prediction_inputs_used=False,
                source_rgb_interpolation=False,precision='float32_no_autocast',gpu=gpu))
            del state,predictions;torch.cuda.empty_cache()
            print(sid,'candidates ready for visual review',flush=True)
    finally: lease.close()


if __name__=='__main__':main()
