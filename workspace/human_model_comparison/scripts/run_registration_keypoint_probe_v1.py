"""CPU-only DWPose observations for a fixed diagnostic manifest; not ground truth."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from audit_static_source_cameras_v1 import H, sha, read, checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=read(args.manifest)
    for row in manifest['cases']:
        checked(row['image'],row['image_sha256'])
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='loading_cpu_keypoint_model',pid=os.getpid(),started_at=time.time(),
                manifest=str(args.manifest),manifest_sha256=sha(args.manifest),cases=[])
    def save():
        status['updated_at']=time.time()
        temp=args.output/'state.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(args.output/'state.json')
    save()
    try:
        import torch
        from mmpose.apis import init_model,inference_topdown
        torch.set_num_threads(4)
        prep=H/'model_preparation/20261110'
        source=prep/'sources/exavatar/fitting/tools/mmpose'
        config=source/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        weight=prep/'weights/dw-ll_ucoco_384.pth'
        checked(weight,'0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07')
        model=init_model(str(config),str(weight),device='cpu')
        ckpt=torch.load(weight,map_location='cpu',weights_only=False)
        state={k.removeprefix('module.'):v for k,v in ckpt.get('state_dict',ckpt).items()}
        model.load_state_dict(state,strict=True)
        del ckpt,state
        model.eval()
        status.update(status='running_cpu_only',weight=str(weight),weight_sha256=sha(weight),
                      config=str(config),config_sha256=sha(config),strict_weight_load_passed=True,
                      torch_version=torch.__version__,device='cpu',torch_threads=4,
                      script_sha256=sha(__file__))
        save()
        for case in manifest['cases']:
            started=time.time()
            with Image.open(case['image']) as im:
                wh=im.size
            if wh!=(512,512):raise ValueError('Probe expects existing 512x512 isolated views')
            with torch.inference_mode():
                predictions=inference_topdown(model,case['image'],np.array([[0.,0.,512.,512.]],dtype=np.float32))
            if len(predictions)!=1:raise ValueError('Expected one fixed-ROI pose result')
            inst=predictions[0].pred_instances
            xy=np.asarray(inst.keypoints)[0];scores=np.asarray(inst.keypoint_scores)[0]
            if xy.shape!=(133,2) or scores.shape!=(133,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
                raise ValueError('Invalid whole-body keypoints')
            record=dict(case=case,keypoints=xy.tolist(),keypoint_scores=scores.tolist(),
                        keypoint_layout='COCO WholeBody 133; first 17 COCO body joints',
                        full_image_roi_xyxy=[0,0,512,512],confidence_is_visibility=False,
                        model_prediction_not_ground_truth=True,metric_eligible=False,
                        elapsed_seconds=time.time()-started)
            if case['role']=='reference':
                raw=np.c_[xy,np.ones(len(xy))] @ np.asarray(case['roi_to_raw_integer_center_affine']).T
                record['keypoints_raw_pixels']=(raw[:,:2]/raw[:,2:3]).tolist()
            path=args.output/(case['case_id']+'.json');path.write_text(json.dumps(record,indent=2)+'\n')
            status['cases'].append(dict(case_id=case['case_id'],record=str(path),sha256=sha(path),
                                       body_joints_confidence_gt_05=int((scores[:17]>.5).sum()),
                                       elapsed_seconds=record['elapsed_seconds']))
            save();print(json.dumps(status['cases'][-1]),flush=True)
        status.update(status='cpu_keypoint_observations_complete_pending_review',metric_eligible=False,
                      source_orientation_accepted=False,scores_produced=False)
    except BaseException as error:
        status.update(status='failed',reason=repr(error));raise
    finally:
        status['finished_at']=time.time();save()


if __name__=='__main__':main()
