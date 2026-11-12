"""CPU detector-conditioned keypoints; ROI belongs to the pose model, not evaluation."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from audit_static_source_cameras_v1 import H, sha, read, checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=read(args.manifest)
    for case in manifest['cases']:checked(case['image'],case['image_sha256'])
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='loading_cpu_models',pid=os.getpid(),started_at=time.time(),
               manifest=str(args.manifest.resolve()),manifest_sha256=sha(args.manifest),
               script_sha256=sha(__file__),device='cpu',cases=[])
    def save():
        state['updated_at']=time.time();tmp=args.output/'state.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.output/'state.json')
    save()
    try:
        import torch
        from mmdet.apis import init_detector,inference_detector
        from mmpose.apis import init_model,inference_topdown
        from mmpose.evaluation.functional import nms
        from mmpose.utils import adapt_mmdet_pipeline
        torch.set_num_threads(4)
        prep=H/'model_preparation/20261110';source=prep/'sources/exavatar/fitting/tools/mmpose'
        configs=[source/'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
                 source/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py']
        weights=[prep/'weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',prep/'weights/dw-ll_ucoco_384.pth']
        hashes=['235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9',
                '0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07']
        for path,digest in zip(weights,hashes):checked(path,digest)
        detector=init_detector(str(configs[0]),str(weights[0]),device='cpu')
        detector.cfg=adapt_mmdet_pipeline(detector.cfg)
        pose=init_model(str(configs[1]),str(weights[1]),device='cpu')
        for model,weight in zip([detector,pose],weights):
            ckpt=torch.load(weight,map_location='cpu',weights_only=False)
            params={k.removeprefix('module.'):v for k,v in ckpt.get('state_dict',ckpt).items()}
            model.load_state_dict(params,strict=True);model.eval();del ckpt,params
        state.update(status='running_cpu_only',strict_weight_load_passed=True,
                     configs=[dict(path=str(p),sha256=sha(p)) for p in configs],
                     weights=[dict(path=str(p),sha256=sha(p)) for p in weights],torch_version=torch.__version__,
                     detector_score_threshold=.3,detector_nms_iou=.3,
                     selection_policy='exactly one person after fixed threshold and NMS; zero/multiple people are explicit NA',
                     comparison_image_modified=False)
        save()
        for case in manifest['cases']:
            started=time.time()
            with torch.inference_mode():
                detected=inference_detector(detector,case['image']).pred_instances.cpu().numpy()
            boxes=np.c_[detected.bboxes,detected.scores]
            boxes=boxes[(detected.labels==0)&(detected.scores>.3)]
            if len(boxes):boxes=boxes[nms(boxes,.3)]
            if not np.isfinite(boxes).all():raise ValueError('Non-finite detection')
            rec=dict(case=case,detected_people=len(boxes),detections_xyxy_score=boxes.tolist(),
                     detector_roi_used_only_inside_pose_inference=True,original_image_unchanged=True,
                     confidence_is_visibility=False,model_prediction_not_ground_truth=True,
                     semantic_identity_validated=False,metric_eligible=False)
            if len(boxes)!=1:
                rec.update(status='keypoints_unavailable',reason='no_person_detected' if not len(boxes) else 'multiple_person_detections',
                           keypoints=None,keypoint_scores=None)
            else:
                with torch.inference_mode():pred=inference_topdown(pose,case['image'],boxes[:,:4])
                if len(pred)!=1:raise ValueError('Unexpected pose count')
                xy=np.asarray(pred[0].pred_instances.keypoints)[0]
                scores=np.asarray(pred[0].pred_instances.keypoint_scores)[0]
                if xy.shape!=(133,2) or scores.shape!=(133,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
                    raise ValueError('Invalid whole-body keypoints')
                rec.update(status='observed_pending_semantic_review',keypoints=xy.tolist(),keypoint_scores=scores.tolist(),
                           keypoint_layout='COCO WholeBody 133; first 17 COCO body joints',
                           body_joints_confidence_gt_05=int((scores[:17]>.5).sum()))
                if case['role']=='reference':
                    raw=np.c_[xy,np.ones(len(xy))] @ np.asarray(case['roi_to_raw_integer_center_affine']).T
                    rec['keypoints_raw_pixels']=(raw[:,:2]/raw[:,2:3]).tolist()
            rec['elapsed_seconds']=time.time()-started
            path=args.output/(case['case_id']+'.json');path.write_text(json.dumps(rec,indent=2)+'\n')
            short=dict(case_id=case['case_id'],status=rec['status'],record=str(path.resolve()),sha256=sha(path),
                       detected_people=len(boxes),body_joints_confidence_gt_05=rec.get('body_joints_confidence_gt_05'))
            state['cases'].append(short);save();print(json.dumps(short),flush=True)
        state.update(status='cpu_keypoint_observations_complete_pending_review',common_camera_valid=False,
                     source_orientation_accepted=False,metric_eligible=False,scores_produced=False)
    except BaseException as error:
        state.update(status='failed',reason=repr(error));raise
    finally:
        state['finished_at']=time.time();save()


if __name__=='__main__':main()
