"""Prepare, observe and score the disclosed common-ROI ED-P variant."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common_human_orbit import H,read,save,sha,checked,METHODS
import human_source_metrics_v1 as core
from human_edp_common import PARTS,SEGMENTS,part_labels,pool_parts,compare_parts

PREP=H/'model_preparation/20261110'
POSE_SOURCE=PREP/'sources/exavatar/fitting/tools/mmpose'
POSE_CONFIGS=[POSE_SOURCE/'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
    POSE_SOURCE/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py']
POSE_WEIGHTS=[Path(_hmc_expand('${HMC_WEIGHT_ROOT}'))/name for name in
    ['rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth','dw-ll_ucoco_384.pth']]
DINO=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST/weights/facebook--dinov3-vitl16-pretrain-lvd1689m'
PY_POSE=_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python')
PY_FEATURE=_hmc_expand('${CONDA_ROOT}/envs/ab_cmp_trellis2/bin/python')


def prepare(root,ids):
    assert ids and len(ids)==len(set(ids))
    root.mkdir(parents=True,exist_ok=False)
    for name in ['images','masks','poses','features']:(root/name).mkdir()
    source=H/'reports/source_camera_metric_cohort_v3/summary.json'
    raw=source.read_bytes();(root/'source_summary.json').write_bytes(raw);summary=json.loads(raw)
    active=read(H/'inputs/active_human_benchmark.json');assert summary['selection_sha256']==active['selection_sha256']
    checked(active['selection'],active['selection_sha256']);selected={r['id']:r for r in read(active['selection'])['objects']}
    rows={(r['id'],r['method']):r for r in summary['rows']};cases=[]
    for sid in ids:
        ref=rows[(sid,'ours')]['reference'];item=selected[sid]
        assert ref['identity_key']==item['identity_key'] and ref['timestamp_us']==item['t_ref_us']
        for k in ['reference_rgb','reference_mask','camera_contract']:checked(ref[k],ref[k+'_sha256'])
        reference=core.prepare_reference(np.array(Image.open(ref['reference_rgb']).convert('RGB')),np.array(Image.open(ref['reference_mask']).convert('L')))
        assert list(reference['roi'])==ref['roi_xyxy']
        prepared=[('reference',reference,dict(reference=ref))]
        for method in METHODS:
            row=rows[(sid,method)]
            if row['status']!='scored':continue
            checked(row['result'],row['result_sha256']);result=read(row['result'])
            checked(result['registration_record'],result['registration_sha256']);cert=read(result['registration_record'])
            assert cert['identity']==sid and cert['identity_key']==ref['identity_key']
            assert cert['timestamp_us']==ref['timestamp_us'] and cert['camera_sha256']==ref['camera_contract_sha256']
            for k in ['source_rgb','source_alpha']:checked(cert[k],cert[k+'_sha256'])
            pred=core.prepare_prediction(np.array(Image.open(cert['source_rgb']).convert('RGB')),np.array(Image.open(cert['source_alpha']).convert('L')),reference['roi'],cert['prediction_encoding'])
            prepared.append((method,pred,dict(result=row['result'],result_sha256=row['result_sha256'],registration=result['registration_record'],registration_sha256=result['registration_sha256'])))
        for method,arrays,evidence in prepared:
            key=sid+'_'+method;ip=root/'images'/(key+'.png');mp=root/'masks'/(key+'.png')
            Image.fromarray(arrays['rgb']).save(ip);Image.fromarray(arrays['mask']).save(mp)
            flags=ref.get('reference_quality_flags',[])
            excluded=['head'] if 'umbrella_occluded_input_state' in flags else []
            cases.append(dict(key=key,id=sid,method=method,image=str(ip),image_sha256=sha(ip),mask=str(mp),mask_sha256=sha(mp),evidence=evidence,
                reference_quality_flags=flags,reference_excluded_parts=excluded))
    weights=[*POSE_WEIGHTS,DINO/'model.safetensors']
    expected=['235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9','0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07','dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179']
    for p,d in zip(weights,expected):checked(p,d)
    protocol=dict(name='human_edp_common_candidate_v2',status='pilot_pending_visual_review',
        parts=list(PARTS),segments=SEGMENTS,common_roi=512,independent_bbox_normalization=False,
        backbone='facebook/dinov3-vitl16-pretrain-lvd1689m',layer='last_hidden_state after final model norm; exclude CLS and four register tokens',
        feature_dtype='float32; eval; no autocast or TF32; eager attention',pool_dtype='float64',
        preprocessing='Already-frozen512 sRGB common ROI /255, ImageNet mean(.485,.456,.406), std(.229,.224,.225); no further crop or resize',
        detector='RTMDet-m person + RTMPose-l wholebody DWPose; first17 COCO joints',
        detector_score=.3,detector_nms=.3,multiple_people='NA, no identity selection by score',
        patch_size=16,keypoint_confidence=.3,max_distance_patches=2,
        foreground='each image uses its own mask >127 at patch centre; reference mask never applied to prediction',
        visibility='estimated keypoint confidence and own visible foreground; endpoints inside canvas; not anatomical visibility ground truth',
        reference_visibility_rules={'umbrella_occluded_input_state':['head']},
        reference_visibility_note='Previously reviewed v5 reference flags identify fully hidden heads; exclude this part for every method without reassigning its patches. Partially visible torso/limbs retain their visible foreground. No whole-person exclusion; full masks and main scores remain unchanged.',
        minimum_parts=1,empty_common_parts='NA with reason, not zero',
        aggregation='mean cosine distance over pairwise common valid parts; report each part and coverage; cross-method tables require shared identities AND shared parts',
        differences_from_AH=['fixed common ROI without independent cropping/alignment','DINOv3 ViT-L16 instead of released eval ViT-H16+','RTMDet/RTMPose instead of SAM3D Body','retain1-3 supported parts with coverage instead of four-part cutoff','empty common parts and degenerate embeddings are NA'],
        paper='https://arxiv.org/html/2604.18468v1#S4.SS2.SSS2',
        upstream=str(H.parent/'assetbench/repos/ah/benchmark/embedding_metrics.py'),
        upstream_sha256=sha(H.parent/'assetbench/repos/ah/benchmark/embedding_metrics.py'),
        upstream_eval_sha256=sha(H.parent/'assetbench/repos/ah/benchmark/eval.py'),
        weights={str(p.resolve()):d for p,d in zip(weights,expected)},
        configurations={str(p):sha(p) for p in [*POSE_CONFIGS,DINO/'config.json']},
        code={str(p):sha(p) for p in [Path(__file__),H/'scripts/human_edp_common.py',H/'scripts/human_source_metrics_v1.py']})
    save(root/'protocol.json',protocol)
    save(root/'inputs.json',dict(ids=ids,cases=cases,summary_sha256=sha(root/'source_summary.json'),selection_sha256=summary['selection_sha256'],protocol_sha256=sha(root/'protocol.json')))
    print(dict(prepared=len(cases),ids=ids),flush=True)


def verified_inputs(root):
    manifest=read(root/'inputs.json');p=read(root/'protocol.json')
    checked(root/'protocol.json',manifest['protocol_sha256'])
    for mapping in ['weights','configurations','code']:
        for path,digest in p[mapping].items():checked(path,digest)
    for c in manifest['cases']:
        checked(c['image'],c['image_sha256']);checked(c['mask'],c['mask_sha256'])
    return manifest,p


def pose(root):
    manifest,protocol=verified_inputs(root)
    import torch,mmpose,mmdet,mmcv
    from mmdet.apis import init_detector,inference_detector
    from mmpose.apis import init_model,inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    torch.set_num_threads(4)
    detector=init_detector(str(POSE_CONFIGS[0]),str(POSE_WEIGHTS[0]),device='cuda');detector.cfg=adapt_mmdet_pipeline(detector.cfg)
    estimator=init_model(str(POSE_CONFIGS[1]),str(POSE_WEIGHTS[1]),device='cuda')
    for model,weight in zip([detector,estimator],POSE_WEIGHTS):
        checkpoint=torch.load(weight,map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint.get('state_dict',checkpoint).items()},strict=True);model.eval()
    for i,c in enumerate(manifest['cases']):
        with torch.inference_mode():d=inference_detector(detector,c['image']).pred_instances.cpu().numpy()
        boxes=np.c_[d.bboxes,d.scores];boxes=boxes[(d.labels==0)&(d.scores>protocol['detector_score'])]
        if len(boxes):boxes=boxes[nms(boxes,protocol['detector_nms'])]
        record=dict(key=c['key'],image_sha256=c['image_sha256'],boxes=boxes.tolist(),keypoints=None,
            status='no_person_detected' if not len(boxes) else 'multiple_person_detections')
        if len(boxes)==1:
            with torch.inference_mode():pred=inference_topdown(estimator,c['image'],boxes[:,:4])
            xy=np.asarray(pred[0].pred_instances.keypoints)[0,:17];scores=np.asarray(pred[0].pred_instances.keypoint_scores)[0,:17]
            k=np.c_[xy,scores];assert k.shape==(17,3) and np.isfinite(k).all()
            record.update(status='observed',keypoints=k.tolist())
        save(root/'poses'/(c['key']+'.json'),record)
        save(root/'pose_progress.json',dict(done=i+1,total=len(manifest['cases']),last=c['key']))
    save(root/'pose_execution.json',dict(torch=torch.__version__,mmpose=mmpose.__version__,mmdet=mmdet.__version__,mmcv=mmcv.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],count=len(manifest['cases'])))


def features(root):
    manifest,protocol=verified_inputs(root)
    import torch,transformers
    from transformers import AutoModel
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModel.from_pretrained(str(DINO),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    assert model.config.patch_size==16 and model.config.hidden_size==1024 and model.config.num_register_tokens==4
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    for i,c in enumerate(manifest['cases']):
        posepath=root/'poses'/(c['key']+'.json');p=read(posepath);assert p['image_sha256']==c['image_sha256']
        rgb=np.array(Image.open(c['image']).convert('RGB'));mask=np.array(Image.open(c['mask']).convert('L'))
        assert rgb.shape==(512,512,3) and mask.shape==(512,512)
        labels=part_labels(mask,p['keypoints'],excluded_parts=c['reference_excluded_parts'])
        x=torch.tensor(rgb,device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255
        with torch.inference_mode():out=model(pixel_values=(x-mean)/std).last_hidden_state[:,5:]
        assert out.shape==(1,1024,1024)
        embedding,counts=pool_parts(out[0].reshape(32,32,1024).cpu().numpy(),labels)
        path=root/'features'/(c['key']+'.npz');np.savez_compressed(path,embedding=embedding,counts=counts,labels=labels)
        save(path.with_suffix('.json'),dict(key=c['key'],image_sha256=c['image_sha256'],mask_sha256=c['mask_sha256'],pose_sha256=sha(posepath),features_sha256=sha(path),counts=counts.tolist()))
        save(root/'feature_progress.json',dict(done=i+1,total=len(manifest['cases']),last=c['key']))
    save(root/'feature_execution.json',dict(torch=torch.__version__,transformers=transformers.__version__,gpu=os.environ['CUDA_VISIBLE_DEVICES'],dtype='float32',count=len(manifest['cases'])))


def score(root):
    manifest,protocol=verified_inputs(root);cases={c['key']:c for c in manifest['cases']};rows=[]
    colors=np.array([[240,80,80],[230,190,60],[60,175,110],[75,130,230],[220,125,45],[160,90,210]],np.uint8)
    contact=Image.new('RGB',(7*208,len(manifest['ids'])*250),'white');draw=ImageDraw.Draw(contact)
    for y,sid in enumerate(manifest['ids']):
        ref=np.load(root/'features'/(sid+'_reference.npz'))
        for col,method in enumerate(['reference',*METHODS]):
            key=sid+'_'+method
            if key not in cases:
                draw.text((col*208+4,y*250+10),sid+' '+method+' NA',fill='black');continue
            c=cases[key];p=root/'features'/(key+'.npz');meta=read(p.with_suffix('.json'));checked(p,meta['features_sha256']);f=np.load(p)
            rgb=np.array(Image.open(c['image']).convert('RGB'));labels=f['labels'].repeat(16,0).repeat(16,1);valid=labels>=0
            overlay=rgb.copy();overlay[valid]=np.round(.75*rgb[valid]+.25*colors[labels[valid]]).astype(np.uint8)
            contact.paste(Image.fromarray(overlay).resize((208,208)),(col*208,y*250+38))
            draw.text((col*208+4,y*250+4),sid+' '+method,fill='black')
            if method!='reference':
                result=compare_parts(ref['embedding'],f['embedding'],ref['counts'],f['counts'])
                rows.append(dict(id=sid,method=method,**result))
                draw.text((col*208+4,y*250+20),f'parts {result["valid_part_count"]}/6',fill='black')
    contact.save(root/'part_review.jpg',quality=94)
    save(root/'results.json',dict(status='pilot_scores_pending_visual_acceptance',ids=manifest['ids'],rows=rows,
        protocol_sha256=sha(root/'protocol.json'),inputs_sha256=sha(root/'inputs.json'),
        source_core_scores_modified=False,claimed_equal_to_AH_official=False))
    print(dict(results=len(rows),valid=sum(r['score']['status']=='ok' for r in rows)),flush=True)


def launch(root,gpu):
    policy=read(H/'state/gpu_resource_policy.json');assert gpu in policy['allowed_gpu_indices'] and gpu not in policy['excluded_gpu_indices']
    assert not (root/'controller.json').exists(),'Inspect previous handle; no automatic restart'
    with (H/'state'/f'gpu_auxiliary_slot_{gpu}.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert int(info[1])>=32768 and shutil.disk_usage(H).free>10*1024**3
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=info[0].strip(),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='/tmp/hmc_pose_mpl')
        state=dict(status='running',pid=os.getpid(),gpu=gpu,started_at=time.time(),stages=[]);save(root/'controller.json',state)
        for action,python in [('pose',PY_POSE),('features',PY_FEATURE),('score',sys.executable)]:
            command=[python,'-B','-u',str(Path(__file__).resolve()),action,'--run-name',root.name]
            entry=dict(action=action,status='running',started_at=time.time());state['stages'].append(entry)
            with (root/(action+'.log')).open('x') as stream:child=subprocess.Popen(command,cwd=H,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
            entry['pid']=child.pid;save(root/'controller.json',state);code=child.wait();entry.update(exit_code=code,status='finished' if code==0 else 'failed_retained');save(root/'controller.json',state)
            if code:
                state['status']='failed_retained';save(root/'controller.json',state);raise RuntimeError(action+' failed; inspect retained log')
        state.update(status='pilot_finished_pending_visual_review',finished_at=time.time());save(root/'controller.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pose','features','score','launch']);p.add_argument('--run-name',default='pilot');p.add_argument('--ids',nargs='+');p.add_argument('--gpu',type=int,default=3);a=p.parse_args()
    assert a.run_name.replace('_','').isalnum();root=H/'reports/source_edp_common_v1'/a.run_name
    if a.action=='prepare':prepare(root,a.ids)
    elif a.action=='launch':launch(root,a.gpu)
    else:globals()[a.action](root)
