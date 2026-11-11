"""Validate official RTMDet/DWPose checkpoints on upstream and frozen real input."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl, hashlib, json, os, subprocess, sys, time

P=Path(__file__).resolve().parents[1]
H=P.parents[1]
R=H.parent/'vehicle_model_comparison/model_preparation/20261109_2100_CST'
STATE=P/'state/exavatar_pose_inference_v1.json'
OUT=P/'reports/exavatar_pose_inference_v1'
SRC=P/'sources/exavatar/fitting/tools/mmpose'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

def worker():
    import numpy as np, torch, cv2
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model, inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    weights=[P/'weights/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',P/'weights/dw-ll_ucoco_384.pth']
    hashes=['235e820939cb2ff33c505441e71f7e9532958c281636a963c6829d100867aed9','0d9408b13cd863c4e95a149dd31232f88f2a12aa6cf8964ed74d7d97748c7a07']
    for w,digest in zip(weights,hashes): assert sha(w)==digest
    configs=[SRC/'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py', SRC/'configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py']
    det=init_detector(str(configs[0]),str(weights[0]),device='cuda:0')
    det.cfg=adapt_mmdet_pipeline(det.cfg)
    pose=init_model(str(configs[1]),str(weights[1]),device='cuda:0')
    strict=[]
    for model,w in zip((det,pose),weights):
        checkpoint=torch.load(w,map_location='cpu',weights_only=False)
        state=checkpoint.get('state_dict',checkpoint)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        # Fail on unmatched parameters instead of accepting a partially initialized network.
        result=model.load_state_dict(state,strict=True)
        strict.append(dict(weight=str(w),missing=list(result.missing_keys),unexpected=list(result.unexpected_keys)))
        del checkpoint,state
    camera=json.loads((H/'configs/human_source_common_v1/candidates/HN0004/camera.json').read_text())
    assert sha(camera['source_rgb'])==camera['source_rgb_sha256']
    cases=[('upstream_coco',SRC/'tests/data/coco/000000000785.jpg'),('HN0004',Path(camera['source_rgb']))]
    records=[]
    for name,path in cases:
        pred=inference_detector(det,str(path)).pred_instances.cpu().numpy()
        boxes=np.concatenate((pred.bboxes,pred.scores[:,None]),axis=1)
        boxes=boxes[(pred.labels==0)&(pred.scores>.3)]
        boxes=boxes[nms(boxes,.3)]
        assert len(boxes)>0, name
        results=inference_topdown(pose,str(path),boxes[:,:4])
        assert len(results)==len(boxes)
        people=[]; image=cv2.imread(str(path))
        for bbox,result in zip(boxes,results):
            inst=result.pred_instances
            points=np.asarray(inst.keypoints);scores=np.asarray(inst.keypoint_scores)
            assert points.shape==(1,133,2) and scores.shape==(1,133)
            assert np.isfinite(points).all() and np.isfinite(scores).all()
            people.append(dict(bbox_xyxy=bbox[:4].tolist(),detection_score=float(bbox[4]),keypoints=points[0].tolist(),keypoint_scores=scores[0].tolist()))
            cv2.rectangle(image,tuple(bbox[:2].astype(int)),tuple(bbox[2:4].astype(int)),(0,255,0),2)
            for xy,score in zip(points[0],scores[0]):
                if score>.3: cv2.circle(image,tuple(xy.astype(int)),2,(0,0,255),-1)
        output=OUT/f'{name}.json'
        save(output,dict(source=str(path),source_sha256=sha(path),people=people,identity_assignment_validated=False))
        assert cv2.imwrite(str(OUT/f'{name}.jpg'),image)
        records.append(dict(name=name,source=str(path),source_sha256=sha(path),detected_people=len(people),predictions=str(output),predictions_sha256=sha(output)))
    save(OUT/'validation.json',dict(status='detector_and_133_keypoint_gpu_inference_passed',strict_weight_loads=strict,cases=records,
        configs=[dict(path=str(c),sha256=sha(c)) for c in configs],weights=[dict(path=str(w),sha256=sha(w)) for w in weights],
        torch_version=torch.__version__,full_exavatar_demo_passed=False,identity_assignment_validated=False,
        loader_compatibility='TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 in this worker for hash-verified official checkpoints and upstream MMEngine loader; parameters unchanged.'))

def main():
    lock=STATE.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    s=dict(status='waiting_pose_install',pid=os.getpid(),started_at=time.time(),script_sha256=sha(__file__))
    save(STATE,s)
    try:
        while True:
            install=json.loads((P/'state/exavatar_pose_install_v1.json').read_text())
            if install['status']=='pose_imports_passed_weight_inference_pending': break
            assert install['status']=='installing',install['status']
            os.kill(install['pid'],0)
            time.sleep(10)
        s['status']='waiting_shared_gpu_lease';save(STATE,s)
        sys.path.insert(0,str(R/'scripts'));import dispatch_rigid111_direct as resources
        while True:
            with (R/'runs/rigid111_gpu_locks/gpu_2.lock').open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX)
                ready,detail=resources.available('2')
                if ready:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=detail['uuid'],PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
                    with (P/'logs/exavatar_pose_inference_worker_v1.log').open('x') as log:
                        child=subprocess.Popen([_hmc_expand('${HMC_ENV_ROOT}/hcmp_exavatar/bin/python'),str(Path(__file__)),'--worker'],cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
                        s.update(status='running',child_pid=child.pid,gpu='2');save(STATE,s)
                        code=child.wait()
                    s.update(status='passed' if code==0 else 'failed',exit_code=code);break
            time.sleep(5)
    except BaseException as exc:
        s.update(status='failed',reason=repr(exc));raise
    finally:
        s['finished_at']=time.time();save(STATE,s)

if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
