"""Prepare exact-exposure visible-mask candidates; never auto-approve references."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/nuscenes_cohort_audit_v2/audit.json'
OUT = ROOT / 'references/exact_exposure_v2'
SOURCE = Path(_hmc_expand('${HMC_THIRD_PARTY}/Grounded-SAM-2'))
WEIGHT = Path(_hmc_expand('${HMC_MODEL_CACHE}/grounded_sam2/sam2.1_hiera_base_plus.pt'))
CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
EXPECTED_WEIGHT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True).splitlines()
    if args.gpu_uuid in [s.strip() for s in occupied]:
        raise RuntimeError('Requested GPU has an existing compute process')
    assert shutil.disk_usage(ROOT).free > 11*1024**3
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['OMP_NUM_THREADS'] = '4'
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    assert sha(WEIGHT) == EXPECTED_WEIGHT_SHA
    source_files = {str(p.relative_to(SOURCE)):sha(p) for p in sorted((SOURCE/'sam2').rglob('*'))
                    if p.is_file() and p.suffix in {'.py','.yaml'}}
    provenance = {'method':'SAM2.1 base-plus single-frame box prompt',
        'source_root':str(SOURCE),'source_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        'source_file_sha256':source_files,'config':CONFIG,'weight_path':str(WEIGHT),
        'weight_sha256':EXPECTED_WEIGHT_SHA,'script_sha256':sha(__file__),
        'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.executable,
        'precision':'float32; no autocast','postprocessing':'upstream build_sam2 defaults; predictor hole/sprinkle area defaults zero',
        'multimask_output':False,'mask_threshold':0.0,
        'prompt':'projected identity bbox from exact native frame, translated to raw camera coordinates',
        'selection':'No benchmark predictions or scores used; every unexcluded candidate and frozen original is processed.',
        'approval':'Automatic mask candidate only; identity, external occlusion, truncation and mask quality require visual review.'}
    save(OUT/'provenance.json',provenance)
    model = build_sam2(CONFIG,str(WEIGHT),device='cuda')
    predictor = SAM2ImagePredictor(model)
    rows = [r for r in json.loads(AUDIT.read_text())['records'] if not r['blocked_by_prior_exclusion']]
    rows.sort(key=lambda r:(r['cohort']!='candidate44',r['id']))
    if args.limit:
        rows = rows[:args.limit]
    state = {'pid':os.getpid(),'status':'running','gpu_uuid':args.gpu_uuid,'started_at':time.time(),
             'audit_sha256':sha(AUDIT),'requested_count':len(rows),'jobs':[]}
    save(OUT/'state.json',state)
    for row in rows:
        ident = row['id']
        job = {'id':ident,'status':'running'}
        state['jobs'].append(job)
        folder = OUT/ident
        folder.mkdir(exist_ok=True)
        try:
            assert shutil.disk_usage(ROOT).free > 11*1024**3
            frame = row['reference_frame']
            source_hash = sha(frame['raw_source_path'])
            assert source_hash == row['reference_raw_sha256'] == frame['raw_source_sha256']
            meta_path = folder/'metadata.json'
            if meta_path.exists():
                previous = json.loads(meta_path.read_text())
                assert previous['source_sha256'] == source_hash and previous['timestamp_us'] == row['t_ref_us']
                assert previous['mask_sha256'] == sha(folder/'mask.png')
                assert previous['preparation_script_sha256'] == provenance['script_sha256']
                job['status'] = 'reused_existing_candidate'
                continue
            image = np.asarray(Image.open(frame['raw_source_path']).convert('RGB'))
            crop = frame['crop_xyxy']
            bbox = np.asarray(frame['bbox_input_xyxy'],dtype=float) + [crop[0],crop[1],crop[0],crop[1]]
            assert np.isfinite(bbox).all() and bbox[2]>bbox[0] and bbox[3]>bbox[1]
            # Only the target exposure is passed to the predictor. No adjacent frame is substituted.
            with torch.inference_mode():
                predictor.set_image(image)
                masks,scores,_ = predictor.predict(box=bbox.astype(np.float32),multimask_output=False)
            assert masks.shape == (1,image.shape[0],image.shape[1])
            mask = masks[0].astype(np.uint8)*255
            Image.fromarray(mask).save(folder/'mask.png')
            x0,y0,x1,y1 = map(int,crop)
            Image.fromarray(mask[y0:y1,x0:x1]).save(folder/'mask_crop.png')
            overlay = image.copy()
            inside = mask>127
            overlay[inside] = (image[inside].astype(float)*.55 + np.array([0,210,120])*.45).astype(np.uint8)
            Image.fromarray(overlay[y0:y1,x0:x1]).save(folder/'overlay_crop.jpg',quality=92)
            meta = {'id':ident,'cohort':row['cohort'],'identity_key':row['identity_key'],'scene':row['scene'],
                'timestamp_us':row['t_ref_us'],'time_status':row['t_ref_status'],
                'source_path':frame['raw_source_path'],'source_sha256':source_hash,
                'raw_shape_hw':list(image.shape[:2]),'crop_xyxy':crop,'bbox_prompt_raw_xyxy':bbox.tolist(),
                'mask_path':str(folder/'mask.png'),'mask_sha256':sha(folder/'mask.png'),
                'foreground_pixels':int(inside.sum()),'sam_predicted_iou':float(scores[0]),
                'existing_exact_input_slots':row['exact_input_slots_for_reference'],
                'mask_origin':'new_exact_exposure_sam2_box_candidate',
                'provenance':str(OUT/'provenance.json'),'preparation_script_sha256':provenance['script_sha256'],
                'visual_review':'pending','reference_ready':False,'metric_eligible':False,
                'notes':'Visible foreground only. No automatic inference of complete unoccluded body.'}
            save(meta_path,meta)
            job.update(status='candidate_ready_pending_visual_review',foreground_pixels=meta['foreground_pixels'])
        except Exception as exc:
            job.update(status='failed',reason=repr(exc))
        finally:
            state['updated_at']=time.time()
            save(OUT/'state.json',state)
            print(json.dumps(job),flush=True)
    state.update(status='finished_with_failures' if any(j['status']=='failed' for j in state['jobs']) else 'mask_candidates_ready_pending_visual_review',finished_at=time.time())
    save(OUT/'state.json',state)


if __name__ == '__main__':
    main()
