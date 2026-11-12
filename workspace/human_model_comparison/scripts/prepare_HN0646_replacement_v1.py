"""Audit an unexcluded existing person and segment its exact real exposure."""
from human_asset_benchmark.paths import expand as _hmc_expand
import json, statistics, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from audit_nuscenes_cohort_v2 import candidate_paths, npz_headers, sha

H=Path(__file__).resolve().parents[1]
def main():
    sid='HN0646';out=H/'reports/replacement_HN0035_20261112_v1'
    selection=H/'inputs/benchmark_human100_v2/selection.json';old=json.loads(selection.read_text())
    row=next(r for r in json.loads((H/'inputs/inventory.json').read_text())['eligible'] if r['id']==sid)
    decisions=json.loads((H/'inputs/manual_decisions.json').read_text())
    assert decisions[sid]=='keep' and sid not in {r['id'] for r in old['objects']}
    assert sid not in old['historical_candidate_exclusions_preserved'] and sid not in {r['id'] for r in old['exclusions_this_review']}
    assert row['scene'] in old['allowed_scenes']
    paths=row['paths'];assert all(Path(p).is_file() for p in paths.values())
    native=json.loads(Path(paths['native_metadata']).read_text());mapping=json.loads(Path(paths['frame_mapping']).read_text())
    assert native['scene']==row['scene'] and 'nuscenes:'+native['track_id']==row['identity_key']
    scene=json.loads((out/'candidate_source.json').read_text())['scene_record']
    assert sha(scene['split'])==scene['split_sha256']
    split=json.loads(Path(scene['split']).read_text());train={f['sample_data_token'] for f in split['files'] if f.get('is_key_frame') and f.get('split')=='train'}
    eligible=[f for f in native['frames'] if f.get('sample_data_token') in train and not f.get('interpolated') and not f.get('enhanced')]
    mid=statistics.median(f['frame_timestamp_micros'] for f in native['frames'])
    frame=min(eligible,key=lambda f:(abs(f['frame_timestamp_micros']-mid),f['frame_timestamp_micros']))
    assert sha(frame['raw_source_path'])==frame['raw_source_sha256'] and sha(frame['path'])==frame['sha256']
    with np.load(paths['runtime'],allow_pickle=False) as z:
        numeric={k:bool(np.isfinite(z[k]).all()) for k in z.files if z[k].dtype.kind in 'fiu'}
        assert all(numeric.values());assert len(z['timestamps'])>1 and np.all(np.diff(z['timestamps'])>0)
        gaussian_count=int(z['frame_xyz'].shape[1]);sh_degree=int(round(np.sqrt(z['sh'].shape[1]))-1)
    provenance={k:sha(paths[k]) for k in ['native_metadata','frame_mapping','asset_metadata','production_config']}
    record=dict(id=sid,scene=row['scene'],identity_key=row['identity_key'],dataset='nuscenes',paths=paths,
                provenance_hashes=provenance,reference_frame=frame,t_ref_us=frame['frame_timestamp_micros'],
                reference_seconds_from_first_native=(frame['frame_timestamp_micros']-native['frames'][0]['frame_timestamp_micros'])/1e6,
                input_budget=dict(real_native_exposures=len(native['frames']),model_input_slots=len(mapping['frames']),
                                  interpolated_slots=sum(bool(f.get('interpolated')) for f in mapping['frames']),
                                  native_duration_s=(native['frames'][-1]['frame_timestamp_micros']-native['frames'][0]['frame_timestamp_micros'])/1e6,
                                  model_input_duration_s=mapping['frames'][-1]['video_seconds']),
                runtime_numeric_decode_valid=True,runtime_sha256=sha(paths['runtime']),gaussians=gaussian_count,sh_degree=sh_degree,
                previous_manual_decision=decisions[sid],split=scene,reference_is_train_keyframe=True,
                old_selection_sha256=sha(selection),selection_basis='Only unselected, unexcluded existing inventory person in unchanged100 scenes; source validity reviewed, no external-model scores used',
                source_visual_review='Target red shirt and white diagonal strap identifiable; nearby people outside target contour, low resolution retained and disclosed',
                metric_eligible=False,reload_valid=False)
    (out/'candidate_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    source=Path(_hmc_expand('${HMC_THIRD_PARTY}/Grounded-SAM-2'));sys.path.insert(0,str(source))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    weight=Path(_hmc_expand('${HMC_MODEL_CACHE}/grounded_sam2/sam2.1_hiera_base_plus.pt'))
    assert sha(weight)=='a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml',str(weight),device='cuda');pred=SAM2ImagePredictor(model)
    image=np.asarray(Image.open(frame['raw_source_path']).convert('RGB'));crop=frame['crop_xyxy']
    box=np.array(frame['bbox_input_xyxy'])+np.array([crop[0],crop[1],crop[0],crop[1]])
    with torch.inference_mode():
        pred.set_image(image);masks,scores,_=pred.predict(box=box.astype(np.float32),multimask_output=False)
    folder=H/'references/replacement_v3'/sid;folder.mkdir(parents=True,exist_ok=False)
    mask=masks[0].astype(np.uint8)*255;Image.fromarray(mask).save(folder/'mask.png')
    white=np.where((mask>127)[...,None],image,255).astype(np.uint8)
    Image.fromarray(white).crop(crop).resize((308,560)).save(out/'exact_mask_review.png')
    meta=dict(id=sid,identity_key=row['identity_key'],scene=row['scene'],timestamp_us=record['t_ref_us'],
              source_path=frame['raw_source_path'],source_sha256=frame['raw_source_sha256'],
              mask_path=str(folder/'mask.png'),mask_sha256=sha(folder/'mask.png'),
              mask_origin='SAM2.1 base-plus exact-exposure box candidate',weight_sha256=sha(weight),
              prompt_raw_xyxy=box.tolist(),sam_score=float(scores[0]),preparation_script_sha256=sha(__file__),
              visual_review='pending',reference_ready=False,metric_eligible=False)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(candidate=sid,gaussians=gaussian_count,sh_degree=sh_degree,mask=str(folder/'mask.png'))))

if __name__=='__main__':main()
