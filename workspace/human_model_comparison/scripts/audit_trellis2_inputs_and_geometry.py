"""Read-only cohort audit of input provenance, official preprocessing and geometry."""
import ast
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from common_human_orbit import H, read, save, sha
from human_source_metrics_v1 import crop

P = H.parent / 'vehicle_model_comparison/model_preparation/20261109_2100_CST'
S = P / 'sources/trellis2-75fbf0183001ed9876c8dbb35de6b68552ee08bd'
OUT = H / 'reports/trellis2_failure_root_cause_v1'


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    (OUT / 'preprocessed').mkdir(exist_ok=True)
    source = S / 'trellis2/pipelines/trellis2_image_to_3d.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trellis2ImageTo3DPipeline')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'preprocess_image')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = dict(Image=Image, np=np)
    exec(compile(module, str(source), 'exec'), ns)
    # Every audited input must use the alpha branch. A call to a rembg model
    # would fail here, instead of silently changing the original input.
    preprocessor = SimpleNamespace(low_vram=False, rembg_model=None)
    manifests = [H / 'inputs/human_static_native_v1/manifest.json', H / 'inputs/human_static_replacement_v3/manifest.json']
    objects = {r['sample_id']: r for p in manifests for r in read(p)['objects']}
    active = read(H / 'inputs/active_human_benchmark.json')
    selected = read(active['selection'])['objects']
    summary = read(H / 'reports/source_camera_metric_cohort_v3/summary.json')
    scored = {(r['id'], r['method']): r for r in summary['rows']}
    result = dict(created_at=time.time(), selection_sha256=sha(active['selection']),
                  source_file=str(source), source_sha256=sha(source), script_sha256=sha(__file__),
                  official_preprocessing_body_unchanged=True, original_assets_modified=False,
                  pipeline_config=str(P / 'weights/microsoft--TRELLIS.2-4B/pipeline.json'), rows=[])
    result['pipeline_config_sha256'] = sha(result['pipeline_config'])
    for sel in selected:
        sid = sel['id']; entry = objects[sid]
        folder = H / 'runs/trellis2_human_native_v1' / sid
        state = read(folder / 'state.json')
        original = read(H / 'runs/trellis_original_human_native_v1' / sid / 'state.json')
        camera = read(entry['camera_candidate'])
        im = Image.open(entry['image_rgba']); rgba = np.array(im)
        row = dict(id=sid, source_status=scored[(sid, 'trellis2')]['status'],
                   native_status=state['status'], input_path=entry['image_rgba'],
                   input_sha256=sha(entry['image_rgba']), input_mode=im.mode, input_wh=list(im.size),
                   input_equals_manifest=sha(entry['image_rgba']) == entry['image_rgba_sha256'],
                   input_equals_native_record=sha(entry['image_rgba']) == state['input_sha256'],
                   same_input_as_original_trellis=state['input_sha256'] == original['input_sha256'],
                   identity_matches=entry['identity_key'] == sel['identity_key'],
                   time_matches=entry['timestamp_us'] == sel['t_ref_us'],
                   parameters={k:state.get(k) for k in ['seed','source_commit','pipeline_type','low_vram','max_num_tokens','real_input_count','generated_input_count']},
                   gpu=state.get('gpu'), raw_reference_height=camera['reference_pixel_height'])
        assert im.mode == 'RGBA' and im.size == (512, 512)
        alpha = rgba[...,3]; binary = alpha > 127
        row.update(alpha_range=[int(alpha.min()),int(alpha.max())], alpha_values=np.unique(alpha).tolist(),
                   alpha_foreground_fraction=float(binary.mean()), uses_existing_alpha=not bool(np.all(alpha == 255)))
        assert row['uses_existing_alpha'] and np.any(alpha > .8*255)
        old_rgb = np.array(Image.open(camera['roi_reference_rgb']).convert('RGB'))
        old_mask = np.array(Image.open(camera['roi_reference_mask']).convert('L'))
        row['matches_frozen_candidate_rgb_and_mask'] = bool(np.array_equal(rgba[...,:3],old_rgb) and np.array_equal(alpha,old_mask))
        ref = scored[(sid,'trellis2')]['reference']
        final_mask = np.array(Image.open(ref['reference_mask']).convert('L'))
        now = crop(final_mask, camera['roi_xyxy'], True) > 127
        row['generation_vs_final_mask_same_roi'] = dict(iou=float(np.count_nonzero(binary & now)/max(1,np.count_nonzero(binary | now))),
            generation_foreground_outside_final_pixels=int(np.count_nonzero(binary & ~now)),
            final_foreground_missing_from_generation_pixels=int(np.count_nonzero(now & ~binary)),
            raw_mask_sha_equal=entry['exact_mask_sha256'] == ref['reference_mask_sha256'])
        processed = ns['preprocess_image'](preprocessor, im)
        path = OUT / 'preprocessed' / (sid+'.png'); processed.save(path)
        row['preprocessed'] = dict(path=str(path),sha256=sha(path),wh=list(processed.size),mode=processed.mode)
        payload = folder / 'native_voxel_mesh.pt'
        if payload.exists():
            packed = torch.load(payload,map_location='cpu',weights_only=True)
            xyz = packed['vertices'].double().numpy()
            center = xyz.mean(0); covariance = (xyz.T@xyz)/len(xyz)-np.outer(center,center)
            eigenvalues = np.maximum(np.linalg.eigvalsh(covariance),0)
            span = np.ptp(xyz,axis=0)
            opacity = packed['attrs'][:,packed['layout']['alpha'][0]].float().numpy()
            row['geometry'] = dict(vertices=len(xyz),faces=len(packed['faces']),
                checkpoint_sha256=sha(payload), recorded_checkpoint_sha256=state.get('output_hashes',{}).get(payload.name),
                aabb_extent=span.tolist(),min_to_max_axis_extent=float(span.min()/max(span.max(),1e-12)),
                pca_std=np.sqrt(eigenvalues).tolist(),pca_min_to_max_std=float(np.sqrt(eigenvalues[0]/max(eigenvalues[-1],1e-20))),
                alpha_quantiles=np.quantile(opacity,[0,.1,.5,.9,1]).tolist(),alpha_voxel_fraction_gt_half=float((opacity>.5).mean()),
                output_before_glb_postprocess=True)
            del packed,xyz,opacity
        result['rows'].append(row)
        save(OUT/'audit.json',result)
        print(sid,row['source_status'],row.get('geometry',{}).get('pca_min_to_max_std'),flush=True)
    result['finished_at'] = time.time()
    result['groups'] = {}
    for group in sorted({r['source_status'] for r in result['rows']}):
        rows = [r for r in result['rows'] if r['source_status']==group]
        result['groups'][group] = dict(count=len(rows),
            inputs_same_as_original_trellis=sum(r['same_input_as_original_trellis']for r in rows),
            all_input_hashes_match=all(r['input_equals_manifest']and r['input_equals_native_record']for r in rows),
            existing_alpha_branch=sum(r['uses_existing_alpha']for r in rows),
            different_generation_vs_final_mask=sum(not r['generation_vs_final_mask_same_roi']['raw_mask_sha_equal']for r in rows),
            input_mask_iou_median=float(np.median([r['generation_vs_final_mask_same_roi']['iou']for r in rows])),
            raw_reference_height_median=float(np.median([r['raw_reference_height']for r in rows])),
            pca_ratio_median=float(np.median([r['geometry']['pca_min_to_max_std']for r in rows if 'geometry'in r]))if any('geometry'in r for r in rows)else None,
            near_planar_pca_ratio_lt_002=sum(r.get('geometry',{}).get('pca_min_to_max_std',1)<.02 for r in rows),
            parameter_variants=dict(Counter(json.dumps(r['parameters'],sort_keys=True)for r in rows)))
    save(OUT/'audit.json',result)
    print(json.dumps(result['groups'],indent=2))


if __name__ == '__main__':main()
