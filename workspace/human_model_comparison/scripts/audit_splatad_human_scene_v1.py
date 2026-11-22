"""Two-case native SplatAD scene/node/decoder audit; never changes metric inputs."""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
OUT = H / 'reports/splatad_human_scene_audit_v1'
sys.path[:0] = [str(R / 'adapters'), str(R / 'scripts')]
from train_splatad_reviewed_scene import MonitoredTrainer
from train_splatad_human_train_only_scene import MonitoredTrainer as TrainOnlyMonitoredTrainer
from nerfstudio.utils.eval_utils import eval_setup


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def image(tensor, path):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    assert np.isfinite(arr).all()
    arr = (arr.clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)
    return arr


class CaptureDecoder(torch.nn.Module):
    """Execute the unchanged native decoder equation and retain its coefficients."""
    def __init__(self, original):
        super().__init__()
        self.core = getattr(original, '_orig_mod', original)
        self.last = None

    def forward(self, features, ray_dirs):
        features = features.view(1, *features.shape[-3:])
        albedo, spec = features.split([self.core.skip_dim,
                                      features.shape[-1] - self.core.skip_dim], dim=-1)
        spec = self.core.net(torch.cat([spec, ray_dirs], dim=-1).permute(0, 3, 1, 2))
        spec = spec.permute(0, 2, 3, 1)
        self.last = (albedo.detach().cpu()[0].clone(), spec.detach().cpu()[0].clone())
        return albedo * (1 + spec[..., :3]) + spec[..., 3:]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--id', required=True)
    args = parser.parse_args(); sid = args.id
    torch.set_num_threads(2)
    reg = read(H / 'reports/scene_source_registration_v1' / sid / 'splatad.json')
    prior = read(reg['render_validation'])
    item = next(x for x in read(H / 'inputs/scene_extract_original75_v1/splatad_plan.json')['objects']
                if x['id'] == sid)
    contract = read(reg['camera_contract'])
    assert sha(item['checkpoint']) == prior['checkpoint_sha256']
    assert sha(item['bundle']) == prior['bundle_sha256']
    out = OUT / sid
    assert not (out / 'audit.json').exists(), 'Completed audit must not be overwritten'
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config, pipeline, checkpoint, step = eval_setup(Path(item['config']), test_mode='val')
    assert step == 30000 and str(checkpoint) == item['checkpoint']
    model, manager = pipeline.model, pipeline.datamanager
    source = Path(item['source_image']).resolve()
    matches = [i for i, p in enumerate(manager.train_dataset.image_filenames) if Path(p).resolve() == source]
    assert len(matches) == 1
    camera = manager.train_dataset.cameras[matches[0]:matches[0]+1].to(model.device)
    parsed = manager.train_dataparser_outputs
    assert float(parsed.dataparser_scale) == 1.0
    model_from_world = np.eye(4)
    model_from_world[:3] = parsed.dataparser_transform.detach().cpu().numpy()[:3]
    cv_gl = np.diag([1., -1., -1., 1.])
    pose = model_from_world @ np.asarray(contract['T_world_from_camera_opencv']) @ cv_gl
    pose_error = float(np.abs(camera.camera_to_worlds[0].cpu().numpy() - pose[:3]).max())
    assert pose_error < 1e-3
    camera.camera_to_worlds[0] = torch.tensor(pose[:3], device=model.device, dtype=camera.camera_to_worlds.dtype)
    width, height = contract['render_wh']
    camera.width.fill_(width); camera.height.fill_(height)
    k = np.asarray(contract['K_raw'])
    for name, value in [('fx', k[0, 0]), ('fy', k[1, 1]), ('cx', k[0, 2]), ('cy', k[1, 2])]:
        getattr(camera, name).fill_(float(value))
    assert abs(float(camera.times.item()) - (contract['timestamp_us']/1e6 - float(parsed.time_offset))) < 1e-4
    model.config.use_camopt_in_eval = False
    model.config.compensate_rs_camera = False
    for key in ('lin_vel_x', 'lin_vel_y', 'lin_vel_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'rs_time'):
        model.rs_editing[key] = 0.
    uuid = item['native_id'].split(':')[-1]
    actor_index = next(i for i, actor in enumerate(parsed.metadata['trajectories']) if actor['uuid'] == uuid)
    original_parameters = model.gauss_params
    target = model.id.detach().reshape(-1) == actor_index
    bundle = torch.load(item['bundle'], map_location='cpu', weights_only=False)
    assert int(target.sum()) == item['gaussians']
    for name, parameter in original_parameters.items():
        assert torch.equal(parameter.detach()[target].cpu(), bundle['gauss_params'][name])
    original_decoder = model.rgb_decoder
    captured = CaptureDecoder(original_decoder)
    original_background = model.config.background_color
    pipeline.eval(); outputs, decoder_parts, arrays = {}, {}, {}
    # Native compiled render is replayed before instrumentation to validate the exact old evidence.
    with torch.no_grad():
        baseline = model.get_outputs_for_camera(copy.deepcopy(camera))
        baseline_array = image(baseline['rgb'], out / 'fullscene_native_decoder.png')
        del baseline
    model.rgb_decoder = captured
    try:
        with torch.no_grad():
            for name, selection in [('fullscene', None), ('without_target', ~target), ('target_only', target)]:
                model.gauss_params = original_parameters if selection is None else torch.nn.ParameterDict({
                    key: torch.nn.Parameter(value.detach()[selection].clone(), requires_grad=False)
                    for key, value in original_parameters.items()})
                model.config.background_color = 'white' if name == 'target_only' else original_background
                result = model.get_outputs_for_camera(copy.deepcopy(camera))
                arrays[name] = image(result['rgb'], out / (name + '.png'))
                image(result['accumulation'], out / (name + '_alpha.png'))
                outputs[name] = {'rgb': result['rgb'].detach().cpu(), 'alpha': result['accumulation'].detach().cpu()}
                decoder_parts[name] = captured.last
                del result
                print(sid, name, 'rendered', flush=True)
    finally:
        model.gauss_params = original_parameters
        model.rgb_decoder = original_decoder
        model.config.background_color = original_background
    replay = {}
    for name, old_path in [('fullscene', prior['full_scene_preview']), ('target_only', prior['actor_preview'])]:
        old = np.asarray(Image.open(old_path).convert('RGB'))
        delta = np.abs(arrays[name].astype(int) - old.astype(int))
        replay[name] = {'max_byte_difference': int(delta.max()), 'mean_byte_difference': float(delta.mean())}
    decoder_delta = np.abs(arrays['fullscene'].astype(int) - baseline_array.astype(int))
    replay['captured_vs_native_decoder'] = {'max_byte_difference': int(decoder_delta.max()),
                                           'mean_byte_difference': float(decoder_delta.mean())}
    assert replay['captured_vs_native_decoder']['max_byte_difference'] <= 1, replay
    assert replay['fullscene']['max_byte_difference'] <= 1 and replay['target_only']['max_byte_difference'] <= 1, replay

    # An explicitly diagnostic intervention: retain target albedo and alpha but swap only CNN coefficients.
    # It is not an independent asset, exported model, correction, or metric input.
    target_albedo, target_coeff = decoder_parts['target_only']
    full_coeff = decoder_parts['fullscene'][1]
    alpha = outputs['target_only']['alpha']
    swapped = (target_albedo * (1 + full_coeff[..., :3]) + full_coeff[..., 3:] + (1-alpha)).clamp(0, 1)
    image(swapped, out / 'target_fullscene_cnn_coeff_diagnostic.png')
    image((target_albedo + 1 - alpha).clamp(0, 1), out / 'target_albedo_white_diagnostic.png')
    reference = next(r['reference'] for r in read(H / 'reports/static_reference_revision_v2/core/summary.json')['rows']
                     if r['id'] == sid and r['method'] == 'splatad')
    mask = torch.from_numpy(np.asarray(Image.open(reference['reference_mask']).convert('L')).copy()) > 127
    aa = alpha.squeeze(-1)
    foreground_difference = (outputs['fullscene']['rgb'] - outputs['without_target']['rgb']).abs().mean(-1)
    context_difference = (swapped - outputs['target_only']['rgb']).abs().mean(-1)
    region = mask & (aa > .5)
    diagnostic = dict(gt_mask_pixels=int(mask.sum()),
        target_alpha_mean_in_gt=float(aa[mask].mean()),
        target_alpha_gt_0_5_fraction_in_gt=float((aa[mask] > .5).float().mean()),
        target_alpha_lt_0_1_fraction_in_gt=float((aa[mask] < .1).float().mean()),
        full_vs_without_target_abs_rgb_mean_in_gt=float(foreground_difference[mask].mean()),
        cnn_context_swap_abs_rgb_mean_in_gt=float(context_difference[mask].mean()),
        cnn_context_swap_abs_rgb_mean_in_gt_target_alpha_gt_0_5=float(context_difference[region].mean()) if region.any() else None,
        diagnostic_only=True, no_paper_metric_recomputed=True)
    record = dict(id=sid, status='passed', scene=prior['scene'], actor_index=actor_index,
        source_rgb=str(source), checkpoint=str(checkpoint), checkpoint_sha256=prior['checkpoint_sha256'],
        bundle=item['bundle'], bundle_sha256=prior['bundle_sha256'], source_camera_index=matches[0],
        source_pose_max_error=pose_error, source_timestamp_us=contract['timestamp_us'], reference=reference,
        target_gaussians=int(target.sum()), scene_gaussians=int(len(target)),
        selected_by='all native actor-id Gaussians; no bbox, centre, alpha or GT-mask filtering',
        original_parameters_identical_to_export=True, trained=False, gaussian_parameters_modified=False,
        images={name: str(out / (name+'.png')) for name in outputs},
        diagnostic_images={name: str(out/(name+'.png')) for name in ['target_fullscene_cnn_coeff_diagnostic','target_albedo_white_diagnostic']},
        replay=replay, diagnostics=diagnostic,
        decoder='Native RGBDecoderCNN equation: albedo*(1+CNN[:3])+CNN[3:]; RGB then adds (1-alpha)*background. Native coefficients depend on rendered local features, appearance embedding and ray direction.',
        context_swap_scope='Diagnostic only: full-scene CNN coefficients applied to unchanged target-only albedo/alpha; contains full-scene context and cannot serve as independent-asset output or official score.',
        source_policy='Unchanged source-camera protocol: camopt and rolling-shutter compensation disabled; source timestamp retained.',
        script_sha256=sha(__file__), elapsed_seconds=time.time()-started)
    (out/'audit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'id': sid, 'status': 'passed', 'replay': replay, 'diagnostics': diagnostic}), flush=True)


if __name__ == '__main__':
    main()
