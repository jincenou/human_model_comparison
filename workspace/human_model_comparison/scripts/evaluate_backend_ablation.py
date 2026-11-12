"""Evaluate fixed-cache backend variants in the existing physical source camera.

Export uses the production covariance and full-SH conversion. A replay of the
existing frozen production asset checks camera and renderer equivalence.
No reference-driven fitting or per-variant framing is performed.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common_human_orbit import H, read, save, sha, checked
from launch_backend_ablation import CODE, check_inputs

EVAL_PYTHON = Path(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'))
PRODUCTION = H.parent/'asset_work/human_nuscenes100_20261026'


def layout(sid):
    root = H/'runs/backend_ablation_v1'/sid
    out = H/'reports/backend_ablation_v1'/sid
    variants = [
        ('native_33k', root/'outputs'/sid/'native', 33000, 32999),
        ('complete_A_53k', root/'outputs'/sid/'asset', 53000, 52999),
        ('rgb_continuation_53k', root/'rgb_continuation_53k/outputs'/sid/'asset', 53000, 52999),
    ]
    return root, out, variants


def context(sid):
    root, out, variants = layout(sid)
    protocol = check_inputs(root, sid)
    reg_path = H/'configs/human_source_common_v1/ours_registration_batch_v3'/(sid+'.json')
    reg = read(reg_path)
    assert reg['timestamp_us'] == protocol['timestamp_us']
    assert reg['active_selection_sha256'] == protocol['selection_sha256']
    checked(reg['camera_contract'], reg['camera_contract_sha256'])
    assert reg['camera_contract_sha256'] == protocol['reference']['camera_contract_sha256']
    # This pilot's exact real exposure falls on one native frame. Fractional
    # exposures need the established SPD interpolation, not a new approximation.
    frame = round(reg['source_time_seconds']*25)
    assert abs(frame/25-reg['source_time_seconds']) < 1e-9
    assert reg['fraction'] == 0 and frame == reg['bracket'][0]
    return root, out, variants, protocol, reg_path, reg, frame


def export(sid):
    import numpy as np
    root, out, variants, protocol, reg_path, reg, frame = context(sid)
    sys.path.insert(0, str(CODE))
    import torch
    from assets import load_asset, transform_gaussians, rotate_sh
    from utils.general_utils import quaternion_to_matrix
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    for name, run, nominal, actual in variants:
        folder = out/name
        folder.mkdir(exist_ok=False)
        complete = read(run/'complete.json')
        assert complete['actual_updates'] == actual
        sources = [run/'config.json', run/'asset_camera_data.json',
                   run/'snapshots'/f'{nominal}.json',
                   run/'point_cloud'/f'iteration_{nominal}'/'point_cloud.ply',
                   run/'deform'/f'iteration_{nominal}'/'deform.pth']
        source_hashes = {str(p): sha(p) for p in sources}
        with torch.inference_mode():
            model, deform, pipe, config = load_asset(run, nominal)
            data = read(run/'asset_camera_data.json')
            assert data['frame_count'] == protocol['frames'] == 61
            assert model.active_sh_degree == protocol['sh_degree'] == 3
            view0 = next(c for c in data['cameras'] if c['view'] == 0)
            transform = np.linalg.inv(np.asarray(view0['camtoworld'], np.float64))
            count = len(model.get_xyz)
            t = model.get_xyz.new_tensor([[frame/(data['frame_count']-1)]]).expand(count, -1)
            warp = deform.step(model.get_xyz.detach(), t)
            xyz = (quaternion_to_matrix(warp['d_xyz'][:, :4]) @ model.get_xyz[..., None]).squeeze(-1) + warp['d_xyz'][:, 4:]
            raw_q = model.get_rotation + warp['d_rotation']
            raw_s = model.get_scaling + warp['d_scaling']
            means, quats, scales, covariance_check = transform_gaussians(
                xyz.cpu().numpy(), raw_q.cpu().numpy(), raw_s.cpu().numpy(), transform)
            sh, sh_error = rotate_sh(model.get_features.cpu().numpy(), transform[:3, :3], 3)
            arrays = dict(means=means, quaternions_wxyz=quats, scales=scales,
                          opacities=model.get_opacity.cpu().numpy().reshape(-1), sh=sh,
                          local_from_appearance=np.eye(3), sh_degree=np.array(3))
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(folder/'frozen.npz', **arrays)
        save(folder/'export.json', dict(id=sid, variant=name, checkpoint=str(run),
            actual_updates=actual, nominal_checkpoint_iteration=nominal,
            source_hashes=source_hashes, frame=frame, source_time_seconds=reg['source_time_seconds'],
            timestamp_us=reg['timestamp_us'], normalized_model_time=frame/(data['frame_count']-1),
            gaussian_count=count, sh_degree=3, asset_from_native=transform.tolist(),
            covariance_diagnostic=covariance_check, sh_rotation_absolute_error=sh_error,
            snapshot_sha256=sha(folder/'frozen.npz'), registration=str(reg_path),
            registration_sha256=sha(reg_path), protocol_sha256=sha(root/'protocol.json'),
            export_code={str(CODE/'assets.py'): sha(CODE/'assets.py'), str(Path(__file__)): sha(__file__)},
            regenerated_frontend=False, per_variant_alignment=False))
        print('exported', name, count, flush=True)
        del model, deform, warp, xyz, raw_q, raw_s, arrays
        torch.cuda.empty_cache()


def render_score(sid):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import human_source_metrics_v1 as metric
    from render_ours_source_batch_v3 import render, GaussianAssetFrame
    root, out, variants, protocol, reg_path, reg, frame = context(sid)
    core_path = H/'configs/human_source_common_v1/core_metric_protocol_frozen_v1.json'
    core = read(core_path)
    checked(metric.__file__, core['code_sha256'])
    ref = protocol['reference']; camera = read(reg['camera_contract'])
    for key in ['reference_rgb', 'reference_mask']:
        checked(ref[key], ref[key+'_sha256'])
    raw = np.asarray(Image.open(ref['reference_rgb']).convert('RGB'))
    mask = np.asarray(Image.open(ref['reference_mask']).convert('L'))
    reference = metric.prepare_reference(raw, mask)
    assert reference['status'] == 'ready'
    assert list(reference['roi']) == ref['roi_xyxy'] == camera['roi_xyxy']
    torch.set_num_threads(4)
    lp = core['lpips']
    lpips = metric.AlexLPIPS(lp['backbone'], lp['linear'], lp['backbone_sha256'], lp['linear_sha256'], 'cpu')
    Image.fromarray(reference['rgb']).save(out/'reference.png')
    summary = read(H/'reports/source_camera_metric_cohort_v3/summary.json')
    production = next(r for r in summary['rows'] if (r['id'], r['method']) == (sid, 'ours'))
    checked(production['result'], production['result_sha256'])
    production_result = read(production['result'])
    checked(production_result['registration_record'], production_result['registration_sha256'])
    production_certificate = read(production_result['registration_record'])
    assert production_certificate['camera_sha256'] == reg['camera_contract_sha256']
    assert production_certificate['timestamp_us'] == reg['timestamp_us']
    for key in ['source_rgb', 'source_alpha']:
        checked(production_certificate[key], production_certificate[key+'_sha256'])
    rows = []; images = [('Real reference', reference['rgb'])]
    for name, run, nominal, actual in variants + [('production_replay', None, None, 52999)]:
        folder = out/name; folder.mkdir(exist_ok=True)
        if name == 'production_replay':
            snapshot = Path(reg['snapshot'])
            checked(snapshot, reg['snapshot_sha256'])
        else:
            exported = read(folder/'export.json')
            snapshot = folder/'frozen.npz'
            checked(snapshot, exported['snapshot_sha256'])
            checked(reg_path, exported['registration_sha256'])
        if (folder/'metrics.json').exists():
            cached = read(folder/'metrics.json')
            assert cached['variant'] == name and cached['snapshot_sha256'] == sha(snapshot)
            checked(folder/'source_rgb.png', cached['rgb_sha256'])
            checked(folder/'source_alpha.png', cached['alpha_sha256'])
            pred = metric.prepare_prediction(np.asarray(Image.open(folder/'source_rgb.png')),
                np.asarray(Image.open(folder/'source_alpha.png')), reference['roi'], 'white_composited_uint8')
            rows.append(cached); images.append((name, pred['rgb']))
            print('reused verified score', name, flush=True)
            continue
        with np.load(snapshot, allow_pickle=False) as z:
            cloud = GaussianAssetFrame(z['means'], z['quaternions_wxyz'], z['scales'], z['opacities'], z['sh'], z['local_from_appearance'])
        with torch.inference_mode():
            rgb, alpha = render(cloud.transformed(np.asarray(reg['T_source_camera_from_export'])),
                                np.eye(4), np.asarray(camera['K_raw']), camera['render_wh'])
        rgb = np.floor(rgb*255+.5).astype(np.uint8)
        alpha = np.floor(alpha*255+.5).astype(np.uint8)
        Image.fromarray(rgb).save(folder/'source_rgb.png')
        Image.fromarray(alpha).save(folder/'source_alpha.png')
        pred = metric.prepare_prediction(rgb, alpha, reference['roi'], 'white_composited_uint8')
        Image.fromarray(pred['rgb']).save(folder/'roi_rgb.png')
        Image.fromarray(pred['mask']).save(folder/'roi_alpha.png')
        scores = dict(PSNR=metric.psnr(reference['rgb'], pred['rgb']),
            SSIM=metric.ssim(reference['rgb'], pred['rgb']), LPIPS=lpips(reference['rgb'], pred['rgb']),
            IoU=metric.iou(reference['mask'], pred['mask']), IoU_raw=metric.iou(mask, alpha))
        row = dict(id=sid, variant=name, actual_updates=actual, metrics=scores,
                   diagnostic_only=name=='production_replay', snapshot=str(snapshot), snapshot_sha256=sha(snapshot),
                   export_sha256=sha(folder/'export.json') if name!='production_replay' else None,
                   rgb_sha256=sha(folder/'source_rgb.png'), alpha_sha256=sha(folder/'source_alpha.png'))
        if name == 'production_replay':
            comparison = {}
            for filename, value in [('source_rgb.png', rgb), ('source_alpha.png', alpha)]:
                old_path = production_certificate[filename.removesuffix('.png')]
                previous = np.asarray(Image.open(old_path))
                error = np.abs(previous.astype(np.float64)-value.astype(np.float64))/255
                comparison[filename] = dict(max=float(error.max()), mean=float(error.mean()),
                    changed_pixels=int(np.count_nonzero(error)), old_sha256=sha(old_path))
            row['production_renderer_replay_diagnostic'] = comparison
            row['replay_scope'] = 'Existing frozen production cloud; verifies renderer and camera, not retired original checkpoint'
        save(folder/'metrics.json', row); rows.append(row); images.append((name, pred['rgb']))
        print('scored', name, scores, flush=True)
    contact = Image.new('RGB', (320*len(images), 354), 'white'); draw = ImageDraw.Draw(contact)
    for col, (label, array) in enumerate(images):
        draw.text((col*320+5, 8), label, fill='black')
        contact.paste(Image.fromarray(array).resize((320, 320)), (col*320, 32))
    contact.save(out/'comparison.jpg', quality=95)
    save(out/'results.json', dict(status='scored_pending_visual_review', id=sid, rows=rows,
        protocol=str(root/'protocol.json'), protocol_sha256=sha(root/'protocol.json'),
        metric_protocol=str(core_path), metric_protocol_sha256=sha(core_path),
        registration=str(reg_path), registration_sha256=sha(reg_path),
        reference=ref, common_roi=reference['roi'], same_source_camera_and_time=True,
        independent_holdout=False, scope='One-person fixed-frontend backend ablation pilot',
        code_sha256=sha(__file__), numerical_metric_code_modified=False))


def launch(sid, gpu, resume_score=False):
    root, out, variants = layout(sid)
    policy = read(H/'state/gpu_resource_policy.json')
    assert gpu in policy['allowed_gpu_indices'] and gpu not in policy['excluded_gpu_indices']
    out.mkdir(parents=True, exist_ok=True)
    tag = sha(__file__)[:12]
    statepath = out/(f'controller_score_{tag}.json' if resume_score else 'controller.json')
    if resume_score:
        previous = read(out/'controller.json')
        assert previous['status'] == 'failed_retained'
        assert not (out/'results.json').exists()
        for name, *_ in variants:
            exported = read(out/name/'export.json')
            checked(out/name/'frozen.npz', exported['snapshot_sha256'])
    assert not statepath.exists(), 'Inspect previous handle and retained outputs before any restart'
    with (H/'state'/f'gpu_auxiliary_slot_{gpu}.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        info = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        assert int(info[1]) >= 16384 and shutil.disk_usage(H).free >= 10*1024**3
        sys.path.insert(0, str(CODE)); os.environ['HUMAN_RUN_ROOT'] = str(root)
        import common
        from launch_ours_frozen_time_v2 import environment as eval_environment
        settings = [('export', common.PYTHONS['grid4d'], common.environment(gpu, 'grid4d')),
                    ('score', EVAL_PYTHON, eval_environment(info[0].strip()))]
        if resume_score:
            settings = settings[1:]
        state = dict(status='running', id=sid, gpu=gpu, pid=os.getpid(), stages=[], started_at=time.time())
        save(statepath, state)
        for action, python, env in settings:
            env['HUMAN_RUN_ROOT'] = str(root)
            command = [str(python), '-B', '-u', str(Path(__file__).resolve()), '--human-id', sid, '--worker', action]
            entry = dict(action=action, status='running', command=command, started_at=time.time())
            state['stages'].append(entry)
            with (out/(action+('_'+tag if resume_score else '')+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=H, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),))
            entry['pid'] = child.pid; save(statepath, state); code = child.wait()
            entry.update(exit_code=code, status='finished' if code == 0 else 'failed_retained', finished_at=time.time())
            save(statepath, state)
            if code:
                state.update(status='failed_retained', finished_at=time.time()); save(statepath, state)
                raise RuntimeError(action+' failed; inspect retained outputs')
        state.update(status='scored_pending_visual_review', finished_at=time.time()); save(statepath, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-id', required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--worker', choices=['export', 'score'])
    parser.add_argument('--resume-score', action='store_true')
    args = parser.parse_args()
    if args.worker == 'export': export(args.human_id)
    elif args.worker == 'score': render_score(args.human_id)
    else: launch(args.human_id, args.gpu, args.resume_score)
