"""Export a finished native pilot, then reload snapshots in a separate process.

Run export followed by reload with the ExAvatar Python environment. These are
native-camera checks, not common-camera metric acceptance. Full learned runtime
is retained in the referenced checkpoint; snapshots capture one observed pose.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parents[1]
ROOT = P / 'reports/exavatar_real_sequence_HN0057_20261113'
SUBJECT = 'HN0057_native_pilot_20261113'
OUT = ROOT / 'native_asset_reload'
FIELDS = ('mean_3d', 'opacity', 'scale', 'rotation', 'rgb')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup():
    main = P / 'sources/exavatar/avatar/main'
    os.chdir(main)
    sys.path.insert(0, str(main))
    from config import cfg
    cfg.dataset = 'Custom'
    sys.path.insert(0, str(Path(cfg.data_dir) / 'Custom'))
    cfg.set_args(SUBJECT)
    return cfg


def save_image(path, array):
    from PIL import Image
    import numpy as np
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype('uint8')).save(path)


def export():
    import numpy as np
    import torch
    state = json.loads((P/'state/exavatar_avatar_HN0057_20261113.json').read_text())
    assert state['status'] == 'trained_reload_pending' and state['exit_code'] == 0
    cfg = setup()
    from base import Tester
    tester = Tester('4')
    tester._make_batch_generator()
    tester._make_model()
    model = tester.model.module
    checkpoint = Path(cfg.model_dir) / 'snapshot_4.pth'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['network'], strict=True)
    assert saved['epoch'] == 4
    del saved
    inputs = json.loads((ROOT/'inputs.json').read_text())
    frames = {int(f['native_index']): f for f in inputs['frames']}
    OUT.mkdir(exist_ok=False)
    records = []
    with torch.no_grad():
        for data in tester.batch_generator:
            idx = int(data['frame_idx'][0])
            cam = {k: v[0].cuda() for k, v in data['cam_param'].items()}
            shape = tuple(int(x) for x in data['img'].shape[-2:])
            pose = model.smplx_param_dict([data['frame_idx'][0]])[0]
            base, refined, _, _ = model.human_gaussian(pose, cam)
            native = tester.model(data, None, 'test')
            record = dict(frame=idx, timestamp_us=frames[idx]['frame_timestamp_micros'],
                          input_sha256=frames[idx]['sha256'], variants={})
            for name, asset, native_key in [('base', base, 'human_img'), ('refined', refined, 'human_img_refined')]:
                assert set(asset) == set(FIELDS)
                assert all(torch.isfinite(asset[k]).all() for k in FIELDS)
                result = model.gaussian_renderer(asset, shape, cam, torch.ones(3, device='cuda'))
                arrays = {k: asset[k].detach().cpu().numpy() for k in FIELDS}
                arrays.update({f'camera_{k}': v.cpu().numpy() for k, v in cam.items()})
                arrays.update(image_shape=np.array(shape), timestamp_us=np.int64(record['timestamp_us']))
                path = OUT/f'{idx:03d}_{name}.npz'
                np.savez_compressed(path, **arrays)
                img = result['img'].cpu().numpy()
                alpha = result['mask'].cpu().numpy()
                np.savez_compressed(OUT/f'{idx:03d}_{name}_render.npz', rgb=img, alpha=alpha)
                save_image(OUT/f'{idx:03d}_{name}.png', img)
                save_image(OUT/f'{idx:03d}_{name}_alpha.png', alpha)
                record['variants'][name] = dict(file=path.name, sha256=sha(path), points=len(arrays['mean_3d']),
                    native_render_max_abs=float((result['img']-native[native_key][0]).abs().max()))
            records.append(record)
            print('exported', idx, flush=True)
    assert len(records) == 30 and set(r['frame'] for r in records) == set(frames)
    manifest = dict(status='exported_reload_pending', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        inputs_sha256=sha(ROOT/'inputs.json'), frames=records,
        camera='native virtual camera; common-camera registration pending',
        appearance='Both native base and refined Gaussian branches; no face texture overlay baked in',
        runtime='Checkpoint retains pose-conditioned geometry/RGB decoders; per-frame NPZ is a posed snapshot',
        pixel_differences='Diagnostics only, no single-pixel numerical eligibility gate', common_camera_valid=False)
    (OUT/'export.json').write_text(json.dumps(manifest, indent=2))


def reload():
    import numpy as np
    import torch
    setup()
    from nets.module import GaussianRenderer
    renderer = GaussianRenderer()
    manifest = json.loads((OUT/'export.json').read_text())
    results = []
    with torch.no_grad():
        for record in manifest['frames']:
            for variant, info in record['variants'].items():
                path = OUT/info['file']
                assert sha(path) == info['sha256']
                with np.load(path, allow_pickle=False) as z:
                    asset = {k: torch.from_numpy(z[k]).cuda() for k in FIELDS}
                    cam = {k.removeprefix('camera_'): torch.from_numpy(z[k]).cuda() for k in z.files if k.startswith('camera_')}
                    shape = tuple(int(x) for x in z['image_shape'])
                    assert int(z['timestamp_us']) == record['timestamp_us']
                render = renderer(asset, shape, cam, torch.ones(3, device='cuda'))
                rgb, alpha = render['img'].cpu().numpy(), render['mask'].cpu().numpy()
                assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
                idx = record['frame']
                with np.load(OUT/f'{idx:03d}_{variant}_render.npz', allow_pickle=False) as before:
                    results.append(dict(frame=idx, variant=variant, rgb_max_abs=float(np.abs(rgb-before['rgb']).max()),
                        alpha_max_abs=float(np.abs(alpha-before['alpha']).max()), foreground_pixels=int((alpha>0.5).sum())))
                save_image(OUT/f'{idx:03d}_{variant}_reloaded.png', rgb)
                print('reloaded', idx, variant, flush=True)
    (OUT/'reload.json').write_text(json.dumps(dict(status='snapshot_reload_executed_review_pending',
        checkpoint_loaded=False, independent_process=True, common_camera_valid=False, frames=results), indent=2))


def orbit():
    import math
    import numpy as np
    import torch
    setup()
    from nets.module import GaussianRenderer
    renderer = GaussianRenderer()
    manifest = json.loads((OUT/'export.json').read_text())
    reference = [r for r in manifest['frames'] if r['timestamp_us'] == 1533201591527893]
    assert len(reference) == 1
    info = reference[0]['variants']['refined']
    path = OUT/info['file']
    assert sha(path) == info['sha256']
    with np.load(path, allow_pickle=False) as z:
        asset = {k: torch.from_numpy(z[k]).cuda() for k in FIELDS}
    xyz = asset['mean_3d']
    lower, upper = xyz.min(0).values, xyz.max(0).values
    center = (lower + upper) / 2
    # One fixed camera radius for every angle; no per-view recentering/resizing.
    radius = float((upper-lower).norm()/2)
    distance = radius * 1.15 / math.sin(math.atan(256/550))
    directory = OUT/'orbit'
    directory.mkdir(exist_ok=False)
    cameras = []
    with torch.no_grad():
        for index in range(16):
            angle = 2*math.pi*index/16
            c, s = math.cos(angle), math.sin(angle)
            rotation = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], device='cuda', dtype=torch.float32)
            t = torch.tensor([0,0,distance],device='cuda') - rotation@center
            cam = dict(R=rotation,t=t,focal=torch.tensor([550.,550.],device='cuda'),
                       princpt=torch.tensor([256.,256.],device='cuda'))
            result = renderer(asset,(512,512),cam,torch.ones(3,device='cuda'))
            save_image(directory/f'{index:02d}.png',result['img'].cpu().numpy())
            cameras.append(dict(angle_deg=index*22.5, **{k:v.cpu().tolist()for k,v in cam.items()}))
    (directory/'cameras.json').write_text(json.dumps(dict(frame=reference[0]['frame'],
        timestamp_us=reference[0]['timestamp_us'],center=center.cpu().tolist(),distance=distance,
        cameras=cameras,scope='Single-model demo orbit, not benchmark common-camera rendering',
        pose='Fixed original pose at the frozen reference exposure',
        rgb='Native refined snapshot appearance; full decoder retained in checkpoint'),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'reload', 'orbit'))
    args = parser.parse_args()
    {'export':export,'reload':reload,'orbit':orbit}[args.mode]()
