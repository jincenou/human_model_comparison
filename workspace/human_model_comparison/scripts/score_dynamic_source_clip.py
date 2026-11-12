"""Paired real-exposure clip metrics; the static metric implementation is reused."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import human_source_metrics_v1 as core

H = Path(__file__).resolve().parents[1]
FLOW_ARGS = (0.5, 3, 15, 3, 5, 1.2, 0)
METHODS = ('ours', 'omnire', 'splatad')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def verify(path, digest):
    if core.sha(path) != digest:
        raise ValueError('Input hash mismatch: ' + str(path))


def check_times(frames):
    times = [f['timestamp_us'] for f in frames]
    if len(times) < 2 or any(not isinstance(t, int) for t in times):
        raise ValueError('At least two integer real exposure timestamps required')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Exposure timestamps must increase strictly')
    if [f['index'] for f in frames] != list(range(len(frames))):
        raise ValueError('Missing or reordered frame indexes')
    return [(b-a)/1e6 for a, b in zip(times, times[1:])]


def optical_flow(a, b):
    a, b = core.bytes_rgb(a), core.bytes_rgb(b)
    if a.shape != b.shape:
        raise ValueError('Flow inputs must use the same canvas')
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None, *FLOW_ARGS)


def flow_difference(a, b):
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError('Expected same-sized two-component flow fields')
    difference = a.astype(np.float64) - b.astype(np.float64)
    return core.score(float(np.sqrt(np.sum(difference*difference, axis=-1)).mean()))


def temporal_pair(reference_flow, prediction_flow, reference_lpips, prediction_lpips):
    return dict(tOF=flow_difference(reference_flow, prediction_flow),
                tLP100=core.score(100*abs(reference_lpips-prediction_lpips)))


def mean_metric(values):
    if not values or any(v['status'] not in ('ok', 'positive_infinity') for v in values):
        return core.na('missing_or_invalid_frame_scores')
    if any(v['status'] == 'positive_infinity' for v in values):
        return core.score(float('inf'))
    return core.score(float(np.mean([v['value'] for v in values], dtype=np.float64)))


def load_reference(folder):
    active = read(H/'inputs/active_human_benchmark.json')
    clip = read(folder/'clip.json')
    qualification = read(folder/'reference_qualification.json')
    verify(active['selection'], active['selection_sha256'])
    identities = {r['id']: r['identity_key'] for r in read(active['selection'])['objects']}
    assert clip['selection_sha256'] == active['selection_sha256']
    assert identities[clip['id']] == clip['identity_key']
    verify(folder/'clip.json', qualification['clip_sha256'])
    assert qualification['id'] == clip['id'] and qualification['reference_eligible'] is True
    assert qualification['all_real_exposures'] and qualification['all_frames_visually_reviewed']
    assert qualification['generated_reference_frames'] == 0
    intervals = check_times(clip['frames'])
    assert len(qualification['frames']) == len(clip['frames'])
    rgbs, masks = [], []
    for f, q in zip(clip['frames'], qualification['frames']):
        assert (f['index'], f['timestamp_us']) == (q['index'], q['timestamp_us'])
        assert f['source_rgb_sha256'] == q['source_rgb_sha256']
        verify(f['source_rgb'], f['source_rgb_sha256']); verify(q['mask'], q['mask_sha256'])
        rgb = np.asarray(Image.open(f['source_rgb']).convert('RGB'))
        mask = np.asarray(Image.open(q['mask']).convert('L'))
        assert list(rgb.shape[1::-1]) == f['render_wh'] and rgb.shape[:2] == mask.shape
        if not np.any(mask > 127):
            raise ValueError('Empty reference mask at frame ' + str(f['index']))
        rgbs.append(rgb); masks.append(mask)
    union = np.maximum.reduce(masks)
    roi = core.reference_roi(union)
    prepared = [dict(rgb=core.crop(np.where((mask > 127)[..., None], rgb, 255).astype(np.uint8), roi),
                     mask=core.crop(mask, roi, True), raw_mask=mask)
                for rgb, mask in zip(rgbs, masks)]
    return clip, qualification, prepared, roi, intervals


def load_prediction(folder, method, clip, roi):
    manifest = folder/method/'validation.json'
    validation = read(manifest)
    assert validation['id'] == clip['id'] and validation['method'] == method
    verify(folder/'clip.json', validation['clip_sha256'])
    assert validation['prediction_encoding'] == 'white_composited_uint8'
    # A later script edit does not invalidate renders made by the archived source.
    candidates = [H/'scripts/prepare_dynamic_source_clip.py', H/'scripts/render_dynamic_scene_clip.py']
    candidates += list((folder/'source_archive').glob('*.py'))
    source = next((p for p in candidates if core.sha(p) == validation['script_sha256']), None)
    if source is None:
        raise ValueError('Exact render source is unavailable')
    assert len(validation['frames']) == len(clip['frames'])
    prepared = []
    for f, r in zip(clip['frames'], validation['frames']):
        assert (f['index'], f['timestamp_us']) == (r['index'], r['timestamp_us'])
        verify(r['rgb'], r['rgb_sha256']); verify(r['alpha'], r['alpha_sha256'])
        rgb = np.asarray(Image.open(r['rgb']).convert('RGB'))
        alpha = np.asarray(Image.open(r['alpha']).convert('L'))
        assert list(rgb.shape[1::-1]) == f['render_wh'] and rgb.shape[:2] == alpha.shape
        if np.any(alpha == 0) and np.max(255-rgb[alpha == 0]) > 2:
            raise ValueError('Nonwhite background in own-alpha render')
        prepared.append(core.prepare_prediction(rgb, alpha, roi, validation['prediction_encoding']))
    return prepared, dict(validation=str(manifest), validation_sha256=core.sha(manifest),
                          render_source=str(source), render_source_sha256=core.sha(source))


def score_sequence(reference, prediction, clip, metric, reference_temporal, raw_iou=True):
    assert len(reference) == len(prediction) == len(clip['frames'])
    frames, pairs = [], []
    for i, (g, p) in enumerate(zip(reference, prediction)):
        values = dict(PSNR=core.psnr(g['rgb'], p['rgb']), SSIM=core.ssim(g['rgb'], p['rgb']),
                      LPIPS=metric(g['rgb'], p['rgb']), IoU=core.iou(g['mask'], p['mask']))
        if raw_iou:
            values['IoU_raw'] = core.iou(g['raw_mask'], p['raw_mask'])
        frames.append(dict(index=i, timestamp_us=clip['frames'][i]['timestamp_us'], metrics=values))
        if i:
            pf = optical_flow(prediction[i-1]['rgb'], p['rgb'])
            pl = metric(prediction[i-1]['rgb'], p['rgb'])['value']
            gf, gl = reference_temporal[i-1]
            pairs.append(dict(previous_index=i-1, index=i,
                              timestamp_us=clip['frames'][i]['timestamp_us'],
                              delta_seconds=(clip['frames'][i]['timestamp_us']-clip['frames'][i-1]['timestamp_us'])/1e6,
                              reference_adjacent_lpips=gl, prediction_adjacent_lpips=pl,
                              metrics=temporal_pair(gf, pf, gl, pl)))
    summary = {k: mean_metric([r['metrics'][k] for r in frames]) for k in frames[0]['metrics']}
    summary.update({k: mean_metric([r['metrics'][k] for r in pairs]) for k in ('tOF', 'tLP100')})
    return dict(status='scored', frame_count=len(frames), pair_count=len(pairs),
                summary=summary, frames=frames, pairs=pairs)


def control_sequences(reference):
    yield 'identical', reference
    yield 'hold_first_frame', [reference[0]]*len(reference)
    yield 'reverse_time', list(reversed(reference))
    flicker, blur = [], []
    for i, frame in enumerate(reference):
        rgb = frame['rgb'].copy()
        if i % 2:
            foreground = frame['mask'] > 127
            rgb[foreground] = np.floor(rgb[foreground].astype(np.float64)*0.55+0.5).astype(np.uint8)
        flicker.append(dict(frame, rgb=rgb))
        blur.append(dict(frame, rgb=cv2.GaussianBlur(frame['rgb'], (15, 15), 3)))
    yield 'alternating_foreground_darkening', flicker
    yield 'gaussian_blur', blur


def run(args):
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(4); cv2.ocl.setUseOpenCL(False)
    folder = H/'reports/dynamic_source_clip_v1'/args.human_id
    out = folder/'metrics'; out.mkdir(exist_ok=True)
    with (out/'score.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path = H/'configs/human_source_common_v1/dynamic_metric_protocol_v1.json'
        protocol = read(protocol_path)
        verify(__file__, protocol['implementation_sha256'])
        assert protocol['farneback']['args'] == list(FLOW_ARGS)
        assert protocol['opencv_version'] == cv2.__version__
        static = H/'configs/human_source_common_v1/core_metric_protocol_frozen_v1.json'
        verify(static, protocol['static_protocol_sha256'])
        cfg = read(static); verify(core.__file__, cfg['code_sha256'])
        clip, qualification, reference, roi, intervals = load_reference(folder)
        signature = dict(clip_sha256=core.sha(folder/'clip.json'),
                         qualification_sha256=core.sha(folder/'reference_qualification.json'),
                         protocol_sha256=core.sha(protocol_path), script_sha256=core.sha(__file__),
                         method_manifests={m:core.sha(folder/m/'validation.json') for m in METHODS},
                         controls=args.controls)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        destination = out/'metrics.json'
        if destination.exists():
            existing = read(destination)
            if existing['input_key'] != key:
                raise ValueError('Existing clip score has different inputs; preserve before revising')
            # Revalidate all rendered inputs before returning a cached result.
            for method in METHODS: load_prediction(folder, method, clip, roi)
            print('Reused exact clip result:', destination, flush=True); return
        lp = cfg['lpips']
        metric = core.AlexLPIPS(lp['backbone'], lp['linear'], lp['backbone_sha256'], lp['linear_sha256'], 'cpu')
        temporal = [(optical_flow(a['rgb'], b['rgb']), metric(a['rgb'], b['rgb'])['value'])
                    for a, b in zip(reference, reference[1:])]
        results, errors = {}, {}
        for method in METHODS:
            try:
                prediction, provenance = load_prediction(folder, method, clip, roi)
                result = score_sequence(reference, prediction, clip, metric, temporal)
                result['provenance'] = provenance; results[method] = result
                print('scored', method, result['summary'], flush=True)
            except (AssertionError, ValueError, FileNotFoundError, KeyError) as error:
                errors[method] = dict(status='na', reason=type(error).__name__ + ': ' + str(error))
                print('method failed', method, errors[method], flush=True)
        controls = {}
        if args.controls:
            for name, sequence in control_sequences(reference):
                controls[name] = score_sequence(reference, sequence, clip, metric, temporal, raw_iou=False)
                print('control', name, controls[name]['summary'], flush=True)
            identical = controls['identical']['summary']
            assert identical['PSNR']['status'] == 'positive_infinity'
            assert identical['IoU']['value'] == 1 and identical['tOF']['value'] == 0
            assert identical['tLP100']['value'] == 0 and abs(identical['LPIPS']['value']) < 1e-7
        result = dict(status='real_exposure_dynamic_source_pilot_scored', input_key=key, inputs=signature,
                      identity=clip['id'], identity_key=clip['identity_key'], selection_sha256=clip['selection_sha256'],
                      source_reconstruction_only=True, heldout_evaluation=False, fixed_roi_xyxy=roi,
                      resolution=[512,512], reference_frame_count=len(reference), reference_pair_count=len(intervals),
                      frame_intervals_seconds=intervals, frame_average='equal real exposures',
                      temporal_average='equal adjacent real-exposure pairs; no frame interpolation or retiming',
                      methods=results, failures=errors, controls=controls, lpips=metric.provenance,
                      common_identity_count=1 if not errors else 0, selected_denominator=100,
                      protocol=str(protocol_path), scope=protocol['scope'], limitations=protocol['limitations'])
        save(destination, result)
        with (out/'summary.csv').open('w', newline='') as stream:
            names=['PSNR','SSIM','LPIPS','IoU','IoU_raw','tOF','tLP100']
            writer=csv.DictWriter(stream, fieldnames=['id','method','frames','pairs']+names)
            writer.writeheader()
            for name, record in results.items():
                row=dict(id=clip['id'],method=name,frames=record['frame_count'],pairs=record['pair_count'])
                for n in names:
                    v=record['summary'][n];row[n]='inf' if v['status']=='positive_infinity' else v['value'] if v['status']=='ok' else 'NA'
                writer.writerow(row)
        print('Saved', destination, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--human-id', default='HN0004')
    parser.add_argument('--controls', action='store_true')
    run(parser.parse_args())
