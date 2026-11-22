"""Resolve audited native support without selecting outputs by metric score.

All conservative candidates remain immutable. Actor-only views reference the
same saved Gaussian bundle and native decoder; no checkpoint copies are made.
"""
import argparse
import copy
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import native_reaudit_common as common

BASE = common.ROOT
ROOT = BASE / 'resolved'


def decisions(method):
    doc = common.read(ROOT / 'decisions' / (method + '.json'))
    rows = [r for r in doc['rows'] if r['method'] == method]
    assert {r['id'] for r in rows} == set(common.freeze()['ids'])
    return {r['id']: r for r in rows}


def validations(method):
    return {p.parent.name: (p, common.read(p))
            for p in (ROOT / 'exports' / method).glob('*/validation.json')}


def render_actor(method, bundle):
    if method == 'omnire':
        import torch
        from native_reaudit_omnire_v1 import render_bundle
        b = torch.load(bundle, map_location='cpu', weights_only=False)
        b['groups'] = [g for g in b['groups'] if g['node'] != 'Background']
        assert len(b['groups']) == 1 and len(b['groups'][0]['posed']['_means']) > 0
        rgb, alpha, _ = render_bundle(b)
        return rgb.detach().cpu(), alpha.detach().cpu()
    from native_reaudit_splatad_v1 import render_bundle
    return render_bundle(bundle, native_actor_only=True)


def export(method, ids=None):
    choices = decisions(method)
    candidates = common.validations(method)
    for sid in ids or sorted(choices):
        decision = choices[sid]
        resolution_path = ROOT / 'decisions' / (method + '.json')
        out = ROOT / 'exports' / method / sid
        out.mkdir(parents=True, exist_ok=True)
        vp = out / 'validation.json'
        if vp.exists():
            old = common.read(vp)
            assert old['resolution_sha256'] == common.sha(resolution_path)
            assert old['status'] == 'passed'
            continue
        cp, candidate = candidates[sid]
        assert candidate['status'] == 'passed' and candidate['reload_verified']
        common.checked(candidate['bundle'], candidate['bundle_sha256'])
        row = copy.deepcopy(candidate)
        row.update(candidate_validation=str(cp), candidate_validation_sha256=common.sha(cp),
                   resolution=str(resolution_path), resolution_sha256=common.sha(resolution_path),
                   selected_support=decision['selected_support'], resolution_reason=decision['reason'],
                   selection_uses_metric_scores=False)
        if decision['selected_support'] == 'native_actor_only':
            assert method != 'streetgs' and row['actor_node_present']
            rgb, alpha = render_actor(method, row['bundle'])
            rgb = np.rint(np.clip(rgb.numpy(), 0, 1) * 255).astype('uint8')
            alpha = np.rint(np.clip(alpha.numpy().squeeze(), 0, 1) * 255).astype('uint8')
            if method == 'omnire':
                native = BASE / method / sid / 'actor_only.png'
                prior = row.get('prior_registration')
                prior_rgb = prior.get('source_rgb') if prior else None
                prior_alpha = prior.get('source_alpha') if prior else None
            else:
                native = Path(row['comparison_images']['native_target_only'])
                prior_rgb = prior_alpha = None
            observed = np.asarray(Image.open(native).convert('RGB'))
            err = np.abs(rgb.astype('int16') - observed.astype('int16'))
            assert int(err.max()) <= 1, ('Independent actor replay differs', method, sid, int(err.max()))
            row['actor_only_reload'] = dict(native_reference=str(native), native_reference_sha256=common.sha(native),
                rgb_max_byte_difference=int(err.max()), rgb_mean_byte_difference=float(err.mean()),
                independent_of_full_scene_checkpoint=True, gaussian_parameters_modified=False)
            # Save this independent actor-only output. Candidate images and
            # checkpoints stay untouched; only small RGB/alpha images are added.
            for key, array, filename in [('actor_preview', rgb, 'actor_source.png'),
                                         ('actor_alpha', alpha, 'actor_alpha.png')]:
                path = out / filename
                Image.fromarray(array).save(path)
                row[key] = str(path)
                row[key + '_sha256'] = common.sha(path)
            if prior_rgb and prior_alpha:
                old_rgb = np.asarray(Image.open(prior_rgb).convert('RGB'))
                old_a = np.asarray(Image.open(prior_alpha).convert('L'))
                row['old_actor_replay'] = dict(rgb_max_byte_difference=int(np.abs(rgb.astype('int16')-old_rgb.astype('int16')).max()),
                    alpha_max_byte_difference=int(np.abs(alpha.astype('int16')-old_a.astype('int16')).max()))
        else:
            assert decision['selected_support'] == 'mask_visible_candidate'
            row['actor_only_reload'] = None
        common.save(vp, row)
        print(method, sid, row['selected_support'], flush=True)


def geometry_configure():
    import score_native_reaudit_geometry_v1 as geometry
    geometry.ROOT = ROOT
    geometry.validations = validations
    original_loader = geometry.load_export

    def selected_loader(method, sid, ref):
        bundle, v, pack, evidence = original_loader(method, sid, ref)
        common.checked(v['resolution'], v['resolution_sha256'])
        common.checked(v['candidate_validation'], v['candidate_validation_sha256'])
        if v['selected_support'] == 'native_actor_only':
            import torch
            if method == 'omnire':
                mask = torch.cat([torch.full((len(g['posed']['_means']),), g['node'] != 'Background', dtype=torch.bool)
                                  for g in bundle['groups']])
            else:
                assert method == 'splatad'
                mask = bundle['target_actor_rows'].bool()
            assert mask.any()
            pack['arrays'] = {k: x[mask].contiguous() for k, x in pack['arrays'].items()}
            if 'velocities' in pack:
                pack['velocities'] = pack['velocities'][mask].contiguous()
            evidence['raw_gaussian_count'] = int(mask.sum())
        evidence.update(selected_support=v['selected_support'], resolution=v['resolution'],
                        resolution_sha256=v['resolution_sha256'])
        return bundle, v, pack, evidence

    geometry.load_export = selected_loader
    return geometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['export', 'score', 'edp-prepare', 'edp-pose', 'edp-features', 'edp-score', 'geometry', 'summary'])
    p.add_argument('--method', choices=common.METHODS, required=True)
    p.add_argument('--ids', nargs='+')
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    cp = ROOT / 'batch_contract.json'
    if not cp.exists():
        os.link(BASE / 'batch_contract.json', cp)
    if a.action == 'export':
        export(a.method, a.ids)
    elif a.action == 'geometry':
        g = geometry_configure()
        for sid in a.ids or sorted(decisions(a.method)):
            g.run(a.method, sid, finalize_missing=True)
        g.summary(a.method)
    else:
        import score_native_reaudit_static_v1 as static
        static.ROOT = ROOT
        static.common.validations = validations
        static.configure(a.method)
        if a.action == 'score':
            ready = static.adapt_inputs(a.ids)
            static.original.score(ready, 'cuda')
        elif a.action == 'edp-prepare':
            static.original.edp_prepare(a.ids, 'all')
            static.reuse_reference_cache(static.original.OUT / 'edp' / 'all')
        elif a.action == 'summary':
            static.original.summary()
        else:
            static.edp_stage(a.action.removeprefix('edp-'), 'all')


if __name__ == '__main__':
    main()
