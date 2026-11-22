"""Route audited new exports through the unchanged accepted human metrics."""
import argparse
import json
import os
from pathlib import Path
import shutil

import native_reaudit_common as common
import score_streetgs_human_static_v1 as original

H, R, ROOT = common.H, common.R, common.ROOT
METHOD = None


def prepare():
    contract = common.freeze()
    p = ROOT/'metrics'/METHOD/'static_protocol.json'
    code = [H/'scripts/human_source_metrics_v1.py',
            R/'reports/source_appearance_alignment_20261115_v1/run_comparison.py',
            H/'scripts/score_streetgs_human_static_v1.py',
            H/'scripts/human_edp_common.py', H/'scripts/run_human_edp_common.py']
    source = contract['inherited_protocols']['appearance']
    common.checked(source['path'], source['sha256'])
    protocol=dict(version='native_human_reaudit_static_v1',method=METHOD,
        ids=contract['ids'],selection=contract['selection'],selection_sha256=contract['selection_sha256'],
        batch_contract=str(ROOT/'batch_contract.json'),batch_contract_sha256=common.sha(ROOT/'batch_contract.json'),
        scoring_reference_summary=contract['reference_summary'],scoring_reference_summary_sha256=contract['reference_summary_sha256'],
        appearance='Existing own-alpha centroid translation and isotropic sqrt(Agt/Apred) scale, common frozen ROI; no rotation/color fit',
        silhouette='Unwarped prediction in the frozen reference ROI; alpha byte>127',
        EDP='Unwarped common ROI and existing six-part DINOv3-L16 detector/embedding definitions',
        aggregation='Version A, each method own metric-valid identities and parts; NA not zero; valid poor/empty output retained',
        mask_conditioning='Reference mask used to select native Gaussians by visible contribution; never applied to prediction pixels',
        lpips=common.read(source['path'])['lpips'], inherited_appearance_protocol=source['path'],
        inherited_appearance_protocol_sha256=source['sha256'],code={str(x):common.sha(x) for x in code},
        old_scores_modified=False)
    if p.exists():assert common.read(p)==protocol,'Changed frozen scoring contract'
    else:common.save(p,protocol)
    return protocol, contract['references']


def adapt_inputs(ids=None):
    contract=common.freeze();found=common.validations(METHOD)
    chosen=ids or contract['ids'];ready=[]
    for sid in chosen:
        if sid not in found:continue
        path,v=found[sid];ref=contract['references'][sid]
        if v['status'] not in ('passed','unavailable','confirmed_unavailable'):continue
        if v['status']=='passed':
            assert v['method']==METHOD and v.get('id',v.get('sample_id'))==sid
            assert v['identity_key']==ref['identity_key']
            assert v['timestamp_us']==ref['timestamp_us']
            assert v['camera_contract_sha256']==ref['camera_contract_sha256']
            assert v['reload_verified'] is True
            if isinstance(v.get('source_reference'),dict):
                assert v['source_reference']==ref,'Export used a different frozen reference'
            if v.get('reference_mask_sha256'):
                assert v['reference_mask_sha256']==ref['reference_mask_sha256']
            for key in ('bundle','actor_preview','actor_alpha'):
                common.checked(v[key],v[key+'_sha256'])
        normalized=dict(v, native_validation=str(path),native_validation_sha256=common.sha(path))
        target=ROOT/'metric_inputs'/METHOD/sid/'validation.json'
        if target.exists():assert common.read(target)==normalized,'Export changed after score binding'
        else:common.save(target,normalized)
        ready.append(sid)
    return ready


_validate = original.validate
def validate(v,ref):
    common.checked(v['native_validation'],v['native_validation_sha256'])
    return _validate(v,ref)


def configure(method):
    global METHOD
    METHOD=method
    original.METHOD=method
    original.OUT=ROOT/'metrics'/method
    original.EXPORT=ROOT/'metric_inputs'/method
    original.freeze=prepare
    original.validate=validate


def reuse_reference_cache(root):
    """Reuse immutable reference observations, never copy model predictions."""
    old=H/common.read(H/'inputs/active_results.json')['edp_root']
    reused=[]
    for case in common.read(root/'inputs.json')['cases']:
        if case['method']!='reference':continue
        key=case['key'];pose=old/'poses'/(key+'.json');feat=old/'features'/(key+'.npz');meta=feat.with_suffix('.json')
        if not (pose.exists() and feat.exists() and meta.exists()):continue
        p=common.read(pose);m=common.read(meta)
        assert p['image_sha256']==case['image_sha256']
        assert m['image_sha256']==case['image_sha256'] and m['mask_sha256']==case['mask_sha256']
        common.checked(pose,m['pose_sha256']);common.checked(feat,m['features_sha256'])
        for src,dst in [(pose,root/'poses'/pose.name),(feat,root/'features'/feat.name),(meta,root/'features'/meta.name)]:
            if dst.exists():assert common.sha(dst)==common.sha(src)
            else:
                try:os.link(src,dst)
                except OSError:shutil.copyfile(src,dst)
        reused.append(key)
    common.save(root/'reference_cache_reuse.json',dict(keys=reused,source=str(old),
        policy='Exact image, mask, pose and feature hashes; immutable references only'))


def edp_stage(action,group):
    import run_human_edp_common as edp
    root=original.OUT/'edp'/group;edp.METHODS=[METHOD]
    if action in ['pose','features']:
        checker=edp.verified_inputs
        def pending(path):
            manifest,protocol=checker(path);cases=[]
            for c in manifest['cases']:
                posepath=path/'poses'/(c['key']+'.json')
                valid_pose=posepath.exists() and common.read(posepath).get('image_sha256')==c['image_sha256']
                valid=False
                if action=='pose':valid=valid_pose
                else:
                    feat=path/'features'/(c['key']+'.npz');meta=feat.with_suffix('.json')
                    if valid_pose and feat.exists() and meta.exists():
                        m=common.read(meta)
                        valid=(m.get('image_sha256')==c['image_sha256'] and m.get('mask_sha256')==c['mask_sha256']
                               and m.get('pose_sha256')==common.sha(posepath) and m.get('features_sha256')==common.sha(feat))
                if not valid:cases.append(c)
            return dict(manifest,cases=cases),protocol
        edp.verified_inputs=pending
    getattr(edp,action)(root)
    if action=='score':original.summary()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','score','summary','edp-prepare','edp-pose','edp-features','edp-score'])
    p.add_argument('--method',choices=common.METHODS,required=True);p.add_argument('--ids',nargs='+');p.add_argument('--device',default='cpu');p.add_argument('--group',default='all')
    a=p.parse_args();assert a.group.replace('_','').isalnum();configure(a.method)
    if a.action=='prepare':prepare()
    elif a.action=='score':
        ids=adapt_inputs(a.ids)
        if ids:original.score(ids,a.device)
    elif a.action=='summary':original.summary()
    elif a.action=='edp-prepare':
        original.edp_prepare(a.ids,a.group);reuse_reference_cache(original.OUT/'edp'/a.group)
    else:edp_stage(a.action.removeprefix('edp-'),a.group)


if __name__=='__main__':main()
