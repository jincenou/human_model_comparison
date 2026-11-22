"""Unchanged observed-surface CD/F-score protocol on reaudited native exports.

Both original metric coordinates and the frozen height-conditioned panel are
retained. No checkpoint reads, endpoint fitting, pruning or per-object tuning.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'
HG=H/'evaluation_static_cd_fscore_time_v1'
G=R/'evaluation_geometry_expanded_20261114_v2'
ROOT=H/'reports/native_human_export_reaudit_v1'
SCRIPT=Path(__file__).resolve()
sys.path[:0]=[str(HG/'scripts'),str(G/'scripts'),str(R/'scripts'),str(R/'evaluation_nuscenes_v1/scripts'),str(H/'scripts')]
from common import read,sha,checked,save,rigid_part,selection
from scene_depth import SceneDepth
from pilot import measure
from depth import query
from geometry_core import metrics,selftest
from native_reaudit_common import freeze,validations
from score_streetgs_human_geometry_v1 import alpha_equivalence

METHODS=('streetgs','omnire','splatad')
TERMINAL=('completed','extent_numerically_unresolved','export_unavailable')


def prepare(method):
    assert method in METHODS
    contract=freeze();old=read(HG/'protocol.json')
    checked(HG/'reference_manifest.json',old['reference_manifest_sha256'])
    for path,digest in old['code'].items():checked(path,digest)
    _,selected=selection();refs=read(HG/'reference_manifest.json')
    assert refs['selection_sha256']==contract['selection_sha256']==old['selection_sha256']
    assert {r['id'] for r in refs['rows']}==set(selected)
    assert collections.Counter(r['candidate_panel'] for r in refs['rows'])=={'main':84,'low_support_exploratory':16}
    out=ROOT/'geometry'/method
    protocol=dict(version='native_human_export_reaudit_geometry_v1',method=method,ids=sorted(selected),
        selection_sha256=old['selection_sha256'],batch_contract=str(ROOT/'batch_contract.json'),batch_contract_sha256=sha(ROOT/'batch_contract.json'),
        reference_manifest=str(HG/'reference_manifest.json'),reference_manifest_sha256=sha(HG/'reference_manifest.json'),
        inherited_protocol=str(HG/'protocol.json'),inherited_protocol_sha256=sha(HG/'protocol.json'),
        rules={k:old[k] for k in ['reference','reference_qualification','main_panel','scale_rule','scene_native_metric','annotation_information','prohibited','native_extent','depth','metrics']},
        input='Independently reloaded native source-exposure Gaussian export; all original parameters retained',
        renderer='Original method SceneDepth perspective kernel; original measure/pilot orthographic metrology (SplatAD shared-alpha helper)',
        alpha_equivalence_threshold=dict(max_byte_difference=2,mean_byte_difference=.01),
        aggregation='Own-valid per frozen 84 main / 16 low-support panel; equal identity; valid empty F=0, CD=null and extended mean infinity',
        old_scores_modified=False,code=old['code'],orthographic_splatad_helper=str(HG/'scripts/measure_splatad.py'),
        orthographic_splatad_helper_sha256=sha(HG/'scripts/measure_splatad.py'))
    path=out/'protocol.json'
    if path.exists():assert read(path)==protocol,'Frozen geometry protocol changed'
    else:save(path,protocol)
    archive=out/'code'/('wrapper_'+sha(SCRIPT)+'.py')
    if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);archive.write_text(SCRIPT.read_text())
    return protocol


def load_export(method,sid,ref):
    import torch
    found=validations(method)
    if sid not in found:raise FileNotFoundError('No final native export record')
    vp,v=found[sid]
    if v['status']!='passed' or not v.get('reload_verified'):raise ValueError('Native export not passed independent reload: '+v['status'])
    contract=read(ROOT/'batch_contract.json');expected=contract['references'][sid]
    assert v['source_reference']==expected
    assert v['identity_key']==ref['identity_key'] and v['timestamp_us']==ref['t_ref_us']
    checked(v['bundle'],v['bundle_sha256'])
    checked(v['actor_alpha'],v['actor_alpha_sha256'])
    # Geometry keeps the inherited physical camera; later masks/ROI do not change it.
    camera=read(ref['camera_contract']);current=read(expected['camera_contract'])
    for key in ['T_world_from_camera_opencv','K_raw','render_wh']:
        np.testing.assert_allclose(camera[key],current[key],atol=1e-8,rtol=0)
    bundle=torch.load(v['bundle'],map_location='cpu',weights_only=False)
    if method=='streetgs':
        from extract_streetgs_human_background_v1 import asset
        arrays=asset(bundle)['arrays'];arrays={k:val.float().contiguous() for k,val in arrays.items()}
        Tworld=np.asarray(bundle['T_world_from_native']);cfg=bundle['config']['trainer']['render'];extra={}
    elif method=='omnire':
        from native_reaudit_omnire_v1 import load_static_geometry
        values,Tworld=load_static_geometry(bundle,device='cpu')
        arrays={key:values[raw].float().contiguous() for key,raw in [('means','_means'),('quats','_quats'),('scales','_scales'),('opacity','_opacities')]}
        cfg=bundle['render_config'];extra={}
    else:
        from native_reaudit_splatad_v1 import load_posed_gaussians
        values=load_posed_gaussians(v['bundle'],device='cpu')
        arrays={k:values[k].float().contiguous() for k in ['means','quats','scales']}
        arrays['opacity']=values['opacities'].float().contiguous()
        Tworld=values['T_world_from_native'].cpu().numpy();cfg=values['render_cfg'];extra=dict(velocities=values['velocities'].float().contiguous())
        assert cfg['tile_size']==16 and abs(cfg['eps2d']-.3)<1e-12,'Inherited SplatAD metrology requires its native pinned projection settings'
    arrays['opacity']=arrays['opacity'].reshape(-1)
    count=len(arrays['means']);assert count>0
    assert arrays['means'].shape==(count,3) and arrays['quats'].shape==(count,4) and arrays['scales'].shape==(count,3)
    assert all(torch.isfinite(a).all() for a in arrays.values()) and bool((arrays['scales']>0).all())
    assert bool((arrays['opacity']>=0).all()) and bool((arrays['opacity']<=1).all())
    _,gauge=rigid_part(Tworld);assert abs(gauge-1)<1e-6
    backend='splatad' if method=='splatad' else 'omnire'
    pack=dict(arrays=arrays,method=backend,render_cfg=cfg,T_world_from_native=np.asarray(Tworld),**extra)
    evidence=dict(bundle=v['bundle'],bundle_sha256=v['bundle_sha256'],validation=str(vp),validation_sha256=sha(vp),
                  raw_gaussian_count=count,identity_key=ref['identity_key'],renderer_backend=backend,
                  input_parameters_modified=False,static_frozen_exposure=True,motion_added=False,
                  source_support_status=v.get('source_support_status'),semantic_review_required=v.get('semantic_review_required'))
    return bundle,v,pack,evidence


def splatad_extent(pack,out,ref):
    import torch
    from evaluate import helper_env
    # Original helper consumes prepared.pt and native world rotation, exactly as
    # the inherited evaluator; do not use a different kernel for perspective CD.
    prepared=out/'prepared.pt';torch.save(pack,prepared)
    save(out/'preparation.json',dict(id=ref['id'],method='splatad',prepared=str(prepared),prepared_sha256=sha(prepared),
         T_world_from_native=np.asarray(pack['T_world_from_native']).tolist(),source='Independent reaudited export; no full checkpoint read'))
    subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(HG/'scripts/measure_splatad.py'),str(out)],
                   env=helper_env(),check=True)
    return read(out/'measurement.json'),out


def run(method,sid,finalize_missing=False):
    protocol=prepare(method);assert sid in protocol['ids']
    root=ROOT/'geometry'/method;ref=next(r for r in read(HG/'reference_manifest.json')['rows'] if r['id']==sid)
    out=root/'runs'/sid;rp=root/'results'/(sid+'.json')
    if rp.exists() and read(rp)['status'] in TERMINAL:
        prior=read(rp);assert prior['protocol_sha256']==sha(root/'protocol.json') and prior['queries_sha256']==ref['queries_sha256']
        if prior['status']=='export_unavailable' and sid in validations(method) and validations(method)[sid][1].get('status')=='passed':
            pass
        else:
            if prior['status']=='completed':
                for key in ['certificate','returns']:checked(prior[key],prior[key+'_sha256'])
            return prior
    out.mkdir(parents=True,exist_ok=True)
    if rp.exists():save(out/('previous_'+sha(rp)+'.json'),read(rp))
    rec=dict(id=sid,method=method,identity_key=ref['identity_key'],t_ref_us=ref['t_ref_us'],reference_panel=ref['candidate_panel'],
             protocol_sha256=sha(root/'protocol.json'),queries_sha256=ref['queries_sha256'],status='running',started_at=time.time(),panels={},
             wrapper_sha256=sha(SCRIPT),batch_contract_sha256=sha(ROOT/'batch_contract.json'))
    save(rp,rec)
    found=validations(method)
    if sid not in found or found[sid][1].get('status')!='passed' or not found[sid][1].get('reload_verified'):
        rec.update(status='export_unavailable' if finalize_missing else 'waiting_for_export',
                   reason='No technically passed independently reloaded native export',finished_at=time.time())
        save(rp,rec);return rec
    adapter=None
    try:
        assert shutil.disk_usage(H).free>10*1024**3
        checked(ref['queries'],ref['queries_sha256']);checked(ref['camera_contract'],ref['camera_contract_sha256'])
        camera=read(ref['camera_contract']);Twc=np.asarray(camera['T_world_from_camera_opencv'])
        with np.load(ref['queries']) as z:arrays={k:z[k] for k in z.files}
        bundle,validation,pack,evidence=load_export(method,sid,ref)
        rec.update(export_validation=evidence['validation'],export_validation_sha256=evidence['validation_sha256'],bundle=evidence['bundle'],bundle_sha256=evidence['bundle_sha256'])
        adapter=SceneDepth(pack);Tworld=np.asarray(pack['T_world_from_native']);sourceT=np.linalg.inv(Twc)@Tworld
        checkT=sourceT
        if method=='splatad':
            # Match the original float32 camera inversion for this equivalence
            # check. Re-inverting its world transform in float64 then casting
            # can move near-plane splats despite <1e-6 matrix disagreement.
            # Physical query cameras and both scoring transforms stay unchanged.
            checkT=bundle['raster_settings']['viewmats'][0].cpu().numpy()
            error=float(np.max(np.abs(checkT-sourceT)))
            assert error<1e-5,('Saved source camera differs physically',error)
            np.testing.assert_allclose(bundle['raster_settings']['Ks'][0].cpu().numpy(),np.asarray(camera['K_raw']),atol=1e-3,rtol=0)
            rec['source_alpha_camera_precision']=dict(source='Exact independently reloaded native float32 view matrix',
                reconstructed_world_matrix_max_difference=error,physical_query_or_scale_rules_modified=False)
        source_check=alpha_equivalence(adapter,checkT,camera,validation,out)
        rec['source_alpha_check']=source_check
        rng,al,hit,diag=query(adapter,arrays,Tworld)
        met,_,_,_=metrics(arrays['q'],arrays['o'],arrays['u'],arrays['r'],rng,hit)
        np.savez_compressed(out/'native_metric_returns.npz',range_pred=rng,alpha=al,hit=hit)
        rec['panels']['native_metric']=dict(metrics=met,T_world_from_native=Tworld.tolist(),diagnostics=diag,
            returns=str(out/'native_metric_returns.npz'),returns_sha256=sha(out/'native_metric_returns.npz'))
        V,oldscale=rigid_part(sourceT);axes=Twc[:3,:3]@V[:3,:3]
        if method=='splatad':extent,extent_root=splatad_extent(pack,out,ref)
        else:extent_root=out/'extent';extent=measure(adapter,axes,extent_root)
        if extent['status']!='passed':
            rec.update(status='extent_numerically_unresolved',reason='Original bounded native-alpha vertical extent convergence not reached; no threshold or scale tuning',
                       extent=str(extent_root/'measurement.json'),extent_sha256=sha(extent_root/'measurement.json'))
        else:
            n=str(extent['precision']['fine_resolution']);height=extent['measurements'][n]['vertical_extent']
            scale=ref['annotation_vertical_extent_m']/height;assert np.isfinite(scale) and scale>0
            A=Twc@np.diag([scale,scale,scale,1.])@V
            assert abs(scale*height-ref['annotation_vertical_extent_m'])<1e-9
            cert=dict(id=sid,method=method,identity_key=ref['identity_key'],t_ref_us=ref['t_ref_us'],protocol_sha256=rec['protocol_sha256'],
                 reference_manifest_sha256=sha(HG/'reference_manifest.json'),queries_sha256=ref['queries_sha256'],evidence=evidence,
                 camera_contract_sha256=ref['camera_contract_sha256'],T_source_camera_from_native=sourceT.tolist(),source_similarity_gauge_removed=oldscale,
                 T_world_from_native=A.tolist(),scale_m_per_native_unit=scale,native_vertical_extent=height,
                 annotation_vertical_extent_m=ref['annotation_vertical_extent_m'],extent=str(extent_root/'measurement.json'),extent_sha256=sha(extent_root/'measurement.json'),
                 translation_scaled=True,full_covariance_transformation='s^2 R Sigma R^T via equivalent camera',root_pose_edited=False,
                 LiDAR_endpoint_fitting=False,added_geometry_pruning=False,motion_added=False)
            save(out/'metric_scale_certificate.json',cert)
            rng,alpha,hit,diag=query(adapter,arrays,A)
            met,points,dpg,dgp=metrics(arrays['q'],arrays['o'],arrays['u'],arrays['r'],rng,hit)
            np.savez_compressed(out/'returns.npz',range_pred=rng,alpha=alpha,hit=hit,predicted_endpoints=points,distances_pg=dpg,distances_gp=dgp)
            rec['panels']['annotation_vertical_extent_conditioned']=dict(metrics=met,diagnostics=diag)
            rec.update(status='completed',certificate=str(out/'metric_scale_certificate.json'),certificate_sha256=sha(out/'metric_scale_certificate.json'),
                       returns=str(out/'returns.npz'),returns_sha256=sha(out/'returns.npz'))
        del bundle,pack
    except Exception as exc:
        rec.update(status='technical_failure_pending_review',reason=repr(exc),traceback=traceback.format_exc())
    finally:
        del adapter;gc.collect()
        import torch
        torch.cuda.empty_cache()
    rec['finished_at']=time.time();save(rp,rec);print(method,sid,rec['status'],rec.get('reason'),flush=True)
    return rec


def summary(method):
    protocol=prepare(method);root=ROOT/'geometry'/method
    rows=[read(p) for p in sorted((root/'results').glob('*.json'))]
    assert len({r['id'] for r in rows})==len(rows)
    assert all(r['id'] in protocol['ids'] and r['protocol_sha256']==sha(root/'protocol.json') for r in rows)
    panels={}
    for panel in ['main','low_support_exploratory']:
        panels[panel]={}
        for measurement in ['annotation_vertical_extent_conditioned','native_metric']:
            pairs=[(r,r['panels'][measurement]['metrics']) for r in rows if r['reference_panel']==panel and measurement in r.get('panels',{})]
            finite=[(r,m) for r,m in pairs if m['cd_m'] is not None];empty=sorted(r['id'] for r,m in pairs if m['status']=='empty_pred')
            cells=dict(CD_cm=dict(value=float(np.mean([m['cd_m'] for r,m in finite])*100) if finite else None,n=len(finite),ids=sorted(r['id'] for r,m in finite),unit='cm',valid_empty_ids=empty,valid_empty_n=len(empty),extended_mean='infinity' if empty else 'finite' if finite else None))
            for metric,key in [('F5','f1_5cm'),('F10','f1_10cm'),('F20','f1_20cm'),('RayHit','ray_hit')]:
                cells[metric]=dict(value=float(np.mean([m[key] for r,m in pairs])*100) if pairs else None,n=len(pairs),ids=sorted(r['id'] for r,m in pairs),unit='percent',valid_empty_n=len(empty))
            panels[panel][measurement]=cells
    counts=dict(collections.Counter(r['status'] for r in rows))
    result=dict(method=method,status='completed' if len(rows)==100 and all(r['status'] in TERMINAL for r in rows) else 'incremental',
                main_panel='main / annotation_vertical_extent_conditioned',groups=panels,coverage=counts,expected=100,
                unattempted_ids=sorted(set(protocol['ids'])-{r['id'] for r in rows}),updated_at=time.time(),protocol_sha256=sha(root/'protocol.json'),old_scores_modified=False)
    save(root/'summary.json',result);return result


def worker(method):
    """One leased GPU, finite attempts per ready export, streaming SplatAD."""
    protocol=prepare(method);root=ROOT/'geometry'/method;statepath=root/'worker.json'
    state=dict(status='running',method=method,pid=os.getpid(),started_at=time.time(),wrapper_sha256=sha(SCRIPT))
    while True:
        found=validations(method)
        rows={p.stem:read(p) for p in (root/'results').glob('*.json')}
        # Technical exceptions are retained for explicit review, never retried
        # automatically. Missing records can become eligible when exports land.
        pending=[sid for sid in protocol['ids'] if sid not in rows or rows[sid]['status'] in ('waiting_for_export','running')]
        ready=[sid for sid in pending if sid in found and found[sid][1].get('status')=='passed' and found[sid][1].get('reload_verified')]
        if ready:
            for sid in ready:
                state.update(status='running',current_id=sid,updated_at=time.time());save(statepath,state)
                run(method,sid)
                result=summary(method);state.update(coverage=result['coverage'],updated_at=time.time());save(statepath,state)
            continue
        export_final=len(found)==100
        if method=='splatad' and (ROOT/'splatad/state.json').exists():
            export_final=export_final or read(ROOT/'splatad/state.json').get('status') in ('finished','finished_with_failures')
        if not pending or export_final:
            for sid in pending:run(method,sid,finalize_missing=True)
            result=summary(method)
            state.update(status='finished' if result['status']=='completed' else 'finished_with_technical_review',
                         current_id=None,coverage=result['coverage'],finished_at=time.time(),updated_at=time.time())
            save(statepath,state);print(method,state['status'],state['coverage'],flush=True);return
        state.update(status='waiting_for_exports',current_id=None,pending_ids=pending,updated_at=time.time());save(statepath,state)
        time.sleep(30)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','summary','selftest','worker']);p.add_argument('--method',choices=METHODS);p.add_argument('--ids',nargs='+');p.add_argument('--finalize-missing',action='store_true')
    a=p.parse_args()
    if a.action=='selftest':print(json.dumps(selftest()));return
    assert a.method
    if a.action=='prepare':prepare(a.method);return
    if a.action=='summary':print(json.dumps(summary(a.method)));return
    if a.action=='worker':worker(a.method);return
    assert a.ids
    for sid in a.ids:run(a.method,sid,a.finalize_missing)
    s=summary(a.method);print(a.method,s['coverage'],flush=True)


if __name__=='__main__':main()
