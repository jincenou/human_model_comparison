"""Scene-batched StreetGS source-mask exports for the frozen human100 cohort.

Select every native Background row with positive visible source-mask contribution.
No Gaussian parameter changes, mask erasing, inferred animation or metric tuning.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
REPORT = H / 'reports/native_human_export_reaudit_v1/streetgs'
OUT = H / 'exports/native_human_export_reaudit_v1/streetgs'
CONTRACT = H / 'reports/native_human_export_reaudit_v1/batch_contract.json'
sys.path[:0] = [str(H/'scripts'), str(R/'scripts')]
from extract_streetgs_human_background_v1 import KEYS, read, save, sha, render


def reserve():
    assert shutil.disk_usage(H).free > 10*1024**3, '10 GiB disk reserve reached'


def prepare():
    from score_streetgs_human_static_v1 import references
    active, reference_summary, refs = references()
    contract = read(CONTRACT)
    assert contract['references'] == refs
    selection = H/'inputs/benchmark_human100_v3/selection.json'
    assert sha(selection) == '03110e1d403248ff84d4f108a23602aad153781c76530fbc572bd5bc22324d55'
    path = REPORT/'plan.json'
    if path.exists():
        result = read(path)
        assert result['selection_sha256'] == sha(selection)
        assert result['reference_summary_sha256'] == sha(reference_summary)
        assert all(r['source_reference'] == contract['references'][r['id']] for r in result['rows'])
        if 'batch_contract_sha256' not in result:
            result.update(batch_contract=str(CONTRACT),batch_contract_sha256=sha(CONTRACT));save(path,result)
        assert result['batch_contract_sha256'] == sha(CONTRACT)
        return result
    rows = []
    for obj in read(selection)['objects']:
        oldpath = H/'exports/streetgs_human_probe_v1/objects'/obj['id']/'validation.json'
        old = read(oldpath)
        assert old['status'] == 'passed' and old['identity_key'] == obj['identity_key']
        reference = refs[obj['id']]
        assert reference['timestamp_us'] == obj['t_ref_us'] and reference['identity_key'] == obj['identity_key']
        rows.append(dict(id=obj['id'], scene=obj['scene'], identity_key=obj['identity_key'],
                         timestamp_us=obj['t_ref_us'], source_reference=reference,
                         old_validation=str(oldpath), old_validation_sha256=sha(oldpath),
                         checkpoint=old['checkpoint'], checkpoint_sha256=old['checkpoint_sha256']))
    assert len(rows) == 100 and len({x['id'] for x in rows}) == 100
    groups = []
    for scene in sorted({r['scene'] for r in rows}):
        entries = [r for r in rows if r['scene'] == scene]
        assert len({r['checkpoint_sha256'] for r in entries}) == 1
        groups.append(dict(scene=scene, ids=[r['id'] for r in entries]))
    result = dict(schema='native_human_export_reaudit_streetgs_v1', selection=str(selection),
                  selection_sha256=sha(selection), rows=rows, scenes=groups,
                  reference_summary=str(reference_summary),reference_summary_sha256=sha(reference_summary),
                  batch_contract=str(CONTRACT),batch_contract_sha256=sha(CONTRACT),
                  rule='All original Background rows with F > 0 under complete native-scene visibility',
                  temporal_policy='Static frozen-exposure export; no fabricated human dynamics',
                  disk_reserve_gib=10, created_at=time.time())
    save(path, result)
    return result


def complete(sid):
    p = OUT/sid/'validation.json'
    if not p.exists(): return False
    v = read(p)
    return v.get('status') == 'passed' and v.get('reload_verified') and Path(v['bundle']).is_file() and sha(v['bundle']) == v['bundle_sha256']


def png(value, path):
    a = value.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[-1] == 1: a = a[...,0]
    assert np.isfinite(a).all()
    a = (a.clip(0,1)*255).round().astype('uint8')
    Image.fromarray(a).save(path)
    return a


class Scene:
    def __init__(self, row):
        from run_background_vehicle_scene import DriveStudio
        self.old = read(row['old_validation'])
        assert sha(self.old['checkpoint']) == row['checkpoint_sha256']
        self.native = DriveStudio(self.old)
        self.checkpoint = torch.load(self.old['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        assert 'SMPLNodes' not in self.checkpoint['models'] and 'DeformableNodes' not in self.checkpoint['models']

    def run(self, row):
        reserve()
        sid = row['id']; dest = OUT/sid; report = REPORT/sid
        dest.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
        oldrec = read(row['old_validation'])
        assert sha(row['old_validation']) == row['old_validation_sha256']
        assert sha(oldrec['bundle']) == oldrec['bundle_sha256']
        oldpack = torch.load(oldrec['bundle'], map_location='cpu', weights_only=False)
        ref = row['source_reference']
        camera, binding = read(ref['camera_contract']), oldpack['binding']
        assert camera['identity_key'] == row['identity_key'] and camera['timestamp_us'] == row['timestamp_us']
        for key in ['camera_contract','reference_rgb','reference_mask']:
            assert sha(ref[key]) == ref[key+'_sha256']
        assert ref['reference_rgb_sha256'] == camera['source_rgb_sha256']
        truth = np.asarray(Image.open(ref['reference_mask']).convert('L')) > 127
        width, height = camera['render_wh']; assert truth.shape == (height,width) and truth.any()
        native = self.native; trainer, dataset = native.trainer, native.dataset
        from datasets.base.pixel_source import get_rays
        from models.gaussians.basics import dataclass_gs
        import models.trainers.base as base
        index = binding['source_frame']*dataset.pixel_source.num_cams + dataset.pixel_source.camera_data[binding['native_camera_id']].unique_cam_idx
        assert index == binding['source_camera_index']
        info, ci = dataset.pixel_source.get_image(index)
        info = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in info.items()}
        ci = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in ci.items()}
        pose = native.T @ np.asarray(camera['T_world_from_camera_opencv'])
        camera_error = float(np.abs(ci['camera_to_world'].cpu().numpy()-pose).max()); assert camera_error < 1e-4
        ci.update(camera_to_world=torch.tensor(pose,dtype=torch.float32,device='cuda'),
                  intrinsics=torch.tensor(camera['K_raw'],dtype=torch.float32,device='cuda'),
                  width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
        x,y = torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
        origins,directions,norm = get_rays(x.flatten(),y.flatten(),ci['camera_to_world'],ci['intrinsics'])
        info = {k:info[k].flatten()[0].expand(height,width) for k in ('img_idx','normed_time','frame_idx')}
        info.update(origins=origins.reshape(height,width,3),viewdirs=directions.reshape(height,width,3),
                    direction_norm=norm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
        nbg = len(native.model._means); raster = base.rasterization; capture = {}
        def recording_raster(**kwargs):
            result = raster(**kwargs)
            if not capture and len(kwargs['means']) >= nbg:
                assert torch.equal(kwargs['means'][:nbg],native.model._means)
                capture.update(kwargs={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items() if k not in ['colors','backgrounds']}, alpha=result[1].detach())
            return result
        with torch.no_grad():
            native_result = trainer(info,ci,novel_view=False)
            png(native_result['rgb'],report/'fullscene_native_CamPose.png')
            del native_result
        base.rasterization = recording_raster
        try:
            with torch.no_grad():
                result = trainer(info,ci,novel_view=True)
                full = png(result['rgb'],report/'fullscene_source_camera.png')
                del result
        finally: base.rasterization = raster
        kwargs = capture['kwargs']; mask = torch.from_numpy(truth).cuda(); na = capture['alpha']
        expected = np.array([float(na[0,:,:,0][mask].double().sum()),float(na[0,:,:,0][~mask].double().sum())])
        reuse = None; prior_path = H/'reports/streetgs_human_mask_all_v2'/sid/'audit.json'
        if prior_path.exists():
            prior = read(prior_path)
            assert prior['status']=='passed' and prior['checkpoint_sha256']==row['checkpoint_sha256']
            assert prior['source_reference_mask_sha256']==ref['reference_mask_sha256']
            assert prior['camera_contract_sha256']==ref['camera_contract_sha256']
            wp = prior_path.parent/'native_contributions.npz'
            if not wp.exists(): wp = Path(prior['prior_contribution_reuse']['weights'])
            weights = np.load(wp,allow_pickle=False); fg,bg = weights['foreground_weight'],weights['background_weight']
            assert len(fg)==nbg and np.isclose(fg.sum(dtype=np.float64),prior['fg_mass_background'],rtol=1e-12)
            assert np.isclose(expected[0],prior['fg_mass_full_scene'],rtol=2e-4)
            conservation = prior['contribution_conservation_relative_error']
            nonbg = dict(value=float(expected[0]-fg.sum(dtype=np.float64)),source='alpha total minus reused Background contribution; may include reduction roundoff')
            reuse = dict(audit=str(prior_path),audit_sha256=sha(prior_path),weights=str(wp),weights_sha256=sha(wp))
        else:
            count = len(kwargs['means'])
            colors = torch.ones((count,3),device='cuda',dtype=kwargs['means'].dtype,requires_grad=True)
            kwargs.update(colors=colors,render_mode='RGB',sh_degree=None,absgrad=False)
            rendered,alpha,_ = raster(**kwargs)
            assert float((alpha-na).abs().max())<1e-5 and float((rendered-alpha).abs().max())<1e-5
            loss = rendered[0,:,:,0][mask].sum()+rendered[0,:,:,1][~mask].sum()
            weights = torch.autograd.grad(loss,colors,only_inputs=True)[0][:,:2].detach().cpu().numpy()
            assert np.isfinite(weights).all() and (weights>=0).all()
            conservation = (np.abs(weights.sum(0,dtype=np.float64)-expected)/np.maximum(expected,1)).tolist()
            assert max(conservation)<2e-4
            fg,bg = weights[:nbg].T
            nonbg = dict(value=float(weights[nbg:,0].sum(dtype=np.float64)),source='direct native non-Background contribution; not exported')
        ids = np.flatnonzero(fg>0).astype(np.int64)
        assert len(ids)>0
        total = float(fg.sum(dtype=np.float64)); selected_mass = float(fg[ids].sum(dtype=np.float64))
        assert np.isclose(selected_mass,total,rtol=1e-12,atol=1e-9)
        index_path=dest/'native_indices.npy'; np.save(index_path,ids,allow_pickle=False)
        evidence_ids=np.flatnonzero((fg>0)|(bg>0)).astype(np.int64)
        contribution_path=report/'native_contributions.npz'
        np.savez_compressed(contribution_path,native_indices=evidence_ids,foreground_weight=fg[evidence_ids],background_weight=bg[evidence_ids])
        ix=torch.as_tensor(ids,device='cuda')
        canonical={k:getattr(native.model,k)[ix].detach().cpu().clone().contiguous() for k in KEYS}
        assert all(torch.equal(canonical[k],self.checkpoint['models']['Background'][k][ids]) for k in KEYS)
        pack=dict(oldpack,canonical=canonical,original_indices=torch.as_tensor(ids),camera=camera,
                  schema='native_human_export_reaudit_streetgs_v1',source_reference=ref,
                  selection_rule='native Background visible foreground contribution > 0',
                  node_origins={'Background':str(index_path)},native_dynamic_human_available=False)
        bundle_path=dest/'native_background_subset.pt';torch.save(pack,bundle_path)
        with torch.no_grad():
            cam=trainer.process_camera(ci,info['img_idx'].flatten()[0],novel_view=True)
            gs=native.model.get_gaussians(cam)
            selected=dataclass_gs(**{k:v[ix] for k,v in gs.items()},detach_keys=[],extras=None)
            result,_=trainer.render_gaussians(gs=selected,cam=cam,near_plane=trainer.render_cfg.near_plane,
                far_plane=trainer.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=trainer.render_cfg.get('radius_clip',0.))
            a,raw=result['opacity'],result['rgb_gaussians']
            bias=trainer.affine_transformation(torch.zeros_like(raw),info)
            rgb=trainer.affine_transformation(raw,info)-bias*(1-a)+(1-a)
            preview=dest/'native_actor_source.png';alpha_path=dest/'native_actor_alpha.png'
            candidate=png(rgb,preview);png(a,alpha_path)
        font=ImageFont.truetype(_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'),22)
        oldimage=np.asarray(Image.open(oldrec['actor_preview']).convert('RGB'))
        source=np.asarray(Image.open(ref['reference_rgb']).convert('RGB'))
        display=np.where(truth[...,None],full,255).astype('uint8')
        Image.fromarray(display).save(report/'fullscene_masked_DISPLAY_ONLY.png')
        sheet=Image.new('RGB',(512*5,552),'white');draw=ImageDraw.Draw(sheet)
        for i,(name,image) in enumerate([('Source RGB',source),('Native source camera',full),('Scene mask (display only)',display),('Old box export',oldimage),('All mask-visible BG',candidate)]):
            draw.text((i*512+12,8),name,font=font,fill='black')
            sheet.paste(Image.fromarray(image).crop(tuple(ref['roi_xyxy'])).resize((512,512),Image.Resampling.BICUBIC),(i*512,40))
        sheet.save(report/'comparison.jpg',quality=95)
        known=sid=='HN0108'
        rec=dict(id=sid,sample_id=sid,method='streetgs',status='exported_pending_reload',scene=row['scene'],
                 batch_contract=str(CONTRACT),batch_contract_sha256=sha(CONTRACT),
                 identity_key=row['identity_key'],actor_uuid=row['identity_key'].split(':')[-1],
                 source_reference=ref,camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],
                 checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],
                 config=oldrec['config'],config_sha256=oldrec['config_sha256'],
                 bundle=str(bundle_path),bundle_sha256=sha(bundle_path),actor_preview=str(preview),actor_preview_sha256=sha(preview),
                 actor_alpha=str(alpha_path),actor_alpha_sha256=sha(alpha_path),prediction_encoding='white_composited_uint8',
                 render_wh=camera['render_wh'],gaussians=len(ids),selected_count=len(ids),indices=str(index_path),indices_sha256=sha(index_path),
                 node_origins={'Background':len(ids)},T_world_from_native=oldpack['T_world_from_native'],binding=binding,
                 frozen_raw_timestamp_us=row['timestamp_us'],source_timestamp_us=row['timestamp_us'],
                 timestamp_us=row['timestamp_us'],
                 reused=False,changed=not np.array_equal(ids,oldpack['original_indices'].numpy()),
                 old_validation=row['old_validation'],old_validation_sha256=row['old_validation_sha256'],
                 old_selected_count=len(oldpack['original_indices']),
                 selection_rule='F > 0 without trimming; full native-scene transmittance and occlusion retained during contribution measurement',
                 fg_mass_scene=float(expected[0]),fg_mass_background=total,fg_mass_selected=selected_mass,fg_contribution_coverage=selected_mass/total,
                 bg_mass_selected=float(bg[ids].sum(dtype=np.float64)),nonbackground_foreground_contribution=nonbg,
                 nonbackground_selection_policy='Excluded; no vehicle-node rows added',
                 old_fg_mass=float(fg[oldpack['original_indices'].numpy()].sum(dtype=np.float64)),
                 contribution=str(contribution_path),contribution_sha256=sha(contribution_path),conservation_relative_error=conservation,
                 prior_contribution_reuse=reuse,native_scene_gaussians=len(capture['kwargs']['means']),
                 selected_parameters_equal_checkpoint=True,raw_parameters_modified=False,reload_verified=False,
                 source_camera_error=camera_error,native_dynamic_human_available=False,temporal_metric_status='NA_no_native_dynamic_human',
                 native_human_status='unavailable_at_frozen_and_neighbor_observations' if known else 'pending_visual_review',
                 source_support_status='native_scene_human_unavailable' if known else 'pending_semantic_review',semantic_review_required=not known,
                 actor_node_present=False,
                 human_presence_verified=False,native_human_unavailable_evidence=str(H/'reports/streetgs_human_scene_audit_v1/neighbors_HN0108/index.html') if known else None,
                 contamination_status='May include background, occluders, neighboring people and mixed Gaussian tails; visual review pending',
                 formal_metrics_updated=False,reference_mask_erased_prediction=False,
                 reason='Original scene has no clear human at frozen and nearby exposures; preserve native export as evidence' if known else 'Replace center-box cut with all positive source-mask contributions; human quality requires visual review',
                 comparison=str(report/'comparison.jpg'),fullscene_source=str(report/'fullscene_source_camera.png'),
                 fullscene_native_CamPose=str(report/'fullscene_native_CamPose.png'),script_sha256=sha(__file__),exported_at=time.time())
        save(dest/'validation.json',rec)
        return rec


def reload(ids):
    """Separate process: reads only bundles, their validation and candidate images."""
    for sid in ids:
        dest=OUT/sid;p=dest/'validation.json';rec=read(p)
        assert rec['status'] in ['exported_pending_reload','passed'] and sha(rec['bundle'])==rec['bundle_sha256']
        pack=torch.load(rec['bundle'],map_location='cpu',weights_only=False)
        assert pack['id']==sid and pack['identity_key']==rec['identity_key']
        rgb,alpha=render(pack);diffs={}
        for name,arr,path in [('rgb',rgb,rec['actor_preview']),('alpha',alpha,rec['actor_alpha'])]:
            old=np.asarray(Image.open(path));delta=np.abs(old.astype(int)-arr.astype(int))
            diffs[name+'_max']=int(delta.max());diffs[name+'_mean']=float(delta.mean())
        assert diffs['rgb_max']<=1 and diffs['alpha_max']<=1,diffs
        record=dict(status='passed',independent_process=True,scene_checkpoint_reads=0,source_image_reads=0,source_mask_reads=0,
                    reference='Upstream native trainer subset render before independent reload',**diffs,
                    bundle_sha256=rec['bundle_sha256'],checked_at=time.time(),script_sha256=sha(__file__))
        save(dest/'reload.json',record)
        rec.update(status='passed',reload_verified=True,reload_record=str(dest/'reload.json'),reload_sha256=sha(dest/'reload.json'))
        save(p,rec);print(sid,'independent reload passed',diffs,flush=True)


def run_groups(plan,groups,state_path):
    rows={r['id']:r for r in plan['rows']}
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),groups=[dict(g,status='queued') for g in groups],script_sha256=sha(__file__))
    save(state_path,state)
    for job in state['groups']:
        pending=[sid for sid in job['ids'] if not complete(sid)]
        if not pending:
            job.update(status='reused_completed',finished_at=time.time());save(state_path,state);continue
        job.update(status='running',started_at=time.time());save(state_path,state)
        engine=None;exported=[];failures=[]
        try:
            reserve();engine=Scene(rows[pending[0]])
            for sid in pending:
                try:
                    engine.run(rows[sid]);exported.append(sid)
                    job.update(exported=exported,failures=failures);save(state_path,state)
                except Exception as exc:
                    failures.append(dict(id=sid,reason=repr(exc)))
                    save(OUT/sid/'validation.json',dict(id=sid,method='streetgs',status='execution_failed',reason=repr(exc),traceback=traceback.format_exc(),source_reference=rows[sid]['source_reference'],checkpoint=rows[sid]['checkpoint'],checkpoint_sha256=rows[sid]['checkpoint_sha256'],reload_verified=False))
                    print(sid,traceback.format_exc(),flush=True)
        except Exception as exc:
            failures.extend(dict(id=sid,reason=repr(exc)) for sid in pending if sid not in exported)
            print(traceback.format_exc(),flush=True)
        finally:
            del engine;gc.collect();torch.cuda.empty_cache()
        if exported:
            command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'reload','--ids',*exported]
            code=subprocess.call(command,env=os.environ.copy())
            if code: failures.append(dict(ids=exported,reason=f'independent_reload_exit_{code}'))
        job.update(status='completed' if not failures else 'completed_with_failures',failures=failures,exported=exported,finished_at=time.time())
        save(state_path,state)
    state.update(status='finished',finished_at=time.time());save(state_path,state)
    summary(plan)


def summary(plan):
    rows=[]
    for row in plan['rows']:
        p=OUT/row['id']/'validation.json'
        if p.exists():
            v=read(p);rows.append(dict(id=row['id'],status=v['status'],reload_verified=v.get('reload_verified',False),validation=str(p),native_human_status=v.get('native_human_status')))
        else:rows.append(dict(id=row['id'],status='pending'))
    from collections import Counter
    result=dict(rows=rows,counts=dict(Counter(r['status'] for r in rows)),total=100,updated_at=time.time())
    save(REPORT/'summary.json',result);print(result['counts'],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','scene','worker','reload','summary'])
    ap.add_argument('--scene');ap.add_argument('--ids',nargs='*');ap.add_argument('--shard',type=int);ap.add_argument('--num-shards',type=int,default=2)
    args=ap.parse_args();torch.set_num_threads(2)
    if args.stage=='reload':reload(args.ids);return
    plan=prepare();REPORT.mkdir(parents=True,exist_ok=True)
    snapshot=REPORT/('script_'+sha(__file__)+'.py')
    if not snapshot.exists():snapshot.write_text(Path(__file__).read_text())
    if args.stage=='prepare':print(len(plan['rows']),len(plan['scenes']));return
    if args.stage=='summary':summary(plan);return
    if args.stage=='scene':
        groups=[g for g in plan['scenes'] if g['scene']==args.scene];assert len(groups)==1
        if args.ids:
            assert set(args.ids).issubset(groups[0]['ids']);groups=[dict(scene=args.scene,ids=args.ids)]
        run_groups(plan,groups,REPORT/('pilot_'+args.scene+'.json'));return
    assert args.shard is not None and 0<=args.shard<args.num_shards
    run_groups(plan,plan['scenes'][args.shard::args.num_shards],REPORT/f'worker_{args.shard}.json')


if __name__=='__main__':main()
