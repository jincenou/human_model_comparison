"""Frozen-source OmniRe exports: complete actor plus positive Background support.

Bundles contain selected original parameters, directional SH and posed geometry,
not baked source RGB. Shared scene dependencies contain no scene Gaussians.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse
import contextlib
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

H = Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'))
R = H.parent / 'vehicle_model_comparison'
REPO = Path(_hmc_expand('${DRIVESTUDIO_ROOT}'))
REPORT = H / 'reports/native_human_export_reaudit_v1/omnire'
EXPORT = H / 'exports/native_human_export_reaudit_v1/omnire'
sys.path[:0] = [str(H/'scripts'), str(REPO)]
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from datasets.driving_dataset import DrivingDataset
from datasets.base.pixel_source import get_rays
from utils.misc import import_str
from models.gaussians.basics import dataclass_gs, spherical_harmonics
from omnire_time_query_v1 import at_physical_exposure

def read(p):
    return json.loads(Path(p).read_text())

def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); q.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def checked(p, s):
    assert sha(p) == s, str(p)

def image(t, p):
    a = t.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[-1] == 1: a = a[...,0]
    Image.fromarray(np.clip(np.rint(a*255),0,255).astype(np.uint8)).save(p)

def prepare():
    active = read(H/'inputs/active_human_benchmark.json'); checked(active['selection'], active['selection_sha256'])
    selected = read(active['selection'])['objects']
    contract = read(H/'reports/native_human_export_reaudit_v1/batch_contract.json')
    assert contract['selection_sha256']==active['selection_sha256']
    rows = {r['id']:r for r in read(H/'reports/source_camera_metric_cohort_v3/summary.json')['rows'] if r['method']=='omnire'}
    plans = {}; times = {}
    for group in ['scene_extract_v2','scene_extract_original75_v1','scene_extract_replacement_v3']:
        p = H/'inputs'/group/'omnire_plan.json'
        for r in read(p)['objects']: plans[r['id']] = dict(r, plan_path=str(p), plan_sha256=sha(p))
    for group in ['model_time_audit_v1','model_time_replacement_v3']:
        p = H/'reports'/group/'omnire.json'
        for r in read(p)['records']: times[r['id']] = dict(r, time_audit=str(p), time_audit_sha256=sha(p))
    objects = []
    for obj in selected:
        sid = obj['id']; p = plans[sid]; ref = contract['references'][sid]; old = None
        assert p['raw_timestamp_us'] == obj['t_ref_us'] == ref['timestamp_us'] == times[sid]['timestamp_us']
        assert p['native_id'] == obj['identity_key'] == ref['identity_key']
        if rows[sid]['status'] == 'scored':
            result = read(rows[sid]['result']); old = read(result['registration_record'])
        objects.append(dict(id=sid, selected=obj, source_reference=ref, plan=p, time=times[sid], prior_registration=old))
    groups = {}
    for obj in objects: groups.setdefault(obj['plan']['checkpoint'], []).append(obj)
    group_rows = [dict(checkpoint=k, objects=v) for k,v in sorted(groups.items())]
    for i,g in enumerate(group_rows): g['index']=i; g['shard']=i%2
    result = dict(selection=active['selection'],selection_sha256=active['selection_sha256'],count=len(objects),groups=group_rows,
        policy='Every original actor Gaussian plus every Background Gaussian with strictly positive native compositing contribution in frozen reference mask; other actor nodes recorded only; no bbox, top-k, percentile or posthoc score filtering',
        representation='Static source-exposure snapshot with original SH, canonical selected parameters and shared native deformation/affine dependency; background is not assigned artificial motion')
    save(REPORT/'plan.json',result); return result

def load_static_geometry(path, device='cuda'):
    """Geometry adapter: native metre coordinates at the frozen physical exposure."""
    b = torch.load(path, map_location='cpu', weights_only=False) if not isinstance(path, dict) else path
    out = {k:torch.cat([g['posed'][k] for g in b['groups']],dim=0).to(device) for k in ['_means','_scales','_quats','_opacities']}
    return out, np.asarray(b['T_world_from_native'],dtype=np.float64)

def load_static_gaussians(path, camera_to_native=None, device='cuda'):
    """Independent fixed-time renderer inputs; reevaluate SH for each camera."""
    b = torch.load(path,map_location='cpu',weights_only=False) if not isinstance(path,dict) else path
    pose = torch.as_tensor(b['camera_to_native'] if camera_to_native is None else camera_to_native,device=device,dtype=torch.float32)
    values = {k:[] for k in ['_means','_scales','_quats','_opacities','_rgbs']}
    for g in b['groups']:
        means = g['posed']['_means'].to(device)
        features = torch.cat([g['canonical']['_features_dc'][:,None,:],g['canonical']['_features_rest']],dim=1).to(device)
        if g['sh_degree'] > 0:
            dirs = means-pose[:3,3]; dirs=dirs/dirs.norm(dim=-1,keepdim=True)
            rgb = (spherical_harmonics(g['active_sh_degree'],dirs,features)+.5).clamp(0,1)
        else: rgb = torch.sigmoid(features[:,0,:])
        values['_rgbs'].append(rgb)
        for k in values:
            if k!='_rgbs': values[k].append(g['posed'][k].to(device))
    return {k:torch.cat(v,dim=0) for k,v in values.items()}, b

def render_bundle(path, camera_to_native=None, K=None, render_wh=None, device='cuda'):
    """Independent native gsplat render; no full-scene model/checkpoint needed."""
    from gsplat.rendering import rasterization
    data,b = load_static_gaussians(path,camera_to_native,device)
    pose = torch.as_tensor(b['camera_to_native'] if camera_to_native is None else camera_to_native,device=device,dtype=torch.float32)
    intr = torch.as_tensor(b['camera_K'] if K is None else K,device=device,dtype=torch.float32)
    width,height = b['render_wh'] if render_wh is None else render_wh
    rc = b['render_config']
    render,alpha,_ = rasterization(means=data['_means'],quats=data['_quats'],scales=data['_scales'],opacities=data['_opacities'].squeeze(-1),colors=data['_rgbs'],
        viewmats=torch.linalg.inv(pose)[None],Ks=intr[None],width=width,height=height,
        packed=rc['packed'],absgrad=rc['absgrad'],sparse_grad=rc['sparse_grad'],
        rasterize_mode='antialiased' if rc['antialiased'] else 'classic',near_plane=rc['near_plane'],far_plane=rc['far_plane'],
        render_mode='RGB+ED',radius_clip=rc.get('radius_clip',0.))
    raw=render[0,:,:,:3].clamp(max=1.);alpha=alpha[0];affine=torch.as_tensor(b['source_affine'],device=device,dtype=raw.dtype)
    rgb=(torch.einsum('ij,hwj->hwi',affine[:,:3],raw)+alpha*affine[:,3]+(1-alpha)).clamp(0,1)
    return rgb,alpha,render[0,:,:,3:]

class Scene:
    def __init__(self, group):
        item=group['objects'][0]['plan'];checked(item['config'],item['config_sha256']);self.cfg=OmegaConf.load(item['config']);self.data=Path(item['processed_data'])
        self.checkpoint=item['checkpoint'];self.checkpoint_sha=sha(self.checkpoint)
        self.dataset=DrivingDataset(data_cfg=self.cfg.data)
        ds=self.dataset;cfg=self.cfg
        self.trainer=import_str(cfg.trainer.type)(**cfg.trainer,num_timesteps=ds.num_img_timesteps,model_config=cfg.model,num_train_images=len(ds.train_image_set),num_full_images=len(ds.full_image_set),test_set_indices=ds.test_timesteps,scene_aabb=ds.get_aabb().reshape(2,3),device=torch.device('cuda'))
        self.trainer.resume_from_checkpoint(ckpt_path=self.checkpoint,load_only_model=True);self.trainer.set_eval()
        self.model=self.trainer.models['DeformableNodes']
        native=ds.get_init_objects(cur_node_type='DeformableNodes',**OmegaConf.to_container(cfg.model.DeformableNodes.init,resolve=True))
        self.true_to_actor={}
        assert self.model.num_instances==len(native)
        for i,(key,value) in enumerate(native.items()):
            assert torch.allclose(self.model.instances_size[i].cpu(),value['size'].cpu(),atol=1e-5)
            assert torch.equal(self.model.instances_fv[:,i].cpu(),value['frame_info'].cpu())
            self.true_to_actor[int(ds.pixel_source.instances_true_id[key])]=i
        del native
        self.world_from_native=np.loadtxt(self.data/'extrinsics'/f'{cfg.data.start_timestep:03d}_0.txt')
        self.shared=EXPORT/'shared'/self.checkpoint_sha;self.shared.mkdir(parents=True,exist_ok=True)
        shared_path=self.shared/'native_decoder.pt'
        if not shared_path.exists():
            # Only small decoder and source-time/appearance state. No full-scene Gaussian arrays.
            torch.save(dict(deform_network={k:v.detach().cpu() for k,v in self.model.deform_network.state_dict().items()},
                affine={k:v.detach().cpu() for k,v in self.trainer.models['Affine'].state_dict().items()},
                normalized_timestamps=self.model.normalized_timestamps.detach().cpu(),config=OmegaConf.to_container(cfg,resolve=True),
                checkpoint=self.checkpoint,checkpoint_sha256=self.checkpoint_sha),shared_path)
        self.shared_path=shared_path;self.shared_sha=sha(shared_path)
        assert not self.trainer.models['Affine'].pixel_affine,'Per-pixel affine requires a separate explicit adapter'

    @torch.no_grad()
    def process(self,obj):
        sid=obj['id'];plan=obj['plan'];ref=obj['source_reference'];tm=obj['time'];tr=self.trainer;model=self.model
        report=REPORT/sid;out=EXPORT/sid;report.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
        assert shutil.disk_usage(EXPORT).free>10*1024**3
        for k in ['camera_contract','reference_mask','reference_rgb']: checked(ref[k],ref[k+'_sha256'])
        checked(plan['instances_info'],plan['instances_info_sha256'])
        camera=read(ref['camera_contract']);assert camera['timestamp_us']==tm['timestamp_us']
        # Source image equality is checked once per object, independent of output quality.
        checked(plan['source_image'],plan['source_sha256']);source=Path(tm['source_image']);assert sha(source)==ref['reference_rgb_sha256']
        frame,camidx=map(int,source.stem.split('_'));assert frame==tm['source_frame']
        cam_unique=self.dataset.pixel_source.camera_data[camidx].unique_cam_idx
        info,ci=self.dataset.pixel_source.get_image(frame*self.dataset.pixel_source.num_cams+cam_unique)
        info={k:v.cuda() if torch.is_tensor(v) else v for k,v in info.items()};ci={k:v.cuda() if torch.is_tensor(v) else v for k,v in ci.items()}
        pose=np.linalg.inv(self.world_from_native)@np.asarray(camera['T_world_from_camera_opencv'])
        assert np.abs(ci['camera_to_world'].cpu().numpy()-pose).max()<1e-4
        width,height=camera['render_wh'];K=camera['K_raw']
        ci.update(camera_to_world=torch.tensor(pose,device='cuda',dtype=torch.float32),intrinsics=torch.tensor(K,device='cuda',dtype=torch.float32),width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
        info={k:info[k].flatten()[0].expand(height,width) for k in ['img_idx','normed_time','frame_idx']}
        x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy')
        origins,viewdirs,dnorm=get_rays(x.flatten(),y.flatten(),ci['camera_to_world'],ci['intrinsics'])
        info.update(origins=origins.reshape(height,width,3),viewdirs=viewdirs.reshape(height,width,3),direction_norm=dnorm.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
        # Initialize discrete source context, then put all diagnostic renders inside the same target-exposure context.
        tr.cur_frame=torch.tensor(frame,device='cuda')
        assert not tr.in_test_set
        for node in tr.models.values():
            if hasattr(node,'in_test_set'):node.in_test_set=False
            if hasattr(node,'set_cur_frame'):node.set_cur_frame(tr.cur_frame)
        infos=read(plan['instances_info']);true=[int(k) for k,v in infos.items() if v['id']==plan['native_id'].split(':')[-1]]
        assert len(true)==1;actor_index=self.true_to_actor.get(true[0])
        original_actor_state=None
        if actor_index is not None:
            original_actor_state={k:getattr(model,k).detach().cpu()[:,actor_index:actor_index+1].clone() if k in ['instances_fv','instances_trans','instances_quats'] else getattr(model,k).detach().cpu()[actor_index:actor_index+1].clone() for k in ['instances_fv','instances_trans','instances_quats','instances_size','instances_embedding']}
        context=at_physical_exposure(model,actor_index,tm) if actor_index is not None else contextlib.nullcontext(dict(query_timestamp_us=tm['timestamp_us'],policy='Background static; target actor absent; other scene nodes retain native source anchor'))
        mask=torch.as_tensor(np.asarray(Image.open(ref['reference_mask']).convert('L'))>127,device='cuda')
        assert tuple(mask.shape)==(height,width) and bool(mask.any())
        with context as temporal:
            full=tr(info,ci,novel_view=True);cam=tr.process_camera(camera_infos=ci,image_ids=info['img_idx'].flatten()[0],novel_view=True)
            entries=[];all_data={k:[] for k in ['_means','_scales','_quats','_opacities','_rgbs']};offset=0
            for cname in tr.gaussian_classes:
                node=tr.models[cname];g=node.get_gaussians(cam)
                if g is None:continue
                n=len(g['_means']);actor=(node.point_ids.reshape(-1)==actor_index) if cname=='DeformableNodes' and actor_index is not None else torch.zeros(n,device='cuda',dtype=torch.bool)
                entries.append(dict(name=cname,node=node,gs=g,start=offset,end=offset+n,actor=actor));offset+=n
                for k in all_data:all_data[k].append(g[k].detach())
            all_data={k:torch.cat(v) for k,v in all_data.items()};actor_mask=torch.cat([e['actor'] for e in entries])
            background=torch.cat([torch.full((e['end']-e['start'],),e['name']=='Background',device='cuda',dtype=torch.bool) for e in entries])
            rc=dict(near_plane=tr.render_cfg.near_plane,far_plane=tr.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=tr.render_cfg.get('radius_clip',0.))
            # Auxiliary zero-color channel: d sum_M RGB / d color_i = sum_M T_i alpha_i.
            # It bypasses source colors, affine, quality scores and center/box filters.
            with torch.enable_grad():
                probe=torch.zeros((offset,3),device='cuda',requires_grad=True)
                aux=dataclass_gs(**dict(all_data,_rgbs=probe),detach_keys=[],extras=None)
                aux_render,_=tr.render_gaussians(gs=aux,cam=cam,**rc)
                contribution=torch.autograd.grad(aux_render['rgb_gaussians'][...,0][mask].sum(),probe)[0][:,0].detach()
            assert torch.isfinite(contribution).all() and float(contribution.min())>=-1e-7
            selected=actor_mask | (background & (contribution>0))
            assert bool(selected.any()),'No actor or positive Background contribution'
            def native_render(keep):
                sub=dataclass_gs(**{k:v[keep] for k,v in all_data.items()},detach_keys=[],extras=None)
                if not bool(keep.any()):
                    return torch.ones((height,width,3),device='cuda'),torch.zeros((height,width,1),device='cuda'),torch.zeros((height,width,3),device='cuda')
                r,_=tr.render_gaussians(gs=sub,cam=cam,**rc);a=r['opacity'];raw=r['rgb_gaussians'];bias=tr.affine_transformation(torch.zeros_like(raw),info)
                return (tr.affine_transformation(raw,info)-bias*(1-a)+(1-a)).clamp(0,1),a,raw
            candidate,alpha,_=native_render(selected);old,old_alpha,_=native_render(actor_mask)
            without,without_alpha,without_raw=native_render(~actor_mask)
            without_scene=tr.affine_transformation(without_raw+full['rgb_sky']*(1-without_alpha),info).clamp(0,1)
            bg,bga,_=native_render(background)
            affine=tr.models['Affine']({'img_idx':info['img_idx'].flatten()[0].expand(1)})[0].detach().cpu()
            groups=[];sources=[];other_nodes=[]
            for e in entries:
                a,b=e['start'],e['end'];local=selected[a:b];c=contribution[a:b];node=e['node'];n=b-a
                sources.append(dict(node=e['name'],total_gaussians=n,selected=int(local.sum()),positive_visible=int((c>0).sum()),visible_weight=float(c.sum()),retained_visible_weight=float(c[local].sum())))
                if hasattr(node,'point_ids'):
                    ids=node.point_ids.reshape(-1)
                    for nid in torch.unique(ids[c>0]).tolist():
                        if e['name']=='DeformableNodes' and nid==actor_index:continue
                        ix=(ids==nid)&(c>0);other_nodes.append(dict(node=e['name'],instance_index=nid,positive_gaussians=int(ix.sum()),visible_weight=float(c[ix].sum()),included=False))
                if not bool(local.any()):continue
                params={k:getattr(node,k).detach()[local].cpu() for k in ['_means','_scales','_quats','_opacities','_features_dc','_features_rest']}
                group=dict(node=e['name'],source_indices=torch.nonzero(local).flatten().cpu(),canonical=params,
                    posed={k:e['gs'][k][local].detach().cpu() for k in ['_means','_scales','_quats','_opacities']},
                    sh_degree=int(node.sh_degree),active_sh_degree=min(int(tr.step)//int(node.ctrl_cfg.sh_degree_interval),int(node.sh_degree)))
                if e['name']=='DeformableNodes':
                    assert actor_index is not None
                    group['original_point_ids']=node.point_ids[local].cpu()
                    group['native_actor_state']=original_actor_state
                    group['native_actor_state_policy']='Unmodified original checkpoint trajectory and embedding; posed geometry separately records the source exposure query'
                groups.append(group)
            bundle=dict(schema='omnire_visible_background_static_v1',id=sid,groups=groups,T_world_from_native=self.world_from_native.tolist(),
                camera_to_native=cam.camtoworlds.detach().cpu().tolist(),camera_K=cam.Ks.detach().cpu().tolist(),render_wh=[width,height],
                source_affine=affine,render_config=OmegaConf.to_container(tr.render_cfg,resolve=True),timestamp_us=tm['timestamp_us'],
                checkpoint=self.checkpoint,checkpoint_sha256=self.checkpoint_sha,source_reference=ref,temporal_query=temporal,
                native_decoder=str(self.shared_path),native_decoder_sha256=self.shared_sha,
                dynamic_support='Native actor state retained, supplemented Background is static. Bundle renders fixed exposure only; no fabricated root/articulation for Background.')
            bundle_path=out/'asset.pt';torch.save(bundle,bundle_path)
            re_rgb,re_alpha,_=render_bundle(bundle_path)
            errors=dict(rgb_max=float((candidate-re_rgb).abs().max()),alpha_max=float((alpha-re_alpha).abs().max()))
            # The RGB path differs only in equivalent affine matrix application.
            assert errors['rgb_max']<2e-5 and errors['alpha_max']<1e-6,errors
            np.savez_compressed(report/'contribution.npz',visible_weight=contribution.cpu().numpy(),selected=selected.cpu().numpy(),actor=actor_mask.cpu().numpy(),background=background.cpu().numpy())
            image(candidate,out/'actor_source.png');image(alpha,out/'actor_alpha.png')
            for name,tensor in [('full_scene.png',full['rgb']),('without_actor.png',without_scene),('actor_only.png',old),('background_only.png',bg)]:image(tensor,report/name)
            old_difference=None
            prior=obj['prior_registration']
            if prior:
                checked(prior['source_rgb'],prior['source_rgb_sha256'])
                old_uint=np.clip(np.rint(old.cpu().numpy()*255),0,255).astype(np.uint8)
                previous=np.asarray(Image.open(prior['source_rgb']).convert('RGB'))
                old_difference=dict(max_byte=float(np.abs(old_uint.astype(float)-previous).max()),mean_byte=float(np.abs(old_uint.astype(float)-previous).mean()))
            total=float(contribution.sum());retained=float(contribution[selected].sum());bg_total=float(contribution[background].sum())
            record=dict(id=sid,method='omnire',status='passed',source_reference=ref,identity_key=ref['identity_key'],camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],checkpoint=self.checkpoint,checkpoint_sha256=self.checkpoint_sha,
                bundle=str(bundle_path),bundle_sha256=sha(bundle_path),actor_preview=str(out/'actor_source.png'),actor_preview_sha256=sha(out/'actor_source.png'),
                actor_alpha=str(out/'actor_alpha.png'),actor_alpha_sha256=sha(out/'actor_alpha.png'),prediction_encoding='white_composited_uint8',render_wh=[width,height],
                source_timestamp_us=tm['timestamp_us'],timestamp_us=tm['timestamp_us'],T_world_from_native=self.world_from_native.tolist(),
                camera_to_native=cam.camtoworlds.cpu().tolist(),camera_K=cam.Ks.cpu().tolist(),source_frame=frame,source_image_index=int(info['img_idx'].flatten()[0]),
                actor_node_present=actor_index is not None,actor_index=actor_index,original_actor_gaussians=int(actor_mask.sum()),exported_gaussians=int(selected.sum()),
                background_added_gaussians=int((selected&background).sum()),change='background_support_added' if bool((selected&background).any()) else 'original_actor_reused',
                change_reason='Uniform positive native Background contribution in frozen mask; original actor kept entirely; other actor nodes not automatically merged',
                source_support_status='native_contribution_retained_not_semantic_segmentation',semantic_review_required=True,reload_verified=True,reload_errors=errors,
                original_actor_reload_difference=old_difference,gaussian_sources=sources,other_node_contributions=other_nodes,
                contributions=dict(total_visible_weight=total,retained_visible_weight=retained,retained_fraction=retained/total if total else None,
                    background_visible_weight=bg_total,background_retained_fraction=1. if bg_total else None,other_node_weight=total-retained,
                    gradient_semantics='d sum(masked auxiliary zero-color native RGB)/d auxiliary per-Gaussian color = summed native T*alpha; no image/opacity fit'),
                target_alpha_mean_in_reference_mask=float(alpha[mask].mean()),target_alpha_over_half_reference_fraction=float((alpha[mask]>.5).float().mean()),
                temporal_query=temporal,other_actor_time_policy='Other root poses retain native source anchor; DeformableNodes deformation clock shares the target exposure query; all comparisons use this identical context',dynamic_claim='static exposure snapshot only; original actor dynamics retained separately; Background has no invented motion',
                source_mask_applied_to_prediction=False,script=str(Path(__file__)),script_sha256=sha(__file__),native_code_sha256={str(p):sha(p) for p in [REPO/'models/trainers/base.py',REPO/'models/nodes/deformable.py']},
                full_scene_preview=str(report/'full_scene.png'),comparison=str(report/'comparison.png'),prior_registration=prior,
                selection_sha256=read(REPORT/'plan.json')['selection_sha256'],scores_produced=False)
            save(out/'validation.json',record);save(report/'validation.json',record)
            tiles=[('GT',ref['reference_rgb']),('Full scene',report/'full_scene.png'),('Without actor',report/'without_actor.png'),('Actor only',report/'actor_only.png'),('Background',report/'background_only.png'),('New export',out/'actor_source.png')]
            import matplotlib
            font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),22)
            sheet=Image.new('RGB',(len(tiles)*320,364),'white');draw=ImageDraw.Draw(sheet)
            for col,(label,path) in enumerate(tiles):
                sheet.paste(Image.open(path).convert('RGB').crop(tuple(ref['roi_xyxy'])).resize((320,320)),(col*320,44));draw.text((col*320+8,10),label,fill='black',font=font)
            sheet.save(report/'comparison.png')
        print(json.dumps(dict(id=sid,status='passed',selected=int(selected.sum()),added=int((selected&background).sum()),reload=errors)),flush=True)
        return record

def run(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    plan=read(REPORT/'plan.json');ids=set(args.ids or [])
    groups=[g for g in plan['groups'] if (args.shard is None or g['shard']==args.shard) and (not ids or any(o['id'] in ids for o in g['objects']))]
    progress=REPORT/f'worker_{args.shard if args.shard is not None else "pilot"}.json';done=[];failures=[]
    for group in groups:
        pending=[o for o in group['objects'] if (not ids or o['id'] in ids) and not (EXPORT/o['id']/'validation.json').exists()]
        if not pending:continue
        save(progress,dict(status='loading_scene',pid=os.getpid(),checkpoint=group['checkpoint'],done=done,failures=failures,updated_at=time.time()))
        try:scene=Scene(group)
        except Exception as ex:
            for o in pending:
                err=dict(id=o['id'],method='omnire',status='failed',reason=str(ex),traceback=traceback.format_exc(),checkpoint=group['checkpoint']);save(REPORT/o['id']/'failure.json',err);failures.append(o['id'])
            continue
        for obj in pending:
            try:scene.process(obj);done.append(obj['id'])
            except Exception as ex:
                err=dict(id=obj['id'],method='omnire',status='failed',reason=str(ex),traceback=traceback.format_exc(),checkpoint=group['checkpoint']);save(REPORT/obj['id']/'failure.json',err);failures.append(obj['id']);print(json.dumps(err),flush=True)
            save(progress,dict(status='running',pid=os.getpid(),checkpoint=group['checkpoint'],done=done,failures=failures,updated_at=time.time()))
        del scene;gc.collect();torch.cuda.empty_cache()
    save(progress,dict(status='completed',pid=os.getpid(),done=done,failures=failures,updated_at=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--shard',type=int,choices=[0,1]);parser.add_argument('--ids',nargs='+');args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run(args)
