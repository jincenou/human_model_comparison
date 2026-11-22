"""Additional diagnostics only: non-target nodes and all-root exposure timing."""
from contextlib import contextmanager, nullcontext
import json,gc
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import matplotlib
from native_reaudit_omnire_v1 import H,REPORT,EXPORT,Scene,read,save,image,dataclass_gs,get_rays,at_physical_exposure
from models.gaussians.basics import interpolate_quats

TIMING={'HN0387','HN0345','HN0272'}
OUT=REPORT/'semantic_node_timing'

@contextmanager
def all_node_exposure(tr,tm):
    anchor=tm['source_frame'];br=tm['continuous_bracket'];lo,hi,f=br['lo'],br['hi'],br['fraction'];saved=[];details=[]
    try:
        for name in tr.gaussian_classes:
            node=tr.models[name]
            if not hasattr(node,'instances_trans'):continue
            times=node.normalized_timestamps;trans=node.instances_trans[anchor].detach().clone();quats=node.instances_quats[anchor].detach().clone()
            valid=node.instances_fv[lo]&node.instances_fv[hi]
            saved.append((node,times,trans,quats))
            node.normalized_timestamps=times.clone();node.normalized_timestamps[anchor]=(1-f)*times[lo]+f*times[hi]
            translation=(1-f)*node.instances_trans[lo,valid].detach().clone()+f*node.instances_trans[hi,valid].detach().clone()
            quaternion=interpolate_quats(node.instances_quats[lo,valid].detach().clone(),node.instances_quats[hi,valid].detach().clone(),f)
            node.instances_trans[anchor,valid]=translation;node.instances_quats[anchor,valid]=quaternion
            details.append(dict(node=name,interpolated_instances=int(valid.sum()),unbracketed_instances=int((~valid).sum()),max_translation_change_m=float((node.instances_trans[anchor]-trans).norm(dim=-1).max())))
        yield details
    finally:
        for node,times,trans,quats in saved:
            node.normalized_timestamps=times;node.instances_trans[anchor].copy_(trans);node.instances_quats[anchor].copy_(quats)

@torch.no_grad()
def process(scene,obj,emit_other_previews=True,correct_exposure=False,output=None):
    sid=obj['id'];folder=(output or OUT)/sid;folder.mkdir(parents=True,exist_ok=True)
    v=read(EXPORT/sid/'validation.json');ref=v['source_reference'];tm=obj['time'];tr=scene.trainer;model=scene.model;camera=read(ref['camera_contract']);frame=tm['source_frame'];camidx=tm['camera_index']
    unique=scene.dataset.pixel_source.camera_data[camidx].unique_cam_idx;info,ci=scene.dataset.pixel_source.get_image(frame*scene.dataset.pixel_source.num_cams+unique)
    info={k:x.cuda() if torch.is_tensor(x) else x for k,x in info.items()};ci={k:x.cuda() if torch.is_tensor(x) else x for k,x in ci.items()}
    pose=np.linalg.inv(scene.world_from_native)@np.asarray(camera['T_world_from_camera_opencv']);width,height=camera['render_wh']
    ci.update(camera_to_world=torch.tensor(pose,device='cuda',dtype=torch.float32),intrinsics=torch.tensor(camera['K_raw'],device='cuda',dtype=torch.float32),width=torch.tensor(width,device='cuda'),height=torch.tensor(height,device='cuda'))
    info={k:info[k].flatten()[0].expand(height,width) for k in ['img_idx','normed_time','frame_idx']}
    x,y=torch.meshgrid(torch.arange(width,device='cuda'),torch.arange(height,device='cuda'),indexing='xy');o,d,n=get_rays(x.flatten(),y.flatten(),ci['camera_to_world'],ci['intrinsics'])
    info.update(origins=o.reshape(height,width,3),viewdirs=d.reshape(height,width,3),direction_norm=n.reshape(height,width,1),pixel_coords=torch.stack([y/height,x/width],dim=-1))
    tr.cur_frame=torch.tensor(frame,device='cuda')
    for node in tr.models.values():
        if hasattr(node,'in_test_set'):node.in_test_set=False
        if hasattr(node,'set_cur_frame'):node.set_cur_frame(tr.cur_frame)
    actor_index=v['actor_index'];mask=torch.tensor(np.asarray(Image.open(ref['reference_mask']).convert('L'))>127,device='cuda')
    def collect():
        full=tr(info,ci,novel_view=True);cam=tr.process_camera(camera_infos=ci,image_ids=info['img_idx'].flatten()[0],novel_view=True)
        parts=[];starts={};offset=0
        for name in tr.gaussian_classes:
            g=tr.models[name].get_gaussians(cam)
            if g is None:continue
            count=len(g['_means']);starts[name]=(offset,offset+count);offset+=count;parts.append(g)
        data={k:torch.cat([g[k].detach() for g in parts]) for k in parts[0]}
        return full,cam,data,starts
    rc=dict(near_plane=tr.render_cfg.near_plane,far_plane=tr.render_cfg.far_plane,render_mode='RGB+ED',radius_clip=tr.render_cfg.get('radius_clip',0.))
    def render(data,cam,keep):
        gs=dataclass_gs(**{k:x[keep] for k,x in data.items()},detach_keys=[],extras=None);r,_=tr.render_gaussians(gs=gs,cam=cam,**rc);a=r['opacity'];raw=r['rgb_gaussians'];bias=tr.affine_transformation(torch.zeros_like(raw),info)
        return (tr.affine_transformation(raw,info)-bias*(1-a)+(1-a)).clamp(0,1)
    with at_physical_exposure(model,actor_index,tm) if actor_index is not None else nullcontext():
        old,cam,data,starts=collect();image(old['rgb'],folder/'target_only_time_full.png')
        previews=[]
        for j,node in enumerate(v['other_node_contributions'] if emit_other_previews else []):
            keep=torch.zeros(len(data['_means']),dtype=torch.bool,device='cuda');a,b=starts[node['node']];keep[a:b]=tr.models[node['node']].point_ids.reshape(-1)==node['instance_index'];rgb=render(data,cam,keep)
            path=folder/f'other_node_{j:02d}.png';image(rgb,path);previews.append(dict(**node,preview=str(path)))
        olddata={k:x.clone() for k,x in data.items()}
    result=dict(id=sid,other_nodes=previews,source_reference=ref,all_node_time_audit=None)
    if sid in TIMING or correct_exposure:
        with all_node_exposure(tr,tm) as changes:
            corrected,cam,newdata,newstarts=collect();assert newstarts==starts;image(corrected['rgb'],folder/'all_node_time_full.png')
            a,b=starts['DeformableNodes'];actor=model.point_ids.reshape(-1)==actor_index if actor_index is not None else torch.zeros(b-a,dtype=torch.bool,device='cuda')
            target_errors={k:float((olddata[k][a:b][actor]-newdata[k][a:b][actor]).abs().max()) if actor.any() else 0. for k in newdata};assert max(target_errors.values())<1e-5,target_errors
            with torch.enable_grad():
                probe=torch.zeros_like(newdata['_rgbs'],requires_grad=True);gs=dataclass_gs(**dict(newdata,_rgbs=probe),detach_keys=[],extras=None)
                aux,_=tr.render_gaussians(gs=gs,cam=cam,**rc);weight=torch.autograd.grad(aux['rgb_gaussians'][...,0][mask].sum(),probe)[0][:,0].detach()
            oldselection=torch.tensor(np.load(REPORT/sid/'contribution.npz')['selected'],device='cuda');bg=torch.zeros_like(oldselection);a,b=starts['Background'];bg[a:b]=True
            target=torch.zeros_like(oldselection);a,b=starts['DeformableNodes'];target[a:b]=actor
            changed_selection=target|(bg&(weight>0));missing=bg&~oldselection&(weight>0)
            candidate=render(newdata,cam,changed_selection);original=render(newdata,cam,oldselection)
            image(candidate,folder/'new_selection_candidate.png')
            independent_delta=(candidate-original).abs()
            delta=(old['rgb'].clamp(0,1)-corrected['rgb'].clamp(0,1)).abs()
            result['all_node_time_audit']=dict(node_changes=changes,target_gaussian_parameter_max_errors=target_errors,
                full_rgb_mask_mae=float(delta[mask].mean()),full_rgb_mask_max=float(delta[mask].max()),
                corrected_background_visible_weight=float(weight[bg].double().sum()),
                background_weight_omitted_by_old_selection=float(weight[missing].double().sum()),
                independent_selection_mask_rgb_mae=float(independent_delta[mask].mean()),
                independent_selection_mask_rgb_max=float(independent_delta[mask].max()),
                independent_selection_mask_fraction_channels_over_1byte=float((independent_delta[mask]>1/255).float().mean()),
                independent_selection_all_rgb_mae=float(independent_delta.mean()),
                current_bundle_rgb_unchanged=True,
                background_missing_gaussians=int(missing.sum()),selection_added=int((changed_selection&~oldselection).sum()),selection_removed=int((oldselection&~changed_selection).sum()),
                policy='Interpolate every bracket-valid node root and node deformation clock at frozen physical exposure; unbracketed instances stay at source anchor; diagnostic only, existing passed exports unchanged')
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),18)
    tiles=[('Source',ref['reference_rgb']),('Current full',folder/'target_only_time_full.png'),('Current new asset',v['actor_preview'])]
    if sid in TIMING or correct_exposure:tiles += [('All roots at exposure',folder/'all_node_time_full.png'),('Corrected selection',folder/'new_selection_candidate.png')]
    for n in previews:tiles.append((n['node']+' '+str(n['instance_index']),n['preview']))
    cols=min(5,len(tiles));rows=(len(tiles)+cols-1)//cols;sheet=Image.new('RGB',(cols*256,rows*286),'white');draw=ImageDraw.Draw(sheet)
    for j,(label,path) in enumerate(tiles):
        x=(j%cols)*256;y=(j//cols)*286;draw.text((x+4,y+4),label,font=font,fill='black');sheet.paste(Image.open(path).convert('RGB').crop(tuple(ref['roi_xyxy'])).resize((256,256)),(x,y+30))
    sheet.save(folder/'comparison.png');save(folder/'audit.json',result);print(sid,result['all_node_time_audit'],flush=True)

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;OUT.mkdir(parents=True,exist_ok=True)
    ids={r['id'] for r in read(REPORT/'completion_audit.json')['other_node_fraction_over_5pct']};plan=read(REPORT/'plan.json')
    for group in plan['groups']:
        todo=[o for o in group['objects'] if o['id'] in ids]
        if not todo:continue
        scene=Scene(group)
        for obj in todo:process(scene,obj)
        del scene;gc.collect();torch.cuda.empty_cache()
    save(OUT/'completion.json',dict(status='completed',ids=sorted(ids),timing_ids=sorted(TIMING),original_outputs_modified=False))

if __name__=='__main__':main()
