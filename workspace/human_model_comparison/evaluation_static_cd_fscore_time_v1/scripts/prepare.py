"""Rebuild reference rays from raw scans; bind intake without importing 2D gates."""
import itertools
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from common import *

def rot(q): return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def transform(cs, ego):
    T = np.eye(4); T[:3,:3] = rot(ego['rotation']) @ rot(cs['rotation'])
    T[:3,3] = rot(ego['rotation']) @ cs['translation'] + ego['translation']
    return T
def local(world,a): return (world-a['translation']) @ rot(a['rotation'])
def inside(world,a): return (np.abs(local(world,a)) <= np.array(a['size'])[[1,0,2]]/2+1e-8).all(1)

def main():
    active, selected = selection()
    ap = read(H/'inputs/active_results.json')
    old = read(H/ap['core_summary']); oldrows = {(r['id'],r['method']):r for r in old['rows']}
    support = read(D/'evidence/reference_support_readonly.json')
    metadata = {r['id']:r for r in read(D/'evidence/prior_reference_metadata_audit.json')['rows']}
    categories = read(support['category_table']['path']); checked(support['category_table']['path'],support['category_table']['sha256'])
    label_index = {r['name']:r['index'] for r in categories}
    cache = {}; refs = []; jobs = []; source_bindings = {}; tiles=[]
    for r in support['rows']:
        sid = r['id']; s = selected[sid]; meta = metadata[sid]; root = Path(meta['metadata_root'])
        if str(root) not in cache:
            tables={}
            for name in ['sample_data','sample_annotation','calibrated_sensor','ego_pose','instance','category']:
                p=root/(name+'.json'); checked(p,support['metadata_sources'][str(p)]);source_bindings[str(p)]=sha(p)
                tables[name]={x['token']:x for x in read(p)}
            cache[str(root)] = tables
        t=cache[str(root)]; sd=t['sample_data'][r['scan_token']]; cam_sd=t['sample_data'][meta['camera_token']]
        ann=t['sample_annotation'][r['annotation_token']]
        assert sd['is_key_frame'] and sd['sample_token']==cam_sd['sample_token']==ann['sample_token']
        assert s['identity_key']=='nuscenes:'+ann['instance_token'] and s['t_ref_us']==cam_sd['timestamp']==r['camera_timestamp_us']
        assert abs(sd['timestamp']-s['t_ref_us'])<=50000
        for k in ['scan','label']: checked(r[k+'_path'],r[k+'_sha256'])
        raw=np.fromfile(r['scan_path'],np.float32).reshape(-1,5);labels=np.fromfile(r['label_path'],np.uint8);assert len(raw)==len(labels)
        T=transform(t['calibrated_sensor'][sd['calibrated_sensor_token']],t['ego_pose'][sd['ego_pose_token']])
        world=raw[:,:3].astype(float)@T[:3,:3].T+T[:3,3]
        ix=np.flatnonzero(inside(world,ann)&(labels==label_index[r['category']]))
        ambiguous=np.zeros(len(ix),bool)
        for a in t['sample_annotation'].values():
            if a['sample_token']!=ann['sample_token'] or a['token']==ann['token']:continue
            category=t['category'][t['instance'][a['instance_token']]['category_token']]['name']
            if category==r['category']:ambiguous |= inside(world[ix],a)
        ix=ix[~ambiguous];assert ix.tolist()==r['retained_point_indices']
        vox=np.floor(local(world[ix],ann)/.05).astype(np.int64)
        _,first=np.unique(vox,axis=0,return_index=True);ix=ix[np.sort(first)]
        assert len(ix)==r['object_frame_5cm_voxels']
        q=world[ix];o=np.broadcast_to(T[:3,3],q.shape).copy();ranges=np.linalg.norm(q-o,axis=1);u=(q-o)/ranges[:,None]
        np.testing.assert_allclose(q,o+ranges[:,None]*u,atol=1e-10)
        qp=E/'references/queries'/f'{sid}.npz';qp.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(q=q,o=o,u=u,r=ranges,raw_index=ix,frame_index=np.zeros(len(ix),np.int64),T_world_from_sensor=T,
                    scan_token=np.array(sd['token']),scan_timestamp_us=np.array(sd['timestamp']),t_ref_us=np.array(s['t_ref_us']))
        if qp.exists():
            with np.load(qp) as z:
                for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
        else:np.savez_compressed(qp,**arrays)
        ref=oldrows[sid,'ours']['reference']; checked(ref['camera_contract'],ref['camera_contract_sha256']);camera=read(ref['camera_contract'])
        Twc=transform(t['calibrated_sensor'][cam_sd['calibrated_sensor_token']],t['ego_pose'][cam_sd['ego_pose_token']])
        np.testing.assert_allclose(Twc,camera['T_world_from_camera_opencv'],atol=1e-8)
        corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.array(ann['size'])[[1,0,2]]/2
        corners=corners@rot(ann['rotation']).T+ann['translation'];height=float(np.ptp(corners[:,2]))
        qc=(q-Twc[:3,3])@Twc[:3,:3];pixels=qc@np.array(camera['K_raw']).T;pixels=pixels[:,:2]/pixels[:,2:]
        main=len(ix)>=20
        item=dict(id=sid,identity_key=s['identity_key'],t_ref_us=s['t_ref_us'],scan_timestamp_us=sd['timestamp'],delta_ms=(sd['timestamp']-s['t_ref_us'])/1000,
                  scan_token=sd['token'],scan_path=r['scan_path'],scan_sha256=r['scan_sha256'],label_path=r['label_path'],label_sha256=r['label_sha256'],
                  annotation_token=ann['token'],annotation=ann,annotation_vertical_extent_m=height,category=r['category'],attribute_names=meta['attribute_names'],
                  queries=str(qp),queries_sha256=sha(qp),query_count=len(ix),retained_raw_points=r['retained_raw_points'],ambiguous_removed=r['same_category_box_ambiguous_points'],
                  camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],reference_rgb=ref['reference_rgb'],reference_rgb_sha256=ref['reference_rgb_sha256'],
                  candidate_panel='main' if main else 'low_support_exploratory',reference_status='pending_scope_visual_review',reference_qualified=False,
                  prior_visual_note=ref.get('visual_note'),query_reconstruction_max_m=float(np.max(abs(q-o-ranges[:,None]*u))),nominal_scan_rays=True)
        refs.append(item)
        checked(ref['reference_rgb'],ref['reference_rgb_sha256'])
        l,top,right,bottom=camera['roi_xyxy'];im=Image.open(ref['reference_rgb']).convert('RGB').crop((l,top,right,bottom)).resize((320,320))
        draw=ImageDraw.Draw(im)
        for px,py in pixels:
            x=(px-l)*320/(right-l);y=(py-top)*320/(bottom-top);draw.ellipse((x-2,y-2,x+2,y+2),fill='#00ffff',outline='black')
        tile=Image.new('RGB',(320,355),'white');tile.paste(im,(0,35));ImageDraw.Draw(tile).text((4,3),f'{sid} n={len(ix)} dt={item["delta_ms"]:.1f}ms',fill='black')
        ImageDraw.Draw(tile).text((4,19),';'.join(meta['attribute_names']),fill='black');tiles.append((sid,tile))
        for method in METHODS:
            previous=oldrows[sid,method]
            job=dict(id=sid,method=method,identity_key=s['identity_key'],t_ref_us=s['t_ref_us'],reference_query_sha256=item['queries_sha256'],reference_panel=item['candidate_panel'],
                     camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],prior_2d_status=previous['status'],prior_2d_reason=previous.get('reason'),
                     geometry_status='queued',geometry_gate_inherits_2d=False,registration_status='pending',depth_status='pending',asset_available=False)
            if previous['status']=='scored':
                checked(previous['result'],previous['result_sha256']);result=read(previous['result']);cp=result['registration_record'];checked(cp,result['registration_sha256']);cert=read(cp)
                job.update(source_certificate=cp,source_certificate_sha256=sha(cp),registration_status='existing_source_relation_pending_metric_scale')
                if method in METHODS[:4]:
                    trace=read(cert['trace_record']);checked(cert['trace_record'],cert['trace_record_sha256'])
                    pk='snapshot' if method=='ours' else 'native_ply' if method=='ah' else 'native_payload'
                    checked(trace[pk],trace[pk+'_sha256']);job.update(asset=trace[pk],asset_sha256=trace[pk+'_sha256'],asset_available=True,
                        source_trace=cert['trace_record'],source_trace_sha256=cert['trace_record_sha256'],T_source_camera_from_native=trace['T_source_camera_from_export'])
                else:
                    checked(cert['render_validation'],cert['render_validation_sha256']);v=read(cert['render_validation'])
                    job.update(asset_available=True,scene_render_validation=cert['render_validation'],scene_render_validation_sha256=sha(cert['render_validation']),scene_redecode_status='pending')
            elif method in ['trellis_original','trellis2']:
                asset=H/'runs'/f'{method}_human_native_v1'/sid/('native_gaussians.pt' if method=='trellis_original' else 'native_voxel_mesh.pt')
                if asset.exists():job.update(asset=str(asset),asset_sha256=sha(asset),asset_available=True,registration_status='recoverable_gap_audit_pending')
                else:job.update(geometry_status='pending_asset_failure_evidence',registration_status='asset_missing')
            else:job['geometry_status']='pending_bounded_scene_extraction_audit'
            jobs.append(job)
    out=E/'references/review';out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(tiles),20):
        sheet=Image.new('RGB',(5*320,4*355),'#eeeeee')
        subset=tiles[start:start+20]
        for i,(_,tile) in enumerate(subset):sheet.paste(tile,((i%5)*320,(i//5)*355))
        sheet.save(out/f'sheet_{start//20+1:02d}.jpg',quality=95)
    immutable(E/'intake.json',dict(selection_sha256=SELECTION_SHA,rows=jobs,sources={str(H/'inputs/active_human_benchmark.json'):sha(H/'inputs/active_human_benchmark.json'),str(H/ap['core_summary']):sha(H/ap['core_summary'])}))
    immutable(E/'references/reference_candidates.json',dict(selection_sha256=SELECTION_SHA,rows=refs,metadata_sources=source_bindings,
         rules=dict(voxel_m=.05,voxel_representative='first raw-index real return',minimum_main_queries=20,max_delta_ms=50,scan='same-sample labelled keyframe',gravity='world +Z')))
    csv_write(E/'geometry_ledger.csv',jobs)
    print('References',dict(Counter(r['candidate_panel'] for r in refs)),'ledger',len(jobs),'available',dict(Counter(r['method'] for r in jobs if r['asset_available'])))

if __name__=='__main__':main()
