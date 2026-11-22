"""CPU-only reference feasibility audit. No model loading or scoring, no frozen gates."""
from human_asset_benchmark.paths import expand as _hmc_expand
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

D=Path(__file__).resolve().parents[1];H=D.parents[1];R=H.parent/'vehicle_model_comparison'
P=H/'handoff/human_static_cd_fscore_time_research_20261114'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rot(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def object_points(world,a):return (world-np.asarray(a['translation']))@rot(a['rotation'])
def inside(world,a):
    local=object_points(world,a);w,l,h=a['size']
    return (np.abs(local)<=np.asarray([l,w,h])/2+1e-8).all(1)
def main():
    active=read(H/'inputs/active_human_benchmark.json');selected=read(active['selection'])
    assert sha(active['selection'])==active['selection_sha256']
    byid={x['id']:x for x in selected['objects']};prior=read(P/'evidence/reference_metadata_audit.json')
    classes=R/'evaluation_cd_fscore_time_20261112/references/lidarseg_extension/v1.0-trainval/category.json'
    label_index={x['name']:x['index'] for x in read(classes)}
    cache={};rows=[];metadata_sources={}
    for r in prior['rows']:
        sid=r['id'];assert byid[sid]['t_ref_us']==r['camera_timestamp_us']
        root=Path(r['metadata_root']);key=str(root)
        if key not in cache:
            tables={}
            for name in ['sample_data','sample_annotation','calibrated_sensor','ego_pose','instance','category']:
                path=root/(name+'.json');metadata_sources[str(path)]=sha(path)
                tables[name]={x['token']:x for x in read(path)}
            cache[key]=tables
        t=cache[key];sd=t['sample_data'][r['same_sample_lidar_token']];cam=t['sample_data'][r['camera_token']]
        assert sd['is_key_frame'] and sd['sample_token']==r['sample_token']==cam['sample_token']
        assert abs(sd['timestamp']-cam['timestamp'])/1000==abs(r['same_sample_lidar_minus_camera_us'])/1000
        path=root.parent/sd['filename']
        if not path.exists():path=Path(_hmc_expand('${NUSCENES_ROOT}'))/sd['filename']
        raw=np.fromfile(path,dtype=np.float32).reshape(-1,5);labels=np.fromfile(r['same_sample_lidar_label_path'],dtype=np.uint8)
        assert len(labels)==len(raw)
        cs=t['calibrated_sensor'][sd['calibrated_sensor_token']];ego=t['ego_pose'][sd['ego_pose_token']]
        world=(raw[:,:3].astype(np.float64)@rot(cs['rotation']).T+cs['translation'])@rot(ego['rotation']).T+ego['translation']
        ann=t['sample_annotation'][r['annotation_token']];assert ann['instance_token']==r['instance_token'] and ann['sample_token']==sd['sample_token']
        category=t['category'][t['instance'][ann['instance_token']]['category_token']]['name']
        boxed=inside(world,ann);matching=labels==label_index[category];candidate=boxed&matching
        ix=np.flatnonzero(candidate);ambiguous=np.zeros(len(ix),dtype=bool)
        for a in t['sample_annotation'].values():
            if a['sample_token']!=ann['sample_token'] or a['token']==ann['token']:continue
            cname=t['category'][t['instance'][a['instance_token']]['category_token']]['name']
            if cname==category:ambiguous|=inside(world[ix],a)
        retained=ix[~ambiguous];local=object_points(world[retained],ann)
        vox=np.floor(local/.05).astype(np.int64);unique_n=len(np.unique(vox,axis=0))
        # Report support distribution only. Identity/pose/accessory correspondence
        # and temporal effects still need review in the executing conversation.
        rows.append(dict(id=sid,scene=r['scene'],instance_token=ann['instance_token'],category=category,
            camera_timestamp_us=cam['timestamp'],scan_timestamp_us=sd['timestamp'],delta_ms=(sd['timestamp']-cam['timestamp'])/1000,
            scan_token=sd['token'],scan_path=str(path),scan_sha256=sha(path),label_path=r['same_sample_lidar_label_path'],label_sha256=sha(r['same_sample_lidar_label_path']),
            annotation_token=ann['token'],annotation_raw_num_lidar_pts=ann['num_lidar_pts'],geometric_box_points=int(boxed.sum()),
            semantic_box_points=int(candidate.sum()),same_category_box_ambiguous_points=int(ambiguous.sum()),
            retained_raw_points=len(retained),object_frame_5cm_voxels=unique_n,reference_only_20_voxel_candidate=unique_n>=20,
            retained_point_indices=retained.tolist(),reference_qualified=False,qualification_status='pending_temporal_identity_scope_visual_review'))
    assert len(rows)==100 and set(x['id'] for x in rows)==set(byid)
    ns=[x['object_frame_5cm_voxels'] for x in rows]
    summary=dict(objects=100,scenes=len(set(x['scene'] for x in rows)),all_scan_and_label_lengths_matched=True,
        annotation_raw_box_count_exact_matches=sum(r['annotation_raw_num_lidar_pts']==r['geometric_box_points'] for r in rows),
        candidate_at_least20=sum(n>=20 for n in ns),below20=sum(n<20 for n in ns),zero=sum(n==0 for n in ns),
        voxel_min=min(ns),voxel_median=float(np.median(ns)),voxel_max=max(ns),points_removed_for_overlapping_same_category_boxes=sum(x['same_category_box_ambiguous_points'] for x in rows))
    out=dict(status='reference_support_audit_complete_not_scoring_qualification',scope='Read-only CPU point/label filtering and density audit. No CD/F1, no Time run, no model render, no final protocol freeze.',
        selection_sha256=active['selection_sha256'],summary=summary,rows=rows,metadata_sources=metadata_sources,category_table=dict(path=str(classes),sha256=sha(classes)),script_sha256=sha(__file__),
        rules=dict(scan='same-sample labelled keyframe only',label='exact annotation category index',point_attribution='inside target oriented box and no other same-category box',voxel_grid='floor in annotation-box coordinates / 0.05 m; count unique cells',threshold='20 is candidate feasibility only; no prediction results consulted'),
        limits=['Box and semantic labels are not independent point-instance ground truth.','Annotation pose is at sample time; scan duration and camera timestamp differ.','The voxel support count is not an approved final eligibility count.','No sensor-beam return query or metric has been executed.'])
    (D/'evidence').mkdir(exist_ok=True)
    (D/'evidence/reference_support_readonly.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
