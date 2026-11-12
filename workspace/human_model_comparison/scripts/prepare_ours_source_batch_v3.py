"""Trace existing source-pose registration; no RGB/mask optimization or reposing."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from human_source_metrics_v1 import sha

H=Path(__file__).resolve().parents[1]
OUT=H/'configs/human_source_common_v1/ours_registration_batch_v3'

def prepare_row(sid,selected):
    row=selected[sid];paths=row['paths'];production=Path(paths['production_config'])
    cfg=json.loads(production.read_text());manifest=Path(cfg['provenance']['manifest'])
    assert sha(manifest)==cfg['provenance']['manifest_sha256']
    data=json.loads(manifest.read_text());base=Path(paths['native_metadata']).parent
    rootpath=base.parent/'conditions/preflight/root_motion.npz'
    rootmeta=base.parent/'conditions/research/root_local.json'
    assert sha(rootpath)==json.loads(rootmeta.read_text())['root_motion_sha256']
    kppath=manifest.parent/'conditioning/keypoints_3d.npy'
    projection=base/'motion_pair/calibrated_moving/projection.npz'
    mapping=json.loads(Path(paths['frame_mapping']).read_text())
    with np.load(rootpath,allow_pickle=False) as z:
        times=z['frame_times'];transforms=z['local_to_smpl_incam']
    kp=np.load(kppath,allow_pickle=False)
    with np.load(projection,allow_pickle=False) as z:observed=z['keypoints_uv']
    assert len(kp)==len(times)==len(mapping['frames'])==len(observed)
    endpoint_errors=[]
    for i in range(len(times)):
        assert abs(times[i]-mapping['frames'][i]['source_relative_exposure_seconds'])<1e-6
        xyz=kp[i]@transforms[i,:3,:3].T+transforms[i,:3,3]
        pixels=xyz@np.asarray(mapping['frames'][i]['K_input']).T
        uv=pixels[:,:2]/pixels[:,2:]
        endpoint_errors.append(float(np.max(np.abs(uv-observed[i]))))
    assert max(endpoint_errors)<.02, (sid,max(endpoint_errors))
    t=row['reference_seconds_from_first_native'];assert times[0]<=t<=times[-1]
    hi=min(int(np.searchsorted(times,t,side='right')),len(times)-1);lo=max(0,hi-1)
    f=0. if times[hi]==times[lo] else float((t-times[lo])/(times[hi]-times[lo]))
    T=np.eye(4)
    T[:3,:3]=Slerp(times[[lo,hi]],Rotation.from_matrix(transforms[[lo,hi],:3,:3]))([t]).as_matrix()[0]
    T[:3,3]=(1-f)*transforms[lo,:3,3]+f*transforms[hi,:3,3]
    normal=np.array(data['source_to_normalized'])
    view0=np.array(next(x for x in data['cameras'] if x['view']==0)['camtoworld'])
    # Existing exporter writes frame_xyz in reference-view00 camera coordinates.
    source_from_export=T@np.linalg.inv(normal)@view0
    camera_path=H/'configs/human_source_common_v1'/('replacement_v3' if sid=='HN0646' else 'reviewed_source_v2')/sid/'camera.json'
    c=json.loads(camera_path.read_text())
    K=np.array(c['K_raw']);A=np.array(row['reference_frame']['undistorted_to_input'])
    np.testing.assert_allclose(A@K,row['reference_frame']['K_input'],atol=1e-8,rtol=0)
    p=H/'exports/ours_frozen_time_v1'/sid/'provenance.json';prov=json.loads(p.read_text())
    assert prov['timestamp_us']==c['timestamp_us'] and prov['source_time_seconds']==t
    snapshot=p.parent/'original_export_coordinates.npz'
    assert sha(snapshot)==prov['output_sha256'][snapshot.name]
    world_from_export=np.array(c['T_world_from_camera_opencv'])@source_from_export
    record=dict(status='estimated_source_registration_algebra_and_endpoint_projection_passed',id=sid,
        timestamp_us=c['timestamp_us'],source_time_seconds=t,bracket=[lo,hi],fraction=f,
        pose_interpolation='existing source-pose root translation linear and rotation SLERP; no joint edits',
        max_endpoint_projection_error_px=max(endpoint_errors),
        T_source_camera_from_export=source_from_export.tolist(),T_world_from_export=world_from_export.tolist(),
        camera_contract=str(camera_path),camera_contract_sha256=sha(camera_path),snapshot=str(snapshot),snapshot_sha256=sha(snapshot),
        provenance_sha256={str(x):sha(x) for x in [production,manifest,rootpath,rootmeta,kppath,projection,Path(paths['frame_mapping']),p]},
        source_pose_estimated=True,source_rgb_used_in_original_pose_estimation=True,
        heldout_eligible=False,world_metric_accuracy_certified=False,render_validated=False,metric_eligible=False,
        registration_policy='Reuse existing conditioning source-pose mapping and recorded exporter reference-view00 transform; no metric-guided fit, no per-prediction bbox scaling')
    return record


def main():
    active=json.loads((H/'inputs/active_human_benchmark.json').read_text())
    assert sha(active['selection'])==active['selection_sha256']
    selected={r['id']:r for r in json.loads(Path(active['selection']).read_text())['objects']}
    assert len(selected)==100 and 'HN0035' not in selected
    OUT.mkdir(exist_ok=False)
    rows=[]
    for sid in selected:
        try:
            record=prepare_row(sid,selected)
            record['active_selection_sha256']=active['selection_sha256']
            dest=OUT/(sid+'.json');dest.write_text(json.dumps(record,indent=2)+'\n')
            rows.append(dict(id=sid,status='trace_prepared',registration=str(dest),registration_sha256=sha(dest)))
        except Exception as error:
            rows.append(dict(id=sid,status='trace_unavailable',reason=repr(error)))
        print(sid,rows[-1]['status'],rows[-1].get('reason',''),flush=True)
    manifest=dict(selection=active['selection'],selection_sha256=active['selection_sha256'],rows=rows,
                  preparation_script_sha256=sha(__file__),scope='Original estimated conditioning pose and scale; no new fitting or regeneration')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
