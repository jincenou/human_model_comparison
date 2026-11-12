"""Sample Ours on the non-looping source clock, retaining full SH and root transforms."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

H=Path(__file__).resolve().parents[1]
SRC=H.parent/'Hasim_trajectory_assets/src'
spec=importlib.util.spec_from_file_location('hasim',SRC/'__init__.py',submodule_search_locations=[str(SRC)])
module=importlib.util.module_from_spec(spec);sys.modules['hasim']=module;spec.loader.exec_module(module)
from hasim.shared.assets.library_v2 import LibraryAsset,load_asset
from hasim.shared.contracts.errors import AssetError


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):d.update(chunk)
    return d.hexdigest()


def save_cloud(path,cloud):
    np.savez_compressed(path,means=cloud.means,quaternions_wxyz=cloud.quaternions,
        scales=cloud.scales,opacities=cloud.opacities,sh=cloud.sh,
        local_from_appearance=cloud.local_from_appearance,sh_degree=cloud.sh_degree)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--human-id',required=True);args=parser.parse_args()
    selection=H/'inputs/benchmark_human100_v3/selection.json'
    row=next(r for r in json.loads(selection.read_text())['objects'] if r['id']==args.human_id)
    paths=row['paths'];meta_path=Path(paths['asset_metadata']);meta=json.loads(meta_path.read_text())
    mapping=json.loads(Path(paths['frame_mapping']).read_text());native=json.loads(Path(paths['native_metadata']).read_text())
    assert sha(meta_path)==row['provenance_hashes']['asset_metadata']
    assert sha(paths['frame_mapping'])==row['provenance_hashes']['frame_mapping']
    assert mapping['source_speed_ratio']==1.0 and not mapping['speedup'] and not mapping['endpoint_padding']
    elapsed=(row['t_ref_us']-native['frames'][0]['frame_timestamp_micros'])/1e6
    assert abs(elapsed-row['reference_seconds_from_first_native'])<1e-8
    asset=LibraryAsset(row['id'],'human','human',Path(paths['runtime']),meta,(meta_path,))
    runtime=load_asset(asset);a=runtime.arrays;query=float(a['timestamps'][0])+elapsed
    assert 0<=elapsed<=runtime.duration_s
    cloud=runtime.sample(elapsed)
    try:runtime.sample(runtime.duration_s+.01)
    except AssetError:pass
    else:raise AssertionError('Out-of-range time unexpectedly looped or held')
    hi=min(int(np.searchsorted(a['timestamps'],query,side='right')),len(a['timestamps'])-1);lo=max(0,hi-1)
    fraction=0. if lo==hi else float((query-a['timestamps'][lo])/(a['timestamps'][hi]-a['timestamps'][lo]))
    def interp(k):return (1-fraction)*a[k][lo]+fraction*a[k][hi]
    root=interp('root_xy');heading=float(interp('root_yaw'));c,s=np.cos(-heading),np.sin(-heading)
    inplace=np.eye(4);inplace[:2,:2]=[[c,-s],[s,c]];inplace[:2,3]=-inplace[:2,:2]@root;inplace[2,3]=float(a['ground_offset'])
    raw_from_inplace=np.linalg.inv(inplace@a['asset_to_standard'])
    original=cloud.transformed(raw_from_inplace)
    restored_standard=cloud.transformed(np.linalg.inv(inplace))
    error=float(np.max(np.abs(original.means-interp('frame_xyz'))));assert error<2e-5
    assert np.allclose(original.local_from_appearance,np.eye(3),atol=1e-6)
    assert cloud.sh_degree==3 and np.array_equal(cloud.sh,a['sh'])
    out=H/'exports/ours_frozen_time_v1'/row['id'];out.mkdir(parents=True,exist_ok=False)
    for name,value in [('inplace',cloud),('original_export_coordinates',original),('standard_coordinates_with_root',restored_standard)]:save_cloud(out/(name+'.npz'),value)
    code=[SRC/'shared/assets/library_v2.py',SRC/'shared/assets/library_cloud.py',SRC/'shared/contracts/errors.py',Path(__file__)]
    record=dict(status='frozen_source_time_sampled_pending_render_world_registration',id=row['id'],identity_key=row['identity_key'],
        selection_sha256=sha(selection),timestamp_us=row['t_ref_us'],source_time_seconds=elapsed,
        source_clock='original non-looping clip, speed ratio1; no cycle_profile or seam blend applied',
        runtime_path=paths['runtime'],runtime_sha256=sha(paths['runtime']),duration_s=runtime.duration_s,
        lo_frame=lo,hi_frame=hi,interpolation_fraction=fraction,gaussians=len(cloud.means),sh_degree=cloud.sh_degree,
        covariance_interpolation='Runtime native SPD covariance interpolation; no independent quaternion SLERP',
        root_xy=root.tolist(),root_yaw_unwrapped_radians=heading,asset_to_standard=a['asset_to_standard'].tolist(),
        inplace_transform=inplace.tolist(),original_from_inplace=raw_from_inplace.tolist(),
        root_roundtrip_position_max_error=error,appearance_contract='Preserve full SH; row view direction d is evaluated at d @ local_from_appearance',
        code_sha256={str(p):sha(p) for p in code},output_sha256={p.name:sha(p) for p in out.glob('*.npz')},
        metric_scale_verified=meta.get('metric_scale_verified',False),
        world_registration_valid=False,reload_valid=False,common_camera_valid=False,metric_eligible=False,scored=False)
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(id=row['id'],gaussians=len(cloud.means),source_time_s=elapsed,sh_degree=cloud.sh_degree,root_roundtrip_error=error)))


if __name__=='__main__':main()
