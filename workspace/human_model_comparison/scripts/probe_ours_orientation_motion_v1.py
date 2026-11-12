"""Check one manually reviewed rigid correction at three original motion times."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image, ImageDraw
from export_ours_frozen_time_v1 import LibraryAsset, load_asset
from render_ours_source_pilot_v1 import H, sha, render


def main():
    torch.cuda.set_per_process_memory_fraction(.10)
    sid='HN0035'
    rp=H/'configs/human_source_common_v1/ours_registration_pilot_v1'/f'{sid}.json'
    reg=json.loads(rp.read_text())
    cp=Path(reg['camera_contract']);assert sha(cp)==reg['camera_contract_sha256']
    camera=json.loads(cp.read_text())
    selection=H/'inputs/benchmark_human100_v2/selection.json'
    row=next(r for r in json.loads(selection.read_text())['objects'] if r['id']==sid)
    meta_path=Path(row['paths']['asset_metadata']); meta=json.loads(meta_path.read_text())
    provenance=json.loads((H/'exports/ours_frozen_time_v1'/sid/'provenance.json').read_text())
    assert sha(row['paths']['runtime'])==provenance['runtime_sha256']
    runtime=load_asset(LibraryAsset(sid,'human','human',Path(row['paths']['runtime']),meta,(meta_path,)))
    rootpath=next(Path(p) for p in reg['provenance_sha256'] if p.endswith('/root_motion.npz'))
    manifest=next(Path(p) for p in reg['provenance_sha256'] if p.endswith('/fdanyone.json'))
    for p in [rootpath,manifest]:assert sha(p)==reg['provenance_sha256'][str(p)]
    data=json.loads(manifest.read_text())
    normal=np.array(data['source_to_normalized'])
    view0=np.array(next(c for c in data['cameras'] if c['view']==0)['camtoworld'])
    with np.load(rootpath,allow_pickle=False) as z:
        times=z['frame_times'];roots=z['local_to_smpl_incam']
    def root_at(t):
        hi=min(int(np.searchsorted(times,t,side='right')),len(times)-1);lo=max(0,hi-1)
        f=float((t-times[lo])/(times[hi]-times[lo]))
        T=np.eye(4);T[:3,:3]=Slerp(times[[lo,hi]],Rotation.from_matrix(roots[[lo,hi],:3,:3]))([t]).as_matrix()[0]
        T[:3,3]=(1-f)*roots[lo,:3,3]+f*roots[hi,:3,3]
        return T
    candidate_path=H/'reports/ours_orientation_probe_20261112_supplement_v1/probe.json'
    candidate=next(c for c in json.loads(candidate_path.read_text())['candidates'] if c['yaw_degrees']==300)
    correction=np.array(candidate['T_corrected_camera_from_camera'])
    reference_root=root_at(reg['source_time_seconds'])
    # One body-root-local correction for the whole sequence, never per-frame fitting.
    local_correction=np.linalg.inv(reference_root)@correction@reference_root
    np.testing.assert_allclose(local_correction[:3,3],0,atol=1e-6)
    out=H/'reports/ours_orientation_motion_probe_20261112_v1';out.mkdir(exist_ok=False)
    sheet=Image.new('RGB',(3*320,2*350),'white');draw=ImageDraw.Draw(sheet)
    entries=[];a=runtime.arrays
    for col,t in enumerate([0.,reg['source_time_seconds'],runtime.duration_s]):
        query=float(a['timestamps'][0])+t
        hi=min(int(np.searchsorted(a['timestamps'],query,side='right')),len(a['timestamps'])-1);lo=max(0,hi-1)
        f=0. if lo==hi else float((query-a['timestamps'][lo])/(a['timestamps'][hi]-a['timestamps'][lo]))
        def interp(k):return (1-f)*a[k][lo]+f*a[k][hi]
        heading=float(interp('root_yaw'));root=interp('root_xy');c,s=np.cos(-heading),np.sin(-heading)
        inplace=np.eye(4);inplace[:2,:2]=[[c,-s],[s,c]];inplace[:2,3]=-inplace[:2,:2]@root;inplace[2,3]=float(a['ground_offset'])
        original=runtime.sample(t).transformed(np.linalg.inv(inplace@a['asset_to_standard']))
        np.testing.assert_allclose(original.means,interp('frame_xyz'),atol=2e-5,rtol=0)
        source_root=root_at(t);base=source_root@np.linalg.inv(normal)@view0
        adjusted=source_root@local_correction@np.linalg.inv(normal)@view0
        for r,(label,T) in enumerate([('original',base),('fixed_yaw',adjusted)]):
            with torch.inference_mode():rgb,alpha=render(original.transformed(T),np.eye(4),np.array(camera['K_raw']),camera['render_wh'])
            p=out/f't{t:.3f}_{label}.png';im=Image.fromarray(np.floor(rgb*255+.5).astype(np.uint8));im.save(p)
            crop=im.crop(camera['roi_xyxy']).resize((320,320),Image.Resampling.BICUBIC)
            sheet.paste(crop,(col*320,r*350+30));draw.text((col*320+5,r*350+8),f'{t:.2f}s {label}',fill='black')
            entries.append(dict(time_seconds=t,variant=label,rgb=str(p),sha256=sha(p)))
    sheet.save(out/'internal_contact.jpg',quality=95)
    report=dict(id=sid,status='three_times_rendered_pending_visual_review',
                yaw_at_reference_degrees=-60, T_root_local_corrected_from_original=local_correction.tolist(),
                policy='One constant rigid transform in body-root coordinates; source root positions and original non-looping timestamps retained',
                selection_basis='Manual front/back review of fixed cardinal and +/-60 degree diagnostic renders; no metric optimization',
                source_rgb_used_for_orientation_selection=True,heldout_eligible=False,metric_eligible=False,
                scale_adjusted=False,original_asset_modified=False,regenerated=False,scored=False,
                full_sequence_validated=False, three_time_outputs=entries,
                input_sha256={str(p):sha(p) for p in [rp,cp,rootpath,manifest,candidate_path,meta_path,selection]},
                runtime_sha256=provenance['runtime_sha256'],script_sha256=sha(__file__),
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
    (out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(out/'probe.json')}))


if __name__=='__main__':main()
