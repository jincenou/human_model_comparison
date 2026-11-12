"""Bounded, unscored rigid yaw diagnosis; preserve original assets and cameras."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from render_ours_source_pilot_v1 import render, GaussianAssetFrame, H, sha


def main():
    sid = 'HN0035'
    torch.cuda.set_per_process_memory_fraction(0.10)
    rp = H/'configs/human_source_common_v1/ours_registration_pilot_v1'/f'{sid}.json'
    reg = json.loads(rp.read_text())
    assert sha(reg['snapshot']) == reg['snapshot_sha256']
    assert sha(reg['camera_contract']) == reg['camera_contract_sha256']
    camera = json.loads(Path(reg['camera_contract']).read_text())
    rootpath = next(Path(p) for p in reg['provenance_sha256'] if p.endswith('/root_motion.npz'))
    assert sha(rootpath) == reg['provenance_sha256'][str(rootpath)]
    with np.load(rootpath, allow_pickle=False) as z:
        assert reg['fraction'] == 0
        pivot = z['local_to_smpl_incam'][reg['bracket'][0], :3, 3].astype(float)
    with np.load(reg['snapshot'], allow_pickle=False) as z:
        cloud = GaussianAssetFrame(z['means'], z['quaternions_wxyz'], z['scales'],
                                   z['opacities'], z['sh'], z['local_from_appearance'])
    original = np.array(reg['T_source_camera_from_export'])
    mapped = cloud.transformed(original)
    # nuScenes world +Z, expressed in the frozen real camera. Root position stays fixed.
    axis = np.array(camera['T_world_from_camera_opencv'])[:3, :3].T @ np.array([0., 0., 1.])
    axis /= np.linalg.norm(axis)
    out = H/'reports/ours_orientation_probe_20261112_supplement_v1'
    out.mkdir(exist_ok=False)
    reference = Image.open(camera['reference_rgb']).convert('RGB')
    sheet = Image.new('RGB', (3*320, 350), 'white')
    sheet.paste(reference.resize((320, 320)), (0, 30))
    draw = ImageDraw.Draw(sheet); draw.text((5, 8), 'Real reference (same ROI)', fill='black')
    records = []
    for col, angle in enumerate((60, 300), 1):
        rot = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
        correction = np.eye(4); correction[:3, :3] = rot
        correction[:3, 3] = pivot-rot@pivot
        corrected = mapped.transformed(correction)
        np.testing.assert_allclose(correction[:3,:3]@pivot+correction[:3,3], pivot, atol=1e-12)
        np.testing.assert_allclose(corrected.scales, mapped.scales, atol=1e-7)
        assert np.array_equal(corrected.sh, mapped.sh)
        with torch.inference_mode():
            rgb, alpha = render(corrected, np.eye(4), np.array(camera['K_raw']), camera['render_wh'])
        paths = {}
        for name, array in [('rgb', rgb), ('alpha', alpha)]:
            p = out/f'yaw_{angle:03}_{name}.png'
            Image.fromarray(np.floor(array*255+.5).astype(np.uint8)).save(p)
            paths[name] = {'path': str(p), 'sha256': sha(p)}
        crop = Image.open(paths['rgb']['path']).crop(camera['roi_xyxy']).resize((320,320), Image.Resampling.BICUBIC)
        sheet.paste(crop, (col*320,30));draw.text((col*320+5,8), f'Rigid yaw {angle} deg', fill='black')
        records.append(dict(yaw_degrees=angle, T_corrected_camera_from_camera=correction.tolist(),
                            T_source_camera_from_export_candidate=(correction@original).tolist(), outputs=paths))
    sheet.save(out/'internal_contact.jpg', quality=95)
    report = dict(id=sid, status='rendered_pending_visual_review',
                  scope='Two supplemental fixed orientations: +/-60 degrees from the 24-view camera spacing, not a metric-driven angle search or accepted registration',
                  original_asset_modified=False, regenerated=False, scored=False, metric_eligible=False,
                  rotation_axis_camera=axis.tolist(), pivot_camera=pivot.tolist(),
                  policy='Fixed world-up yaw about recorded root; unchanged camera, scale, root position, time and reference ROI',
                  dynamic_sequence_validated=False, preserved_sh_degree=3,
                  input_sha256={str(rp):sha(rp), reg['snapshot']:sha(reg['snapshot']),
                                reg['camera_contract']:sha(reg['camera_contract']), str(rootpath):sha(rootpath)},
                  candidates=records, peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), script_sha256=sha(__file__))
    (out/'probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'report':str(out/'probe.json'), 'peak_torch_allocated_bytes':report['peak_torch_allocated_bytes']}))


if __name__ == '__main__':
    main()
