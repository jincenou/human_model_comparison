"""Bounded semantic probe using existing native views, without choosing a source angle."""
import json
from pathlib import Path
import numpy as np
from audit_static_source_cameras_v1 import H, P, TRELLIS, sha, read, checked, crop_affines

IDS = ['HN0004','HN0035','HN0039','HN0040']


def native_camera(yaw, pitch=.2, radius=2., fov_degrees=40., size=512):
    """Reconstruct fixed original-TRELLIS preview; world Z-up and CV camera axes.

    Effective pixel K includes the rasterizer's ndc2Pix(..., S) half-pixel
    subtraction. This is metadata about these native views, not a raw-camera fix.
    """
    eye = radius * np.array([np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)])
    z = -eye / np.linalg.norm(eye)
    x = np.cross(-np.array([0.,0.,1.]),z);x /= np.linalg.norm(x)
    y = np.cross(z,x);y /= np.linalg.norm(y)
    view = np.eye(4);view[:3,:3] = np.stack([x,y,z]);view[:3,3] = -view[:3,:3] @ eye
    f = .5/np.tan(np.deg2rad(fov_degrees)/2)
    intrinsics = np.array([[f,0,.5],[0,f,.5],[0,0,1.]])
    effective = np.array([[size*f,0,(size-1)/2],[0,size*f,(size-1)/2],[0,0,1.]])
    return dict(yaw_radians=yaw,pitch_radians=pitch,radius=radius,fov_degrees=fov_degrees,
                normalized_intrinsics=intrinsics.tolist(),world_to_camera=view.tolist(),
                effective_pixel_K=effective.tolist(),projection_matrix=(effective @ view[:3]).tolist(),
                image_wh=[size,size],pixel_convention='native C++ ndc2Pix; integer pixel sampling; centered intrinsics only',
                source_orientation_known=False)


def main():
    output = H/'configs/human_source_common_v1/trellis_keypoint_probe_v1.json'
    if output.exists():
        raise RuntimeError('Probe already prepared')
    qualification = H/'configs/human_source_common_v1/reference_qualification_frozen_v1.json'
    references = {r['id']:r for r in read(qualification)['objects']}
    cases=[]
    for sid in IDS:
        ref=references[sid];checked(ref['camera_contract'],ref['camera_contract_sha256'])
        c=read(ref['camera_contract']);checked(c['reference_rgb'],c['reference_rgb_sha256'])
        _,centers=crop_affines(c['roi_xyxy'],(512,512))
        cases.append(dict(case_id=sid+'_reference',id=sid,role='reference',
                          image=c['reference_rgb'],image_sha256=c['reference_rgb_sha256'],
                          identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],
                          camera_contract=ref['camera_contract'],camera_contract_sha256=ref['camera_contract_sha256'],
                          roi_to_raw_integer_center_affine=np.linalg.inv(centers).tolist(),
                          raw_K=c['K_raw']))
        native=H/'runs/trellis_original_human_native_v1'/sid
        reload=read(native/'reload_validation.json')
        for index,yaw in enumerate([0.,.29670597]):
            path=native/f'reload_{index}.png';checked(path,reload['views'][index]['sha256'])
            cases.append(dict(case_id=f'{sid}_native_{index}',id=sid,role='native_asset_view',
                              image=str(path),image_sha256=sha(path),
                              identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],
                              native_payload=str(native/'native_gaussians.pt'),native_payload_sha256=reload['output_sha256'],
                              camera=native_camera(yaw),view_index=index))
    source_paths=[Path(__file__),H/'scripts/run_trellis_original_human_native_v1.py',
                  TRELLIS/'trellis/utils/render_utils.py',TRELLIS/'trellis/renderers/gaussian_render.py',
                  P/'sources/recon02_diff_gaussian_rasterization_local/cuda_rasterizer/auxiliary.h',
                  P/'sources/recon02_diff_gaussian_rasterization_local/cuda_rasterizer/forward.cu']
    manifest=dict(schema='trellis_semantic_probe_v1',scope='diagnostic only; two nearby views cannot establish accepted 3D anchors',
                  ids=IDS,cases=cases,reference_qualification_sha256=sha(qualification),
                  source_sha256={str(p):sha(p) for p in source_paths},
                  detector_policy='fixed entire 512x512 isolated image ROI for all cases; no per-method image recentering/resizing',
                  score_based_view_selection=False,metric_eligible=False,scores_produced=False)
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(output),cases=len(cases),sha256=sha(output))))


if __name__=='__main__':main()
