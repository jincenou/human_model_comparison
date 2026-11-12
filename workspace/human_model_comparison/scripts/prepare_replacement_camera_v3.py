"""Derive immutable candidate camera/ROI references from official nuScenes records."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

H=Path(__file__).resolve().parents[1]
DATA=Path(_hmc_expand('${NUSCENES_ROOT}'))
OUT=H/'configs/human_source_common_v1/replacement_v3'


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):d.update(chunk)
    return d.hexdigest()


def transform(record):
    w,x,y,z=record['rotation'];q=np.array([x,y,z,w],dtype=np.float64)
    assert abs(np.linalg.norm(q)-1)<1e-6
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(q).as_matrix();t[:3,3]=record['translation']
    return t


def main():
    selection=H/'inputs/benchmark_human100_v3/selection.json'
    rows=[r for r in json.loads(selection.read_text())['objects'] if r['id']=='HN0646'];tokens={r['reference_frame']['sample_data_token'] for r in rows}
    meta=DATA/'v1.0-trainval'
    table_paths={k:meta/(k+'.json') for k in ['sample_data','ego_pose','calibrated_sensor']}
    samples={r['token']:r for r in json.loads(table_paths['sample_data'].read_text()) if r['token'] in tokens}
    ego_tokens={r['ego_pose_token'] for r in samples.values()}
    poses={r['token']:r for r in json.loads(table_paths['ego_pose'].read_text()) if r['token'] in ego_tokens}
    sensors={r['token']:r for r in json.loads(table_paths['calibrated_sensor'].read_text())}
    metadata_hashes={k:sha(v) for k,v in table_paths.items()}
    OUT.mkdir(parents=True,exist_ok=False);records=[]
    for row in rows:
        ref=row['reference_frame'];sample=samples[ref['sample_data_token']]
        assert sample['timestamp']==row['t_ref_us']
        raw=DATA/sample['filename'];assert raw.resolve()==Path(ref['raw_source_path']).resolve()
        assert sha(raw)==ref['raw_source_sha256']
        sensor=sensors[sample['calibrated_sensor_token']];ego=poses[sample['ego_pose_token']]
        k=np.asarray(sensor['camera_intrinsic'],dtype=np.float64)
        crop=np.asarray(ref['undistorted_to_input'],dtype=np.float64)
        assert np.allclose(crop@k,ref['K_input'],atol=1e-8,rtol=0)
        camera=transform(ego)@transform(sensor)
        rgb=Image.open(raw).convert('RGB');mask_path=Path(row['reference_mask_candidate']['path'])
        assert sha(mask_path)==row['reference_mask_candidate']['sha256']
        mask=Image.open(mask_path).convert('L');binary=np.asarray(mask)>127
        assert rgb.size==(sample['width'],sample['height']) and binary.shape==(rgb.height,rgb.width)
        ys,xs=np.nonzero(binary);assert len(xs)
        bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        side=math.ceil(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.1)
        left=math.floor((bbox[0]+bbox[2]-side)/2);top=math.floor((bbox[1]+bbox[3]-side)/2)
        roi=[left,top,left+side,top+side]
        folder=OUT/row['id'];folder.mkdir()
        white=Image.fromarray(np.where(binary[...,None],np.asarray(rgb),255).astype('uint8'))
        rgb_roi=Image.new('RGB',(side,side),'white');mask_roi=Image.new('L',(side,side),0)
        clip=[max(0,left),max(0,top),min(rgb.width,left+side),min(rgb.height,top+side)]
        offset=(clip[0]-left,clip[1]-top)
        rgb_roi.paste(white.crop(clip),offset);mask_roi.paste(mask.crop(clip),offset)
        rgb_roi.resize((512,512),Image.Resampling.BICUBIC).save(folder/'reference_rgb.png')
        mask_roi.resize((512,512),Image.Resampling.NEAREST).save(folder/'reference_mask.png')
        contract=dict(schema='human_source_common_v1_candidate',status='camera_derived_reference_qualification_pending',
            sample_id=row['id'],identity_key=row['identity_key'],scene=row['scene'],timestamp_us=row['t_ref_us'],
            selection_sha256=sha(selection),source_rgb=str(raw),source_rgb_sha256=sha(raw),
            source_mask=str(mask_path),source_mask_sha256=sha(mask_path),render_wh=list(rgb.size),
            K_raw=k.tolist(),T_world_from_camera_opencv=camera.tolist(),
            sample_data_token=sample['token'],ego_pose_token=ego['token'],calibrated_sensor_token=sensor['token'],
            metadata_table_sha256=metadata_hashes,raw_mask_bbox_xyxy=bbox,reference_pixel_height=bbox[3]-bbox[1],
            roi_xyxy=roi,roi_expansion=1.1,roi_size=512,rgb_interpolation='PIL bicubic',mask_interpolation='PIL nearest',
            roi_reference_rgb=str(folder/'reference_rgb.png'),roi_reference_rgb_sha256=sha(folder/'reference_rgb.png'),
            roi_reference_mask=str(folder/'reference_mask.png'),roi_reference_mask_sha256=sha(folder/'reference_mask.png'),
            raw_pixel_geometry='official dataset pixels and intrinsics; no guessed distortion correction',
            prediction_compositing='Prediction own alpha only; never multiply prediction by reference mask',
            reference_ready=False,metric_eligible=False,scored=False,orbit_settings_frozen=False)
        path=folder/'camera.json';path.write_text(json.dumps(contract,indent=2)+'\n')
        records.append(dict(id=row['id'],contract=str(path),sha256=sha(path)))
    (OUT/'manifest.json').write_text(json.dumps(dict(status='candidate_not_metric_qualification',count=len(records),records=records),indent=2)+'\n')
    print(json.dumps(dict(cameras=len(records),output=str(OUT))))


if __name__=='__main__':main()
