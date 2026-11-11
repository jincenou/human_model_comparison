"""One real exposure per frozen identity; reference-defined crop, no pose edits."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

H = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection = H / 'inputs/benchmark_human100_v2/selection.json'
    rows = json.loads(selection.read_text())['objects']
    # Fixed before observing AH outputs; one initial pilot, no batch eligibility implied.
    row = next(r for r in rows if r['id'] == 'HN0004')
    raw = Path(row['reference_frame']['raw_source_path'])
    mask_path = Path(row['reference_mask_candidate']['path'])
    assert sha(raw) == row['reference_frame']['raw_source_sha256']
    assert sha(mask_path) == row['reference_mask_candidate']['sha256']
    image = Image.open(raw).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    binary = np.asarray(mask) > 127
    assert binary.shape == (image.height, image.width) and binary.any()
    ys, xs = np.nonzero(binary)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
    side = math.ceil(max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 1.10)
    left = math.floor((bbox[0]+bbox[2]-side)/2)
    top = math.floor((bbox[1]+bbox[3]-side)/2)
    roi = [left, top, left+side, top+side]
    white = Image.fromarray(np.where(binary[..., None], np.asarray(image), 255).astype('uint8'))
    rgb_roi = Image.new('RGB', (side, side), 'white')
    mask_roi = Image.new('L', (side, side), 0)
    clipped = [max(0,left), max(0,top), min(image.width,left+side), min(image.height,top+side)]
    offset = (clipped[0]-left, clipped[1]-top)
    rgb_roi.paste(white.crop(clipped), offset)
    mask_roi.paste(mask.crop(clipped), offset)
    root = H / 'inputs/ah/jobs/human_pilot_v1_HN0004'
    folder = root / row['id']
    folder.mkdir(parents=True, exist_ok=False)
    rgb_path, output_mask = folder/'frame.jpeg', folder/'mask.png'
    rgb_roi.resize((512,512),Image.Resampling.BICUBIC).save(rgb_path,quality=100,subsampling=0)
    mask_roi.resize((512,512),Image.Resampling.NEAREST).save(output_mask)
    record = dict(sample_id=row['id'], scene=row['scene'], identity_key=row['identity_key'],
        timestamp_us=row['t_ref_us'], selection_path=str(selection), selection_sha256=sha(selection),
        raw_source_path=str(raw), raw_source_sha256=sha(raw), exact_mask_path=str(mask_path),
        exact_mask_sha256=sha(mask_path), raw_shape=[image.height,image.width],
        mask_bbox_xyxy=bbox, raw_person_pixel_height=bbox[3]-bbox[1], roi_xyxy=roi,
        preprocessing='visible-mask white background; reference bbox long side x1.10 square; RGB bicubic/mask nearest 512; JPEG quality100 subsampling0',
        image=str(rgb_path), mask=str(output_mask), source_image_sha256=sha(rgb_path),
        source_mask_sha256=sha(output_mask), real_input_count=1, real_duration_s=0,
        interpolation_frames=0, input_generated_images=0, data_nature='real_photo',
        camera_prior='AH camera estimator prediction; no real camera supplied to generation',
        stage='fixed_native_pilot_only', formal_reference_eligible=False,
        visual_review='HN0004 exact-exposure mask inspected; target isolated; formal qualification pending',
        pose_policy='preserve native prediction; no rigging, reposing or best-frame choice')
    (root/'source_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
