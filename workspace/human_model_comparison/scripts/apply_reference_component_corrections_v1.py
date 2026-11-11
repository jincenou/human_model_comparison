"""Apply explicit, source-hash-bound reference corrections; never rewrite inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan_path = ROOT / 'inputs/reference_component_corrections_v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(plan['evidence']) == plan['evidence_sha256']
    assert plan['connectivity'] == 8
    records = {r['id']: r for r in json.loads((ROOT / 'reports/reference_qualification_v1/manifest.json').read_text())['objects']}
    panel = Image.new('RGB', (1200, 340 * len(plan['records'])), '#e8edf1')
    draw = ImageDraw.Draw(panel)
    results = []
    for index, row in enumerate(plan['records']):
        assert sha(row['source_raw']) == row['source_sha256']
        assert sha(row['base_mask']) == row['base_mask_sha256']
        base = np.array(Image.open(row['base_mask']))
        assert set(np.unique(base)).issubset({0, 255})
        labels, _ = ndimage.label(base > 127, np.ones((3, 3)))
        corrected = base.copy()
        removed = 0
        for component in row['remove_components']:
            ys, xs = np.where(labels == component['label'])
            assert len(ys) == component['area']
            assert [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)] == component['bbox']
            corrected[ys, xs] = 0
            removed += len(ys)
        assert int(np.count_nonzero(base != corrected)) == removed
        assert np.all(corrected <= base) and corrected.any()
        destination = Path(row['base_mask']).parent / 'revisions/components_v1'
        destination.mkdir(parents=True, exist_ok=False)
        Image.fromarray(corrected).save(destination / 'mask.png')
        raw = np.array(Image.open(row['source_raw']).convert('RGB'))
        assert raw.shape[:2] == corrected.shape
        white = np.where((corrected > 127)[..., None], raw, 255).astype(np.uint8)
        roi = records[row['id']]['review_roi']
        views = [Image.fromarray(raw), Image.fromarray(base).convert('RGB'), Image.fromarray(white)]
        draw.text((8, index * 340 + 4), f"{row['id']} | raw / old mask / corrected reference | removed {removed}px", fill='black')
        for col, view in enumerate(views):
            panel.paste(view.crop(roi).resize((310, 310), Image.Resampling.NEAREST), (col*400, index*340+25))
        metadata = dict(row, mask_path=str(destination/'mask.png'), mask_sha256=sha(destination/'mask.png'),
                        removed_pixels=removed, correction_plan=str(plan_path), correction_plan_sha256=sha(plan_path),
                        script_sha256=sha(__file__), mask_origin='visually_reviewed_explicit_background_component_removal',
                        original_mask_unchanged=sha(row['base_mask'])==row['base_mask_sha256'],
                        generation_inputs_unchanged=True, visual_review='pending', reference_ready=False, metric_eligible=False)
        (destination/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        results.append(metadata)
    output = ROOT/'reports/reference_qualification_v1/component_corrections_v1'
    output.mkdir(exist_ok=False)
    panel.save(output/'review.png')
    (output/'manifest.json').write_text(json.dumps({'results':results,'review_image':str(output/'review.png'),
        'review_image_sha256':sha(output/'review.png'),'status':'candidates_pending_visual_review'},indent=2)+'\n')
    print(json.dumps({'count':len(results),'removed_pixels':sum(r['removed_pixels'] for r in results),'report':str(output)}))


if __name__ == '__main__':
    main()
