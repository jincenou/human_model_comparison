"""Apply logged source-coordinate mask edits; never qualify automatically."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import review_dynamic_references as review


def apply(plan_path):
    plan = review.read(plan_path)
    assert plan['prediction_inputs_used'] is False
    assert plan['variant'].replace('_', '').isalnum()
    folder = review.ROOT/plan['id']
    review.CANDIDATES = plan['base_candidate_name']
    source = list(review.inputs(folder))
    edits = {e['index']: e for e in plan['edits']}
    assert len(edits) == len(plan['edits']) and set(edits).issubset({f['index'] for f, _ in source})
    out = folder/('reference_masks_'+plan['variant'])
    out.mkdir(exist_ok=False)
    rows, records = [], []
    for frame, candidate in source:
        old = np.array(Image.open(candidate['path']).convert('L'))
        new = old.copy()
        if frame['index'] in edits:
            edit = edits[frame['index']]
            assert edit['source_rgb_sha256'] == frame['source_rgb_sha256']
            assert edit['source_mask_sha256'] == candidate['sha256']
            assert edit['reason'].strip()
            for operation, value in [('remove_polygons', 0), ('add_polygons', 255)]:
                for polygon in edit.get(operation, []):
                    assert len(polygon) >= 3
                    assert all(0 <= x < old.shape[1] and 0 <= y < old.shape[0] for x, y in polygon)
                    support = Image.new('L', (old.shape[1], old.shape[0]), 0)
                    ImageDraw.Draw(support).polygon([tuple(p) for p in polygon], fill=255)
                    new[np.array(support) > 127] = value
            records.append(dict(**edit, added_pixels=int(((new > 127) & (old <= 127)).sum()),
                                removed_pixels=int(((new <= 127) & (old > 127)).sum())))
        target = out/f"{frame['index']:05d}.png"
        if np.array_equal(old, new):
            shutil.copyfile(candidate['path'], target)
        else:
            Image.fromarray(new).save(target)
        rows.append(dict(index=frame['index'], timestamp_us=frame['timestamp_us'],
                         path=str(target), sha256=review.sha(target)))
    destination = folder/('mask_candidates_'+plan['variant']+'.json')
    assert not destination.exists()
    review.save(destination, dict(id=plan['id'], status='manual_candidates_pending_all_frame_visual_review',
        clip_sha256=review.sha(folder/'clip.json'), frames=rows,
        base_candidate_manifest=str(folder/review.CANDIDATES),
        base_candidate_sha256=review.sha(folder/review.CANDIDATES),
        correction_plan=str(plan_path), correction_plan_sha256=review.sha(plan_path),
        manual_source_pixel_edits=records, prediction_inputs_used=False,
        script_sha256=review.sha(__file__), reference_eligible=False))
    print(records, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    apply(p.parse_args().plan.resolve())
