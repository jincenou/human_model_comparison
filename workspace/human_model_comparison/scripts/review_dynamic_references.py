"""Source-only contact sheets and explicitly recorded visual review decisions.

Sheet construction never qualifies references. Accept only after actually viewing
every source/mask pair. Review crops are display-only, never metric crops.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

H = Path(__file__).resolve().parents[1]
ROOT = H / 'reports/dynamic_source_clip_v1'
CANDIDATES = 'mask_candidates.json'
REVIEW_PREFIX = 'source_mask_review'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(p)


def inputs(folder):
    clip = read(folder / 'clip.json')
    masks = read(folder / CANDIDATES)
    assert masks['clip_sha256'] == sha(folder / 'clip.json')
    assert len(clip['frames']) == len(masks['frames'])
    for f, m in zip(clip['frames'], masks['frames']):
        assert (f['index'], f['timestamp_us']) == (m['index'], m['timestamp_us'])
        assert sha(f['source_rgb']) == f['source_rgb_sha256']
        assert sha(m['path']) == m['sha256']
        yield f, m


def sheets(sid):
    folder = ROOT / sid
    records = list(inputs(folder))
    sheets_record = []
    # 15 frames per sheet, full source crop beside masked source; no predictions.
    for start in range(0, len(records), 15):
        sheet = Image.new('RGB', (1400, 690), '#d6d6d6')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 4), sid + ' | each pair: source / masked source', fill='black')
        indices = []
        for j, (f, m) in enumerate(records[start:start+15]):
            rgb = np.asarray(Image.open(f['source_rgb']).convert('RGB'))
            mask = np.asarray(Image.open(m['path']).convert('L')) > 127
            assert mask.any()
            ys, xs = np.where(mask)
            box = f['prompt_bbox_xyxy']
            x0, y0 = min(box[0], xs.min()), min(box[1], ys.min())
            x1, y1 = max(box[2], xs.max()+1), max(box[3], ys.max()+1)
            pad = .12 * max(x1-x0, y1-y0)
            crop = tuple(map(int, (max(0,x0-pad), max(0,y0-pad),
                                   min(rgb.shape[1],x1+pad), min(rgb.shape[0],y1+pad))))
            segmented = np.where(mask[...,None], rgb, 255).astype('uint8')
            x, y = (j % 5)*280, 24+(j//5)*222
            draw.text((x+3, y), f"{f['index']:02d}  {f['time_seconds']:.3f}s", fill='black')
            for k, pixels in enumerate((rgb, segmented)):
                panel = ImageOps.pad(Image.fromarray(pixels).crop(crop), (138,200), color='white')
                sheet.paste(panel, (x+k*140,y+18))
            indices.append(f['index'])
        path = folder / f'{REVIEW_PREFIX}_{start//15:02d}.jpg'
        sheet.save(path, quality=94)
        sheets_record.append(dict(path=str(path),sha256=sha(path),frame_indices=indices))
    save(folder/(REVIEW_PREFIX+'.json'), dict(id=sid,clip_sha256=sha(folder/'clip.json'),
         masks_sha256=sha(folder/CANDIDATES),candidate_manifest=str(folder/CANDIDATES),sheets=sheets_record,
         predictions_used=False,reference_eligible=False,script_sha256=sha(__file__)))
    print(sid, len(records), 'frames; awaiting visual review', flush=True)


def accept(sid, note):
    assert note.strip(), 'A concrete visual observation is required'
    folder = ROOT / sid
    destination = folder/'reference_qualification.json'
    assert not destination.exists(), 'Preserve prior qualification'
    review = read(folder/(REVIEW_PREFIX+'.json'))
    assert review['clip_sha256'] == sha(folder/'clip.json')
    assert review['masks_sha256'] == sha(folder/CANDIDATES)
    rows = []
    for f, m in inputs(folder):
        rows.append(dict(index=f['index'],timestamp_us=f['timestamp_us'],
             source_rgb=f['source_rgb'],source_rgb_sha256=f['source_rgb_sha256'],
             mask=m['path'],mask_sha256=m['sha256'],prior_mask=m['path'],
             prior_mask_sha256=m['sha256'],added_pixels=0,removed_pixels=0))
    assert [i for s in review['sheets'] for i in s['frame_indices']] == [r['index'] for r in rows]
    for sheet in review['sheets']:
        assert sha(sheet['path']) == sheet['sha256']
    save(destination, dict(status='reviewed_exact_exposure_references',id=sid,frames=rows,
         clip=str(folder/'clip.json'),clip_sha256=sha(folder/'clip.json'),
         all_real_exposures=True,all_frames_visually_reviewed=True,
         mask_review_basis=note,candidate_manifest=str(folder/CANDIDATES),candidate_manifest_sha256=sha(folder/CANDIDATES),
         manual_source_pixel_edits={},edits_reason='No pixel edits after visual review',
         limitations=['Visible blurred boundaries and small carried items retain pixel-level uncertainty'],
         reference_eligible=True,real_frame_count=len(rows),generated_reference_frames=0,
         review_images=review['sheets'],reviewed_at_utc=datetime.now(timezone.utc).isoformat()))
    print(sid, 'review recorded', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['sheets','accept'])
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--note', default='')
    parser.add_argument('--variant', default='native', help='Versioned mask candidate name, e.g. video_seed_v1 or video_corrected_v2')
    args = parser.parse_args()
    assert args.variant.replace('_','').isalnum(), 'Variant must be a plain identifier'
    if args.variant != 'native':
        CANDIDATES = 'mask_candidates_'+args.variant+'.json'
        REVIEW_PREFIX = 'source_mask_review_'+args.variant
    for sid in args.ids:
        if args.action == 'sheets': sheets(sid)
        else: accept(sid, args.note)
