"""Eight deterministic examples; reuse exact certified 512px gallery panels.

No new rendering, independent fitting, per-method cropping or color processing.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from active_human_results import resolve

H = Path(__file__).resolve().parents[1]
OUT = resolve('advisor')
GALLERY = resolve('gallery')
METHODS = ['ours','ah','trellis_original','trellis2','omnire','splatad']
LABELS = ['Reference','Ours','AH','TRELLIS','TRELLIS.2','OmniRe','SplatAD']
FONT = _hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    fig = OUT/'figures'; fig.mkdir(exist_ok=True)
    members_path = resolve('core_summary').parent/'tables/paired_members.csv'
    members = list(csv.DictReader(members_path.open()))
    common = sorted({r['id'] for r in members if r['metric']=='PSNR' and len(r['group'].split('+'))==6})
    assert len(common)==46
    indices = [round(i*(len(common)-1)/7) for i in range(8)]
    selected = [common[i] for i in indices]
    manifest = json.loads((GALLERY/'manifest.json').read_text())
    font = ImageFont.truetype(FONT, 34)
    overview = Image.new('RGB', (7*320, 64+8*320), 'white')
    overview_draw = ImageDraw.Draw(overview)
    smallfont = ImageFont.truetype(FONT, 27)
    for col,label in enumerate(LABELS):
        overview_draw.text((col*320+160,32),label,font=smallfont,anchor='mm',fill='black')
    records = []
    for row,sid in enumerate(selected):
        page = (GALLERY/(sid+'.html')).read_text()
        data = json.loads(re.search(r'<script type="application/json" id="gallery-data">(.*?)</script>',page).group(1))
        ref = re.search(r'alt="真实源视角" src="([^"]+)"',page).group(1)
        panel_paths = {p['method']:p['source'] for p in data['panels']}
        sources = [(GALLERY/ref).resolve()] + [(GALLERY/panel_paths[m]).resolve() for m in METHODS]
        single = Image.new('RGB',(7*512,576),'white');draw=ImageDraw.Draw(single)
        evidence=[]
        for col,(label,path) in enumerate(zip(LABELS,sources)):
            assert sha(path)==manifest['image_files'][str(path)]
            im=Image.open(path).convert('RGB');assert im.size==(512,512)
            draw.text((col*512+256,32),label,font=font,anchor='mm',fill='black')
            single.paste(im,(col*512,64))
            assert np.array_equal(np.asarray(single.crop((col*512,64,(col+1)*512,576))),np.asarray(im))
            overview.paste(im.resize((320,320),Image.Resampling.LANCZOS),(col*320,64+row*320))
            evidence.append(dict(label=label,path=str(path),sha256=sha(path)))
        destination=fig/(sid+'.png');single.save(destination)
        records.append(dict(row=row+1,id=sid,file=str(destination),sha256=sha(destination),panels=evidence))
    overview.save(fig/'eight_objects.png')
    overview.resize((1400,1640),Image.Resampling.LANCZOS).save(fig/'eight_objects_preview.jpg',quality=95)
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),selection_rule='Sorted six-method common PSNR identities; evenly spaced indices round(i*(46-1)/7), i=0..7; no quality-based selection.',
        common_count=len(common),selected_indices=indices,selected_ids=selected,columns=LABELS,
        members_path=str(members_path),members_sha256=sha(members_path),gallery_manifest_sha256=sha(GALLERY/'manifest.json'),
        source_view_only=True,generated_new_renders=False,independent_normalization=False,
        single_panel_pixels_exact=True,overview_resize='identical full-panel LANCZOS for all methods',rows=records,
        overview_sha256=sha(fig/'eight_objects.png'),script_sha256=sha(__file__))
    (OUT/'figures_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,overview=str(fig/'eight_objects.png')),ensure_ascii=False))


if __name__=='__main__': main()
