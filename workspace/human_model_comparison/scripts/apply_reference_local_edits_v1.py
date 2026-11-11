"""Apply explicit hash-bound reference edits and produce reviewable revisions."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

H=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    plan_path=H/'inputs/reference_local_edits_v1.json';plan=json.loads(plan_path.read_text())
    originals={r['id']:r for r in json.loads((H/'reports/reference_qualification_v1/manifest.json').read_text())['objects']}
    out=H/'reports/reference_qualification_v1/local_edits_v1_review';out.mkdir(exist_ok=False)
    results=[]
    for row in plan['records']:
        assert sha(row['base_mask_path'])==row['base_mask_sha256'] and sha(row['source_path'])==row['source_sha256']
        base=np.array(Image.open(row['base_mask_path']));assert set(np.unique(base)).issubset({0,255})
        mask=base.copy();fg,_=ndimage.label(base>127,np.ones((3,3)));bg,_=ndimage.label(base<=127,np.ones((3,3)))
        changed=np.zeros(base.shape,bool)
        for edit in row['edits']:
            remove=edit['kind']=='remove_component';assert remove or edit['kind']=='fill_hole'
            yy,xx=np.where((fg if remove else bg)==edit['label'])
            assert len(xx)==edit['area']
            assert [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]==edit['bbox']
            if not remove:assert xx.min()>0 and yy.min()>0 and xx.max()<base.shape[1]-1 and yy.max()<base.shape[0]-1
            assert not changed[yy,xx].any();mask[yy,xx]=0 if remove else 255;changed[yy,xx]=True
        if 'union_mask_path' in row:
            assert sha(row['union_mask_path'])==row['union_mask_sha256']
            part=np.array(Image.open(row['union_mask_path']));assert part.shape==base.shape and set(np.unique(part)).issubset({0,255})
            changed|=(part>127)&(mask==0);mask=np.maximum(mask,part)
        assert np.array_equal(mask[~changed],base[~changed]) and np.array_equal(mask!=base,changed)
        root=H/'references/exact_exposure_v2'/row['id'];dest=root/'revisions'/plan['revision'];dest.mkdir(parents=True,exist_ok=False)
        Image.fromarray(mask).save(dest/'mask.png');meta=json.loads((root/'metadata.json').read_text())
        meta.update(mask_path=str(dest/'mask.png'),mask_sha256=sha(dest/'mask.png'),base_mask_path=row['base_mask_path'],base_mask_sha256=row['base_mask_sha256'],
            correction=row,correction_plan=str(plan_path),correction_plan_sha256=sha(plan_path),correction_script_sha256=sha(__file__),
            mask_origin='explicit_reviewed_reference_component_edits_or_accessory_union',changed_pixels=int(changed.sum()),
            added_pixels=int(((mask>127)&(base==0)).sum()),removed_pixels=int(((base>127)&(mask==0)).sum()),
            foreground_pixels=int((mask>127).sum()),original_mask_unchanged=sha(row['base_mask_path'])==row['base_mask_sha256'],
            generation_inputs_unchanged=True,visual_review='pending',reference_ready=False,metric_eligible=False)
        (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');results.append(meta)
    for offset in range(0,len(results),3):
        canvas=Image.new('RGB',(1440,1440),'#e8edf1');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(results[offset:offset+3]):
            raw=Image.open(row['source_path']).convert('RGB');mask=Image.open(row['mask_path']);a=np.array(raw);m=np.array(mask)>127
            overlay=a.copy();overlay[m]=(a[m]*.6+np.array([230,30,30])*.4).astype('uint8');roi=originals[row['id']]['review_roi']
            draw.text((6,j*480+4),row['id']+' raw / revised overlay / mask',fill='black')
            for col,img in enumerate([raw,Image.fromarray(overlay),mask.convert('RGB')]):canvas.paste(img.crop(roi).resize((450,450),Image.Resampling.NEAREST),(col*480,j*480+25))
        canvas.save(out/f'page_{offset//3+1:02d}.png')
    (out/'manifest.json').write_text(json.dumps({'objects':results,'status':'candidate_revisions_pending_visual_review'},indent=2)+'\n')
    print(json.dumps({'count':len(results),'changed_pixels':sum(r['changed_pixels'] for r in results),'report':str(out)}))

if __name__=='__main__':main()
