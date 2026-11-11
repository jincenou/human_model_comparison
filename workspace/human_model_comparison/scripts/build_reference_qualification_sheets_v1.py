"""Review only original exposures and masks, never model results or scores."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parents[1];O=H/'reports/reference_qualification_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);selection=H/'inputs/benchmark_human100_v2/selection.json'
 rows=json.loads(selection.read_text())['objects'];font=ImageFont.load_default(size=20);records=[]
 for page in range(20):
  sheet=Image.new('RGB',(1280,1740),'#e9edf2');draw=ImageDraw.Draw(sheet)
  for col,label in enumerate(['Real context','Mask overlay','Candidate mask','White reference']):draw.text((col*320+8,8),label,font=font,fill='#102030')
  for index,row in enumerate(rows[page*5:(page+1)*5]):
   sid=row['id'];raw=Path(row['reference_frame']['raw_source_path']);mp=Path(row['reference_mask_candidate']['path'])
   assert sha(raw)==row['reference_frame']['raw_source_sha256'] and sha(mp)==row['reference_mask_candidate']['sha256']
   rgb=Image.open(raw).convert('RGB');mask=Image.open(mp).convert('L');a=np.asarray(rgb);m=np.asarray(mask)>127;ys,xs=np.nonzero(m);assert len(xs)
   bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1];side=math.ceil(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.5)
   left=math.floor((bbox[0]+bbox[2]-side)/2);top=math.floor((bbox[1]+bbox[3]-side)/2);box=(left,top,left+side,top+side)
   overlay=a.copy();overlay[m]=(overlay[m]*.65+np.array([255,40,40])*.35).astype('uint8')
   white=np.where(m[...,None],a,255).astype('uint8')
   images=[rgb,Image.fromarray(overlay),mask.convert('RGB'),Image.fromarray(white)]
   y=38+index*340;draw.text((8,y),f'{sid}  {row["scene"]}  height={bbox[3]-bbox[1]}px',font=font,fill='#102030')
   for col,im in enumerate(images):sheet.paste(im.crop(box).resize((310,310),Image.Resampling.NEAREST if col==2 else Image.Resampling.BICUBIC),(col*320+5,y+27))
   records.append(dict(id=sid,page=page+1,row=index+1,raw=str(raw),raw_sha256=sha(raw),mask=str(mp),mask_sha256=sha(mp),
    bbox=bbox,review_roi=box,mask_touches_image_border=bool(m[0].any()or m[-1].any()or m[:,0].any()or m[:,-1].any()),
    reference_pixel_height=bbox[3]-bbox[1],identity_key=row['identity_key'],timestamp_us=row['t_ref_us'],status='pending_visual_review'))
  sheet.save(O/f'sheet_{page+1:02d}.jpg',quality=95)
 (O/'manifest.json').write_text(json.dumps(dict(selection_sha256=sha(selection),scope='Reference-only visual qualification; expanded review crop is not scoring ROI',columns=['raw','overlay','mask','white'],objects=records),indent=2)+'\n')
 print('20 reference-only sheets generated for 100 frozen identities')
if __name__=='__main__':main()
