"""Read-only human provenance inventory and CPU review contact sheets."""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,asyncio,collections,hashlib,json,math,os
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
R=Path(__file__).resolve().parents[1]
LIB=Path(_hmc_expand('${HMC_WORKSPACE}/asset_library/human'))
WORK=Path(_hmc_expand('${HMC_WORKSPACE}/asset_work'))
cv2.setNumThreads(1)
FONT_PATH=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
FONT=ImageFont.truetype(FONT_PATH,18)
SMALL=ImageFont.truetype(FONT_PATH,14)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def inventory():
    reviews={x['identity']:x for x in read(WORK/'human_review_20261026/manual_review.json')['entries']}
    rows=[];excluded=[]
    for dataset,batch in [('nuscenes','human_nuscenes100_20261026'),('waymo','human300_20261024')]:
        base=WORK/batch
        items={x['id']:x for x in read(base/'selection.json')['items']}
        for entry in read(LIB/'library.json')['assets']:
            meta=LIB/entry['metadata'];ident=meta.parent.name
            if ident not in items:continue
            s=items[ident];m=read(meta)
            root=base/'inputs'/ident if dataset=='nuscenes' else Path(s['package'])
            clip=root/'clip0';review=base/'review'/f'{ident}.json'
            required={'source_video':clip/'observed_native.mp4','input_video':clip/'inputs/main.mp4',
                      'mask_npz':clip/'segmentation/target_masks.npz','frame_mapping':clip/'inputs/frame_mapping.json',
                      'native_metadata':clip/'native.json','asset_preview':LIB/'previews'/f'{ident}.mp4',
                      'runtime':meta.parent/'runtime.npz','asset_metadata':meta,'production_config':base/'config'/f'{ident}.json'}
            reason=[]
            if m['quality_state']=='flagged_for_review':reason.append('existing_asset_flagged')
            if reviews.get(ident,{}).get('status')=='confirmed_visible_issue':reason.append('existing_confirmed_visual_issue')
            temporal=base/'temporal'/f'{ident}.json'
            if temporal.exists() and read(temporal)['diagnosis']['decision']!='keep_full':reason.append('trimmed_input_requires_separate_mapping_audit')
            if not review.exists():reason.append('missing_input_provenance_review')
            else:
                rv=read(review);inp=next((x for x in rv['sources'] if x['kind']=='model input'),None)
                if not inp or Path(inp['path']).resolve()!=required['input_video'].resolve():reason.append('production_input_repair_or_trim_requires_separate_mapping_audit')
            missing=[k for k,p in required.items() if not p.is_file()]
            if missing:reason.append('missing:'+','.join(missing))
            if reason:excluded.append({'id':ident,'dataset':dataset,'reasons':reason});continue
            adm=base/'review'/f'{ident}_admission.json';vision=read(adm).get('vision',{}) if adm.exists() else {}
            if any(vision.get(k) is True for k in ['extra_person_present','major_body_missing','foreground_background_leak']) or vision.get('identity_consistent') is False:
                excluded.append({'id':ident,'dataset':dataset,'reasons':['existing_input_identity_or_segmentation_alert']});continue
            rows.append({'id':ident,'dataset':dataset,'scene':s['scene'],'identity_key':s['identity_key'],
                         'height_median_px':float(s['height_median_px']),'rank_score':float(s.get('rank_score',0)),
                         'motion_type':m['motion_type'],'duration_s':m['duration_s'],'quality_state':m['quality_state'],
                         'paths':{k:str(v) for k,v in required.items()},'input_review':str(review),
                         'existing_input_vision':vision,'existing_quality_flags':m.get('quality_flags',[])})
    # Review a surplus, balancing scene diversity and retaining smaller valid inputs.
    selected=[]
    for dataset,count,hard_count in [('nuscenes',110,25),('waymo',30,8)]:
        pool=[r for r in rows if r['dataset']==dataset];used=set();scene_counts=collections.Counter()
        def take(seq,n,group):
            added=0
            for row in seq:
                if row['identity_key'] in used or scene_counts[row['scene']]>=3:continue
                out=dict(row,selection_group=group);selected.append(out);used.add(row['identity_key']);scene_counts[row['scene']]+=1;added+=1
                if added>=n:break
            return added
        take(sorted(pool,key=lambda x:(x['height_median_px'],x['id'])),hard_count,'smaller_input')
        take(sorted(pool,key=lambda x:(-x['rank_score'],x['id'])),count-hard_count,'base')
    selected.sort(key=lambda r:(r['dataset']!='nuscenes',r['selection_group']=='smaller_input',r['id']))
    for i,r in enumerate(selected,1):r['candidate_number']=i
    save(R/'inputs/inventory.json',{'eligible':rows,'excluded':excluded,'counts':dict(collections.Counter(r['dataset'] for r in rows))})
    save(R/'inputs/candidates.json',{'status':'awaiting_direct_visual_review','candidates':selected,'target':{'nuscenes':80,'waymo':20},'selection_uses_existing_asset_flags':True})
    print(json.dumps({'eligible':len(rows),'candidates':len(selected),'counts':dict(collections.Counter(r['dataset'] for r in selected))}),flush=True)
    return selected
def frames(path,indices):
    c=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert c.isOpened(),path
    count=int(c.get(cv2.CAP_PROP_FRAME_COUNT));out=[]
    for idx in indices:
        assert 0<=idx<count,(path,idx,count)
        c.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,f=c.read();assert ok,(path,idx)
        out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    c.release();return out,count
def fit(im,w,h):return ImageOps.pad(Image.fromarray(im),(w,h),method=Image.Resampling.LANCZOS,color=(235,235,235))
def media(row):
    dest=R/'reports/candidates'/row['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'evidence.json').exists():return read(dest/'evidence.json')
    p={k:Path(v) for k,v in row['paths'].items()};mapping=read(p['frame_mapping'])['frames'];native=read(p['native_metadata'])['frames']
    n=len(mapping);slots=[0,(n-1)//2,n-1]
    native_ids=[int(mapping[i]['left_native_index']) for i in slots]
    src,_=frames(p['source_video'],native_ids);inp,ni=frames(p['input_video'],slots);asset,na=frames(p['asset_preview'],slots)
    assert ni==n and na==n,(row['id'],'temporal lengths',ni,n,na)
    with np.load(p['mask_npz']) as z:masks=z['masks']
    assert len(masks)==n
    review=read(Path(row['input_review']));input_record=next(x for x in review['sources'] if x['kind']=='model input')
    hashes={k:sha(p[k]) for k in ['source_video','input_video','mask_npz','frame_mapping','native_metadata','asset_preview','asset_metadata','production_config']}
    assert hashes['input_video']==input_record['sha256'],(row['id'],'actual model input SHA differs')
    expected_mask=review.get('alpha_metrics',{}).get('mask_source_sha256')
    if expected_mask:assert expected_mask==hashes['mask_npz'],(row['id'],'retained mask SHA differs')
    sheet=Image.new('RGB',(900,3*354),(248,248,248));draw=ImageDraw.Draw(sheet);slices=[]
    for j,(raw,input_frame,pre,idx) in enumerate(zip(src,inp,asset,slots)):
        mask=masks[idx]>127 if masks.max()>1 else masks[idx]>0
        assert mask.shape==input_frame.shape[:2]
        # Fixed original crop / model mask display; never modify source masks.
        overlay=raw.copy();assert raw.shape==input_frame.shape
        overlay[mask]=(overlay[mask]*.55+np.array([0,220,130])*.45).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(255,60,10),1)
        assert pre.shape[1]%5==0,(row['id'],'unexpected retained preview layout')
        cell=pre.shape[1]//5
        parts=[raw,overlay,input_frame,pre[32:,3*cell:4*cell],pre[32:,4*cell:5*cell]]
        labels=['Real observation','Mask on observation','Actual model input','Asset view 00','Asset view 08']
        y=j*354
        for k,(arr,label) in enumerate(zip(parts,labels)):
            sheet.paste(fit(arr,180,320),(k*180,y+34));draw.text((k*180+3,y+16),label,font=SMALL,fill='black')
        text=f'{row["id"]} | slot {idx} | observed frame {native_ids[j]} | interpolated input: {mapping[idx].get("interpolated",False)}'
        draw.text((3,y),text,font=SMALL,fill='black')
        # Raw map and actual exposure are retained; the nearest observation is not relabelled as a generated slot.
        slices.append({'slot':idx,'observed_native_index':native_ids[j],'mapping':mapping[idx],'native':native[native_ids[j]]})
    sheet.save(dest/'contact.jpg',quality=91)
    first=sheet.crop((0,0,900,354));first.save(dest/'first.jpg',quality=90)
    evidence={'id':row['id'],'status':'media_ready','frame_count':n,'sampled_slots':slots,'sampled_source_mapping':slices,
              'hashes':hashes,'contact':str(dest/'contact.jpg'),'preview_origin':'Existing asset_library Grid4D preview; last two columns only. Not a new render or generated reference.',
              'observation_note':'At interpolated slots, mask overlay uses the nearest left native exposure; frame0 is exact. Overlay is a review aid, not IoU evaluation.'}
    save(dest/'evidence.json',evidence);return evidence
def main():
    rows=inventory();results=[]
    for row in rows:
        print('processing',row['id'],flush=True)
        try:r=media(row);results.append({'id':row['id'],'status':r['status']})
        except Exception as e:results.append({'id':row['id'],'status':'failed','reason':repr(e)})
        save(R/'reports/preparation_status.json',{'completed':len(results),'total':len(rows),'results':results})
    good={x['id'] for x in results if x['status']=='media_ready'}
    for start in range(0,len(rows),10):
        block=rows[start:start+10];canvas=Image.new('RGB',(1800,5*390),(248,248,248));d=ImageDraw.Draw(canvas)
        for k,row in enumerate(block):
            x=(k%2)*900;y=(k//2)*390
            d.text((x+4,y+4),f'C{row["candidate_number"]:03d} {row["id"]} {row["dataset"]} {row["motion_type"]} h={row["height_median_px"]:.0f}px',font=FONT,fill='black')
            if row['id'] in good:canvas.paste(Image.open(R/'reports/candidates'/row['id']/'first.jpg'),(x,y+30))
        canvas.save(R/'reports'/f'candidate_sheet_{start//10+1:02d}.jpg',quality=90)
    print(json.dumps({'ready':len(good),'failed':[x for x in results if x['status']=='failed']}),flush=True)
if __name__=='__main__':main()
