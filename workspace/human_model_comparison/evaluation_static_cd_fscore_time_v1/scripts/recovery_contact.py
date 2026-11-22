from common import *
from PIL import Image,ImageDraw
import numpy as np
import sys
for method,sid in [('trellis_original','HN0517'),('trellis2','HN0263')]:
    out=E/'registration_recovery'/('local_views' if '--local' in sys.argv else 'existing')/method/sid
    if '--local' in sys.argv:out=out/'0'
    reg=read(out/'registration.json')
    folder=H/'reports'/('trellis_source_batch' if method=='trellis_original' else 'trellis2_source_registration')/'observations'/sid
    records=[read(r['record']) for r in read(folder/'keypoints.json')['records']];ref=next(r for r in records if r['case']['role']=='reference');images=[]
    im=Image.open(ref['case']['image']).convert('RGB');dr=ImageDraw.Draw(im);A=np.linalg.inv(np.asarray(ref['case']['roi_to_raw_integer_center_affine']))
    for j,uv,px in zip(reg['joints'],reg['source_pixels_raw'],reg['projected_pixels_raw']):
        i=j['joint_index'];u,v=(A@[*uv,1])[:2];x,y=(A@[*px,1])[:2];dr.ellipse((u-3,v-3,u+3,v+3),outline='lime',width=2);dr.text((u,v),str(i),fill='lime');dr.ellipse((x-3,y-3,x+3,y+3),outline='red',width=2);dr.text((x,y),str(i),fill='red')
    dr.text((4,4),sid+' SOURCE: green=observed red=global fit',fill='red');images.append(im)
    for index in [0,4,8,12]:
        r=next(r for r in records if r['case']['role']=='native_asset_view' and r['case']['view_index']==index);im=Image.open(r['case']['image']).convert('RGB');dr=ImageDraw.Draw(im);cam=r['case']['camera']
        for j in reg['joints']:
            xyz=np.asarray(cam['world_to_camera'])@[*j['geometry']['xyz'],1];p=np.asarray(cam['effective_pixel_K'])@xyz[:3];u,v=p[:2]/p[2];dr.ellipse((u-2,v-2,u+2,v+2),outline='red');dr.text((u,v),str(j['joint_index']),fill='red')
        dr.text((4,4),f'native {index}: triangulated landmarks',fill='red');images.append(im)
    sheet=Image.new('RGB',(512*5,512),'white')
    for i,im in enumerate(images):sheet.paste(im.resize((512,512)),(512*i,0))
    sheet.save(out/'correspondence_review.png')
