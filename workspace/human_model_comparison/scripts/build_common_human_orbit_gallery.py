"""Offline synchronized orbit viewer; no score, camera or asset optimization."""
import argparse
import base64
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw
from common_human_orbit import H,OUT,METHODS,read,save,sha,checked,inputs

LABELS=dict(ours='Ours',ah='Asset Harvester',trellis_original='TRELLIS',omnire='OmniRe',splatad='SplatAD',trellis2='TRELLIS.2')
def data_url(path):
    mime='image/jpeg'if str(path).endswith('.jpg')else'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode()


def build(sid):
    root=OUT/sid;protocol=read(root/'protocol.json');digest=sha(root/'protocol.json')
    _,_,reference,_,_,_=inputs(sid,'ours');panels=[];manifests={}
    for method in METHODS:
        path=root/method/'manifest.json';m=read(path)
        assert m['id']==sid and m['method']==method and m['protocol_sha256']==digest
        assert m['timestamp_us']==protocol['timestamp_us'] and len(m['frames'])==16
        frames=[]
        for i,r in enumerate(m['frames']):
            assert r['index']==i and r['angle_degrees']==i*22.5
            checked(r['rgb_roi'],r['rgb_roi_sha256']);frames.append(data_url(r['rgb_roi']))
        panels.append(dict(key=method,label=LABELS[method],frames=frames))
        manifests[method]=dict(path=str(path),sha256=sha(path))
    # The real reference remains labeled as the observed source view at every angle.
    source=H/'reports/source_camera_metric_cohort_v3/media'/sid/'reference.png'
    assert Image.open(source).size==(512,512)
    contact=Image.new('RGB',(6*256,4*280),'white');draw=ImageDraw.Draw(contact)
    for col,method in enumerate(METHODS):
        for row,i in enumerate([0,4,8,12]):
            image=Image.open(root/method/f'{i:02d}_rgb_roi.png').resize((256,256))
            contact.paste(image,(col*256,row*280+24));draw.text((col*256+5,row*280+5),f'{LABELS[method]} | {i*22.5:g} deg',fill='black')
    contact.save(root/'cardinal_comparison.jpg')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人体固定姿态共同环绕</title><style>body{margin:25px auto;padding:0 20px;max-width:1800px;font:17px/1.6 system-ui;color:#20252b;background:#f5f6f7}h1{font-size:28px}p{max-width:1200px}.controls{position:sticky;top:0;background:#f5f6f7ee;padding:12px 0;z-index:2;display:flex;align-items:center;gap:12px;flex-wrap:wrap}input{width:340px}button{padding:8px 14px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(6,minmax(180px,1fr));gap:10px;overflow-x:auto}.card{margin:0;background:white;padding:8px;text-align:center}.card img{width:100%;display:block}.reference{display:flex;gap:20px;align-items:center}.reference img{width:200px;background:white}.wide{width:100%;background:white}small{color:#555}</style>'''
    doc+=f'<h1>{html.escape(sid)} · 六方法固定姿态共同环绕</h1><p>同一人物、同一真实时刻，所有模型保持各自预测姿态。拖动滑块可同步查看16个角度。白底、裁剪区域和环绕相机设置共同使用；侧背面缺失、扁平和伪影均保留。</p>'
    doc+='<div class="reference"><img alt="真实源视角参考" src="'+data_url(source)+'"><p><b>真实参考：仅此源视角</b><br>0°沿用真实源相机方向；90°、180°等是相对相机角度，并非统一的人体正面或背面标签。其他角度没有真实参考，因此不计算背面PSNR或LPIPS。</p></div>'
    doc+='<div class="controls"><button id="prev">上一角度</button><input aria-label="共同环绕角度" id="angle" type="range" min="0" max="15" value="0" step="1"><b id="label">0°</b><button id="next">下一角度</button>'+''.join(f'<button data-index="{i}">{i*22.5:g}°</button>'for i in [0,4,8,12])+'</div><div class="grid">'
    for item in panels:doc+=f'<figure class="card"><figcaption>{html.escape(item["label"])}</figcaption><img id="{item["key"]}" src="{item["frames"][0]}" alt="{html.escape(item["label"])} 当前共同角度"></figure>'
    doc+='</div><p><small>显示尺度规则：以源相机为原点，统一各资产的深度单位，同时保持源视角投影不变。没有按各模型的预测框居中缩放，也没有修正关节或挑选最佳角度。这是共同显示坐标下的比较，不代表物理尺度精度已验证；已有源视角成绩不变。</small></p><details><summary>四个相对方位总览与复现记录</summary><img class="wide" alt="六方法四方位总览" src="'+data_url(root/'cardinal_comparison.jpg')+'"><p><a href="protocol.json">共同相机与显示协议</a> · <a href="gallery_manifest.json">输入与方法记录</a></p></details>'
    doc+='<script>const panels='+json.dumps(panels)+';const slider=document.getElementById("angle");function show(i){i=(i+16)%16;slider.value=i;document.getElementById("label").textContent=(i*22.5)+"°";for(const p of panels)document.getElementById(p.key).src=p.frames[i];}slider.oninput=()=>show(+slider.value);document.getElementById("prev").onclick=()=>show(+slider.value-1);document.getElementById("next").onclick=()=>show(+slider.value+1);document.querySelectorAll("[data-index]").forEach(b=>b.onclick=()=>show(+b.dataset.index));show(0);</script></html>'
    (root/'index.html').write_text(doc)
    save(root/'gallery_manifest.json',dict(id=sid,status='gallery_built_pending_visual_acceptance',methods=METHODS,angles=16,protocol=str(root/'protocol.json'),protocol_sha256=digest,method_manifests=manifests,reference_source=str(source),reference_source_sha256=sha(source),gallery_sha256=sha(root/'index.html'),contact_sha256=sha(root/'cardinal_comparison.jpg'),physical_scale_claim=False,metrics_produced=False,script_sha256=sha(__file__)))
    print(root/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',required=True);a=p.parse_args()
    for sid in a.ids:build(sid)
