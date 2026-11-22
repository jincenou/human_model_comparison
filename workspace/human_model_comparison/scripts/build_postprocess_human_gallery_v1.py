"""Offline source/raw/prediction-segmented comparison; no 3D-cleanup claim."""
from human_asset_benchmark.paths import expand as _hmc_expand
import base64
import collections
import html
import io
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from native_reaudit_common import read,save,sha
from human_source_metrics_v1 import crop

H=Path(__file__).resolve().parents[1]
ROOT=H/'reports/native_human_export_reaudit_v1/postprocess_v1'
METHODS=['splatad','omnire','streetgs']
LABEL={'splatad':'SplatAD','omnire':'OmniRe','streetgs':'StreetGS'}
FONT=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf')
NOTE=('同一固定规则作用于三种场景方法的全部100个输出。RTMDet人体框来自预测渲染，SAM2仅读取预测RGB；'
      '既定ROI提供粗定位，因此不声称完全无标注辅助。原照片只用于本页对照，原照片和参考mask像素不进入分割。'
      '保留区域RGB/alpha不变，区域外置白/透明；无人体检测时保留空白结果，异常按固定规则回退原图。'
      '空白结果仍进入图像评测，不等于原生三维生成失败。此步骤仅为二维图像前景后处理，没有清理高斯、修复人体或改变三维几何。')


def checked_records():
    protocol=read(ROOT/'protocol.json');inputs=read(ROOT/'inputs.json')
    assert sha(ROOT/'inputs.json')==protocol['input_manifest_sha256'] and protocol['expected']==300
    expected={(x['method'],x['id']):x for x in inputs['records']};assert len(expected)==300
    rows=[]
    for method in METHODS:
        for sid in inputs['ids']:
            path=ROOT/'outputs'/method/sid/'postprocess.json';row=read(path);original=expected[(method,sid)]
            assert row['id']==sid and row['method']==method and row['protocol_sha256']==sha(ROOT/'protocol.json')
            assert row['status'] in ['postprocess_segmented','postprocess_no_person','postprocess_exception_fallback','postprocess_empty_mask']
            for key in ['source_validation','original_rgb','original_alpha']:
                assert row[key]==original[key] and row[key+'_sha256']==original[key+'_sha256']==sha(row[key])
            outputs=row.get('outputs',row)
            for key in ['clean_rgb','clean_alpha','mask']:assert sha(outputs[key])==outputs[key+'_sha256']
            assert row['reference_mask_pixels_read'] is False and row['original_source_photo_pixels_read'] is False
            assert row['geometry_changed'] is False and row['all_entries_retained_for_scoring'] is True
            assert row['kept_pixels_rgb_exact'] and row['kept_pixels_alpha_exact']
            v=read(row['source_validation']);assert v['identity_key']==row['identity_key'] and v['timestamp_us']==row['timestamp_us']
            row['_record']=str(path);row['_record_sha256']=sha(path);row['_reference']=v['source_reference']
            rows.append(row)
    return rows


def main():
    assert shutil.disk_usage(H).free>10*2**30
    rows=checked_records();out=ROOT/'gallery';out.mkdir(exist_ok=True)
    font=ImageFont.truetype(FONT,18);cards=[];manifest=[];counts={}
    for method in METHODS:
        chosen=[x for x in rows if x['method']==method];counts[method]=dict(collections.Counter(x['status'] for x in chosen))
        cards.append('<h2>'+LABEL[method]+' · 100/100</h2>')
        for row in chosen:
            ref=row['_reference'];roi=row['roi_xyxy'];outputs=row.get('outputs',row)
            paths=[ref['reference_rgb'],row['original_rgb'],outputs['clean_rgb']]
            assert sha(paths[0])==ref['reference_rgb_sha256']
            canvas=Image.new('RGB',(768,292),'white');draw=ImageDraw.Draw(canvas)
            for i,(title,path) in enumerate(zip(['Source','Native raw','2D segmented'],paths)):
                draw.text((i*256+8,5),title,font=font,fill='#172535')
                pixels=np.asarray(Image.open(path).convert('RGB'))
                im=Image.fromarray(crop(pixels,roi)).resize((256,256),Image.Resampling.LANCZOS)
                canvas.paste(im,(i*256,36))
            buffer=io.BytesIO();canvas.save(buffer,format='JPEG',quality=93,subsampling=0)
            uri='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()
            key=method+'_'+row['id'];cards.append('<section data-method="'+method+'" data-id="'+row['id']+'"><h3>'+LABEL[method]+' · '+row['id']+'</h3><p>'+html.escape(row['status'])+'</p><img alt="'+key+' source raw segmented" src="'+uri+'"></section>')
            manifest.append(dict(method=method,id=row['id'],postprocess_record=row['_record'],postprocess_sha256=row['_record_sha256'],status=row['status'],
                source_validation=row['source_validation'],source_validation_sha256=row['source_validation_sha256'],source=paths[0],source_sha256=ref['reference_rgb_sha256'],
                raw=paths[1],raw_sha256=row['original_rgb_sha256'],clean=paths[2],clean_sha256=outputs['clean_rgb_sha256'],roi_xyxy=roi,
                embedded_row_jpeg_sha256=__import__('hashlib').sha256(buffer.getvalue()).hexdigest()))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人体预测图像前景后处理 · 300条对照</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:24px auto;padding:16px;background:#eef2f6}section{background:white;padding:12px;margin:16px 0}img{width:100%;height:auto}h3,p{margin:4px 0}input{padding:10px;width:280px}.note{padding:16px;background:#fff6db}</style><h1>三模型人体：原生图像与二维分割后处理</h1><p>300/300条，按方法与冻结编号展示；三列使用同一ROI，不单独对齐或缩放人体。</p><p class="note">'''+html.escape(NOTE)+'''</p><p><input placeholder="搜索 HN编号或方法" oninput="let v=this.value.trim().toLowerCase();document.querySelectorAll('section').forEach(s=>s.hidden=!(s.dataset.id+' '+s.dataset.method).toLowerCase().includes(v))"></p>'''+''.join(cards)+'</html>'
    (out/'index.html').write_text(page)
    save(out/'manifest.json',dict(status='complete',count=300,method_counts=counts,self_contained_html=True,
        protocol_sha256=sha(ROOT/'protocol.json'),html=str(out/'index.html'),html_sha256=sha(out/'index.html'),rows=manifest,
        source_photo_used_only_for_display=True,prediction_only_segmentation=True,annotation_assisted_roi=True,geometry_modified=False,
        raw_native_gallery_modified=False,script_sha256=sha(__file__)))
    print(str(out/'index.html'),counts)


if __name__=='__main__':main()
