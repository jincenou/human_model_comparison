"""Display the two unchanged SplatAD node ablations without masking source images."""
import base64
from html import escape
import json
from pathlib import Path

from PIL import Image, ImageDraw
from build_streetgs_scene_audit_gallery_v1 import expanded_roi, fitted, font


H = Path(__file__).resolve().parents[1]
OUT = H / 'reports/splatad_human_scene_audit_v1'
IDS = ('HN0108', 'HN0273')


def uri(path):
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def main():
    cards, overview_rows = [], []
    for sid in IDS:
        directory = OUT / sid
        record = json.loads((directory/'audit.json').read_text())
        assert record['status'] == 'passed'
        ims = {'source': Image.open(record['source_rgb']).convert('RGB')}
        for name in ('fullscene', 'without_target', 'target_only', 'target_only_alpha'):
            ims[name] = Image.open(directory/(name+'.png')).convert('RGB')
        size = ims['source'].size
        assert all(im.size == size for im in ims.values())
        roi = expanded_roi(record['reference']['roi_xyxy'], size, 1.3)
        width, margin, gap = 1800, 28, 16
        full_width = (width-2*margin-gap)//2
        full_height = round(full_width*size[1]/size[0])
        crop_width = (width-2*margin-4*gap)//5
        full_y = 104; crop_label_y = full_y+full_height+28; crop_y = crop_label_y+35
        canvas = Image.new('RGB', (width,crop_y+crop_width+72), '#f8fafc')
        draw = ImageDraw.Draw(canvas)
        draw.text((margin,24), f'{sid} | {record["scene"]} | SplatAD', fill='#111827', font=font(28))
        for i,(name,title) in enumerate([('source','Original full frame'),('fullscene','SplatAD full scene (frozen source camera)')]):
            x = margin+i*(full_width+gap)
            draw.text((x,68),title,fill='#111827',font=font(23))
            panel = ims[name].resize((full_width,full_height),Image.Resampling.LANCZOS)
            box=tuple(round(v*(full_width/size[0] if j%2==0 else full_height/size[1])) for j,v in enumerate(roi))
            ImageDraw.Draw(panel).rectangle(box,outline='#ff3344',width=3)
            canvas.paste(panel,(x,full_y))
        for i,(name,title) in enumerate([('source','Source'),('fullscene','Full scene'),('without_target','Without target node'),('target_only','Target node only'),('target_only_alpha','Target alpha')]):
            x=margin+i*(crop_width+gap)
            draw.text((x,crop_label_y),title,fill='#111827',font=font(20))
            canvas.paste(fitted(ims[name].crop(roi),(crop_width,crop_width)),(x,crop_y))
        draw.text((margin,crop_y+crop_width+20),f'Same crop: {roi}; all target-id Gaussians retained; no image masking or enhancement.',fill='#475569',font=font(19))
        path=directory/'comparison.jpg';canvas.save(path,quality=95,subsampling=0)
        overview_rows.append(fitted(canvas,(1200,round(canvas.height*1200/canvas.width))))
        diag=record['diagnostics']; imuri=uri(path)
        cards.append(f'''<section><h2>{sid} · {escape(record['scene'])}</h2>
<a href="{imuri}" download="{sid}_splatad_scene_audit.jpg"><img src="{imuri}" alt="{sid} SplatAD 原场景与独立目标核查"></a>
<p>当前导出包含该目标节点的全部 {record['target_gaussians']:,} 个高斯。目标 alpha 在参考前景内大于 0.5 的像素比例为 {diag['target_alpha_gt_0_5_fraction_in_gt']:.1%}。这是同一图上的诊断统计，不是新增论文指标。</p>
<p><a href="{imuri}" download="{sid}_splatad_scene_audit.jpg">下载拼图</a> · <a href="{sid}/audit.json">完整核查记录</a> · <a href="{sid}/fullscene.png">原场景 PNG</a> · <a href="{sid}/without_target.png">移除目标节点 PNG</a> · <a href="{sid}/target_only.png">独立目标 PNG</a> · <a href="{sid}/target_only_alpha.png">目标 alpha PNG</a></p>
<details><summary>CNN 上下文干预诊断</summary><p>保持目标节点 albedo 与 alpha 不变，临时使用整场景 CNN 系数解码，以判断局部场景语境对解码的影响。此图依赖整场景信息，不能作为独立资产、优化结果或正式指标输入。</p>
<p><a href="{sid}/target_fullscene_cnn_coeff_diagnostic.png">CNN 系数替换诊断 PNG</a> · <a href="{sid}/target_albedo_white_diagnostic.png">目标 albedo 白底诊断 PNG</a></p></details></section>''')
    overview=Image.new('RGB',(1200,sum(im.height for im in overview_rows)+20),'#cbd5e1')
    y=0
    for im in overview_rows:overview.paste(im,(0,y));y+=im.height+20
    overview.save(OUT/'overview.jpg',quality=95,subsampling=0)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SplatAD 人体节点导出核查</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#eef2f7;color:#172033;margin:auto;padding:24px;max-width:1800px}section{background:white;padding:22px;margin:26px 0;border-radius:12px}img{width:100%;height:auto;display:block}a{color:#1254a0}p{max-width:1400px}aside{background:white;padding:16px;border-left:4px solid #ca3344}</style>
<h1>SplatAD：完整场景、移除目标与独立节点</h1>
<p>两例均复用已有训练检查点、同一曝光和冻结源相机。没有重新训练、修改高斯参数或按 GT mask 抹除图片。原照片的模糊照实保留；五列使用相同坐标范围裁切。</p>
<aside>“Without target node” 保留原场景其它全部高斯，只移除目标 actor-id；“Target node only” 保留这个 actor-id 的全部高斯，不按三维框或中心再次裁剪。alpha 白色表示较不透明，黑色表示透明。若同一目标在移除节点后仍存在，需检查背景或其它节点承担，并先排除后方其他人显露的情况；若整场景中的肢体已经透明，不能仅凭白底更显眼就归因于导出丢失。</aside>
<p>SplatAD 使用原生 CNN 解码，不是普通 SH 直接着色。核查脚本同时验证当前独立图和整场景图能复现历史结果，并记录 CNN 上下文诊断；两类问题需要分开判断。本页对比 JPG 已内嵌，可单文件离线查看；PNG 和核查记录链接需要相应文件夹。</p>'''+''.join(cards)+'</html>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print(OUT/'index.html')


if __name__=='__main__':main()
