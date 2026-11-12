"""All frozen identities, with source views, synchronized orbits and explicit NA.

Reuses scored and rendered files. No asset generation, camera fitting or metric
changes. The optional ZIP preserves relative paths for offline browsing.
"""
import argparse
import html
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from common_human_orbit import H, OUT, METHODS, read, save, sha, checked
import human_source_metrics_v1 as metric
from export_active_metric_tables import NAMES
from active_human_results import resolve

DEST=resolve('gallery')
STYLE='''body{font:16px/1.6 system-ui;margin:24px auto;padding:0 20px;max-width:1900px;color:#243042;background:#f3f5f7}a{color:#195da8}h1{font-size:27px}p{max-width:1300px}.controls{position:sticky;top:0;background:#f3f5f7f2;padding:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}button,select,input{font:inherit;padding:7px}button{cursor:pointer}input[type=range]{width:270px}.scroll{overflow-x:auto}.grid{display:grid;grid-template-columns:repeat(7,minmax(180px,1fr));gap:10px;min-width:1320px}.card{background:white;padding:8px;text-align:center;margin:0}.card img{width:100%;display:block}.placeholder{min-height:240px;display:flex;align-items:center;justify-content:center;background:#eef1f4;padding:12px;overflow-wrap:anywhere}.metrics{font-size:13px}.reason{font-size:12px;overflow-wrap:anywhere}.catalog{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}.catalog a{background:white;padding:10px;text-decoration:none}.catalog img{width:100%;display:block}.badge{font-size:13px;color:#555}small{color:#596575}[hidden]{display:none!important}'''
JS='''const data=JSON.parse(document.getElementById("gallery-data").textContent);
const slider=document.getElementById("angle");let mode="source";
function show(){const i=Number(slider.value);document.getElementById("view-label").textContent=mode==="source"?"真实源相机":`相对源相机 ${i*22.5}°`;
for(const p of data.panels){const img=document.getElementById(p.method+"-image");const empty=document.getElementById(p.method+"-empty");const path=mode==="source"?p.source:p.frames[i];
if(path){img.src=path;img.hidden=false;empty.hidden=true;}else{img.removeAttribute("src");img.hidden=true;empty.hidden=false;empty.textContent=mode==="source"?p.sourceMessage:p.orbitMessage;}
document.getElementById(p.method+"-metrics").hidden=mode!=="source";}
}
document.getElementById("source").onclick=()=>{mode="source";show();};
document.getElementById("orbit").onclick=()=>{mode="orbit";show();};
slider.oninput=()=>{mode="orbit";show();};
document.getElementById("prev").onclick=()=>{mode="orbit";slider.value=(Number(slider.value)+15)%16;show();};
document.getElementById("next").onclick=()=>{mode="orbit";slider.value=(Number(slider.value)+1)%16;show();};
document.querySelectorAll("[data-angle]").forEach(b=>b.onclick=()=>{mode="orbit";slider.value=b.dataset.angle;show();});show();
'''


def relative(path):
    return os.path.relpath(path, DEST)


def header(title):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><link rel="stylesheet" href="style.css"><body>'


def preview(path,array,sid,name):
    if not path.exists():
        path=DEST/'source'/sid/(name+'.png');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():Image.fromarray(array).save(path)
    assert np.array_equal(np.asarray(Image.open(path)),array),str(path)
    return path


def build(archive=False):
    DEST.mkdir(exist_ok=True)
    summary_path=resolve('core_summary');summary=read(summary_path)
    checked(summary['selection'],summary['selection_sha256'])
    selected=read(summary['selection'])['objects'];assert len(selected)==100
    rows={(r['id'],r['method']):r for r in summary['rows']}
    files=set();evidence=[];catalog=[]
    (DEST/'style.css').write_text(STYLE);(DEST/'viewer.js').write_text(JS)
    for human in sorted(selected,key=lambda r:r['id']):
        sid=human['id'];ref=rows[sid,'ours']['reference'];camera=read(ref['camera_contract'])
        for key in ['camera_contract','reference_rgb','reference_mask']:checked(ref[key],ref[key+'_sha256'])
        assert ref['timestamp_us']==human['t_ref_us'] and ref['identity_key']==human['identity_key']
        gt=metric.prepare_reference(np.asarray(Image.open(ref['reference_rgb']).convert('RGB')),
                                    np.asarray(Image.open(ref['reference_mask']).convert('L')))
        assert list(gt['roi'])==ref['roi_xyxy']
        if ref.get('reference_revision'):
            checked(ref['reference_revision'],ref['reference_revision_sha256'])
            assert ref['evaluation_roi_authority']=='reviewed mask revision; camera contract retains historical ROI'
        else:assert ref['roi_xyxy']==camera['roi_xyxy']
        media=DEST/'source'/sid
        reference=preview(media/'reference.png',gt['rgb'],sid,'reference')
        files.add(reference);panels=[];records=[];source_count=orbit_count=0
        protocol_path=OUT/sid/'protocol.json';protocol=read(protocol_path)
        assert protocol['selection_sha256']==summary['selection_sha256']
        assert protocol['timestamp_us']==ref['timestamp_us']
        assert protocol['reference_camera_sha256']==ref['camera_contract_sha256']
        for method in METHODS:
            row=rows[sid,method];source=None;frames=[None]*16;values={};record=dict(method=method,source_status=row['status'],reason=row.get('reason'))
            if row['status']=='scored':
                checked(row['result'],row['result_sha256']);result=read(row['result'])
                checked(result['registration_record'],result['registration_sha256']);cert=read(result['registration_record'])
                assert cert['identity_key']==ref['identity_key'] and cert['timestamp_us']==ref['timestamp_us']
                assert cert['camera_sha256']==ref['camera_contract_sha256']
                for key in ['source_rgb','source_alpha']:checked(cert[key],cert[key+'_sha256'])
                pred=metric.prepare_prediction(np.asarray(Image.open(cert['source_rgb']).convert('RGB')),
                    np.asarray(Image.open(cert['source_alpha']).convert('L')),gt['roi'],cert['prediction_encoding'])
                source=preview(media/(method+'.png'),pred['rgb'],sid,method)
                files.add(source);source_count+=1
                values={key:(f'{result["metrics"][key]["value"]:.4f}' if result['metrics'][key]['status']=='ok' else '∞' if result['metrics'][key]['status']=='positive_infinity' else 'NA')for key in ['PSNR','SSIM','LPIPS','IoU']}
                record.update(result=row['result'],result_sha256=row['result_sha256'],source_png_sha256=sha(source))
            manifest_path=OUT/sid/method/'manifest.json'
            if manifest_path.exists():
                manifest=read(manifest_path)
                assert row['status']=='scored' and manifest['protocol_sha256']==sha(protocol_path)
                assert manifest['timestamp_us']==ref['timestamp_us'] and len(manifest['frames'])==16
                assert manifest['registration']['sha256']==result['registration_sha256']
                for i,f in enumerate(manifest['frames']):
                    assert f['index']==i and f['angle_degrees']==i*22.5
                    p=Path(f['rgb_roi']);checked(p,f['rgb_roi_sha256']);assert Image.open(p).size==(512,512)
                    files.add(p);frames[i]=relative(p)
                orbit_count+=1;record.update(orbit_manifest=str(manifest_path),orbit_manifest_sha256=sha(manifest_path))
            source_message={'source_registration_unavailable':'NA · 配准不可用',
                'native_unavailable_in_existing_scene':'NA · 无独立人体',
                'native_generation_failed':'NA · 生成失败',
                'registration_pending':'待完成'}.get(row['status'],'暂无结果')
            orbit_message='尚无共同环绕输出' if source else source_message
            panels.append(dict(method=method,label=NAMES[method],source=relative(source)if source else None,frames=frames,metrics=values,sourceMessage=source_message,orbitMessage=orbit_message))
            records.append(record)
        doc=[header(sid+' 人体资产对比'),'<p><a href="index.html">← 全部100人</a></p>',f'<h1>{sid} · 模型对比</h1>',
             f'<p class="badge">同一时刻 · 统一视角 · 白底512×512 · 源视角 {source_count}/6 · 环绕 {orbit_count}/6</p>',
             '<div class="controls"><button id="source">源视角＋指标</button><button id="orbit">共同环绕</button><button id="prev">上一角度</button><input id="angle" aria-label="共同角度" type="range" min="0" max="15" value="0" step="1"><button id="next">下一角度</button><b id="view-label"></b>',
             ''.join(f'<button data-angle="{i}">{i*22.5:g}°</button>'for i in [0,4,8,12]),'</div><div class="scroll"><div class="grid">',
             '<figure class="card"><figcaption>真实参考（源视角）</figcaption><img alt="真实源视角" src="'+relative(reference)+'"></figure>']
        for p in panels:
            method=p['method'];scores=' · '.join(k+(' ↓'if k=='LPIPS'else' ↑')+': '+v for k,v in p['metrics'].items())
            src=' src="'+p['source']+'"'if p['source']else' hidden'
            doc += [f'<figure class="card"><figcaption>{html.escape(p["label"])}</figcaption><img id="{method}-image" alt="{html.escape(p["label"])}对应视角"{src}>',
                    f'<div id="{method}-empty" class="placeholder"'+(' hidden'if p['source']else'')+'>'+html.escape(p['sourceMessage'])+'</div>',
                    f'<p id="{method}-metrics" class="metrics">{html.escape(scores)}</p></figure>']
        doc += ['</div></div><p><small>环绕以源相机为0°；其他角度无真实参考。<a href="'+relative(protocol_path)+'">详细协议</a></small></p>',
                '<script type="application/json" id="gallery-data">'+json.dumps(dict(id=sid,panels=panels),ensure_ascii=False).replace('<','\\u003c')+'</script><script src="viewer.js"></script></body></html>']
        page=DEST/(sid+'.html');page.write_text(''.join(doc))
        catalog.append(f'<a class="person" data-id="{sid}" href="{sid}.html"><b>{sid}</b><img loading="lazy" alt="{sid}真实参考" src="{relative(reference)}"><span class="badge">源视角 {source_count}/6 · 环绕 {orbit_count}/6</span></a>')
        evidence.append(dict(id=sid,source_count=source_count,orbit_count=orbit_count,reference_sha256=sha(reference),page_sha256=sha(page),methods=records))
        print('built',sid,source_count,orbit_count,flush=True)
    doc=[header('固定100人人体资产完整对比'),'<h1>固定100人 · 人体资产完整对比</h1>',
         '<p>6种模型 · 源视角指标 · 16方位环绕</p>',
         '<p><a href="'+relative(summary_path.parent/'tables/RESULTS.md')+'">核心指标与覆盖率</a> · <a href="'+relative(resolve('edp_root')/'tables/RESULTS.md')+'">ED-P与部位覆盖</a> · <a href="../dynamic_source_clip_v1/tables/RESULTS.md">动态指标</a></p>',
         '<label>查找人物 <input id="search" placeholder="例如 HN0004"></label><p id="count">100人</p><div class="catalog">',''.join(catalog),
         '</div><script>document.getElementById("search").oninput=function(){const q=this.value.trim().toUpperCase();let n=0;for(const a of document.querySelectorAll(".person")){a.hidden=!a.dataset.id.includes(q);if(!a.hidden)n++;}document.getElementById("count").textContent=n+"人";};</script></body></html>']
    (DEST/'index.html').write_text(''.join(doc))
    save(DEST/'manifest.json',dict(status='full_cohort_gallery_built_pending_full_visual_review',selection_sha256=summary['selection_sha256'],
         summary_sha256=sha(summary_path),identities=100,source_pairs=sum(r['source_count']for r in evidence),orbit_pairs=sum(r['orbit_count']for r in evidence),
         rows=evidence,script_sha256=sha(__file__),index_sha256=sha(DEST/'index.html'),viewer_sha256=sha(DEST/'viewer.js'),
         image_files={str(p):sha(p)for p in sorted(files)},generated_scores=False,generated_renders=False))
    if archive:
        files.update(DEST.glob('*'))
        for folder in [summary_path.parent/'tables',resolve('edp_root')/'tables',H/'reports/dynamic_source_clip_v1/tables']:
            files.update(folder.glob('*'))
        target=DEST.with_suffix('.zip')
        with zipfile.ZipFile(target.with_suffix('.zip.tmp'),'w',compression=zipfile.ZIP_STORED)as z:
            for p in sorted(files):
                if p.is_file():z.write(p,p.relative_to(H/'reports'))
        target.with_suffix('.zip.tmp').replace(target)
        print('archive',target,flush=True)
    print(DEST/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');a=parser.parse_args();build(a.archive)
