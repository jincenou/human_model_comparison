"""Build a self-contained comparison from already scored source-camera pairs."""
import argparse,base64,hashlib,html,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import human_metric_batch_v1 as b

LABELS={'ours':'Ours','ah':'Asset Harvester','trellis_original':'TRELLIS','trellis2':'TRELLIS.2','omnire':'OmniRe','splatad':'SplatAD'}
CORE_METRICS=['PSNR','SSIM','LPIPS','IoU']

def select_cases(summary,requested_methods,count,requested_ids=None):
 methods=list(summary['coverage']) if requested_methods is None else list(requested_methods)
 if not methods or len(methods)!=len(set(methods)):raise ValueError('Choose distinct methods')
 if set(methods)-set(summary['coverage']):raise ValueError('Requested method is not in the active summary')
 if count<1:raise ValueError('Count must be positive')
 valid={method:{r['id'] for r in summary['rows'] if r['method']==method and r['status']=='scored'
                and all(r.get('metrics',{}).get(k,{}).get('status') in ('ok','positive_infinity') for k in CORE_METRICS)} for method in methods}
 common=sorted(set.intersection(*(valid[method] for method in methods)))
 ids=list(requested_ids) if requested_ids is not None else common[:count]
 if not ids or len(ids)!=len(set(ids)) or not set(ids).issubset(common):raise ValueError('No requested common scored cases are available')
 return methods,common,ids


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='+');ap.add_argument('--count',type=int,default=3)
 ap.add_argument('--methods',nargs='+',help='Default: all methods in the active summary')
 ap.add_argument('--out-dir',type=Path,default=b.H/'reports/common_source_comparison');a=ap.parse_args()
 run=b.H/'reports/source_camera_metric_cohort_v3';snapshot=(run/'summary.json').read_bytes()
 summary=json.loads(snapshot);snapshot_hash=hashlib.sha256(snapshot).hexdigest()
 methods,common,ids=select_cases(summary,a.methods,a.count,a.ids)
 rows={(r['id'],r['method']):r for r in summary['rows']};out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 labels={method:LABELS.get(method,method) for method in methods};columns=len(methods)+1
 selection_rule='explicit IDs within the selected-method common-valid set; no score filtering' if a.ids is not None else 'first count IDs in sorted selected-method common-valid set; no score filtering'
 selection_text='本页展示指定的共同有效人物' if a.ids is not None else '本页按编号展示所选方法共同有效人物中的前'+str(len(ids))+'人'
 parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人体共同源视角对比</title><style>body{font:15px system-ui;margin:24px;background:#f5f5f5;color:#222}section{background:white;margin:20px 0;padding:16px}.grid{display:grid;grid-template-columns:repeat('+str(columns)+',minmax(160px,1fr));gap:8px;min-width:'+str(columns*168)+'px}.scroll{overflow:auto}img{width:100%;background:white}small{display:block;color:#555}h2{margin:5px 0 10px}p{line-height:1.6}.cell{text-align:center}</style><h1>人体共同源视角对比</h1><p>同一人物、真实曝光和源相机；统一白底及参考确定的画面范围。保留模型自身的姿态、体型和外观差异。</p>',
 '<p>'+selection_text+'，未按分数挑选。共同有效集合 '+str(len(common))+'/'+str(summary['denominator'])+' 人。属于已观测输入的重建比较，不代表完整100人的排名。</p>',
 '<p>AH沿用自身朝向/尺寸预测作固定相机转换；TRELLIS系列使用源图身体关键点进行整体刚体配准，未改关节。Ours沿用原始估计根姿态，OmniRe/SplatAD沿用场景坐标。</p>']
 if 'trellis2' in methods:parts.append('<p>TRELLIS.2 展示 native base color (unlit)：原生基础颜色与自身透明度合成白底，未加入 HDR/BRDF 光照或曝光匹配。</p>')
 mosaic=Image.new('RGB',(256*columns,304*len(ids)),'white');draw=ImageDraw.Draw(mosaic);evidence=[]
 for rownum,sid in enumerate(ids):
  ref=rows[(sid,methods[0])]['reference'];c=b.read(ref['camera_contract'])
  for key in ['reference_rgb','reference_mask','camera_contract']:b.verified(ref[key],ref[key+'_sha256'])
  reference=b.m.prepare_reference(np.asarray(Image.open(ref['reference_rgb']).convert('RGB')),np.asarray(Image.open(ref['reference_mask']).convert('L')))
  assert list(reference['roi'])==ref['roi_xyxy']==c['roi_xyxy'];arrays=[reference['rgb']];metrics=[];records=[]
  for method in methods:
   row=rows[(sid,method)];b.verified(row['result'],row['result_sha256']);result=b.read(row['result']);b.verified(result['registration_record'],result['registration_sha256']);cert=b.read(result['registration_record'])
   assert result['id']==sid and result['method']==method and cert['identity']==sid and cert['identity_key']==ref['identity_key']
   assert cert['camera_sha256']==ref['camera_contract_sha256'] and cert['timestamp_us']==ref['timestamp_us']
   for key in ['source_rgb','source_alpha']:b.verified(cert[key],cert[key+'_sha256'])
   rgb=np.asarray(Image.open(cert['source_rgb']).convert('RGB'));alpha=np.asarray(Image.open(cert['source_alpha']).convert('L'))
   assert list(rgb.shape[1::-1])==c['render_wh'] and alpha.shape==rgb.shape[:2]
   pred=b.m.prepare_prediction(rgb,alpha,reference['roi'],cert['prediction_encoding'])
   arrays.append(pred['rgb']);metrics.append(result['metrics']);records.append(dict(method=method,result=row['result'],result_sha256=row['result_sha256']))
  parts.append('<section><h2>'+sid+'</h2><small>原始人物高度 '+str(ref['reference_pixel_height'])+' px · 真实曝光 '+str(ref['timestamp_us'])+' μs</small><div class="scroll"><div class="grid">')
  for col,array in enumerate(arrays):
   name='真实参考' if col==0 else labels[methods[col-1]];im=Image.fromarray(array);buf=io.BytesIO();im.save(buf,format='PNG');encoded=base64.b64encode(buf.getvalue()).decode()
   parts.append('<div class="cell"><b>'+html.escape(name)+'</b>')
   variant='native base color (unlit)' if col and methods[col-1]=='trellis2' else '&nbsp;'
   parts.append('<small style="height:18px">'+variant+'</small>')
   parts.append('<img loading="lazy" src="data:image/png;base64,'+encoded+'">')
   if col:
    for key in ['PSNR','SSIM','LPIPS','IoU']:
     v=metrics[col-1][key];value=f"{v['value']:.4f}" if v['status']=='ok' else '∞' if v['status']=='positive_infinity' else 'NA'
     parts.append('<small>'+key+': '+value+'</small>')
   parts.append('</div>');mosaic.paste(im.resize((256,256)),(256*col,304*rownum+40));draw.text((256*col+8,304*rownum+5),sid+' '+('Reference' if col==0 else labels[methods[col-1]]),fill='black')
   if col and methods[col-1]=='trellis2':draw.text((256*col+8,304*rownum+20),'native base color (unlit)',fill='black')
  parts.append('</div></div></section>');evidence.append(dict(id=sid,reference=ref,records=records))
 parts.append('</html>');(out/'index.html').write_text(''.join(parts));mosaic.save(out/'preview.jpg',quality=94)
 b.save(out/'manifest.json',dict(ids=ids,methods=methods,method_names=labels,selection_sha256=summary['selection_sha256'],selection_rule=selection_rule,
  requested_count=a.count,common_ids=common,common_count=len(common),denominator=summary['denominator'],summary_source=str(run/'summary.json'),
  snapshot_summary_hash=snapshot_hash,snapshot_summary_hash_algorithm='sha256',summary_sha256=snapshot_hash,
  method_coverage={method:summary['coverage'][method] for method in methods},rows=evidence,script_sha256=b.m.sha(__file__),html_sha256=b.m.sha(out/'index.html'),
  appearance_variants={'trellis2':'native base color (unlit), native own alpha on white'} if 'trellis2' in methods else {},
  scope='source-observed appearance after disclosed method coordinate mapping'))
 print(out/'index.html')

if __name__=='__main__':main()
