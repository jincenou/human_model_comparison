"""Create an immutable advisor snapshot from verified existing result tables.

No scoring, filtering, metric fitting or dependency installation. XLSX uses
standard inline-string/numeric cells and is accompanied by UTF-8 CSV tables.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import zipfile
from active_human_results import resolve

H = Path(__file__).resolve().parents[1]
OUT = resolve('advisor')
METHODS = ['ours','ah','trellis_original','trellis2','omnire','splatad']
NAMES = dict(zip(METHODS,['Ours','AH','TRELLIS','TRELLIS.2','OmniRe','SplatAD']))
METRICS = ['PSNR','SSIM','LPIPS','IoU','IoU_raw']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def snapshot(name, directory):
    manifest = read(directory/'manifest.json')
    target = OUT/'sources'/name;target.mkdir(parents=True,exist_ok=False)
    for filename,digest in manifest['files'].items():
        source=directory/filename
        data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        (target/filename).write_bytes(data)
    shutil.copyfile(directory/'manifest.json',target/'manifest.json')
    return manifest,target


def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def xlsx(path,sheets):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    content=['<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    workbook=[f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>']
    relationships=['<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('_rels/.rels',f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        for i,(name,rows) in enumerate(sheets.items(),1):
            workbook.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
            relationships.append(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>')
            content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            xml=[f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="29" customWidth="1"/><col min="2" max="26" width="19" customWidth="1"/></cols><sheetData>']
            for j,row in enumerate(rows,1):
                xml.append(f'<row r="{j}">')
                for k,value in enumerate(row):
                    cell=f'{chr(65+k)}{j}'
                    if isinstance(value,(int,float)):
                        assert math.isfinite(value);xml.append(f'<c r="{cell}"><v>{value}</v></c>')
                    else:xml.append(f'<c r="{cell}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
                xml.append('</row>')
            xml.append('</sheetData></worksheet>');z.writestr(f'xl/worksheets/sheet{i}.xml',''.join(xml))
        z.writestr('xl/workbook.xml',''.join(workbook)+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',''.join(relationships)+'</Relationships>')
        z.writestr('[Content_Types].xml',''.join(content)+'</Types>')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for n in z.namelist():ET.fromstring(z.read(n))
        for i,rows in enumerate(sheets.values(),1):
            parsed=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            actual=parsed.findall(f'{{{ns}}}sheetData/{{{ns}}}row')
            assert len(actual)==len(rows)
            for source,row in zip(rows,actual):
                assert len(row)==len(source)
                for expected,cell in zip(source,row):
                    if isinstance(expected,(int,float)):assert float(cell.find(f'{{{ns}}}v').text)==float(expected)
                    else:assert ''.join(cell.itertext())==str(expected)


def main():
    OUT.mkdir(exist_ok=True)
    core,cdir=snapshot('core',resolve('core_summary').parent/'tables')
    edp,edir=snapshot('edp',resolve('edp_root')/'tables')
    dynamic,ddir=snapshot('dynamic',H/'reports/dynamic_source_clip_v1/tables')
    assert core['selection_sha256']==edp['selection_sha256']
    coverage={r['method']:int(r['n']) for r in csvread(cdir/'coverage.csv') if r['status']=='scored'}
    ec={r['method']:int(r['valid_people']) for r in csvread(edir/'coverage.csv')}
    sheets={}
    sheets['覆盖率']=[['模型','静态可计分人数','ED-P有效人数','固定总人数']]+[[NAMES[m],coverage[m],ec[m],100] for m in METHODS]
    means=csvread(cdir/'paired_means.csv')
    for count in [5,6]:
        selected=[r for r in means if len(r['group'].split('+'))==count]
        n={int(r['n']) for r in selected};assert len(n)==1
        ns=n.pop();table=[['模型','共同人数']+METRICS]
        for method in METHODS:
            own={r['metric']:r for r in selected if r['method']==method}
            if not own:continue
            assert all(own[k]['mean_status']=='ok' for k in METRICS)
            table.append([NAMES[method],ns]+[float(own[k]['mean']) for k in METRICS])
        sheets[f'静态{count}模型_{ns}人']=table
    erows=csvread(edir/'paired_means.csv')
    for count in [5,6]:
        items=[r for r in erows if len(r['group'].split('+'))==count]
        sheets[f'EDP{count}模型']=[['模型','共同人数','共同部位总数','ED-P_common']]+[[NAMES[r['method']],int(r['n']),int(r['shared_part_slots']),float(r['mean'])] for r in items]
    all_dynamic=[]
    for record in dynamic['sources']:
        p=Path(record['path']);assert sha(p)==record['sha256'];d=read(p)
        assert not d['failures'] and set(d['methods'])=={'ours','omnire','splatad'}
        all_dynamic.append(d)
    n=len(all_dynamic);assert n==dynamic['n']
    temporal=['tOF','tLP100'];dtable=[['模型','共同人数']+METRICS+temporal]
    for method in ['ours','omnire','splatad']:
        averages=[]
        for key in METRICS+temporal:
            values=[]
            for d in all_dynamic:
                item=d['methods'][method];parts=item['pairs'] if key in temporal else item['frames']
                scores=[f['metrics'][key] for f in parts];assert all(s['status']=='ok' for s in scores)
                value=math.fsum(s['value'] for s in scores)/len(scores)
                assert abs(value-item['summary'][key]['value'])<1e-10;values.append(value)
            avg=math.fsum(values)/len(values);averages.append(avg)
            if key in temporal:
                old=next(r['mean'] for r in dynamic['means'] if r['method']==method and r['metric']==key)
                assert abs(avg-old)<1e-10
        dtable.append([NAMES[method],n]+averages)
    sheets[f'动态{n}人快照']=dtable
    ab=H/'reports/backend_ablation_v1/HN0004/metrics.csv';shutil.copyfile(ab,OUT/'sources/backend_ablation_HN0004.csv')
    ar=csvread(ab);variants={'native_33k':'原生后端','complete_A_53k':'优化后端','rgb_continuation_53k':'同预算alpha权重0'}
    sheets['消融_仅HN0004']=[['变体','实际更新次数']+METRICS+['总训练分钟','峰值allocated_GiB','峰值reserved_GiB']]
    for r in ar:sheets['消融_仅HN0004'].append([variants[r['variant']],int(r['actual_updates'])]+[float(r[k]) for k in METRICS]+[float(r['training_seconds_including_shared_native'])/60,float(r['peak_allocated_gib']),float(r['peak_reserved_gib'])])
    notes=[
      ['项目','说明'],
      ['评测范围','nuScenes固定100人v3；同真实人物/参考曝光/相机的源观测重建，不能称真实留出泛化测试。'],
      ['比较集合','同一张表所有方法使用相同人物；ED-P还使用逐人相同有效部位；不同表不能直接混合排名。'],
      ['输入预算','600条人物×方法输入记录已汇总：真实帧、插值、生成视图、原场景配置/划分及先验分别记录。场景观测数不等于该人物可见帧数，不称等输入预算对比。'],
      ['名单变更','HN0035在查看Ours结果后按用户指示换为HN0646，属于输出后人工修订，不能称完全与结果无关的抽样。'],
      ['缺测','TRELLIS 9人配准不可用；TRELLIS.2 3人生成失败、45人配准不可用；OmniRe 1人、SplatAD 16人无可独立提取人体。保持NA与100总分母。'],
      ['图示','8人从六模型共同46人按ID排序等距抽取；原始渲染直接排版。部分资产仍有朝向、姿态、形状和伪影错误；统一相机不代表资产正确。'],
      ['动态快照',f'{n}人；{sum(r["frames"] for r in dynamic["sources"])}个真实曝光；{sum(r["pairs"] for r in dynamic["sources"])}个相邻帧对。'],
      ['动态进度','84人完成三方法渲染，82人评分完成；HN0252纸张/前臂、HN0321牵引绳参考仍不可靠，按用户指示停止追加修复，保留参考缺测；16人无共同可提取对象。'],
      ['后端消融','仅1人完成三组，固定8人扩批已在GPU2–7推进；待完成者不填分。优化比原生多20000次更新；需同预算对照，不能声称普遍优于原生。'],
      ['参考修正','HN0116背包、HN0632标牌混入、HN0838背景链条已修正，并对受影响的全部可用模型统一重算核心指标与ED-P；原资产、相机和姿态配准保持不变。'],
      ['尚未完成','后端消融固定8人中1人完成、6人训练、1人排队（本次快照）；动态困难参考按用户要求停止补测。新模型按用户要求暂缓。'],
      ['未计算','动态逐帧ED-P、关节/加速度/完整3D表面误差；无对应可靠三维真值。侧背面无真实参考，只展示，不报参考型指标。'],
      ['GitHub','公开代码独立提交并验证远端；本汇报包保留自己的结果快照，数据图像、模型资产与权重不随Git发布。']]
    sheets['汇报说明']=notes
    definitions=[['指标','目的','当前实际计算','汇总和限制'],
      ['PSNR ↑','衡量像素颜色重建误差','512×512公共ROI内sRGB归一化到[0,1]；MSE=所有RGB通道差值平方的均值；PSNR=-10log10(MSE)，单位dB。MSE=0记录正无穷。','逐人算分后等权平均。包括白底，对模糊、占比与错位敏感；不是3D完整度。'],
      ['SSIM ↑','衡量局部亮度、对比度和结构相似度','11×11高斯窗，sigma=1.5；计算加权均值μ、总体方差σ²及协方差；[(2μGμP+C1)(2σGP+C2)]/[(μG²+μP²+C1)(σG²+σP²+C2)]；C1=0.01²，C2=0.03²；去5像素边界后对有效中心和RGB平均。','越接近1越相似；不是准确率。可能偏好平滑结果，不能独自判断外观精细程度。'],
      ['LPIPS ↓','衡量深度特征中的感知外观差异','固定AlexNet v0.1及线性权重SHA；公共RGB转[-1,1]、NCHW；各层特征通道归一化，平方差经学习通道权重、空间平均并跨层求和；float32/eval。','逐人等权平均。对纹理、形状和姿态差异敏感，但不是身份识别率、3D真值或百分比。'],
      ['IoU ↑','衡量可见人体投影轮廓与参考重合度','参考mask与预测自身alpha量化字节>127变为二值；公共512ROI中交集像素数/并集像素数。','空预测对非空参考计0；技术失败NA。仅测轮廓，不测颜色或内部细节。'],
      ['IoU_raw ↑','检查完整原图画布上的轮廓误差','与IoU同公式，但使用完整原相机画布，未裁剪或缩放。','可发现公共ROI以外的预测多余前景；和ROI版数值并列保留。'],
      ['ED-P_common ↓','比较头、躯干、左右臂、左右腿的部位外观特征','同512ROI；DINOv3 ViT-L/16最终归一化patch特征，排除CLS/register。RTMDet/RTMPose关键点置信度≥0.3；自身前景patch中心分配给32px内最近可靠骨架部位；每部位特征均值，距离=1-cos(reference,prediction)。','逐人对所有比较方法共同有效部位平均，再人物等权平均。无公共部位NA；报告覆盖。不是AH论文完全相同的配置，不测关节位置误差。'],
      ['tOF ↓','衡量生成序列与真实序列的二维运动变化是否一致','逐帧同真实曝光和相机，整段参考mask并集固定ROI。参考与预测各自相邻灰度帧用Farneback(.5,3,15,3,5,1.2,0)求光流；每像素算两流的二维欧氏距离，完整ROI平均。','先帧对平均后人物等权。单位为512ROI上的每实际帧间隔像素位移误差，不除Δt；含相机运动与背景，非3D动作误差。'],
      ['tLP100 ↓','衡量帧间感知变化幅度是否接近真实序列','每个相邻帧对：100×abs(LPIPS(G[t-1],G[t])-LPIPS(P[t-1],P[t]))；固定同一AlexNet v0.1。','先帧对平均后人物等权。乘100仅为显示尺度；不是准确率。不比较运动方向，也不是同一时刻预测与GT的LPIPS；不能单独证明动作正确。'],
      ['覆盖率 ↑','说明技术可用性和结果分母','各方法成功可计分人数/固定100；另记录共同人物数及ED-P有效部位数。','不按高分筛人；条件均值与覆盖率一起报告，失败NA不填零。'],
      ['训练时间/显存','记录后端成本','实测各阶段时间相加，续训总时间包含共同原生阶段；峰值torch allocated和reserved分别记录GiB。','当前只有HN0004且GPU共享，时间不是受控跨模型速度排名；reserved不同于实际张量占用。']]
    sheets['指标计算说明']=definitions
    columns=['PSNR','SSIM','LPIPS','IoU','ED-P','tOF','tLP100']
    totals=[]
    for method in METHODS:
        own={r['metric']:float(r['mean']) for r in means if r['method']==method and len(r['group'].split('+'))==6}
        own['ED-P']=next(float(r['mean']) for r in erows if r['method']==method and len(r['group'].split('+'))==6)
        own.update({r['metric']:r['mean'] for r in dynamic['means'] if r['method']==method})
        totals.append([NAMES[method]]+[own.get(k,'') for k in columns])
    sheets={'总表':[['模型']+columns]+totals,**sheets}
    best={k:(max if k in ['PSNR','SSIM','IoU'] else min)(r[i+1] for r in totals if r[i+1]!='') for i,k in enumerate(columns)}
    doc=['| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | ED-P ↓ | tOF ↓ | tLP100 ↓ |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in totals:
        cells=[row[0]]
        for k,value in zip(columns,row[1:]):
            cell=f'{value:.4f}' if value!='' else ''
            cells.append('**'+cell+'**' if value!='' and value==best[k] else cell)
        doc.append('| '+' | '.join(cells)+' |')
    doc+=['','加粗为各列最优；空白为未评测。静态四项46人，ED-P40人，动态82人；固定分母100人。',
           '已合并三处参考修正；2人动态参考缺测、16人缺共同独立对象。未按模型输出质量筛选。','']
    (OUT/'总表.md').write_text('\n'.join(doc))
    budget=resolve('input_budget')
    for name in ['per_identity_method.csv','输入预算说明.txt']:
        if (budget/name).exists():shutil.copyfile(budget/name,OUT/name)
    for name,rows in sheets.items():
        with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(rows)
    xlsx(OUT/'导师汇报表.xlsx',sheets)
    details=['人体资产实验汇报材料',datetime.now(timezone.utc).isoformat(),'',
       '公共前处理：每个人使用同一真实曝光与源相机。真实图用可见人物mask合成白底；预测仅用自身alpha合成白底，不能套GT mask擦除预测错误。静态ROI由真实mask bbox长边扩大1.10倍成正方形，越界RGB补白、mask补0，统一512×512；RGB bicubic、mask nearest。动态ROI由整段真实mask并集确定，整段固定，不能逐帧各自居中。衣物、背包与携带物按可见外观保留。',
       '所有指标配置对模型一致；但模型本身的输入信息量不同，当前不能称等输入预算对比。','']
    for row in notes[1:]:details.append(row[0]+'：'+row[1])
    for row in definitions[1:]:details.extend(['',row[0],'目的：'+row[1],'计算：'+row[2],'解释：'+row[3]])
    details += ['', '数值实现：scripts/human_source_metrics_v1.py、scripts/human_edp_common.py、scripts/score_dynamic_source_clip.py。',
      'LPIPS官方实现：https://github.com/richzhang/PerceptualSimilarity',
      'tOF/tLP计算参考：https://github.com/thunil/TecoGAN/blob/master/metrics.py；本实验不使用其裁边/删首尾帧，LPIPS版本及ROI固定为本实验协议，不能直接横比原论文数值。',
      'ED-P采用本项目common-ROI变体，具体配置与AH论文不同。',
      '图中仅写列名；行号与身份对应见figures_manifest.json，单人图片以人物ID命名。']
    (OUT/'汇报说明与指标计算.txt').write_text('\n'.join(details)+'\n')
    inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT/'tables_manifest.json').write_text(json.dumps(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_selection_sha256=core['selection_sha256'],dynamic_snapshot_n=n,generated_scores=False,
        workbook_xml_and_values_verified=True,script_sha256=sha(__file__),files=inventory),ensure_ascii=False,indent=2)+'\n')
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(workbook=str(OUT/'导师汇报表.xlsx'),archive=str(archive),dynamic=dtable,sheet_count=len(sheets)),ensure_ascii=False))


if __name__=='__main__':main()
