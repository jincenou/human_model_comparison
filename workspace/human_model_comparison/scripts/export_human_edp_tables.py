"""Auxiliary ED-P tables with common identities AND common body parts."""
import argparse
import json
import math
from collections import Counter
from datetime import datetime,timezone
from common_human_orbit import H,read,save,sha,checked,METHODS
from human_source_metrics_v1 import PARTS
from active_human_results import resolve
from export_active_metric_tables import write_csv,NAMES


def main(root):
    frozen=read(H/'configs/human_source_common_v1/edp_protocol_frozen_v1.json')
    checked(root/'protocol.json',frozen['specification_sha256'])
    inputs=read(root/'inputs.json');results=read(root/'results.json')
    checked(root/'inputs.json',results['inputs_sha256']);checked(root/'source_summary.json',inputs['summary_sha256'])
    assert results['protocol_sha256']==frozen['specification_sha256']
    summary=read(root/'source_summary.json');selected=set(inputs['ids'])
    latest_path=resolve('core_summary')
    latest=read(latest_path)
    assert latest['selection_sha256']==summary['selection_sha256']
    old_scores={(r['id'],r['method']):r['result_sha256'] for r in summary['rows'] if r['status']=='scored'}
    new_scores={(r['id'],r['method']):r['result_sha256'] for r in latest['rows'] if r['status']=='scored'}
    assert old_scores==new_scores, 'New or changed scored outputs require an incremental ED-P run'
    # Terminal NA statuses can settle after feature extraction; no score inputs
    # are changed, and both the original and current summaries remain traceable.
    summary=latest
    assert len(selected)==100 and selected=={r['id']for r in summary['rows']}
    scored={(r['id'],r['method']):r for r in results['rows']}
    assert set(scored)=={(r['id'],r['method'])for r in summary['rows']if r['status']=='scored'}
    out=root/'tables';out.mkdir(exist_ok=True);objects=[];parts=[]
    for r in summary['rows']:
        x=scored.get((r['id'],r['method']))
        objects.append(dict(id=r['id'],method=r['method'],source_status=r['status'],
            edp_status=x['score']['status'] if x else 'na',
            edp=x['score']['value'] if x and x['score']['status']=='ok' else None,
            reason=x['score'].get('reason') if x else r['status'],
            valid_parts=x['valid_part_count'] if x else 0,coverage=x['coverage'] if x else 0))
        for name in PARTS:
            detail=x['parts'][name] if x else dict(distance=None,reference_patches=None,prediction_patches=None,reason=r['status'])
            parts.append(dict(id=r['id'],method=r['method'],part=name,**detail))
    write_csv(out/'per_object.csv',list(objects[0]),objects);write_csv(out/'per_part.csv',list(parts[0]),parts)
    coverage=[]
    for method in METHODS:
        rows=[r for r in objects if r['method']==method]
        coverage.append(dict(method=method,valid_people=sum(r['edp_status']=='ok'for r in rows),denominator=100,
            paired_valid_part_slots=sum(r['valid_parts']for r in rows),possible_part_slots=600,
            source_scored=summary['coverage'][method]))
    write_csv(out/'coverage.csv',list(coverage[0]),coverage)
    paired=[];members=[];excluded=[]
    groups=[g for g in summary['pairings']if len(g.split('+'))>=5]
    for group in groups:
        methods=group.split('+');valid=[]
        for sid in sorted(selected):
            items=[scored.get((sid,m))for m in methods]
            if any(x is None or x['score']['status']!='ok'for x in items):
                excluded.append(dict(group=group,id=sid,reason='at_least_one_method_has_no_valid_EDP'));continue
            common=set.intersection(*(set(x['valid_parts'])for x in items))
            common=[p for p in PARTS if p in common]
            if not common:
                excluded.append(dict(group=group,id=sid,reason='no_common_parts_across_methods'));continue
            valid.append(sid);members.append(dict(group=group,id=sid,parts=';'.join(common),part_count=len(common)))
            for method,x in zip(methods,items):
                value=math.fsum(x['parts'][p]['distance']for p in common)/len(common)
                paired.append(dict(group=group,id=sid,method=method,value=value,common_parts=len(common)))
    write_csv(out/'paired_members.csv',['group','id','parts','part_count'],members)
    write_csv(out/'paired_per_identity.csv',['group','id','method','value','common_parts'],paired)
    write_csv(out/'paired_excluded.csv',['group','id','reason'],excluded)
    means=[]
    for group in groups:
        for method in group.split('+'):
            rows=[r for r in paired if r['group']==group and r['method']==method]
            means.append(dict(group=group,method=method,n=len(rows),denominator=100,
                mean=math.fsum(r['value']for r in rows)/len(rows)if rows else None,
                shared_part_slots=sum(r['common_parts']for r in rows),possible_part_slots=600))
    write_csv(out/'paired_means.csv',list(means[0]),means)
    doc=['# 人体辅助指标 ED-P：共同ROI版本','',
        '固定100人；仅对已经完成统一源相机渲染的输出提取特征，技术缺测保留NA。'
        '下面每张比较表同时使用相同人物和每个人的相同有效部位，避免各模型分别丢弃困难部位后直接比较均值。','',
        '本实验采用DINOv3 ViT-L/16最终层patch特征，RTMDet＋RTMPose/DWPose关键点，以及冻结的512×512公共ROI。'
        '它保留[Asset Harvester论文](https://arxiv.org/html/2604.18468v1#S4.SS2.SSS2)的六部位池化与余弦距离思想，'
        '但编码器、关键点模型、裁剪和缺部位处理不同，不能与原论文ED-P数值直接比较。','',
        '## 覆盖率','',
        '| 方法 | ED-P有效人物/100 | 有效部位对/600 | 已有源视角结果 |','|---|---:|---:|---:|']
    doc += [f'| {NAMES[r["method"]]} | {r["valid_people"]}/100 | {r["paired_valid_part_slots"]}/600 | {r["source_scored"]}/100 |'for r in coverage]
    for group in groups:
        entries=[r for r in means if r['group']==group]
        n=entries[0]['n'];slots=entries[0]['shared_part_slots']
        doc+=['',f'## {len(entries)}方法共同人物：{n}/100','',
            f'共同使用{slots}个部位对；满覆盖分母为100×6=600。逐人先平均共同部位，再对人物等权平均。','',
            '| 方法 | ED-P_common ↓ |','|---|---:|']
        for r in entries:
            cell=f'{r["mean"]:.4f}'if r['mean']is not None else'NA'
            doc.append(f'| {NAMES[r["method"]]} | {cell} |')
    doc += ['', '## 缺测与解释范围','',
        '- 六部位为头、躯干、左/右臂、左/右腿。每张图用自身前景mask选取patch，预测不套真实mask。',
        '- 部位分配来自估计关键点和前景patch支持，并不是解剖可见性的真值；少量残余patch也可能使某部位有效，覆盖率不代表肢体完整度。',
        '- 无公共有效部位时记NA；不因少于四个部位而丢弃整个人。缺部位后均分变低不代表生成更完整。',
        '- 根据冻结真实参考的雨伞遮头标签，HN0190和HN0215的头部项对所有方法统一不计；其余可见部分继续计分，主指标及完整mask不变。',
        '- 无逐关节调整、各自bbox归一化或指标驱动的角度/尺度搜索；姿态、大小和输出缺陷保持原样。',
        '- 小样例已检查部位划分；批量完成不等于全部人物的关键点均经人工标注。', '',
        '[逐人分数与覆盖](per_object.csv) · [六部位明细](per_part.csv) · [配对均值](paired_means.csv) · '
        '[共同人物与部位](paired_members.csv) · [配对缺测](paired_excluded.csv)', '',
        '[全部人物部位叠加图](../part_review.jpg) · [指标协议](../protocol.json)', '']
    (out/'RESULTS.md').write_text('\n'.join(doc))
    files=['per_object.csv','per_part.csv','coverage.csv','paired_members.csv','paired_per_identity.csv','paired_excluded.csv','paired_means.csv','RESULTS.md']
    save(out/'manifest.json',dict(status='auxiliary_cohort_tables_exported',created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_results=str(root/'results.json'),source_results_sha256=sha(root/'results.json'),specification_sha256=frozen['specification_sha256'],
        latest_source_summary=str(latest_path),latest_source_summary_sha256=sha(latest_path),
        selection_sha256=inputs['selection_sha256'],coverage=coverage,means=means,
        script_sha256=sha(__file__),files={name:sha(out/name)for name in files}))
    print(json.dumps(dict(report=str(out/'RESULTS.md'),coverage=coverage),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name');a=p.parse_args()
    if a.run_name:assert a.run_name.replace('_','').isalnum()
    main(H/'reports/source_edp_common_v1'/a.run_name if a.run_name else resolve('edp_root'))
