"""Export paired tables from one atomic summary snapshot; never recompute scores."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parents[1]
NAMES = {'ours': 'Ours', 'ah': 'Asset Harvester', 'trellis_original': 'TRELLIS',
         'omnire': 'OmniRe', 'splatad': 'SplatAD', 'trellis2': 'TRELLIS.2'}
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw')


def value(score):
    if score['status'] == 'positive_infinity':
        return 'inf'
    return score['value'] if score['status'] == 'ok' else 'NA'


def write_csv(path, fields, rows):
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def main(quiet=False):
    source = H/'reports/source_camera_metric_cohort_v3/summary.json'
    raw = source.read_bytes()  # One snapshot even if the scorer publishes meanwhile.
    summary = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    out = source.parent/'tables'; out.mkdir(exist_ok=True)
    paired, members = [], []
    for group, metrics in summary['pairings'].items():
        for name in METRICS:
            p = metrics[name]
            for sid in p['common_ids']:
                members.append(dict(group=group, metric=name, id=sid))
            for method, item in p['paired'].items():
                assert item['n'] == len(p['common_ids']) and item['denominator'] == 100
                paired.append(dict(group=group, metric=name, method=method, n=item['n'],
                    denominator=item['denominator'], mean=value(item['mean']),
                    mean_status=item['mean']['status'],
                    positive_infinity_count=item['positive_infinity_count']))
    write_csv(out/'paired_means.csv', list(paired[0]), paired)
    write_csv(out/'paired_members.csv', ['group', 'metric', 'id'], members)
    coverage = []
    for method, count in summary['coverage'].items():
        rows = [r for r in summary['rows'] if r['method'] == method]
        assert len(rows) == 100 and sum(r['status'] == 'scored' for r in rows) == count
        for status, n in sorted(Counter(r['status'] for r in rows).items()):
            coverage.append(dict(method=method, status=status, n=n, denominator=100))
    write_csv(out/'coverage.csv', ['method', 'status', 'n', 'denominator'], coverage)
    missing = [{k: r.get(k) for k in ['id', 'method', 'status', 'reason']}
               for r in summary['rows'] if r['status'] != 'scored']
    write_csv(out/'missing.csv', ['id', 'method', 'status', 'reason'], missing)
    now = datetime.now(timezone.utc).isoformat()
    doc = ['# 当前人体源视角配对结果', '', f'快照时间：{now}。固定名单100人，参考资格v5。', '',
           '本表是源观测重建实验。各方法保留自身姿态与外观误差；TRELLIS与TRELLIS.2使用已声明的身体关键点刚体配准，不能称为真实留出评测。', '',
           '同一张表只比较共同有效人物。不同表的人物集合可能不同，不能跨表直接比较均值。', '',
           '## 覆盖率', '', '| 方法 | 已计分 / 固定人数 |', '|---|---:|']
    doc += [f'| {NAMES.get(m, m)} | {n}/100 |' for m, n in summary['coverage'].items()]
    groups = list(summary['pairings'])
    main_groups = [g for g in groups if len(g.split('+')) >= 5]
    for group in main_groups:
        p = summary['pairings'][group]
        ids = p['PSNR']['common_ids']
        assert all(p[name]['common_ids'] == ids for name in METRICS), 'Use per-metric tables if populations differ'
        doc += ['', f'## {len(group.split("+"))}方法共同人物：{len(ids)}/100', '',
                '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ |', '|---|---:|---:|---:|---:|']
        for method in group.split('+'):
            scores = [value(p[name]['paired'][method]['mean']) for name in METRICS[:4]]
            cells = [f'{x:.4f}' if isinstance(x, (float, int)) else str(x) for x in scores]
            doc.append('| ' + ' | '.join([NAMES.get(method, method)] + cells) + ' |')
    # Coverage and failures are part of the result, not an invitation to select
    # only attractive outputs. These means describe TRELLIS.2's own valid set;
    # cross-method comparisons above retain their common identity sets.
    t2 = [r for r in summary['rows'] if r['method'] == 'trellis2']
    counts = Counter(r['status'] for r in t2)
    valid_means = []
    for name in METRICS:
        scores = [r['metrics'][name] for r in t2 if r['status'] == 'scored'
                  and r['metrics'][name]['status'] in ('ok', 'positive_infinity')]
        ninf = sum(s['status'] == 'positive_infinity' for s in scores)
        avg = 'inf' if ninf else math.fsum(s['value'] for s in scores)/len(scores) if scores else 'NA'
        valid_means.append(dict(method='trellis2', metric=name, n=len(scores),
            denominator=100, mean=avg, positive_infinity_count=ninf,
            scope='method_valid_set_only_not_cross_method_ranking'))
    write_csv(out/'trellis2_valid_means.csv', list(valid_means[0]), valid_means)
    pending = sum(n for status, n in counts.items() if 'pending' in status)
    disclosure = (
        '我们在固定的100人人体集合上测试TRELLIS.2，并保留全部人物作为覆盖率分母。'
        f'当前{counts["scored"]}人完成统一源相机渲染和指标计算；'
        f'{counts["native_generation_failed"]}人原生生成失败，'
        f'{counts["source_registration_unavailable"]}人虽有原生输出，但无法通过既定身体关键点配准得到可用的共同源相机结果，'
        f'另有{pending}人尚待处理完成。'
        '可计分输出不按视觉质量或指标高低筛选；技术失败保留为NA，不填零、不替换人物，也不反复改种子重生成。'
        '单方法均值使用该方法实际有效集合，横向比较则使用所比较方法共同有效的相同人物，并同时报告n/100。'
        '因而这些成绩是有效输出条件下的重建质量，不能代表在全部100人上的无条件性能。'
    )
    doc += ['', '## TRELLIS.2有效数量、失败案例与正文说明', '', disclosure, '',
            '“生成了文件”不等于完成共同相机评测。原生生成失败与配准不可用分开记录；不能把所有缺测统称为模型生成失败。', '',
            '| TRELLIS.2自身有效集合 | n/100 | 均值 |', '|---|---:|---:|']
    for r in valid_means[:4]:
        avg = f'{r["mean"]:.4f}' if isinstance(r['mean'], (float, int)) else r['mean']
        doc.append(f'| {r["metric"]} | {r["n"]}/100 | {avg} |')
    doc += ['', '上表只描述TRELLIS.2自身覆盖范围，不与不同人物集合的均值直接排名。', '',
            '[失败案例图册](../../trellis2_failure_root_cause_v1/failure_cases/index.html) · '
            '[失败及待完成清单](missing.csv) · [自身有效集合均值](trellis2_valid_means.csv)', '',
            '归因说明：官方流程的有限复现确认HN0046的薄片缺陷已存在于原生输出，且与原资产几何完全一致；'
            'HN0214存在旧输入mask包含背景的问题，只修正mask就明显改变了输出。'
            '因此不能将所有失败归因于模型本身。修正mask的诊断结果未替换正式资产或指标。']
    doc += ['', '## 数据与范围', '',
            '- [配对均值](paired_means.csv)、[每个指标的共同人物](paired_members.csv)、[覆盖率](coverage.csv)、[逐人缺测原因](missing.csv)。',
            '- 原始逐人指标在上一级 `metrics.csv`；快照以本目录 `manifest.json` 中记录的摘要为准。',
            '- 缺测保持NA，不填零、不从100人名单移除；遮挡和雨伞不自动取消参考资格。',
            '- TRELLIS.2终止状态和待完成数量以上述清单为准；失败案例保留，不再反复补跑。后端消融单独报告。',
            '- [ED-P辅助指标与部位覆盖率](../../source_edp_common_v1/cohort_v1/tables/RESULTS.md)：使用共同人物及共同有效部位；该独立结果表为ED-P的当前入口，历史核心指标JSON中的ED-P占位NA不代表辅助计算仍未完成。',
            '- [动态指标进度与小样例结果](../../dynamic_source_clip_v1/tables/RESULTS.md)：tOF、tLP100与静态表分开报告；目前不是100人动态评测。',
            '- PSNR/SSIM/LPIPS使用同一真实相机、参考确定的公共ROI及白底；预测使用自身alpha。无真实背面参考时不计算背面参考指标。', '']
    (out/'RESULTS.md').write_text('\n'.join(doc))
    history = source.parent/'history'; history.mkdir(exist_ok=True)
    snapshot = history/(digest+'.json')
    if not snapshot.exists():
        snapshot.write_bytes(raw)
    files = ['paired_means.csv', 'paired_members.csv', 'coverage.csv', 'missing.csv', 'trellis2_valid_means.csv', 'RESULTS.md']
    manifest = dict(generated_at_utc=now, summary=str(snapshot), summary_sha256=digest,
                    selection_sha256=summary['selection_sha256'], coverage=summary['coverage'],
                    protocol_sha256=summary['protocol_sha256'],
                    files={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in files})
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    if not quiet:
        print(json.dumps(dict(report=str(out/'RESULTS.md'), coverage=summary['coverage']), ensure_ascii=False))


if __name__ == '__main__':
    main()
