"""Summarize verified dynamic scores; no rendering or metric recomputation."""
import json
import math
from datetime import datetime, timezone
from common_human_orbit import H, read, save, sha, checked
from export_active_metric_tables import NAMES, write_csv


def main():
    root = H/'reports/dynamic_source_clip_v1'
    methods = ('ours', 'omnire', 'splatad')
    metrics = ('tOF', 'tLP100')
    rows, records = [], []
    for path in sorted(root.glob('HN*/metrics/metrics.json')):
        d = read(path); folder = path.parents[1]; sid = folder.name
        assert d['identity'] == sid and set(d['methods']) == set(methods)
        assert d['source_reconstruction_only'] and not d['heldout_evaluation']
        checked(folder/'clip.json', d['inputs']['clip_sha256'])
        checked(folder/'reference_qualification.json', d['inputs']['qualification_sha256'])
        checked(d['protocol'], d['inputs']['protocol_sha256'])
        clip = read(folder/'clip.json')
        times = [f['timestamp_us'] for f in clip['frames']]
        assert all(b > a for a, b in zip(times, times[1:]))
        assert len(times) == d['reference_frame_count'] == d['reference_pair_count'] + 1
        for method in methods:
            checked(folder/method/'validation.json', d['inputs']['method_manifests'][method])
            item = d['methods'][method]
            assert item['status'] == 'scored'
            assert [f['timestamp_us'] for f in item['frames']] == times
            assert item['frame_count'] == len(item['frames']) == len(times)
            assert item['pair_count'] == len(item['pairs']) == len(times)-1
            for i, pair in enumerate(item['pairs'], 1):
                assert (pair['previous_index'], pair['index'], pair['timestamp_us']) == (i-1, i, times[i])
                assert abs(pair['delta_seconds']-(times[i]-times[i-1])/1e6) < 1e-12
            for metric in metrics:
                values = [p['metrics'][metric]['value'] for p in item['pairs']]
                assert all(p['metrics'][metric]['status'] == 'ok' for p in item['pairs'])
                avg = math.fsum(values)/len(values)
                assert abs(avg-item['summary'][metric]['value']) < 1e-10
                rows.append(dict(id=sid, method=method, metric=metric, value=avg,
                    frames=len(times), pairs=len(times)-1, denominator=100))
        records.append(dict(id=sid, path=str(path), sha256=sha(path), frames=len(times),
            pairs=len(times)-1, protocol_sha256=d['inputs']['protocol_sha256']))
    assert records and len({r['protocol_sha256'] for r in records}) == 1
    out = root/'tables'; out.mkdir(exist_ok=True)
    write_csv(out/'per_identity.csv', list(rows[0]), rows)
    means = []
    for method in methods:
        for metric in metrics:
            selected = [r for r in rows if r['method'] == method and r['metric'] == metric]
            means.append(dict(method=method, metric=metric, n=len(selected), denominator=100,
                mean=math.fsum(r['value'] for r in selected)/len(selected),
                aggregation='equal_identity_means_not_pooled_frame_pairs'))
    write_csv(out/'paired_means.csv', list(means[0]), means)
    doc = ['# 动态源视角指标：当前批次结果', '',
        f'共同人物 {len(records)}/100；共{sum(r["frames"] for r in records)}个真实曝光、'
        f'{sum(r["pairs"] for r in records)}个相邻帧对。每种方法使用相同人物和逐人相同曝光。', '',
        '每人先对实际相邻帧对取均值，再对人物等权平均。下表描述当前已完成批次，不能据此得出100人整体排名。', '',
        '| 方法 | tOF ↓ | tLP100 ↓ | n/100 |', '|---|---:|---:|---:|']
    for method in methods:
        values = {r['metric']: r['mean'] for r in means if r['method'] == method}
        doc.append(f'| {NAMES[method]} | {values["tOF"]:.4f} | {values["tLP100"]:.4f} | {len(records)}/100 |')
    doc += ['', 'tOF是固定512×512公共ROI内预测与参考相邻光流的像素差；tLP100是相邻LPIPS变化差的100倍。两者均越低越好。'
        '它们衡量图像时间变化的一致程度，不能单独证明动作或三维几何正确。', '',
        '## tOF的实际计算方式', '',
        '1. 同一人物使用完全相同的真实曝光时刻和对应相机。参考用逐帧可见mask合成白底；预测只用自身alpha。整段真实mask的并集确定一个固定公共ROI，长边扩1.10倍成正方形并缩放到512×512，各帧和方法均使用该ROI。',
        '2. 分别将真实与预测的相邻两帧转为灰度，用相同的OpenCV Farneback算法估计二维光流。每个像素得到水平、垂直两个位移分量。',
        '3. 每个像素的误差为 `sqrt((Fx_ref-Fx_pred)^2 + (Fy_ref-Fy_pred)^2)`；对完整512×512区域的所有像素取平均，得到这一帧对的tOF。',
        '4. 对同一人物的全部有效相邻帧对等权平均，再对共同人物等权平均。不是把所有人物的帧对混在一起平均。', '',
        '例：某像素真实光流为向右4像素，预测向右1像素，则该像素误差为3；若预测向左4像素，误差为8。实际分数还要平均所有像素和帧对。', '',
        '当前没有除以帧间秒数，单位是每个实际帧间隔的像素位移误差，不是像素/秒或三维关节误差。不同方法使用完全相同的真实帧间隔；不补帧、不循环、不重定时。', '',
        '当前对完整公共ROI计算，包含白底。大面积静止背景、模糊、轮廓误差与相机运动都会影响分数，因此低tOF不能单独证明肢体动作或外观正确。', '',
        '公式和Farneback参数参照[TecoGAN官方实现](https://github.com/thunil/TecoGAN/blob/master/metrics.py)。本实验保留完整公共ROI和所有有效真实帧对，不使用其裁边及片段首尾删帧操作，因此不直接与论文表格数值比较。', '',
        '## 真实帧覆盖', '',
        'HN0046只使用三方法共同支持的29个真实曝光（28帧对）；末帧超出Ours原生时长，另记缺测，不补帧或循环。', '',
        '逐帧PSNR、SSIM、LPIPS、IoU和IoU_raw也已保存，见各人物页面。', '',
        '| 人物 | 真实帧数 | 帧对数 | 查看效果 |', '|---|---:|---:|---|']
    doc += [f'| {r["id"]} | {r["frames"]} | {r["pairs"]} | [视频与逐项分数](../{r["id"]}/index.html) |' for r in records]
    doc += ['', '[逐人时序成绩](per_identity.csv) · [配对均值](paired_means.csv)', '',
        '动态逐帧ED-P尚未计算；固定参考时刻的[ED-P辅助表](../../source_edp_common_v1/cohort_v1/tables/RESULTS.md)已经独立提供。关节/加速度及完整表面几何指标未计算，当前没有相应可靠三维真值。'
        '没有真实侧背面参考，不计算侧背面PSNR/LPIPS。', '']
    closeout = root/'reference_closeout.json'
    extra_files = []
    if closeout.exists():
        audit = read(closeout)
        assert sorted(r['id'] for r in records) == audit['scored_ids']
        assert len(records)+len(audit['missing']) == 100
        write_csv(out/'missing.csv', ['id','status','reason'], audit['missing'])
        extra_files.append('missing.csv')
        doc += ['## 本轮缺测记录', '',
            'HN0252的纸张/前臂与HN0321的牵引绳参考分割仍不可靠，按用户指示停止追加修复，记参考缺测；不是模型生成失败。另16人没有三方法共同可提取对象。', '',
            '全部100人仍保留在覆盖率分母中；本轮82人评分完成，未按模型效果或分数筛选。', '',
            '[逐人缺测原因](missing.csv)', '']
    (out/'RESULTS.md').write_text('\n'.join(doc))
    save(out/'manifest.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        n=len(records), denominator=100, sources=records, means=means,
        generated_scores=False, script_sha256=sha(__file__),
        reference_closeout_sha256=sha(closeout) if closeout.exists() else None,
        files={name:sha(out/name) for name in ['per_identity.csv','paired_means.csv','RESULTS.md']+extra_files}))
    print(json.dumps(dict(report=str(out/'RESULTS.md'), n=len(records), means=means), ensure_ascii=False))


if __name__ == '__main__':
    main()
