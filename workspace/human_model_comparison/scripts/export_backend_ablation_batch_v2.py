"""Export paired results for the predeclared eight-person backend ablation."""
import math
import statistics
from common_human_orbit import H, read, save, sha, checked
from export_active_metric_tables import write_csv


def main():
    planpath = H/'configs/backend_ablation_batch_v2.json'; plan = read(planpath)
    checked(plan['selection'], plan['selection_sha256'])
    out = H/'reports/backend_ablation_batch_v2'; out.mkdir(exist_ok=True)
    metrics = ['PSNR', 'SSIM', 'LPIPS', 'IoU', 'IoU_raw']
    rows = []; sources = {}; completed = []
    for sid in plan['ids']:
        folder = H/'reports/backend_ablation_v1'/sid
        path = folder/'results.json'
        group = next(g for g, ids in plan['groups'].items() if sid in ids)
        if not path.exists():
            for name in plan['variants']:
                rows.append(dict(id=sid, cohort=group, variant=name, status='not_scored',
                                 actual_updates='', **{k:'' for k in metrics},
                                 training_minutes='', peak_allocated_gib='', peak_reserved_gib=''))
            continue
        data = read(path); sources[str(path)] = sha(path)
        protocol = read(data['protocol']); checked(data['protocol'], data['protocol_sha256'])
        checked(data['metric_protocol'], data['metric_protocol_sha256'])
        checked(data['registration'], data['registration_sha256'])
        assert protocol['selection_sha256'] == plan['selection_sha256']
        assert protocol['all_original_rgba_file_hashes_equal'] and not protocol['diffusion_generation_rerun']
        by_variant = {r['variant']: r for r in data['rows'] if not r['diagnostic_only']}
        assert set(by_variant) == set(plan['variants'])
        root = H/'runs/backend_ablation_v1'/sid
        native = read(root/'outputs'/sid/'native/ablation_execution.json')
        for name, updates in zip(plan['variants'], plan['actual_updates']):
            record = by_variant[name]
            assert record['actual_updates'] == updates
            assert record == read(folder/name/'metrics.json')
            checked(folder/name/'source_rgb.png', record['rgb_sha256'])
            checked(folder/name/'source_alpha.png', record['alpha_sha256'])
            checked(record['snapshot'], record['snapshot_sha256'])
            values = {}
            for key in metrics:
                score = record['metrics'][key]
                if key == 'PSNR' and score['status'] == 'positive_infinity':
                    values[key] = math.inf
                else:
                    assert score['status'] == 'ok' and math.isfinite(score['value'])
                    values[key] = score['value']
            stages = [native]
            if name != 'native_33k':
                base = root/'rgb_continuation_53k' if name == 'rgb_continuation_53k' else root
                stages += [read(base/'outputs'/sid/stage/'ablation_execution.json') for stage in ['alpha','asset']]
            rows.append(dict(id=sid, cohort=group, variant=name, status='scored', actual_updates=updates,
                             **values,
                             training_minutes=sum(r['wall_seconds_including_loading'] for r in stages)/60,
                             peak_allocated_gib=max(r['peak_allocated_bytes'] for r in stages)/1024**3,
                             peak_reserved_gib=max(r['peak_reserved_bytes'] for r in stages)/1024**3))
        for stage in ['alpha','asset']:
            for filename in ['sampling.json','learning_rates.json']:
                assert sha(root/'outputs'/sid/stage/filename) == sha(root/'rgb_continuation_53k/outputs'/sid/stage/filename)
        completed.append(sid)
    write_csv(out/'per_identity.csv', list(rows[0]), rows)
    paired = []
    for group, ids in [('all', plan['ids']), *plan['groups'].items()]:
        common = sorted(set(ids) & set(completed))
        if not common:
            continue
        for name in plan['variants']:
            selected = [r for r in rows if r['variant'] == name and r['id'] in common]
            paired.append(dict(cohort=group, variant=name, n=len(common), denominator=len(ids),
                               ids=','.join(common), **{k:statistics.mean(r[k] for r in selected) for k in metrics}))
    if paired:
        write_csv(out/'paired_means.csv', list(paired[0]), paired)
    text = ['# 固定前端缓存后端消融：预先固定8人', '',
            f'当前三组共同完成 **{len(completed)}/8人**。名单为原有与新增两组分别按编号取前4人，包含此前已完成的HN0004；未按模型分数选择。尚未完成的结果留空，不混入均值。', '',
            '| 组别 | 变体 | 人数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | IoU_raw ↑ |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        text.append(f'| {r["cohort"]} | {r["variant"]} | {r["n"]}/{r["denominator"]} | ' + ' | '.join(f'{r[k]:.4f}' for k in metrics) + ' |')
    text += ['', '原生后端32999步；优化与同预算对照均52999步。同预算对照只将alpha损失权重从0.1改为0，保留采样、学习率、初始化和全部输入。多出的20000步不能全部归因于算法。非整数曝光使用原运行时的高斯协方差插值，不取最近帧。', '',
             '本表是已观测源视角重建，非独立真实留出。每人缓存与原哈希一致，前端视频未重新生成。训练时间和峰值显存见逐人CSV；同卡并行会影响实际耗时。结果生成后仍需目视核查，视觉质量差本身不会被排除。', '',
             '[逐人结果与资源用量](per_identity.csv) · [汇总来源](manifest.json)', '']
    (out/'RESULTS.md').write_text('\n'.join(text))
    save(out/'manifest.json', dict(status='scored_pending_visual_review' if len(completed)==len(plan['ids']) else 'partial_batch',
                                  plan=str(planpath), plan_sha256=sha(planpath), ids=plan['ids'], completed_ids=completed,
                                  pending_ids=sorted(set(plan['ids'])-set(completed)), source_hashes=sources,
                                  script_sha256=sha(__file__), denominator=8, paired_n=len(completed)))
    print(f'Paired backend results: {len(completed)}/8')


if __name__ == '__main__':
    main()
