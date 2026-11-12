"""Report the measured fixed-cache pilot, including matched budget and limitations."""
import argparse
import json
from datetime import datetime, timezone
from common_human_orbit import H, read, save, sha, checked
from export_active_metric_tables import write_csv


def main(sid):
    root=H/'runs/backend_ablation_v1'/sid
    out=H/'reports/backend_ablation_v1'/sid
    data=read(out/'results.json')
    checked(data['protocol'],data['protocol_sha256'])
    checked(data['metric_protocol'],data['metric_protocol_sha256'])
    checked(data['registration'],data['registration_sha256'])
    native=read(root/'outputs'/sid/'native/ablation_execution.json')
    rows=[]
    for row in data['rows']:
        if row['diagnostic_only']:continue
        name=row['variant'];folder=out/name
        assert read(folder/'metrics.json')==row
        checked(folder/'source_rgb.png',row['rgb_sha256']);checked(folder/'source_alpha.png',row['alpha_sha256'])
        stages=[native]
        if name!='native_33k':
            base=root/'rgb_continuation_53k' if name=='rgb_continuation_53k' else root
            stages += [read(base/'outputs'/sid/stage/'ablation_execution.json')for stage in ['alpha','asset']]
        rows.append(dict(id=sid,variant=name,actual_updates=row['actual_updates'],
            **{key:score['value'] for key,score in row['metrics'].items()},
            training_seconds_including_shared_native=sum(r['wall_seconds_including_loading']for r in stages),
            peak_allocated_gib=max(r['peak_allocated_bytes']for r in stages)/1024**3,
            peak_reserved_gib=max(r['peak_reserved_bytes']for r in stages)/1024**3))
    for stage in ['alpha','asset']:
        for name in ['sampling.json','learning_rates.json']:
            assert sha(root/'outputs'/sid/stage/name)==sha(root/'rgb_continuation_53k/outputs'/sid/stage/name)
    write_csv(out/'metrics.csv',list(rows[0]),rows)
    labels={'native_33k':'原生后端','complete_A_53k':'优化后端','rgb_continuation_53k':'同预算对照（alpha损失权重0）'}
    doc=['# 固定前端缓存的后端消融：'+sid,'',
         '**当前为1个人物的小样例，不能据此得出100人的总体结论。** 三组均使用同一真实参考时刻、相机、参考ROI、白底及冻结核心指标；模型姿态和大小误差保留。','',
         '24路×61帧共1464张RGBA恢复后与原始缓存逐文件哈希一致，未重新生成前端视频。相机、初始化、种子42、输入分辨率352×640、SH阶数3保持固定。', '',
         '| 变体 | 实际更新 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | IoU_raw ↑ |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        doc.append(f'| {labels[r["variant"]]} | {r["actual_updates"]} | '+ ' | '.join(f'{r[k]:.4f}'for k in ['PSNR','SSIM','LPIPS','IoU','IoU_raw'])+' |')
    doc+=['','原生与优化版相比，多出了20000次更新，不能将全部变化归因于优化模块。同预算对照与优化版共享同一32999步初始checkpoint，继续15000＋5000步，采样和学习率记录哈希一致，唯一训练代码变化是alpha损失系数从0.1改为0。该对照隔离的是此损失在既定续训流程中的作用，不代表所有优化模块分别消融。','',
          '该样例中，优化版相对于同预算对照的LPIPS与轮廓IoU有所改善，PSNR略降；相对于原生32999步也并非全部指标提高。数值按实报告，不按分数修改相机、选择时刻或替换人物。','',
          '| 变体 | 总训练时间（分钟） | 峰值allocated（GiB） | 峰值reserved（GiB） |','|---|---:|---:|---:|']
    for r in rows:
        doc.append(f'| {labels[r["variant"]]} | {r["training_seconds_including_shared_native"]/60:.2f} | {r["peak_allocated_gib"]:.2f} | {r["peak_reserved_gib"]:.2f} |')
    doc += ['', '时间含各阶段加载开销；两组续训复用了同一次原生训练，表内总时间均计入这段共有成本。GPU可与其他任务共享，因此时间只作为本次实际执行记录。', '',
        '![真实参考、三组消融与原资产渲染核对](comparison.jpg)', '',
        '最右列为原正式资产的渲染核对，不计入消融表；其RGB和alpha与已有正式源视角逐像素相同。三组新训练输出使用生产导出的完整SH与协方差转换，未经独立重配准。原正式训练checkpoint已清理，因此没有宣称从该旧checkpoint重新导出完成等价验证。', '',
        '这是已观测源视角重建实验。生成视图中预留0/8/16号视图并不等于真实未见相机的GT；不把生成视图当真实留出。', '',
        '[逐项CSV](metrics.csv) · [结果及来源](results.json) · [验收记录](acceptance.json)', '']
    (out/'RESULTS.md').write_text('\n'.join(doc))
    save(out/'tables_manifest.json',dict(status='one_identity_ablation_tables_exported',created_at_utc=datetime.now(timezone.utc).isoformat(),
        id=sid,n=1,results_sha256=sha(out/'results.json'),script_sha256=sha(__file__),
        files={name:sha(out/name)for name in ['metrics.csv','RESULTS.md']}))
    print(json.dumps(dict(report=str(out/'RESULTS.md'),n=1),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);a=p.parse_args();main(a.human_id)
