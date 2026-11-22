# 人体实验入口与执行顺序

下列路径均相对于 `workspace/human_model_comparison`。本索引归纳截至 2026-11-22 收集到的源码，不把历史队列记录当作当前正在运行的任务。生产管线的统一文档位于 [OmniSim_Asset](https://github.com/jincenou/OmniSim_Asset)。

## 1. 参考与固定名单

`audit_nuscenes_cohort_v2.py` / `audit_candidate_runtime_decode_v2.py` → `prepare_exact_reference_masks_v2.py` → 经审核的名单与源曝光记录 → `prepare_human_source_cameras_v1.py` / `build_source_comparison_contracts_v2.py`。

输入为 nuScenes 原始曝光、实例轨迹、可见前景掩码和可用相机参数；输出为名单、相机/裁剪契约和内容哈希。`freeze_human100_v2.py` 与 `freeze_human100_replacement_v3.py` 属于历史名单构建阶段，新实验不能机械复用旧状态。公开模板位于 `metadata/templates/human_model_comparison/`；逐人物相机与注册缓存需要从本地输入生成。

## 2. 原生资产与方法适配

| 方法/路线 | 代码入口 | 必要外部输入 |
|---|---|---|
| Ours | `export_ours_frozen_time_v1.py`、`render_ours_frozen_time_v3.py`、`prepare_ours_source_batch_v3.py` | 4DAnyone 前端缓存、Grid4D 资产和源时间映射 |
| Asset Harvester | `prepare_ah_human_pilot_v1.py`、`run_ah_human_pilot_v1.py`、`reload_ah_human_pilot_v1.py` | 原生仓库、AHC 权重、源图/掩码 |
| TRELLIS / TRELLIS.2 | `run_trellis_original_human_native_v1.py`、`run_trellis2_human_native_v1.py` | 对应模型环境与原生导出 |
| Hunyuan3D-2.1 / ReconViaGen | `run_human_static_two_models_20261115.py`、`human_static_two_models_20261115/` | 已冻结真实输入、各自权重、原生 GLB 或高斯表示 |
| OmniRe / SplatAD | `extract_omnire_human100_v2.py`、`extract_splatad_human100_v2.py`，以及 `native_reaudit_*` | 原生场景检查点、实例节点和源相机 |
| StreetGS | `native_reaudit_streetgs_v1.py`、`native_reaudit_streetgs_neighbors_v1.py` | DriveStudio 对应原生模型与实例映射 |

`model_preparation/20261110/scripts/` 还保留 ExAvatar、JacobianAvatar、Flexi、MHR 的环境接入和可行性试验。这些脚本不表示方法已进入最终统一成绩表，不应把所有安装版本串行执行。

## 3. 配准、单帧指标与 ED-P

1. 准备并核验真实源相机、资产局部坐标、时间和尺度关系。
2. 使用对应 `prepare_*_source*` / `render_*_source*` 适配器渲染，记录 RGB、alpha 与配准证据。
3. `run_human_metric_pipeline_v1.py --active-cohort` 增量计算已满足契约的单帧指标；`--status` 只读取现有本地汇总。
4. `run_human_edp_common.py` 按 `prepare` → `pose` → `features` → `score` 阶段处理六个部位；`export_human_edp_tables.py` 汇总。
5. `export_active_metric_tables.py` 导出核心指标与覆盖率。

PSNR、SSIM、IoU 与 LPIPS 的核心位于 `human_source_metrics_v1.py`。ED-P 的前景 patch 分配、特征池化及共同有效部位比较位于 `human_edp_common.py`。历史核心评分与后续 `evaluate_human_static_two_models.py` 使用的二维外观对齐分支分别保存；后者通过车辆仓库的 `reports/source_appearance_alignment_20261115_v1/run_comparison.py` 复用平移/等比缩放实现。不要把两种评分配置视作相同协议。

新增两模型的执行顺序是原生生成与重载 → `evaluate_human_static_two_models.py` 的视图/姿态/配准/源视图/评分阶段 → `run_human_static_two_models_staged.py` 的分阶段调度。后一启动器只在需要 GPU 的阶段获取租约。`wait_b0_then_human_static_two_models.py` 与 report 内的 worker 切换脚本是历史协调工具，不是新实验默认入口。

## 4. 源视频时序实验

`export_dynamic_source_assets.py` → `prepare_dynamic_source_clip.py` → `launch_dynamic_source_clip.py` → 源掩码审核/必要修订 → `score_dynamic_source_clip.py` → `export_dynamic_metric_tables.py`。

`launch_dynamic_full_cohort.py` 是批量控制器，`launch_dynamic_scoring_batch.py` 只发现已审核参考。评分使用原始真实曝光时刻与统一片段 ROI；`score_dynamic_source_clip.py` 实现 Farnebäck 光流差 tOF，以及两序列相邻 LPIPS 变化的差值 tLP100。单张图像生成的静态资产不具备相同源时间驱动序列，不能将其空缺直接补为零。网络视频定性资产的生产归入 OmniSim_Asset，不能自动加入已冻结的定量结果。

## 5. 人体可观测几何与耗时

目录为 `evaluation_static_cd_fscore_time_v1/scripts/`。

| 阶段 | 主要脚本 |
|---|---|
| 参考与协议准备 | `prepare.py`、`freeze_reference_and_time_plan.py`、`freeze_protocol.py` |
| 原生尺度/深度接口 | `measure_decoded.py`、`measure_splatad.py`、`depth.py`、`scene_depth.py` |
| 批量测量与评分 | `run_gpu.py`、`evaluate.py` |
| 时间测量 | `time_direct.py` / `time_direct_v2.py`、`time_sequence.py`、`time_ours*.py` |
| 验证与汇总 | `validate_cpu.py`、`validate_voxel.py`、`verify_geometry.py`、`audit_time.py`、`finalize_tables.py` |

该分支以同曝光的带标注单次 LiDAR 回波作为人体可观测表面参考；不能直接套用刚体多帧运动补偿后的几何参考。`protocol.json` 模板保存其尺度、原生透明度支持、深度射线与严格距离阈值规则。标注竖直高度条件下的结果与原生场景物理尺度诊断分开。历史协议内的 SHA256 绑定原记录；重新生成参考/计划后需重新冻结，不能关掉断言绕过。

## 6. 原生场景重提取与诊断

`native_reaudit_common.py` 提供共享接口；`native_reaudit_omnire_v1.py`、`native_reaudit_splatad_v1.py`、`native_reaudit_streetgs_v1.py` 处理方法原生节点。`native_reaudit_static_pipeline_v1.py` / `native_reaudit_resolved_pipeline_v1.py` 调度，`score_native_reaudit_static_v1.py` / `score_native_reaudit_geometry_v1.py` 评分。`settle_native_reaudit_support_v1.py`、`audit_native_reaudit_completion_v1.py` 汇总诊断。

被这些流程直接引用的少量 `reports/**/code/*.py` 与 probe 源码保留原路径；逐对象重复的 `source_archive/`、渲染结果与进程状态未进入本次源码包。历史结果依赖确切旧源码哈希时，应从原始归档恢复相应代码，不可用新发布路径适配后的文件冒充。

## 7. 固定前端缓存的后端消融

`restore_frontend_rgba_cache.py` → `prepare_backend_ablation.py` → `launch_backend_ablation.py` / `launch_backend_budget_control.py` → `evaluate_backend_ablation.py` → `export_backend_ablation_tables.py`。

批量入口为 `run_backend_ablation_batch_v2.py`，汇总入口为 `export_backend_ablation_batch_v2.py`。各变体复用前端视频、相机、骨架、掩码和初始化，分别记录预算。`backend_ablation_sampling.py` 保留分数曝光时刻的插值契约。已有八人消融结果留在原日期目录，不扩写为全体人体资产上的结论。

## 8. 展示与代码历史

`common_human_orbit_batch.py`、`launch_common_human_orbit.py`、`render_common_human_orbit.py` / `render_common_scene_orbit.py` 提供固定姿态环绕视图；`build_common_human_orbit_gallery.py`、`build_full_human_comparison_gallery.py`、`build_native_reaudit_gallery_v1.py` 构建图册。

`pilot`、`replacement`、`v1/v2` 与 `audit` 不自动意味着废弃：部分仍被新控制器作为实现调用。新实验按本索引选阶段，完整文件角色与来源见 [代码清单](CODE_INVENTORY_20261122.md)。
