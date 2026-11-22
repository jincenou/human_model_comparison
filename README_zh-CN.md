# 人体资产生成与比较实验

本仓库整理 nuScenes 人体资产的参考准备、各方法接入、源相机配准、单帧/时序指标、几何与耗时评测、后端消融及定性展示代码。**2026-11-22 更新为代码归档与流程整理，不重新生成资产，也不更新既有实验成绩。**

[实验入口与顺序](docs/EXPERIMENT_INDEX_zh-CN.md) · [全部源码索引](docs/CODE_INVENTORY_20261122.md) · [环境与复现](docs/REPRODUCING.md) · [English](README.md)

## 三个仓库的分工

| 仓库 | 内容 |
|---|---|
| [OmniSim_Asset](https://github.com/jincenou/OmniSim_Asset) | 刚体/人体生产管线、资产规范化、导出与复用 |
| 本仓库 | 人体实验、方法适配、静态/时序/几何/时间指标与消融 |
| [vehicle_model_comparison](https://github.com/jincenou/vehicle_model_comparison) | 刚体实验，以及人体复用的相机、原生渲染和几何测量工具 |

## 目录结构

```text
src/human_asset_benchmark/              可独立运行的指标、pair manifest 与路径解析
workspace/human_model_comparison/
  scripts/                              参考、生成、配准、渲染、评分、消融与审计
  scripts/human_static_two_models_20261115/
                                        Hunyuan3D-2.1 / ReconViaGen 原生生成与重载
  evaluation_static_cd_fscore_time_v1/scripts/
                                        人体可观测几何、耗时与协议验证
  model_preparation/20261110/scripts/    各方法环境接入及可行性试验历史
  reports/                              少量被实验直接引用的诊断源码；不含生成结果
  handoff/                              几何参考的只读核查脚本
metadata/templates/                     路径参数化的协议与名单模板
metadata/source_snapshot_20261122.json   本次收集的文件来源及原始/发布后 SHA256
metadata/source_provenance.json         累积来源记录
results/                               已有公开历史结果，按原日期保留
```

历史版本保留原路径，因为脚本之间存在导入和文件定位依赖；它们通过索引区分角色，**不应按文件名顺序全部运行**。批量启动器需要本地输入清单、已审查参考、模型环境和资源策略，源码存在不代表这些条件已满足。

## 快速验证

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py

# 合成样例仅验证接口与计算，不代表模型实验结果。
python tools/make_synthetic_example.py artifacts/example
hmc-score artifacts/example/pair.json --output artifacts/example/scores.json
```

LPIPS 使用本地权重与内容哈希；时序实验另需 OpenCV，姿态/特征提取、原生渲染和模型训练分别使用各自环境。依赖记录见 `environments/` 和 [方法说明](docs/METHODS.md)。不要把多个模型的 CUDA 环境合并成一个环境。

## 实验范围

| 实验 | 主要入口 |
|---|---|
| 单帧外观与轮廓 | `run_human_metric_pipeline_v1.py`、`human_source_metrics_v1.py` |
| 六部位 ED-P | `run_human_edp_common.py`、`human_edp_common.py` |
| 真实曝光时序 tOF / tLP | `prepare_dynamic_source_clip.py`、`score_dynamic_source_clip.py` |
| 新增静态方法 | `run_human_static_two_models_20261115.py`、`run_human_static_two_models_staged.py` |
| 可观测几何与 Time | `evaluation_static_cd_fscore_time_v1/scripts/` |
| 原生场景参与者重提取 | `native_reaudit_*`、`score_native_reaudit_*` |
| 固定前端缓存的后端消融 | `prepare_backend_ablation.py`、`run_backend_ablation_batch_v2.py` |
| 统一环绕视图 | `common_human_orbit_batch.py`、`render_common_human_orbit.py` |

完整执行顺序、输入依赖和当前/历史分支区分见 [实验索引](docs/EXPERIMENT_INDEX_zh-CN.md)。2026-11-12/13 的历史结果继续保留原协议；后续两种静态方法、配准修订、原生重提取和几何诊断具有独立协议，不能因为源码合并而自动合并成绩。

## 数据与运行路径

本仓库发布源码及轻量协议模板。数据集图片、人体模型、模型权重、原生资产、训练缓存、GPU 租约和运行状态留在本地。生产管线见 `OmniSim_Asset`，第三方模型遵循各自来源与条款，见 [THIRD_PARTY.md](THIRD_PARTY.md)。

路径由 `.env.example` 中的 `HMC_WORKSPACE`、`CONDA_ROOT`、`NUSCENES_ROOT` 等变量提供；配置模板通过 `hmc-materialize` 展开。跨仓库布局和原始哈希边界见 [复现说明](docs/REPRODUCING.md)。本次代码整理只验证源码与 CPU 指标契约，不声明在新机器上完成全部 GPU 训练与评测。
