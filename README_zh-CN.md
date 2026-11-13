# 人体资产对比实验

[English](README.md) · [复现步骤](docs/REPRODUCING.md) · [评测协议](docs/EVALUATION.md) · [方法与依赖](docs/METHODS.md)

本仓库整理仅基于 **nuScenes 冻结100人** 的人体资产实验代码，比较同一人物在同一真实时刻、同一观测动作下的重建表现。主流程包括参考审核、资产生成或场景提取、独立重载、统一相机渲染、指标和覆盖率统计，以及固定前端缓存的后端消融。

**当前为研究预发布版本。** 六模型共有526份统一源视角核心指标、509份有效ED-P。三处参考修正已合并；静态五方法共同77人、六方法共同46人，ED-P分别共同70人和40人。动态82人完成；2人参考分割不可靠，按用户要求停止追加修复，另16人缺少共同独立对象。后端消融按原有/新增两组各取编号最小4人，8人三组已全部完成并检查；新方法暂缓。

[8人后端消融](results/20261113/ablation/RESULTS.md) · [指标总表（各列最优加粗）](results/20261112/ALL_METRICS.md) · [输入预算说明](docs/INPUT_BUDGETS.md) · [动态缺测原因](results/20261112/dynamic/missing.csv)

## 统一大小与视角

1. 先核实模型坐标、单位、根运动和真实相机之间的关系，再做三维配准。
2. 同一人物的所有方法使用同一真实相机 K/T、曝光时刻、原始画布和参考定义的 ROI。
3. 不按各方法预测框分别居中或缩放，不根据评分挑角度，不逐关节修正姿态。
4. 环绕展示统一中心、距离、角度与画布；没有真实背面参考时不计算背面 PSNR 或 LPIPS。

当前发布的核心指标包含 PSNR、SSIM、LPIPS 和两种 IoU。ED-P采用冻结的DINOv3 ViT-L/16与DWPose六部位协议，同时统一比较人物和有效部位；属于本实验辅助版本。预测只使用自身 alpha 合成白底，不能乘真实 mask 擦除多余结构。

## 代码与复现

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python tools/make_synthetic_example.py artifacts/example
hmc-score artifacts/example/pair.json --output artifacts/example/scores.json
```

这个示例是合成数据的接口与数值检查，不代表真实实验成绩。完整实验需要自行取得 nuScenes、各方法代码及权重，并配置 [`.env.example`](.env.example) 中的路径。LPIPS 权重通过明确的本地路径和 SHA256 加载。

`workspace/human_model_comparison` 保留参考修复、场景提取、指定时刻重载、渲染和环境准备脚本；`workspace/human` 保留第一方生产流程；另外包含高斯运行时和必要共享辅助代码。绝对路径已参数化，原始和发布后代码哈希记录于 `metadata/source_provenance.json`。历史版本脚本用于追溯，具体执行顺序见复现文档。

## 数据与结果边界

- 100人身份与冻结曝光不因某个方法失败而更换。
- 参考保留衣服、背包及手持物，外部遮挡和截断需按参考资格规则处理。
- 主比较使用各方法共同有效的人物集合，报告实际 `n/100` 和各方法覆盖率。
- 技术失败记录 NA 和原因；合法空预测按定义计分。
- 数据集原图、视频、权重、大体积实验输出、旧筛选 HTML、安装缓存和凭据不随代码发布。
- 已有场景的 OmniRe／SplatAD 人体优先直接提取，不为接入而重新训练整场景。

项目级开源许可和论文引用信息尚待确认；第三方代码、模型、数据分别受原许可约束。当前不虚构论文标题、作者列表、会议或正式排行榜。


当前执行顺序见[执行入口](docs/CURRENT_WORKFLOW.md)；[早期批处理流程](docs/METRIC_BATCH_WORKFLOW_zh-CN.md)仅保留为历史实现记录。最新状态见[实验快照](docs/STATUS.md)、[数值结果](results/20261112/README.md)及[当前执行入口](docs/CURRENT_WORKFLOW.md)。
