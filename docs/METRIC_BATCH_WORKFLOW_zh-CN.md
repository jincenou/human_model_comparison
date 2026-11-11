# 人体资产增量评测流程

此流程服务冻结的 nuScenes 100人名单。它复用已经生成或提取的资产，处理统一源相机下的准备、验证、逐人评分和同对象汇总。新资产生成后重复执行即可；缺资产、缺配准、参考不合格和计算失败分别保留，不换人。

## 已固定的共同条件

- 身份和真实曝光：沿用冻结100人名单，原75人与新增25人的身份、时刻均不修改。
- 参考：`configs/human_source_common_v1/reference_qualification_frozen_v1.json`。91人可用于当前完整人物外观指标；8人人体受外部遮挡、1人携带物受外部遮挡记NA，仍保留在100人总分母。自然自身遮挡不剔除，衣物、背包和携带物保留。
- 相机：`reviewed_source_v2/<HN编号>/camera.json` 固定原始 K/T、1600×900画布和微秒时间；公共ROI只由参考mask确定，长边扩1.10倍后输出512×512。不能按照各个预测的人体框重新居中、缩放或挑角度。
- 核心指标：`core_metric_protocol_frozen_v1.json` 已绑定PSNR、SSIM、LPIPS、IoU实现和权重哈希。ED-P仍缺特征/可见性协议，保持NA。
- 图像：预测由自身alpha合成白底，绝不使用参考mask擦掉多余结构。报告原始人物像素高度，512放大不增加真实细节。

## 当前服务器的重复执行命令

在人体实验目录，用已有 `ab_eval` 环境执行；这些步骤使用CPU，不抢占现有GPU任务：

```bash
python -B scripts/prepare_scene_registration_v2.py
python -B scripts/human_metric_batch_v1.py prepare \
  --registration-index reports/scene_source_registration_incremental_v2/latest_manifest.json
python -B scripts/human_metric_batch_v1.py run
```

也可以用一个入口完成扫描、准备、增量评分、CSV导出和图册更新：

```bash
python -B scripts/run_human_metric_pipeline_v1.py
# 只准备清单、资格与展示，不启动新的评分
python -B scripts/run_human_metric_pipeline_v1.py --prepare-only
```

第一步重新扫描已有 OmniRe / SplatAD 源视角结果，检查身份、真实时刻、原生重载记录、相机投影、坐标系尺度、渲染脚本哈希和alpha编码。证据不足的历史结果不会自动通过。检查批次按内容哈希保存，因此相同输入再次扫描不会改写旧批次。

第二步生成100人×所选方法的准备清单。默认方法为Ours、AH、原版TRELLIS、OmniRe和SplatAD。通过 `--methods` 可登记TRELLIS.2及已列入注册表的后续方法；登记不代表其配准已经完成。可重复传入 `--registration-index` 合并不同方法适配器的证书清单，重复且冲突的证书会被拒绝。

第三步只计算 `ready` 项。输入键绑定参考、相机、配准证书、指标协议及实现哈希。已有成功结果在相关输入仍一致时复用；新结果、修复后的配准或者改变的协议产生新结果文件，不覆盖历史成绩。只有一个评分进程能持有运行锁，数据盘不足10 GiB停止新增计算。失败会保留原因，修复后可重跑。

如果只想重建汇总和图册：

```bash
python -B scripts/human_metric_batch_v1.py report
```

## 输出位置

默认 `reports/source_camera_metric_batch_v1/`：

| 文件/目录 | 内容 |
| --- | --- |
| `latest_plan.json`、`plans/` | 当前准备清单和不可变历史快照 |
| `state.json` | 评分PID、进度、复用数、失败数 |
| `results/<method>/<id>/<input_key>.json` | 逐人物、逐方法的核心指标与输入证据 |
| `failures/` | 真实计算失败及原因；不以白图伪造结果 |
| `summary.json` | 覆盖率、缺测原因和同一人物集合的配对汇总 |
| `metrics.csv` | 100人×所选方法的平铺表；含指标状态、NA原因和相机哈希 |
| `index.html`、`media/` | 同一公共ROI的真实参考与已通过方法；待配准方法留空 |

原生预览图册仍用于检查模型资产本身，不能替代上述共同相机图册。源视角是重建评测；没有真实背面参考时不计算背面PSNR/LPIPS。

## 后续方法如何接入

先按方法需要的真实输入完成生成/拟合/提取，并保存固定真实时刻的原生资产及独立重载证据。完成源相机配准后，再由经过验收的适配器生成证书；不能只把一个JSON的状态改成`passed`。

适配器必须核查：身份、微秒时刻、原生姿态时间轴与根运动、相机K/T、统一尺度/位置、完整协方差及方向相关外观、白底alpha编码。模型自身姿态错误应保留；原生相机或T-pose不能替换源姿态。

一个被接受的注册索引包含：

- `qualification`与`qualification_sha256`：指向相同冻结参考资格文件。
- `rows`：每行包含`id`、`method`、状态；通过者为`certified_for_core_metrics`，并提供`registration_record`与其SHA256。
- 配准证书：`identity`（HN编号）、`identity_key`、`timestamp_us`、`camera_sha256`、`method`、`status`、`registration_certified`；原始画布预测RGB/alpha路径与SHA256；`prediction_encoding=white_composited_uint8`；渲染验证文件与脚本路径/哈希；未按指标搜索姿态尺度的明确记录。

现有证书示例在 `reports/scene_source_registration_incremental_v2/batches/`。人体批处理会检查证书、身份/时间/相机及所有输入哈希；它无法替代原生渲染器的几何验收。新方法需要自己的认证适配器，不能复用OmniRe证书冒充已验收。

`REGISTRY`只用于寻找原生结果、区分“缺资产”和“待配准”；实际评分始终以独立配准证书为门槛。新模型添加注册表条目和一个适配器后，沿用同一准备、评分、缓存与汇总流程。

## 当前需要补齐的工作

- Ours、AH和原版TRELLIS：原生结果大部分/全部已存在，源相机配准仍待验收。
- OmniRe：89份通过当前证书检查；HN0004历史试运行记录的渲染脚本哈希与当前脚本不同，需要核实归档代码或重做验证。另有原生初始化过滤案例。
- SplatAD：60份通过当前证书检查；部分新增人物旧渲染缺完整的结构化原生参数/物理时间验证，需补验收。不能将其当前NA解释为模型质量差。
- ED-P、固定姿态16方位环绕、固定前端缓存后端消融仍在原任务范围内。

完整五方法共同有效集合未形成前，不把两种场景方法的配对结果标成完整主表，也不按各自不同成功子集的均值做排名。
