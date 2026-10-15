# UWB 单对单测距偏置离线学习

本包实现一个**有向 UWB A→B 测距对**的静态 LOS 数据采集、离线数据集构建、基线/MLP 训练、固定测试集评估、模型导出和独立推理。它不做定位、滤波、在线训练，也不接入现有 ESKF、优化器或飞控。

当前状态：工程主体已编写，**Gate 1 已于 2026-07-23 通过现场验收**。已确认 UWB `1 -> 2` 为 50 Hz、两路动捕约 52 Hz 且同为 `world` frame，使用系统 ROS 时间/UWB 接收时间，单位外参与六维工作域也已写入配置。当前进入 Gate 2，尚未创建正式 session 或完成实体试采。详见 [环境与 ROS 接口清单](docs/environment_inventory.md) 和 [真实进度报告](docs/progress_report.md)。

严禁把历史 bag、软件合成数据、实时预览 CSV 或计划位姿当作正式实验结果。当前没有真实训练指标、曲线或“修正有效”的结论。

## 关键约定

- 相对位姿：`T_A_B = inverse(T_M_A) * T_M_B`。
- 输入：`rel_tx rel_ty rel_tz` 加按行展开的 `rel_R00 ... rel_R22`，共 12 维。
- 标签：`bias_m = range_m - distance_gt_m`，允许为正或负。
- 修正：`corrected_range_m = range_m - predicted_bias_m`。
- 四元数：`qx,qy,qz,qw`；位置/距离/偏置单位为米，ROS 时间单位为秒。
- 原始文件和运行目录只创建、不覆盖；重复命令应使用新的 session、attempt 或输出路径。
- `data_origin` 和 `is_synthetic` 必须由 session 元数据显式声明：物理实验只能是 `physical_experiment`/`false`，合成软件测试只能是 `synthetic_test`/`true`；不能因数据来自 ROS 或 bag 就推断为真实实验。

### 用刚体杆臂直接查看实物测距偏置

当动捕刚体原点不在标签/基站天线中心时，可对已经时间同步的逐帧 CSV 直接计算：

```bash
python3 scripts/estimate_range_bias_from_mocap.py \
  --input <SYNCED_RELATIVE_POSE_AND_UWB.csv> \
  --output outputs/reports/uwb_bias_<UNIQUE_TAG>.csv \
  --tag-position-body <TAG_X_M> <TAG_Y_M> <TAG_Z_M> \
  --anchor-position-base <ANCHOR_X_M> <ANCHOR_Y_M> <ANCHOR_Z_M> \
  --pose-direction base_to_body
```

默认输入列为 `range_m`、`rel_tx rel_ty rel_tz` 和
`rel_qx rel_qy rel_qz rel_qw`（四元数顺序 `qx qy qz qw`）。若 CSV 提供旋转矩阵，可加
`--rotation-columns rel_R00 rel_R01 rel_R02 rel_R10 rel_R11 rel_R12 rel_R20 rel_R21 rel_R22`。
输出逐帧真实距离与 `uwb_bias_m = range_m - true_distance_m`，并生成
`<output>.summary.json`。若输入存在 `is_valid` 列，默认只统计有效行。详细公式见
[坐标约定](docs/coordinate_convention.md)。

对于 `uwb_imu_tof_eskf_real_node.cpp` 使用的 4 个标签（5--8）、6 个基站
（0、1、2、3、4、9）和单路 `/vrpn_client/UWB_rocket/pose`，可直接从 bag 重建全部
24 条链路：

```bash
python3 scripts/analyze_multitag_uwb_bias_bag.py \
  --bag <INPUT.bag> \
  --output-csv <BIAS_SAMPLES.csv> \
  --summary-json <BIAS_SUMMARY.json>
```

该脚本使用 UWB 的 bag 接收时间插值动捕位姿，并复现节点中的
`rho_ij = norm(p_A_B + R_A_B b_j - a_i)`；默认同步门限为 0.04 s，动捕插值区间上限为
0.10 s。几何常量与节点 `loadGeometry()` 保持一致，节点标定量改变时必须同步更新脚本。

逐测距 CSV 可进一步绘制 24 条链路随时间变化的总览图和细节图：

```bash
python3 scripts/plot_multitag_uwb_bias_time.py \
  --input <BIAS_SAMPLES.csv> \
  --overview-output <BIAS_TIME_OVERVIEW.png> \
  --detail-output <BIAS_TIME_DETAIL.png> \
  --smooth-window-s 1.0
```

## Gate 顺序与当前阻塞

| Gate | 当前状态 | 进入下一 Gate 前的条件 |
| --- | --- | --- |
| Gate 1 接口确认 | 已放行 | 10 秒现场报告已确认三路接口、ID/方向、时间、频率、frame、工作域与单位外参 |
| Gate 2 坐标与采集 | 当前阶段；代码主体已写，未做真实试采 | 填写物理 session 元数据，生成计划、录 bag，并完成少量实体静态位姿试采 |
| Gate 3 试采质量 | 未开始 | 生成真实质量报告，确认同步/稳定阈值与字段是否合格 |
| Gate 4 正式数据集 | 流水线代码已写，未运行真实数据 | Gate 3 合格后才可正式采集、聚合和固定划分 |
| Gate 5 模型与报告 | 流水线代码已写，未运行真实训练 | 固定真实划分后比较三种基线与 MLP，再一次性打开测试集 |

Gate 1 放行只证明接口可用；不要把下面的 Gate 2–5 命令理解为已经完成实验。

## 0. 环境与安装

以下命令假定 `<CATKIN_WS>` 是当前 catkin 工作空间，且 ROS Noetic 已安装。ROS Python 包来自 Noetic；`requirements.txt` 只列 Python 依赖，其中 PyTorch 仅用于 Gate 5。

```bash
cd <CATKIN_WS>
source /opt/ros/noetic/setup.bash
python3 -m pip install --user -r src/uwb_bias_learning/requirements.txt
catkin_make
source devel/setup.bash
cd src/uwb_bias_learning
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

输入是本仓库和 `requirements.txt`；成功时 `catkin_make` 结束且包可导入。常见失败原因是未安装/未 source ROS Noetic、Python 3.8 不支持所选 PyTorch wheel、网络不可用或 `nlink_parser` 未构建。安装依赖本身不代表 ROS 接口或实验已验证。

## 1. Gate 1：检查 ROS 接口

先在 ROS master 和设备启动后列出当前话题；该发现动作不要求先填写未知 ID：

```bash
python3 scripts/inspect_ros_interfaces.py \
  --list-topics \
  --output-json outputs/reports/live_topics_<UNIQUE_TAG>.json
```

输出是实时话题/类型表和一个不可覆盖的 JSON；成功时退出码为 0。若 ROS master 不可达、环境未 source 或输出已存在则失败。

根据现场证据填写 `configs/ros_topics.yaml`，并在完整三路数据在线后运行定时检查：

```bash
python3 scripts/inspect_ros_interfaces.py \
  --config configs/ros_topics.yaml \
  --duration-s 10 \
  --output-json outputs/reports/interface_check_<UNIQUE_TAG>.json
```

输入是已确认的三路配置；输出含实际类型、频率、时间戳、frame、ID 和字段统计。只有三路发布且类型匹配时才成功。配置仍为 `null`、ID/方向未确认、类型不符、话题断流或输出已存在都会失败；不得用历史候选值绕过。

本次 Gate 1 已确认并固化：

- `configs/ros_topics.yaml`：最终三话题、`source_id 1 -> peer_id 2`、UWB `receive_time`、两路 `world` frame 和单位外参；
- `configs/collection.yaml`：Pilot 01 后将初始 `max_sync_gap_s` 调为 0.04，`max_mocap_sample_gap_s=0.10`；测距配置边界 `[-10, 10] m` 且非正值仍拒绝；
- `configs/pose_domain.yaml`：距离 0.1–2.0 m 及当前六维角度域；
- `outputs/reports/interface_check_20260723_gate1_confirmed.json`：10 秒现场验收原始报告。

进入 Gate 2 前仍须填写 session 元数据，明确 `data_origin`、`is_synthetic`、设备、环境、操作者和 LOS 条件。同步与稳定性阈值是试采起点，Gate 3 必须依据真实质量报告复核。

## 2. Gate 2：位姿计划、session、bag 与静态采集

### 2.1 生成物理摆放计划

仅在 `pose_domain.yaml` 六维范围确认后运行：

```bash
python3 scripts/generate_pose_plan.py \
  --config configs/pose_domain.yaml \
  --output data/interim/pose_plan_<UNIQUE_TAG>.csv
```

输出为不可覆盖的计划 CSV 及 `.manifest.json`；成功时打印计划数量。范围仍为 `null`、范围倒置或输出已存在会失败。计划值只指导摆放，绝不能替代动捕实测相对位姿。

### 2.2 创建不可变 session

先从 `configs/session_metadata.template.yaml` 另存一份新的 `<SESSION_METADATA.yaml>` 并完整填写。真实物理实验使用 `data_origin: physical_experiment`、`is_synthetic: false`；合成软件测试使用 `data_origin: synthetic_test`、`is_synthetic: true`，二者不可混用。

```bash
python3 -m uwb_bias_learning.session prepare \
  --collection-config configs/collection.yaml \
  --metadata <SESSION_METADATA.yaml> \
  --session-id <SESSION_ID>
```

输入为已放行的采集配置与完整元数据；默认输出 `data/raw/<SESSION_ID>/`，包含不可变 manifest、配置快照和事件日志，终端打印 JSON。配置未放行、元数据缺字段、session ID 非法或目录已存在会失败。

### 2.3 启动 rosbag

```bash
bash scripts/record_session.sh \
  --session-dir data/raw/<SESSION_ID>
```

脚本从 session manifest 读取 UWB、动捕 A、动捕 B 和状态话题，输出 `raw_topics*.bag`，并打印 session 路径、bag 前缀和话题。用 Ctrl+C 正常结束。manifest 缺失、话题未解析、同名前缀已有 bag、`rosbag` 不可用或 ROS master/数据源异常会失败。不要删除已有 bag 后复用同一个 session 来规避不可覆盖检查。

### 2.4 采集一个静态 pose

保持 rosbag 运行，在另一个已 source 的终端执行：

```bash
python3 scripts/capture_static_pose.py \
  --config configs/collection.yaml \
  --session-id <SESSION_ID> \
  --session-dir data/raw/<SESSION_ID> \
  --pose-id <POSE_ID> \
  --target-valid-ranges 200
```

输入为在线三路数据、session 和实体静态位姿；输出在该 pose 的新 attempt 目录，包括状态事件、逐测距 `raw_ranges.csv`、摘要和完成标记。成功时状态为 `COMPLETE` 且有效数量达到目标。这个实时 CSV 是**操作预览和现场诊断副本**，不是正式训练的 canonical 原始表；正式原始表必须在录制结束后由 bag 重建。

常见失败原因是必填配置仍有 `null`、单位外参未确认、frame/时间戳不符、错误 ID 方向、三路断流、同步/稳定性不合格、量程配置非法、超时、中途移动或已有完成 pose。`--yes` 只跳过摆放确认，不跳过健康/稳定检查；`--skip` 只记录一次跳过尝试。

### 2.5 查看完成与待采 pose

```bash
python3 scripts/list_pose_status.py \
  --plan data/interim/pose_plan_<UNIQUE_TAG>.csv \
  --session-dir data/raw/<SESSION_ID> \
  --output-json outputs/reports/pose_status_<UNIQUE_TAG>.json
```

输入是计划和 session；输出表格及可选不可覆盖 JSON，列出 `PENDING`、`COMPLETE`、不完整尝试等状态。计划/session 不存在、manifest/JSON 损坏或输出已存在会失败。只有存在合法 `COMPLETE` 标记的 attempt 才算完成。

## 3. Gate 3/4：从 bag 重建并构建固定数据集

### 3.1 canonical 原始表

```bash
python3 scripts/bag_to_raw_dataset.py \
  --config configs/collection.yaml \
  --session-id <SESSION_ID> \
  --session-dir data/raw/<SESSION_ID> \
  --input-bag data/raw/<SESSION_ID>/raw_topics.bag \
  --output-csv data/raw/<SESSION_ID>/canonical_raw_<UNIQUE_TAG>.csv \
  --summary-json data/raw/<SESSION_ID>/canonical_raw_<UNIQUE_TAG>.summary.json
```

可重复写多个 `--input-bag` 处理同一 session 的分包。输出 CSV 是后续质量检查的 **canonical 原始重建**，摘要记录输入/输出哈希和状态恢复统计。成功时打印 JSON 摘要。输出已存在、bag/session/config 哈希或接口不一致、状态事件非法、话题缺失或 bag 损坏会失败。

bag 重建保留未完成/中止 attempt 的原始行，但使其无效并写明原因；只有终态为 `COMPLETE` 的 attempt 才有资格进入后续训练候选。缺失终态不能按完成处理。

### 3.2 检查原始数据

```bash
python3 scripts/validate_raw_dataset.py \
  --input <CANONICAL_RAW.csv> \
  --config configs/preprocessing.yaml \
  --pose-domain-config configs/pose_domain.yaml \
  --source-id <CONFIRMED_SOURCE_ID> \
  --peer-id <CONFIRMED_PEER_ID> \
  --report outputs/reports/raw_quality_<UNIQUE_TAG>.json
```

输入是 canonical 原始表及确认配置；输出不可覆盖的质量 JSON。成功时打印 `status: PASS`。schema/来源混乱、ID 不符、pose 数量不足、同步/几何/稳定性问题、工作域违规、重复时间戳或输出已存在会失败。Gate 3 必须先审阅该真实报告，再决定是否调整阈值或补采，不能直接继续正式训练。

### 3.3 每个静态位姿聚合一行

```bash
python3 scripts/build_pose_dataset.py \
  --input <CANONICAL_RAW.csv> \
  --config configs/preprocessing.yaml \
  --source-id <CONFIRMED_SOURCE_ID> \
  --peer-id <CONFIRMED_PEER_ID> \
  --output data/processed/poses_<UNIQUE_TAG>.csv \
  --rejected-output data/processed/rejected_poses_<UNIQUE_TAG>.csv \
  --annotated-output data/interim/annotated_ranges_<UNIQUE_TAG>.csv \
  --summary-report outputs/reports/pose_build_<UNIQUE_TAG>.json
```

输出为通过 pose、拒绝 pose、保留逐行理由的审计表和可选摘要；成功时至少有一个 pose 通过全部质量门。正式数据必须提供已确认的有向 ID；`--allow-unconfigured-id-filter` 只供明确的软件/流程测试，不能生成正式 A–B 数据集。

### 3.4 固定 train/val/test 划分

```bash
python3 scripts/make_splits.py \
  --input data/processed/poses_<UNIQUE_TAG>.csv \
  --annotated-raw data/interim/annotated_ranges_<UNIQUE_TAG>.csv \
  --output-dir data/splits/<SPLIT_ID> \
  --split-mode auto
```

输出 `train.csv`、`val.csv`、`test.csv`、`raw_test.csv`、`normalization.json` 和 `splits.json`；成功时打印三组数量和规则。脚本按 session 优先，session 不足时才退化为同环境 pose/几何分组并发出限制说明。同一 pose 或近重复几何泄漏、比例/样本不足、real/synthetic 来源不一致或任一输出已存在会失败。归一化只由 train 拟合；测试集此后保持封闭。

## 4. Gate 5：训练、评估、导出与推理

### 4.1 三个基线

```bash
python3 scripts/train_baselines.py \
  --train data/splits/<SPLIT_ID>/train.csv \
  --val data/splits/<SPLIT_ID>/val.csv \
  --split-manifest data/splits/<SPLIT_ID>/splits.json \
  --output-dir outputs/runs/baselines_<RUN_ID>
```

输出零偏置、训练集常数偏置和 Ridge 三个基线及仅 train/val 指标；成功时明确提示没有读取 test。manifest 哈希/pose key/归一化不匹配、输出目录已存在或来源非法会失败。

### 4.2 MLP

```bash
python3 scripts/train_mlp.py \
  --train data/splits/<SPLIT_ID>/train.csv \
  --val data/splits/<SPLIT_ID>/val.csv \
  --split-manifest data/splits/<SPLIT_ID>/splits.json \
  --config configs/train_mlp.yaml \
  --output-dir outputs/runs/mlp_<RUN_ID> \
  --device cpu
```

输出每个 seed 的最佳验证 checkpoint、history、版本/哈希、选中 `model_state.pt` 和训练摘要；成功时明确按验证损失选择且没有读取 test。PyTorch 缺失、配置/lineage 不一致、非有限损失或输出目录已存在会失败。

对明确标记为 synthetic 的软件测试数据，上述两条训练命令必须额外显式传 `--allow-synthetic`；真实流程不得传此开关。合成数据仍只能说明软件链路可运行，不能作为实验指标。

### 4.3 一次性固定测试集评估

模型和超参数冻结后才运行：

```bash
python3 scripts/evaluate.py \
  --test data/splits/<SPLIT_ID>/test.csv \
  --raw-test data/splits/<SPLIT_ID>/raw_test.csv \
  --split-manifest data/splits/<SPLIT_ID>/splits.json \
  --baselines outputs/runs/baselines_<RUN_ID>/baselines.json \
  --mlp-state outputs/runs/mlp_<RUN_ID>/model_state.pt \
  --normalization outputs/runs/mlp_<RUN_ID>/normalization.json \
  --training-history outputs/runs/mlp_<RUN_ID>/seed_<SELECTED_SEED>/history.csv \
  --output-root outputs/evaluation_<EVAL_ID>
```

输出 `reports/` 下的固定测试指标/预测和 `figures/` 下的覆盖、偏置、修正前后、距离/方向及训练曲线。成功前会校验 test/raw-test、模型、基线、归一化和 split manifest 的哈希与来源。任一 lineage 不一致、raw-test 含无效/非 test 行、缺少测试 pose、输出目标已存在或 PyTorch/绘图依赖缺失会失败。

只有真实固定测试集上的修正指标优于 raw，才可以声称“修正有效”；MLP 不如常数或 Ridge 时必须如实保留结果。synthetic 评估必须显式加 `--allow-synthetic`，且只能标记为软件测试。

### 4.4 导出模型包

```bash
python3 scripts/export_model.py \
  --model-state outputs/runs/mlp_<RUN_ID>/model_state.pt \
  --normalization outputs/runs/mlp_<RUN_ID>/normalization.json \
  --train-config outputs/runs/mlp_<RUN_ID>/train_config.yaml \
  --split-manifest data/splits/<SPLIT_ID>/splits.json \
  --metrics outputs/evaluation_<EVAL_ID>/reports/metrics.json \
  --model-version <MODEL_VERSION> \
  --output-dir outputs/runs/export_<EXPORT_ID>
```

输出包含 eager checkpoint、TorchScript、归一化、训练配置、split manifest、固定测试 metrics 和 `model_spec.json` 的新目录；成功时 eager/TorchScript 探针一致且打印越界策略。输出目录已存在、版本为空、训练/测试 lineage 不一致或 TorchScript 校验失败会拒绝导出。synthetic 模型导出同样必须显式加 `--allow-synthetic`，导出规格会标记 test-only。

### 4.5 独立预测

```bash
python3 scripts/predict_bias.py \
  --model outputs/runs/export_<EXPORT_ID>/model_torchscript.pt \
  --tx <TX> --ty <TY> --tz <TZ> \
  --rotation-matrix <R00> <R01> <R02> <R10> <R11> <R12> <R20> <R21> <R22> \
  --range-m <OPTIONAL_RAW_RANGE_M>
```

输入是实测 `T_A_B` 的原始（未自行归一化）pose12；输出 `predicted_bias_m`、是否处于训练域、警告，以及可选的 `corrected_range_m`。旋转不属于 SO(3)、模型/规格缺失或字段/训练域不完整会失败。越界输入只报警，不裁剪。

## 5. 运行全部软件测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

输入是仓库内单元测试和临时目录中的明确合成数据；成功时 unittest 汇总为 `OK`。若环境未安装 PyTorch，依赖它的推理一致性测试可以显示 `skipped`，这不等于 Gate 5 模型链路已完整验证；正式交付前必须安装依赖后重跑。任何失败都应先修复，不能跳 Gate。

## 数据权威性与不可覆盖规则

```text
ROS 在线数据 ──► rosbag（保真记录） ──► bag_to_raw_dataset ──► canonical raw CSV
       └──────► capture_static_pose raw_ranges.csv（operational preview）
```

- rosbag 加状态事件是原始证据；`bag_to_raw_dataset.py` 的新输出是正式处理入口。
- live `raw_ranges.csv` 用于现场计数、暂停/重试和快速摘要，不与 canonical CSV 拼接，也不直接训练。
- 非 `COMPLETE` attempt 的行仍保留审计，但必须无效，不能进入训练。
- `data/raw/`、固定 split、训练/评估/导出目录均不得覆盖；重新运行请更换唯一标签。
- 大 bag、CSV、模型和图表默认不提交 Git。

字段定义见 [数据字典](docs/data_dictionary.md)，坐标和符号见 [坐标约定](docs/coordinate_convention.md)，实体操作见 [采集协议](docs/collection_protocol.md)。
