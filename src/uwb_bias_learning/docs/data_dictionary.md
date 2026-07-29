# 数据字典

本文只定义文件契约，不包含真实样本、统计量或实验结果。CSV 为 UTF-8；四元数统一按 `qx,qy,qz,qw`；距离和偏置为米，时间为秒，带 `_raw` 的设备计数不冒充秒。

## 1. 原始逐测距表 `raw_ranges.csv`

每一行对应配置中精确 `source_id -> peer_id` 的一条 UWB 测距尝试。无 CIR 的消息保持 `cir_quality` 为空，不用 RSSI 或 0 伪填。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 表结构版本，当前为 1 |
| `data_origin` | 由不可变 session 元数据显式声明；只允许物理实验 `physical_experiment` 或合成软件测试 `synthetic_test`，不能因消息来自 ROS/bag 就推断为真实 |
| `is_synthetic` | 显式 YAML 布尔值；`physical_experiment` 必须为 0/false，`synthetic_test` 必须为 1/true |
| `ingestion_path` | 数据进入表格的路径：实时操作预览 `ros_live`、canonical bag 重建 `rosbag_replay`，或软件测试 `synthetic_unit_test`；它不表示真实/合成来源 |
| `session_id` | 独立实验会话 ID |
| `environment_id` | 来自不可变 session 元数据的环境 ID |
| `pose_id` | 静态位姿 ID |
| `attempt_id` | 该 pose 的一次不可覆盖尝试 ID |
| `sample_index` | 本次采集内的目标测距顺序号 |
| `capture_state` | 测距到达时的 `SETTLING`、`COLLECTING` 或 `PAUSED` |
| `uwb_timestamp` | 用于同步的 UWB 时间，秒 |
| `uwb_receive_timestamp` | ROS 接收或 bag 记录时间，秒 |
| `uwb_timestamp_source` | `receive_time` 或未来明确实现的设备时钟映射；当前映射未实现时拒绝运行 |
| `range_m` | 原始 UWB 测距，米 |
| `source_id`,`peer_id` | 配置确认的外层 source 和内层 peer，有向且不自动反向 |
| `uwb_source_id`,`uwb_peer_id` | 上述有向 ID 的显式别名，便于审计 |
| `anchor_id`,`node_id` | 任务书兼容列；物理角色未确认时保持空，不能从 source/peer 猜测 |
| `source_role`,`peer_role` | 消息中原始 role 值；不能据此自行推断 A/B |
| `local_time_raw`,`system_time_raw` | UWB 帧原始设备计数；未映射前不视为 ROS 秒 |
| `fp_rssi`,`rx_rssi` | 消息原始质量字段；单位未确认 |
| `cir_quality` | CIR 质量；NodeFrame2 不提供，因此为空 |
| `mocap_a_frame_id`,`mocap_b_frame_id` | 两路动捕原始 frame；必须与配置一致 |
| `mocap_[a/b]_timestamp_before/after` | 包围 UWB 时间的动捕 Header 时间戳 |
| `mocap_[a/b]_interpolation_alpha` | 位置线性插值和四元数 SLERP 的比例 |
| `mocap_[a/b]_gap_before_s/after_s` | UWB 时间到两侧动捕样本的时间差 |
| `sync_gap_[a/b]_s` | 两侧时间差中的较小值，即任务书规定的“最近动捕样本”间隔 |
| `A_px A_py A_pz` / `B_px B_py B_pz` | UWB 时刻插值得到的世界系位置，米 |
| `A_qx ... A_qw` / `B_qx ... B_qw` | UWB 时刻 SLERP 得到的世界系姿态 |
| `rel_tx rel_ty rel_tz` | `T_A_B = inverse(T_M_A) * T_M_B` 的平移，米 |
| `rel_qx ... rel_qw` | 同一相对旋转的四元数 |
| `rel_R00 ... rel_R22` | 同一相对旋转矩阵，按行展开 |
| `distance_gt_m` | `norm([rel_tx,rel_ty,rel_tz])` |
| `range_error_m` | 有符号误差 `range_m - distance_gt_m` |
| `stability_[a/b/relative]_position_m` | A、B 和相对位姿窗口的位置 RMS 离散度 |
| `stability_[a/b/relative]_rotation_deg` | A、B 和相对位姿窗口的 SO(3) 测地角 RMS |
| `is_los` | session 是否明确为 LOS |
| `is_mocap_valid` | 两路动捕是否均能在该 UWB 时刻合法同步 |
| `is_stable` | A、B、相对位姿三个检查是否全部稳定 |
| `is_sync_valid` | 同步是否有效且未外插/超阈值 |
| `is_range_valid` | 测距是否通过有限值、正值和已配置量程检查 |
| `is_valid` | 仅 `COLLECTING` 且 LOS、动捕、同步、稳定性、测距和上游状态均有效时为 1 |
| `invalid_reason` | 分号分隔的拒绝原因；原值保留，不静默丢弃 |
| `operator_note` | 操作者备注 |

空字段表示消息没有提供、无法同步或语义尚未确认，不表示数值 0。

### 原始表的权威层级

- `capture_static_pose.py` 在 attempt 内生成的 `raw_ranges.csv` 是 **operational preview**，用于现场计数、暂停/重试和快速摘要，不直接进入正式训练。
- rosbag 同时保存三路原始话题和状态事件；`bag_to_raw_dataset.py` 从 bag 新建的 CSV 是后续质量检查和数据集构建的 **canonical 原始重建**。
- 两类表用 `ingestion_path` 区分，不得拼接后把同一测距重复计数。
- canonical 重建保留非 `COMPLETE` attempt 的行以便审计，但这些行必须为无效并带 `attempt_not_complete`、`attempt_aborted` 等理由；只有 `COMPLETE` attempt 的合格行才可进入训练候选。

## 2. 行级审计表

`validate_raw_dataset.py` 和 `build_pose_dataset.py` 保留所有原始列，并增加：

| 字段 | 含义 |
| --- | --- |
| `processing_status` | `candidate`、`valid`、`outlier`、`rejected` 或 `pose_rejected` |
| `processing_invalid_reason` | 校验/几何/整 pose 拒绝原因，不覆盖原 `invalid_reason` |
| `source_processing_status` | 输入中已有处理状态的备份，防止覆盖来源证据 |
| `is_outlier` | 是否被配置化 Hampel/MAD 规则剔除 |
| `outlier_reason` | 离群字段、偏差和阈值 |
| 相对位姿、`distance_gt_m`,`range_error_m` | 缺失时由原始 A/B 位姿重新计算；已有值必须一致 |

## 3. 每静态位姿一行的聚合表

主键为 `session_id + pose_id`。

| 字段组 | 含义 |
| --- | --- |
| `environment_id`,`data_origin`,`is_synthetic` | 从 session/逐行继承的环境和来源；冲突或混合 real/synthetic 会被拒绝 |
| `rel_t*`,`rel_q*`,`rel_R*` | 平移算术均值、合法四元数旋转平均及其矩阵 |
| `distance_gt_mean`,`distance_gt_std` | 静态段真值距离均值和样本标准差 |
| `range_mean_raw`,`range_median_raw`,`range_std_raw`,`range_mad_raw`,`range_q05/q25/q75/q95_raw` | 离群处理前的测距统计 |
| `range_mean_filtered`,`range_median_filtered`,`range_median` | 配置化处理后的测距统计 |
| `bias_mean_raw`,`bias_median_raw`,`bias_std_raw`,`bias_mad_raw`,`bias_q05/q25/q75/q95_raw` | 处理前的有符号偏置统计 |
| `bias_mean`,`bias_median_filtered`,`bias_std`,`bias_sem` | 训练标签及过滤后统计；`bias_sem=bias_std/sqrt(range_count_valid)` |
| `row_count_total`,`base_rejected_count`,`range_count_raw`,`range_count_valid`,`outlier_count` | 数量审计 |
| `position_stability_m`,`rotation_stability_deg` | 相对位姿段内离散度 |
| `mean_fp_rssi`,`mean_rx_rssi` | 有值时的均值；第一版模型不使用 |
| `outlier_method`,`outlier_center`,`outlier_mad`,`outlier_cutoff` | 离群规则审计参数 |
| `quality_status`,`rejection_reason` | pose 是否进入主数据集及原因 |

## 4. 固定划分

- `train.csv`、`val.csv`、`test.csv`：聚合行及 `norm_<pose12字段>`；归一化只拟合 train。
- `raw_test.csv`：仅固定 test pose、`processing_status=valid` 且 `is_outlier=false` 的逐条测距，用于修正前后比较。
- `normalization.json`：固定字段顺序、train 均值/尺度/最小值/最大值。
- `splits.json`：划分规则、seed、各 split 的 session/pose key、real/synthetic 三态来源，以及输入、train/val/test、normalization、annotated raw、raw test 的 SHA-256。

`splits.json` 的 provenance 是下游强制契约：synthetic split 默认拒绝训练、评估和导出，只有明确的软件测试命令使用 `--allow-synthetic` 才能继续；该开关不能把 synthetic 产物变成真实实验结果。

固定网络输入顺序为：

```text
rel_tx rel_ty rel_tz
rel_R00 rel_R01 rel_R02
rel_R10 rel_R11 rel_R12
rel_R20 rel_R21 rel_R22
```

模型输出 `predicted_bias_m`；修正始终为 `corrected_range_m = range_m - predicted_bias_m`。
