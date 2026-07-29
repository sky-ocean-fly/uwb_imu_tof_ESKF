# UWB 偏置离线学习真实进度报告

更新时间：2026-07-23 UTC

## 结论

按“代码主体”衡量，目前约完成 **95%**；按任务书的严格 Gate 验收衡量，**Gate 1 已正式放行，当前进入 Gate 2**。剩余工作主要是实体试采、正式多 session 数据采集和真实训练/测试。

当前没有伪造或声称存在任何真实 bag、真实训练曲线、真实测试指标或测距修正结论。历史 bag 只能证明部分旧接口 schema，不能生成本任务的双动捕标签；软件测试中的 synthetic 数据也不能作为实验结果。

本轮软件验证结果：`catkin_make --pkg uwb_bias_learning` 此前通过；Python 编译和
shell 语法检查通过；配置更新后 `unittest` 共运行 103 项，其中 102 项通过，
1 项因当前环境未安装 PyTorch 而跳过。10 秒真实 ROS 接口验收另行通过，但尚不
代表实体试采、数据质量或模型效果通过。

## 按 Gate 统计

| Gate | 已编写 | 已验证 | 尚未完成/阻塞 |
| --- | --- | --- | --- |
| Gate 1 接口确认 | 环境、仓库、消息定义、接口检查和配置 | 10 秒现场报告确认 UWB `1 -> 2` 为 50.0 Hz；A/B 动捕均为 `PoseStamped`、`world`、约 52.0 Hz；系统时间、UWB 接收时间、单位外参与工作域已确认 | 无；Gate 1 已放行。现场报告为 `outputs/reports/interface_check_20260723_gate1_confirmed.json` |
| Gate 2 坐标与采集 | 变换/SO(3)、SLERP 同步、稳定性/质量门、接口检查、session、bag 记录、实时采集、状态恢复、bag 重建、位姿计划/状态查询 | 纯算法和合成/临时输入测试；真实三流接口检查已通过 | 尚未填写正式 session 元数据、生成最终计划、录正式 bag 或完成实体试采；阈值仍须由试采校准；live CSV 仅为 operational preview，canonical 原始表须从 bag 重建 |
| Gate 3 试采质量 | 原始行审计、拒绝原因、覆盖/同步/稳定性/几何报告代码 | 软件级数据契约与合成测试 | 真实试采为 0；没有真实质量报告，不能确定阈值、丢帧率、异常比例或偏置分布 |
| Gate 4 正式数据集 | 逐行注释、离群审计、每 pose 聚合、合法旋转平均、固定划分、近重复防泄漏、train-only 归一化、raw test 选择与 SHA-256 lineage | 软件级聚合/划分测试 | 正式数据为 0；没有覆盖图、拒绝/补采清单或真实 train/val/test；需 Gate 3 合格并完成多 session 实体采集 |
| Gate 5 模型与报告 | 零/常数/Ridge、三 seed 小型 MLP、验证集 early stopping、固定测试评估、图表、模型导出、越界预警推理、lineage 校验 | 不依赖真实数据的模型/指标逻辑测试；synthetic 必须显式允许 | 真实训练、真实测试、模型包和实验图表均为 0；PyTorch 依赖和完整导出/推理需在目标环境最终复验 |

## 已完成的工程主体

- 独立 catkin/Python 包，不修改现有 ESKF、优化、飞控或仿真代码。
- Gate 1 环境清单和证据等级，区分现场、源码、历史 bag 和候选值。
- 已确认的 live 接口、ID、频率、工作域与单位外参已写入配置；测试仍验证缺失必填值时正式入口失败关闭。
- 坐标方向 `T_A_B = inverse(T_M_A) * T_M_B`、四元数 `qx,qy,qz,qw`、SO(3) 合法性与 12 维 `pose12` 特征。
- 偏置符号 `range - distance_gt` 和修正方向 `range - predicted_bias`。
- UWB 时刻位置线性插值、四元数 SLERP，不外插并记录同步间隔。
- A、B 绝对位姿与 A→B 相对位姿的静态性检查；移动/断流/时间异常进入暂停或拒绝理由。
- session/attempt 不可覆盖、配置快照、事件日志、rosbag 和状态话题。
- 行级原始/注释表、每 pose 聚合、标准差/SEM、离群标记与拒绝清单。
- 固定 train/val/test、近重复 pose 防泄漏、train-only normalization 与文件哈希 lineage。
- 三个基线、小型 MLP、仅依据 val 选模；evaluate 才读取固定 test/raw-test。
- 导出 TorchScript、model spec、归一化、split/metrics；推理越界报警但不裁剪。

## Gate 1 已确认值与仍需试采复核的阈值

- UWB：`/nlink_linktrack_nodeframe2`，`nlink_parser/LinktrackNodeframe2`，物理 A 为 ID 1、B 为 ID 2，保留 `1 -> 2`，接收时间策略，实测 50.0 Hz。
- 动捕：`/vrpn_client/UWB_A/pose` 与 `/vrpn_client/UWB_B/pose`，均为 `geometry_msgs/PoseStamped`、`world`，10 秒实测约 52.0 Hz，Header 非零。
- Gate 1 的 10 秒检查中 A/B 各 520 条有效位姿、各 5 个 sequence 缺口，最大 Header 间隔约 70 ms；初始 `max_mocap_sample_gap_s=0.10`。2026-07-24 的 Pilot 01 表明 `max_sync_gap_s=0.02` 会拒绝约 54 Hz 动捕的正常抖动，因此 Pilot 02 改用 0.04；两项阈值都必须继续复核。
- 用户确认两刚体原点在天线中心且轴对齐；距离域 0.1–2.0 m。角度域采用初始方位 ±60°、俯仰 ±20°、roll ±30°、pitch ±40°、yaw ±90°。
- UWB 配置边界按用户输入保存为 `[-10, 10] m`，同时保持 `reject_nonpositive=true`，所以可训练物理测距的有效区间实际为 `(0, 10] m`。

## 数据来源和训练资格

- session 元数据必须显式声明 `data_origin` 和 `is_synthetic`；允许的配对只有物理实验 `physical_experiment`/`false` 与合成测试 `synthetic_test`/`true`。ROS 在线或 rosbag 文件本身不能证明是物理实验。
- live `capture_static_pose.py` 的 `raw_ranges.csv` 只用于现场操作预览、计数和快速诊断。
- 录制结束后，由 `bag_to_raw_dataset.py` 结合状态事件重建的新 CSV 才是后续质量检查的 canonical 原始输入。
- `SETTLING`、`PAUSED`、`ABORTED`、缺失终态或其他非 `COMPLETE` attempt 的行保留用于审计，但不得成为有效训练行。
- synthetic 数据只能进入软件测试目录；训练、评估和导出都必须显式传 `--allow-synthetic`，产物不得表述为真实实验结果。
- 正式评估必须同时使用固定 pose-level `test.csv` 和同一组 test pose 的 `raw_test.csv`，并通过 split manifest 和 SHA-256 lineage 校验。

## 尚未完成的实际工作

1. 确认物理摆放能力和 session 设计，填写本次物理实验的完整 session 元数据。
2. 生成位姿计划、启动 bag 记录并完成少量实体位姿试采，生成 Gate 3 质量报告并据实调整阈值。
3. 完成至少多个独立 session 的正式 LOS 静态采集；一个 pose 的约 200 条测距聚合后仍只算一个几何样本。
4. 审核覆盖与拒绝/补采清单，固定真实 train/val/test 后封闭 test。
5. 仅用 train/val 训练和选模，最后一次性评估 test，再决定能否声称修正有效。
6. 在安装 PyTorch 的目标环境重跑被跳过的一致性测试，并完成独立导出/推理复验。

## 下一次需要用户提供/操作的内容

进入真实试采前，用户下一步需要提供本次 session 的环境、地点、操作者、UWB 型号/固件/测距模式、刚体名与安装描述等元数据，并确认是否具备按计划调节六维位姿的物理条件。随后配合程序提示完成实体摆放与静止等待。

当前正确状态是“Gate 1 已放行、Gate 2 待实体试采”，不是“实验或模型训练已完成”。
