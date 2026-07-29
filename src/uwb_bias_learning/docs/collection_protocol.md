# 静态位姿采集协议

## 适用范围

仅用于一个明确方向的 UWB A–B 测距对、LOS、静态位姿。不得用于 NLOS、动态采集、在线训练、定位或滤波实验。

## 采集前放行条件

以下条件必须全部满足：

1. `configs/ros_topics.yaml` 中三个最终话题、在线消息类型、UWB source/peer ID 和方向已经现场确认。
2. A、B 动捕消息都有非零 Header，处于同一 `frame_id`，频率和时间间隔已经记录。
3. UWB 时间策略已明确；若使用接收时间，session 元数据明确记录 `receive_time`。
4. `/use_sim_time=false`，没有运行 `simulation_both.launch` 或任何在真实 UWB 话题发布合成消息的节点。
5. 用户确认两个刚体原点在天线中心、坐标轴与天线系对齐；否则先提供外参，不能采集后再猜。
6. `pose_domain.yaml` 六个维度均有用户确认的范围。
7. session 元数据完整，LOS 为 true，输出 session/pose 不存在。

`inspect_ros_interfaces.py` 或采集程序任一预检失败时停止，不能用命令行默认值绕过。

## 每个 session

1. 从 `session_metadata.template.yaml` 复制一份新文件并填写设备、固件、模式、环境、操作者、刚体名和安装说明。
2. 用 `record_session.sh` 创建新的 session。脚本使用独占目录，已有 session ID 会报错。
3. rosbag 持续记录 UWB、动捕 A、动捕 B 和状态事件话题。
4. 每次开机、跨日期、改变全局区域或明显改变环境布置时使用新的 `session_id`。
5. 正常中断 rosbag，保留 session 事件日志和配置快照。

## 每个静态 pose

1. 选择计划表中仍为 `PENDING` 的 `pose_id`。
2. 操作者将设备移动到计划位姿；计划值仅用于摆放指导。
3. 采集程序发布 `SETTLING` 状态，并等待至少 `settling_time_s`。
4. 在 `stability_window_s` 内同时检查：
   - A 的绝对位置和 SO(3) 旋转稳定度；
   - B 的绝对位置和 SO(3) 旋转稳定度；
   - `T_A_B` 的相对位置和 SO(3) 旋转稳定度。
5. 稳定度合格后发布 `COLLECTING`，每条目标方向 UWB 测距在自己的时间戳处分别插值 A、B 位姿。
6. 位置使用相邻样本线性插值；姿态使用归一化四元数 SLERP。
7. 任一侧最近动捕样本时间差超过 `max_sync_gap_s`、数据断流、ID 不符、测距无效或设备移动时，该测距不计入有效数量，并记录拒绝原因。
8. 稳定性失效时发布 `PAUSED`；重新稳定后才继续，不把移动段混入静态段。
9. 收满 `target_valid_ranges` 后发布 `COMPLETE`，关闭该 pose 的 CSV，并输出只基于实际收集值的快速摘要。
10. 已存在的 `pose_id` 不能覆盖。重试使用新的 attempt 或显式的新 pose ID，并在备注中关联。

## 初始阈值

当前配置保存的是任务书建议起点，不是实测结论：

```yaml
target_valid_ranges: 200
min_valid_ranges: 180
settling_time_s: 1.5
stability_window_s: 1.0
max_position_std_m: 0.005
max_rotation_std_deg: 0.5
max_sync_gap_s: 0.04
```

`max_sync_gap_s` 原起点为 0.02 s；2026-07-24 Pilot 01 在约 54 Hz 动捕下频繁触发正常间隔超限，故 Pilot 02 调为 0.04 s。其余数值仍是起点。Gate 3 试采质量报告完成后才能决定正式阈值；每次调整必须创建新 session 并保存快照，不能根据最终测试集表现反向选择。

## 覆盖与 session 设计

- 计划使用 Latin Hypercube 加中心、单轴边界、典型点和少量重复点，避免六维完整网格爆炸。
- 最终训练输入只取动捕实测 `T_A_B`，绝不使用名义计划值替代。
- 试采建议 30–50 个不同位姿；正式采集尽量数百个位姿、至少 3 个独立 session，实际数量由物理可操作性决定。
- 同一相对位姿在不同全局位置重复少量样本，用于识别无法由相对位姿解释的环境多径。
- 每个位姿的 200 条测距用于估计均值和波动，聚合后仍只是一个主要几何样本。

## 失败和中断

- 断流、时间倒退、frame 改变、四元数非法或同步间隔超限：暂停并记录原因。
- Ctrl+C：先发布/写入中断事件，安全关闭 CSV；rosbag 由记录脚本接收 SIGINT 后关闭。
- 任何原始 CSV 和 bag 只创建、不覆盖。错误文件保留并在状态列标记，不删除以“清理数据”。
