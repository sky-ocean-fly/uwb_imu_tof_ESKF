# Gate 1：环境与 ROS 接口清单

- 审计时间：2026-07-23（UTC）
- 工作空间：`/home/yu/catkin_workspace`
- 任务范围：单对单 UWB 测距偏置离线学习
- 当前结论：**Gate 1 已于 2026-07-23 完成 10 秒三路现场验收并放行。**

初次审计时只有动捕 A 在线；随后用户启动最终 UWB 与动捕 B，并确认物理 A/B 分别对应 UWB ID 1/2、单位外参和工作域。最终报告 `outputs/reports/interface_check_20260723_gate1_confirmed.json` 确认三路发布、类型匹配、时间/frame/ID 合格。本文保留历史证据章节，但以本次最终现场报告为当前状态。

## 1. 证据等级

本文严格区分以下证据，避免把旧代码、历史数据或候选配置写成当前事实：

| 标记 | 含义 |
| --- | --- |
| 现场确认 | 2026-07-23 从当前 ROS master 读取并采样得到 |
| 源码确认 | 可由当前工作树中的消息定义、解析器、launch 或配置直接证明 |
| 历史 bag 确认 | 可由仓库已有 rosbag 证明，但不代表当前设备配置 |
| 候选/待确认 | 仅有旧代码或配置线索，不能作为新采集接口 |

## 2. 环境与工作空间

| 项目 | 结果 | 状态 |
| --- | --- | --- |
| 操作系统 | Ubuntu 20.04.6 LTS，Linux 6.8 x86_64 | 已确认 |
| ROS | ROS 1 Noetic，`ROS_VERSION=1`、`ROS_DISTRO=noetic` | 已确认 |
| Python | `python3 3.8.10`，`ROS_PYTHON_VERSION=3`；系统另有 Python 2.7.18 | 已确认 |
| 工作空间类型 | 标准 catkin 工作空间；`src/CMakeLists.txt` 指向 Noetic 的 `catkin/cmake/toplevel.cmake` | 已确认 |
| 构建工具 | `catkin_make` 可用；`catkin`（catkin_tools）命令未安装 | 已确认 |
| 构建输出 | `build/`、`devel/`，`CATKIN_DEVEL_PREFIX=/home/yu/catkin_workspace/devel` | 已确认 |
| CMake / GCC | CMake 3.16.3，GCC 9.4.0 | 已确认 |
| Git 基线 | `8a516b099e8dd788370e533b6bf8280207f123b0` | 已确认 |
| 当前工作树 | 已存在大量用户修改及生成物变化 | 已确认；本任务不得清理或覆盖 |
| Python 工程配置 | 当前没有 `requirements.txt`、`pyproject.toml`、`setup.py`、`setup.cfg`、Pipfile 或 Conda 环境文件 | 已确认 |

当前 catkin 顶层包含三个包：

| 包 | 版本 | 主要用途 |
| --- | --- | --- |
| `nlink_parser` | 2.1.0 | LinkTrack/ToF 串口协议、ROS 消息及 UWB 记录脚本 |
| `imu_mahony` | 0.0.0 | IMU、地距处理及一个旧 UWB/地距记录器 |
| `uwb_pose_estimation` | 0.0.1 | 现有定位、优化、ESKF 和仿真；不属于本任务修改范围 |

根目录的 `uwb_imu_tof_to_gpt/` 位于 catkin 的 `src/` 之外，且其 `package.xml` 仍声明包名 `uwb_pose_estimation`。它是遗留副本，不作为本任务开发目标。

## 3. 仓库结构盘点

### 3.1 主要目录

```text
catkin_workspace/
├── build/                         # catkin 生成物，已有用户变化
├── devel/                         # catkin 生成物，已有用户变化
├── result/                        # 大量历史 UWB bag/CSV
├── src/
│   ├── CMakeLists.txt -> /opt/ros/noetic/share/catkin/cmake/toplevel.cmake
│   ├── nlink_parser/              # UWB 解析器、消息、launch、旧记录脚本
│   ├── imu_mahony/                # IMU/ToF 与旧同步记录器
│   ├── uwb_pose_estimation/       # 定位、ESKF、仿真；禁止为本任务改动
│   └── uwb_bias_learning/         # 本任务新目录；Gate 1 目前仅含本文档
├── NatNet.py                      # 未完成的 NatNet 示例，不可用作接口实现
├── tof_platform_sim.py            # ToF 平台仿真，与本任务无关
└── uwb_imu_tof_to_gpt/            # src 外遗留副本
```

`NatNet.py:57-69` 明确说明其解析只是临时示例，代码没有实际解析并填充刚体数据；其中示例刚体 ID 不能作为本任务动捕或 UWB ID 证据。

### 3.2 README

- `src/nlink_parser/README.md`
- `src/nlink_parser/README.en.md`
- `src/nlink_parser/extern/serial/README.md`
- `src/nlink_parser/src/utils/nlink_unpack/README.md`
- `src/nlink_parser/src/utils/nlink_unpack/README.en.md`
- `src/nlink_parser/src/utils/protocol_extracter/README.md`
- `src/nlink_parser/src/utils/protocol_extracter/README.en.md`

工作空间根目录和另外两个现行 catkin 包均无 README。

### 3.3 Launch 文件

- `src/nlink_parser/launch/`：`iot.launch`、`linktrack0.launch`、`linktrack1.launch`、`linktrack2.launch`、`linktrack3.launch`、`linktrack_aoa.launch`、`linktrack_mixed.launch`、`linktrack_rviz.launch`、`tofsense.launch`、`tofsensem.launch`、`vrpn_client.launch`
- `src/imu_mahony/launch/`：`ground_distance.launch`、`uwb_sync_record.launch`
- `src/uwb_pose_estimation/launch/`：`all_systems.launch`、`simulation_both.launch`、`uwb_imu_tof_eskf.launch`、`uwb_pose_estimation.launch`
- src 外遗留副本：`uwb_imu_tof_to_gpt/all_systems.launch`、`uwb_imu_tof_to_gpt/uwb_imu_tof_eskf.launch`

### 3.4 包、构建和 YAML 文件

- 现行包：`src/nlink_parser/package.xml`、`src/imu_mahony/package.xml`、`src/uwb_pose_estimation/package.xml`
- 内嵌 serial 包：`src/nlink_parser/extern/serial/package.xml`
- src 外遗留包：`uwb_imu_tof_to_gpt/package.xml`
- 现行顶层构建：三个现行包各自的 `CMakeLists.txt`
- 内嵌库构建：serial、`nlink_unpack`、`protocol_extracter` 的 `CMakeLists.txt`
- 现行 YAML：`src/uwb_pose_estimation/config/config.yaml`（空文件）、`sim_data_generator.yaml`、`uwb_imu_tof_eskf.yaml`、`uwb_imu_tof_eskf_sim.yaml`
- src 外遗留 YAML：`uwb_imu_tof_to_gpt/uwb_imu_tof_eskf.yaml`

## 4. 现有相关代码和可复用边界

| 现有部分 | 可复用内容 | 不可直接复用的原因 |
| --- | --- | --- |
| `nlink_parser` 消息与 NodeFrame2 解析器 | UWB 消息类型、ID、测距、RSSI 的真实字段；协议帧校验 | 没有 ROS Header、没有无效值状态、没有三流同步 |
| `record_uwb_distance.py` | 遍历 `msg.nodes` 并取外层/内层 ID、`dis`、RSSI 的方式 | 使用接收时间；不保存设备时间；不筛选固定方向；不记录双动捕；`open(..., 'w')` 会覆盖；NodeFrame4 会伪填 RSSI 为 0，不可沿用 |
| `imu_mahony/src/uwb_sync_recorder.cpp` | 展开 NodeFrame2 字段的写法 | 只拼接最新地距，不是时间同步；两处 `ros::Time::now()`；最多 9 个节点；`ios::trunc` 覆盖原文件；地距消息类型与当前发布端不一致 |
| `pose_compare_logger.cpp` | 两个 `PoseStamped` 的 ApproximateTime 用法、四元数 x/y/z/w 到 Eigen 的转换 | 只同步两个派生位姿，不含 UWB；没有插值或 SLERP；默认绝对路径且覆盖输出 |
| `utils.cpp` 的 `makeT`/`inverseT` | 齐次变换构造和求逆可作为交叉参考 | 现有实物真值流程应用旧的非单位轴变换，不能直接用于本任务 |
| `sim_data_generator_node.cpp:202-203` | 数学公式 `R_A_B=R_W_A^T R_W_B`、`p_A_AB=R_W_A^T(p_W_B-p_W_A)` 可用于核对 | 仿真数据、仿真偏置和仿真结果不得作为真实实验数据或指标 |
| 历史 bag/CSV | 验证 schema、历史频率、字段取值与旧话题 | 只有一个动捕刚体，无 `pose_id`/静态段标记，不能生成本任务标签 |

仓库没有双动捕加 UWB 的三流同步器，没有位置插值加四元数 SLERP 的实现，也没有现成的 PyTorch、MLP、基线训练、数据集划分或推理代码。

`src/uwb_pose_estimation/src/utils.cpp:378-431` 中名为 `Calculate_positive_bias` 的函数所有分支都返回 0，且没有调用者。它不是已有偏置模型，也不是实验结论，不得复用或视作“零偏置已验证”。

`/mocap_truth_pose` 与 `/eskf_mocap_truth_pose` 是现有估计器加工后重新发布的派生真值：它们只缓存最新 B 位姿、应用旧坐标变换，并用新的发布时间戳发布。新任务必须记录两路原始 VRPN 位姿，不能订阅这些派生话题代替原始真值。

## 5. UWB 接口

### 5.1 话题与消息

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 候选最终话题 | `/nlink_linktrack_nodeframe2` | 源码确认、历史 bag 确认；当前未发布 |
| 消息类型 | `nlink_parser/LinktrackNodeframe2` | `msg/LinktrackNodeframe2.msg`，MD5 `5e015baafc881617fa57f6ba7fd27747` |
| 发布结构 | 一个外层上报节点，加一个动态 `nodes[]` 数组 | 消息定义和 `linktrack/init.cpp:188-227` |
| 测距字段 | `nodes[i].dis` | `msg/LinktrackNode2.msg` |
| 测距单位 | 米 | 原始 signed int24 除以 `MULTIPLY_DIS=1000`；现有消费者明确按米使用 |
| 外层 ID | `msg.id` | 旧定位代码将其视为 source/tag ID |
| 对端 ID | `nodes[i].id` | 旧定位代码将其视为 target/anchor ID |
| 质量字段 | `nodes[i].fp_rssi`、`nodes[i].rx_rssi` | 消息定义确认 |
| RSSI 单位 | 未确认 | 解析器按 `raw/-2` 缩放，但仓库没有物理单位说明 |
| CIR | 不存在 | NodeFrame2 消息没有该字段 |
| 显式有效标志 | 不存在 | 无 validity/status 字段 |
| ROS Header | 不存在 | 消息定义没有 `std_msgs/Header` |

完整外层字段为：

```text
role, id, local_time, system_time, voltage,
pos_3d[3], eop_3d[3], vel_3d[3], angle_3d[3], quaternion[4],
imu_gyro_3d[3], imu_acc_3d[3], nodes[]
```

每个 `nodes[]` 元素包含：

```text
role, id, dis, fp_rssi, rx_rssi
```

一条消息可能包含多个对端测距，也可能 `nodes=[]`。提取目标对时必须按外层 `msg.id` 和内层 `node.id` 查找，不得把固定数组下标当作 ID。

### 5.2 ID、角色和方向

旧定位工程使用：

- source/tag 候选：5、6、7、8；
- target/anchor 候选：0、1、2、3、4、9；
- 缓存方向：`distances_cache[msg.id][node.id]`。

这只是旧的多节点定位配置，不是本任务单对单 A/B 的确定映射。

历史 NodeFrame2 数据还显示：

- 代表 bag `result/UWB_true_13.bag` 中外层 ID 5、6、7、8 各约 100 Hz，总计约 400 Hz；
- `linktrack_mixed.launch` 启动四个串口节点，它们向同一个全局话题发布，历史 bag 显示该话题有四个 publisher connection；
- 同一 `nodes[]` 中可同时出现旧工程所谓的标签和基站；
- 历史数据中外层和内层 `role` 均为 0，不能靠 `role` 推断 A/B；
- USB 端口、launch 节点名与外层 ID 的历史对应并不稳定，不能从 `/dev/ttyCH343USBx` 推断 ID。

因此，本任务的 A/B 数值 ID和提取方向仍未确认。若两个方向都发布，也不能在没有用户决定的情况下自行平均或合并。

### 5.3 UWB 时间戳

NodeFrame2 没有 `header.stamp`，只有帧级 `uint32 local_time` 和 `uint32 system_time`。解析器原样复制二者，没有转换为 ROS 时间。原始每节点协议结构另有一个 `system_time`，但该字段没有暴露到 ROS 的 `LinktrackNode2` 消息中。

历史数据中这两个字段的增量与毫秒计数一致，仿真代码也用 `elapsed_ms` 填充；但仓库没有证明真实设备的时间原点、跨设备同步、回绕行为或它与 ROS/VRPN 的对应关系。因此：

- 不能把 `local_time` 或 `system_time` 直接当作与动捕同基的时间；
- 当前旧记录脚本中的 `%time` 是 rosbag/回调接收时间，不是 UWB 消息自带的 ROS Header；
- Gate 2 适配器必须同时保存原始设备时间和 ROS 接收时间；
- 是否能建立设备时钟到 ROS 时钟的映射，必须在 UWB 在线后检查；不能现在写死。

### 5.4 无效测距

源码没有定义无效距离 sentinel：

- 帧长度或校验和无效时，解析器丢弃整帧；
- 目标节点没有测距时，已经观察到的表现是该 `node.id` 缺席或整个 `nodes` 数组为空；
- 现有 ESKF 把非有限值或 `dis <= 0` 丢弃，只能视为消费端防御规则，不是设备协议保证。

对仓库现有 NodeFrame2 bag 的只读扫描共覆盖 910857 条消息和 5511446 个嵌套测距；观察到 39406 条空 `nodes` 消息，未观察到 NaN、Inf、零或负距离。该结果不能证明硬件在其他故障模式下不会输出特殊正数 sentinel；仍需设备文档或现场断链测试确认。

### 5.5 其他协议话题

解析器还支持 NodeFrame3、NodeFrame4、NodeFrame5 和 TagFrame0。README 明确说明实际发布哪一种取决于 NAssistant 的设备协议配置。历史数据强烈指向 NodeFrame2，但最终仍必须以现场 `rostopic type` 为准；不能仅根据话题字符串选择消息类。

特别地，历史 `result/test_data/UWB_testtrue_1.bag` 中名为 `/nlink_linktrack_nodeframe3` 的话题实际类型是 `LinktrackNodeframe2`，说明字符串猜类型会出错。

## 6. 动捕接口

### 6.1 动捕 A：现场确认

| 项目 | 现场结果 |
| --- | --- |
| 话题 | `/vrpn_client/UWB_A/pose` |
| 类型 | `geometry_msgs/PoseStamped` |
| publisher | `/vrpn_client` |
| `header.stamp` | 非零，连续递增 |
| `header.frame_id` | `world` |
| 位置字段 | `pose.position.x/y/z` |
| 四元数字段 | `pose.orientation.x/y/z/w` |
| 四元数范数 | 10 秒窗口约 `0.99999995` 至 `1.00000010` |
| 当前实际频率 | 约 53 Hz；`rostopic hz` 窗口约 52.5–57.4 Hz |
| 配置轮询频率 | `update_frequency=100.0` Hz；它不等于实际发布频率 |

10 秒现场检查中，时间戳无零值、无重复、`frame_id` 全为 `world`；帧间隔最大约 60 ms，存在少量小于 1 ms 的成对到达和大于 40 ms 的间隔。两个相邻检查窗口中，一个没有 `header.seq` 缺口，另一个 500 消息窗口观察到 2 个序号缺口。因此不能仅依据 launch 的 100 Hz 参数假定无丢帧，正式采集程序必须持续记录间隔和缺口。

现场接收时间相对 Header 的延迟约为亚毫秒量级；一个 500 消息窗口的中位值约 0.284 ms、范围约 0.091–0.634 ms。

### 6.2 动捕 B：现场确认

最终接口为 `/vrpn_client/UWB_B/pose`、`geometry_msgs/PoseStamped`。10 秒验收收到 520 条有效位姿，`header.stamp` 全部非零、`frame_id=world`、四元数范数有效，Header 频率约 51.95 Hz，最大间隔约 70 ms，观察到 5 个 sequence 缺口。

旧历史接口 `/vrpn_client/UWB_rocket/pose` 只属于旧 bag，不用于本任务。

### 6.3 当前 VRPN 配置

2026-07-23 的现场参数为：

```yaml
server: 192.168.50.208
port: 3883
trackers: [UWB_A, UWB_B]
update_frequency: 100.0
frame_id: world
```

`use_server_time` 没有显式设置；本机安装的 `vrpn_client_ros` 示例将默认行为说明为 `use_server_time: false`，现场 Header 与客户端接收时间最大差约 1.54 ms，与客户端系统 ROS 时间一致。

当前没有 `/tf` 话题，VRPN 原始接口是 `PoseStamped`，不是 `TransformStamped`。

### 6.4 四元数顺序

ROS `geometry_msgs/Quaternion` 的字段顺序由消息定义明确为：

```text
x, y, z, w
```

现有代码在构造 `Eigen::Quaterniond` 时按 Eigen 的构造顺序 `(w, x, y, z)` 重新排列并归一化，这一字段读取方式可作为参考。NodeFrame2 外层自带的 `float32[4] quaternion` 没有在消息定义中说明数组顺序，不能拿它代替 VRPN 四元数或作为本任务真值。

## 7. 当前 ROS 现场检查

最终 10 秒只读检查结果：

```text
Nodes:
  /rosout
  /vrpn_client
  /linktrack0

Published data topics:
  /vrpn_client/UWB_A/pose [geometry_msgs/PoseStamped]
  /vrpn_client/UWB_B/pose [geometry_msgs/PoseStamped]
  /nlink_linktrack_nodeframe2 [nlink_parser/LinktrackNodeframe2]
```

全局 `/use_sim_time` 参数未设置，工具报告的有效值为 false；当前是系统 ROS 时间，不是仿真 `/clock`。UWB 消息没有 Header，因此明确使用 ROS 接收时间，同时保留设备 `local_time`/`system_time` 原始计数。

## 8. 三个最终数据流的 Gate 1 状态

| 数据流 | 当前结论 | Gate 1 状态 |
| --- | --- | --- |
| UWB | `/nlink_linktrack_nodeframe2` + `LinktrackNodeframe2`，500 条均为 `1 -> 2`，约 50.0 Hz | 已现场确认 |
| 动捕 A | `/vrpn_client/UWB_A/pose` + `PoseStamped`，world、有效 Header、约 52.0 Hz | 已现场确认 |
| 动捕 B | `/vrpn_client/UWB_B/pose` + `PoseStamped`，world、有效 Header、约 52.0 Hz | 已现场确认 |

三个数据话题、类型、方向、frame 与时间策略均已确认，Gate 1 放行。同步与稳定性阈值仍需 Gate 2/3 实体试采验证。

## 9. 历史数据能证明什么

仓库内共有 59 个 bag，其中 56 个位于 `result/` 并包含 NodeFrame2 UWB 数据，16 个同时包含旧 B 话题 `/vrpn_client/UWB_rocket/pose`。代表数据：

```text
result/UWB_true_13.bag
duration: 104.202498 s
/nlink_linktrack_nodeframe2: 41680 messages, 4 connections
/vrpn_client/UWB_rocket/pose: 4625 messages
```

按外层 ID 分组后，ID 5、6、7、8 各约 100 Hz；旧 B 动捕平均约 44.4 Hz。历史 B 的 Header 非零且 frame 为 world。

这些 bag 没有动捕 A、没有两个世界位姿、没有 `pose_id`/采集状态事件，也没有本任务的 session 元数据。因此它们只能用于 schema 与解析器回归测试，不能计算本任务要求的

```text
T_A_B = inverse(T_M_A) * T_M_B
```

更不能作为新任务的训练集、测试集或实验指标。

## 10. 坐标与外参冲突

任务书给定第一版约定：A、B 动捕刚体原点位于各自 UWB 天线中心，轴已对齐，刚体到天线外参均为单位变换。

现有实物定位代码却：

- 只订阅 B 动捕；
- 假设动捕世界 `M` 与旧系统的 `AM` 重合；
- 在 `src/uwb_pose_estimation/src/utils.cpp:486-501` 和 `src/uwb_pose_estimation/src/uwb_imu_tof_eskf_node.cpp:108-138` 对 A、B 应用同一个非单位轴旋转矩阵；
- 现有多节点 UWB 的天线位置也相对 A/B 机体系有非零安装坐标。

这与新任务的单位外参声明表面冲突。可能原因是旧代码服务于另一套多天线平台，而新任务使用新的单对单、天线中心刚体；但仓库本身不能证明这一点。

用户已明确确认本次装置使用单位外参。因此：

- 不复用旧非单位轴矩阵；
- 不把旧 `/mocap_truth_pose` 当原始真值；
- 只使用两路原始 VRPN 位姿，按本任务公式生成真值。

## 11. 已创建的独立包

本任务采用新的独立 catkin/Python 包 `src/uwb_bias_learning/`，避免改动现有 ESKF、优化法、飞控或仿真代码。

当前已创建：

```text
src/uwb_bias_learning/
├── CMakeLists.txt
├── package.xml
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── ros_topics.yaml
│   ├── collection.yaml
│   ├── pose_domain.yaml
│   ├── preprocessing.yaml
│   └── train_mlp.yaml
├── docs/
│   ├── environment_inventory.md
│   ├── coordinate_convention.md
│   ├── collection_protocol.md
│   ├── data_dictionary.md
│   └── progress.md
├── scripts/
│   ├── inspect_ros_interfaces.py
│   ├── generate_pose_plan.py
│   ├── record_session.sh
│   ├── capture_static_pose.py
│   ├── bag_to_raw_dataset.py
│   ├── validate_raw_dataset.py
│   ├── build_pose_dataset.py
│   ├── make_splits.py
│   ├── train_baselines.py
│   ├── train_mlp.py
│   ├── evaluate.py
│   ├── export_model.py
│   └── predict_bias.py
├── src/uwb_bias_learning/
│   ├── __init__.py
│   ├── config.py
│   ├── ros_adapters.py
│   ├── transforms.py
│   ├── synchronization.py
│   ├── quality.py
│   ├── features.py
│   ├── datasets.py
│   ├── models.py
│   └── metrics.py
├── tests/
│   ├── test_transforms.py
│   ├── test_bias_sign.py
│   ├── test_synchronization.py
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_inference_consistency.py
├── data/{raw,interim,processed,splits}/
└── outputs/{runs,figures,reports}/
```

真实 ROS adapter 已实现并通过接口检查。当前仍不计划修改：

- `src/nlink_parser/`
- `src/imu_mahony/`
- `src/uwb_pose_estimation/`
- `uwb_imu_tof_to_gpt/`
- 任何已有 bag、CSV、build/devel 文件或用户未提交修改

## 12. Gate 2 仍需用户回答或配合的项目

接口、ID、外参与工作域已经确认。进入实体试采仍需：

1. **物理可调能力**：是否有转台、滑轨，哪些维度可独立调节，哪些只能手动摆放。
2. **session 能力**：能否跨日期、重新开机、改变全局摆放区域，并在至少 3 个独立 session 中重复少量相对位姿。
3. **session 元数据**：UWB 型号、固件、测距模式、功率/电池、环境、地点、操作者、刚体名、安装描述、反射物、温度等。
4. **现场操作**：根据计划移动实体并保持静止，配合完成少量 Gate 2 试采。
