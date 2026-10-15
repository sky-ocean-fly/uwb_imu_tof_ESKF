# 坐标、四元数与偏置约定

## 状态

本文给出代码必须遵守的数学约定。用户已于 2026-07-23 确认 A、B 刚体原点分别位于天线中心，且刚体轴与对应天线轴对齐，因此本次装置使用单位外参。

## 坐标系

- `{M}`：动捕世界坐标系。
- `{A}`：UWB A 天线坐标系。
- `{B}`：UWB B 天线坐标系。
- `T_X_Y`：把 `{Y}` 中的点变换到 `{X}`，满足 `p_X = R_X_Y p_Y + t_X_Y`。

动捕原始位姿定义为 `T_M_A` 和 `T_M_B`。B 相对 A 的位姿严格为：

```text
T_A_B = inverse(T_M_A) * T_M_B
R_A_B = transpose(R_M_A) * R_M_B
t_A_B = transpose(R_M_A) * (p_M_B - p_M_A)
```

真实天线中心距离：

```text
distance_gt_m = norm(t_A_B)
```

代码和文档不得把该变换写成 `T_B_A`，也不得直接用世界坐标差作为相对平移而遗漏 A 的旋转。

## 四元数

- CSV、配置和 Python 公共 API：`qx, qy, qz, qw`。
- ROS `geometry_msgs/Quaternion`：字段 `x, y, z, w`。
- 所有输入先检查有限值并归一化。
- `q` 和 `-q` 表示同一旋转；特征必须完全一致。
- 插值使用最短弧 SLERP；不得对四个分量直接线性插值后不归一化。
- 静态段旋转聚合先做四元数符号对齐，再求合法旋转平均；不得把 9 个旋转矩阵元素的算术平均直接当作旋转矩阵。

## 网络输入

固定 12 维顺序：

```text
rel_tx rel_ty rel_tz
rel_R00 rel_R01 rel_R02
rel_R10 rel_R11 rel_R12
rel_R20 rel_R21 rel_R22
```

旋转矩阵按行展开。任何模型包必须保存并校验此字段顺序。

## 偏置符号

每条测距：

```text
range_error_m = range_m - distance_gt_m
```

每个静态位姿的标签：

```text
bias_mean = mean(range_error_m)
```

预测后修正：

```text
range_corrected_m = range_m - predicted_bias_m
```

偏置允许为正或负，代码不得取绝对值、ReLU 或强制非负。

## 刚体原点不在 UWB 天线中心时

若动捕给出机体相对基站的位姿 `T_base_body = [R_base_body, t_base_body]`，标签
天线相位中心在机体系的坐标为 `p_body_tag`，基站天线相位中心在基站系的坐标为
`p_base_anchor`，则：

```text
p_base_tag = R_base_body * p_body_tag + t_base_body
delta_base = p_base_tag - p_base_anchor
distance_gt_m = norm(delta_base)
uwb_bias_m = range_m - distance_gt_m
```

若动捕相对位姿恰好相反，为 `T_body_base`，则在机体系中等价计算：

```text
p_body_anchor = R_body_base * p_base_anchor + t_body_base
delta_body = p_body_tag - p_body_anchor
distance_gt_m = norm(delta_body)
```

两种写法的距离必须一致。坐标必须是天线相位中心（或设备定义的测距参考点），而不只是
外壳边缘；所有量均使用米。可用 `scripts/estimate_range_bias_from_mocap.py` 对同步后的 CSV
逐帧计算并汇总。脚本默认读取本包 canonical CSV 的 `rel_t*`、`rel_q*`、`range_m` 字段，
默认把相对位姿解释为 `T_base_body`；实际 A/B 角色不同时必须显式选择正确方向。

## 单位

- 距离、位置、偏置：米。
- 时间：秒；原始设备计数另存为 `_raw` 字段，不冒充秒。
- 程序内部角度：弧度。
- 配置、计划表和面向操作者的报告角度：度，并在字段名写 `_deg`。

## SO(3) 合法性

每个用于训练的旋转矩阵必须满足：

```text
transpose(R) * R ~= I
det(R) ~= 1
```

超出配置容差时样本进入拒绝清单，不自动投影后静默接受。若内部平均算法执行 SVD 投影，必须在聚合后重新检查并保留段内旋转离散度。
