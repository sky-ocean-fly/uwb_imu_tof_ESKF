#!/usr/bin/env python3
"""
将包含多个静止位姿的连续 ROS1 rosbag 转换为 UWB 偏置学习 CSV。

默认话题：

- 动捕 A：/vrpn_client/UWB_A/pose
- 动捕 B：/vrpn_client/UWB_B/pose
- UWB：/nlink_linktrack_nodeframe2
- 采集标记：/uwb_bias/collection_marker（std_msgs/String）

标记消息的 data 必须是 JSON，例如：

    {"event":"STATIC_START","pose_id":2,"stamp":1784892381.17,
     "source":"phone"}
    {"event":"MOVE_START","pose_id":2,"stamp":1784892383.35,
     "source":"phone"}

分段边界始终使用标记消息在 rosbag 中的记录时间，不使用 JSON 内的
``stamp``。同一 ``pose_id`` 的 STATIC_START 到 MOVE_START 构成一个静止段；
如果 bag 结束时仍有未闭合的 STATIC_START，则以 bag 结束时间闭合。

对每个静止段，脚本执行：

1. 将每条有效 UWB 测距与 A、B 动捕位姿同步；
2. 从该段全部成功对齐的数据中选取中间 50 条；
3. 检查相邻 UWB 时间间隔不超过 0.1 s；
4. 检查 A、B 绝对位姿及 A 到 B 相对位姿均满足静止阈值；
5. 对通过检查的 50 条测距生成一行训练数据。

标签定义为：

    bias_mean = mean(range_k - distance_gt_k)
              = mean(range_k) - mean(distance_gt_k)

输出：

1. processed_bias.csv：每个合格静止位姿一行，可直接交给
   uwb_bias_training.py；
2. raw_aligned.csv：每条成功对齐的 UWB 测距一行，包含所属分段、
   是否为中间 50 条、是否用于标签以及拒绝原因。

用法示例：

    source /opt/ros/noetic/setup.bash
    source ~/catkin_workspace/devel/setup.bash

    python3 rosbag_to_bias_csv.py data/run_003.bag \
        --session-id run_003 \
        --processed-output datasets/processed/run_003.csv \
        --raw-output datasets/raw/run_003_raw.csv

一次处理文件夹中的全部 bag：

    python3 rosbag_to_bias_csv.py data \
        --recursive \
        --processed-output datasets/processed/all.csv \
        --raw-output datasets/raw/all_raw.csv

录包时必须包含以下四个话题：

    rosbag record -O data/run_003 \
        /nlink_linktrack_nodeframe2 \
        /vrpn_client/UWB_A/pose \
        /vrpn_client/UWB_B/pose \
        /uwb_bias/collection_marker

依赖：

- ROS Noetic 的 Python 3 环境与 rosbag
- 已编译并 source nlink_parser 工作空间
- numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROCESSED_FIELDS: List[str] = [
    "session_id",
    "pose_id",
    "marker_pose_id",
    "segment_index",
    "bag_file",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "segment_closed_by",
    "selected_start_sec",
    "selected_end_sec",
    "available_range_count",
    "rel_tx",
    "rel_ty",
    "rel_tz",
    "rel_R00",
    "rel_R01",
    "rel_R02",
    "rel_R10",
    "rel_R11",
    "rel_R12",
    "rel_R20",
    "rel_R21",
    "rel_R22",
    "rel_qx",
    "rel_qy",
    "rel_qz",
    "rel_qw",
    "distance_gt_mean",
    "distance_gt_std",
    "range_mean",
    "range_std",
    "bias_mean",
    "bias_std",
    "range_count",
    "position_stability",
    "rotation_stability",
    "pose_a_position_stability",
    "pose_a_rotation_stability",
    "pose_b_position_stability",
    "pose_b_rotation_stability",
    "fp_rssi_mean",
    "rx_rssi_mean",
    "rssi_diff_mean",
    "measurement_span_sec",
    "measurement_rate_hz",
    "max_uwb_gap_sec",
    "pose_a_sync_mean_sec",
    "pose_a_sync_max_sec",
    "pose_b_sync_mean_sec",
    "pose_b_sync_max_sec",
    "quality_ok",
]


RAW_FIELDS: List[str] = [
    "session_id",
    "pose_id",
    "marker_pose_id",
    "segment_index",
    "bag_file",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "segment_closed_by",
    "available_range_count",
    "selected_max_uwb_gap_sec",
    "in_marked_static_segment",
    "selected_middle_block",
    "used_for_label",
    "segment_quality_ok",
    "segment_reject_reason",
    "uwb_timestamp",
    "time_from_bag_start_sec",
    "uwb_local_time",
    "uwb_system_time",
    "local_role",
    "local_id",
    "remote_role",
    "remote_id",
    "range_m",
    "fp_rssi",
    "rx_rssi",
    "rssi_diff",
    "A_px",
    "A_py",
    "A_pz",
    "A_qx",
    "A_qy",
    "A_qz",
    "A_qw",
    "B_px",
    "B_py",
    "B_pz",
    "B_qx",
    "B_qy",
    "B_qz",
    "B_qw",
    "rel_tx",
    "rel_ty",
    "rel_tz",
    "rel_R00",
    "rel_R01",
    "rel_R02",
    "rel_R10",
    "rel_R11",
    "rel_R12",
    "rel_R20",
    "rel_R21",
    "rel_R22",
    "distance_gt_m",
    "range_error_m",
    "pose_a_nearest_dt_sec",
    "pose_b_nearest_dt_sec",
]


class ProcessingError(RuntimeError):
    """输入数据无法安全转换时抛出的错误。"""

#动捕单帧
@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray

#在UWB时刻插值后的动捕快照
@dataclass(frozen=True)
class PoseSample:
    position: np.ndarray
    quaternion: np.ndarray
    nearest_dt_sec: float

#单条UWB测距
@dataclass(frozen=True)
class UwbMeasurement:
    timestamp: float
    local_time: int
    system_time: int
    local_role: int
    local_id: int
    remote_role: int
    remote_id: int
    range_m: float
    fp_rssi: float
    rx_rssi: float


@dataclass(frozen=True)
class CollectionMarker:
    """使用 rosbag 记录时间表示的一条采集标记。"""

    timestamp: float
    event: str
    marker_pose_id: str
    source: str


@dataclass(frozen=True)
class StaticInterval:
    """由 STATIC_START 和 MOVE_START（或 bag 结束）定义的静止段。"""

    segment_index: int
    marker_pose_id: str
    start_timestamp: float
    end_timestamp: float
    closed_by: str


@dataclass(frozen=True)
class MatchedMeasurement:
    uwb: UwbMeasurement
    pose_a: PoseSample
    pose_b: PoseSample
    relative_translation: np.ndarray
    relative_quaternion: np.ndarray
    relative_rotation: np.ndarray
    distance_gt_m: float
    range_error_m: float


@dataclass(frozen=True)
class StabilityMetrics:
    """选中测距块内 A、B 和相对位姿的 RMS 波动。"""

    pose_a_position_m: float
    pose_a_rotation_deg: float
    pose_b_position_m: float
    pose_b_rotation_deg: float
    relative_position_m: float
    relative_rotation_deg: float


@dataclass(frozen=True)
class SegmentReport:
    """一个标记静止段的处理结果，用于终端诊断。"""

    segment_index: int
    pose_id: str
    marker_pose_id: str
    available_count: int
    selected_count: int
    quality_ok: int
    reject_reason: str
    max_uwb_gap_sec: float
    stability: Optional[StabilityMetrics]


@dataclass
class BagResult:
    processed_rows: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    counters: Counter
    segment_reports: List[SegmentReport]
    marker_warnings: List[str]

#对动捕数据进行插值
class PoseSeries:
    """按时间排序的位姿序列，并提供位置线性插值和四元数 SLERP。"""

    def __init__(self, records: Sequence[PoseRecord], name: str) -> None:
        if not records:
            raise ProcessingError(f"{name} 没有有效动捕位姿。")

        ordered = sorted(records, key=lambda item: item.timestamp)
        deduplicated: List[PoseRecord] = []
        for record in ordered:
            if (
                deduplicated
                and abs(record.timestamp - deduplicated[-1].timestamp) <= 1e-9
            ):
                deduplicated[-1] = record
            else:
                deduplicated.append(record)

        if len(deduplicated) < 2:
            raise ProcessingError(f"{name} 有效时间戳少于2个，无法时间同步。")

        self.name = name
        self.times = np.asarray(
            [item.timestamp for item in deduplicated], dtype=np.float64
        )
        self.positions = np.stack(
            [item.position for item in deduplicated], axis=0
        ).astype(np.float64)
        self.quaternions = np.stack(
            [item.quaternion for item in deduplicated], axis=0
        ).astype(np.float64)

    def interpolate(
        self,
        timestamp: float,
        max_gap_sec: float,
    ) -> Optional[PoseSample]:
        """在 timestamp 处插值；离最近动捕帧过远时返回 None。"""
        index = int(np.searchsorted(self.times, timestamp, side="left"))

        if index == 0:
            dt = float(self.times[0] - timestamp)
            if abs(dt) > max_gap_sec:
                return None
            return PoseSample(
                self.positions[0].copy(),
                self.quaternions[0].copy(),
                abs(dt),
            )

        if index == len(self.times):
            dt = float(timestamp - self.times[-1])
            if abs(dt) > max_gap_sec:
                return None
            return PoseSample(
                self.positions[-1].copy(),
                self.quaternions[-1].copy(),
                abs(dt),
            )

        t0 = float(self.times[index - 1])
        t1 = float(self.times[index])
        dt0 = timestamp - t0
        dt1 = t1 - timestamp

        if abs(dt0) <= 1e-12:
            return PoseSample(
                self.positions[index - 1].copy(),
                self.quaternions[index - 1].copy(),
                0.0,
            )
        if abs(dt1) <= 1e-12:
            return PoseSample(
                self.positions[index].copy(),
                self.quaternions[index].copy(),
                0.0,
            )

        # 两侧动捕帧都必须足够靠近，避免跨越长时间数据缺口进行插值。
        if max(dt0, dt1) > max_gap_sec:
            return None

        interval = t1 - t0
        if interval <= 0.0:
            return None
        alpha = float(np.clip(dt0 / interval, 0.0, 1.0))
        position = (
            (1.0 - alpha) * self.positions[index - 1]
            + alpha * self.positions[index]
        )
        quaternion = quaternion_slerp(
            self.quaternions[index - 1],
            self.quaternions[index],
            alpha,
        )
        return PoseSample(
            position=position,
            quaternion=quaternion,
            nearest_dt_sec=float(min(dt0, dt1)),
        )

#归一化四元数
def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """归一化 ROS 顺序 [x, y, z, w] 四元数。"""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all():
        raise ValueError("四元数含 NaN 或 Inf。")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("四元数模长接近0。")
    q = q / norm
    # q 与 -q 表示相同旋转；固定符号便于复现实验和查看 CSV。
    if q[3] < 0.0:
        q = -q
    return q


def quaternion_conjugate(quaternion: Sequence[float]) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

#四元数转3×3旋转矩阵
def quaternion_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> np.ndarray:
    """ROS xyzw 顺序的 Hamilton 四元数乘法。"""
    x1, y1, z1, w1 = normalize_quaternion(left)
    x2, y2, z2, w2 = normalize_quaternion(right)
    result = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion(result)


def quaternion_slerp(
    quaternion_0: Sequence[float],
    quaternion_1: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """两个 xyzw 四元数之间的最短路径球面线性插值。"""
    q0 = normalize_quaternion(quaternion_0)
    q1 = normalize_quaternion(quaternion_1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    weight_0 = math.sin((1.0 - alpha) * theta) / sin_theta
    weight_1 = math.sin(alpha * theta) / sin_theta
    return normalize_quaternion(weight_0 * q0 + weight_1 * q1)


def quaternion_to_rotation_matrix(
    quaternion: Sequence[float],
) -> np.ndarray:
    """将 ROS xyzw 四元数转换为3×3旋转矩阵。"""
    x, y, z, w = normalize_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """使用 Markley 方法计算多个旋转的平均四元数。"""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("平均四元数输入必须为非空 N×4 数组。")
    normalized = np.stack(
        [normalize_quaternion(item) for item in values],
        axis=0,
    )
    accumulator = normalized.T @ normalized
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean = eigenvectors[:, int(np.argmax(eigenvalues))]
    return normalize_quaternion(mean)


def quaternion_angular_distance_deg(
    quaternion_a: Sequence[float],
    quaternion_b: Sequence[float],
) -> float:
    """两个旋转之间的最小夹角，单位为度。"""
    qa = normalize_quaternion(quaternion_a)
    qb = normalize_quaternion(quaternion_b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def message_timestamp(
    message: Any,
    bag_timestamp: float,
    source: str,
) -> Optional[float]:
    if source == "bag":
        return bag_timestamp

    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None or not hasattr(stamp, "to_sec"):
        return None
    value = float(stamp.to_sec())
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value

#从VRPN消息中提取位姿
def extract_pose_record(
    message: Any,
    bag_timestamp: float,
    time_source: str,
) -> Optional[PoseRecord]:
    timestamp = message_timestamp(message, bag_timestamp, time_source)
    if timestamp is None or not math.isfinite(timestamp):
        return None
    try:
        pose = message.pose
        position = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )
        quaternion = normalize_quaternion(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.isfinite(position).all():
        return None
    return PoseRecord(timestamp, position, quaternion)


def normalize_marker_pose_id(value: Any) -> Optional[str]:
    """把 JSON 中的 pose_id 转成稳定、非空的字符串。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return str(int(value))
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


def extract_collection_marker(
    message: Any,
    bag_timestamp: float,
    counters: Counter,
) -> Optional[CollectionMarker]:
    """解析 std_msgs/String JSON；时间只采用 rosbag 的记录时间。"""
    counters["marker_messages_total"] += 1
    if not math.isfinite(bag_timestamp):
        counters["marker_messages_invalid_timestamp"] += 1
        return None
    data = getattr(message, "data", None)
    if not isinstance(data, str):
        counters["marker_messages_invalid_data"] += 1
        return None

    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        counters["marker_messages_invalid_json"] += 1
        return None
    if not isinstance(payload, dict):
        counters["marker_messages_invalid_json"] += 1
        return None

    event = str(payload.get("event", "")).strip().upper()
    if event not in {"STATIC_START", "MOVE_START"}:
        counters["marker_messages_unknown_event"] += 1
        return None

    marker_pose_id = normalize_marker_pose_id(payload.get("pose_id"))
    if marker_pose_id is None:
        counters["marker_messages_invalid_pose_id"] += 1
        return None

    source_value = payload.get("source", "")
    source = str(source_value).strip() if source_value is not None else ""
    counters["marker_messages_valid"] += 1
    return CollectionMarker(
        timestamp=bag_timestamp,
        event=event,
        marker_pose_id=marker_pose_id,
        source=source,
    )


# 提取 UWB 测距
def extract_uwb_measurement(
    message: Any,
    bag_timestamp: float,
    local_id: int,
    remote_id: int,
    range_scale: float,
    min_range_m: float,
    max_range_m: Optional[float],
    counters: Counter,
) -> Optional[UwbMeasurement]:
    counters["uwb_frames_total"] += 1
    if not math.isfinite(bag_timestamp):
        counters["uwb_frames_invalid_timestamp"] += 1
        return None
    try:
        message_local_id = int(message.id)
    except (AttributeError, TypeError, ValueError):
        counters["uwb_frames_missing_local_id"] += 1
        return None

    if message_local_id != local_id:
        counters["uwb_frames_wrong_local_id"] += 1
        return None

    nodes = getattr(message, "nodes", None)
    if nodes is None:
        counters["uwb_frames_missing_nodes"] += 1
        return None

    matches: List[Any] = []
    for node in nodes:
        try:
            node_id = int(node.id)
        except (AttributeError, TypeError, ValueError):
            continue
        if node_id == remote_id:
            matches.append(node)

    if not matches:
        counters["uwb_frames_remote_not_found"] += 1
        return None
    if len(matches) > 1:
        counters["uwb_frames_duplicate_remote"] += 1

    node = matches[0]
    try:
        range_m = float(node.dis) * range_scale
    except (AttributeError, TypeError, ValueError):
        counters["uwb_frames_invalid_range"] += 1
        return None

    if (
        not math.isfinite(range_m)
        or range_m <= min_range_m
        or (max_range_m is not None and range_m > max_range_m)
    ):
        counters["uwb_frames_invalid_range"] += 1
        return None

    counters["uwb_measurements_valid_before_sync"] += 1
    return UwbMeasurement(
        timestamp=bag_timestamp,
        local_time=int(getattr(message, "local_time", 0)),
        system_time=int(getattr(message, "system_time", 0)),
        local_role=int(getattr(message, "role", -1)),
        local_id=message_local_id,
        remote_role=int(getattr(node, "role", -1)),
        remote_id=remote_id,
        range_m=range_m,
        fp_rssi=finite_or_nan(getattr(node, "fp_rssi", float("nan"))),
        rx_rssi=finite_or_nan(getattr(node, "rx_rssi", float("nan"))),
    )


def build_static_intervals(
    markers: Sequence[CollectionMarker],
    bag_end_time: float,
    counters: Counter,
) -> Tuple[List[StaticInterval], List[str]]:
    """按标记顺序构造不重叠静止段，并对异常标记给出可读警告。"""
    intervals: List[StaticInterval] = []
    warnings: List[str] = []
    active: Optional[CollectionMarker] = None

    for marker in sorted(markers, key=lambda item: item.timestamp):
        if marker.event == "STATIC_START":
            if active is not None:
                counters["marker_duplicate_static_start"] += 1
                warnings.append(
                    "在上一静止段尚未收到 MOVE_START 时又收到 "
                    f"STATIC_START：放弃 pose_id={active.marker_pose_id}，"
                    f"改从 pose_id={marker.marker_pose_id} 开始。"
                )
            active = marker
            continue

        if active is None:
            counters["marker_orphan_move_start"] += 1
            warnings.append(
                "收到没有对应 STATIC_START 的 MOVE_START："
                f"pose_id={marker.marker_pose_id}。"
            )
            continue

        if marker.marker_pose_id != active.marker_pose_id:
            counters["marker_pose_id_mismatch"] += 1
            warnings.append(
                "STATIC_START 与 MOVE_START 的 pose_id 不一致："
                f"start={active.marker_pose_id}，move={marker.marker_pose_id}；"
                "该静止段已放弃。"
            )
            active = None
            continue

        if marker.timestamp <= active.timestamp:
            counters["marker_nonpositive_interval"] += 1
            warnings.append(
                f"pose_id={active.marker_pose_id} 的静止段时长不大于0，已放弃。"
            )
            active = None
            continue

        intervals.append(
            StaticInterval(
                segment_index=len(intervals) + 1,
                marker_pose_id=active.marker_pose_id,
                start_timestamp=active.timestamp,
                end_timestamp=marker.timestamp,
                closed_by="MOVE_START",
            )
        )
        active = None

    if active is not None:
        if bag_end_time > active.timestamp:
            intervals.append(
                StaticInterval(
                    segment_index=len(intervals) + 1,
                    marker_pose_id=active.marker_pose_id,
                    start_timestamp=active.timestamp,
                    end_timestamp=bag_end_time,
                    closed_by="BAG_END",
                )
            )
            counters["marker_intervals_closed_by_bag_end"] += 1
        else:
            counters["marker_nonpositive_interval"] += 1
            warnings.append(
                f"pose_id={active.marker_pose_id} 的 STATIC_START 位于 bag 末尾，"
                "没有可处理的数据。"
            )

    counters["static_intervals_built"] = len(intervals)
    return intervals, warnings


def marker_pose_token(marker_pose_id: str) -> str:
    """生成适合放入 CSV pose_id 的短标识。"""
    try:
        numeric = int(marker_pose_id)
    except ValueError:
        numeric = -1
    if numeric >= 0 and str(numeric) == marker_pose_id:
        return f"{numeric:04d}"

    safe = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in marker_pose_id
    ).strip("_")
    return safe or "unknown"


def unique_segment_pose_id(
    base_pose_id: str,
    marker_pose_id: str,
    occurrences: Dict[str, int],
) -> str:
    """组合 bag 名和标记编号；重复编号自动增加 repeat 后缀。"""
    occurrences[marker_pose_id] = occurrences.get(marker_pose_id, 0) + 1
    base = f"{base_pose_id}_pose_{marker_pose_token(marker_pose_id)}"
    occurrence = occurrences[marker_pose_id]
    if occurrence == 1:
        return base
    return f"{base}_repeat_{occurrence:02d}"


def relative_pose(
    pose_a: PoseSample,
    pose_b: PoseSample,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    由 T_M_A、T_M_B 计算 T_A_B。

    返回：
    - t_A_B
    - q_A_B（xyzw）
    - R_A_B
    - 天线中心真实距离
    """
    rotation_m_a = quaternion_to_rotation_matrix(pose_a.quaternion)
    translation_a_b = rotation_m_a.T @ (
        pose_b.position - pose_a.position
    )
    quaternion_a_b = quaternion_multiply(
        quaternion_conjugate(pose_a.quaternion),
        pose_b.quaternion,
    )
    rotation_a_b = quaternion_to_rotation_matrix(quaternion_a_b)
    distance_gt_m = float(np.linalg.norm(pose_b.position - pose_a.position))
    return translation_a_b, quaternion_a_b, rotation_a_b, distance_gt_m


def make_raw_row(
    match: MatchedMeasurement,
    session_id: str,
    bag_file: str,
    bag_start_time: float,
    annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    uwb = match.uwb
    row: Dict[str, Any] = {
        "session_id": session_id,
        "pose_id": annotation.get("pose_id", ""),
        "marker_pose_id": annotation.get("marker_pose_id", ""),
        "segment_index": annotation.get("segment_index", ""),
        "bag_file": bag_file,
        "segment_start_sec": annotation.get("segment_start_sec", ""),
        "segment_end_sec": annotation.get("segment_end_sec", ""),
        "segment_duration_sec": annotation.get("segment_duration_sec", ""),
        "segment_closed_by": annotation.get("segment_closed_by", ""),
        "available_range_count": annotation.get(
            "available_range_count", ""
        ),
        "selected_max_uwb_gap_sec": annotation.get(
            "selected_max_uwb_gap_sec", ""
        ),
        "in_marked_static_segment": annotation.get(
            "in_marked_static_segment", 0
        ),
        "selected_middle_block": annotation.get(
            "selected_middle_block", 0
        ),
        "used_for_label": annotation.get("used_for_label", 0),
        "segment_quality_ok": annotation.get("segment_quality_ok", ""),
        "segment_reject_reason": annotation.get(
            "segment_reject_reason", ""
        ),
        "uwb_timestamp": uwb.timestamp,
        "time_from_bag_start_sec": uwb.timestamp - bag_start_time,
        "uwb_local_time": uwb.local_time,
        "uwb_system_time": uwb.system_time,
        "local_role": uwb.local_role,
        "local_id": uwb.local_id,
        "remote_role": uwb.remote_role,
        "remote_id": uwb.remote_id,
        "range_m": uwb.range_m,
        "fp_rssi": uwb.fp_rssi,
        "rx_rssi": uwb.rx_rssi,
        "rssi_diff": uwb.rx_rssi - uwb.fp_rssi,
        "A_px": match.pose_a.position[0],
        "A_py": match.pose_a.position[1],
        "A_pz": match.pose_a.position[2],
        "A_qx": match.pose_a.quaternion[0],
        "A_qy": match.pose_a.quaternion[1],
        "A_qz": match.pose_a.quaternion[2],
        "A_qw": match.pose_a.quaternion[3],
        "B_px": match.pose_b.position[0],
        "B_py": match.pose_b.position[1],
        "B_pz": match.pose_b.position[2],
        "B_qx": match.pose_b.quaternion[0],
        "B_qy": match.pose_b.quaternion[1],
        "B_qz": match.pose_b.quaternion[2],
        "B_qw": match.pose_b.quaternion[3],
        "rel_tx": match.relative_translation[0],
        "rel_ty": match.relative_translation[1],
        "rel_tz": match.relative_translation[2],
        "distance_gt_m": match.distance_gt_m,
        "range_error_m": match.range_error_m,
        "pose_a_nearest_dt_sec": match.pose_a.nearest_dt_sec,
        "pose_b_nearest_dt_sec": match.pose_b.nearest_dt_sec,
    }
    for row_index in range(3):
        for column_index in range(3):
            row[f"rel_R{row_index}{column_index}"] = match.relative_rotation[
                row_index, column_index
            ]
    return row


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def pose_stability_rms(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> Tuple[float, float]:
    """计算一组位姿相对其均值的位置和旋转 RMS 波动。"""
    position_values = np.asarray(positions, dtype=np.float64)
    quaternion_values = np.asarray(quaternions, dtype=np.float64)
    position_mean = position_values.mean(axis=0)
    position_rms = float(
        np.sqrt(
            np.mean(
                np.sum((position_values - position_mean) ** 2, axis=1)
            )
        )
    )

    quaternion_mean = average_quaternions(quaternion_values)
    angles = np.asarray(
        [
            quaternion_angular_distance_deg(item, quaternion_mean)
            for item in quaternion_values
        ],
        dtype=np.float64,
    )
    rotation_rms = float(np.sqrt(np.mean(angles**2)))
    return position_rms, rotation_rms


def calculate_stability(
    matches: Sequence[MatchedMeasurement],
) -> StabilityMetrics:
    """同时检查 A、B 的动捕位姿和 T_A_B 相对位姿。"""
    if not matches:
        raise ValueError("稳定性计算需要至少一条对齐测距。")

    pose_a_position, pose_a_rotation = pose_stability_rms(
        np.stack([item.pose_a.position for item in matches], axis=0),
        np.stack([item.pose_a.quaternion for item in matches], axis=0),
    )
    pose_b_position, pose_b_rotation = pose_stability_rms(
        np.stack([item.pose_b.position for item in matches], axis=0),
        np.stack([item.pose_b.quaternion for item in matches], axis=0),
    )
    relative_position, relative_rotation = pose_stability_rms(
        np.stack([item.relative_translation for item in matches], axis=0),
        np.stack([item.relative_quaternion for item in matches], axis=0),
    )
    return StabilityMetrics(
        pose_a_position_m=pose_a_position,
        pose_a_rotation_deg=pose_a_rotation,
        pose_b_position_m=pose_b_position,
        pose_b_rotation_deg=pose_b_rotation,
        relative_position_m=relative_position,
        relative_rotation_deg=relative_rotation,
    )


def stability_rejection_reasons(
    stability: StabilityMetrics,
    max_position_stability_m: float,
    max_rotation_stability_deg: float,
) -> List[str]:
    """返回所有超限原因；空列表表示三组位姿均通过检查。"""
    reasons: List[str] = []
    position_values = {
        "A": stability.pose_a_position_m,
        "B": stability.pose_b_position_m,
        "relative": stability.relative_position_m,
    }
    bad_positions = {
        name: value
        for name, value in position_values.items()
        if value > max_position_stability_m
    }
    if bad_positions:
        details = ",".join(
            f"{name}={value:.6f}" for name, value in bad_positions.items()
        )
        reasons.append(
            f"position_rms_exceeded({details};"
            f"limit={max_position_stability_m:.6f}m)"
        )

    rotation_values = {
        "A": stability.pose_a_rotation_deg,
        "B": stability.pose_b_rotation_deg,
        "relative": stability.relative_rotation_deg,
    }
    bad_rotations = {
        name: value
        for name, value in rotation_values.items()
        if value > max_rotation_stability_deg
    }
    if bad_rotations:
        details = ",".join(
            f"{name}={value:.6f}" for name, value in bad_rotations.items()
        )
        reasons.append(
            f"rotation_rms_exceeded({details};"
            f"limit={max_rotation_stability_deg:.6f}deg)"
        )
    return reasons


def summarize_matches(
    matches: Sequence[MatchedMeasurement],
    session_id: str,
    pose_id: str,
    bag_file: str,
    bag_start_time: float,
    interval: StaticInterval,
    available_count: int,
    stability: StabilityMetrics,
    max_position_stability_m: float,
    max_rotation_stability_deg: float,
) -> Dict[str, Any]:
    if len(matches) < 2:
        raise ProcessingError("有效对齐测距少于2条，无法计算 bias_std。")

    translations = np.stack(
        [item.relative_translation for item in matches], axis=0
    )
    quaternions = np.stack(
        [item.relative_quaternion for item in matches], axis=0
    )
    distances = np.asarray(
        [item.distance_gt_m for item in matches], dtype=np.float64
    )
    ranges = np.asarray(
        [item.uwb.range_m for item in matches], dtype=np.float64
    )
    errors = np.asarray(
        [item.range_error_m for item in matches], dtype=np.float64
    )
    timestamps = np.asarray(
        [item.uwb.timestamp for item in matches], dtype=np.float64
    )
    sync_a = np.asarray(
        [item.pose_a.nearest_dt_sec for item in matches], dtype=np.float64
    )
    sync_b = np.asarray(
        [item.pose_b.nearest_dt_sec for item in matches], dtype=np.float64
    )

    translation_mean = translations.mean(axis=0)
    quaternion_mean = average_quaternions(quaternions)
    rotation_mean = quaternion_to_rotation_matrix(quaternion_mean)

    distance_gt_mean = float(distances.mean())
    range_mean = float(ranges.mean())
    # 明确采用与训练脚本完全一致的标签符号。
    bias_mean = range_mean - distance_gt_mean
    bias_std = float(np.std(errors, ddof=1))
    measurement_span = float(timestamps.max() - timestamps.min())
    measurement_rate = (
        float((len(matches) - 1) / measurement_span)
        if measurement_span > 0.0
        else float("nan")
    )
    timestamp_differences = np.diff(timestamps)
    max_uwb_gap_sec = (
        float(np.max(timestamp_differences))
        if len(timestamp_differences)
        else float("nan")
    )

    fp_rssi_mean = finite_mean(item.uwb.fp_rssi for item in matches)
    rx_rssi_mean = finite_mean(item.uwb.rx_rssi for item in matches)
    rssi_diff_mean = finite_mean(
        item.uwb.rx_rssi - item.uwb.fp_rssi for item in matches
    )

    quality_ok = int(
        not stability_rejection_reasons(
            stability,
            max_position_stability_m,
            max_rotation_stability_deg,
        )
    )

    row: Dict[str, Any] = {
        "session_id": session_id,
        "pose_id": pose_id,
        "marker_pose_id": interval.marker_pose_id,
        "segment_index": interval.segment_index,
        "bag_file": bag_file,
        "segment_start_sec": interval.start_timestamp - bag_start_time,
        "segment_end_sec": interval.end_timestamp - bag_start_time,
        "segment_duration_sec": (
            interval.end_timestamp - interval.start_timestamp
        ),
        "segment_closed_by": interval.closed_by,
        "selected_start_sec": timestamps.min() - bag_start_time,
        "selected_end_sec": timestamps.max() - bag_start_time,
        "available_range_count": available_count,
        "rel_tx": translation_mean[0],
        "rel_ty": translation_mean[1],
        "rel_tz": translation_mean[2],
        "rel_qx": quaternion_mean[0],
        "rel_qy": quaternion_mean[1],
        "rel_qz": quaternion_mean[2],
        "rel_qw": quaternion_mean[3],
        "distance_gt_mean": distance_gt_mean,
        "distance_gt_std": float(np.std(distances, ddof=1)),
        "range_mean": range_mean,
        "range_std": float(np.std(ranges, ddof=1)),
        "bias_mean": bias_mean,
        "bias_std": bias_std,
        "range_count": len(matches),
        # 保留原字段名，继续表示相对位姿稳定程度。
        "position_stability": stability.relative_position_m,
        "rotation_stability": stability.relative_rotation_deg,
        "pose_a_position_stability": stability.pose_a_position_m,
        "pose_a_rotation_stability": stability.pose_a_rotation_deg,
        "pose_b_position_stability": stability.pose_b_position_m,
        "pose_b_rotation_stability": stability.pose_b_rotation_deg,
        "fp_rssi_mean": fp_rssi_mean,
        "rx_rssi_mean": rx_rssi_mean,
        "rssi_diff_mean": rssi_diff_mean,
        "measurement_span_sec": measurement_span,
        "measurement_rate_hz": measurement_rate,
        "max_uwb_gap_sec": max_uwb_gap_sec,
        "pose_a_sync_mean_sec": float(sync_a.mean()),
        "pose_a_sync_max_sec": float(sync_a.max()),
        "pose_b_sync_mean_sec": float(sync_b.mean()),
        "pose_b_sync_max_sec": float(sync_b.max()),
        "quality_ok": quality_ok,
    }
    for row_index in range(3):
        for column_index in range(3):
            row[f"rel_R{row_index}{column_index}"] = rotation_mean[
                row_index, column_index
            ]

    # 防止未来修改造成标签定义与逐次误差统计不一致。
    direct_error_mean = float(errors.mean())
    if not math.isclose(bias_mean, direct_error_mean, abs_tol=1e-10):
        raise ProcessingError(
            "内部检查失败：range_mean-distance_gt_mean 与 "
            "mean(range-distance) 不一致。"
        )
    return row


def load_rosbag_module() -> Any:
    try:
        import rosbag  # type: ignore
    except ModuleNotFoundError as exc:
        raise ProcessingError(
            "找不到 ROS1 的 rosbag Python 模块。请使用系统 Python 3，先执行：\n"
            "  source /opt/ros/noetic/setup.bash\n"
            "  source <你的catkin工作空间>/devel/setup.bash\n"
            "然后重新运行本脚本。"
        ) from exc
    return rosbag

# 核心处理流程
def process_bag(
    bag_path: Path,
    args: argparse.Namespace,
    rosbag_module: Any,
    session_id: str,
    base_pose_id: str,
) -> BagResult:
    counters: Counter = Counter()
    pose_a_records: List[PoseRecord] = []
    pose_b_records: List[PoseRecord] = []
    uwb_measurements: List[UwbMeasurement] = []
    markers: List[CollectionMarker] = []

    try:
        bag = rosbag_module.Bag(str(bag_path), "r")
    except Exception as exc:
        raise ProcessingError(f"无法打开 rosbag：{bag_path}\n{exc}") from exc

    try:
        bag_start_time = float(bag.get_start_time())
        bag_end_time = float(bag.get_end_time())
        if (
            not math.isfinite(bag_start_time)
            or not math.isfinite(bag_end_time)
            or bag_end_time <= bag_start_time
        ):
            raise ProcessingError(
                f"{bag_path.name} 的起止时间无效："
                f"start={bag_start_time}, end={bag_end_time}。"
            )
        trimmed_start = bag_start_time + args.trim_start_sec
        trimmed_end = bag_end_time - args.trim_end_sec
        if trimmed_end <= trimmed_start:
            raise ProcessingError(
                f"{bag_path.name} 裁剪后的时间段为空："
                f"bag时长={bag_end_time - bag_start_time:.3f}s，"
                f"trim_start={args.trim_start_sec:.3f}s，"
                f"trim_end={args.trim_end_sec:.3f}s。"
            )

        topics = [
            args.pose_a_topic,
            args.pose_b_topic,
            args.uwb_topic,
            args.marker_topic,
        ]
        for topic, message, bag_time_object in bag.read_messages(topics=topics):
            bag_timestamp = float(bag_time_object.to_sec())
            if topic == args.pose_a_topic:
                record = extract_pose_record(
                    message,
                    bag_timestamp,
                    args.pose_time_source,
                )
                if record is None:
                    counters["pose_a_invalid"] += 1
                else:
                    pose_a_records.append(record)
            elif topic == args.pose_b_topic:
                record = extract_pose_record(
                    message,
                    bag_timestamp,
                    args.pose_time_source,
                )
                if record is None:
                    counters["pose_b_invalid"] += 1
                else:
                    pose_b_records.append(record)
            elif topic == args.uwb_topic:
                if not (trimmed_start <= bag_timestamp <= trimmed_end):
                    counters["uwb_frames_trimmed"] += 1
                    continue
                measurement = extract_uwb_measurement(
                    message=message,
                    bag_timestamp=bag_timestamp,
                    local_id=args.local_id,
                    remote_id=args.remote_id,
                    range_scale=args.range_scale,
                    min_range_m=args.min_range_m,
                    max_range_m=args.max_range_m,
                    counters=counters,
                )
                if measurement is not None:
                    uwb_measurements.append(measurement)
            elif topic == args.marker_topic:
                marker = extract_collection_marker(
                    message=message,
                    bag_timestamp=bag_timestamp,
                    counters=counters,
                )
                if marker is not None:
                    markers.append(marker)
    except ProcessingError:
        raise
    except Exception as exc:
        raise ProcessingError(f"读取 {bag_path.name} 时出错：{exc}") from exc
    finally:
        bag.close()

    if not pose_a_records:
        raise ProcessingError(
            f"{bag_path.name} 中没有有效的 {args.pose_a_topic} 数据。"
        )
    if not pose_b_records:
        raise ProcessingError(
            f"{bag_path.name} 中没有有效的 {args.pose_b_topic} 数据。"
        )
    if not uwb_measurements:
        details = ", ".join(
            f"{key}={value}" for key, value in sorted(counters.items())
        )
        raise ProcessingError(
            f"{bag_path.name} 中没有符合 A={args.local_id}、"
            f"B={args.remote_id} 的有效测距。统计：{details}"
        )
    if not markers:
        details = ", ".join(
            f"{key}={value}"
            for key, value in sorted(counters.items())
            if key.startswith("marker_")
        )
        raise ProcessingError(
            f"{bag_path.name} 中没有可解析的 {args.marker_topic} 标记。"
            "请确认该话题已录入 bag，消息类型为 std_msgs/String，"
            f"data 为约定 JSON。统计：{details or '无标记消息'}"
        )

    intervals, marker_warnings = build_static_intervals(
        markers=markers,
        bag_end_time=bag_end_time,
        counters=counters,
    )
    if not intervals:
        raise ProcessingError(
            f"{bag_path.name} 没有形成任何有效静止段。"
            "请检查 STATIC_START/MOVE_START 的顺序和 pose_id。"
        )

    pose_a_series = PoseSeries(pose_a_records, args.pose_a_topic)
    pose_b_series = PoseSeries(pose_b_records, args.pose_b_topic)
    matches: List[MatchedMeasurement] = []

    for measurement in uwb_measurements:
        pose_a = pose_a_series.interpolate(
            measurement.timestamp,
            args.max_pose_gap_sec,
        )
        if pose_a is None:
            counters["uwb_sync_failed_pose_a"] += 1
            continue
        pose_b = pose_b_series.interpolate(
            measurement.timestamp,
            args.max_pose_gap_sec,
        )
        if pose_b is None:
            counters["uwb_sync_failed_pose_b"] += 1
            continue

        translation, quaternion, rotation, distance_gt_m = relative_pose(
            pose_a,
            pose_b,
        )
        range_error_m = measurement.range_m - distance_gt_m
        matches.append(
            MatchedMeasurement(
                uwb=measurement,
                pose_a=pose_a,
                pose_b=pose_b,
                relative_translation=translation,
                relative_quaternion=quaternion,
                relative_rotation=rotation,
                distance_gt_m=distance_gt_m,
                range_error_m=range_error_m,
            )
        )

    matches.sort(key=lambda item: item.uwb.timestamp)
    counters["uwb_measurements_matched"] = len(matches)
    if not matches:
        raise ProcessingError(
            f"{bag_path.name} 没有成功与 A、B 动捕同时对齐的 UWB 测距。"
            "请检查时间戳和 --max-pose-gap-sec。"
        )

    processed_rows: List[Dict[str, Any]] = []
    segment_reports: List[SegmentReport] = []
    annotations: List[Dict[str, Any]] = [{} for _ in matches]
    match_times = np.asarray(
        [item.uwb.timestamp for item in matches],
        dtype=np.float64,
    )
    pose_id_occurrences: Dict[str, int] = {}

    for interval in intervals:
        segment_pose_id = unique_segment_pose_id(
            base_pose_id,
            interval.marker_pose_id,
            pose_id_occurrences,
        )
        first_index = int(
            np.searchsorted(
                match_times,
                interval.start_timestamp,
                side="left",
            )
        )
        past_last_index = int(
            np.searchsorted(
                match_times,
                interval.end_timestamp,
                side="right",
            )
        )
        segment_indices = list(range(first_index, past_last_index))
        available_count = len(segment_indices)

        base_annotation: Dict[str, Any] = {
            "pose_id": segment_pose_id,
            "marker_pose_id": interval.marker_pose_id,
            "segment_index": interval.segment_index,
            "segment_start_sec": (
                interval.start_timestamp - bag_start_time
            ),
            "segment_end_sec": interval.end_timestamp - bag_start_time,
            "segment_duration_sec": (
                interval.end_timestamp - interval.start_timestamp
            ),
            "segment_closed_by": interval.closed_by,
            "available_range_count": available_count,
            "in_marked_static_segment": 1,
            "selected_middle_block": 0,
            "used_for_label": 0,
        }
        for match_index in segment_indices:
            annotations[match_index].update(base_annotation)

        selected_indices: List[int] = []
        stability: Optional[StabilityMetrics] = None
        max_uwb_gap_sec = float("nan")
        rejection_reasons: List[str] = []

        if available_count < args.samples_per_pose:
            rejection_reasons.append(
                "insufficient_aligned_measurements("
                f"available={available_count},"
                f"required={args.samples_per_pose})"
            )
            counters["segments_rejected_insufficient_samples"] += 1
        else:
            middle_offset = (
                available_count - args.samples_per_pose
            ) // 2
            selected_indices = segment_indices[
                middle_offset : middle_offset + args.samples_per_pose
            ]
            selected_matches = [matches[index] for index in selected_indices]
            selected_times = np.asarray(
                [item.uwb.timestamp for item in selected_matches],
                dtype=np.float64,
            )
            time_differences = np.diff(selected_times)
            if np.any(time_differences <= 0.0):
                rejection_reasons.append(
                    "non_increasing_uwb_timestamps"
                )
                counters["segments_rejected_timestamp_order"] += 1
            if len(time_differences):
                max_uwb_gap_sec = float(np.max(time_differences))
            if (
                math.isfinite(max_uwb_gap_sec)
                and max_uwb_gap_sec > args.max_uwb_gap_sec
            ):
                rejection_reasons.append(
                    "max_uwb_gap_exceeded("
                    f"value={max_uwb_gap_sec:.6f},"
                    f"limit={args.max_uwb_gap_sec:.6f})"
                )
                counters["segments_rejected_uwb_gap"] += 1

            stability = calculate_stability(selected_matches)
            stability_reasons = stability_rejection_reasons(
                stability,
                args.max_position_stability_m,
                args.max_rotation_stability_deg,
            )
            if stability_reasons:
                counters["segments_rejected_stability"] += 1
                rejection_reasons.extend(stability_reasons)

            for match_index in selected_indices:
                annotations[match_index]["selected_middle_block"] = 1

        quality_ok = int(not rejection_reasons)
        reject_reason = " | ".join(rejection_reasons)
        for match_index in segment_indices:
            annotations[match_index]["segment_quality_ok"] = quality_ok
            annotations[match_index]["segment_reject_reason"] = reject_reason
            annotations[match_index]["selected_max_uwb_gap_sec"] = (
                max_uwb_gap_sec
                if math.isfinite(max_uwb_gap_sec)
                else ""
            )

        if quality_ok:
            selected_matches = [matches[index] for index in selected_indices]
            if stability is None:
                raise ProcessingError(
                    "内部检查失败：合格静止段缺少稳定性统计。"
                )
            processed_row = summarize_matches(
                matches=selected_matches,
                session_id=session_id,
                pose_id=segment_pose_id,
                bag_file=bag_path.name,
                bag_start_time=bag_start_time,
                interval=interval,
                available_count=available_count,
                stability=stability,
                max_position_stability_m=args.max_position_stability_m,
                max_rotation_stability_deg=args.max_rotation_stability_deg,
            )
            if int(processed_row["quality_ok"]) != 1:
                raise ProcessingError(
                    "内部检查失败：已接受静止段的 quality_ok 不为1。"
                )
            processed_rows.append(processed_row)
            for match_index in selected_indices:
                annotations[match_index]["used_for_label"] = 1
            counters["segments_accepted"] += 1
        else:
            counters["segments_rejected"] += 1

        segment_reports.append(
            SegmentReport(
                segment_index=interval.segment_index,
                pose_id=segment_pose_id,
                marker_pose_id=interval.marker_pose_id,
                available_count=available_count,
                selected_count=len(selected_indices),
                quality_ok=quality_ok,
                reject_reason=reject_reason,
                max_uwb_gap_sec=max_uwb_gap_sec,
                stability=stability,
            )
        )

    counters["segments_total"] = len(intervals)
    raw_rows = [
        make_raw_row(
            match=item,
            session_id=session_id,
            bag_file=bag_path.name,
            bag_start_time=bag_start_time,
            annotation=annotations[index],
        )
        for index, item in enumerate(matches)
    ]
    return BagResult(
        processed_rows=processed_rows,
        raw_rows=raw_rows,
        counters=counters,
        segment_reports=segment_reports,
        marker_warnings=marker_warnings,
    )


def discover_bags(inputs: Sequence[str], recursive: bool) -> List[Path]:
    discovered: List[Path] = []
    for text in inputs:
        path = Path(text).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".bag":
                raise ProcessingError(f"输入文件不是 .bag：{path}")
            discovered.append(path.resolve())
            continue
        if path.is_dir():
            pattern = "**/*.bag" if recursive else "*.bag"
            discovered.extend(item.resolve() for item in path.glob(pattern))
            continue
        raise ProcessingError(f"找不到输入路径：{path}")

    unique = sorted(set(discovered), key=lambda item: item.as_posix())
    if not unique:
        scope = "及其子文件夹" if recursive else "当前文件夹"
        raise ProcessingError(f"输入路径的{scope}没有找到 .bag 文件。")
    return unique


def identifiers_for_bag(
    bag_path: Path,
    args: argparse.Namespace,
    total_bags: int,
) -> Tuple[str, str]:
    if args.session_id:
        session_id = args.session_id
    elif args.session_from == "bag":
        session_id = bag_path.stem
    else:
        session_id = bag_path.parent.name or "session"

    if args.pose_id:
        if total_bags != 1:
            raise ProcessingError("--pose-id 只能在输入恰好一个 bag 时使用。")
        pose_id = args.pose_id
    else:
        pose_id = bag_path.stem
    return session_id, pose_id


def native_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv_atomic(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ProcessingError(f"没有可写入 {output_path} 的数据。")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=str(output_path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: native_csv_value(row.get(key, ""))
                        for key in fieldnames
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把含多个标记静止位姿的连续 ROS1 bag 转换为 UWB 偏置训练 CSV"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="一个或多个 .bag 文件，或包含 .bag 的文件夹",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="输入文件夹时递归搜索子文件夹",
    )
    parser.add_argument(
        "--processed-output",
        default="processed_bias.csv",
        help="每个静态位姿一行的训练 CSV",
    )
    parser.add_argument(
        "--raw-output",
        default="raw_aligned.csv",
        help="每条对齐测距一行的检查 CSV",
    )
    parser.add_argument(
        "--skip-raw-output",
        action="store_true",
        help="不生成逐次测距 raw CSV",
    )
    parser.add_argument(
        "--skip-bad-bags",
        action="store_true",
        help="某个 bag 无效时跳过并继续处理其他 bag",
    )

    parser.add_argument(
        "--pose-a-topic",
        default="/vrpn_client/UWB_A/pose",
    )
    parser.add_argument(
        "--pose-b-topic",
        default="/vrpn_client/UWB_B/pose",
    )
    parser.add_argument(
        "--uwb-topic",
        default="/nlink_linktrack_nodeframe2",
    )
    parser.add_argument(
        "--marker-topic",
        default="/uwb_bias/collection_marker",
        help="std_msgs/String JSON 采集标记话题",
    )
    parser.add_argument("--local-id", type=int, default=1)
    parser.add_argument("--remote-id", type=int, default=2)
    parser.add_argument(
        "--range-scale",
        type=float,
        default=1.0,
        help="nodes[].dis 乘以该系数后转换为米；当前设备已经是米",
    )
    parser.add_argument(
        "--min-range-m",
        type=float,
        default=0.01,
        help="删除不大于该值的测距",
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=None,
        help="可选的最大有效测距；默认不设置上限",
    )

    parser.add_argument(
        "--pose-time-source",
        choices=["bag", "header"],
        default="bag",
        help=(
            "动捕时间来源；UWB消息没有header，始终使用bag记录时间，"
            "因此默认动捕也使用bag时间"
        ),
    )
    parser.add_argument(
        "--max-pose-gap-sec",
        type=float,
        default=0.05,
        help="UWB时刻到两侧动捕帧允许的最大时间间隔",
    )
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=0.0,
        help="忽略每个bag开头的若干秒UWB数据",
    )
    parser.add_argument(
        "--trim-end-sec",
        type=float,
        default=0.0,
        help="忽略每个bag结尾的若干秒UWB数据",
    )
    parser.add_argument(
        "--samples-per-pose",
        "--min-samples",
        dest="samples_per_pose",
        type=int,
        default=50,
        help=(
            "每个静止段从中间选取的有效对齐测距数量；"
            "--min-samples 是兼容旧脚本的别名"
        ),
    )
    parser.add_argument(
        "--max-uwb-gap-sec",
        type=float,
        default=0.1,
        help="选中测距块内允许的最大相邻UWB时间间隔",
    )

    parser.add_argument(
        "--session-id",
        default=None,
        help="指定所有输入bag的session_id；省略时按 --session-from 生成",
    )
    parser.add_argument(
        "--session-from",
        choices=["parent", "bag"],
        default="parent",
        help="未指定 --session-id 时，从父文件夹名或bag名生成session_id",
    )
    parser.add_argument(
        "--pose-id",
        default=None,
        help="单个bag时手工指定pose_id前缀；默认使用bag文件名",
    )
    parser.add_argument(
        "--max-position-stability-m",
        type=float,
        default=0.10,
        help="A、B及相对位置RMS波动阈值；超限静止段不进入训练CSV",
    )
    parser.add_argument(
        "--max-rotation-stability-deg",
        type=float,
        default=2.0,
        help="A、B及相对旋转RMS波动阈值；超限静止段不进入训练CSV",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    topics = [
        args.pose_a_topic,
        args.pose_b_topic,
        args.uwb_topic,
        args.marker_topic,
    ]
    if any(not isinstance(topic, str) or not topic.strip() for topic in topics):
        raise ProcessingError("所有话题名称都必须是非空字符串。")
    if len(set(topics)) != len(topics):
        raise ProcessingError("动捕、UWB和标记话题名称不能重复。")
    if not (0 <= args.local_id <= 255):
        raise ProcessingError("--local-id 必须位于0到255。")
    if not (0 <= args.remote_id <= 255):
        raise ProcessingError("--remote-id 必须位于0到255。")
    if args.local_id == args.remote_id:
        raise ProcessingError("--local-id 与 --remote-id 不能相同。")
    if not math.isfinite(args.range_scale) or args.range_scale <= 0.0:
        raise ProcessingError("--range-scale 必须大于0。")
    if not math.isfinite(args.min_range_m) or args.min_range_m < 0.0:
        raise ProcessingError("--min-range-m 不能小于0。")
    if args.max_range_m is not None:
        if (
            not math.isfinite(args.max_range_m)
            or args.max_range_m <= args.min_range_m
        ):
            raise ProcessingError("--max-range-m 必须大于 --min-range-m。")
    if (
        not math.isfinite(args.max_pose_gap_sec)
        or args.max_pose_gap_sec <= 0.0
    ):
        raise ProcessingError("--max-pose-gap-sec 必须大于0。")
    if (
        not math.isfinite(args.trim_start_sec)
        or not math.isfinite(args.trim_end_sec)
        or args.trim_start_sec < 0.0
        or args.trim_end_sec < 0.0
    ):
        raise ProcessingError("裁剪时间不能小于0。")
    if args.samples_per_pose < 2:
        raise ProcessingError("--samples-per-pose 至少为2。")
    if (
        not math.isfinite(args.max_uwb_gap_sec)
        or args.max_uwb_gap_sec <= 0.0
    ):
        raise ProcessingError("--max-uwb-gap-sec 必须大于0。")
    if (
        not math.isfinite(args.max_position_stability_m)
        or args.max_position_stability_m < 0.0
    ):
        raise ProcessingError("--max-position-stability-m 不能小于0。")
    if (
        not math.isfinite(args.max_rotation_stability_deg)
        or args.max_rotation_stability_deg < 0.0
    ):
        raise ProcessingError("--max-rotation-stability-deg 不能小于0。")

    processed = Path(args.processed_output).expanduser().resolve()
    if not args.skip_raw_output:
        raw = Path(args.raw_output).expanduser().resolve()
        if processed == raw:
            raise ProcessingError(
                "--processed-output 与 --raw-output 不能是同一个文件。"
            )


def format_counter_summary(counters: Counter) -> str:
    selected = [
        "marker_messages_total",
        "marker_messages_valid",
        "static_intervals_built",
        "uwb_frames_total",
        "uwb_frames_trimmed",
        "uwb_frames_wrong_local_id",
        "uwb_frames_remote_not_found",
        "uwb_frames_invalid_range",
        "uwb_measurements_valid_before_sync",
        "uwb_sync_failed_pose_a",
        "uwb_sync_failed_pose_b",
        "uwb_measurements_matched",
        "segments_accepted",
        "segments_rejected",
    ]
    extras = sorted(
        key
        for key, value in counters.items()
        if value
        and key not in selected
        and (
            key.startswith("marker_")
            or key.startswith("segments_rejected_")
        )
    )
    return ", ".join(
        f"{key}={int(counters[key])}" for key in selected + extras
    )


def format_stability_metrics(stability: StabilityMetrics) -> str:
    return (
        "position_rms_m("
        f"A={stability.pose_a_position_m:.6f},"
        f"B={stability.pose_b_position_m:.6f},"
        f"relative={stability.relative_position_m:.6f}), "
        "rotation_rms_deg("
        f"A={stability.pose_a_rotation_deg:.4f},"
        f"B={stability.pose_b_rotation_deg:.4f},"
        f"relative={stability.relative_rotation_deg:.4f})"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
        bag_paths = discover_bags(args.inputs, args.recursive)
        if args.pose_id and len(bag_paths) != 1:
            raise ProcessingError("--pose-id 只能用于单个bag。")

        rosbag_module = load_rosbag_module()
        processed_rows: List[Dict[str, Any]] = []
        raw_rows: List[Dict[str, Any]] = []
        failed_bags: List[Tuple[Path, str]] = []
        total_rejected_segments = 0

        print(f"发现 {len(bag_paths)} 个 bag。")
        for index, bag_path in enumerate(bag_paths, start=1):
            session_id, base_pose_id = identifiers_for_bag(
                bag_path,
                args,
                len(bag_paths),
            )
            print(
                f"\n[{index}/{len(bag_paths)}] 处理 {bag_path.name} "
                f"(session_id={session_id}, pose_id前缀={base_pose_id})"
            )
            try:
                result = process_bag(
                    bag_path=bag_path,
                    args=args,
                    rosbag_module=rosbag_module,
                    session_id=session_id,
                    base_pose_id=base_pose_id,
                )
            except ProcessingError as exc:
                if not args.skip_bad_bags:
                    raise
                failed_bags.append((bag_path, str(exc)))
                print(f"[跳过] {exc}", file=sys.stderr)
                continue

            processed_rows.extend(result.processed_rows)
            raw_rows.extend(result.raw_rows)
            print(format_counter_summary(result.counters))

            for warning in result.marker_warnings:
                print(
                    f"[标记警告] {warning}",
                    file=sys.stderr,
                )

            rows_by_pose = {
                str(row["pose_id"]): row for row in result.processed_rows
            }
            for report in result.segment_reports:
                if report.quality_ok:
                    row = rows_by_pose[report.pose_id]
                    print(
                        f"[接受] segment={report.segment_index}, "
                        f"pose_id={report.pose_id}, "
                        f"available={report.available_count}, "
                        f"selected={report.selected_count}, "
                        f"d_gt={float(row['distance_gt_mean']):.4f} m, "
                        f"range={float(row['range_mean']):.4f} m, "
                        f"bias={float(row['bias_mean']) * 100.0:.3f} cm, "
                        f"bias_std={float(row['bias_std']) * 100.0:.3f} cm"
                    )
                else:
                    total_rejected_segments += 1
                    print(
                        f"[拒绝] segment={report.segment_index}, "
                        f"pose_id={report.pose_id}, "
                        f"available={report.available_count}, "
                        f"selected={report.selected_count}, "
                        f"reason={report.reject_reason}",
                        file=sys.stderr,
                    )
                if report.stability is not None:
                    print(
                        "        "
                        + format_stability_metrics(report.stability)
                    )

        if not args.skip_raw_output:
            if raw_rows:
                raw_path = Path(args.raw_output)
                write_csv_atomic(raw_path, RAW_FIELDS, raw_rows)
            else:
                print(
                    "[警告] 没有成功对齐的原始测距，未生成 raw CSV。",
                    file=sys.stderr,
                )

        if not processed_rows:
            raw_hint = (
                f"；请检查 {Path(args.raw_output).expanduser().resolve()}"
                if not args.skip_raw_output and raw_rows
                else ""
            )
            raise ProcessingError(
                "所有标记静止段均被拒绝，没有生成训练数据"
                f"{raw_hint}。"
            )

        processed_path = Path(args.processed_output)
        write_csv_atomic(processed_path, PROCESSED_FIELDS, processed_rows)

        print("\n转换完成。")
        print(
            f"处理后训练CSV：{Path(args.processed_output).expanduser().resolve()}"
        )
        if not args.skip_raw_output:
            print(
                f"逐次对齐CSV：{Path(args.raw_output).expanduser().resolve()}"
            )
        print(
            f"成功读取bag={len(bag_paths) - len(failed_bags)}, "
            f"失败bag={len(failed_bags)}, "
            f"接受静止段/训练行数={len(processed_rows)}, "
            f"拒绝静止段={total_rejected_segments}, "
            f"原始对齐行数={len(raw_rows)}"
        )
        if failed_bags:
            print("\n被跳过的bag：", file=sys.stderr)
            for path, reason in failed_bags:
                print(f"- {path}: {reason}", file=sys.stderr)

    except (ProcessingError, OSError, ValueError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
