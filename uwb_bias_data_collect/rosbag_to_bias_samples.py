#!/usr/bin/env python3
"""
将 ROS1 rosbag 转换为逐测距 UWB 偏置训练 CSV。

核心规则
--------
每一条成功与 A、B 动捕位姿对齐的 UWB 测距，生成一条训练数据：

    bias_mean = range_m - distance_gt_m

其中：

- ``range_m`` 是该时刻的一次 UWB 测距；
- ``distance_gt_m`` 是把 A、B 动捕位姿插值到该 UWB 时刻后得到的真值距离；
- ``rel_tx ... rel_R22`` 是同一时刻的 A 到 B 相对位姿；
- ``range_count`` 固定为 1；
- ``bias_std`` 留空，因为单次测距无法估计自身标准差。

脚本默认处理 bag 中所有有效 UWB 测距，不要求采集标记，因此可用于连续运动
数据。若 bag 中包含 ``/uwb_bias/collection_marker``，脚本会把静止段编号写入
CSV；添加 ``--only-marked-static`` 后，只输出标记静止段内的测距。

时间同步
--------
UWB 消息没有 header，始终使用 rosbag 记录时间。动捕默认也使用 rosbag 记录
时间，并在 UWB 时刻对位置做线性插值、对四元数做 SLERP。

动态数据对时间偏移更敏感。可使用：

    --uwb-time-offset-sec 0.005

令动捕查询时刻为：

    pose_query_time = uwb_bag_time + uwb_time_offset_sec

正值表示使用比 UWB bag 时间更晚的动捕位姿。

输出兼容性
----------
输出继续保留 ``bias_mean``、``range_count``、``rel_tx ... rel_R22`` 等字段，
可直接交给原有 ``uwb_bias_training.py``。由于每行 ``range_count=1``，
不应依赖该行的 ``bias_std/sqrt(range_count)`` 进行加权；建议使用等权重和
Huber 损失，或者另外标定全局测距噪声。

基本用法
--------
处理一个 bag 中的全部有效测距：

    python3 rosbag_to_bias_samples.py data/run_003.bag \
        --session-id run_003 \
        --processed-output datasets/processed/run_003_samples.csv

只保留人工标记的静止段：

    python3 rosbag_to_bias_samples.py data/run_003.bag \
        --only-marked-static \
        --processed-output datasets/processed/run_003_static_samples.csv

批量处理文件夹：

    python3 rosbag_to_bias_samples.py data \
        --recursive \
        --processed-output datasets/processed/all_samples.csv

若还需要保存“所有成功对齐数据”的诊断副本，可额外指定：

    --raw-output datasets/raw/all_aligned.csv

最少需要录制：

    rosbag record -O data/run_003 \
        /nlink_linktrack_nodeframe2 \
        /vrpn_client/UWB_A/pose \
        /vrpn_client/UWB_B/pose

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TRAINING_FIELDS: List[str] = [
    "session_id",
    "sample_id",
    "pose_id",
    "marker_pose_id",
    "segment_index",
    "in_marked_static_segment",
    "bag_file",
    "sample_index",
    "uwb_timestamp",
    "pose_query_timestamp",
    "time_from_bag_start_sec",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "segment_closed_by",
    "uwb_local_time",
    "uwb_system_time",
    "local_role",
    "local_id",
    "remote_role",
    "remote_id",
    "range_m",
    "range_mean",
    "range_std",
    "fp_rssi",
    "rx_rssi",
    "rssi_diff",
    "fp_rssi_mean",
    "rx_rssi_mean",
    "rssi_diff_mean",
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
    "rel_qx",
    "rel_qy",
    "rel_qz",
    "rel_qw",
    "rel_R00",
    "rel_R01",
    "rel_R02",
    "rel_R10",
    "rel_R11",
    "rel_R12",
    "rel_R20",
    "rel_R21",
    "rel_R22",
    "relative_distance_m",
    "relative_azimuth_rad",
    "relative_azimuth_deg",
    "relative_elevation_rad",
    "relative_elevation_deg",
    "distance_gt_m",
    "distance_gt_mean",
    "distance_gt_std",
    "range_error_m",
    "bias_mean",
    "bias_std",
    "range_count",
    "position_stability",
    "rotation_stability",
    "pose_a_position_stability",
    "pose_a_rotation_stability",
    "pose_b_position_stability",
    "pose_b_rotation_stability",
    "pose_a_nearest_dt_sec",
    "pose_b_nearest_dt_sec",
    "max_pose_sync_dt_sec",
    "quality_ok",
    "used_for_training",
]


class ProcessingError(RuntimeError):
    """输入数据无法安全转换时抛出的错误。"""


@dataclass(frozen=True)
class PoseRecord:
    """一条动捕位姿。四元数使用 ROS xyzw 顺序。"""

    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class PoseSample:
    """在 UWB 对齐时刻插值得到的动捕位姿。"""

    position: np.ndarray
    quaternion: np.ndarray
    nearest_dt_sec: float


@dataclass(frozen=True)
class UwbMeasurement:
    """指定 local_id 与 remote_id 之间的一条测距。"""

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
    """由 STATIC_START 与 MOVE_START（或 bag 结束）构成的静止段。"""

    segment_index: int
    marker_pose_id: str
    start_timestamp: float
    end_timestamp: float
    closed_by: str


@dataclass(frozen=True)
class MatchedMeasurement:
    """一条 UWB 测距及其同一时刻的 A、B 动捕位姿。"""

    uwb: UwbMeasurement
    pose_query_timestamp: float
    pose_a: PoseSample
    pose_b: PoseSample
    relative_translation: np.ndarray
    relative_quaternion: np.ndarray
    relative_rotation: np.ndarray
    distance_gt_m: float
    range_error_m: float


@dataclass
class BagResult:
    processed_rows: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    counters: Counter
    marker_warnings: List[str]


class PoseSeries:
    """按时间排序的位姿序列，支持位置线性插值和四元数 SLERP。"""

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
            raise ProcessingError(f"{name} 有效时间戳少于2个，无法插值。")

        self.name = name
        self.times = np.asarray(
            [item.timestamp for item in deduplicated],
            dtype=np.float64,
        )
        self.positions = np.stack(
            [item.position for item in deduplicated],
            axis=0,
        ).astype(np.float64)
        self.quaternions = np.stack(
            [item.quaternion for item in deduplicated],
            axis=0,
        ).astype(np.float64)

    def interpolate(
        self,
        timestamp: float,
        max_gap_sec: float,
    ) -> Optional[PoseSample]:
        """在指定时刻插值；附近动捕数据不足时返回 None。"""
        index = int(np.searchsorted(self.times, timestamp, side="left"))

        if index == 0:
            dt = float(self.times[0] - timestamp)
            if abs(dt) > max_gap_sec:
                return None
            return PoseSample(
                position=self.positions[0].copy(),
                quaternion=self.quaternions[0].copy(),
                nearest_dt_sec=abs(dt),
            )

        if index == len(self.times):
            dt = float(timestamp - self.times[-1])
            if abs(dt) > max_gap_sec:
                return None
            return PoseSample(
                position=self.positions[-1].copy(),
                quaternion=self.quaternions[-1].copy(),
                nearest_dt_sec=abs(dt),
            )

        t0 = float(self.times[index - 1])
        t1 = float(self.times[index])
        dt0 = timestamp - t0
        dt1 = t1 - timestamp

        if abs(dt0) <= 1e-12:
            return PoseSample(
                position=self.positions[index - 1].copy(),
                quaternion=self.quaternions[index - 1].copy(),
                nearest_dt_sec=0.0,
            )
        if abs(dt1) <= 1e-12:
            return PoseSample(
                position=self.positions[index].copy(),
                quaternion=self.quaternions[index].copy(),
                nearest_dt_sec=0.0,
            )

        # 两侧帧都要足够近，避免跨过动捕长时间丢帧区间。
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


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """归一化 ROS 顺序 [x, y, z, w] 四元数。"""
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        raise ValueError("四元数含 NaN 或 Inf。")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("四元数模长接近0。")
    value = value / norm
    if value[3] < 0.0:
        value = -value
    return value


def quaternion_conjugate(quaternion: Sequence[float]) -> np.ndarray:
    value = normalize_quaternion(quaternion)
    return np.asarray(
        [-value[0], -value[1], -value[2], value[3]],
        dtype=np.float64,
    )


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
    """将 ROS xyzw 四元数转换为 3×3 旋转矩阵。"""
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


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def extract_pose_record(
    message: Any,
    bag_timestamp: float,
    time_source: str,
) -> Optional[PoseRecord]:
    """从 geometry_msgs/PoseStamped 风格消息中提取动捕位姿。"""
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
    """解析 std_msgs/String JSON；标记边界使用 rosbag 记录时间。"""
    counters["marker_messages_total"] += 1
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
    """从一帧 LinkTrack nodeframe 中提取指定天线对的一条测距。"""
    counters["uwb_frames_total"] += 1
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

    matching_nodes: List[Any] = []
    for node in nodes:
        try:
            node_id = int(node.id)
        except (AttributeError, TypeError, ValueError):
            continue
        if node_id == remote_id:
            matching_nodes.append(node)

    if not matching_nodes:
        counters["uwb_frames_remote_not_found"] += 1
        return None
    if len(matching_nodes) > 1:
        counters["uwb_frames_duplicate_remote"] += 1

    node = matching_nodes[0]
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
        local_time=safe_int(getattr(message, "local_time", 0)),
        system_time=safe_int(getattr(message, "system_time", 0)),
        local_role=safe_int(getattr(message, "role", -1), -1),
        local_id=message_local_id,
        remote_role=safe_int(getattr(node, "role", -1), -1),
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
    """按标记顺序构造不重叠静止段；标记不存在时返回空列表。"""
    intervals: List[StaticInterval] = []
    warnings: List[str] = []
    active: Optional[CollectionMarker] = None

    for marker in sorted(markers, key=lambda item: item.timestamp):
        if marker.event == "STATIC_START":
            if active is not None:
                counters["marker_duplicate_static_start"] += 1
                warnings.append(
                    "上一静止段尚未结束又收到 STATIC_START："
                    f"放弃 pose_id={active.marker_pose_id}，"
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
                f"start={active.marker_pose_id}，"
                f"move={marker.marker_pose_id}；该段已放弃。"
            )
            active = None
            continue
        if marker.timestamp <= active.timestamp:
            counters["marker_nonpositive_interval"] += 1
            warnings.append(
                f"pose_id={active.marker_pose_id} 的静止段时长不大于0，"
                "该段已放弃。"
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
                f"pose_id={active.marker_pose_id} 的 STATIC_START 位于 bag "
                "末尾，没有形成有效静止段。"
            )

    counters["static_intervals_built"] = len(intervals)
    return intervals, warnings


def marker_pose_token(marker_pose_id: str) -> str:
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


def segment_pose_ids(
    intervals: Sequence[StaticInterval],
    base_pose_id: str,
) -> Dict[int, str]:
    """为每个静止段生成稳定且不重复的 pose_id。"""
    occurrences: Dict[str, int] = {}
    result: Dict[int, str] = {}
    for interval in intervals:
        marker_id = interval.marker_pose_id
        occurrences[marker_id] = occurrences.get(marker_id, 0) + 1
        base = f"{base_pose_id}_pose_{marker_pose_token(marker_id)}"
        occurrence = occurrences[marker_id]
        result[interval.segment_index] = (
            base if occurrence == 1 else f"{base}_repeat_{occurrence:02d}"
        )
    return result


def relative_pose(
    pose_a: PoseSample,
    pose_b: PoseSample,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """由 T_M_A、T_M_B 计算 T_A_B 和两动捕原点之间的真值距离。"""
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


def interval_for_timestamp(
    intervals: Sequence[StaticInterval],
    timestamp: float,
    start_index: int,
) -> Tuple[Optional[StaticInterval], int]:
    """在线性扫描中查找包含 timestamp 的静止段。"""
    index = start_index
    while index < len(intervals) and timestamp > intervals[index].end_timestamp:
        index += 1
    if index >= len(intervals):
        return None, index
    interval = intervals[index]
    if interval.start_timestamp <= timestamp <= interval.end_timestamp:
        return interval, index
    return None, index


def make_sample_row(
    match: MatchedMeasurement,
    session_id: str,
    bag_file: str,
    bag_start_time: float,
    sample_index: int,
    sample_id: str,
    pose_id: str,
    interval: Optional[StaticInterval],
    used_for_training: bool,
) -> Dict[str, Any]:
    """把一条成功对齐的测距转换为一条 CSV 行。"""
    uwb = match.uwb
    translation = match.relative_translation
    horizontal_distance = math.hypot(
        float(translation[0]),
        float(translation[1]),
    )
    azimuth_rad = math.atan2(
        float(translation[1]),
        float(translation[0]),
    )
    elevation_rad = math.atan2(
        float(translation[2]),
        horizontal_distance,
    )
    rssi_diff = uwb.rx_rssi - uwb.fp_rssi
    max_sync_dt = max(
        match.pose_a.nearest_dt_sec,
        match.pose_b.nearest_dt_sec,
    )

    row: Dict[str, Any] = {
        "session_id": session_id,
        "sample_id": sample_id,
        "pose_id": pose_id,
        "marker_pose_id": (
            interval.marker_pose_id if interval is not None else ""
        ),
        "segment_index": (
            interval.segment_index if interval is not None else ""
        ),
        "in_marked_static_segment": int(interval is not None),
        "bag_file": bag_file,
        "sample_index": sample_index,
        "uwb_timestamp": uwb.timestamp,
        "pose_query_timestamp": match.pose_query_timestamp,
        "time_from_bag_start_sec": uwb.timestamp - bag_start_time,
        "segment_start_sec": (
            interval.start_timestamp - bag_start_time
            if interval is not None
            else ""
        ),
        "segment_end_sec": (
            interval.end_timestamp - bag_start_time
            if interval is not None
            else ""
        ),
        "segment_duration_sec": (
            interval.end_timestamp - interval.start_timestamp
            if interval is not None
            else ""
        ),
        "segment_closed_by": (
            interval.closed_by if interval is not None else ""
        ),
        "uwb_local_time": uwb.local_time,
        "uwb_system_time": uwb.system_time,
        "local_role": uwb.local_role,
        "local_id": uwb.local_id,
        "remote_role": uwb.remote_role,
        "remote_id": uwb.remote_id,
        "range_m": uwb.range_m,
        # 以下 mean 字段是为了兼容原先“静止段均值”训练脚本。
        "range_mean": uwb.range_m,
        "range_std": "",
        "fp_rssi": uwb.fp_rssi,
        "rx_rssi": uwb.rx_rssi,
        "rssi_diff": rssi_diff,
        "fp_rssi_mean": uwb.fp_rssi,
        "rx_rssi_mean": uwb.rx_rssi,
        "rssi_diff_mean": rssi_diff,
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
        "rel_tx": translation[0],
        "rel_ty": translation[1],
        "rel_tz": translation[2],
        "rel_qx": match.relative_quaternion[0],
        "rel_qy": match.relative_quaternion[1],
        "rel_qz": match.relative_quaternion[2],
        "rel_qw": match.relative_quaternion[3],
        "relative_distance_m": match.distance_gt_m,
        "relative_azimuth_rad": azimuth_rad,
        "relative_azimuth_deg": math.degrees(azimuth_rad),
        "relative_elevation_rad": elevation_rad,
        "relative_elevation_deg": math.degrees(elevation_rad),
        "distance_gt_m": match.distance_gt_m,
        "distance_gt_mean": match.distance_gt_m,
        "distance_gt_std": "",
        "range_error_m": match.range_error_m,
        # 单帧标签：偏置真值中仍包含该次测距的随机噪声。
        "bias_mean": match.range_error_m,
        "bias_std": "",
        "range_count": 1,
        # 单帧无法估计段内稳定性，留空而不是错误地写成0。
        "position_stability": "",
        "rotation_stability": "",
        "pose_a_position_stability": "",
        "pose_a_rotation_stability": "",
        "pose_b_position_stability": "",
        "pose_b_rotation_stability": "",
        "pose_a_nearest_dt_sec": match.pose_a.nearest_dt_sec,
        "pose_b_nearest_dt_sec": match.pose_b.nearest_dt_sec,
        "max_pose_sync_dt_sec": max_sync_dt,
        "quality_ok": 1,
        "used_for_training": int(used_for_training),
    }
    for row_index in range(3):
        for column_index in range(3):
            row[f"rel_R{row_index}{column_index}"] = match.relative_rotation[
                row_index,
                column_index,
            ]
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


def process_bag(
    bag_path: Path,
    args: argparse.Namespace,
    rosbag_module: Any,
    session_id: str,
    base_pose_id: str,
) -> BagResult:
    """读取一个 bag，完成逐测距时间同步并生成训练行。"""
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
            f"{bag_path.name} 中没有符合 local_id={args.local_id}、"
            f"remote_id={args.remote_id} 的有效测距。统计：{details}"
        )

    intervals, marker_warnings = build_static_intervals(
        markers=markers,
        bag_end_time=bag_end_time,
        counters=counters,
    )
    if args.only_marked_static and not intervals:
        raise ProcessingError(
            f"{bag_path.name} 启用了 --only-marked-static，"
            f"但 {args.marker_topic} 没有形成有效静止段。"
        )

    pose_a_series = PoseSeries(pose_a_records, args.pose_a_topic)
    pose_b_series = PoseSeries(pose_b_records, args.pose_b_topic)
    matches: List[MatchedMeasurement] = []

    for measurement in uwb_measurements:
        pose_query_timestamp = (
            measurement.timestamp + args.uwb_time_offset_sec
        )
        pose_a = pose_a_series.interpolate(
            pose_query_timestamp,
            args.max_pose_gap_sec,
        )
        if pose_a is None:
            counters["uwb_sync_failed_pose_a"] += 1
            continue
        pose_b = pose_b_series.interpolate(
            pose_query_timestamp,
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
                pose_query_timestamp=pose_query_timestamp,
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
            "请检查时间戳、--uwb-time-offset-sec 和 --max-pose-gap-sec。"
        )

    interval_pose_id = segment_pose_ids(intervals, base_pose_id)
    processed_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    interval_scan_index = 0
    want_raw = bool(args.raw_output) and not args.skip_raw_output
    eligible_index = 0

    for matched_index, match in enumerate(matches, start=1):
        interval, interval_scan_index = interval_for_timestamp(
            intervals,
            match.uwb.timestamp,
            interval_scan_index,
        )
        eligible_for_training = (
            interval is not None or not args.only_marked_static
        )
        if eligible_for_training:
            eligible_index += 1
        use_for_training = (
            eligible_for_training
            and (eligible_index - 1) % args.sample_stride == 0
        )

        sample_id = f"{base_pose_id}_sample_{matched_index:08d}"
        if interval is None:
            pose_id = sample_id
        else:
            pose_id = interval_pose_id[interval.segment_index]

        row = make_sample_row(
            match=match,
            session_id=session_id,
            bag_file=bag_path.name,
            bag_start_time=bag_start_time,
            sample_index=matched_index,
            sample_id=sample_id,
            pose_id=pose_id,
            interval=interval,
            used_for_training=use_for_training,
        )
        if want_raw:
            raw_rows.append(row)
        if not eligible_for_training:
            counters["matched_outside_marked_static"] += 1
            continue
        if not use_for_training:
            counters["training_rows_stride_skipped"] += 1
            continue
        processed_rows.append(row)

    counters["training_rows_written"] = len(processed_rows)
    if not processed_rows:
        mode = "标记静止段内" if args.only_marked_static else "全部"
        raise ProcessingError(
            f"{bag_path.name} 的{mode}成功对齐测距中没有可写训练数据。"
        )

    return BagResult(
        processed_rows=processed_rows,
        raw_rows=raw_rows,
        counters=counters,
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
        base_pose_id = args.pose_id
    else:
        base_pose_id = bag_path.stem
    return session_id, base_pose_id


def native_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
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
            "把 ROS1 bag 转换为“每条UWB测距+同一时刻动捕位姿=一行”的"
            "偏置训练 CSV"
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
        default="processed_bias_samples.csv",
        help="逐测距训练 CSV",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help=(
            "可选：保存所有成功对齐测距；启用 --only-marked-static 时，"
            "它还会保留静止段之外的行并令 used_for_training=0"
        ),
    )
    parser.add_argument(
        "--skip-raw-output",
        action="store_true",
        help="兼容旧命令：即使指定 --raw-output 也不写诊断 CSV",
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
        help="可选的 std_msgs/String JSON 采集标记话题",
    )
    parser.add_argument(
        "--only-marked-static",
        action="store_true",
        help="只把 STATIC_START 到 MOVE_START 之间的测距写入训练 CSV",
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
        help="可选最大有效测距；默认不设置上限",
    )

    parser.add_argument(
        "--pose-time-source",
        choices=["bag", "header"],
        default="bag",
        help=(
            "动捕时间来源；UWB没有header，因此默认双方都使用bag记录时间"
        ),
    )
    parser.add_argument(
        "--uwb-time-offset-sec",
        type=float,
        default=0.0,
        help=(
            "动捕查询时刻相对UWB bag时间的偏移；"
            "pose_time=uwb_time+offset"
        ),
    )
    parser.add_argument(
        "--max-pose-gap-sec",
        type=float,
        default=0.05,
        help="插值时UWB查询时刻到相邻动捕帧允许的最大时间间隔",
    )
    parser.add_argument(
        "--trim-start-sec",
        type=float,
        default=0.0,
        help="忽略每个 bag 开头若干秒的 UWB 数据",
    )
    parser.add_argument(
        "--trim-end-sec",
        type=float,
        default=0.0,
        help="忽略每个 bag 结尾若干秒的 UWB 数据",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="每隔多少条合格测距保留一条；1表示全部保留",
    )

    parser.add_argument(
        "--session-id",
        default=None,
        help="指定所有输入 bag 的 session_id；省略时自动生成",
    )
    parser.add_argument(
        "--session-from",
        choices=["parent", "bag"],
        default="parent",
        help="未指定 --session-id 时，从父文件夹名或 bag 名生成",
    )
    parser.add_argument(
        "--pose-id",
        default=None,
        help="单个 bag 时指定 pose_id 前缀；默认使用 bag 文件名",
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
    if not math.isfinite(args.uwb_time_offset_sec):
        raise ProcessingError("--uwb-time-offset-sec 必须是有限数值。")
    if (
        not math.isfinite(args.trim_start_sec)
        or not math.isfinite(args.trim_end_sec)
        or args.trim_start_sec < 0.0
        or args.trim_end_sec < 0.0
    ):
        raise ProcessingError("裁剪时间不能小于0。")
    if args.sample_stride < 1:
        raise ProcessingError("--sample-stride 至少为1。")

    processed = Path(args.processed_output).expanduser().resolve()
    if args.raw_output and not args.skip_raw_output:
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
        "matched_outside_marked_static",
        "training_rows_stride_skipped",
        "training_rows_written",
    ]
    extras = sorted(
        key
        for key, value in counters.items()
        if value and key not in selected and key.startswith("marker_")
    )
    return ", ".join(
        f"{key}={int(counters[key])}" for key in selected + extras
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
        bag_paths = discover_bags(args.inputs, args.recursive)
        if args.pose_id and len(bag_paths) != 1:
            raise ProcessingError("--pose-id 只能用于单个 bag。")

        rosbag_module = load_rosbag_module()
        processed_rows: List[Dict[str, Any]] = []
        raw_rows: List[Dict[str, Any]] = []
        failed_bags: List[Tuple[Path, str]] = []

        print(f"发现 {len(bag_paths)} 个 bag。")
        for index, bag_path in enumerate(bag_paths, start=1):
            session_id, base_pose_id = identifiers_for_bag(
                bag_path,
                args,
                len(bag_paths),
            )
            print(
                f"\n[{index}/{len(bag_paths)}] 处理 {bag_path.name} "
                f"(session_id={session_id}, sample前缀={base_pose_id})"
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
                print(f"[标记警告] {warning}", file=sys.stderr)

        if not processed_rows:
            raise ProcessingError("所有 bag 均没有生成训练数据。")

        processed_path = Path(args.processed_output)
        write_csv_atomic(
            processed_path,
            TRAINING_FIELDS,
            processed_rows,
        )
        if args.raw_output and not args.skip_raw_output:
            if raw_rows:
                write_csv_atomic(
                    Path(args.raw_output),
                    TRAINING_FIELDS,
                    raw_rows,
                )
            else:
                print(
                    "[警告] 没有可写入诊断 CSV 的成功对齐测距。",
                    file=sys.stderr,
                )

        print("\n转换完成。")
        print(
            "逐测距训练CSV："
            f"{Path(args.processed_output).expanduser().resolve()}"
        )
        if args.raw_output and not args.skip_raw_output:
            print(
                "全部对齐诊断CSV："
                f"{Path(args.raw_output).expanduser().resolve()}"
            )
        print(
            f"成功bag={len(bag_paths) - len(failed_bags)}, "
            f"失败bag={len(failed_bags)}, "
            f"训练行数={len(processed_rows)}, "
            f"诊断行数={len(raw_rows)}"
        )
        if failed_bags:
            print("\n被跳过的 bag：", file=sys.stderr)
            for path, reason in failed_bags:
                print(f"- {path}: {reason}", file=sys.stderr)

    except (ProcessingError, OSError, ValueError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
