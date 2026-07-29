#!/usr/bin/env python3
"""
将 ROS1 rosbag 转换为 UWB 单天线对单天线偏置学习 CSV。

当前默认配置对应：

- 动捕 A：/vrpn_client/UWB_A/pose
- 动捕 B：/vrpn_client/UWB_B/pose
- UWB：/nlink_linktrack_nodeframe2
- 连接电脑的本机 UWB_A ID：1
- 远端 UWB_B ID：2
- nlink_parser/LinktrackNodeframe2.nodes[].dis 的单位：米
- 一个 bag 只包含一个静态相对位姿
- 使用静态 bag 内全部有效测距

对每条 UWB 测距 z_k，脚本根据时间同步 A、B 的动捕位姿，计算：

    T_A_B = inverse(T_M_A) @ T_M_B
    d_gt,k = ||p_M_B - p_M_A||_2
    e_k = z_k - d_gt,k

随后把一个 bag 聚合成一行训练数据：

    bias_mean = mean(e_k)
    bias_std  = sample_std(e_k)

并输出：

1. processed_bias.csv：每个 bag/静态位姿一行，可直接交给
   uwb_bias_training.py。
2. raw_aligned.csv：每条成功对齐的 UWB 测距一行，便于检查时间同步、
   坐标变换和异常值。

最简单的用法
------------

    source /opt/ros/noetic/setup.bash
    source ~/catkin_workspace/devel/setup.bash

    python3 rosbag_to_bias_csv.py data/pose_001.bag \
        --session-id run_001 \
        --processed-output datasets/processed/run_001_pose_001.csv \
        --raw-output datasets/raw/run_001_pose_001_raw.csv

一次处理一个文件夹中的全部 bag：

    python3 rosbag_to_bias_csv.py data/run_001 \
        --session-id run_001 \
        --processed-output datasets/processed/run_001.csv \
        --raw-output datasets/raw/run_001_raw.csv

如果 bag 开头和结尾包含按下录制按钮造成的不稳定数据，可以裁剪：

    python3 rosbag_to_bias_csv.py data/pose_001.bag \
        --trim-start-sec 1.0 \
        --trim-end-sec 1.0 \
        --processed-output processed_bias.csv \
        --raw-output raw_aligned.csv

依赖
----

- ROS Noetic 的 Python 3 环境与 rosbag
- 已编译并 source nlink_parser 工作空间
- numpy

说明
----

LinktrackNodeframe2 没有 ROS header，因此 UWB 时间使用 rosbag 的记录时间。
默认也使用动捕消息的 rosbag 记录时间，保证两个数据源使用同一时间基准。
若确认 VRPN header.stamp 与 rosbag 时间属于同一时钟，可显式指定
--pose-time-source header。
"""

from __future__ import annotations

import argparse
import csv
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
    "bag_file",
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
    "fp_rssi_mean",
    "rx_rssi_mean",
    "rssi_diff_mean",
    "measurement_span_sec",
    "measurement_rate_hz",
    "pose_a_sync_mean_sec",
    "pose_a_sync_max_sec",
    "pose_b_sync_mean_sec",
    "pose_b_sync_max_sec",
    "quality_ok",
]


RAW_FIELDS: List[str] = [
    "session_id",
    "pose_id",
    "bag_file",
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


@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class PoseSample:
    position: np.ndarray
    quaternion: np.ndarray
    nearest_dt_sec: float


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
class MatchedMeasurement:
    uwb: UwbMeasurement
    pose_a: PoseSample
    pose_b: PoseSample
    relative_translation: np.ndarray
    relative_quaternion: np.ndarray
    relative_rotation: np.ndarray
    distance_gt_m: float
    range_error_m: float


@dataclass
class BagResult:
    processed_row: Dict[str, Any]
    raw_rows: List[Dict[str, Any]]
    counters: Counter


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


def extract_pose_record(
    message: Any,
    bag_timestamp: float,
    time_source: str,
) -> Optional[PoseRecord]:
    timestamp = message_timestamp(message, bag_timestamp, time_source)
    if timestamp is None:
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
    pose_id: str,
    bag_file: str,
    bag_start_time: float,
) -> Dict[str, Any]:
    uwb = match.uwb
    row: Dict[str, Any] = {
        "session_id": session_id,
        "pose_id": pose_id,
        "bag_file": bag_file,
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


def summarize_matches(
    matches: Sequence[MatchedMeasurement],
    session_id: str,
    pose_id: str,
    bag_file: str,
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

    translation_deviation = translations - translation_mean
    position_stability = float(
        np.sqrt(np.mean(np.sum(translation_deviation**2, axis=1)))
    )
    rotation_angles = np.asarray(
        [
            quaternion_angular_distance_deg(item, quaternion_mean)
            for item in quaternions
        ],
        dtype=np.float64,
    )
    rotation_stability = float(np.sqrt(np.mean(rotation_angles**2)))

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

    fp_rssi_mean = finite_mean(item.uwb.fp_rssi for item in matches)
    rx_rssi_mean = finite_mean(item.uwb.rx_rssi for item in matches)
    rssi_diff_mean = finite_mean(
        item.uwb.rx_rssi - item.uwb.fp_rssi for item in matches
    )

    quality_ok = int(
        position_stability <= max_position_stability_m
        and rotation_stability <= max_rotation_stability_deg
    )

    row: Dict[str, Any] = {
        "session_id": session_id,
        "pose_id": pose_id,
        "bag_file": bag_file,
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
        "position_stability": position_stability,
        "rotation_stability": rotation_stability,
        "fp_rssi_mean": fp_rssi_mean,
        "rx_rssi_mean": rx_rssi_mean,
        "rssi_diff_mean": rssi_diff_mean,
        "measurement_span_sec": measurement_span,
        "measurement_rate_hz": measurement_rate,
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
            "内部检查失败：range_mean-distance_gt_mean 与 mean(range-distance) 不一致。"
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


def process_bag(
    bag_path: Path,
    args: argparse.Namespace,
    rosbag_module: Any,
    session_id: str,
    pose_id: str,
) -> BagResult:
    counters: Counter = Counter()
    pose_a_records: List[PoseRecord] = []
    pose_b_records: List[PoseRecord] = []
    uwb_measurements: List[UwbMeasurement] = []

    try:
        bag = rosbag_module.Bag(str(bag_path), "r")
    except Exception as exc:
        raise ProcessingError(f"无法打开 rosbag：{bag_path}\n{exc}") from exc

    try:
        bag_start_time = float(bag.get_start_time())
        bag_end_time = float(bag.get_end_time())
        trimmed_start = bag_start_time + args.trim_start_sec
        trimmed_end = bag_end_time - args.trim_end_sec
        if trimmed_end <= trimmed_start:
            raise ProcessingError(
                f"{bag_path.name} 裁剪后的时间段为空："
                f"bag时长={bag_end_time - bag_start_time:.3f}s，"
                f"trim_start={args.trim_start_sec:.3f}s，"
                f"trim_end={args.trim_end_sec:.3f}s。"
            )

        topics = [args.pose_a_topic, args.pose_b_topic, args.uwb_topic]
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

    counters["uwb_measurements_matched"] = len(matches)
    if len(matches) < args.min_samples:
        raise ProcessingError(
            f"{bag_path.name} 只有 {len(matches)} 条有效对齐测距，"
            f"少于 --min-samples={args.min_samples}。"
            "请检查节点ID、时间戳和 --max-pose-gap-sec，"
            "或在确认数据可靠后调低 --min-samples。"
        )

    processed_row = summarize_matches(
        matches=matches,
        session_id=session_id,
        pose_id=pose_id,
        bag_file=bag_path.name,
        max_position_stability_m=args.max_position_stability_m,
        max_rotation_stability_deg=args.max_rotation_stability_deg,
    )
    raw_rows = [
        make_raw_row(
            match=item,
            session_id=session_id,
            pose_id=pose_id,
            bag_file=bag_path.name,
            bag_start_time=bag_start_time,
        )
        for item in matches
    ]
    return BagResult(processed_row, raw_rows, counters)


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
            "把每个静态位姿一个 ROS1 bag 转换为 UWB 偏置训练 CSV"
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
        "--min-samples",
        type=int,
        default=200,
        help="每个静态位姿至少需要的成功对齐测距数量",
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
        help="单个bag时手工指定pose_id；默认使用bag文件名",
    )
    parser.add_argument(
        "--max-position-stability-m",
        type=float,
        default=0.01,
        help="相对位置RMS波动阈值，超过后quality_ok=0并打印警告",
    )
    parser.add_argument(
        "--max-rotation-stability-deg",
        type=float,
        default=1.0,
        help="相对旋转RMS波动阈值，超过后quality_ok=0并打印警告",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not (0 <= args.local_id <= 255):
        raise ProcessingError("--local-id 必须位于0到255。")
    if not (0 <= args.remote_id <= 255):
        raise ProcessingError("--remote-id 必须位于0到255。")
    if args.local_id == args.remote_id:
        raise ProcessingError("--local-id 与 --remote-id 不能相同。")
    if args.range_scale <= 0.0:
        raise ProcessingError("--range-scale 必须大于0。")
    if args.min_range_m < 0.0:
        raise ProcessingError("--min-range-m 不能小于0。")
    if args.max_range_m is not None:
        if args.max_range_m <= args.min_range_m:
            raise ProcessingError("--max-range-m 必须大于 --min-range-m。")
    if args.max_pose_gap_sec <= 0.0:
        raise ProcessingError("--max-pose-gap-sec 必须大于0。")
    if args.trim_start_sec < 0.0 or args.trim_end_sec < 0.0:
        raise ProcessingError("裁剪时间不能小于0。")
    if args.min_samples < 2:
        raise ProcessingError("--min-samples 至少为2。")
    if args.max_position_stability_m < 0.0:
        raise ProcessingError("--max-position-stability-m 不能小于0。")
    if args.max_rotation_stability_deg < 0.0:
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
        "uwb_frames_total",
        "uwb_frames_trimmed",
        "uwb_frames_wrong_local_id",
        "uwb_frames_remote_not_found",
        "uwb_frames_invalid_range",
        "uwb_measurements_valid_before_sync",
        "uwb_sync_failed_pose_a",
        "uwb_sync_failed_pose_b",
        "uwb_measurements_matched",
    ]
    return ", ".join(f"{key}={int(counters[key])}" for key in selected)


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

        print(f"发现 {len(bag_paths)} 个 bag。")
        for index, bag_path in enumerate(bag_paths, start=1):
            session_id, pose_id = identifiers_for_bag(
                bag_path,
                args,
                len(bag_paths),
            )
            print(
                f"\n[{index}/{len(bag_paths)}] 处理 {bag_path.name} "
                f"(session_id={session_id}, pose_id={pose_id})"
            )
            try:
                result = process_bag(
                    bag_path=bag_path,
                    args=args,
                    rosbag_module=rosbag_module,
                    session_id=session_id,
                    pose_id=pose_id,
                )
            except ProcessingError as exc:
                if not args.skip_bad_bags:
                    raise
                failed_bags.append((bag_path, str(exc)))
                print(f"[跳过] {exc}", file=sys.stderr)
                continue

            processed_rows.append(result.processed_row)
            raw_rows.extend(result.raw_rows)
            row = result.processed_row
            print(format_counter_summary(result.counters))
            print(
                "聚合结果："
                f"N={int(row['range_count'])}, "
                f"d_gt={float(row['distance_gt_mean']):.4f} m, "
                f"range={float(row['range_mean']):.4f} m, "
                f"bias={float(row['bias_mean']) * 100.0:.3f} cm, "
                f"bias_std={float(row['bias_std']) * 100.0:.3f} cm"
            )
            print(
                "静止程度："
                f"position_rms={float(row['position_stability']) * 1000.0:.3f} mm, "
                f"rotation_rms={float(row['rotation_stability']):.3f} deg"
            )
            if int(row["quality_ok"]) == 0:
                print(
                    "[警告] 该bag超过静止阈值，quality_ok=0。"
                    "建议检查raw CSV或在训练时按稳定性阈值过滤。",
                    file=sys.stderr,
                )

        if not processed_rows:
            raise ProcessingError("没有任何bag成功生成训练数据。")

        processed_path = Path(args.processed_output)
        write_csv_atomic(processed_path, PROCESSED_FIELDS, processed_rows)
        if not args.skip_raw_output:
            raw_path = Path(args.raw_output)
            write_csv_atomic(raw_path, RAW_FIELDS, raw_rows)

        print("\n转换完成。")
        print(
            f"处理后训练CSV：{Path(args.processed_output).expanduser().resolve()}"
        )
        if not args.skip_raw_output:
            print(
                f"逐次对齐CSV：{Path(args.raw_output).expanduser().resolve()}"
            )
        print(
            f"成功bag={len(processed_rows)}, "
            f"失败bag={len(failed_bags)}, "
            f"训练行数={len(processed_rows)}, "
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
