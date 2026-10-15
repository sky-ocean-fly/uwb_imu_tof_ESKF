#!/usr/bin/env python3
"""Reconstruct per-link UWB bias from the real multi-tag rosbag and mocap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.bias_estimation import summarize_bias_samples  # noqa: E402
from uwb_bias_learning.synchronization import TimedPose, synchronize_pose  # noqa: E402
from uwb_bias_learning.transforms import (  # noqa: E402
    quaternion_to_matrix,
    rotation_matrix_to_quaternion,
)


# Geometry copied exactly from loadGeometry() in
# uwb_pose_estimation/src/uwb_imu_tof_eskf_real_node.cpp.  The state convention
# there is T_A_B and rho_ij = ||p_A_B + R_A_B b_j - a_i||.
TAG_POSITIONS_B = {
    5: np.asarray([-0.017, 0.494, 0.037], dtype=float),
    6: np.asarray([-0.457, -0.006, 0.046], dtype=float),
    7: np.asarray([-0.023, -0.506, 0.058], dtype=float),
    8: np.asarray([0.472, 0.010, 0.046], dtype=float),
}
ANCHOR_POSITIONS_A = {
    0: np.asarray([-0.677, -3.480, 2.098], dtype=float),
    1: np.asarray([-2.863, 1.767, 0.913], dtype=float),
    2: np.asarray([-2.880, -2.275, 0.756], dtype=float),
    3: np.asarray([1.787, -2.296, 0.922], dtype=float),
    4: np.asarray([1.795, 1.748, 1.038], dtype=float),
    9: np.asarray([-0.736, 3.512, 1.813], dtype=float),
}
R_AM_A = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=float
)
R_BM_B = R_AM_A.copy()


CSV_COLUMNS = (
    "bag_timestamp_s",
    "mocap_timestamp_before_s",
    "mocap_timestamp_after_s",
    "mocap_interpolation_alpha",
    "mocap_sync_gap_s",
    "mocap_bracket_span_s",
    "tag_id",
    "anchor_id",
    "measured_range_m",
    "true_range_m",
    "bias_m",
    "fp_rssi",
    "rx_rssi",
    "p_A_B_x_m",
    "p_A_B_y_m",
    "p_A_B_z_m",
    "q_A_B_x",
    "q_A_B_y",
    "q_A_B_z",
    "q_A_B_w",
    "tag_x_A_m",
    "tag_y_A_m",
    "tag_z_A_m",
    "anchor_x_A_m",
    "anchor_y_A_m",
    "anchor_z_A_m",
)


class BagBiasError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all 4-tag x 6-anchor signed biases in a rosbag using "
            "the geometry and mocap convention from uwb_imu_tof_eskf_real_node.cpp."
        )
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--uwb-topic", default="/nlink_linktrack_nodeframe2")
    parser.add_argument("--mocap-topic", default="/vrpn_client/UWB_rocket/pose")
    parser.add_argument(
        "--max-sync-gap-s",
        type=float,
        default=0.04,
        help="Maximum nearest mocap-to-UWB time gap (default: %(default)s)",
    )
    parser.add_argument(
        "--max-mocap-bracket-s",
        type=float,
        default=0.10,
        help="Maximum span of the two mocap samples used for interpolation (default: %(default)s)",
    )
    return parser


def _new_paths(args) -> Tuple[Path, Path, Path]:
    bag_path = Path(args.bag).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    if not bag_path.is_file():
        raise BagBiasError("bag does not exist: {}".format(bag_path))
    if csv_path == summary_path:
        raise BagBiasError("CSV and JSON output paths must differ")
    for path in (csv_path, summary_path):
        if path.exists():
            raise BagBiasError("refusing to overwrite existing output: {}".format(path))
    if not math.isfinite(args.max_sync_gap_s) or args.max_sync_gap_s < 0.0:
        raise BagBiasError("max-sync-gap-s must be finite and non-negative")
    if not math.isfinite(args.max_mocap_bracket_s) or args.max_mocap_bracket_s <= 0.0:
        raise BagBiasError("max-mocap-bracket-s must be finite and positive")
    return bag_path, csv_path, summary_path


def _compact_stats(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    result = summarize_bias_samples(array)
    result["p05_m"] = float(np.percentile(array, 5.0))
    result["p95_m"] = float(np.percentile(array, 95.0))
    return result


def _group_stats(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, ...], List[float]] = defaultdict(list)
    for row in rows:
        key = tuple(int(row[name]) for name in keys)
        grouped[key].append(float(row["bias_m"]))
    result = []
    for key in sorted(grouped):
        item: Dict[str, object] = {name: value for name, value in zip(keys, key)}
        item["bias"] = _compact_stats(grouped[key])
        result.append(item)
    return result


def analyze(args) -> Dict[str, object]:
    try:
        import rosbag
    except ImportError as exc:
        raise BagBiasError("ROS rosbag Python module is unavailable; source the ROS environment") from exc

    bag_path, csv_path, summary_path = _new_paths(args)
    poses: List[TimedPose] = []
    uwb_messages = []
    counts = Counter()
    mocap_envelope_delays = []
    observed_types: Dict[str, set] = defaultdict(set)

    try:
        bag = rosbag.Bag(str(bag_path), "r")
    except Exception as exc:
        raise BagBiasError("cannot open bag: {}".format(exc)) from exc
    try:
        info = bag.get_type_and_topic_info()
        for topic, value in info.topics.items():
            observed_types[str(topic)].add(str(value.msg_type))
        for topic, message, bag_stamp in bag.read_messages(
            topics=[args.uwb_topic, args.mocap_topic]
        ):
            receive_time = float(bag_stamp.to_sec())
            if topic == args.mocap_topic:
                counts["mocap_messages"] += 1
                header_time = float(message.header.stamp.to_sec())
                if not math.isfinite(header_time) or header_time <= 0.0:
                    counts["mocap_invalid_timestamp"] += 1
                    continue
                try:
                    poses.append(
                        TimedPose(
                            header_time,
                            [
                                message.pose.position.x,
                                message.pose.position.y,
                                message.pose.position.z,
                            ],
                            [
                                message.pose.orientation.x,
                                message.pose.orientation.y,
                                message.pose.orientation.z,
                                message.pose.orientation.w,
                            ],
                        )
                    )
                except ValueError:
                    counts["mocap_invalid_pose"] += 1
                    continue
                mocap_envelope_delays.append(receive_time - header_time)
            else:
                counts["uwb_messages"] += 1
                if int(message.id) not in TAG_POSITIONS_B:
                    counts["uwb_nonconfigured_tag_messages"] += 1
                    continue
                counts["uwb_configured_tag_messages"] += 1
                uwb_messages.append((receive_time, message))
    finally:
        bag.close()

    expected_types = {
        args.uwb_topic: "nlink_parser/LinktrackNodeframe2",
        args.mocap_topic: "geometry_msgs/PoseStamped",
    }
    for topic, expected in expected_types.items():
        if observed_types.get(topic) != {expected}:
            raise BagBiasError(
                "topic {} must have type {}, observed {}".format(
                    topic, expected, sorted(observed_types.get(topic, set()))
                )
            )
    if len(poses) < 2:
        raise BagBiasError("fewer than two valid mocap samples")
    pose_times = [pose.timestamp for pose in poses]
    if any(current <= previous for previous, current in zip(pose_times, pose_times[1:])):
        raise BagBiasError("mocap header timestamps are not strictly increasing")

    rows: List[Dict[str, object]] = []
    sync_gaps = []
    bracket_spans = []
    for receive_time, message in uwb_messages:
        sync = synchronize_pose(poses, receive_time, args.max_sync_gap_s)
        if not sync.is_valid:
            counts["uwb_sync_rejected:" + sync.invalid_reason] += 1
            continue
        bracket_span = float(sync.timestamp_after - sync.timestamp_before)
        if bracket_span > args.max_mocap_bracket_s:
            counts["uwb_sync_rejected:mocap_bracket_span_exceeded"] += 1
            continue

        rotation_m_bm = quaternion_to_matrix(sync.quaternion)
        rotation_a_b = R_AM_A.T.dot(rotation_m_bm).dot(R_BM_B)
        position_a_b = R_AM_A.T.dot(np.asarray(sync.position))
        quaternion_a_b = rotation_matrix_to_quaternion(rotation_a_b)
        sync_gaps.append(float(sync.sync_gap_s))
        bracket_spans.append(bracket_span)

        configured_nodes = set()
        for node in message.nodes:
            anchor_id = int(node.id)
            if anchor_id not in ANCHOR_POSITIONS_A:
                counts["ignored_nonanchor_nodes"] += 1
                continue
            configured_nodes.add(anchor_id)
            measured_range = float(node.dis)
            if not math.isfinite(measured_range) or measured_range <= 0.0:
                counts["invalid_ranges"] += 1
                continue
            tag_id = int(message.id)
            tag_a = position_a_b + rotation_a_b.dot(TAG_POSITIONS_B[tag_id])
            anchor_a = ANCHOR_POSITIONS_A[anchor_id]
            true_range = float(np.linalg.norm(tag_a - anchor_a))
            row = {
                "bag_timestamp_s": receive_time,
                "mocap_timestamp_before_s": sync.timestamp_before,
                "mocap_timestamp_after_s": sync.timestamp_after,
                "mocap_interpolation_alpha": sync.interpolation_ratio,
                "mocap_sync_gap_s": sync.sync_gap_s,
                "mocap_bracket_span_s": bracket_span,
                "tag_id": tag_id,
                "anchor_id": anchor_id,
                "measured_range_m": measured_range,
                "true_range_m": true_range,
                "bias_m": measured_range - true_range,
                "fp_rssi": float(node.fp_rssi),
                "rx_rssi": float(node.rx_rssi),
                "p_A_B_x_m": float(position_a_b[0]),
                "p_A_B_y_m": float(position_a_b[1]),
                "p_A_B_z_m": float(position_a_b[2]),
                "q_A_B_x": float(quaternion_a_b[0]),
                "q_A_B_y": float(quaternion_a_b[1]),
                "q_A_B_z": float(quaternion_a_b[2]),
                "q_A_B_w": float(quaternion_a_b[3]),
                "tag_x_A_m": float(tag_a[0]),
                "tag_y_A_m": float(tag_a[1]),
                "tag_z_A_m": float(tag_a[2]),
                "anchor_x_A_m": float(anchor_a[0]),
                "anchor_y_A_m": float(anchor_a[1]),
                "anchor_z_A_m": float(anchor_a[2]),
            }
            rows.append(row)
        counts["missing_configured_anchor_ranges"] += len(ANCHOR_POSITIONS_A) - len(
            configured_nodes
        )

    if not rows:
        raise BagBiasError("no synchronized valid UWB ranges were reconstructed")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    biases = [float(row["bias_m"]) for row in rows]
    measured = [float(row["measured_range_m"]) for row in rows]
    truth = [float(row["true_range_m"]) for row in rows]
    envelope = np.asarray(mocap_envelope_delays, dtype=float)
    summary: Dict[str, object] = {
        "schema_version": 1,
        "bag": str(bag_path),
        "output_csv": str(csv_path),
        "bias_definition": "bias_m = measured_range_m - true_range_m",
        "truth_model": "rho_ij = norm(p_A_B + R_A_B * b_j - a_i)",
        "geometry_source": "src/uwb_pose_estimation/src/uwb_imu_tof_eskf_real_node.cpp",
        "timestamp_policy": "UWB bag receive time, mocap linear position interpolation and quaternion SLERP",
        "max_sync_gap_s": float(args.max_sync_gap_s),
        "max_mocap_bracket_s": float(args.max_mocap_bracket_s),
        "counts": dict(sorted(counts.items())),
        "used_range_count": len(rows),
        "measured_range_min_m": float(np.min(measured)),
        "measured_range_max_m": float(np.max(measured)),
        "true_range_min_m": float(np.min(truth)),
        "true_range_max_m": float(np.max(truth)),
        "bias_overall": _compact_stats(biases),
        "by_tag_anchor": _group_stats(rows, ("tag_id", "anchor_id")),
        "by_tag": _group_stats(rows, ("tag_id",)),
        "by_anchor": _group_stats(rows, ("anchor_id",)),
        "synchronization": {
            "sync_gap_mean_s": float(np.mean(sync_gaps)),
            "sync_gap_max_s": float(np.max(sync_gaps)),
            "mocap_bracket_mean_s": float(np.mean(bracket_spans)),
            "mocap_bracket_max_s": float(np.max(bracket_spans)),
            "mocap_bag_minus_header_mean_s": float(np.mean(envelope)),
            "mocap_bag_minus_header_abs_max_s": float(np.max(np.abs(envelope))),
        },
        "tag_positions_B_m": {
            str(key): [float(value) for value in point]
            for key, point in sorted(TAG_POSITIONS_B.items())
        },
        "anchor_positions_A_m": {
            str(key): [float(value) for value in point]
            for key, point in sorted(ANCHOR_POSITIONS_A.items())
        },
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = analyze(args)
    except (BagBiasError, OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_csv": summary["output_csv"],
                "summary_json": summary["summary_json"],
                "used_range_count": summary["used_range_count"],
                "bias_overall": summary["bias_overall"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
