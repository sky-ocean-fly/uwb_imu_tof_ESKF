"""ROS-independent primitives shared by live capture and rosbag replay.

This module deliberately contains no ``rospy`` or ``rosbag`` imports.  It
turns confirmed, directed UWB observations and timestamped mocap samples into
auditable raw CSV rows.  The live and offline tools therefore use exactly the
same range validation, interpolation, geometry, and static-window checks.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import ConfigError, open_exclusive, require_values
from .quality import PoseStabilityMetrics, assess_pose_stability, validate_range_m
from .ros_adapters import UwbObservation
from .synchronization import PoseSyncResult, TimedPose, synchronize_pose, synchronize_pose_pair
from .transforms import quaternion_to_matrix, relative_pose


CAPTURE_STATES = ("SETTLING", "COLLECTING", "PAUSED", "COMPLETE", "ABORTED")
ACTIVE_CAPTURE_STATES = ("SETTLING", "COLLECTING", "PAUSED")


# The first group follows section 7 of the task specification.  Additional
# columns retain interface direction, interpolation audit data, relative pose,
# and the three independent static-window assessments.
RAW_CSV_COLUMNS = (
    "schema_version",
    "data_origin",
    "is_synthetic",
    "ingestion_path",
    "session_id",
    "environment_id",
    "pose_id",
    "attempt_id",
    "sample_index",
    "capture_state",
    "uwb_timestamp",
    "uwb_receive_timestamp",
    "uwb_timestamp_source",
    "range_m",
    "anchor_id",
    "node_id",
    "uwb_source_id",
    "uwb_peer_id",
    "source_id",
    "peer_id",
    "source_role",
    "peer_role",
    "local_time_raw",
    "system_time_raw",
    "fp_rssi",
    "rx_rssi",
    "cir_quality",
    "mocap_a_frame_id",
    "mocap_b_frame_id",
    "mocap_a_timestamp_before",
    "mocap_a_timestamp_after",
    "mocap_b_timestamp_before",
    "mocap_b_timestamp_after",
    "mocap_a_interpolation_alpha",
    "mocap_b_interpolation_alpha",
    "mocap_a_gap_before_s",
    "mocap_a_gap_after_s",
    "mocap_b_gap_before_s",
    "mocap_b_gap_after_s",
    "sync_gap_a_s",
    "sync_gap_b_s",
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
    "distance_gt_m",
    "range_error_m",
    "stability_a_position_m",
    "stability_a_rotation_deg",
    "stability_b_position_m",
    "stability_b_rotation_deg",
    "stability_relative_position_m",
    "stability_relative_rotation_deg",
    "is_stable",
    "is_los",
    "is_mocap_valid",
    "is_sync_valid",
    "is_range_valid",
    "is_valid",
    "invalid_reason",
    "operator_note",
)


class CaptureError(ValueError):
    """Raised when capture input/configuration cannot be used safely."""


@dataclass(frozen=True)
class CaptureThresholds:
    stability_window_s: float
    max_position_std_m: float
    max_rotation_std_deg: float
    max_sync_gap_s: float
    max_mocap_sample_gap_s: float
    range_min_m: Optional[float]
    range_max_m: Optional[float]
    reject_nonpositive: bool

    @classmethod
    def from_collection(cls, collection: Mapping[str, Any]) -> "CaptureThresholds":
        range_config = collection.get("range_validation", {})
        if not isinstance(range_config, Mapping):
            raise CaptureError("range_validation must be a mapping")
        maximum_sample_gap = _optional_float(collection.get("max_mocap_sample_gap_s"))
        if maximum_sample_gap is None:
            raise CaptureError(
                "max_mocap_sample_gap_s is unresolved; confirm the real mocap frequency "
                "and continuity tolerance before capture"
            )
        result = cls(
            stability_window_s=float(collection["stability_window_s"]),
            max_position_std_m=float(collection["max_position_std_m"]),
            max_rotation_std_deg=float(collection["max_rotation_std_deg"]),
            max_sync_gap_s=float(collection["max_sync_gap_s"]),
            max_mocap_sample_gap_s=maximum_sample_gap,
            range_min_m=_optional_float(range_config.get("min_m")),
            range_max_m=_optional_float(range_config.get("max_m")),
            reject_nonpositive=bool(range_config.get("reject_nonpositive", True)),
        )
        values = (
            result.stability_window_s,
            result.max_position_std_m,
            result.max_rotation_std_deg,
            result.max_sync_gap_s,
            result.max_mocap_sample_gap_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise CaptureError("capture thresholds must be finite")
        if result.stability_window_s <= 0.0:
            raise CaptureError("stability_window_s must be positive")
        if min(values[1:]) < 0.0:
            raise CaptureError("stability and synchronization thresholds cannot be negative")
        if result.max_mocap_sample_gap_s <= 0.0:
            raise CaptureError("max_mocap_sample_gap_s must be positive")
        if (
            result.range_min_m is not None
            and result.range_max_m is not None
            and result.range_min_m > result.range_max_m
        ):
            raise CaptureError("range_validation.min_m must not exceed max_m")
        return result


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise CaptureError("optional numeric configuration values must be finite")
    return converted


def validate_capture_interfaces(collection: Mapping[str, Any]) -> None:
    """Apply capture-specific fail-closed checks after generic config loading.

    Expected rates are informational and may remain unknown.  Every interface
    value used to interpret or publish data must be explicit, however.
    """

    topics = collection.get("ros_topics")
    if not isinstance(topics, Mapping):
        raise CaptureError("collection config has no resolved ros_topics mapping")
    try:
        require_values(
            topics,
            (
                "uwb.topic",
                "uwb.message_type",
                "uwb.adapter",
                "uwb.source_id",
                "uwb.peer_id",
                "uwb.direction",
                "uwb.timestamp_source",
                "mocap_a.topic",
                "mocap_a.message_type",
                "mocap_a.expected_frame_id",
                "mocap_b.topic",
                "mocap_b.message_type",
                "mocap_b.expected_frame_id",
                "status.topic",
                "status.message_type",
            ),
        )
    except ConfigError as exc:
        raise CaptureError(str(exc))

    if topics["uwb"]["direction"] != "source_to_peer":
        raise CaptureError("only exact directed source_id -> peer_id extraction is supported")
    if topics["uwb"]["timestamp_source"] == "mapped_device_time":
        raise CaptureError(
            "mapped_device_time is configured, but no explicit clock mapping is implemented; "
            "refusing to substitute receive time"
        )
    if topics["uwb"]["timestamp_source"] != "receive_time":
        raise CaptureError("unsupported UWB timestamp source")
    if int(topics["uwb"]["source_id"]) == int(topics["uwb"]["peer_id"]):
        raise CaptureError("uwb.source_id and uwb.peer_id must differ")
    if bool(topics.get("time", {}).get("require_same_mocap_frame", True)):
        if topics["mocap_a"]["expected_frame_id"] != topics["mocap_b"]["expected_frame_id"]:
            raise CaptureError("configured mocap frame IDs differ while same-frame checking is enabled")
    input_topics = [topics[name]["topic"] for name in ("uwb", "mocap_a", "mocap_b")]
    if len(set(input_topics)) != len(input_topics):
        raise CaptureError("UWB and mocap input topics must be distinct")
    if topics["status"]["topic"] in input_topics:
        raise CaptureError("status topic must not collide with an input topic")


@dataclass(frozen=True)
class StaticWindowAssessment:
    is_stable: bool
    invalid_reason: str
    mocap_a: Optional[PoseStabilityMetrics]
    mocap_b: Optional[PoseStabilityMetrics]
    relative: Optional[PoseStabilityMetrics]


def _window_samples(
    samples: Sequence[TimedPose],
    target_timestamp: float,
    window_s: float,
    max_gap_s: float,
    max_sample_gap_s: float,
    stream_name: str,
) -> Tuple[Sequence[TimedPose], str]:
    start = target_timestamp - window_s
    start_sync = synchronize_pose(samples, start, max_gap_s)
    if not start_sync.is_valid:
        return [], "%s:stability_window_start:%s" % (stream_name, start_sync.invalid_reason)
    end_sync = synchronize_pose(samples, target_timestamp, max_gap_s)
    if not end_sync.is_valid:
        return [], "%s:stability_window_end:%s" % (stream_name, end_sync.invalid_reason)

    # Interpolated boundary samples make the requested time span explicit and
    # avoid assuming a device frequency.  Original samples strictly inside the
    # window are retained for the dispersion calculation.
    selected = [
        TimedPose(start, start_sync.position, start_sync.quaternion),
        *[sample for sample in samples if start < sample.timestamp < target_timestamp],
        TimedPose(target_timestamp, end_sync.position, end_sync.quaternion),
    ]
    for previous, current in zip(selected, selected[1:]):
        if current.timestamp - previous.timestamp > max_sample_gap_s:
            return [], "%s:stability_window_internal_gap_exceeded" % stream_name
    return selected, ""


def assess_static_window(
    samples_a: Sequence[TimedPose],
    samples_b: Sequence[TimedPose],
    target_timestamp: float,
    thresholds: CaptureThresholds,
) -> StaticWindowAssessment:
    """Check absolute A/B and relative ``T_A_B`` stability in one time window."""

    target = float(target_timestamp)
    selected_a, reason_a = _window_samples(
        samples_a,
        target,
        thresholds.stability_window_s,
        thresholds.max_sync_gap_s,
        thresholds.max_mocap_sample_gap_s,
        "mocap_a",
    )
    selected_b, reason_b = _window_samples(
        samples_b,
        target,
        thresholds.stability_window_s,
        thresholds.max_sync_gap_s,
        thresholds.max_mocap_sample_gap_s,
        "mocap_b",
    )
    initial_reasons = [reason for reason in (reason_a, reason_b) if reason]
    if initial_reasons:
        return StaticWindowAssessment(False, ";".join(initial_reasons), None, None, None)

    assessment_a = assess_pose_stability(
        [sample.position for sample in selected_a],
        [sample.quaternion for sample in selected_a],
        thresholds.max_position_std_m,
        thresholds.max_rotation_std_deg,
    )
    assessment_b = assess_pose_stability(
        [sample.position for sample in selected_b],
        [sample.quaternion for sample in selected_b],
        thresholds.max_position_std_m,
        thresholds.max_rotation_std_deg,
    )

    relative_positions = []
    relative_quaternions = []
    relative_timestamps = []
    relative_sync_reasons = []
    for sample_a in selected_a:
        synced_b = synchronize_pose(samples_b, sample_a.timestamp, thresholds.max_sync_gap_s)
        if not synced_b.is_valid:
            relative_sync_reasons.append(synced_b.invalid_reason)
            continue
        translation, quaternion = relative_pose(
            sample_a.position,
            sample_a.quaternion,
            synced_b.position,
            synced_b.quaternion,
        )
        relative_positions.append(translation)
        relative_quaternions.append(quaternion)
        relative_timestamps.append(sample_a.timestamp)

    relative_metrics = None
    relative_assessment = None
    if len(relative_positions) < 2:
        relative_reason = "relative:stability_window_has_fewer_than_two_synchronized_samples"
    elif relative_timestamps[0] > target - thresholds.stability_window_s + thresholds.max_sync_gap_s:
        relative_reason = "relative:stability_window_start_not_covered"
    elif target - relative_timestamps[-1] > thresholds.max_sync_gap_s:
        relative_reason = "relative:stability_window_end_not_covered"
    elif relative_sync_reasons:
        relative_reason = "relative:sync_invalid:" + ",".join(sorted(set(relative_sync_reasons)))
    else:
        relative_assessment = assess_pose_stability(
            relative_positions,
            relative_quaternions,
            thresholds.max_position_std_m,
            thresholds.max_rotation_std_deg,
        )
        relative_metrics = relative_assessment.metrics
        relative_reason = ""

    reasons = []
    if not assessment_a.is_stable:
        reasons.append("mocap_a:" + assessment_a.invalid_reason)
    if not assessment_b.is_stable:
        reasons.append("mocap_b:" + assessment_b.invalid_reason)
    if relative_reason:
        reasons.append(relative_reason)
    elif relative_assessment is not None and not relative_assessment.is_stable:
        reasons.append("relative:" + relative_assessment.invalid_reason)

    return StaticWindowAssessment(
        is_stable=not reasons,
        invalid_reason=";".join(reasons),
        mocap_a=assessment_a.metrics,
        mocap_b=assessment_b.metrics,
        relative=relative_metrics,
    )


@dataclass(frozen=True)
class ProcessedMeasurement:
    row: Dict[str, Any]
    is_valid: bool
    stability: StaticWindowAssessment
    sync_a: PoseSyncResult
    sync_b: PoseSyncResult


def _set_pose_fields(row: Dict[str, Any], prefix: str, sync: PoseSyncResult) -> None:
    row["%s_timestamp_before" % prefix] = sync.timestamp_before
    row["%s_timestamp_after" % prefix] = sync.timestamp_after
    row["%s_interpolation_alpha" % prefix] = sync.interpolation_ratio
    row["%s_gap_before_s" % prefix] = sync.gap_before_s
    row["%s_gap_after_s" % prefix] = sync.gap_after_s


def _append_reason(reasons: list, reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def process_uwb_observation(
    observation: UwbObservation,
    samples_a: Sequence[TimedPose],
    samples_b: Sequence[TimedPose],
    thresholds: CaptureThresholds,
    *,
    session_id: str,
    environment_id: str,
    pose_id: str,
    attempt_id: str,
    sample_index: int,
    capture_state: str,
    mocap_a_frame_id: str,
    mocap_b_frame_id: str,
    is_los: bool,
    data_origin: str,
    is_synthetic: bool,
    ingestion_path: str,
    operator_note: str = "",
    upstream_invalid_reason: str = "",
) -> ProcessedMeasurement:
    """Build one immutable-schema row for one extracted target-pair range."""

    if capture_state not in ACTIVE_CAPTURE_STATES:
        raise CaptureError("measurement rows require an active capture state")
    if not str(data_origin).strip():
        raise CaptureError("data_origin must be explicit and non-empty")
    if not str(ingestion_path).strip():
        raise CaptureError("ingestion_path must be explicit and non-empty")
    if not str(environment_id).strip():
        raise CaptureError("environment_id must be explicit and non-empty")
    pair = synchronize_pose_pair(
        samples_a,
        samples_b,
        observation.timestamp,
        thresholds.max_sync_gap_s,
    )
    stability = assess_static_window(
        samples_a,
        samples_b,
        observation.timestamp,
        thresholds,
    )
    range_result = validate_range_m(
        observation.range_m,
        min_m=thresholds.range_min_m,
        max_m=thresholds.range_max_m,
        reject_nonpositive=thresholds.reject_nonpositive,
    )
    los_valid = bool(is_los)

    row: Dict[str, Any] = {name: "" for name in RAW_CSV_COLUMNS}
    row.update(
        {
            "schema_version": 1,
            "data_origin": str(data_origin),
            "is_synthetic": int(bool(is_synthetic)),
            "ingestion_path": str(ingestion_path),
            "session_id": session_id,
            "environment_id": str(environment_id),
            "pose_id": pose_id,
            "attempt_id": attempt_id,
            "sample_index": int(sample_index),
            "capture_state": capture_state,
            "uwb_timestamp": observation.timestamp,
            "uwb_receive_timestamp": observation.receive_timestamp,
            "uwb_timestamp_source": observation.timestamp_source,
            "range_m": observation.range_m,
            # Endpoint semantics are intentionally not inferred from NodeFrame
            # roles.  source/peer retain the exact configured direction.
            "anchor_id": "",
            "node_id": "",
            "uwb_source_id": observation.source_id,
            "uwb_peer_id": observation.peer_id,
            "source_id": observation.source_id,
            "peer_id": observation.peer_id,
            "source_role": observation.source_role,
            "peer_role": observation.peer_role,
            "local_time_raw": observation.local_time_raw,
            "system_time_raw": observation.system_time_raw,
            "fp_rssi": observation.fp_rssi,
            "rx_rssi": observation.rx_rssi,
            # This confirmed adapter exposes no CIR field; missing quality is
            # empty, never synthesized from another quantity.
            "cir_quality": "",
            "mocap_a_frame_id": mocap_a_frame_id,
            "mocap_b_frame_id": mocap_b_frame_id,
            "sync_gap_a_s": pair.pose_a.sync_gap_s,
            "sync_gap_b_s": pair.pose_b.sync_gap_s,
            "is_stable": int(stability.is_stable),
            "is_los": int(los_valid),
            "is_mocap_valid": int(pair.is_valid),
            "is_sync_valid": int(pair.is_valid),
            "is_range_valid": int(range_result.is_valid),
            "operator_note": operator_note,
        }
    )
    _set_pose_fields(row, "mocap_a", pair.pose_a)
    _set_pose_fields(row, "mocap_b", pair.pose_b)

    if stability.mocap_a is not None:
        row["stability_a_position_m"] = stability.mocap_a.position_stability_m
        row["stability_a_rotation_deg"] = stability.mocap_a.rotation_stability_deg
    if stability.mocap_b is not None:
        row["stability_b_position_m"] = stability.mocap_b.position_stability_m
        row["stability_b_rotation_deg"] = stability.mocap_b.rotation_stability_deg
    if stability.relative is not None:
        row["stability_relative_position_m"] = stability.relative.position_stability_m
        row["stability_relative_rotation_deg"] = stability.relative.rotation_stability_deg

    if pair.is_valid:
        pose_values = (
            ("A", pair.pose_a.position, pair.pose_a.quaternion),
            ("B", pair.pose_b.position, pair.pose_b.quaternion),
        )
        for name, position, quaternion in pose_values:
            for axis, value in zip(("px", "py", "pz"), position):
                row["%s_%s" % (name, axis)] = value
            for axis, value in zip(("qx", "qy", "qz", "qw"), quaternion):
                row["%s_%s" % (name, axis)] = value

        translation, quaternion = relative_pose(
            pair.pose_a.position,
            pair.pose_a.quaternion,
            pair.pose_b.position,
            pair.pose_b.quaternion,
        )
        rotation = quaternion_to_matrix(quaternion)
        for axis, value in zip(("tx", "ty", "tz"), translation):
            row["rel_%s" % axis] = value
        for axis, value in zip(("qx", "qy", "qz", "qw"), quaternion):
            row["rel_%s" % axis] = value
        for i in range(3):
            for j in range(3):
                row["rel_R%d%d" % (i, j)] = rotation[i, j]
        distance_gt = float(np.linalg.norm(translation))
        row["distance_gt_m"] = distance_gt
        if range_result.is_valid:
            row["range_error_m"] = float(observation.range_m - distance_gt)

    reasons = []
    if capture_state != "COLLECTING":
        _append_reason(reasons, "capture_state:" + capture_state.lower())
    if not pair.is_valid:
        _append_reason(reasons, pair.invalid_reason)
    if not stability.is_stable:
        _append_reason(reasons, stability.invalid_reason)
    if not range_result.is_valid:
        _append_reason(reasons, range_result.invalid_reason)
    if not los_valid:
        _append_reason(reasons, "not_los")
    if upstream_invalid_reason:
        _append_reason(reasons, str(upstream_invalid_reason))
    is_valid = (
        capture_state == "COLLECTING"
        and pair.is_valid
        and stability.is_stable
        and range_result.is_valid
        and los_valid
        and not upstream_invalid_reason
    )
    row["is_valid"] = int(is_valid)
    row["invalid_reason"] = ";".join(reasons)
    return ProcessedMeasurement(row, is_valid, stability, pair.pose_a, pair.pose_b)


def _csv_value(value: Any, precision: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return format(float(value), ".%dg" % int(precision))
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


class ExclusiveRawCsvWriter:
    """Write raw rows immediately, refusing an existing destination."""

    def __init__(self, path: os.PathLike, float_precision: int = 17):
        self.path = Path(path).expanduser().resolve()
        self.float_precision = int(float_precision)
        if self.float_precision < 6 or self.float_precision > 17:
            raise CaptureError("float_precision must be within [6, 17]")
        self._stream = open_exclusive(self.path, "x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=RAW_CSV_COLUMNS, extrasaction="raise")
        self._writer.writeheader()
        self._stream.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        missing = [name for name in RAW_CSV_COLUMNS if name not in row]
        if missing:
            raise CaptureError("raw row is missing columns: %s" % ", ".join(missing))
        converted = {name: _csv_value(row[name], self.float_precision) for name in RAW_CSV_COLUMNS}
        self._writer.writerow(converted)
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "ExclusiveRawCsvWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class CaptureSummary:
    """Accumulate only observed row values for a quick, non-inferential report."""

    def __init__(self) -> None:
        self.total_rows = 0
        self.valid_rows = 0
        self.invalid_reasons: Counter = Counter()
        self.valid_ranges = []
        self.valid_truth = []
        self.valid_errors = []
        self.stability = {name: [] for name in (
            "stability_a_position_m",
            "stability_a_rotation_deg",
            "stability_b_position_m",
            "stability_b_rotation_deg",
            "stability_relative_position_m",
            "stability_relative_rotation_deg",
        )}

    def add(self, processed: ProcessedMeasurement) -> None:
        row = processed.row
        self.total_rows += 1
        if processed.is_valid:
            self.valid_rows += 1
            self.valid_ranges.append(float(row["range_m"]))
            self.valid_truth.append(float(row["distance_gt_m"]))
            self.valid_errors.append(float(row["range_error_m"]))
        else:
            reason = str(row.get("invalid_reason", "unspecified")) or "unspecified"
            self.invalid_reasons[reason] += 1
        for name, values in self.stability.items():
            value = row.get(name, "")
            if value not in (None, ""):
                values.append(float(value))

    @staticmethod
    def _mean(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    @staticmethod
    def _population_std(values: Sequence[float]) -> Optional[float]:
        return float(np.std(values, ddof=0)) if values else None

    def payload(self) -> Dict[str, Any]:
        return {
            "total_target_pair_rows": self.total_rows,
            "valid_range_count": self.valid_rows,
            "invalid_reason_counts": dict(sorted(self.invalid_reasons.items())),
            "valid_range_mean_m": self._mean(self.valid_ranges),
            "valid_range_std_m_population": self._population_std(self.valid_ranges),
            "valid_distance_gt_mean_m": self._mean(self.valid_truth),
            "valid_range_error_mean_m": self._mean(self.valid_errors),
            "valid_range_error_std_m_population": self._population_std(self.valid_errors),
            "stability_observed_maxima": {
                name: (max(values) if values else None) for name, values in self.stability.items()
            },
        }


@dataclass(frozen=True)
class StatusAssignment:
    session_id: str
    pose_id: str
    attempt_id: str
    state: str
    operator_note: str


class StatusAssignmentTracker:
    """Assign bag messages to the latest validated capture-state event."""

    def __init__(self, expected_session_id: str):
        self.expected_session_id = str(expected_session_id)
        self.active: Optional[StatusAssignment] = None
        self.event_count = 0
        self.ignored_other_session = 0
        self.invalid_event_count = 0
        self.terminal_outcomes: Dict[str, str] = {}
        self.invalid_attempt_reasons: Dict[str, str] = {}

    def _invalidate_attempt(self, attempt_id: str, reason: str) -> None:
        if attempt_id:
            self.invalid_attempt_reasons.setdefault(str(attempt_id), str(reason))
        self.invalid_event_count += 1

    def feed(self, raw_payload: Any) -> Optional[StatusAssignment]:
        try:
            payload = json.loads(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
            session_id = str(payload["session_id"])
            state = str(payload["state"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.invalid_event_count += 1
            return self.active
        if session_id != self.expected_session_id:
            self.ignored_other_session += 1
            return self.active
        if state not in CAPTURE_STATES:
            self.invalid_event_count += 1
            return self.active
        if state in ACTIVE_CAPTURE_STATES:
            pose_id = payload.get("pose_id")
            attempt_id = payload.get("attempt_id")
            if pose_id in (None, "") or attempt_id in (None, ""):
                self.invalid_event_count += 1
                return self.active
            pose_id = str(pose_id)
            attempt_id = str(attempt_id)
            if attempt_id in self.terminal_outcomes or attempt_id in self.invalid_attempt_reasons:
                self._invalidate_attempt(attempt_id, "active_event_after_terminal_or_conflict")
                return self.active
            if self.active is None:
                if state != "SETTLING":
                    self._invalidate_attempt(attempt_id, "attempt_did_not_start_with_settling")
                    return self.active
            elif self.active.pose_id != pose_id or self.active.attempt_id != attempt_id:
                self._invalidate_attempt(self.active.attempt_id, "attempt_replaced_without_terminal")
                self._invalidate_attempt(attempt_id, "attempt_started_while_another_attempt_active")
                return self.active
            else:
                allowed = {
                    "SETTLING": {"SETTLING", "COLLECTING", "PAUSED"},
                    "COLLECTING": {"COLLECTING", "PAUSED"},
                    "PAUSED": {"PAUSED", "COLLECTING"},
                }
                if state not in allowed[self.active.state]:
                    self._invalidate_attempt(attempt_id, "illegal_state_transition")
                    return self.active
            self.active = StatusAssignment(
                session_id=session_id,
                pose_id=pose_id,
                attempt_id=attempt_id,
                state=state,
                operator_note=str(payload.get("operator_note", "")),
            )
            self.event_count += 1
        else:
            pose_id = payload.get("pose_id")
            attempt_id = payload.get("attempt_id")
            if pose_id in (None, "") or attempt_id in (None, ""):
                self.invalid_event_count += 1
                return self.active
            pose_id = str(pose_id)
            attempt_id = str(attempt_id)
            if attempt_id in self.terminal_outcomes:
                self._invalidate_attempt(attempt_id, "duplicate_or_conflicting_terminal")
                return self.active
            if self.active is None:
                self._invalidate_attempt(attempt_id, "terminal_without_active_attempt")
                return self.active
            if self.active.pose_id != pose_id or self.active.attempt_id != attempt_id:
                self._invalidate_attempt(attempt_id, "terminal_does_not_match_active_attempt")
                return self.active
            self.terminal_outcomes[attempt_id] = state
            self.active = None
            self.event_count += 1
        return self.active


__all__ = [
    "ACTIVE_CAPTURE_STATES",
    "CAPTURE_STATES",
    "RAW_CSV_COLUMNS",
    "CaptureError",
    "CaptureSummary",
    "CaptureThresholds",
    "ExclusiveRawCsvWriter",
    "ProcessedMeasurement",
    "StaticWindowAssessment",
    "StatusAssignment",
    "StatusAssignmentTracker",
    "assess_static_window",
    "process_uwb_observation",
    "validate_capture_interfaces",
]
