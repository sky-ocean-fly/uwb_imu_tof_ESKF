"""Timestamp matching and pose interpolation without ROS dependencies.

Positions are interpolated linearly.  Orientations use shortest-path SLERP in
external ROS quaternion order ``qx,qy,qz,qw``.  The synchronization threshold
is applied to the nearest mocap timestamp, as specified by the task document;
extrapolation outside the available pose buffer is never performed.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .transforms import normalize_quaternion


def _finite_timestamp(value: float, name: str) -> float:
    timestamp = float(value)
    if not np.isfinite(timestamp):
        raise ValueError("{} must be finite".format(name))
    return timestamp


def _position(value: Sequence[float], name: str = "position") -> np.ndarray:
    position = np.asarray(value, dtype=float)
    if position.shape != (3,):
        raise ValueError("{} must have shape (3,), got {}".format(name, position.shape))
    if not np.all(np.isfinite(position)):
        raise ValueError("{} must contain only finite values".format(name))
    return position


@dataclass(frozen=True)
class TimedPose:
    """One timestamped pose sample independent of any ROS message type."""

    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self) -> None:
        timestamp = _finite_timestamp(self.timestamp, "timestamp")
        position = _position(self.position).copy()
        quaternion = normalize_quaternion(self.quaternion).copy()
        # Frozen dataclass plus read-only arrays prevents accidental mutation of
        # a history buffer after its ordering has been validated.
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)


PoseSample = TimedPose


@dataclass(frozen=True)
class PoseSyncResult:
    """Auditable outcome of matching one target timestamp to one pose stream."""

    is_valid: bool
    invalid_reason: str
    target_timestamp: float
    position: Optional[np.ndarray]
    quaternion: Optional[np.ndarray]
    timestamp_before: Optional[float]
    timestamp_after: Optional[float]
    interpolation_ratio: Optional[float]
    gap_before_s: Optional[float]
    gap_after_s: Optional[float]
    sync_gap_s: Optional[float]

    @property
    def alpha(self) -> Optional[float]:
        """Alias for ``interpolation_ratio`` used in dataset audit columns."""

        return self.interpolation_ratio


SynchronizedPose = PoseSyncResult


@dataclass(frozen=True)
class PosePairSyncResult:
    """Synchronization results for mocap A and B at one UWB timestamp."""

    pose_a: PoseSyncResult
    pose_b: PoseSyncResult

    @property
    def is_valid(self) -> bool:
        return self.pose_a.is_valid and self.pose_b.is_valid

    @property
    def invalid_reason(self) -> str:
        reasons = []
        if not self.pose_a.is_valid:
            reasons.append("mocap_a:" + self.pose_a.invalid_reason)
        if not self.pose_b.is_valid:
            reasons.append("mocap_b:" + self.pose_b.invalid_reason)
        return ";".join(reasons)


def linear_interpolate_position(
    position_before: Sequence[float], position_after: Sequence[float], alpha: float
) -> np.ndarray:
    """Linearly interpolate two positions for ``alpha`` in ``[0, 1]``."""

    ratio = float(alpha)
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")
    before = _position(position_before, "position_before")
    after = _position(position_after, "position_after")
    return (1.0 - ratio) * before + ratio * after


def quaternion_slerp(
    quaternion_before: Sequence[float], quaternion_after: Sequence[float], alpha: float
) -> np.ndarray:
    """Shortest-path spherical interpolation of two quaternions.

    The near-parallel branch uses normalized linear interpolation to avoid the
    small denominator in the trigonometric formula.  Antipodal inputs are
    sign-aligned first, so ``q`` to ``-q`` remains the same physical rotation.
    """

    ratio = float(alpha)
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise ValueError("alpha must be finite and within [0, 1]")

    before = normalize_quaternion(quaternion_before)
    after = normalize_quaternion(quaternion_after)
    dot = float(np.dot(before, after))
    if dot < 0.0:
        after = -after
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        return normalize_quaternion((1.0 - ratio) * before + ratio * after)

    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    weight_before = np.sin((1.0 - ratio) * theta) / sin_theta
    weight_after = np.sin(ratio * theta) / sin_theta
    return normalize_quaternion(weight_before * before + weight_after * after)


slerp = quaternion_slerp


def _valid_result(
    target_timestamp: float,
    before: TimedPose,
    after: TimedPose,
    position: np.ndarray,
    quaternion: np.ndarray,
    alpha: float,
) -> PoseSyncResult:
    gap_before = max(0.0, target_timestamp - before.timestamp)
    gap_after = max(0.0, after.timestamp - target_timestamp)
    position = np.asarray(position, dtype=float).copy()
    quaternion = normalize_quaternion(quaternion).copy()
    position.setflags(write=False)
    quaternion.setflags(write=False)
    return PoseSyncResult(
        is_valid=True,
        invalid_reason="",
        target_timestamp=target_timestamp,
        position=position,
        quaternion=quaternion,
        timestamp_before=before.timestamp,
        timestamp_after=after.timestamp,
        interpolation_ratio=float(alpha),
        gap_before_s=float(gap_before),
        gap_after_s=float(gap_after),
        sync_gap_s=float(min(gap_before, gap_after)),
    )


def _invalid_result(
    target_timestamp: float,
    reason: str,
    before: Optional[TimedPose] = None,
    after: Optional[TimedPose] = None,
) -> PoseSyncResult:
    gap_before = None if before is None else max(0.0, target_timestamp - before.timestamp)
    gap_after = None if after is None else max(0.0, after.timestamp - target_timestamp)
    available_gaps = [gap for gap in (gap_before, gap_after) if gap is not None]
    sync_gap = min(available_gaps) if available_gaps else None
    alpha = None
    if before is not None and after is not None:
        span = after.timestamp - before.timestamp
        if span > 0.0:
            alpha = (target_timestamp - before.timestamp) / span
        elif target_timestamp == before.timestamp:
            alpha = 0.0
    return PoseSyncResult(
        is_valid=False,
        invalid_reason=reason,
        target_timestamp=target_timestamp,
        position=None,
        quaternion=None,
        timestamp_before=None if before is None else before.timestamp,
        timestamp_after=None if after is None else after.timestamp,
        interpolation_ratio=None if alpha is None else float(alpha),
        gap_before_s=None if gap_before is None else float(gap_before),
        gap_after_s=None if gap_after is None else float(gap_after),
        sync_gap_s=None if sync_gap is None else float(sync_gap),
    )


def interpolate_pose(before: TimedPose, after: TimedPose, target_timestamp: float) -> PoseSyncResult:
    """Interpolate a bracketed target time and return all audit metadata.

    This low-level function does not apply a synchronization-gap threshold;
    use :func:`synchronize_pose` for acceptance/rejection.
    """

    target = _finite_timestamp(target_timestamp, "target_timestamp")
    if after.timestamp < before.timestamp:
        raise ValueError("after.timestamp must not be earlier than before.timestamp")
    if target < before.timestamp or target > after.timestamp:
        raise ValueError("target_timestamp must lie inside the pose bracket")

    span = after.timestamp - before.timestamp
    if span == 0.0:
        if target != before.timestamp:
            raise ValueError("equal pose timestamps can only match that exact timestamp")
        # Duplicate samples are rejected by synchronize_pose, but exact reuse
        # remains well-defined here when callers intentionally pass one sample.
        return _valid_result(target, before, after, before.position, before.quaternion, 0.0)

    alpha = (target - before.timestamp) / span
    position = linear_interpolate_position(before.position, after.position, alpha)
    quaternion = quaternion_slerp(before.quaternion, after.quaternion, alpha)
    return _valid_result(target, before, after, position, quaternion, alpha)


def _coerce_timed_pose(value: Any) -> TimedPose:
    if isinstance(value, TimedPose):
        return value
    if isinstance(value, Mapping):
        try:
            return TimedPose(value["timestamp"], value["position"], value["quaternion"])
        except KeyError as error:
            raise ValueError("pose mappings require timestamp, position, and quaternion") from error
    try:
        timestamp, position, quaternion = value
    except (TypeError, ValueError) as error:
        raise ValueError(
            "pose samples must be TimedPose, a matching mapping, or a (timestamp, position, quaternion) tuple"
        ) from error
    return TimedPose(timestamp, position, quaternion)


def synchronize_pose(
    samples: Sequence[Any], target_timestamp: float, max_sync_gap_s: float
) -> PoseSyncResult:
    """Match and interpolate one pose stream at a UWB timestamp.

    Samples must be strictly increasing in timestamp.  A result is invalid if
    interpolation would require extrapolation or if the nearest bracket sample
    is farther than ``max_sync_gap_s`` from the UWB timestamp.
    """

    target = _finite_timestamp(target_timestamp, "target_timestamp")
    threshold = float(max_sync_gap_s)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("max_sync_gap_s must be finite and non-negative")

    poses = [_coerce_timed_pose(sample) for sample in samples]
    if not poses:
        return _invalid_result(target, "no_pose_samples")
    timestamps = [pose.timestamp for pose in poses]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("pose sample timestamps must be strictly increasing")

    insertion_index = bisect_left(timestamps, target)
    if insertion_index < len(poses) and timestamps[insertion_index] == target:
        exact = poses[insertion_index]
        return _valid_result(target, exact, exact, exact.position, exact.quaternion, 0.0)
    if insertion_index == 0:
        return _invalid_result(target, "target_before_pose_buffer", after=poses[0])
    if insertion_index == len(poses):
        return _invalid_result(target, "target_after_pose_buffer", before=poses[-1])

    before = poses[insertion_index - 1]
    after = poses[insertion_index]
    nearest_gap = min(target - before.timestamp, after.timestamp - target)
    if nearest_gap > threshold:
        return _invalid_result(target, "sync_gap_exceeded", before=before, after=after)
    return interpolate_pose(before, after, target)


def synchronize_pose_arrays(
    timestamps: Sequence[float],
    positions: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    target_timestamp: float,
    max_sync_gap_s: float,
) -> PoseSyncResult:
    """Array-oriented convenience wrapper around :func:`synchronize_pose`."""

    if not (len(timestamps) == len(positions) == len(quaternions)):
        raise ValueError("timestamps, positions, and quaternions must have equal lengths")
    samples = [TimedPose(t, p, q) for t, p, q in zip(timestamps, positions, quaternions)]
    return synchronize_pose(samples, target_timestamp, max_sync_gap_s)


def synchronize_pose_pair(
    samples_a: Sequence[Any],
    samples_b: Sequence[Any],
    target_timestamp: float,
    max_sync_gap_s: float,
) -> PosePairSyncResult:
    """Synchronize mocap A and B independently at one UWB timestamp."""

    return PosePairSyncResult(
        pose_a=synchronize_pose(samples_a, target_timestamp, max_sync_gap_s),
        pose_b=synchronize_pose(samples_b, target_timestamp, max_sync_gap_s),
    )


__all__ = [
    "PosePairSyncResult",
    "PoseSample",
    "PoseSyncResult",
    "SynchronizedPose",
    "TimedPose",
    "interpolate_pose",
    "linear_interpolate_position",
    "quaternion_slerp",
    "slerp",
    "synchronize_pose",
    "synchronize_pose_arrays",
    "synchronize_pose_pair",
]
