"""Interface-independent range validity and static-pose quality measures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .transforms import average_quaternions, quaternion_angular_distance


Numeric = Union[float, np.ndarray]


def _finite_numeric(value: Numeric, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _scalar_or_array(value: np.ndarray) -> Numeric:
    if value.ndim == 0:
        return float(value)
    return value


def range_error_m(range_m: Numeric, distance_gt_m: Numeric) -> Numeric:
    """Compute the signed label ``range_m - distance_gt_m`` in metres."""

    measured = _finite_numeric(range_m, "range_m")
    truth = _finite_numeric(distance_gt_m, "distance_gt_m")
    try:
        result = np.subtract(measured, truth)
    except ValueError as error:
        raise ValueError("range_m and distance_gt_m are not broadcast-compatible") from error
    return _scalar_or_array(result)


def correct_range_m(range_m: Numeric, predicted_bias_m: Numeric) -> Numeric:
    """Apply the required correction ``range_m - predicted_bias_m``."""

    measured = _finite_numeric(range_m, "range_m")
    predicted_bias = _finite_numeric(predicted_bias_m, "predicted_bias_m")
    try:
        result = np.subtract(measured, predicted_bias)
    except ValueError as error:
        raise ValueError("range_m and predicted_bias_m are not broadcast-compatible") from error
    return _scalar_or_array(result)


# Explicit aliases keep the sign convention visible at call sites.
compute_range_error_m = range_error_m
apply_bias_correction_m = correct_range_m
correct_range = correct_range_m


@dataclass(frozen=True)
class RangeValidationResult:
    """Validity outcome for one raw UWB range value."""

    is_valid: bool
    invalid_reason: str


def validate_range_m(
    range_m: float,
    min_m: Optional[float] = None,
    max_m: Optional[float] = None,
    reject_nonpositive: bool = True,
) -> RangeValidationResult:
    """Validate one raw range without inventing device-specific sentinels."""

    minimum = None
    if min_m is not None:
        minimum = float(min_m)
        if not np.isfinite(minimum):
            raise ValueError("min_m must be finite when provided")
    maximum = None
    if max_m is not None:
        maximum = float(max_m)
        if not np.isfinite(maximum):
            raise ValueError("max_m must be finite when provided")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("min_m must not exceed max_m")

    value = float(range_m)
    if not np.isfinite(value):
        return RangeValidationResult(False, "range_nonfinite")
    if reject_nonpositive and value <= 0.0:
        return RangeValidationResult(False, "range_nonpositive")
    if minimum is not None and value < minimum:
        return RangeValidationResult(False, "range_below_min")
    if maximum is not None and value > maximum:
        return RangeValidationResult(False, "range_above_max")
    return RangeValidationResult(True, "")


def _positions_array(positions: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("positions must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values")
    return array


def _quaternions_array(quaternions: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(quaternions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError("quaternions must have shape (N, 4) with N >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError("quaternions must contain only finite values")
    return array


def _denominator(sample_count: int, ddof: int) -> int:
    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    denominator = sample_count - int(ddof)
    if denominator <= 0:
        raise ValueError("ddof must be smaller than the sample count")
    return denominator


def position_stability_m(positions: Sequence[Sequence[float]], ddof: int = 0) -> float:
    """Return rotation-invariant RMS position dispersion in metres.

    This is ``sqrt(sum(||p_i - mean(p)||^2) / (N-ddof))``.  It is the single
    scalar compared with ``max_position_std_m``; unlike taking the largest
    coordinate standard deviation, it does not depend on world-axis choice.
    """

    array = _positions_array(positions)
    denominator = _denominator(array.shape[0], ddof)
    residual = array - np.mean(array, axis=0)
    return float(np.sqrt(np.sum(residual * residual) / denominator))


def rotation_stability_rad(
    quaternions: Sequence[Sequence[float]], ddof: int = 0
) -> float:
    """Return RMS geodesic orientation dispersion about a valid mean rotation."""

    array = _quaternions_array(quaternions)
    denominator = _denominator(array.shape[0], ddof)
    mean_quaternion = average_quaternions(array)
    angular_residuals = np.array(
        [quaternion_angular_distance(mean_quaternion, quaternion) for quaternion in array],
        dtype=float,
    )
    return float(np.sqrt(np.sum(angular_residuals * angular_residuals) / denominator))


def rotation_stability_deg(
    quaternions: Sequence[Sequence[float]], ddof: int = 0
) -> float:
    """Return :func:`rotation_stability_rad` converted to degrees."""

    return float(np.degrees(rotation_stability_rad(quaternions, ddof=ddof)))


@dataclass(frozen=True)
class PoseStabilityMetrics:
    """Position and SO(3) stability summary for one static-pose window."""

    sample_count: int
    mean_position: np.ndarray
    mean_quaternion: np.ndarray
    position_stability_m: float
    rotation_stability_rad: float
    rotation_stability_deg: float
    max_position_deviation_m: float
    max_rotation_deviation_deg: float


def compute_pose_stability(
    positions: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    ddof: int = 0,
) -> PoseStabilityMetrics:
    """Compute a static-window summary using Euclidean and SO(3) residuals."""

    position_array = _positions_array(positions)
    quaternion_array = _quaternions_array(quaternions)
    if position_array.shape[0] != quaternion_array.shape[0]:
        raise ValueError("positions and quaternions must have the same sample count")
    denominator = _denominator(position_array.shape[0], ddof)

    mean_position = np.mean(position_array, axis=0)
    mean_quaternion = average_quaternions(quaternion_array)
    position_residuals = np.linalg.norm(position_array - mean_position, axis=1)
    angular_residuals_rad = np.array(
        [quaternion_angular_distance(mean_quaternion, quaternion) for quaternion in quaternion_array],
        dtype=float,
    )
    position_rms = float(np.sqrt(np.sum(position_residuals ** 2) / denominator))
    rotation_rms_rad = float(np.sqrt(np.sum(angular_residuals_rad ** 2) / denominator))

    mean_position = mean_position.copy()
    mean_quaternion = mean_quaternion.copy()
    mean_position.setflags(write=False)
    mean_quaternion.setflags(write=False)
    return PoseStabilityMetrics(
        sample_count=int(position_array.shape[0]),
        mean_position=mean_position,
        mean_quaternion=mean_quaternion,
        position_stability_m=position_rms,
        rotation_stability_rad=rotation_rms_rad,
        rotation_stability_deg=float(np.degrees(rotation_rms_rad)),
        max_position_deviation_m=float(np.max(position_residuals)),
        max_rotation_deviation_deg=float(np.degrees(np.max(angular_residuals_rad))),
    )


@dataclass(frozen=True)
class StabilityAssessment:
    """A stability summary plus the configured threshold decision."""

    is_stable: bool
    invalid_reason: str
    metrics: PoseStabilityMetrics


def assess_pose_stability(
    positions: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    max_position_std_m: float,
    max_rotation_std_deg: float,
    ddof: int = 0,
) -> StabilityAssessment:
    """Compare one window with configured position and SO(3) thresholds."""

    position_threshold = float(max_position_std_m)
    rotation_threshold = float(max_rotation_std_deg)
    if not np.isfinite(position_threshold) or position_threshold < 0.0:
        raise ValueError("max_position_std_m must be finite and non-negative")
    if not np.isfinite(rotation_threshold) or rotation_threshold < 0.0:
        raise ValueError("max_rotation_std_deg must be finite and non-negative")

    metrics = compute_pose_stability(positions, quaternions, ddof=ddof)
    reasons = []
    if metrics.position_stability_m > position_threshold:
        reasons.append("position_unstable")
    if metrics.rotation_stability_deg > rotation_threshold:
        reasons.append("rotation_unstable")
    return StabilityAssessment(not reasons, ";".join(reasons), metrics)


__all__ = [
    "PoseStabilityMetrics",
    "RangeValidationResult",
    "StabilityAssessment",
    "apply_bias_correction_m",
    "assess_pose_stability",
    "compute_pose_stability",
    "compute_range_error_m",
    "correct_range",
    "correct_range_m",
    "position_stability_m",
    "range_error_m",
    "rotation_stability_deg",
    "rotation_stability_rad",
    "validate_range_m",
]
