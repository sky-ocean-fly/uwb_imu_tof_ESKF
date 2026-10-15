"""Geometry and statistics for mocap-referenced UWB range bias estimation.

The relative-pose convention follows the rest of this package: ``T_X_Y``
maps a point expressed in frame Y into frame X.  Antenna locations are points
expressed in their attached rigid-body frames; no antenna-axis orientation is
needed to calculate a scalar range.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .transforms import require_rotation_matrix


def _point(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError("{} must have shape (3,), got {}".format(name, result.shape))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must contain only finite values".format(name))
    return result


def true_range_from_relative_pose(
    translation: Sequence[float],
    rotation: Sequence[Sequence[float]],
    tag_position_body_m: Sequence[float],
    anchor_position_base_m: Sequence[float],
    *,
    pose_direction: str = "base_to_body",
) -> Dict[str, Any]:
    """Calculate antenna-to-antenna distance from one relative rigid-body pose.

    ``base_to_body`` means the input is ``T_base_body``.  ``body_to_base``
    means it is ``T_body_base``.  The returned endpoint coordinates and delta
    are all expressed in the indicated ``reference_frame``.
    """

    t = _point(translation, "translation")
    r = require_rotation_matrix(rotation)
    tag_body = _point(tag_position_body_m, "tag_position_body_m")
    anchor_base = _point(anchor_position_base_m, "anchor_position_base_m")

    if pose_direction == "base_to_body":
        tag_reference = r.dot(tag_body) + t
        anchor_reference = anchor_base
        reference_frame = "base"
    elif pose_direction == "body_to_base":
        tag_reference = tag_body
        anchor_reference = r.dot(anchor_base) + t
        reference_frame = "body"
    else:
        raise ValueError(
            "pose_direction must be 'base_to_body' or 'body_to_base'"
        )

    delta = tag_reference - anchor_reference
    return {
        "reference_frame": reference_frame,
        "tag_position_reference_m": tag_reference,
        "anchor_position_reference_m": anchor_reference,
        "delta_reference_m": delta,
        "true_distance_m": float(np.linalg.norm(delta)),
    }


def summarize_bias_samples(samples_m: Iterable[float]) -> Dict[str, float]:
    """Return descriptive statistics for signed bias samples in metres."""

    values = np.asarray(list(samples_m), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("at least one bias sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("bias samples must contain only finite values")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mean_m": float(np.mean(values)),
        "std_m_sample": (
            float(statistics.stdev(float(value) for value in values))
            if values.size >= 2
            else 0.0
        ),
        "median_m": float(np.median(values)),
        "min_m": float(np.min(values)),
        "max_m": float(np.max(values)),
        "rmse_m": float(math.sqrt(float(np.mean(values * values)))),
        "mae_m": float(np.mean(absolute)),
        "abs_p95_m": float(np.percentile(absolute, 95.0)),
    }


__all__ = ["summarize_bias_samples", "true_range_from_relative_pose"]
