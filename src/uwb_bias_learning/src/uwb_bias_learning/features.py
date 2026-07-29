"""Feature construction for the first-version 12-dimensional pose model."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import POSE12_FIELDS
from .transforms import quaternion_to_matrix, require_rotation_matrix


def _translation(value: Sequence[float]) -> np.ndarray:
    translation = np.asarray(value, dtype=float)
    if translation.shape != (3,):
        raise ValueError("translation must have shape (3,), got {}".format(translation.shape))
    if not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain only finite values")
    return translation


def pose12_features(
    translation: Sequence[float], rotation_matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    """Build ``[tx,ty,tz,R00,...,R22]`` with row-major matrix entries."""

    vector = _translation(translation)
    rotation = require_rotation_matrix(rotation_matrix)
    return np.concatenate((vector, rotation.reshape(9, order="C"))).astype(float, copy=False)


def pose12_from_pose(
    translation: Sequence[float], quaternion: Sequence[float]
) -> np.ndarray:
    """Build pose12 directly from a ``qx,qy,qz,qw`` quaternion pose."""

    return pose12_features(translation, quaternion_to_matrix(quaternion))


def pose12_from_transform(transform: Sequence[Sequence[float]]) -> np.ndarray:
    """Build pose12 from a rigid 4-by-4 homogeneous transform."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4), got {}".format(matrix.shape))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform must contain only finite values")
    if not np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1.0e-6):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    return pose12_features(matrix[:3, 3], matrix[:3, :3])


def validate_pose12(feature: Sequence[float], rotation_atol: float = 1.0e-6) -> np.ndarray:
    """Validate a pose12 vector, including that its rotation block is in SO(3)."""

    array = np.asarray(feature, dtype=float)
    if array.shape != (12,):
        raise ValueError("pose12 feature must have shape (12,), got {}".format(array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("pose12 feature must contain only finite values")
    require_rotation_matrix(array[3:].reshape(3, 3), atol=rotation_atol)
    return array


def pose12_to_dict(feature: Sequence[float]) -> dict:
    """Map a validated feature vector to the exact persisted column names."""

    array = validate_pose12(feature)
    return {field: float(value) for field, value in zip(POSE12_FIELDS, array)}


def pose12_from_mapping(row: Mapping[str, float]) -> np.ndarray:
    """Read a pose12 vector from a dataset row without relying on column order."""

    missing = [field for field in POSE12_FIELDS if field not in row]
    if missing:
        raise ValueError("missing pose12 fields: {}".format(", ".join(missing)))
    return validate_pose12([row[field] for field in POSE12_FIELDS])


def pose12_batch(
    translations: Sequence[Sequence[float]],
    rotation_matrices: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Build an ``N x 12`` feature array with per-row SO(3) validation."""

    translation_array = np.asarray(translations, dtype=float)
    rotation_array = np.asarray(rotation_matrices, dtype=float)
    if translation_array.ndim != 2 or translation_array.shape[1] != 3:
        raise ValueError("translations must have shape (N, 3)")
    if rotation_array.shape != (translation_array.shape[0], 3, 3):
        raise ValueError("rotation_matrices must have shape (N, 3, 3)")
    if translation_array.shape[0] == 0:
        return np.empty((0, 12), dtype=float)
    return np.vstack(
        [pose12_features(translation, rotation) for translation, rotation in zip(translation_array, rotation_array)]
    )


# Concise aliases for consumers that select the configured ``pose12`` feature set.
pose12 = pose12_features
build_pose12 = pose12_features


__all__ = [
    "POSE12_FIELDS",
    "build_pose12",
    "pose12",
    "pose12_batch",
    "pose12_features",
    "pose12_from_mapping",
    "pose12_from_pose",
    "pose12_from_transform",
    "pose12_to_dict",
    "validate_pose12",
]
