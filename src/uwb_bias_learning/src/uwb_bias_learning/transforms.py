"""Rigid-body geometry used by the offline UWB bias pipeline.

All public quaternion arguments and return values use ROS field order
``[qx, qy, qz, qw]``.  A pose is interpreted as ``T_M_X``: it maps a point
expressed in frame X into the mocap/world frame M.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


_QUATERNION_SIZE = 4
_POSITION_SIZE = 3


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError("{} must have shape ({},), got {}".format(name, size, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Return a unit quaternion in external ``qx,qy,qz,qw`` order.

    A zero, non-finite, or incorrectly shaped quaternion is rejected rather
    than silently replaced by the identity rotation.
    """

    q = _finite_vector(quaternion, _QUATERNION_SIZE, "quaternion")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be greater than zero")
    return q / norm


def canonicalize_quaternion(
    quaternion: Sequence[float], sign_tolerance: float = 1.0e-12
) -> np.ndarray:
    """Normalize a quaternion and choose one deterministic representative.

    ``q`` and ``-q`` encode the same rotation.  The scalar component is made
    positive when it is not approximately zero.  At a 180-degree rotation,
    the first significant vector component is made positive instead.  Thus
    this function maps an exact antipodal pair to the same output.
    """

    if sign_tolerance < 0.0:
        raise ValueError("sign_tolerance must be non-negative")
    q = normalize_quaternion(quaternion)
    # Prefer qw, then qx, qy, qz.  The fallback only applies to an impossible
    # all-zero unit quaternion, but keeps the logic total and explicit.
    for index in (3, 0, 1, 2):
        if abs(float(q[index])) > sign_tolerance:
            if q[index] < 0.0:
                q = -q
            break
    return q


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a ``qx,qy,qz,qw`` quaternion to a 3-by-3 rotation matrix."""

    x, y, z, w = normalize_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def is_valid_rotation_matrix(rotation: Sequence[Sequence[float]], atol: float = 1.0e-6) -> bool:
    """Return whether ``rotation`` is finite and belongs to SO(3)."""

    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return False
    orthogonality = matrix.T.dot(matrix)
    determinant = float(np.linalg.det(matrix))
    return bool(
        np.allclose(orthogonality, np.eye(3), rtol=0.0, atol=atol)
        and abs(determinant - 1.0) <= atol
    )


# Short compatibility name used naturally by callers and tests.
is_rotation_matrix = is_valid_rotation_matrix


def require_rotation_matrix(
    rotation: Sequence[Sequence[float]], atol: float = 1.0e-6
) -> np.ndarray:
    """Return ``rotation`` as an array, raising if it is not in SO(3)."""

    matrix = np.asarray(rotation, dtype=float)
    if not is_valid_rotation_matrix(matrix, atol=atol):
        raise ValueError("rotation matrix must be finite, orthonormal, and have determinant +1")
    return matrix


def rotation_matrix_to_quaternion(
    rotation: Sequence[Sequence[float]], atol: float = 1.0e-6
) -> np.ndarray:
    """Convert a valid rotation matrix to canonical ``qx,qy,qz,qw`` form."""

    matrix = require_rotation_matrix(rotation, atol=atol)
    trace = float(np.trace(matrix))

    # Branch on the largest stable denominator.  This avoids numerical loss
    # near pi, where the trace-based scalar component is almost zero.
    if trace > 0.0:
        scale = 2.0 * np.sqrt(max(0.0, trace + 1.0))
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    return canonicalize_quaternion(np.array([x, y, z, w], dtype=float))


matrix_to_quaternion = rotation_matrix_to_quaternion


def make_transform(
    position: Sequence[float],
    quaternion: Optional[Sequence[float]] = None,
    *,
    rotation_matrix: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Construct a homogeneous transform from a position and one rotation.

    Exactly one of ``quaternion`` and ``rotation_matrix`` must be supplied.
    The translation is the origin of the child frame expressed in the parent
    frame, matching ROS ``Pose`` semantics.
    """

    translation = _finite_vector(position, _POSITION_SIZE, "position")
    if (quaternion is None) == (rotation_matrix is None):
        raise ValueError("provide exactly one of quaternion or rotation_matrix")
    if quaternion is not None:
        rotation = quaternion_to_matrix(quaternion)
    else:
        rotation = require_rotation_matrix(rotation_matrix)  # type: ignore[arg-type]

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


pose_to_transform = make_transform


def transform_from_rotation_translation(
    rotation: Sequence[Sequence[float]], translation: Sequence[float]
) -> np.ndarray:
    """Construct a transform using the conventional ``(R, t)`` argument order."""

    return make_transform(translation, rotation_matrix=rotation)


def _require_transform(transform: Sequence[Sequence[float]], atol: float = 1.0e-6) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4), got {}".format(matrix.shape))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform must contain only finite values")
    if not np.allclose(matrix[3, :], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=atol):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    require_rotation_matrix(matrix[:3, :3], atol=atol)
    return matrix


def invert_transform(transform: Sequence[Sequence[float]]) -> np.ndarray:
    """Invert a rigid homogeneous transform without a generic matrix inverse."""

    matrix = _require_transform(transform)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T.dot(translation)
    return inverse


inverse_transform = invert_transform


def relative_transform_from_matrices(
    transform_m_a: Sequence[Sequence[float]], transform_m_b: Sequence[Sequence[float]]
) -> np.ndarray:
    """Compute ``T_A_B = inverse(T_M_A) @ T_M_B`` from two transforms."""

    t_m_a = _require_transform(transform_m_a)
    t_m_b = _require_transform(transform_m_b)
    result = invert_transform(t_m_a).dot(t_m_b)
    # Rebuild the final row to remove insignificant multiplication noise.
    result[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return result


def relative_transform(
    position_a: Sequence[float],
    quaternion_a: Sequence[float],
    position_b: Sequence[float],
    quaternion_b: Sequence[float],
) -> np.ndarray:
    """Compute ``T_A_B`` from mocap poses ``T_M_A`` and ``T_M_B``.

    The returned transform maps coordinates in B into coordinates in A.
    """

    transform_m_a = make_transform(position_a, quaternion_a)
    transform_m_b = make_transform(position_b, quaternion_b)
    return relative_transform_from_matrices(transform_m_a, transform_m_b)


def relative_pose(
    position_a: Sequence[float],
    quaternion_a: Sequence[float],
    position_b: Sequence[float],
    quaternion_b: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return relative translation and quaternion for ``T_A_B``."""

    transform_a_b = relative_transform(position_a, quaternion_a, position_b, quaternion_b)
    translation = transform_a_b[:3, 3].copy()
    quaternion = rotation_matrix_to_quaternion(transform_a_b[:3, :3])
    return translation, quaternion


def rotation_angle_rad(rotation: Sequence[Sequence[float]]) -> float:
    """Return the principal SO(3) angle of a rotation in radians."""

    matrix = require_rotation_matrix(rotation)
    cosine = float((np.trace(matrix) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rotation_angle_between(
    rotation_a: Sequence[Sequence[float]], rotation_b: Sequence[Sequence[float]]
) -> float:
    """Return the geodesic angle between two rotation matrices in radians."""

    matrix_a = require_rotation_matrix(rotation_a)
    matrix_b = require_rotation_matrix(rotation_b)
    return rotation_angle_rad(matrix_a.T.dot(matrix_b))


rotation_angle_difference = rotation_angle_between


def quaternion_angular_distance(
    quaternion_a: Sequence[float], quaternion_b: Sequence[float]
) -> float:
    """Return the shortest angular distance between two quaternions in radians."""

    q_a = normalize_quaternion(quaternion_a)
    q_b = normalize_quaternion(quaternion_b)
    # Absolute value makes the result invariant to either quaternion's sign.
    cosine_half_angle = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine_half_angle))


def average_quaternions(
    quaternions: Sequence[Sequence[float]], weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return a valid rotation average using the Markley quaternion method.

    The outer-product formulation is invariant to each input quaternion's
    sign.  Optional weights must be finite, non-negative, and have positive
    total weight.
    """

    array = np.asarray(quaternions, dtype=float)
    if array.ndim != 2 or array.shape[1] != _QUATERNION_SIZE or array.shape[0] == 0:
        raise ValueError("quaternions must have shape (N, 4) with N >= 1")
    normalized = np.vstack([normalize_quaternion(row) for row in array])

    if weights is None:
        weight_array = np.ones(normalized.shape[0], dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != (normalized.shape[0],):
            raise ValueError("weights must have shape ({},)".format(normalized.shape[0]))
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
            raise ValueError("weights must be finite and non-negative")
    total_weight = float(np.sum(weight_array))
    if total_weight <= 0.0:
        raise ValueError("weights must have a positive sum")

    accumulator = np.zeros((4, 4), dtype=float)
    for quaternion, weight in zip(normalized, weight_array):
        accumulator += float(weight) * np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator / total_weight)
    average = eigenvectors[:, int(np.argmax(eigenvalues))]
    return canonicalize_quaternion(average)


mean_quaternions = average_quaternions


def average_rotations(
    rotations: Sequence[Sequence[Sequence[float]]], weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Average rotation matrices and return a matrix that remains in SO(3)."""

    array = np.asarray(rotations, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (3, 3) or array.shape[0] == 0:
        raise ValueError("rotations must have shape (N, 3, 3) with N >= 1")
    quaternions = np.vstack([rotation_matrix_to_quaternion(matrix) for matrix in array])
    return quaternion_to_matrix(average_quaternions(quaternions, weights=weights))


__all__ = [
    "average_quaternions",
    "average_rotations",
    "canonicalize_quaternion",
    "inverse_transform",
    "invert_transform",
    "is_rotation_matrix",
    "is_valid_rotation_matrix",
    "make_transform",
    "matrix_to_quaternion",
    "mean_quaternions",
    "normalize_quaternion",
    "pose_to_transform",
    "quaternion_angular_distance",
    "quaternion_to_matrix",
    "relative_pose",
    "relative_transform",
    "relative_transform_from_matrices",
    "require_rotation_matrix",
    "rotation_angle_between",
    "rotation_angle_difference",
    "rotation_angle_rad",
    "rotation_matrix_to_quaternion",
    "transform_from_rotation_translation",
]
