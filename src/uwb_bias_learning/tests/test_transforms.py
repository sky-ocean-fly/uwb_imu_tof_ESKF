#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning.transforms import (
    average_quaternions,
    average_rotations,
    invert_transform,
    is_valid_rotation_matrix,
    make_transform,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_to_matrix,
    relative_pose,
    relative_transform,
    relative_transform_from_matrices,
    rotation_angle_between,
)


def quaternion_z(angle_rad):
    return np.array([0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)])


class TransformTests(unittest.TestCase):
    def test_quaternion_normalization_and_zero_rejection(self):
        normalized = normalize_quaternion([0.0, 0.0, 0.0, 7.0])
        np.testing.assert_allclose(normalized, [0.0, 0.0, 0.0, 1.0], atol=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0)
        with self.assertRaises(ValueError):
            normalize_quaternion([0.0, 0.0, 0.0, 0.0])

    def test_identical_poses_have_identity_relative_transform(self):
        position = np.array([1.2, -3.4, 0.7])
        quaternion = quaternion_z(np.deg2rad(37.0))
        relative = relative_transform(position, quaternion, position, quaternion)
        np.testing.assert_allclose(relative, np.eye(4), atol=1.0e-12)
        self.assertAlmostEqual(float(np.linalg.norm(relative[:3, 3])), 0.0, places=12)

    def test_identity_a_preserves_b_translation(self):
        relative = relative_transform(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(relative[:3, 3], [1.0, 2.0, 3.0], atol=1.0e-12)

    def test_global_left_transform_does_not_change_relative_transform(self):
        transform_m_a = make_transform([0.3, -1.0, 2.0], quaternion_z(0.2))
        transform_m_b = make_transform([2.3, 0.7, -0.4], quaternion_z(-0.8))
        original = relative_transform_from_matrices(transform_m_a, transform_m_b)

        global_change = make_transform([-8.0, 4.0, 1.5], quaternion_z(1.1))
        changed = relative_transform_from_matrices(
            global_change.dot(transform_m_a), global_change.dot(transform_m_b)
        )
        np.testing.assert_allclose(changed, original, atol=1.0e-12)

    def test_manual_direction_example_is_t_a_b_not_t_b_a(self):
        # A is at (1,0,0) and its x axis points along world +y.  B is at
        # (1,1,0), so B's origin is at +1 metre on A's x axis.
        relative = relative_transform(
            [1.0, 0.0, 0.0],
            quaternion_z(np.pi / 2.0),
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(relative[:3, 3], [1.0, 0.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(
            relative[:3, :3], quaternion_to_matrix(quaternion_z(-np.pi / 2.0)), atol=1.0e-12
        )
        np.testing.assert_allclose(invert_transform(relative).dot(relative), np.eye(4), atol=1.0e-12)

    def test_quaternion_sign_does_not_change_rotation_or_relative_pose(self):
        q_a = normalize_quaternion([0.1, -0.2, 0.3, 0.9])
        q_b = normalize_quaternion([-0.4, 0.2, 0.1, 0.8])
        np.testing.assert_allclose(quaternion_to_matrix(q_a), quaternion_to_matrix(-q_a), atol=0.0)
        positive = relative_transform([0, 0, 0], q_a, [1, 2, 3], q_b)
        antipodal = relative_transform([0, 0, 0], -q_a, [1, 2, 3], -q_b)
        np.testing.assert_allclose(positive, antipodal, atol=1.0e-12)

        translation_positive, quaternion_positive = relative_pose([0, 0, 0], q_a, [1, 2, 3], q_b)
        translation_negative, quaternion_negative = relative_pose([0, 0, 0], -q_a, [1, 2, 3], -q_b)
        np.testing.assert_allclose(translation_positive, translation_negative, atol=1.0e-12)
        np.testing.assert_allclose(quaternion_positive, quaternion_negative, atol=1.0e-12)

    def test_rotation_matrix_round_trip_and_so3_checks(self):
        for angle in (0.0, 0.3, np.pi - 1.0e-8, np.pi):
            rotation = quaternion_to_matrix(quaternion_z(angle))
            self.assertTrue(is_valid_rotation_matrix(rotation))
            round_trip = quaternion_to_matrix(matrix_to_quaternion(rotation))
            np.testing.assert_allclose(round_trip, rotation, atol=1.0e-8)
            self.assertAlmostEqual(rotation_angle_between(rotation, round_trip), 0.0, places=7)

        reflection = np.diag([-1.0, 1.0, 1.0])
        self.assertFalse(is_valid_rotation_matrix(reflection))
        self.assertFalse(is_valid_rotation_matrix(np.ones((2, 2))))

    def test_rotation_average_is_sign_invariant_and_stays_on_so3(self):
        q0 = quaternion_z(np.deg2rad(20.0))
        q1 = quaternion_z(np.deg2rad(24.0))
        average_a = average_quaternions([q0, q1])
        average_b = average_quaternions([-q0, q1])
        np.testing.assert_allclose(average_a, average_b, atol=1.0e-12)
        expected = quaternion_to_matrix(quaternion_z(np.deg2rad(22.0)))
        np.testing.assert_allclose(quaternion_to_matrix(average_a), expected, atol=1.0e-12)

        rotation_average = average_rotations([quaternion_to_matrix(q0), quaternion_to_matrix(q1)])
        self.assertTrue(is_valid_rotation_matrix(rotation_average, atol=1.0e-12))
        np.testing.assert_allclose(rotation_average, expected, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
