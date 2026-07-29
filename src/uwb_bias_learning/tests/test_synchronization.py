#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning.synchronization import (
    TimedPose,
    interpolate_pose,
    quaternion_slerp,
    synchronize_pose,
    synchronize_pose_pair,
)
from uwb_bias_learning.transforms import quaternion_to_matrix


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.identity = np.array([0.0, 0.0, 0.0, 1.0])
        self.half_turn_z = np.array([0.0, 0.0, 1.0, 0.0])
        self.before = TimedPose(10.0, [0.0, 0.0, 0.0], self.identity)
        self.after = TimedPose(12.0, [2.0, 4.0, 6.0], self.half_turn_z)

    def test_position_linear_interpolation_and_quaternion_slerp(self):
        result = interpolate_pose(self.before, self.after, 11.0)
        self.assertTrue(result.is_valid)
        np.testing.assert_allclose(result.position, [1.0, 2.0, 3.0], atol=1.0e-12)
        expected_quaternion = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(
            quaternion_to_matrix(result.quaternion),
            quaternion_to_matrix(expected_quaternion),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(result.interpolation_ratio, 0.5)
        self.assertAlmostEqual(result.gap_before_s, 1.0)
        self.assertAlmostEqual(result.gap_after_s, 1.0)
        self.assertAlmostEqual(result.sync_gap_s, 1.0)

    def test_slerp_handles_antipodal_quaternion_without_spurious_rotation(self):
        midpoint = quaternion_slerp(self.identity, -self.identity, 0.5)
        np.testing.assert_allclose(quaternion_to_matrix(midpoint), np.eye(3), atol=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(midpoint)), 1.0)

    def test_sync_gap_over_threshold_is_rejected_with_audit_times(self):
        result = synchronize_pose([self.before, self.after], 11.0, max_sync_gap_s=0.99)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.invalid_reason, "sync_gap_exceeded")
        self.assertIsNone(result.position)
        self.assertIsNone(result.quaternion)
        self.assertAlmostEqual(result.timestamp_before, 10.0)
        self.assertAlmostEqual(result.timestamp_after, 12.0)
        self.assertAlmostEqual(result.interpolation_ratio, 0.5)
        self.assertAlmostEqual(result.sync_gap_s, 1.0)

    def test_sync_gap_equal_to_threshold_is_accepted(self):
        result = synchronize_pose([self.before, self.after], 11.0, max_sync_gap_s=1.0)
        self.assertTrue(result.is_valid)
        np.testing.assert_allclose(result.position, [1.0, 2.0, 3.0], atol=1.0e-12)

    def test_exact_timestamp_is_valid_even_with_zero_threshold(self):
        result = synchronize_pose([self.before, self.after], 10.0, max_sync_gap_s=0.0)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.timestamp_before, 10.0)
        self.assertEqual(result.timestamp_after, 10.0)
        self.assertEqual(result.sync_gap_s, 0.0)
        np.testing.assert_allclose(result.position, self.before.position, atol=0.0)

    def test_extrapolation_is_rejected(self):
        early = synchronize_pose([self.before, self.after], 9.99, max_sync_gap_s=1.0)
        late = synchronize_pose([self.before, self.after], 12.01, max_sync_gap_s=1.0)
        self.assertFalse(early.is_valid)
        self.assertEqual(early.invalid_reason, "target_before_pose_buffer")
        self.assertFalse(late.is_valid)
        self.assertEqual(late.invalid_reason, "target_after_pose_buffer")

    def test_both_pose_streams_must_synchronize(self):
        valid_a = [self.before, self.after]
        far_b = [
            TimedPose(8.0, [0, 0, 0], self.identity),
            TimedPose(14.0, [0, 0, 0], self.identity),
        ]
        result = synchronize_pose_pair(valid_a, far_b, 11.0, max_sync_gap_s=1.0)
        self.assertFalse(result.is_valid)
        self.assertTrue(result.pose_a.is_valid)
        self.assertFalse(result.pose_b.is_valid)
        self.assertIn("mocap_b:sync_gap_exceeded", result.invalid_reason)

    def test_duplicate_or_reversed_timestamps_are_not_silently_sorted(self):
        duplicate = TimedPose(10.0, [1, 0, 0], self.identity)
        with self.assertRaises(ValueError):
            synchronize_pose([self.before, duplicate, self.after], 11.0, max_sync_gap_s=1.0)
        with self.assertRaises(ValueError):
            synchronize_pose([self.after, self.before], 11.0, max_sync_gap_s=1.0)


if __name__ == "__main__":
    unittest.main()
