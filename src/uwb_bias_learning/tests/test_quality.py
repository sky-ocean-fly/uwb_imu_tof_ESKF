#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning.quality import (
    assess_pose_stability,
    compute_pose_stability,
    position_stability_m,
    rotation_stability_deg,
    validate_range_m,
)


def quaternion_z_deg(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)])


class QualityTests(unittest.TestCase):
    def test_position_stability_is_rms_euclidean_dispersion(self):
        positions = np.array([[-0.003, 0.0, 0.0], [0.003, 0.0, 0.0]])
        self.assertAlmostEqual(position_stability_m(positions), 0.003, places=12)
        self.assertAlmostEqual(position_stability_m(positions, ddof=1), np.sqrt(18.0e-6), places=12)

    def test_rotation_stability_uses_so3_not_wrapped_euler_subtraction(self):
        # These yaw values are two degrees apart, not 358 degrees apart.
        quaternions = np.vstack((quaternion_z_deg(179.0), quaternion_z_deg(-179.0)))
        self.assertAlmostEqual(rotation_stability_deg(quaternions), 1.0, places=8)

    def test_antipodal_quaternion_samples_have_zero_rotation_dispersion(self):
        quaternion = quaternion_z_deg(43.0)
        self.assertAlmostEqual(rotation_stability_deg([quaternion, -quaternion]), 0.0, places=10)

    def test_combined_stability_assessment_reports_each_failed_threshold(self):
        positions = np.array([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
        quaternions = np.vstack((quaternion_z_deg(-2.0), quaternion_z_deg(2.0)))
        assessment = assess_pose_stability(
            positions,
            quaternions,
            max_position_std_m=0.005,
            max_rotation_std_deg=1.0,
        )
        self.assertFalse(assessment.is_stable)
        self.assertEqual(assessment.invalid_reason, "position_unstable;rotation_unstable")
        self.assertEqual(assessment.metrics.sample_count, 2)

        metrics = compute_pose_stability(positions, quaternions)
        self.assertAlmostEqual(metrics.position_stability_m, 0.01, places=12)
        self.assertAlmostEqual(metrics.rotation_stability_deg, 2.0, places=8)

    def test_range_validation_does_not_guess_device_specific_sentinels(self):
        self.assertTrue(validate_range_m(2.0).is_valid)
        self.assertEqual(validate_range_m(float("nan")).invalid_reason, "range_nonfinite")
        self.assertEqual(validate_range_m(0.0).invalid_reason, "range_nonpositive")
        self.assertEqual(validate_range_m(0.9, min_m=1.0).invalid_reason, "range_below_min")
        self.assertEqual(validate_range_m(5.1, max_m=5.0).invalid_reason, "range_above_max")
        with self.assertRaises(ValueError):
            validate_range_m(2.0, min_m=3.0, max_m=1.0)


if __name__ == "__main__":
    unittest.main()
