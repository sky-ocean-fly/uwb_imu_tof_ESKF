#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning.bias_estimation import (
    summarize_bias_samples,
    true_range_from_relative_pose,
)
from uwb_bias_learning.transforms import quaternion_to_matrix


class BiasEstimationTests(unittest.TestCase):
    def test_base_to_body_applies_both_endpoint_lever_arms(self):
        # Body origin is at (2, 0, 0) in base.  A +90 degree body yaw maps the
        # tag lever arm (1, 0, 0) to (0, 1, 0) in base.  Anchor is at (0, 1, 0).
        rotation = quaternion_to_matrix(
            [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)]
        )
        result = true_range_from_relative_pose(
            [2.0, 0.0, 0.0],
            rotation,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            pose_direction="base_to_body",
        )
        np.testing.assert_allclose(result["tag_position_reference_m"], [2.0, 1.0, 0.0])
        np.testing.assert_allclose(result["anchor_position_reference_m"], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(result["true_distance_m"], 2.0)
        self.assertEqual(result["reference_frame"], "base")

    def test_inverse_pose_direction_gives_same_distance(self):
        rotation_base_body = quaternion_to_matrix(
            [0.0, 0.0, np.sin(0.2), np.cos(0.2)]
        )
        translation_base_body = np.asarray([1.2, -0.3, 0.7])
        tag_body = [0.2, -0.1, 0.05]
        anchor_base = [-0.4, 0.3, 0.1]
        forward = true_range_from_relative_pose(
            translation_base_body,
            rotation_base_body,
            tag_body,
            anchor_base,
            pose_direction="base_to_body",
        )
        rotation_body_base = rotation_base_body.T
        translation_body_base = -rotation_body_base.dot(translation_base_body)
        inverse = true_range_from_relative_pose(
            translation_body_base,
            rotation_body_base,
            tag_body,
            anchor_base,
            pose_direction="body_to_base",
        )
        self.assertAlmostEqual(forward["true_distance_m"], inverse["true_distance_m"], places=12)

    def test_bias_statistics_keep_sign(self):
        summary = summarize_bias_samples([0.1, -0.2, 0.4])
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean_m"], 0.1)
        self.assertAlmostEqual(summary["median_m"], 0.1)
        self.assertAlmostEqual(summary["mae_m"], 0.7 / 3.0)


if __name__ == "__main__":
    unittest.main()
