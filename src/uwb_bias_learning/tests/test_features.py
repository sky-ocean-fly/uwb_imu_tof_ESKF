#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning import POSE12_FIELDS
from uwb_bias_learning.features import (
    pose12_batch,
    pose12_features,
    pose12_from_mapping,
    pose12_from_pose,
    pose12_from_transform,
    pose12_to_dict,
)
from uwb_bias_learning.transforms import make_transform, quaternion_to_matrix


class Pose12FeatureTests(unittest.TestCase):
    def test_exact_translation_then_row_major_rotation_order(self):
        quaternion = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        rotation = quaternion_to_matrix(quaternion)
        feature = pose12_features([1.0, 2.0, 3.0], rotation)
        expected = np.concatenate(([1.0, 2.0, 3.0], rotation.reshape(-1)))
        self.assertEqual(feature.shape, (12,))
        np.testing.assert_allclose(feature, expected, atol=0.0)
        self.assertEqual(
            tuple(POSE12_FIELDS),
            (
                "rel_tx",
                "rel_ty",
                "rel_tz",
                "rel_R00",
                "rel_R01",
                "rel_R02",
                "rel_R10",
                "rel_R11",
                "rel_R12",
                "rel_R20",
                "rel_R21",
                "rel_R22",
            ),
        )

    def test_q_and_minus_q_produce_identical_features(self):
        quaternion = np.array([0.2, -0.3, 0.1, 0.8])
        positive = pose12_from_pose([0.4, 0.5, 0.6], quaternion)
        antipodal = pose12_from_pose([0.4, 0.5, 0.6], -quaternion)
        np.testing.assert_allclose(positive, antipodal, atol=0.0)

    def test_transform_and_mapping_paths_preserve_feature_order(self):
        quaternion = np.array([0.0, np.sin(0.2), 0.0, np.cos(0.2)])
        transform = make_transform([-1.0, 4.0, 2.5], quaternion)
        feature = pose12_from_transform(transform)
        mapping = pose12_to_dict(feature)
        self.assertEqual(tuple(mapping.keys()), tuple(POSE12_FIELDS))
        np.testing.assert_allclose(pose12_from_mapping(mapping), feature, atol=0.0)

    def test_invalid_rotation_is_rejected_instead_of_used_as_feature(self):
        with self.assertRaises(ValueError):
            pose12_features([0.0, 0.0, 0.0], np.zeros((3, 3)))

    def test_batch_feature_shape_and_empty_batch(self):
        rotations = np.stack((np.eye(3), np.eye(3)))
        features = pose12_batch([[1, 2, 3], [4, 5, 6]], rotations)
        self.assertEqual(features.shape, (2, 12))
        np.testing.assert_allclose(features[:, :3], [[1, 2, 3], [4, 5, 6]], atol=0.0)
        empty = pose12_batch(np.empty((0, 3)), np.empty((0, 3, 3)))
        self.assertEqual(empty.shape, (0, 12))


if __name__ == "__main__":
    unittest.main()
