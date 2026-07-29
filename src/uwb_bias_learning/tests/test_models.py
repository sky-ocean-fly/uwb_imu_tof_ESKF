#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    ConstantBiasModel,
    FeatureNormalizer,
    RidgeBiasModel,
    ZeroBiasModel,
)


def identity_pose_features(translations):
    identity = np.eye(3).reshape(-1)
    return np.asarray(
        [list(translation) + identity.tolist() for translation in translations],
        dtype=np.float64,
    )


class ModelsTest(unittest.TestCase):
    def test_normalization_is_fitted_only_from_supplied_training_rows(self):
        train = identity_pose_features([[0, 0, 0], [2, 4, 6]])
        validation = identity_pose_features([[1000, 2000, 3000]])
        normalizer = FeatureNormalizer.fit(train)
        np.testing.assert_allclose(normalizer.mean[:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(normalizer.scale[:3], [1.0, 2.0, 3.0])
        transformed = normalizer.transform(validation)
        np.testing.assert_allclose(transformed[0, :3], [999.0, 999.0, 999.0])
        # Transforming validation data must not mutate the fitted statistics.
        np.testing.assert_allclose(normalizer.mean[:3], [1.0, 2.0, 3.0])

    def test_constant_columns_get_unit_scale(self):
        train = identity_pose_features([[0, 0, 0], [1, 0, 0]])
        normalizer = FeatureNormalizer.fit(train)
        self.assertTrue(np.all(normalizer.scale > 0.0))
        np.testing.assert_allclose(normalizer.transform(train)[:, 3:], 0.0, atol=0.0)

    def test_normalization_round_trip_json_preserves_feature_order(self):
        train = identity_pose_features([[0, 1, 2], [3, 4, 5]])
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            path = Path(directory) / "normalization.json"
            original = FeatureNormalizer.fit(train)
            original.save_json(path)
            restored = FeatureNormalizer.load_json(path)
            self.assertEqual(restored.feature_names, POSE12_FEATURES)
            np.testing.assert_allclose(restored.transform(train), original.transform(train))

    def test_normalization_requires_explicit_feature_order(self):
        train = identity_pose_features([[0, 1, 2], [3, 4, 5]])
        payload = FeatureNormalizer.fit(train).to_dict()
        payload.pop("feature_names")
        with self.assertRaises(ValueError):
            FeatureNormalizer.from_dict(payload)

    def test_three_numpy_baselines(self):
        train = identity_pose_features(
            [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]
        )
        target = 0.25 + 0.1 * train[:, 0]
        zero = ZeroBiasModel()
        constant = ConstantBiasModel.fit(target)
        ridge = RidgeBiasModel.fit(train, target, alpha=0.0)
        np.testing.assert_allclose(zero.predict(train), 0.0)
        np.testing.assert_allclose(constant.predict(train), np.mean(target))
        np.testing.assert_allclose(ridge.predict(train), target, atol=1.0e-10)

    def test_domain_check_warns_but_transform_does_not_clip(self):
        train = identity_pose_features([[0, 0, 0], [1, 1, 1]])
        normalizer = FeatureNormalizer.fit(train)
        outside = identity_pose_features([[10, 0.5, 0.5]])
        result = normalizer.domain_check(outside)
        self.assertFalse(result["in_training_domain"])
        self.assertEqual(result["behavior"], "warn_without_clipping")
        self.assertEqual(result["violations"][0]["feature"], "rel_tx")
        # If it had been clipped to train_max, this normalized value would be 1.
        self.assertGreater(normalizer.transform(outside)[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
