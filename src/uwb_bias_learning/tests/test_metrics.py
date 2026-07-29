#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.metrics import (  # noqa: E402
    bias_prediction_metrics,
    bootstrap_confidence_intervals,
    correct_ranges,
    metrics_by_group,
    range_error_metrics,
)


class MetricsTest(unittest.TestCase):
    def test_bias_metrics_use_prediction_minus_truth(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0])
        prediction = np.array([0.1, 0.8, 2.3, 2.6])
        error = prediction - truth
        result = bias_prediction_metrics(truth, prediction)
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mae"], np.mean(np.abs(error)))
        self.assertAlmostEqual(result["rmse"], np.sqrt(np.mean(error ** 2)))
        self.assertAlmostEqual(result["mean_error"], np.mean(error))
        self.assertAlmostEqual(result["error_std"], np.std(error, ddof=0))
        self.assertAlmostEqual(result["p90_abs_error"], np.percentile(np.abs(error), 90))
        expected_r2 = 1.0 - np.sum(error ** 2) / np.sum((truth - np.mean(truth)) ** 2)
        self.assertAlmostEqual(result["r2"], expected_r2)

    def test_constant_reference_has_undefined_r2(self):
        result = bias_prediction_metrics([0.2, 0.2], [0.1, 0.3])
        self.assertTrue(math.isnan(result["r2"]))

    def test_signed_bias_is_always_subtracted(self):
        raw = np.array([5.12, 4.90])
        truth = np.array([5.00, 5.00])
        signed_bias = np.array([0.12, -0.10])
        np.testing.assert_allclose(correct_ranges(raw, signed_bias), truth, atol=1.0e-12)
        corrected_metrics = range_error_metrics(raw, truth, signed_bias)
        self.assertAlmostEqual(corrected_metrics["rmse"], 0.0, places=12)
        raw_metrics = range_error_metrics(raw, truth)
        self.assertGreater(raw_metrics["rmse"], 0.0)

    def test_range_correction_rejects_accidental_matrix_broadcast(self):
        with self.assertRaises(ValueError):
            correct_ranges(np.ones((3, 1)), np.ones(3))
        np.testing.assert_allclose(
            correct_ranges(np.ones(3), 0.25), np.full(3, 0.75)
        )

    def test_group_metrics_do_not_mix_groups(self):
        result = metrics_by_group(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 4.0, 5.0],
            ["session_b", "session_b", "session_a", "session_a"],
        )
        self.assertEqual(list(result), ["session_a", "session_b"])
        self.assertAlmostEqual(result["session_a"]["mae"], 1.0)
        self.assertAlmostEqual(result["session_b"]["mae"], 0.0)

    def test_bootstrap_is_seeded_and_does_not_change_direct_metrics(self):
        first = bootstrap_confidence_intervals(
            [0.0, 1.0, 2.0], [0.1, 1.2, 1.7], repetitions=40, seed=2026
        )
        second = bootstrap_confidence_intervals(
            [0.0, 1.0, 2.0], [0.1, 1.2, 1.7], repetitions=40, seed=2026
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first["mae"]["lower"], first["mae"]["upper"])

    def test_empty_or_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            bias_prediction_metrics([], [])
        with self.assertRaises(ValueError):
            bias_prediction_metrics([0.0], [float("nan")])
        with self.assertRaises(ValueError):
            range_error_metrics([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
