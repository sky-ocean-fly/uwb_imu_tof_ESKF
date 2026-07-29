#!/usr/bin/env python3

import unittest

import numpy as np

from uwb_bias_learning.quality import correct_range_m, range_error_m


class BiasSignConventionTests(unittest.TestCase):
    def test_positive_bias_is_subtracted(self):
        measured_range_m = 5.12
        distance_gt_m = 5.00
        bias_m = range_error_m(measured_range_m, distance_gt_m)
        self.assertAlmostEqual(bias_m, 0.12)
        self.assertAlmostEqual(correct_range_m(measured_range_m, bias_m), distance_gt_m)

    def test_negative_bias_is_still_subtracted_as_a_signed_quantity(self):
        measured_range_m = 4.90
        distance_gt_m = 5.00
        bias_m = range_error_m(measured_range_m, distance_gt_m)
        self.assertAlmostEqual(bias_m, -0.10)
        # Subtracting -0.10 must add 0.10.  Taking abs(bias) or always adding
        # the model output would make this assertion fail.
        self.assertAlmostEqual(correct_range_m(measured_range_m, bias_m), distance_gt_m)

    def test_vectorized_sign_convention(self):
        truth = np.array([2.0, 3.0, 4.0])
        measured = np.array([2.2, 2.7, 4.0])
        bias = range_error_m(measured, truth)
        np.testing.assert_allclose(bias, [0.2, -0.3, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(correct_range_m(measured, bias), truth, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
