#!/usr/bin/env python3
"""Leakage and train-only normalization tests on explicit synthetic poses."""

import tempfile
import unittest
from pathlib import Path

from uwb_bias_learning.datasets import (
    DatasetError,
    OutputExistsError,
    POSE12_FIELDS,
    assert_distinct_new_outputs,
    fit_normalization,
    split_pose_rows,
    write_csv_exclusive,
)


def synthetic_pose_row(session_id, pose_id, tx, bias=0.1):
    row = {
        "data_origin": "synthetic_test",
        "session_id": session_id,
        "pose_id": pose_id,
        "environment_id": "synthetic_environment",
        "rel_tx": float(tx),
        "rel_ty": 0.0,
        "rel_tz": 0.0,
        "rel_qx": 0.0,
        "rel_qy": 0.0,
        "rel_qz": 0.0,
        "rel_qw": 1.0,
        "bias_mean": float(bias),
        "quality_status": "accepted",
    }
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for name, value in zip(POSE12_FIELDS[3:], identity):
        row[name] = value
    return row


class DatasetSplitTest(unittest.TestCase):
    def test_session_split_keeps_sessions_poses_and_near_duplicates_together(self):
        rows = [
            synthetic_pose_row("synthetic_s0", "p0", 0.000),
            synthetic_pose_row("synthetic_s1", "p1", 0.005),  # near duplicate of s0
            synthetic_pose_row("synthetic_s2", "p2", 1.0),
            synthetic_pose_row("synthetic_s3", "p3", 2.0),
            synthetic_pose_row("synthetic_s4", "p4", 3.0),
            synthetic_pose_row("synthetic_s5", "p5", 4.0),
        ]
        result = split_pose_rows(
            rows,
            seed=2026,
            split_mode="auto",
            near_translation_m=0.01,
            near_rotation_deg=0.1,
        )
        self.assertEqual(result.manifest["split_rule"], "session_components")
        self.assertTrue(result.manifest["synthetic"])

        owner = {}
        for name, split_rows in (
            ("train", result.train_rows),
            ("val", result.val_rows),
            ("test", result.test_rows),
        ):
            self.assertTrue(split_rows)
            for row in split_rows:
                key = (row["session_id"], row["pose_id"])
                self.assertNotIn(key, owner)
                owner[key] = name
                for feature in POSE12_FIELDS:
                    self.assertIn("norm_" + feature, row)

        self.assertEqual(owner[("synthetic_s0", "p0")], owner[("synthetic_s1", "p1")])
        split_sessions = [
            {row["session_id"] for row in split_rows}
            for split_rows in (result.train_rows, result.val_rows, result.test_rows)
        ]
        self.assertFalse(split_sessions[0] & split_sessions[1])
        self.assertFalse(split_sessions[0] & split_sessions[2])
        self.assertFalse(split_sessions[1] & split_sessions[2])

    def test_normalization_is_fitted_from_resulting_train_rows_only(self):
        rows = [
            synthetic_pose_row("synthetic_s{}".format(index), "p{}".format(index), index * 10.0)
            for index in range(8)
        ]
        result = split_pose_rows(rows, seed=17, near_translation_m=0.0, near_rotation_deg=0.0)
        train_tx = [float(row["rel_tx"]) for row in result.train_rows]
        tx_index = list(result.normalization["feature_order"]).index("rel_tx")
        expected_mean = sum(train_tx) / len(train_tx)
        self.assertAlmostEqual(result.normalization["mean"][tx_index], expected_mean)
        self.assertEqual(result.normalization["fitted_on"], "train")
        self.assertEqual(result.normalization["train_min"][tx_index], min(train_tx))
        self.assertEqual(result.normalization["train_max"][tx_index], max(train_tx))

        all_mean = sum(float(row["rel_tx"]) for row in rows) / len(rows)
        if expected_mean != all_mean:
            self.assertNotEqual(result.normalization["mean"][tx_index], all_mean)
        val_first = result.val_rows[0]
        scale = result.normalization["std"][tx_index]
        self.assertAlmostEqual(
            val_first["norm_rel_tx"],
            (float(val_first["rel_tx"]) - expected_mean) / scale,
        )

    def test_one_session_uses_explicit_pose_geometry_fallback(self):
        rows = [
            synthetic_pose_row("synthetic_single_session", "p{}".format(index), float(index))
            for index in range(6)
        ]
        result = split_pose_rows(rows, seed=42, split_mode="auto")
        self.assertEqual(result.manifest["split_rule"], "pose_geometry_groups")
        self.assertTrue(result.manifest["warnings"])
        keys = [
            {(row["session_id"], row["pose_id"]) for row in split_rows}
            for split_rows in (result.train_rows, result.val_rows, result.test_rows)
        ]
        self.assertFalse(keys[0] & keys[1])
        self.assertFalse(keys[0] & keys[2])
        self.assertFalse(keys[1] & keys[2])

    def test_cross_session_near_duplicates_that_collapse_all_sessions_fail_clearly(self):
        rows = [
            synthetic_pose_row("synthetic_s{}".format(index), "p{}".format(index), 1.0)
            for index in range(3)
        ]
        with self.assertRaisesRegex(DatasetError, "independent session components"):
            split_pose_rows(rows, split_mode="auto")

    def test_mixed_synthetic_and_real_provenance_is_rejected(self):
        rows = [
            synthetic_pose_row("synthetic_s{}".format(index), "p{}".format(index), float(index))
            for index in range(6)
        ]
        rows[-1]["data_origin"] = "experimental"
        with self.assertRaisesRegex(DatasetError, "mixed dataset provenance"):
            split_pose_rows(rows)

    def test_unmarked_provenance_stays_unknown_not_real(self):
        rows = [
            synthetic_pose_row("unmarked_s{}".format(index), "p{}".format(index), float(index))
            for index in range(6)
        ]
        for row in rows:
            row.pop("data_origin")
        result = split_pose_rows(rows)
        self.assertEqual(result.manifest["data_provenance"], "unknown")
        self.assertIsNone(result.manifest["synthetic"])

    def test_duplicate_pose_key_is_rejected(self):
        rows = [
            synthetic_pose_row("synthetic_s0", "same", 0.0),
            synthetic_pose_row("synthetic_s0", "same", 1.0),
            synthetic_pose_row("synthetic_s1", "p1", 2.0),
        ]
        with self.assertRaisesRegex(DatasetError, "duplicate aggregate pose key"):
            split_pose_rows(rows, split_mode="pose")

    def test_split_outputs_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="uwb_synthetic_split_test_") as directory:
            output = Path(directory) / "train.csv"
            write_csv_exclusive(output, [{"data_origin": "synthetic_test"}], ["data_origin"])
            with self.assertRaises(OutputExistsError):
                assert_distinct_new_outputs([], [output])

    def test_fit_normalization_rejects_empty_train(self):
        with self.assertRaisesRegex(DatasetError, "empty training"):
            fit_normalization([])


if __name__ == "__main__":
    unittest.main()
