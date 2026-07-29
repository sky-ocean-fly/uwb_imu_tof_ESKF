#!/usr/bin/env python3
"""Unit tests using explicitly marked synthetic rows only."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from uwb_bias_learning.datasets import (
    AggregationConfig,
    DatasetError,
    OutlierConfig,
    OutputExistsError,
    aggregate_pose_rows,
    compare_coverage_to_pose_domain,
    file_sha256,
    ordered_fieldnames,
    read_csv_rows,
    validate_raw_rows,
    write_csv_exclusive,
    write_json_exclusive,
)


def synthetic_raw_row(
    sample_index,
    bias_m,
    *,
    session_id="synthetic_session_01",
    pose_id="synthetic_pose_01",
    position_b=(2.0, 0.0, 0.0),
    quaternion_b=(0.0, 0.0, 0.0, 1.0),
):
    distance = math.sqrt(sum(value * value for value in position_b))
    timestamp = 1000.0 + 0.1 * sample_index
    return {
        "data_origin": "synthetic_test",
        "is_synthetic": "true",
        "ingestion_path": "synthetic_unit_test",
        "session_id": session_id,
        "environment_id": "synthetic_environment",
        "pose_id": pose_id,
        "attempt_id": "synthetic_attempt",
        "sample_index": str(sample_index),
        "capture_state": "COLLECTING",
        "uwb_timestamp": timestamp,
        "uwb_receive_timestamp": timestamp + 0.001,
        "uwb_timestamp_source": "synthetic_test_clock",
        "range_m": distance + bias_m,
        "source_id": "synthetic_A",
        "peer_id": "synthetic_B",
        "anchor_id": "synthetic_A",
        "node_id": "synthetic_B",
        "fp_rssi": -80.0 - sample_index,
        "rx_rssi": -75.0 - sample_index,
        "cir_quality": "",
        "mocap_a_timestamp_before": timestamp - 0.005,
        "mocap_a_timestamp_after": timestamp + 0.005,
        "mocap_b_timestamp_before": timestamp - 0.005,
        "mocap_b_timestamp_after": timestamp + 0.005,
        "sync_gap_a_s": 0.005,
        "sync_gap_b_s": 0.005,
        "A_px": 0.0,
        "A_py": 0.0,
        "A_pz": 0.0,
        "A_qx": 0.0,
        "A_qy": 0.0,
        "A_qz": 0.0,
        "A_qw": 1.0,
        "B_px": position_b[0],
        "B_py": position_b[1],
        "B_pz": position_b[2],
        "B_qx": quaternion_b[0],
        "B_qy": quaternion_b[1],
        "B_qz": quaternion_b[2],
        "B_qw": quaternion_b[3],
        "is_los": "true",
        "is_stable": "true",
        "is_sync_valid": "true",
        "is_range_valid": "true",
        "is_mocap_valid": "true",
        "invalid_reason": "",
        "operator_note": "synthetic unit-test row; not an experimental result",
    }


def synthetic_config(min_valid_ranges=3, outlier=None):
    return AggregationConfig(
        min_valid_ranges=min_valid_ranges,
        expected_anchor_id="synthetic_A",
        expected_node_id="synthetic_B",
        require_los=True,
        outlier=outlier or OutlierConfig(enabled=False),
    )


class DatasetAggregationTest(unittest.TestCase):
    def test_preprocessing_mapping_aliases_are_applied(self):
        config = AggregationConfig.from_mapping(
            {
                "min_valid_ranges": 7,
                "required_flags": {"is_los": False},
                "outlier_filter": {
                    "enabled": True,
                    "method": "hampel_mad",
                    "threshold_mad": 3.5,
                    "minimum_scale_m": 0.002,
                },
            }
        )
        self.assertEqual(config.min_valid_ranges, 7)
        self.assertFalse(config.require_los)
        self.assertEqual(config.outlier.threshold, 3.5)
        self.assertEqual(config.outlier.minimum_absolute_deviation_m, 0.002)

    def test_synthetic_aggregate_mean_sample_std_and_sem(self):
        rows = [
            synthetic_raw_row(0, 0.1),
            synthetic_raw_row(1, 0.2, quaternion_b=(0.0, 0.0, 0.0, -1.0)),
            synthetic_raw_row(2, 0.3),
        ]
        original = copy.deepcopy(rows)
        result = aggregate_pose_rows(rows, synthetic_config())

        self.assertEqual(rows, original, "aggregation must not mutate raw rows")
        self.assertEqual(len(result.pose_rows), 1)
        self.assertEqual(result.rejected_pose_rows, [])
        pose = result.pose_rows[0]
        self.assertEqual(pose["quality_status"], "accepted")
        self.assertEqual(pose["range_count_raw"], 3)
        self.assertEqual(pose["range_count_valid"], 3)
        self.assertAlmostEqual(pose["bias_mean_raw"], 0.2, places=12)
        self.assertAlmostEqual(pose["bias_mean"], 0.2, places=12)
        self.assertAlmostEqual(pose["bias_std"], 0.1, places=12)
        self.assertAlmostEqual(pose["bias_sem"], 0.1 / math.sqrt(3.0), places=12)
        self.assertAlmostEqual(pose["rel_tx"], 2.0, places=12)
        self.assertAlmostEqual(pose["rel_qw"], 1.0, places=12)
        self.assertAlmostEqual(pose["rel_R00"], 1.0, places=12)
        self.assertAlmostEqual(pose["rel_R11"], 1.0, places=12)
        self.assertAlmostEqual(pose["rel_R22"], 1.0, places=12)
        self.assertTrue(all(row["processing_status"] == "valid" for row in result.annotated_rows))

    def test_explicit_source_peer_ids_take_priority_without_reverse_fallback(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(3)]
        for row in rows:
            row["anchor_id"] = ""
            row["node_id"] = ""
            row["source_id"] = "synthetic_A"
            row["peer_id"] = "synthetic_B"
        accepted = aggregate_pose_rows(rows, synthetic_config())
        self.assertEqual(len(accepted.pose_rows), 1)

        reversed_rows = copy.deepcopy(rows)
        for row in reversed_rows:
            row["source_id"] = "synthetic_B"
            row["peer_id"] = "synthetic_A"
            # Legacy-looking values must not override explicit directed fields.
            row["anchor_id"] = "synthetic_A"
            row["node_id"] = "synthetic_B"
        rejected = aggregate_pose_rows(reversed_rows, synthetic_config())
        self.assertEqual(rejected.pose_rows, [])
        self.assertTrue(
            all(
                "id_pair_mismatch" in row["processing_invalid_reason"]
                for row in rejected.annotated_rows
            )
        )

    def test_synthetic_hampel_retains_before_and_after_statistics(self):
        rows = [synthetic_raw_row(index, bias) for index, bias in enumerate((0.1, 0.1, 0.1, 2.0))]
        outlier = OutlierConfig(
            enabled=True,
            method="hampel_mad",
            field="range_error_m",
            threshold=3.0,
            mad_scale=1.4826,
            min_samples=4,
            minimum_absolute_deviation_m=0.001,
        )
        result = aggregate_pose_rows(rows, synthetic_config(3, outlier))
        pose = result.pose_rows[0]

        self.assertEqual(pose["range_count_raw"], 4)
        self.assertEqual(pose["range_count_valid"], 3)
        self.assertEqual(pose["outlier_count"], 1)
        self.assertAlmostEqual(pose["bias_mean_raw"], 0.575, places=12)
        self.assertAlmostEqual(pose["bias_mean"], 0.1, places=12)
        self.assertIn("hampel_mad", result.annotated_rows[-1]["outlier_reason"])
        self.assertEqual(result.annotated_rows[-1]["processing_status"], "outlier")
        self.assertEqual(result.annotated_rows[-1]["invalid_reason"], "")

    def test_synthetic_invalid_reason_preserved_and_minimum_count_rejects_pose(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(4)]
        rows[-1]["is_sync_valid"] = "false"
        rows[-1]["invalid_reason"] = "synthetic_sync_failure"
        rows[-1]["processing_status"] = "synthetic_upstream_status"
        result = aggregate_pose_rows(rows, synthetic_config(min_valid_ranges=4))

        self.assertEqual(result.pose_rows, [])
        self.assertEqual(len(result.rejected_pose_rows), 1)
        rejected = result.rejected_pose_rows[0]
        self.assertEqual(rejected["range_count_raw"], 3)
        self.assertIn("insufficient_valid_ranges:3<4", rejected["rejection_reason"])
        annotated = result.annotated_rows[-1]
        self.assertEqual(annotated["invalid_reason"], "synthetic_sync_failure")
        self.assertEqual(
            annotated["source_processing_status"], "synthetic_upstream_status"
        )
        self.assertIn("upstream_invalid:synthetic_sync_failure", annotated["processing_invalid_reason"])
        self.assertIn("is_sync_valid_false", annotated["processing_invalid_reason"])

    def test_synthetic_raw_validator_reports_explicit_provenance_and_duplicates(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(3)]
        rows.append(copy.deepcopy(rows[-1]))
        result = validate_raw_rows(rows, synthetic_config(2), fail_incomplete_poses=False)

        self.assertTrue(result.report["synthetic"])
        self.assertFalse(result.passed)
        self.assertEqual(result.report["duplicate_counts"]["exact_duplicate_count"], 1)
        self.assertEqual(result.report["duplicate_counts"]["duplicate_sample_key_count"], 1)

    def test_live_preview_is_rejected_for_formal_dataset_construction(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(3)]
        for row in rows:
            row["data_origin"] = "physical_experiment"
            row["is_synthetic"] = "false"
            row["ingestion_path"] = "ros_live"

        validation = validate_raw_rows(
            rows, synthetic_config(), fail_incomplete_poses=False
        )
        self.assertFalse(validation.passed)
        self.assertTrue(
            any("operational preview" in error for error in validation.severe_errors)
        )
        with self.assertRaisesRegex(DatasetError, "operational preview"):
            aggregate_pose_rows(rows, synthetic_config())

    def test_physical_rosbag_replay_is_the_canonical_real_path(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(3)]
        for row in rows:
            row["data_origin"] = "physical_experiment"
            row["is_synthetic"] = "false"
            row["ingestion_path"] = "rosbag_replay"

        validation = validate_raw_rows(
            rows, synthetic_config(), fail_incomplete_poses=False
        )
        self.assertTrue(validation.passed, validation.severe_errors)
        self.assertEqual(validation.report["ingestion_path"], "rosbag_replay")
        result = aggregate_pose_rows(rows, synthetic_config())
        self.assertEqual(result.summary["ingestion_path"], "rosbag_replay")

    def test_synthetic_ingestion_requires_explicit_synthetic_provenance(self):
        rows = [synthetic_raw_row(index, 0.1) for index in range(3)]
        for row in rows:
            row["data_origin"] = ""
            row["is_synthetic"] = ""

        validation = validate_raw_rows(
            rows, synthetic_config(), fail_incomplete_poses=False
        )
        self.assertFalse(validation.passed)
        self.assertTrue(
            any("provenance must be explicitly" in error for error in validation.severe_errors)
        )
        with self.assertRaisesRegex(DatasetError, "provenance must be explicitly"):
            aggregate_pose_rows(rows, synthetic_config())

    def test_synthetic_invalid_stored_rotation_is_rejected_not_projected(self):
        row = synthetic_raw_row(0, 0.1)
        row.update(
            {
                "rel_tx": 2.0,
                "rel_ty": 0.0,
                "rel_tz": 0.0,
                "rel_qx": 0.0,
                "rel_qy": 0.0,
                "rel_qz": 0.0,
                "rel_qw": 1.0,
                "rel_R00": -1.0,
                "rel_R01": 0.0,
                "rel_R02": 0.0,
                "rel_R10": 0.0,
                "rel_R11": 1.0,
                "rel_R12": 0.0,
                "rel_R20": 0.0,
                "rel_R21": 0.0,
                "rel_R22": 1.0,
            }
        )
        result = aggregate_pose_rows([row], synthetic_config(min_valid_ranges=1))
        self.assertEqual(result.pose_rows, [])
        self.assertIn(
            "rotation determinant is not +1",
            result.annotated_rows[0]["processing_invalid_reason"],
        )

    def test_pose_domain_comparison_does_not_guess_euler_sequence(self):
        comparison = compare_coverage_to_pose_domain(
            {
                "distance_m": {"min": 0.5, "max": 2.1},
                "azimuth_deg": {"min": -20.0, "max": 20.0},
                "elevation_deg": {"min": -5.0, "max": 5.0},
            },
            {
                "distance_m": {"min": 1.0, "max": 2.0},
                "azimuth_deg": {"min": -30.0, "max": 30.0},
                "elevation_deg": {"min": None, "max": None},
                "relative_roll_deg": {"min": -10.0, "max": 10.0},
                "relative_pitch_deg": {"min": -10.0, "max": 10.0},
                "relative_yaw_deg": {"min": -30.0, "max": 30.0},
            },
        )
        self.assertEqual(len(comparison["violations"]), 1)
        self.assertEqual(
            comparison["axes"]["relative_yaw_deg"]["status"], "not_evaluated"
        )
        self.assertIn("not confirmed", comparison["axes"]["relative_yaw_deg"]["reason"])

    def test_exclusive_csv_and_json_writers_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="uwb_synthetic_dataset_test_") as directory:
            root = Path(directory)
            csv_path = root / "synthetic_rows.csv"
            json_path = root / "synthetic_report.json"
            rows = [{"data_origin": "synthetic_test", "value": 1.25}]
            write_csv_exclusive(csv_path, rows, ("data_origin", "value"))
            loaded, fields = read_csv_rows(csv_path)
            self.assertEqual(fields, ["data_origin", "value"])
            self.assertEqual(loaded[0]["data_origin"], "synthetic_test")
            with self.assertRaises(OutputExistsError):
                write_csv_exclusive(csv_path, rows, ("data_origin", "value"))

            write_json_exclusive(json_path, {"synthetic": True})
            self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["synthetic"])
            with self.assertRaises(OutputExistsError):
                write_json_exclusive(json_path, {"synthetic": True})

    def test_synthetic_command_line_validation_build_and_split_pipeline(self):
        package_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="uwb_synthetic_e2e_test_") as directory:
            root = Path(directory)
            rows = []
            for session_index in range(6):
                for sample_index, bias in enumerate((0.09, 0.10, 0.11)):
                    rows.append(
                        synthetic_raw_row(
                            sample_index,
                            bias,
                            session_id="synthetic_session_{:02d}".format(session_index),
                            pose_id="synthetic_pose_{:02d}".format(session_index),
                            position_b=(1.0 + session_index, 0.0, 0.0),
                        )
                    )
            raw_csv = root / "synthetic_raw.csv"
            config_json = root / "synthetic_preprocessing.json"
            pose_domain_json = root / "synthetic_pose_domain.json"
            write_csv_exclusive(raw_csv, rows, ordered_fieldnames(rows))
            write_json_exclusive(
                config_json,
                {
                    "min_valid_ranges": 3,
                    "expected_anchor_id": "synthetic_A",
                    "expected_node_id": "synthetic_B",
                    "required_flags": {"is_los": True},
                    "outlier_filter": {"enabled": False, "method": "hampel_mad"},
                    "near_duplicate_pose": {
                        "translation_tolerance_m": 0.01,
                        "rotation_tolerance_deg": 0.1,
                    },
                    "splits": {
                        "seed": 2026,
                        "train_fraction": 0.7,
                        "val_fraction": 0.15,
                        "test_fraction": 0.15,
                    },
                },
            )
            write_json_exclusive(
                pose_domain_json,
                {
                    "distance_m": {"min": 0.5, "max": 6.5},
                    "azimuth_deg": {"min": -1.0, "max": 1.0},
                    "elevation_deg": {"min": -1.0, "max": 1.0},
                    "relative_roll_deg": {"min": -1.0, "max": 1.0},
                    "relative_pitch_deg": {"min": -1.0, "max": 1.0},
                    "relative_yaw_deg": {"min": -1.0, "max": 1.0},
                },
            )

            def run_script(name, *arguments):
                return subprocess.run(
                    [sys.executable, str(package_root / "scripts" / name)]
                    + [str(value) for value in arguments],
                    cwd=str(package_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False,
                )

            validation = run_script(
                "validate_raw_dataset.py",
                "--input",
                raw_csv,
                "--config",
                config_json,
                "--pose-domain-config",
                pose_domain_json,
                "--report",
                root / "synthetic_validation.json",
                "--annotated-output",
                root / "synthetic_validation_rows.csv",
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)

            pose_csv = root / "synthetic_poses.csv"
            build = run_script(
                "build_pose_dataset.py",
                "--input",
                raw_csv,
                "--config",
                config_json,
                "--output",
                pose_csv,
                "--rejected-output",
                root / "synthetic_rejected_poses.csv",
                "--annotated-output",
                root / "synthetic_build_rows.csv",
                "--summary-report",
                root / "synthetic_build_summary.json",
            )
            self.assertEqual(build.returncode, 0, build.stderr)

            split_dir = root / "synthetic_splits"
            split = run_script(
                "make_splits.py",
                "--input",
                pose_csv,
                "--config",
                config_json,
                "--annotated-raw",
                root / "synthetic_build_rows.csv",
                "--output-dir",
                split_dir,
            )
            self.assertEqual(split.returncode, 0, split.stderr)
            manifest = json.loads((split_dir / "splits.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["synthetic"])
            self.assertTrue(manifest["train_sessions"])
            self.assertTrue((split_dir / "normalization.json").is_file())
            self.assertEqual(
                manifest["artifacts"]["aggregate_input"]["sha256"],
                file_sha256(pose_csv),
            )
            for artifact_name, filename in (
                ("train", "train.csv"),
                ("val", "val.csv"),
                ("test", "test.csv"),
                ("normalization", "normalization.json"),
                ("raw_test", "raw_test.csv"),
            ):
                self.assertEqual(
                    manifest["artifacts"][artifact_name]["sha256"],
                    file_sha256(split_dir / filename),
                )
            raw_test_rows, raw_test_fields = read_csv_rows(split_dir / "raw_test.csv")
            self.assertTrue(raw_test_rows)
            self.assertIn("distance_gt_m", raw_test_fields)
            self.assertTrue(
                all(row["processing_status"] == "valid" for row in raw_test_rows)
            )
            self.assertTrue(all(row["is_outlier"] == "false" for row in raw_test_rows))

            repeated = run_script(
                "make_splits.py",
                "--input",
                pose_csv,
                "--config",
                config_json,
                "--annotated-raw",
                root / "synthetic_build_rows.csv",
                "--output-dir",
                split_dir,
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("refusing to overwrite", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
