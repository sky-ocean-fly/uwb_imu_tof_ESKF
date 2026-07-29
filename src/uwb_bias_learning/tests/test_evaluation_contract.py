import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.lineage import SplitContract, file_sha256  # noqa: E402
from uwb_bias_learning.models import POSE12_FEATURES  # noqa: E402


def load_script(name):
    path = PACKAGE_ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(
        "uwb_bias_learning_test_{}".format(name), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load_script("evaluate")
export_model = load_script("export_model")


def pose12_values():
    return [0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


class EvaluationContractTests(unittest.TestCase):
    def make_contract(self, root, provenance="synthetic", raw_status="valid"):
        artifacts = {}
        poses = {"train": "P1", "val": "P2", "test": "P3"}
        for split, pose_id in poses.items():
            path = root / (split + ".csv")
            fields = ["session_id", "pose_id", *POSE12_FEATURES, "bias_mean"]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            ["S1", pose_id, *pose12_values(), 0.02],
                        )
                    )
                )
            artifacts[split] = path

        raw = root / "raw_test.csv"
        raw_fields = [
            "session_id",
            "pose_id",
            "range_m",
            "distance_gt_m",
            "processing_status",
            "is_outlier",
            "is_los",
            "is_stable",
            "is_mocap_valid",
            "is_sync_valid",
            "is_range_valid",
            "data_origin",
            "is_synthetic",
            *POSE12_FEATURES,
        ]
        raw_row = {
            "session_id": "S1",
            "pose_id": "P3",
            "range_m": "0.42",
            "distance_gt_m": "0.4",
            "processing_status": raw_status,
            "is_outlier": "false",
            "is_los": "true",
            "is_stable": "true",
            "is_mocap_valid": "true",
            "is_sync_valid": "true",
            "is_range_valid": "true",
            "data_origin": "synthetic_fixture" if provenance == "synthetic" else "rosbag",
            "is_synthetic": "true" if provenance == "synthetic" else "false",
        }
        raw_row.update(dict(zip(POSE12_FEATURES, pose12_values())))
        with raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw_fields)
            writer.writeheader()
            writer.writerow(raw_row)
        artifacts["raw_test"] = raw

        normalization = root / "normalization.json"
        normalization.write_text("{}\n", encoding="utf-8")
        artifacts["normalization"] = normalization
        payload = {
            "data_provenance": provenance,
            "splits": {
                split: {
                    "pose_keys": [{"session_id": "S1", "pose_id": pose_id}]
                }
                for split, pose_id in poses.items()
            },
            "artifacts": {
                name: {"file": path.name, "sha256": file_sha256(path)}
                for name, path in artifacts.items()
            },
        }
        manifest = root / "splits.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return SplitContract.load(
            manifest, allow_synthetic=provenance == "synthetic"
        ), artifacts

    def test_raw_rows_must_match_contract_provenance(self):
        fields = [
            "session_id", "pose_id", "range_m", "distance_gt_m",
            "processing_status", "is_outlier", "is_los", "is_stable",
            "is_mocap_valid", "is_sync_valid", "is_range_valid",
            "data_origin", "is_synthetic", *POSE12_FEATURES,
        ]
        row = {name: "0" for name in fields}
        row.update(
            {
                "processing_status": "valid",
                "is_outlier": "false",
                "is_los": "true",
                "is_stable": "true",
                "is_mocap_valid": "true",
                "is_sync_valid": "true",
                "is_range_valid": "true",
                "data_origin": "fixture",
                "is_synthetic": "true",
            }
        )
        evaluate.validate_raw_test_rows(fields, [row], Path("raw.csv"), "synthetic")
        with self.assertRaisesRegex(ValueError, "data_provenance"):
            evaluate.validate_raw_test_rows(fields, [row], Path("raw.csv"), "real")

    def test_model_lineage_binds_split_and_normalization(self):
        with tempfile.TemporaryDirectory(prefix="uwb_eval_contract_") as directory:
            contract, artifacts = self.make_contract(Path(directory))
            expected = contract.base_lineage()
            normalization_hash = file_sha256(artifacts["normalization"])
            payload = {
                "data_provenance": "synthetic",
                "lineage": {**expected, "normalization_sha256": normalization_hash},
            }
            evaluate.validate_model_lineage(
                payload, expected, normalization_hash, "artifact"
            )
            payload["lineage"]["val_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "val_sha256"):
                evaluate.validate_model_lineage(
                    payload, expected, normalization_hash, "artifact"
                )

    def test_export_contract_checks_every_supplied_artifact_hash(self):
        with tempfile.TemporaryDirectory(prefix="uwb_export_contract_") as directory:
            contract, artifacts = self.make_contract(Path(directory))
            expected = contract.base_lineage()
            norm_hash = file_sha256(artifacts["normalization"])
            model_hash = "a" * 64
            config_hash = "b" * 64
            raw_hash = file_sha256(artifacts["raw_test"])
            checkpoint = {
                "data_provenance": "synthetic",
                "lineage": {
                    **expected,
                    "normalization_sha256": norm_hash,
                    "train_config_saved_sha256": config_hash,
                    "train_config_sha256": config_hash,
                }
            }
            metrics_lineage = {
                **expected,
                "raw_test_sha256": raw_hash,
                "mlp_state_sha256": model_hash,
                "normalization_sha256": norm_hash,
                "train_config_sha256": config_hash,
            }
            metrics = {
                "evaluation_split": "fixed_test_only",
                "data_provenance": "synthetic",
                "test_only": True,
                "lineage": metrics_lineage,
                "split_manifest_sha256": contract.sha256,
                "test_sha256": expected["test_sha256"],
                "raw_test_sha256": raw_hash,
                "mlp_state_sha256": model_hash,
                "normalization_sha256": norm_hash,
                "train_config_sha256": config_hash,
            }
            export_model.validate_export_lineage(
                contract,
                metrics,
                checkpoint,
                mlp_state_sha256=model_hash,
                normalization_sha256=norm_hash,
                train_config_sha256=config_hash,
            )
            metrics["train_config_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "train_config_sha256"):
                export_model.validate_export_lineage(
                    contract,
                    metrics,
                    checkpoint,
                    mlp_state_sha256=model_hash,
                    normalization_sha256=norm_hash,
                    train_config_sha256=config_hash,
                )

    def test_invalid_raw_input_creates_no_report_directory(self):
        with tempfile.TemporaryDirectory(prefix="uwb_eval_no_partial_") as directory:
            root = Path(directory)
            contract, artifacts = self.make_contract(root, raw_status="rejected")
            baselines = root / "baselines.json"
            baselines.write_text("{}\n", encoding="utf-8")
            model_state = root / "model_state.pt"
            model_state.write_bytes(b"not-read-before-raw-validation")
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "processing_status"):
                evaluate.main(
                    [
                        "--test", str(artifacts["test"]),
                        "--raw-test", str(artifacts["raw_test"]),
                        "--baselines", str(baselines),
                        "--mlp-state", str(model_state),
                        "--normalization", str(artifacts["normalization"]),
                        "--split-manifest", str(contract.path),
                        "--output-root", str(output),
                        "--skip-figures",
                        "--allow-synthetic",
                    ]
                )
            self.assertFalse((output / "reports").exists())

    def test_export_requires_a_path_that_does_not_exist(self):
        with tempfile.TemporaryDirectory(prefix="uwb_export_no_overwrite_") as directory:
            root = Path(directory)
            sources = []
            for name in (
                "model_state.pt",
                "normalization.json",
                "train_config.yaml",
                "splits.json",
                "metrics.json",
            ):
                path = root / name
                path.write_text("{}\n", encoding="utf-8")
                sources.append(path)
            output = root / "existing_empty_output"
            output.mkdir()
            with self.assertRaises(SystemExit):
                export_model.main(
                    [
                        "--model-state", str(sources[0]),
                        "--normalization", str(sources[1]),
                        "--train-config", str(sources[2]),
                        "--split-manifest", str(sources[3]),
                        "--metrics", str(sources[4]),
                        "--output-dir", str(output),
                        "--model-version", "software-test",
                    ]
                )
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
