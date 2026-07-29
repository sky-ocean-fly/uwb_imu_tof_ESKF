"""Synthetic-only checks for training input lineage and overwrite safety."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.lineage import LineageError, file_sha256  # noqa: E402
from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    FeatureNormalizer,
)


def _load_script(name):
    path = PACKAGE_ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("test_" + name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_baselines = _load_script("train_baselines")
train_mlp = _load_script("train_mlp")


class TrainingLineageTests(unittest.TestCase):
    def _fixture(self, root):
        rows_by_split = {
            "train": [("S1", "P1", 0.01), ("S1", "P2", 0.02), ("S2", "P3", 0.04)],
            "val": [("S3", "P4", 0.03), ("S3", "P5", 0.05)],
        }
        features_by_split = {}
        paths = {}
        fieldnames = ["session_id", "pose_id"] + list(POSE12_FEATURES) + ["bias_mean"]
        for split, descriptions in rows_by_split.items():
            path = root / (split + ".csv")
            numeric_rows = []
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for row_index, (session_id, pose_id, target) in enumerate(descriptions):
                    offset = 10 if split == "val" else 0
                    values = [
                        float((row_index + 1 + offset) * (column + 1)) / 100.0
                        for column in range(len(POSE12_FEATURES))
                    ]
                    numeric_rows.append(values)
                    writer.writerow(
                        {
                            "session_id": session_id,
                            "pose_id": pose_id,
                            "bias_mean": target,
                            **dict(zip(POSE12_FEATURES, values)),
                        }
                    )
            paths[split] = path
            features_by_split[split] = np.asarray(numeric_rows, dtype=np.float64)

        normalization = root / "normalization.json"
        with normalization.open("w", encoding="utf-8") as stream:
            json.dump(
                FeatureNormalizer.fit(features_by_split["train"]).to_dict(),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        paths["normalization"] = normalization

        # The test artifact intentionally does not exist.  A train/val command
        # may propagate its pre-recorded hash, but must never open test.csv.
        test_path = root / "must_not_be_opened_test.csv"
        manifest_payload = {
            "schema_version": 1,
            "data_provenance": "synthetic",
            "splits": {
                name: {
                    "pose_keys": [
                        {"session_id": session_id, "pose_id": pose_id}
                        for session_id, pose_id, _ in descriptions
                    ]
                }
                for name, descriptions in rows_by_split.items()
            },
            "artifacts": {
                "train": {"file": paths["train"].name, "sha256": file_sha256(paths["train"])},
                "val": {"file": paths["val"].name, "sha256": file_sha256(paths["val"])},
                "test": {"file": test_path.name, "sha256": "a" * 64},
                "normalization": {
                    "file": normalization.name,
                    "sha256": file_sha256(normalization),
                },
            },
        }
        manifest_payload["splits"]["test"] = {
            "pose_keys": [{"session_id": "S9", "pose_id": "P9"}]
        }
        manifest = root / "splits.json"
        manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        paths["manifest"] = manifest
        return paths, manifest_payload

    def test_baseline_cli_binds_synthetic_run_and_does_not_read_test(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_software_test_") as directory:
            root = Path(directory)
            paths, _ = self._fixture(root)
            output = root / "baseline_run"
            result = train_baselines.main(
                [
                    "--train", str(paths["train"]),
                    "--val", str(paths["val"]),
                    "--split-manifest", str(paths["manifest"]),
                    "--output-dir", str(output),
                    "--ridge-alpha", "0.01",
                    "--allow-synthetic",
                ]
            )
            self.assertEqual(result, 0)
            bundle = json.loads((output / "baselines.json").read_text(encoding="utf-8"))
            lineage = bundle["lineage"]
            self.assertEqual(bundle["data_provenance"], "synthetic")
            self.assertEqual(lineage["data_provenance"], "synthetic")
            self.assertEqual(lineage["test_sha256"], "a" * 64)
            self.assertEqual(
                lineage["normalization_sha256"],
                file_sha256(output / "normalization.json"),
            )
            self.assertEqual(
                lineage["train_config_saved_sha256"],
                file_sha256(output / "train_config.json"),
            )
            self.assertEqual(
                file_sha256(output / "split_manifest.json"),
                file_sha256(paths["manifest"]),
            )

    def test_synthetic_contract_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_software_test_") as directory:
            root = Path(directory)
            paths, _ = self._fixture(root)
            output = root / "must_not_exist"
            with self.assertRaises(LineageError):
                train_baselines.main(
                    [
                        "--train", str(paths["train"]),
                        "--val", str(paths["val"]),
                        "--split-manifest", str(paths["manifest"]),
                        "--output-dir", str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_semantically_wrong_normalization_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_software_test_") as directory:
            root = Path(directory)
            paths, manifest_payload = self._fixture(root)
            normalization = json.loads(paths["normalization"].read_text(encoding="utf-8"))
            normalization["mean"][0] += 1.0
            paths["normalization"].write_text(
                json.dumps(normalization, indent=2), encoding="utf-8"
            )
            manifest_payload["artifacts"]["normalization"]["sha256"] = file_sha256(
                paths["normalization"]
            )
            paths["manifest"].write_text(
                json.dumps(manifest_payload, indent=2), encoding="utf-8"
            )
            output = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "does not match values recomputed"):
                train_baselines.main(
                    [
                        "--train", str(paths["train"]),
                        "--val", str(paths["val"]),
                        "--split-manifest", str(paths["manifest"]),
                        "--output-dir", str(output),
                        "--allow-synthetic",
                    ]
                )
            self.assertFalse(output.exists())

    def test_mlp_and_baseline_share_strict_normalization_check(self):
        mean = np.arange(len(POSE12_FEATURES), dtype=np.float64)
        supplied = FeatureNormalizer(
            mean=mean,
            scale=np.ones_like(mean),
            train_min=mean,
            train_max=mean,
        )
        recomputed = FeatureNormalizer(
            mean=mean + 0.5,
            scale=np.ones_like(mean),
            train_min=mean,
            train_max=mean,
        )
        with self.assertRaises(ValueError):
            train_baselines.verify_normalizer_matches_train(supplied, recomputed)
        with self.assertRaises(ValueError):
            train_mlp.verify_normalizer_matches_train(supplied, recomputed)

    def test_existing_even_empty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_software_test_") as directory:
            root = Path(directory)
            paths, _ = self._fixture(root)
            output = root / "already_exists"
            output.mkdir()
            with self.assertRaises(SystemExit):
                train_baselines.main(
                    [
                        "--train", str(paths["train"]),
                        "--val", str(paths["val"]),
                        "--split-manifest", str(paths["manifest"]),
                        "--output-dir", str(output),
                        "--allow-synthetic",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
