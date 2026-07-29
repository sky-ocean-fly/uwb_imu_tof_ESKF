#!/usr/bin/env python3
"""Train zero, training-mean, and closed-form Ridge baselines.

Only train and validation CSV files are accepted by this command.  Keeping a
test argument out of the interface makes accidental test-set model selection
harder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.metrics import bias_prediction_metrics  # noqa: E402
from uwb_bias_learning.lineage import SplitContract, file_sha256  # noqa: E402
from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    TARGET_COLUMN,
    ConstantBiasModel,
    FeatureNormalizer,
    RidgeBiasModel,
    ZeroBiasModel,
)


def read_pose_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        required = list(POSE12_FEATURES) + [TARGET_COLUMN]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                "{} is missing required columns: {}".format(path, ", ".join(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows: {}".format(path))
    try:
        features = np.asarray(
            [[float(row[name]) for name in POSE12_FEATURES] for row in rows],
            dtype=np.float64,
        )
        targets = np.asarray(
            [float(row[TARGET_COLUMN]) for row in rows], dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("{} contains a non-numeric pose12/target value".format(path)) from exc
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("{} contains NaN or infinite pose12/target values".format(path))
    return features, targets, rows


def pose_keys(rows: Sequence[Dict[str, str]], source: Path) -> set:
    if not rows:
        return set()
    if "session_id" not in rows[0] or "pose_id" not in rows[0]:
        raise ValueError(
            "{} must contain session_id and pose_id for leakage checks".format(source)
        )
    keys = {(row["session_id"], row["pose_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("{} contains duplicate session_id + pose_id rows".format(source))
    return keys


def git_revision() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ).strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(value) for value in payload]
    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, (np.floating, float)):
        value = float(payload)
        return value if math.isfinite(value) else None
    return payload


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_normalizer_matches_train(
    supplied: FeatureNormalizer, recomputed: FeatureNormalizer
) -> None:
    """Reject normalization that was not fitted from this exact train CSV."""

    if supplied.feature_names != recomputed.feature_names:
        raise ValueError("split normalization feature order does not match pose12")
    for name in ("mean", "scale", "train_min", "train_max"):
        recorded = getattr(supplied, name)
        expected = getattr(recomputed, name)
        if not np.allclose(recorded, expected, rtol=1.0e-9, atol=1.0e-12):
            raise ValueError(
                "split normalization {} does not match values recomputed from train.csv".format(
                    name
                )
            )


def verified_training_data(
    train_path: Path,
    val_path: Path,
    split_manifest_path: Path,
    allow_synthetic: bool = False,
) -> Tuple[
    SplitContract,
    FeatureNormalizer,
    Path,
    np.ndarray,
    np.ndarray,
    List[Dict[str, str]],
    np.ndarray,
    np.ndarray,
    List[Dict[str, str]],
]:
    """Load only train/val after binding them to the immutable split contract."""

    contract = SplitContract.load(
        split_manifest_path, allow_synthetic=allow_synthetic
    )
    contract.verify_pose_csv("train", train_path)
    contract.verify_pose_csv("val", val_path)
    normalization_path = contract.artifact_path("normalization")
    contract.verify_artifact("normalization", normalization_path)

    train_x, train_y, train_rows = read_pose_csv(train_path)
    val_x, val_y, val_rows = read_pose_csv(val_path)
    overlap = pose_keys(train_rows, train_path) & pose_keys(val_rows, val_path)
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(
            "train/val leakage: {} session_id + pose_id keys overlap; examples: {}".format(
                len(overlap), examples
            )
        )
    normalizer = FeatureNormalizer.load_json(normalization_path)
    verify_normalizer_matches_train(normalizer, FeatureNormalizer.fit(train_x))
    return (
        contract,
        normalizer,
        normalization_path,
        train_x,
        train_y,
        train_rows,
        val_x,
        val_y,
        val_rows,
    )


def lineage_payload(
    contract: SplitContract,
    normalization_sha256: str,
    train_config_input_sha256: str,
    train_config_saved_sha256: str,
) -> Dict[str, Any]:
    lineage = contract.base_lineage()
    lineage.update(
        {
            "normalization_sha256": normalization_sha256,
            "train_config_input_sha256": train_config_input_sha256,
            "train_config_saved_sha256": train_config_saved_sha256,
            # Compatibility name used by downstream evaluators/exporters.
            "train_config_sha256": train_config_saved_sha256,
        }
    )
    return lineage


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="fixed train.csv")
    parser.add_argument("--val", required=True, type=Path, help="fixed val.csv")
    parser.add_argument(
        "--split-manifest",
        required=True,
        type=Path,
        help="splits.json whose hashes and pose keys bind the fixed split",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for baseline artifacts (existing artifacts are not overwritten)",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0e-3,
        help="fixed L2 coefficient; this command does not inspect the test set",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="software tests only: permit a manifest marked data_provenance=synthetic",
    )
    args = parser.parse_args(argv)

    if args.ridge_alpha < 0.0 or not np.isfinite(args.ridge_alpha):
        parser.error("--ridge-alpha must be finite and non-negative")
    for source in (args.train, args.val, args.split_manifest):
        if not source.is_file():
            parser.error("input file does not exist: {}".format(source))
    if args.output_dir.exists():
        parser.error(
            "refusing to write into existing {}; choose a new output directory".format(
                args.output_dir
            )
        )

    (
        contract,
        normalizer,
        normalization_source,
        train_x,
        train_y,
        train_rows,
        val_x,
        val_y,
        val_rows,
    ) = verified_training_data(
        args.train,
        args.val,
        args.split_manifest,
        allow_synthetic=args.allow_synthetic,
    )
    zero = ZeroBiasModel()
    constant = ConstantBiasModel.fit(train_y)
    ridge = RidgeBiasModel.fit(
        train_x, train_y, alpha=args.ridge_alpha, normalizer=normalizer
    )
    models = {"zero": zero, "constant": constant, "ridge": ridge}
    split_data = {"train": (train_x, train_y), "val": (val_x, val_y)}
    metric_payload: Dict[str, Any] = {
        "schema_version": 1,
        "units": "metres",
        "error_definition": "predicted_bias_m - true_bias_m",
        "selection_data": "validation_only",
        "test_set_used": False,
        "data_provenance": contract.data_provenance,
        "splits": {},
    }
    for split_name, (features, targets) in split_data.items():
        metric_payload["splits"][split_name] = {
            model_name: bias_prediction_metrics(targets, model.predict(features))
            for model_name, model in models.items()
        }

    train_config = {
        "schema_version": 1,
        "command": "train_baselines",
        "feature_set": "pose12",
        "feature_names": list(POSE12_FEATURES),
        "target_column": TARGET_COLUMN,
        "ridge_alpha": args.ridge_alpha,
        "selection_data": "validation_only",
        "test_set_used": False,
    }

    # All validation and fitting above are read-only.  From this point onward a
    # run owns a brand-new directory, so a concurrent invocation cannot merge
    # or overwrite artifacts from another run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protected = args.output_dir / "baselines.json"
    shutil.copyfile(str(normalization_source), str(args.output_dir / "normalization.json"))
    shutil.copyfile(str(contract.path), str(args.output_dir / "split_manifest.json"))
    write_json(args.output_dir / "train_config.json", train_config)
    saved_config_sha256 = file_sha256(args.output_dir / "train_config.json")
    run_normalization_sha256 = file_sha256(args.output_dir / "normalization.json")
    lineage = lineage_payload(
        contract,
        normalization_sha256=run_normalization_sha256,
        train_config_input_sha256=saved_config_sha256,
        train_config_saved_sha256=saved_config_sha256,
    )
    metric_payload["lineage"] = lineage

    bundle = {
        "schema_version": 1,
        "feature_set": "pose12",
        "feature_names": list(POSE12_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_unit": "m",
        "trained_from": "train_only",
        "test_set_used": False,
        "data_provenance": contract.data_provenance,
        "lineage": lineage,
        "models": {
            "zero": zero.to_dict(),
            "constant": constant.to_dict(),
            "ridge": ridge.to_dict(),
        },
    }
    write_json(protected, bundle)
    write_json(args.output_dir / "baseline_metrics.json", metric_payload)

    with (args.output_dir / "baseline_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "split",
            "model",
            "count",
            "mae",
            "rmse",
            "mean_error",
            "error_std",
            "r2",
            "p50_abs_error",
            "p90_abs_error",
            "p95_abs_error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "val"):
            for model_name in ("zero", "constant", "ridge"):
                row = {
                    "split": split_name,
                    "model": model_name,
                    **metric_payload["splits"][split_name][model_name],
                }
                writer.writerow(row)

    run_config = {
        "command": "train_baselines",
        "ridge_alpha": args.ridge_alpha,
        "train_csv": str(args.train),
        "val_csv": str(args.val),
        "train_sha256": file_sha256(args.train),
        "val_sha256": file_sha256(args.val),
        "train_rows": int(train_x.shape[0]),
        "val_rows": int(val_x.shape[0]),
        "test_set_used": False,
        "data_provenance": contract.data_provenance,
        "lineage": lineage,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_json(
        args.output_dir / "versions.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git": git_revision(),
        },
    )
    print("Saved three baselines to {}".format(args.output_dir))
    print("Validation metrics were computed; the test set was not read.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
