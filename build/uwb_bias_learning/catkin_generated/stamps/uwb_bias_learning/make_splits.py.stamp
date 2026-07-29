#!/usr/bin/env python3
"""Create fixed leakage-checked train/val/test pose splits and train-only scaling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.datasets import (  # noqa: E402
    DatasetError,
    assert_distinct_new_outputs,
    dataset_provenance,
    file_sha256,
    load_config_mapping,
    ordered_fieldnames,
    read_csv_rows,
    select_raw_test_rows,
    split_pose_rows,
    write_csv_exclusive,
    write_json_exclusive,
)


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="accepted pose CSV")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for new train.csv, val.csv, test.csv and manifests",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing.yaml",
        help="preprocessing/split YAML or JSON",
    )
    parser.add_argument(
        "--annotated-raw",
        type=Path,
        help=(
            "row-level audit CSV from build_pose_dataset.py; when supplied, "
            "raw_test.csv is selected from the fixed test pose keys"
        ),
    )
    parser.add_argument(
        "--raw-test-output",
        type=Path,
        help="new row-level test CSV (default: OUTPUT_DIR/raw_test.csv)",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument(
        "--split-mode", choices=("auto", "session", "pose"), default="auto"
    )
    parser.add_argument("--near-translation-m", type=float)
    parser.add_argument("--near-rotation-deg", type=float)
    return parser.parse_args(argv)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    if args.raw_test_output is not None and args.annotated_raw is None:
        print(
            "make_splits: --raw-test-output requires --annotated-raw",
            file=sys.stderr,
        )
        return 2
    targets = {
        "train": args.output_dir / "train.csv",
        "val": args.output_dir / "val.csv",
        "test": args.output_dir / "test.csv",
        "manifest": args.output_dir / "splits.json",
        "normalization": args.output_dir / "normalization.json",
    }
    if args.annotated_raw is not None:
        targets["raw_test"] = args.raw_test_output or (args.output_dir / "raw_test.csv")
    try:
        inputs = [args.input, args.config]
        if args.annotated_raw is not None:
            inputs.append(args.annotated_raw)
        assert_distinct_new_outputs(inputs, list(targets.values()))
        config = load_config_mapping(args.config)
        split_config = _mapping(config.get("splits"))
        near_config = _mapping(config.get("near_duplicate_pose"))
        seed = args.seed if args.seed is not None else int(split_config.get("seed", 42))
        train_ratio = (
            args.train_ratio
            if args.train_ratio is not None
            else float(split_config.get("train_fraction", 0.70))
        )
        val_ratio = (
            args.val_ratio
            if args.val_ratio is not None
            else float(split_config.get("val_fraction", 0.15))
        )
        test_ratio = (
            args.test_ratio
            if args.test_ratio is not None
            else float(split_config.get("test_fraction", 0.15))
        )
        near_translation = (
            args.near_translation_m
            if args.near_translation_m is not None
            else float(near_config.get("translation_tolerance_m", 0.01))
        )
        near_rotation = (
            args.near_rotation_deg
            if args.near_rotation_deg is not None
            else float(near_config.get("rotation_tolerance_deg", 1.0))
        )

        rows, _ = read_csv_rows(args.input)
        result = split_pose_rows(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            split_mode=args.split_mode,
            near_translation_m=near_translation,
            near_rotation_deg=near_rotation,
        )
        result.manifest["source_csv"] = str(args.input)
        result.manifest["source_csv_sha256"] = file_sha256(args.input)
        result.manifest["config_file"] = str(args.config)
        raw_test_rows = None
        raw_test_fields = None
        if args.annotated_raw is not None:
            annotated_rows, annotated_fields = read_csv_rows(args.annotated_raw)
            annotated_provenance = dataset_provenance(annotated_rows)
            if annotated_provenance != result.manifest["data_provenance"]:
                raise DatasetError(
                    "aggregate and annotated raw provenance differ: {} versus {}".format(
                        result.manifest["data_provenance"], annotated_provenance
                    )
                )
            raw_test_rows = select_raw_test_rows(annotated_rows, result.test_rows)
            raw_test_fields = ordered_fieldnames(raw_test_rows, annotated_fields)
            result.manifest["raw_test"] = {
                "file": targets["raw_test"].name,
                "row_count": len(raw_test_rows),
                "source_annotated_csv": str(args.annotated_raw),
                "source_annotated_sha256": file_sha256(args.annotated_raw),
                "selection": (
                    "fixed test session_id+pose_id; processing_status=valid; "
                    "is_outlier=false"
                ),
            }
        csv_fields = ordered_fieldnames(
            result.train_rows + result.val_rows + result.test_rows
        )
        write_csv_exclusive(targets["train"], result.train_rows, csv_fields)
        write_csv_exclusive(targets["val"], result.val_rows, csv_fields)
        write_csv_exclusive(targets["test"], result.test_rows, csv_fields)
        if raw_test_rows is not None and raw_test_fields is not None:
            write_csv_exclusive(targets["raw_test"], raw_test_rows, raw_test_fields)
        write_json_exclusive(targets["normalization"], result.normalization)

        artifacts = {
            "aggregate_input": {
                "file": str(args.input),
                "sha256": file_sha256(args.input),
            },
            "train": {
                "file": targets["train"].name,
                "sha256": file_sha256(targets["train"]),
            },
            "val": {
                "file": targets["val"].name,
                "sha256": file_sha256(targets["val"]),
            },
            "test": {
                "file": targets["test"].name,
                "sha256": file_sha256(targets["test"]),
            },
            "normalization": {
                "file": targets["normalization"].name,
                "sha256": file_sha256(targets["normalization"]),
            },
        }
        for name in ("train", "val", "test"):
            result.manifest["splits"][name]["csv_file"] = artifacts[name]["file"]
            result.manifest["splits"][name]["sha256"] = artifacts[name]["sha256"]
        result.manifest["normalization"]["sha256"] = artifacts["normalization"][
            "sha256"
        ]
        if raw_test_rows is not None:
            artifacts["annotated_raw_input"] = {
                "file": str(args.annotated_raw),
                "sha256": file_sha256(args.annotated_raw),
            }
            artifacts["raw_test"] = {
                "file": targets["raw_test"].name,
                "sha256": file_sha256(targets["raw_test"]),
            }
            result.manifest["raw_test"]["sha256"] = artifacts["raw_test"]["sha256"]
        result.manifest["artifacts"] = artifacts
        write_json_exclusive(targets["manifest"], result.manifest)

        print(
            "fixed splits: train={train}, val={val}, test={test}, rule={rule}".format(
                train=len(result.train_rows),
                val=len(result.val_rows),
                test=len(result.test_rows),
                rule=result.manifest["split_rule"],
            )
        )
        print("output directory: {}".format(args.output_dir))
        if raw_test_rows is None:
            print(
                "warning: --annotated-raw was not supplied; raw_test.csv for range "
                "correction evaluation was not generated",
                file=sys.stderr,
            )
        for warning in result.manifest.get("warnings", []):
            print("warning: {}".format(warning), file=sys.stderr)
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print("make_splits: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
