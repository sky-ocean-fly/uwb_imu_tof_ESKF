#!/usr/bin/env python3
"""Estimate signed UWB range bias from synchronized mocap relative poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.bias_estimation import (  # noqa: E402
    summarize_bias_samples,
    true_range_from_relative_pose,
)
from uwb_bias_learning.transforms import quaternion_to_matrix  # noqa: E402


APPENDED_COLUMNS = (
    "bias_reference_frame",
    "tag_reference_x_m",
    "tag_reference_y_m",
    "tag_reference_z_m",
    "anchor_reference_x_m",
    "anchor_reference_y_m",
    "anchor_reference_z_m",
    "antenna_delta_x_m",
    "antenna_delta_y_m",
    "antenna_delta_z_m",
    "true_distance_m",
    "uwb_bias_m",
)


class BiasEstimationError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use synchronized mocap relative pose and endpoint lever arms to compute "
            "true UWB distance and signed bias (measured range - true distance)."
        )
    )
    parser.add_argument("--input", required=True, help="Input synchronized CSV")
    parser.add_argument("--output", required=True, help="New per-sample result CSV")
    parser.add_argument(
        "--summary-json",
        help="New summary JSON (default: <output>.summary.json)",
    )
    parser.add_argument(
        "--tag-position-body",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Tag antenna phase-center position in the vehicle/body frame, metres",
    )
    parser.add_argument(
        "--anchor-position-base",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Anchor antenna phase-center position in the base rigid-body frame, metres",
    )
    parser.add_argument(
        "--pose-direction",
        choices=("base_to_body", "body_to_base"),
        default="base_to_body",
        help=(
            "Meaning of the input relative pose: base_to_body is T_base_body; "
            "body_to_base is T_body_base (default: %(default)s)"
        ),
    )
    parser.add_argument("--range-column", default="range_m")
    parser.add_argument(
        "--translation-columns",
        nargs=3,
        default=("rel_tx", "rel_ty", "rel_tz"),
        metavar=("TX", "TY", "TZ"),
    )
    parser.add_argument(
        "--quaternion-columns",
        nargs=4,
        default=("rel_qx", "rel_qy", "rel_qz", "rel_qw"),
        metavar=("QX", "QY", "QZ", "QW"),
        help="Quaternion columns in qx qy qz qw order",
    )
    parser.add_argument(
        "--rotation-columns",
        nargs=9,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Use nine row-major rotation-matrix columns instead of quaternion columns",
    )
    parser.add_argument(
        "--valid-column",
        default="is_valid",
        help="If this column exists, only truthy rows are used (default: %(default)s)",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Ignore the valid-column filter and calculate every row",
    )
    return parser


def _finite(row: Mapping[str, str], columns: Sequence[str], row_number: int) -> np.ndarray:
    missing = [name for name in columns if name not in row or str(row[name]).strip() == ""]
    if missing:
        raise BiasEstimationError(
            "row {} is missing values for: {}".format(row_number, ", ".join(missing))
        )
    try:
        result = np.asarray([float(row[name]) for name in columns], dtype=float)
    except ValueError as exc:
        raise BiasEstimationError("row {} contains a non-numeric pose value".format(row_number)) from exc
    if not np.all(np.isfinite(result)):
        raise BiasEstimationError("row {} contains a non-finite pose value".format(row_number))
    return result


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def _rotation(row: Mapping[str, str], args, row_number: int) -> np.ndarray:
    if args.rotation_columns:
        return _finite(row, args.rotation_columns, row_number).reshape(3, 3)
    quaternion = _finite(row, args.quaternion_columns, row_number)
    return quaternion_to_matrix(quaternion)


def _paths(args) -> Tuple[Path, Path, Path]:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else Path(str(output_path) + ".summary.json")
    )
    if not input_path.is_file():
        raise BiasEstimationError("input CSV does not exist: {}".format(input_path))
    if output_path == summary_path:
        raise BiasEstimationError("output CSV and summary JSON must be different paths")
    for path in (output_path, summary_path):
        if path.exists():
            raise BiasEstimationError("refusing to overwrite existing output: {}".format(path))
    return input_path, output_path, summary_path


def estimate(args) -> Dict[str, object]:
    input_path, output_path, summary_path = _paths(args)
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise BiasEstimationError("input CSV has no header")
            original_columns = list(reader.fieldnames)
            collisions = [name for name in APPENDED_COLUMNS if name in original_columns]
            if collisions:
                raise BiasEstimationError(
                    "input already contains output columns: {}".format(", ".join(collisions))
                )
            required = [args.range_column, *args.translation_columns]
            required.extend(args.rotation_columns or args.quaternion_columns)
            missing = [name for name in required if name not in original_columns]
            if missing:
                raise BiasEstimationError(
                    "input CSV is missing columns: {}".format(", ".join(missing))
                )

            rows: List[Dict[str, object]] = []
            biases: List[float] = []
            ranges: List[float] = []
            truths: List[float] = []
            input_rows = 0
            skipped_invalid = 0
            valid_filter_present = args.valid_column in original_columns
            for row_number, source in enumerate(reader, start=2):
                input_rows += 1
                if (
                    valid_filter_present
                    and not args.include_invalid
                    and not _truthy(source.get(args.valid_column, ""))
                ):
                    skipped_invalid += 1
                    continue
                translation = _finite(source, args.translation_columns, row_number)
                rotation = _rotation(source, args, row_number)
                try:
                    measured = float(source[args.range_column])
                except (TypeError, ValueError) as exc:
                    raise BiasEstimationError(
                        "row {} has invalid measured range".format(row_number)
                    ) from exc
                if not math.isfinite(measured) or measured <= 0.0:
                    raise BiasEstimationError(
                        "row {} measured range must be finite and positive".format(row_number)
                    )
                geometry = true_range_from_relative_pose(
                    translation,
                    rotation,
                    args.tag_position_body,
                    args.anchor_position_base,
                    pose_direction=args.pose_direction,
                )
                truth = float(geometry["true_distance_m"])
                bias = measured - truth
                tag = geometry["tag_position_reference_m"]
                anchor = geometry["anchor_position_reference_m"]
                delta = geometry["delta_reference_m"]
                result: Dict[str, object] = dict(source)
                result.update(
                    {
                        "bias_reference_frame": geometry["reference_frame"],
                        "tag_reference_x_m": float(tag[0]),
                        "tag_reference_y_m": float(tag[1]),
                        "tag_reference_z_m": float(tag[2]),
                        "anchor_reference_x_m": float(anchor[0]),
                        "anchor_reference_y_m": float(anchor[1]),
                        "anchor_reference_z_m": float(anchor[2]),
                        "antenna_delta_x_m": float(delta[0]),
                        "antenna_delta_y_m": float(delta[1]),
                        "antenna_delta_z_m": float(delta[2]),
                        "true_distance_m": truth,
                        "uwb_bias_m": bias,
                    }
                )
                rows.append(result)
                ranges.append(measured)
                truths.append(truth)
                biases.append(bias)
    except OSError as exc:
        raise BiasEstimationError("cannot read input CSV: {}".format(exc)) from exc

    if not rows:
        raise BiasEstimationError("no rows remain after validity filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=original_columns + list(APPENDED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, object] = {
        "schema_version": 1,
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "bias_definition": "uwb_bias_m = measured_range_m - true_distance_m",
        "pose_direction": args.pose_direction,
        "tag_position_body_m": [float(value) for value in args.tag_position_body],
        "anchor_position_base_m": [float(value) for value in args.anchor_position_base],
        "input_row_count": input_rows,
        "used_row_count": len(rows),
        "skipped_invalid_row_count": skipped_invalid,
        "valid_filter_applied": bool(valid_filter_present and not args.include_invalid),
        "measured_range_mean_m": float(np.mean(ranges)),
        "true_distance_mean_m": float(np.mean(truths)),
        "bias": summarize_bias_samples(biases),
    }
    try:
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        # The CSV is intentionally not deleted: preserving a created result is
        # safer than silently removing potentially valuable experiment output.
        raise
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = estimate(args)
    except (BiasEstimationError, ValueError, OSError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
