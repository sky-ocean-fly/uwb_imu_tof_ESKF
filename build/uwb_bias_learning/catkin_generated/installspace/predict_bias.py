#!/usr/bin/env python3
"""Predict signed UWB range bias from one raw relative pose12 sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.models import POSE12_FEATURES, require_torch  # noqa: E402


def load_spec(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            "model specification does not exist: {}; keep model_spec.json beside the model or pass --model-spec".format(
                path
            )
        )
    with path.open("r", encoding="utf-8") as stream:
        spec = json.load(stream)
    fields = tuple(spec.get("input", {}).get("fields_in_exact_order", ()))
    if fields != POSE12_FEATURES:
        raise ValueError("model_spec input field order does not match required pose12 order")
    return spec


def validate_rotation(rotation: np.ndarray, tolerance: float) -> None:
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix must contain nine finite row-major values")
    orthogonality_error = float(
        np.linalg.norm(np.matmul(rotation.T, rotation) - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > tolerance or abs(determinant - 1.0) > tolerance:
        raise ValueError(
            "rotation matrix is not in SO(3): orthogonality error {:.6g}, determinant {:.6g}".format(
                orthogonality_error, determinant
            )
        )


def predict_one(
    model_path: Path,
    spec_path: Path,
    pose12: Sequence[float],
) -> Dict[str, Any]:
    features = np.asarray(pose12, dtype=np.float64).reshape(-1)
    if features.shape != (len(POSE12_FEATURES),) or not np.all(np.isfinite(features)):
        raise ValueError("pose input must contain exactly 12 finite values")
    validate_rotation(features[3:].reshape(3, 3), tolerance=1.0e-3)
    spec = load_spec(spec_path)
    domain = spec.get("training_domain", {}).get("per_feature", {})
    violations = []
    for index, field in enumerate(POSE12_FEATURES):
        if field not in domain or "min" not in domain[field] or "max" not in domain[field]:
            raise ValueError("model_spec lacks training-domain bounds for {}".format(field))
        minimum = float(domain[field]["min"])
        maximum = float(domain[field]["max"])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError("model_spec has invalid training-domain bounds for {}".format(field))
        value = float(features[index])
        if value < minimum or value > maximum:
            violations.append(
                "{}={:.9g} outside [{:.9g}, {:.9g}]".format(
                    field, value, minimum, maximum
                )
            )

    torch = require_torch()
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise ValueError("failed to load TorchScript model {}: {}".format(model_path, exc)) from exc
    model.eval()
    tensor = torch.from_numpy(features.astype(np.float32).reshape(1, -1))
    with torch.no_grad():
        output = model(tensor).detach().cpu().numpy().reshape(-1)
    if output.size != 1 or not np.isfinite(output[0]):
        raise RuntimeError("model did not return one finite bias value")
    warning = (
        "input outside training feature bounds; evaluated unchanged (no clipping): "
        + "; ".join(violations)
        if violations
        else "none"
    )
    return {
        "predicted_bias_m": float(output[0]),
        "in_training_domain": not violations,
        "warning": warning,
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="model_torchscript.pt")
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=None,
        help="defaults to model_spec.json beside --model",
    )
    parser.add_argument("--tx", required=True, type=float)
    parser.add_argument("--ty", required=True, type=float)
    parser.add_argument("--tz", required=True, type=float)
    parser.add_argument(
        "--rotation-matrix",
        required=True,
        nargs=9,
        type=float,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    parser.add_argument(
        "--rotation-tolerance", type=float, default=1.0e-3,
        help="SO(3) orthogonality and determinant tolerance",
    )
    parser.add_argument(
        "--range-m", type=float, default=None,
        help="optional raw range; prints corrected_range_m = range_m - predicted_bias_m",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error("model file does not exist: {}".format(args.model))
    if args.rotation_tolerance <= 0.0 or not math.isfinite(args.rotation_tolerance):
        parser.error("--rotation-tolerance must be finite and positive")
    values = np.asarray(
        [args.tx, args.ty, args.tz] + list(args.rotation_matrix), dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        parser.error("pose values must all be finite")
    validate_rotation(values[3:].reshape(3, 3), args.rotation_tolerance)
    spec_path = args.model_spec or args.model.with_name("model_spec.json")
    result = predict_one(args.model, spec_path, values)
    if args.range_m is not None:
        if not math.isfinite(args.range_m):
            parser.error("--range-m must be finite")
        result["corrected_range_m"] = float(args.range_m - result["predicted_bias_m"])
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("predicted_bias_m: {:.12g}".format(result["predicted_bias_m"]))
        print("in_training_domain: {}".format(str(result["in_training_domain"]).lower()))
        print("warning: {}".format(result["warning"]))
        if "corrected_range_m" in result:
            print("corrected_range_m: {:.12g}".format(result["corrected_range_m"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
