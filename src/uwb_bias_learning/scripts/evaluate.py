#!/usr/bin/env python3
"""Final test-only comparison of bias prediction and raw-range correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.metrics import (  # noqa: E402
    bias_prediction_metrics,
    correct_ranges,
    error_metrics,
)
from uwb_bias_learning.lineage import (  # noqa: E402
    SplitContract,
    require_lineage_match,
)
from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    TARGET_COLUMN,
    ConstantBiasModel,
    FeatureNormalizer,
    RidgeBiasModel,
    ZeroBiasModel,
    load_mlp_checkpoint,
    predict_mlp,
)


METHOD_ORDER = ("Raw", "Constant", "Ridge", "MLP")
BIAS_METHOD_ORDER = ("Zero", "Constant", "Ridge", "MLP")
BASE_LINEAGE_FIELDS = (
    "split_manifest_sha256",
    "data_provenance",
    "train_sha256",
    "val_sha256",
    "test_sha256",
    "split_normalization_sha256",
)


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError("CSV contains no rows: {}".format(path))
    return fieldnames, rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def numeric_matrix(rows: Sequence[Mapping[str, str]], fields: Sequence[str], source: Path) -> np.ndarray:
    try:
        output = np.asarray(
            [[float(row[name]) for name in fields] for row in rows], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "{} has missing or non-numeric fields among: {}".format(
                source, ", ".join(fields)
            )
        ) from exc
    if not np.all(np.isfinite(output)):
        raise ValueError("{} contains NaN or infinite required values".format(source))
    return output


def numeric_vector(rows: Sequence[Mapping[str, str]], field: str, source: Path) -> np.ndarray:
    return numeric_matrix(rows, [field], source).reshape(-1)


def keys_for(rows: Sequence[Mapping[str, str]], source: Path, require_unique: bool) -> List[Tuple[str, str]]:
    try:
        keys = [(row["session_id"], row["pose_id"]) for row in rows]
    except KeyError as exc:
        raise ValueError("{} must contain session_id and pose_id".format(source)) from exc
    if require_unique and len(set(keys)) != len(keys):
        raise ValueError("{} contains duplicate session_id + pose_id rows".format(source))
    return keys


def explicit_bool(value: Any, field: str, source: Path) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise ValueError(
        "{} field {} must contain an explicit boolean; got {!r}".format(
            source, field, value
        )
    )


def load_json_mapping(path: Path, label: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("{} JSON root must be an object: {}".format(label, path))
    return payload


def load_baselines(
    path: Path,
) -> Tuple[Dict[str, Any], ZeroBiasModel, ConstantBiasModel, RidgeBiasModel]:
    payload = load_json_mapping(path, "baseline bundle")
    if tuple(payload.get("feature_names", ())) != POSE12_FEATURES:
        raise ValueError("baseline feature order does not match required pose12 order")
    models = payload.get("models", {})
    if not all(name in models for name in ("zero", "constant", "ridge")):
        raise ValueError("baseline bundle must contain zero, constant, and ridge")
    return (
        payload,
        ZeroBiasModel(),
        ConstantBiasModel.from_dict(models["constant"]),
        RidgeBiasModel.from_dict(models["ridge"]),
    )


def require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return digest


def validate_model_lineage(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    normalization_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    """Bind a trained artifact to one split contract and normalization file."""

    recorded = payload.get("lineage")
    require_lineage_match(recorded, expected, BASE_LINEAGE_FIELDS, label)
    if payload.get("data_provenance") != expected.get("data_provenance"):
        raise ValueError("{} data_provenance does not match the fixed split".format(label))
    recorded_normalization = require_sha256(
        recorded.get("normalization_sha256"),
        "{} lineage normalization_sha256".format(label),
    )
    if recorded_normalization != normalization_sha256:
        raise ValueError(
            "{} lineage normalization_sha256 does not match --normalization".format(label)
        )
    return recorded


def checkpoint_train_config_sha256(lineage: Mapping[str, Any]) -> str:
    """Return the saved training-config hash, rejecting ambiguous aliases."""

    saved = lineage.get("train_config_saved_sha256")
    alias = lineage.get("train_config_sha256")
    if saved in (None, "") and alias in (None, ""):
        raise ValueError("MLP checkpoint lineage has no saved train-config SHA-256")
    if saved not in (None, "") and alias not in (None, "") and saved != alias:
        raise ValueError("MLP checkpoint train-config SHA-256 aliases disagree")
    return require_sha256(
        saved if saved not in (None, "") else alias,
        "MLP checkpoint saved train-config SHA-256",
    )


def validate_raw_test_rows(
    raw_fields: Sequence[str],
    raw_rows: Sequence[Mapping[str, str]],
    source: Path,
    data_provenance: str,
) -> None:
    """Require raw-test rows to be valid and provenance-explicit."""

    required_raw = [
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
    missing = [field for field in required_raw if field not in raw_fields]
    if missing:
        raise ValueError("raw test CSV is missing: {}".format(", ".join(missing)))
    expected_synthetic = data_provenance == "synthetic"
    for row_number, row in enumerate(raw_rows, start=2):
        status = str(row["processing_status"]).strip().lower()
        if status != "valid":
            raise ValueError(
                "{} row {} processing_status must be valid; got {!r}".format(
                    source, row_number, status
                )
            )
        if explicit_bool(row["is_outlier"], "is_outlier", source):
            raise ValueError("{} row {} is marked as an outlier".format(source, row_number))
        for flag in (
            "is_los",
            "is_stable",
            "is_mocap_valid",
            "is_sync_valid",
            "is_range_valid",
        ):
            if not explicit_bool(row[flag], flag, source):
                raise ValueError(
                    "{} row {} has {} != true".format(source, row_number, flag)
                )
        if not str(row["data_origin"]).strip():
            raise ValueError(
                "{} row {} data_origin must be non-empty".format(source, row_number)
            )
        is_synthetic = explicit_bool(row["is_synthetic"], "is_synthetic", source)
        if is_synthetic != expected_synthetic:
            raise ValueError(
                "{} row {} is_synthetic does not match split data_provenance={!r}".format(
                    source, row_number, data_provenance
                )
            )


def safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(safe_json_value(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_png_exclusive(figure: Any, path: Path, dpi: int = 160) -> None:
    """Save one PNG without any overwrite race after the preflight check."""

    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=dpi)


def metric_row(
    quantity: str,
    method: str,
    grouping: str,
    group: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        "quantity": quantity,
        "method": method,
        "grouping": grouping,
        "group": group,
    }
    row.update(metrics)
    return row


def interval_labels(values: np.ndarray, width: float, unit: str) -> np.ndarray:
    if width <= 0.0 or not np.isfinite(width):
        raise ValueError("bin width must be finite and positive")
    start = math.floor(float(np.min(values)) / width) * width
    indices = np.floor((values - start) / width).astype(int)
    labels = []
    for index in indices:
        low = start + int(index) * width
        high = low + width
        labels.append("[{:.6g},{:.6g}) {}".format(low, high, unit))
    return np.asarray(labels, dtype=object)


def orientation_angle_deg(features: np.ndarray) -> np.ndarray:
    rotation = features[:, 3:].reshape(-1, 3, 3)
    traces = np.trace(rotation, axis1=1, axis2=2)
    cosine = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def validate_pose_geometry(features: np.ndarray, tolerance: float = 1.0e-5) -> None:
    rotations = features[:, 3:].reshape(-1, 3, 3)
    identities = np.matmul(np.transpose(rotations, (0, 2, 1)), rotations)
    orthogonality = np.linalg.norm(identities - np.eye(3), axis=(1, 2))
    determinants = np.linalg.det(rotations)
    bad = (orthogonality > tolerance) | (np.abs(determinants - 1.0) > tolerance)
    if np.any(bad):
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            "pose12 row {} has an invalid SO(3) block (orthogonality error {:.6g}, determinant {:.6g})".format(
                first, orthogonality[first], determinants[first]
            )
        )


def manifest_pose_keys(manifest: Mapping[str, Any], split: str) -> set:
    try:
        rows = manifest["splits"][split]["pose_keys"]
        return {(str(row["session_id"]), str(row["pose_id"])) for row in rows}
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "split manifest lacks splits.{}.pose_keys".format(split)
        ) from exc


def grouped_range_rows(
    distance_gt: np.ndarray,
    corrected: Mapping[str, np.ndarray],
    labels: Sequence[Any],
    grouping: str,
) -> List[Dict[str, Any]]:
    string_labels = np.asarray([str(value) for value in labels], dtype=object)
    rows: List[Dict[str, Any]] = []
    for label in sorted(set(string_labels.tolist())):
        mask = string_labels == label
        for method in METHOD_ORDER:
            rows.append(
                metric_row(
                    "range_correction",
                    method,
                    grouping,
                    label,
                    error_metrics(distance_gt[mask], corrected[method][mask], include_r2=False),
                )
            )
    return rows


def find_history(mlp_state: Path, explicit: Path) -> Path:
    if explicit is not None:
        return explicit
    summary_path = mlp_state.parent / "training_summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        selected_seed = summary.get("selected_seed")
        candidate = mlp_state.parent / "seed_{}".format(selected_seed) / "history.csv"
        if candidate.is_file():
            return candidate
    return None


def generate_figures(
    figures_dir: Path,
    pose_features: np.ndarray,
    true_bias: np.ndarray,
    bias_predictions: Mapping[str, np.ndarray],
    distance_gt: np.ndarray,
    corrected: Mapping[str, np.ndarray],
    distance_labels: np.ndarray,
    orientation_labels: np.ndarray,
    history_path: Path,
) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for evaluation figures; install it or use --skip-figures"
        ) from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []

    fig, axis = plt.subplots(figsize=(6, 5))
    for method in BIAS_METHOD_ORDER:
        axis.scatter(true_bias, bias_predictions[method], s=16, alpha=0.7, label=method)
    limits = [
        min(float(np.min(true_bias)), *(float(np.min(v)) for v in bias_predictions.values())),
        max(float(np.max(true_bias)), *(float(np.max(v)) for v in bias_predictions.values())),
    ]
    axis.plot(limits, limits, "k--", linewidth=1, label="ideal")
    axis.set_xlabel("True mean bias (m)")
    axis.set_ylabel("Predicted mean bias (m)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "bias_true_vs_pred.png"
    save_png_exclusive(fig, path)
    plt.close(fig)
    made.append(path.name)

    fig, axis = plt.subplots(figsize=(7, 5))
    for method in METHOD_ORDER:
        axis.hist(
            corrected[method] - distance_gt,
            bins=40,
            density=False,
            histtype="step",
            linewidth=1.4,
            label=method,
        )
    axis.set_xlabel("Range error (m)")
    axis.set_ylabel("Count")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "range_error_before_after.png"
    save_png_exclusive(fig, path)
    plt.close(fig)
    made.append(path.name)

    def grouped_rmse_plot(labels: np.ndarray, filename: str, xlabel: str) -> None:
        unique = sorted(set(labels.tolist()))
        positions = np.arange(len(unique))
        fig, axis = plt.subplots(figsize=(max(7, 0.8 * len(unique)), 5))
        for method in METHOD_ORDER:
            rmse = []
            for label in unique:
                mask = labels == label
                rmse.append(
                    error_metrics(distance_gt[mask], corrected[method][mask], include_r2=False)[
                        "rmse"
                    ]
                )
            axis.plot(positions, rmse, marker="o", label=method)
        axis.set_xticks(positions)
        axis.set_xticklabels(unique, rotation=35, ha="right")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Range RMSE (m)")
        axis.legend()
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        path = figures_dir / filename
        save_png_exclusive(fig, path)
        plt.close(fig)
        made.append(path.name)

    grouped_rmse_plot(distance_labels, "error_by_distance.png", "Ground-truth distance bin")
    grouped_rmse_plot(
        orientation_labels, "error_by_orientation.png", "Relative rotation-angle bin"
    )

    fig, axis = plt.subplots(figsize=(6, 5))
    scatter = axis.scatter(
        pose_features[:, 0], pose_features[:, 1], c=true_bias, s=24, cmap="coolwarm"
    )
    axis.set_xlabel("rel_tx (m)")
    axis.set_ylabel("rel_ty (m)")
    axis.set_title("Test pose coverage (XY projection)")
    fig.colorbar(scatter, ax=axis, label="Mean bias (m)")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "data_coverage.png"
    save_png_exclusive(fig, path)
    plt.close(fig)
    made.append(path.name)

    if history_path is not None and history_path.is_file():
        _, history_rows = read_csv(history_path)
        epochs = numeric_vector(history_rows, "epoch", history_path)
        train_loss = numeric_vector(history_rows, "train_loss", history_path)
        val_loss = numeric_vector(history_rows, "val_loss", history_path)
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.plot(epochs, train_loss, label="train")
        axis.plot(epochs, val_loss, label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Configured loss")
        axis.legend()
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        path = figures_dir / "training_curve.png"
        save_png_exclusive(fig, path)
        plt.close(fig)
        made.append(path.name)
    return made


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=Path, help="fixed pose-level test.csv")
    parser.add_argument(
        "--raw-test",
        required=True,
        type=Path,
        help="valid raw ranges belonging only to the fixed test poses",
    )
    parser.add_argument("--baselines", required=True, type=Path, help="baselines.json")
    parser.add_argument("--mlp-state", required=True, type=Path, help="selected model_state.pt")
    parser.add_argument("--normalization", required=True, type=Path, help="MLP normalization.json")
    parser.add_argument(
        "--split-manifest", required=True, type=Path,
        help="splits.json used to verify that test rows are exactly the fixed test split",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="root under which reports/ and figures/ are created",
    )
    parser.add_argument("--training-history", type=Path, default=None)
    parser.add_argument("--distance-bin-width-m", type=float, default=1.0)
    parser.add_argument("--orientation-bin-width-deg", type=float, default=30.0)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="allow synthetic data only for an explicitly marked software test run",
    )
    args = parser.parse_args(argv)
    for source in (
        args.test,
        args.raw_test,
        args.baselines,
        args.mlp_state,
        args.normalization,
        args.split_manifest,
    ):
        if not source.is_file():
            parser.error("input file does not exist: {}".format(source))
    if args.training_history is not None and not args.training_history.is_file():
        parser.error("training history does not exist: {}".format(args.training_history))
    if args.distance_bin_width_m <= 0.0 or not np.isfinite(args.distance_bin_width_m):
        parser.error("--distance-bin-width-m must be finite and positive")
    if args.orientation_bin_width_deg <= 0.0 or not np.isfinite(
        args.orientation_bin_width_deg
    ):
        parser.error("--orientation-bin-width-deg must be finite and positive")
    reports_dir = args.output_root / "reports"
    figures_dir = args.output_root / "figures"
    report_targets = [
        reports_dir / name
        for name in (
            "test_metrics.csv", "test_summary.md", "metrics.json",
            "test_bias_predictions.csv", "test_range_predictions.csv",
        )
    ]
    figure_targets = [
        figures_dir / name
        for name in (
            "bias_true_vs_pred.png", "range_error_before_after.png",
            "error_by_distance.png", "error_by_orientation.png",
            "data_coverage.png", "training_curve.png",
        )
    ]
    existing_targets = [
        path for path in report_targets + ([] if args.skip_figures else figure_targets)
        if path.exists()
    ]
    if existing_targets:
        parser.error(
            "refusing to overwrite existing evaluation artifacts: {}".format(
                ", ".join(str(path) for path in existing_targets)
            )
        )

    # Complete every content, provenance, schema, and model-lineage check before
    # creating a report directory or writing a result file.
    contract = SplitContract.load(
        args.split_manifest, allow_synthetic=args.allow_synthetic
    )
    test_sha256 = contract.verify_pose_csv("test", args.test)
    raw_test_sha256 = contract.verify_artifact("raw_test", args.raw_test)
    normalization_sha256 = contract.verify_artifact(
        "normalization", args.normalization
    )
    expected_lineage = contract.base_lineage()
    test_fields, test_rows = read_csv(args.test)
    required_test = list(POSE12_FEATURES) + [TARGET_COLUMN, "session_id", "pose_id"]
    missing = [field for field in required_test if field not in test_fields]
    if missing:
        raise ValueError("test CSV is missing: {}".format(", ".join(missing)))
    pose_features = numeric_matrix(test_rows, POSE12_FEATURES, args.test)
    validate_pose_geometry(pose_features)
    true_bias = numeric_vector(test_rows, TARGET_COLUMN, args.test)
    test_keys = keys_for(test_rows, args.test, require_unique=True)
    manifest_keys = {
        name: contract.pose_keys(name)
        for name in ("train", "val", "test")
    }
    if manifest_keys["train"] & manifest_keys["val"] or manifest_keys["train"] & manifest_keys["test"] or manifest_keys["val"] & manifest_keys["test"]:
        raise ValueError("split manifest contains overlapping session_id + pose_id keys")
    if set(test_keys) != manifest_keys["test"]:
        raise ValueError(
            "test CSV pose keys do not exactly match the fixed test split in splits.json"
        )

    raw_fields, raw_rows = read_csv(args.raw_test)
    validate_raw_test_rows(
        raw_fields, raw_rows, args.raw_test, contract.data_provenance
    )
    raw_keys = keys_for(raw_rows, args.raw_test, require_unique=False)
    test_key_set = set(test_keys)
    unknown = set(raw_keys) - test_key_set
    absent = test_key_set - set(raw_keys)
    if unknown:
        raise ValueError("raw-test contains {} poses outside fixed test.csv".format(len(unknown)))
    if absent:
        raise ValueError("raw-test has no valid ranges for {} fixed test poses".format(len(absent)))
    raw_features = numeric_matrix(raw_rows, POSE12_FEATURES, args.raw_test)
    validate_pose_geometry(raw_features)
    range_m = numeric_vector(raw_rows, "range_m", args.raw_test)
    distance_gt = numeric_vector(raw_rows, "distance_gt_m", args.raw_test)
    geometric_distance = np.linalg.norm(raw_features[:, :3], axis=1)
    if not np.allclose(geometric_distance, distance_gt, rtol=0.0, atol=1.0e-4):
        first = int(np.flatnonzero(np.abs(geometric_distance - distance_gt) > 1.0e-4)[0])
        raise ValueError(
            "raw-test row {} distance_gt_m disagrees with norm(relative translation)".format(
                first
            )
        )

    baseline_payload, zero, constant, ridge = load_baselines(args.baselines)
    baseline_lineage = validate_model_lineage(
        baseline_payload,
        expected_lineage,
        normalization_sha256,
        "baseline bundle",
    )
    normalizer = FeatureNormalizer.load_json(args.normalization)
    mlp, checkpoint = load_mlp_checkpoint(args.mlp_state)
    checkpoint_lineage = validate_model_lineage(
        checkpoint,
        expected_lineage,
        normalization_sha256,
        "MLP checkpoint",
    )
    train_config_sha256 = checkpoint_train_config_sha256(checkpoint_lineage)

    bias_predictions = {
        "Zero": zero.predict(pose_features),
        "Constant": constant.predict(pose_features),
        "Ridge": ridge.predict(pose_features),
        "MLP": predict_mlp(mlp, pose_features, normalizer),
    }
    all_metric_rows: List[Dict[str, Any]] = []
    bias_metric_payload: Dict[str, Any] = {}
    for method in BIAS_METHOD_ORDER:
        metrics = bias_prediction_metrics(true_bias, bias_predictions[method])
        bias_metric_payload[method] = metrics
        all_metric_rows.append(metric_row("bias_prediction", method, "overall", "all", metrics))

    range_bias = {
        "Raw": np.zeros(range_m.shape[0], dtype=np.float64),
        "Constant": constant.predict(raw_features),
        "Ridge": ridge.predict(raw_features),
        "MLP": predict_mlp(mlp, raw_features, normalizer),
    }
    corrected = {
        method: correct_ranges(range_m, range_bias[method]) for method in METHOD_ORDER
    }
    range_metric_payload: Dict[str, Any] = {"overall": {}}
    for method in METHOD_ORDER:
        metrics = error_metrics(distance_gt, corrected[method], include_r2=False)
        range_metric_payload["overall"][method] = metrics
        all_metric_rows.append(metric_row("range_correction", method, "overall", "all", metrics))

    sessions = np.asarray([row["session_id"] for row in raw_rows], dtype=object)
    distance_labels = interval_labels(distance_gt, args.distance_bin_width_m, "m")
    angle_deg = orientation_angle_deg(raw_features)
    orientation_labels = interval_labels(
        angle_deg, args.orientation_bin_width_deg, "deg"
    )
    session_rows = grouped_range_rows(distance_gt, corrected, sessions, "session")
    distance_rows = grouped_range_rows(
        distance_gt, corrected, distance_labels, "distance_bin"
    )
    orientation_rows = grouped_range_rows(
        distance_gt, corrected, orientation_labels, "orientation_bin"
    )
    all_metric_rows.extend(session_rows + distance_rows + orientation_rows)

    history_path = find_history(args.mlp_state, args.training_history)
    if history_path is not None and history_path.is_file():
        _, history_rows = read_csv(history_path)
        numeric_vector(history_rows, "epoch", history_path)
        numeric_vector(history_rows, "train_loss", history_path)
        numeric_vector(history_rows, "val_loss", history_path)

    baselines_sha256 = file_sha256(args.baselines)
    mlp_state_sha256 = file_sha256(args.mlp_state)
    evaluation_lineage = dict(expected_lineage)
    evaluation_lineage.update(
        {
            "raw_test_sha256": raw_test_sha256,
            "baselines_sha256": baselines_sha256,
            "mlp_state_sha256": mlp_state_sha256,
            "normalization_sha256": normalization_sha256,
            "train_config_sha256": train_config_sha256,
        }
    )

    # No report path is created until all input validation and inference above
    # have succeeded.
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "test_bias_predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "session_id", "pose_id", "true_bias_m", "zero_bias_m",
            "constant_bias_m", "ridge_bias_m", "mlp_bias_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(test_rows):
            writer.writerow(
                {
                    "session_id": row["session_id"], "pose_id": row["pose_id"],
                    "true_bias_m": true_bias[index],
                    "zero_bias_m": bias_predictions["Zero"][index],
                    "constant_bias_m": bias_predictions["Constant"][index],
                    "ridge_bias_m": bias_predictions["Ridge"][index],
                    "mlp_bias_m": bias_predictions["MLP"][index],
                }
            )

    with (reports_dir / "test_range_predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "session_id", "pose_id", "range_m", "distance_gt_m", "raw_error_m",
            "constant_predicted_bias_m", "ridge_predicted_bias_m", "mlp_predicted_bias_m",
            "constant_corrected_m", "ridge_corrected_m", "mlp_corrected_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(raw_rows):
            writer.writerow(
                {
                    "session_id": row["session_id"], "pose_id": row["pose_id"],
                    "range_m": range_m[index], "distance_gt_m": distance_gt[index],
                    "raw_error_m": range_m[index] - distance_gt[index],
                    "constant_predicted_bias_m": range_bias["Constant"][index],
                    "ridge_predicted_bias_m": range_bias["Ridge"][index],
                    "mlp_predicted_bias_m": range_bias["MLP"][index],
                    "constant_corrected_m": corrected["Constant"][index],
                    "ridge_corrected_m": corrected["Ridge"][index],
                    "mlp_corrected_m": corrected["MLP"][index],
                }
            )

    metric_fields = [
        "quantity", "method", "grouping", "group", "count", "mae", "rmse",
        "mean_error", "error_std", "r2", "p50_abs_error", "p90_abs_error",
        "p95_abs_error",
    ]
    with (reports_dir / "test_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metric_rows)

    generated_figures: List[str] = []
    if not args.skip_figures:
        generated_figures = generate_figures(
            figures_dir, pose_features, true_bias, bias_predictions,
            distance_gt, corrected, distance_labels, orientation_labels, history_path,
        )

    metrics_json = {
        "schema_version": 2,
        "evaluation_split": "fixed_test_only",
        "data_provenance": contract.data_provenance,
        "test_only": contract.data_provenance == "synthetic",
        "lineage": evaluation_lineage,
        "split_manifest_sha256": contract.sha256,
        "test_sha256": test_sha256,
        "raw_test_sha256": raw_test_sha256,
        "mlp_state_sha256": mlp_state_sha256,
        "normalization_sha256": normalization_sha256,
        "train_config_sha256": train_config_sha256,
        "baselines_sha256": baselines_sha256,
        "bias_error_definition": "predicted_bias_m - true_bias_m",
        "range_error_definition": "corrected_range_m - distance_gt_m",
        "correction_formula": "corrected_range_m = range_m - predicted_bias_m",
        "bias_prediction": bias_metric_payload,
        "range_correction_overall": range_metric_payload["overall"],
        "figures_generated": generated_figures,
    }
    write_json(reports_dir / "metrics.json", metrics_json)
    raw_metrics = range_metric_payload["overall"]["Raw"]
    with (reports_dir / "test_summary.md").open("x", encoding="utf-8") as stream:
        stream.write("# Final fixed-test evaluation\n\n")
        stream.write(
            "The model and hyperparameters were already selected from validation data. "
            "This report evaluates the fixed test split and does not perform selection.\n\n"
        )
        stream.write("Correction uses `corrected_range_m = range_m - predicted_bias_m`.\n\n")
        stream.write("| Method | Range MAE (m) | Range RMSE (m) | Better than raw RMSE? |\n")
        stream.write("| --- | ---: | ---: | --- |\n")
        for method in METHOD_ORDER:
            metrics = range_metric_payload["overall"][method]
            improved = "n/a" if method == "Raw" else (
                "yes" if metrics["rmse"] < raw_metrics["rmse"] else "no"
            )
            stream.write(
                "| {} | {:.9g} | {:.9g} | {} |\n".format(
                    method, metrics["mae"], metrics["rmse"], improved
                )
            )
        stream.write("\n")
        if history_path is None or not history_path.is_file():
            stream.write(
                "`training_curve.png` was not generated because no selected-seed history "
                "file was available; no curve was fabricated.\n"
            )
    print("Saved fixed-test reports to {}".format(reports_dir))
    if not args.skip_figures:
        print("Saved figures based on measured input rows to {}".format(figures_dir))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
