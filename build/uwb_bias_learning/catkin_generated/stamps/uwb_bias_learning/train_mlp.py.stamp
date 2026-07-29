#!/usr/bin/env python3
"""Train the small pose12 MLP with validation-only early stopping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.metrics import bias_prediction_metrics  # noqa: E402
from uwb_bias_learning.lineage import SplitContract, file_sha256  # noqa: E402
from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    TARGET_COLUMN,
    FeatureNormalizer,
    build_mlp,
    checkpoint_payload,
    predict_mlp,
    require_torch,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "feature_set": "pose12",
    "hidden_dims": [64, 64, 32],
    "activation": "relu",
    "loss": "huber",
    "huber_delta": 1.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "max_epochs": 500,
    "early_stopping_patience": 40,
    "early_stopping_min_delta": 0.0,
    "seeds": [17, 42, 2026],
    "sample_weighting": "none",
    "deterministic_algorithms": True,
}


def read_pose_csv(
    path: Path, need_sem: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        required = list(POSE12_FEATURES) + [TARGET_COLUMN, "session_id", "pose_id"]
        if need_sem:
            required.append("bias_sem")
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
        targets = np.asarray([float(row[TARGET_COLUMN]) for row in rows], dtype=np.float64)
        sem = (
            np.asarray([float(row["bias_sem"]) for row in rows], dtype=np.float64)
            if need_sem
            else np.ones(len(rows), dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("{} contains non-numeric required values".format(path)) from exc
    if not all(np.all(np.isfinite(array)) for array in (features, targets, sem)):
        raise ValueError("{} contains NaN or infinite required values".format(path))
    if need_sem and np.any(sem <= 0.0):
        raise ValueError("bias_sem must be strictly positive for inverse_sem weighting")
    keys = [(row["session_id"], row["pose_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("{} contains duplicate session_id + pose_id rows".format(path))
    return features, targets, rows, sem


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError("training configuration does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream)
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("training configuration must be a YAML mapping")
    config = dict(DEFAULT_CONFIG)
    config.update(supplied)
    if config["feature_set"] != "pose12":
        raise ValueError("first-version feature_set must be exactly 'pose12'")
    if "feature_fields" in config and tuple(config["feature_fields"]) != POSE12_FEATURES:
        raise ValueError("feature_fields must exactly match the fixed pose12 order")
    if "target_field" in config and config["target_field"] != TARGET_COLUMN:
        raise ValueError("target_field must be exactly '{}'".format(TARGET_COLUMN))
    if "sample_weighting" not in supplied and bool(config.get("use_bias_sem_weights", False)):
        config["sample_weighting"] = "inverse_sem"
    config["hidden_dims"] = [int(value) for value in config["hidden_dims"]]
    if not config["hidden_dims"] or any(value <= 0 for value in config["hidden_dims"]):
        raise ValueError("hidden_dims must contain positive integers")
    config["seeds"] = [int(value) for value in config["seeds"]]
    if len(set(config["seeds"])) < 3:
        raise ValueError("at least three distinct random seeds are required")
    for name in ("batch_size", "max_epochs", "early_stopping_patience"):
        config[name] = int(config[name])
        if config[name] <= 0:
            raise ValueError("{} must be positive".format(name))
    for name in (
        "learning_rate",
        "weight_decay",
        "early_stopping_min_delta",
        "huber_delta",
    ):
        config[name] = float(config[name])
        if not np.isfinite(config[name]) or config[name] < 0.0:
            raise ValueError("{} must be finite and non-negative".format(name))
    if config["learning_rate"] == 0.0 or config["huber_delta"] == 0.0:
        raise ValueError("learning_rate and huber_delta must be strictly positive")
    config["loss"] = str(config["loss"]).lower()
    if config["loss"] not in ("huber", "mse"):
        raise ValueError("loss must be 'huber' or 'mse'")
    config["sample_weighting"] = str(config["sample_weighting"]).lower()
    if config["sample_weighting"] not in ("none", "inverse_sem"):
        raise ValueError("sample_weighting must be 'none' or 'inverse_sem'")
    return config


def loss_values(torch: Any, prediction: Any, target: Any, config: Mapping[str, Any]) -> Any:
    if config["loss"] == "mse":
        return torch.square(prediction - target)
    # Explicit Huber formula keeps behaviour stable across PyTorch versions.
    delta = float(config["huber_delta"])
    absolute = torch.abs(prediction - target)
    quadratic = torch.minimum(absolute, torch.tensor(delta, device=absolute.device))
    linear = absolute - quadratic
    return 0.5 * torch.square(quadratic) + delta * linear


def set_seed(torch: Any, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


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
    """Reject a split normalization that was not fitted on this train CSV."""

    if supplied.feature_names != recomputed.feature_names:
        raise ValueError("split normalization feature order does not match pose12")
    for name in ("mean", "scale", "train_min", "train_max"):
        if not np.allclose(
            getattr(supplied, name),
            getattr(recomputed, name),
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise ValueError(
                "split normalization {} does not match values recomputed from train.csv".format(
                    name
                )
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


def git_revision() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL, universal_newlines=True,
        ).strip())
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


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
    parser.add_argument("--config", required=True, type=Path, help="train_mlp YAML")
    parser.add_argument("--output-dir", required=True, type=Path, help="new run directory")
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device; CPU is the reproducible default"
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="software tests only: permit a manifest marked data_provenance=synthetic",
    )
    args = parser.parse_args(argv)
    for source in (args.train, args.val, args.config, args.split_manifest):
        if not source.is_file():
            parser.error("input file does not exist: {}".format(source))
    if args.output_dir.exists():
        parser.error(
            "refusing to write into an existing run directory; choose a new --output-dir"
        )

    contract = SplitContract.load(
        args.split_manifest, allow_synthetic=args.allow_synthetic
    )
    contract.verify_pose_csv("train", args.train)
    contract.verify_pose_csv("val", args.val)
    normalization_source = contract.artifact_path("normalization")
    contract.verify_artifact("normalization", normalization_source)
    config = load_config(args.config)
    use_sem = config["sample_weighting"] == "inverse_sem"
    train_x, train_y, train_rows, train_sem = read_pose_csv(args.train, need_sem=use_sem)
    val_x, val_y, val_rows, _ = read_pose_csv(args.val, need_sem=False)
    train_keys = {(row["session_id"], row["pose_id"]) for row in train_rows}
    val_keys = {(row["session_id"], row["pose_id"]) for row in val_rows}
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError(
            "train/val leakage: {} session_id + pose_id keys overlap".format(len(overlap))
        )

    normalizer = FeatureNormalizer.load_json(normalization_source)
    verify_normalizer_matches_train(normalizer, FeatureNormalizer.fit(train_x))
    torch = require_torch()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    # Nothing above mutates the workspace.  Claim one new run directory only
    # after all content and provenance checks have succeeded.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(str(normalization_source), str(args.output_dir / "normalization.json"))
    shutil.copyfile(str(contract.path), str(args.output_dir / "split_manifest.json"))
    with (args.output_dir / "train_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, default_flow_style=False, sort_keys=True)
    input_config_sha256 = file_sha256(args.config)
    saved_config_sha256 = file_sha256(args.output_dir / "train_config.yaml")
    run_normalization_sha256 = file_sha256(args.output_dir / "normalization.json")
    lineage = lineage_payload(
        contract,
        normalization_sha256=run_normalization_sha256,
        train_config_input_sha256=input_config_sha256,
        train_config_saved_sha256=saved_config_sha256,
    )
    train_normalized = normalizer.transform(train_x).astype(np.float32)
    val_normalized = normalizer.transform(val_x).astype(np.float32)
    if use_sem:
        weights = 1.0 / np.square(train_sem)
        weights = weights / np.mean(weights)
    else:
        weights = np.ones(train_y.shape[0], dtype=np.float64)

    seed_results: List[Dict[str, Any]] = []
    selected_checkpoint: Path = None
    selected_val_loss = float("inf")
    for seed in config["seeds"]:
        set_seed(torch, seed, bool(config["deterministic_algorithms"]))
        seed_dir = args.output_dir / "seed_{}".format(seed)
        seed_dir.mkdir(parents=True, exist_ok=False)
        model = build_mlp(config["hidden_dims"], config["activation"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        train_tensor = torch.from_numpy(train_normalized).to(device)
        train_target = torch.from_numpy(train_y.astype(np.float32)).to(device).reshape(-1, 1)
        train_weight = torch.from_numpy(weights.astype(np.float32)).to(device).reshape(-1, 1)
        val_tensor = torch.from_numpy(val_normalized).to(device)
        val_target = torch.from_numpy(val_y.astype(np.float32)).to(device).reshape(-1, 1)
        generator = np.random.RandomState(seed)
        best_state = None
        best_epoch = -1
        best_val_loss = float("inf")
        patience_reference_loss = float("inf")
        stale_epochs = 0
        history: List[Dict[str, Any]] = []

        for epoch in range(config["max_epochs"]):
            model.train()
            order = generator.permutation(train_x.shape[0])
            total_loss = 0.0
            total_weight = 0.0
            for start in range(0, len(order), config["batch_size"]):
                batch_indices = order[start : start + config["batch_size"]]
                indices = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
                optimizer.zero_grad()
                prediction = model(train_tensor.index_select(0, indices))
                batch_weight = train_weight.index_select(0, indices)
                element_loss = loss_values(
                    torch, prediction, train_target.index_select(0, indices), config
                )
                batch_loss = torch.sum(element_loss * batch_weight) / torch.sum(batch_weight)
                batch_loss.backward()
                optimizer.step()
                weight_sum = float(torch.sum(batch_weight).detach().cpu().item())
                total_loss += float(batch_loss.detach().cpu().item()) * weight_sum
                total_weight += weight_sum

            model.eval()
            with torch.no_grad():
                validation_prediction = model(val_tensor)
                validation_loss = float(
                    torch.mean(loss_values(torch, validation_prediction, val_target, config))
                    .detach().cpu().item()
                )
            train_loss = total_loss / total_weight
            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": validation_loss}
            )
            # Always retain the true minimum validation checkpoint.  min_delta
            # controls only the patience reset and must not change what
            # "best" means in the saved model or report.
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            if validation_loss < (
                patience_reference_loss - config["early_stopping_min_delta"]
            ):
                patience_reference_loss = validation_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config["early_stopping_patience"]:
                    break

        if best_state is None:
            raise RuntimeError("training produced no finite validation checkpoint")
        model.to("cpu")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        checkpoint = checkpoint_payload(
            model,
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            seed=seed,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            loss=config["loss"],
            target_unit="m",
            data_provenance=contract.data_provenance,
            lineage=lineage,
        )
        checkpoint_path = seed_dir / "model_state.pt"
        torch.save(checkpoint, str(checkpoint_path))
        with (seed_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(history)
        val_prediction = predict_mlp(model, val_x, normalizer)
        validation_metrics = bias_prediction_metrics(val_y, val_prediction)
        result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "best_val_loss": best_val_loss,
            "validation_bias_metrics": validation_metrics,
            "checkpoint": str(checkpoint_path.relative_to(args.output_dir)),
            "history": str((seed_dir / "history.csv").relative_to(args.output_dir)),
            "data_provenance": contract.data_provenance,
            "lineage": lineage,
        }
        write_json(seed_dir / "metrics.json", result)
        seed_results.append(result)
        if best_val_loss < selected_val_loss:
            selected_val_loss = best_val_loss
            selected_checkpoint = checkpoint_path

    if selected_checkpoint is None:
        raise RuntimeError("no seed checkpoint was selected")
    shutil.copyfile(str(selected_checkpoint), str(args.output_dir / "model_state.pt"))
    selected_seed = int(selected_checkpoint.parent.name.split("seed_", 1)[1])
    metric_names = (
        "mae", "rmse", "mean_error", "error_std", "r2",
        "p50_abs_error", "p90_abs_error", "p95_abs_error",
    )
    aggregate: Dict[str, Any] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [result["validation_bias_metrics"][metric_name] for result in seed_results],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        aggregate[metric_name] = {
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite, ddof=0)) if finite.size else None,
            "finite_seed_count": int(finite.size),
        }
    summary = {
        "schema_version": 1,
        "feature_names": list(POSE12_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_unit": "m",
        "selection_rule": "minimum validation loss",
        "test_set_used": False,
        "data_provenance": contract.data_provenance,
        "lineage": lineage,
        "selected_seed": selected_seed,
        "selected_checkpoint": "model_state.pt",
        "seeds": seed_results,
        "validation_across_seeds": aggregate,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    write_json(
        args.output_dir / "data_manifest.json",
        {
            "train_csv": str(args.train), "val_csv": str(args.val),
            "train_sha256": file_sha256(args.train), "val_sha256": file_sha256(args.val),
            "train_rows": int(train_x.shape[0]), "val_rows": int(val_x.shape[0]),
            "test_set_used": False,
            "data_provenance": contract.data_provenance,
            "lineage": lineage,
        },
    )
    write_json(
        args.output_dir / "versions.json",
        {
            "python": platform.python_version(), "numpy": np.__version__,
            "pyyaml": yaml.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "device": str(device), "git": git_revision(),
        },
    )
    print("Saved {} seed runs to {}".format(len(seed_results), args.output_dir))
    print("Selected seed {} by validation loss only; the test set was not read.".format(selected_seed))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
