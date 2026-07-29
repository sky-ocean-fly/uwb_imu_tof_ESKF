#!/usr/bin/env python3
"""Export the selected validation-chosen MLP as a self-contained model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = PROJECT_ROOT / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    FeatureNormalizer,
    build_normalized_inference_module,
    load_mlp_checkpoint,
    require_torch,
)
from uwb_bias_learning.lineage import (  # noqa: E402
    SplitContract,
    require_lineage_match,
)


BASE_LINEAGE_FIELDS = (
    "split_manifest_sha256",
    "data_provenance",
    "train_sha256",
    "val_sha256",
    "test_sha256",
    "split_normalization_sha256",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(label))
    return digest


def require_recorded_hash(
    mapping: Mapping[str, Any], field: str, actual: str, label: str
) -> None:
    recorded = require_sha256(mapping.get(field), "{} {}".format(label, field))
    if recorded != actual:
        raise ValueError(
            "{} {} does not match the supplied artifact".format(label, field)
        )


def checkpoint_train_config_sha256(lineage: Mapping[str, Any]) -> str:
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


def validate_export_lineage(
    contract: SplitContract,
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    mlp_state_sha256: str,
    normalization_sha256: str,
    train_config_sha256: str,
) -> Dict[str, Any]:
    """Validate all export inputs without importing or invoking PyTorch."""

    expected = contract.base_lineage()
    metrics_lineage = metrics.get("lineage")
    checkpoint_lineage = checkpoint.get("lineage")
    require_lineage_match(
        metrics_lineage, expected, BASE_LINEAGE_FIELDS, "evaluation metrics"
    )
    require_lineage_match(
        checkpoint_lineage, expected, BASE_LINEAGE_FIELDS, "MLP checkpoint"
    )
    if checkpoint.get("data_provenance") != contract.data_provenance:
        raise ValueError("MLP checkpoint data_provenance does not match split manifest")

    if metrics.get("evaluation_split") != "fixed_test_only":
        raise ValueError("metrics file is not marked as a fixed-test evaluation")
    if metrics.get("data_provenance") != contract.data_provenance:
        raise ValueError("metrics data_provenance does not match split manifest")
    expected_test_only = contract.data_provenance == "synthetic"
    if metrics.get("test_only") is not expected_test_only:
        raise ValueError(
            "metrics test_only marker does not match data provenance"
        )

    raw_test_sha256 = require_sha256(
        contract.artifact("raw_test").get("sha256"),
        "split raw_test SHA-256",
    )
    top_level_hashes = {
        "split_manifest_sha256": contract.sha256,
        "test_sha256": expected["test_sha256"],
        "raw_test_sha256": raw_test_sha256,
        "mlp_state_sha256": mlp_state_sha256,
        "normalization_sha256": normalization_sha256,
        "train_config_sha256": train_config_sha256,
    }
    for field, actual in top_level_hashes.items():
        require_recorded_hash(metrics, field, actual, "evaluation metrics")
    for field, actual in (
        ("raw_test_sha256", raw_test_sha256),
        ("mlp_state_sha256", mlp_state_sha256),
        ("normalization_sha256", normalization_sha256),
        ("train_config_sha256", train_config_sha256),
    ):
        require_recorded_hash(metrics_lineage, field, actual, "metrics lineage")

    require_recorded_hash(
        checkpoint_lineage,
        "normalization_sha256",
        normalization_sha256,
        "MLP checkpoint lineage",
    )
    if checkpoint_train_config_sha256(checkpoint_lineage) != train_config_sha256:
        raise ValueError(
            "MLP checkpoint saved train-config SHA-256 does not match --train-config"
        )

    verified = dict(expected)
    verified.update(
        {
            "raw_test_sha256": raw_test_sha256,
            "mlp_state_sha256": mlp_state_sha256,
            "normalization_sha256": normalization_sha256,
            "train_config_sha256": train_config_sha256,
        }
    )
    return verified


def _session_values(node: Any) -> Set[str]:
    sessions: Set[str] = set()
    if isinstance(node, dict):
        if "session_id" in node and node["session_id"] not in (None, ""):
            sessions.add(str(node["session_id"]))
        for key, value in node.items():
            if key in ("sessions", "session_ids", "train_sessions"):
                if isinstance(value, (list, tuple)):
                    sessions.update(str(item) for item in value if item not in (None, ""))
                elif value not in (None, ""):
                    sessions.add(str(value))
            else:
                sessions.update(_session_values(value))
    elif isinstance(node, list):
        for item in node:
            sessions.update(_session_values(item))
    return sessions


def extract_training_sessions(manifest: Mapping[str, Any]) -> Sequence[str]:
    direct = manifest.get("train_sessions")
    if direct:
        if isinstance(direct, (list, tuple, set)):
            return sorted({str(value) for value in direct})
        return [str(direct)]
    train_node = None
    if isinstance(manifest.get("splits"), dict):
        train_node = manifest["splits"].get("train")
    if train_node is None:
        train_node = manifest.get("train")
    sessions = _session_values(train_node) if train_node is not None else set()
    assignments = manifest.get("assignments")
    if isinstance(assignments, list):
        sessions.update(
            str(row["session_id"])
            for row in assignments
            if isinstance(row, dict)
            and str(row.get("split", "")).lower() == "train"
            and row.get("session_id") not in (None, "")
        )
    return sorted(sessions)


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-state", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument(
        "--metrics", required=True, type=Path, help="fixed-test metrics.json from evaluate.py"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-version",
        required=True,
        help="explicit human/project version; it is never guessed from the clock",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="allow synthetic artifacts only for an explicitly marked software test export",
    )
    args = parser.parse_args(argv)
    sources = (
        args.model_state,
        args.normalization,
        args.train_config,
        args.split_manifest,
        args.metrics,
    )
    for source in sources:
        if not source.is_file():
            parser.error("input file does not exist: {}".format(source))
    if not args.model_version.strip():
        parser.error("--model-version cannot be empty")
    if args.output_dir.exists():
        parser.error(
            "refusing to use an existing exported model path; --output-dir must not exist"
        )

    contract = SplitContract.load(
        args.split_manifest, allow_synthetic=args.allow_synthetic
    )
    manifest = contract.payload
    training_sessions = extract_training_sessions(manifest)
    if not training_sessions:
        raise ValueError(
            "split manifest does not encode training session IDs; refusing to guess them"
        )
    metrics = load_json(args.metrics)
    if not isinstance(metrics, dict):
        raise ValueError("metrics JSON root must be an object")
    normalization_sha256 = contract.verify_artifact(
        "normalization", args.normalization
    )
    mlp_state_sha256 = file_sha256(args.model_state)
    train_config_sha256 = file_sha256(args.train_config)
    normalizer = FeatureNormalizer.load_json(args.normalization)
    model, checkpoint = load_mlp_checkpoint(args.model_state)
    verified_lineage = validate_export_lineage(
        contract,
        metrics,
        checkpoint,
        mlp_state_sha256=mlp_state_sha256,
        normalization_sha256=normalization_sha256,
        train_config_sha256=train_config_sha256,
    )
    with args.train_config.open("r", encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream)
    if not isinstance(train_config, dict):
        raise ValueError("train config must be a YAML mapping")
    expected_model = checkpoint["model_config"]
    if train_config.get("feature_set") != "pose12":
        raise ValueError("train config feature_set does not match pose12 checkpoint")
    if [int(value) for value in train_config.get("hidden_dims", [])] != [
        int(value) for value in expected_model["hidden_dims"]
    ]:
        raise ValueError("train config hidden_dims do not match checkpoint")
    if str(train_config.get("activation", "")).lower() != str(
        expected_model["activation"]
    ).lower():
        raise ValueError("train config activation does not match checkpoint")
    torch = require_torch()
    wrapper = build_normalized_inference_module(model, normalizer).eval()
    example = torch.zeros((1, len(POSE12_FEATURES)), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=True)
        traced = torch.jit.freeze(traced.eval())
        probes = torch.from_numpy(
            np.vstack((normalizer.mean, normalizer.train_min, normalizer.train_max)).astype(
                np.float32
            )
        )
        eager_output = wrapper(probes).detach().cpu().numpy()
        traced_output = traced(probes).detach().cpu().numpy()
    if not np.allclose(eager_output, traced_output, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("TorchScript verification differs from eager inference")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(str(args.model_state), str(args.output_dir / "model_state.pt"))
    traced.save(str(args.output_dir / "model_torchscript.pt"))
    # Preserve the byte-identical, contract-bound normalization artifact.  A
    # parse-and-reserialize here would silently change its recorded SHA-256.
    shutil.copyfile(str(args.normalization), str(args.output_dir / "normalization.json"))
    shutil.copyfile(str(args.train_config), str(args.output_dir / "train_config.yaml"))
    shutil.copyfile(str(args.split_manifest), str(args.output_dir / "split_manifest.json"))
    shutil.copyfile(str(args.metrics), str(args.output_dir / "metrics.json"))

    per_feature_domain = {
        feature: {
            "min": float(normalizer.train_min[index]),
            "max": float(normalizer.train_max[index]),
        }
        for index, feature in enumerate(POSE12_FEATURES)
    }
    spec = {
        "schema_version": 2,
        "model_version": args.model_version.strip(),
        "model_family": "small_mlp",
        "data_provenance": contract.data_provenance,
        "test_only": contract.data_provenance == "synthetic",
        "lineage": verified_lineage,
        "input": {
            "feature_set": "pose12",
            "dimension": len(POSE12_FEATURES),
            "fields_in_exact_order": list(POSE12_FEATURES),
            "torchscript_accepts": "raw, unnormalized pose12 float values",
        },
        "output": {
            "field": "predicted_bias_m",
            "dimension": 1,
            "unit": "m",
            "signed": True,
        },
        "coordinate_convention": {
            "world_frame": "M (motion-capture world)",
            "reference_frame": "A (UWB A antenna frame)",
            "target_frame": "B (UWB B antenna frame)",
            "transform": "A_T_B = inverse(M_T_A) * M_T_B",
            "mapping": "A_p = A_R_B * B_p + A_t_B",
            "translation_unit": "m",
            "rotation_representation": "row-major 3x3 matrix",
        },
        "bias_convention": {
            "label": "bias_m = range_m - distance_gt_m",
            "correction": "corrected_range_m = range_m - predicted_bias_m",
        },
        "network": checkpoint["model_config"],
        "selected_training_seed": int(checkpoint["seed"]),
        "training_sessions": training_sessions,
        "training_domain": {
            "source": "minimum and maximum of the training split only",
            "per_feature": per_feature_domain,
        },
        "normalization": {
            "file": "normalization.json",
            "source": "training split only",
            "formula": "(x - mean) / scale",
        },
        "out_of_domain_behavior": {
            "action": "warn_without_clipping",
            "description": "Inputs outside any per-feature training bound are evaluated unchanged and reported with a warning.",
        },
        "artifacts": {
            "eager_state_dict": "model_state.pt",
            "torchscript": "model_torchscript.pt",
            "train_config": "train_config.yaml",
            "split_manifest": "split_manifest.json",
            "fixed_test_metrics": "metrics.json",
        },
        "export_verification": {
            "probe_count": 3,
            "eager_torchscript_close": True,
            "rtol": 1.0e-6,
            "atol": 1.0e-7,
        },
    }
    write_json(args.output_dir / "model_spec.json", spec)

    versions_source = args.model_state.parent / "versions.json"
    if versions_source.is_file():
        shutil.copyfile(str(versions_source), str(args.output_dir / "versions.json"))
    print("Exported verified model package to {}".format(args.output_dir))
    print("Training domain violations will warn and will not be clipped.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(2)
