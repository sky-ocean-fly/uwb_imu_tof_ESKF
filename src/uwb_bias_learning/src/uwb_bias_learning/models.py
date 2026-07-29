"""Models and train-only feature normalization for UWB bias learning.

This module deliberately does not import PyTorch at module import time.  The
data preparation and NumPy baselines therefore remain usable on ROS machines
where PyTorch has not been installed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import POSE12_FIELDS


POSE12_FEATURES: Tuple[str, ...] = tuple(POSE12_FIELDS)
POSE12_DIM = len(POSE12_FEATURES)
TARGET_COLUMN = "bias_mean"


class TorchUnavailableError(RuntimeError):
    """Raised when a PyTorch-only operation is requested without PyTorch."""


def require_torch() -> Any:
    """Import and return PyTorch, with an actionable missing-dependency error."""

    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is required for MLP training/export/inference but is not "
            "installed. Install it for this Python environment following "
            "https://pytorch.org/get-started/locally/ (for example, a suitable "
            "'python3 -m pip install torch' command), then retry."
        ) from exc


def as_pose12(values: Any, *, name: str = "features") -> np.ndarray:
    """Return finite pose12 data as a two-dimensional float64 array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.shape[0] != POSE12_DIM:
            raise ValueError(
                "{} must contain exactly {} values in pose12 order; got {}".format(
                    name, POSE12_DIM, array.shape[0]
                )
            )
        array = array.reshape(1, POSE12_DIM)
    if array.ndim != 2 or array.shape[1] != POSE12_DIM:
        raise ValueError(
            "{} must have shape (N, {}); got {}".format(
                name, POSE12_DIM, tuple(array.shape)
            )
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinite values".format(name))
    return array


class FeatureNormalizer:
    """Per-feature standardization fitted exclusively from training samples."""

    def __init__(
        self,
        mean: Sequence[float],
        scale: Sequence[float],
        train_min: Sequence[float],
        train_max: Sequence[float],
        feature_names: Sequence[str] = POSE12_FEATURES,
    ) -> None:
        self.feature_names = tuple(feature_names)
        if self.feature_names != POSE12_FEATURES:
            raise ValueError(
                "feature order must exactly match pose12: {}".format(
                    ", ".join(POSE12_FEATURES)
                )
            )
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.train_min = np.asarray(train_min, dtype=np.float64)
        self.train_max = np.asarray(train_max, dtype=np.float64)
        for label, value in (
            ("mean", self.mean),
            ("scale", self.scale),
            ("train_min", self.train_min),
            ("train_max", self.train_max),
        ):
            if value.shape != (POSE12_DIM,):
                raise ValueError("{} must contain {} values".format(label, POSE12_DIM))
            if not np.all(np.isfinite(value)):
                raise ValueError("{} contains NaN or infinite values".format(label))
        if np.any(self.scale <= 0.0):
            raise ValueError("normalization scales must be strictly positive")
        if np.any(self.train_min > self.train_max):
            raise ValueError("train_min must not exceed train_max")

    @classmethod
    def fit(cls, train_features: Any) -> "FeatureNormalizer":
        """Fit mean/scale/range from the training split only."""

        train = as_pose12(train_features, name="train_features")
        if train.shape[0] == 0:
            raise ValueError("cannot fit normalization on an empty training split")
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0, ddof=0)
        # A constant training feature is valid; keeping its denominator at one
        # maps it to zero without injecting information from another split.
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(
            mean=mean,
            scale=scale,
            train_min=np.min(train, axis=0),
            train_max=np.max(train, axis=0),
        )

    def transform(self, values: Any) -> np.ndarray:
        features = as_pose12(values)
        return (features - self.mean) / self.scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_names": list(self.feature_names),
            "fitted_on": "train_only",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "train_min": self.train_min.tolist(),
            "train_max": self.train_max.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureNormalizer":
        fitted_on = payload.get("fitted_on")
        if fitted_on not in ("train", "train_only"):
            raise ValueError(
                "normalization provenance must explicitly be 'train' or 'train_only'"
            )
        if "feature_names" not in payload:
            raise ValueError(
                "normalization must explicitly record the fixed pose12 feature_names"
            )
        return cls(
            mean=payload["mean"],
            scale=payload["scale"],
            train_min=payload["train_min"],
            train_max=payload["train_max"],
            feature_names=payload["feature_names"],
        )

    def save_json(self, path: Any) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

    @classmethod
    def load_json(cls, path: Any) -> "FeatureNormalizer":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def domain_check(self, values: Any, atol: float = 0.0) -> Dict[str, Any]:
        """Describe training-bound violations without modifying the input."""

        features = as_pose12(values)
        below = features < (self.train_min - float(atol))
        above = features > (self.train_max + float(atol))
        outside = below | above
        details: List[Dict[str, Any]] = []
        for row_index, feature_index in np.argwhere(outside):
            details.append(
                {
                    "row": int(row_index),
                    "feature": self.feature_names[int(feature_index)],
                    "value": float(features[row_index, feature_index]),
                    "train_min": float(self.train_min[feature_index]),
                    "train_max": float(self.train_max[feature_index]),
                }
            )
        return {
            "in_training_domain": bool(not np.any(outside)),
            "violations": details,
            "behavior": "warn_without_clipping",
        }


class ZeroBiasModel:
    """Baseline that always predicts zero metres."""

    def predict(self, features: Any) -> np.ndarray:
        count = as_pose12(features).shape[0]
        return np.zeros(count, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "zero_bias", "value_m": 0.0}


class ConstantBiasModel:
    """Baseline that predicts the training-set mean bias."""

    def __init__(self, value_m: float) -> None:
        self.value_m = float(value_m)
        if not np.isfinite(self.value_m):
            raise ValueError("constant bias must be finite")

    @classmethod
    def fit(cls, train_targets: Any) -> "ConstantBiasModel":
        targets = _as_targets(train_targets, "train_targets")
        if targets.size == 0:
            raise ValueError("cannot fit constant baseline on an empty training split")
        return cls(float(np.mean(targets)))

    def predict(self, features: Any) -> np.ndarray:
        count = as_pose12(features).shape[0]
        return np.full(count, self.value_m, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "training_mean", "value_m": self.value_m}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConstantBiasModel":
        return cls(payload["value_m"])


class RidgeBiasModel:
    """Closed-form ridge regression over normalized pose12 features."""

    def __init__(
        self,
        coefficients: Sequence[float],
        intercept: float,
        alpha: float,
        normalizer: FeatureNormalizer,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.shape != (POSE12_DIM,):
            raise ValueError("ridge coefficients must contain 12 values")
        self.intercept = float(intercept)
        self.alpha = float(alpha)
        self.normalizer = normalizer
        if self.alpha < 0.0 or not np.isfinite(self.alpha):
            raise ValueError("ridge alpha must be a finite non-negative value")
        if not np.all(np.isfinite(self.coefficients)) or not np.isfinite(self.intercept):
            raise ValueError("ridge parameters must be finite")

    @classmethod
    def fit(
        cls,
        train_features: Any,
        train_targets: Any,
        alpha: float = 1.0e-3,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> "RidgeBiasModel":
        features = as_pose12(train_features, name="train_features")
        targets = _as_targets(train_targets, "train_targets")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("training features and targets have different row counts")
        if features.shape[0] == 0:
            raise ValueError("cannot fit ridge baseline on an empty training split")
        alpha = float(alpha)
        if alpha < 0.0 or not np.isfinite(alpha):
            raise ValueError("ridge alpha must be a finite non-negative value")
        fitted_normalizer = normalizer or FeatureNormalizer.fit(features)
        normalized = fitted_normalizer.transform(features)
        design = np.column_stack((np.ones(features.shape[0]), normalized))
        penalty = np.eye(POSE12_DIM + 1, dtype=np.float64) * alpha
        penalty[0, 0] = 0.0  # The intercept is deliberately not regularized.
        gram = np.matmul(design.T, design) + penalty
        rhs = np.matmul(design.T, targets)
        try:
            parameters = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            parameters = np.matmul(np.linalg.pinv(gram), rhs)
        return cls(
            coefficients=parameters[1:],
            intercept=float(parameters[0]),
            alpha=alpha,
            normalizer=fitted_normalizer,
        )

    def predict(self, features: Any) -> np.ndarray:
        normalized = self.normalizer.transform(features)
        return self.intercept + np.matmul(normalized, self.coefficients)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ridge",
            "alpha": self.alpha,
            "intercept_m": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "coefficient_feature_order": list(POSE12_FEATURES),
            "normalization": self.normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RidgeBiasModel":
        return cls(
            coefficients=payload["coefficients"],
            intercept=payload["intercept_m"],
            alpha=payload["alpha"],
            normalizer=FeatureNormalizer.from_dict(payload["normalization"]),
        )


def _as_targets(values: Any, name: str = "targets") -> np.ndarray:
    targets = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(targets)):
        raise ValueError("{} contains NaN or infinite values".format(name))
    return targets


def build_mlp(
    hidden_dims: Sequence[int] = (64, 64, 32),
    activation: str = "relu",
) -> Any:
    """Build the small scalar-output PyTorch MLP, importing torch lazily."""

    torch = require_torch()
    dims = [int(value) for value in hidden_dims]
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("hidden_dims must be a non-empty list of positive integers")
    activation_name = str(activation).lower()
    activation_types = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
    }
    if activation_name not in activation_types:
        raise ValueError(
            "unsupported activation {!r}; choose relu, tanh, or gelu".format(
                activation
            )
        )

    class PoseBiasMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: List[Any] = []
            input_dim = POSE12_DIM
            for output_dim in dims:
                layers.append(torch.nn.Linear(input_dim, output_dim))
                layers.append(activation_types[activation_name]())
                input_dim = output_dim
            layers.append(torch.nn.Linear(input_dim, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, features: Any) -> Any:
            return self.network(features)

    return PoseBiasMLP()


# Constructor-style aliases make the public intent clear while preserving the
# lazy PyTorch import (a module-level ``torch.nn.Module`` subclass would import
# torch even for NumPy-only baseline use).
PoseBiasMLP = build_mlp
create_mlp = build_mlp


def build_normalized_inference_module(
    model: Any, normalizer: FeatureNormalizer
) -> Any:
    """Wrap an MLP so exported TorchScript accepts raw pose12 features."""

    torch = require_torch()
    mean = torch.tensor(normalizer.mean, dtype=torch.float32).reshape(1, -1)
    scale = torch.tensor(normalizer.scale, dtype=torch.float32).reshape(1, -1)

    class NormalizedPoseBiasModel(torch.nn.Module):
        def __init__(self, network: Any) -> None:
            super().__init__()
            self.network = network
            self.register_buffer("feature_mean", mean)
            self.register_buffer("feature_scale", scale)

        def forward(self, raw_features: Any) -> Any:
            return self.network(
                (raw_features - self.feature_mean) / self.feature_scale
            )

    return NormalizedPoseBiasModel(model)


def checkpoint_payload(
    model: Any,
    hidden_dims: Sequence[int],
    activation: str,
    seed: int,
    **metadata: Any
) -> Dict[str, Any]:
    """Build a portable state-dict checkpoint with architecture metadata."""

    payload: Dict[str, Any] = {
        "format_version": 1,
        "feature_names": list(POSE12_FEATURES),
        "model_config": {
            "input_dim": POSE12_DIM,
            "hidden_dims": [int(value) for value in hidden_dims],
            "activation": str(activation).lower(),
            "output_dim": 1,
        },
        "seed": int(seed),
        "state_dict": model.state_dict(),
    }
    payload.update(metadata)
    return payload


def load_mlp_checkpoint(
    checkpoint_path: Any, map_location: str = "cpu"
) -> Tuple[Any, Dict[str, Any]]:
    """Reconstruct an MLP from a checkpoint produced by this project."""

    torch = require_torch()
    path = str(checkpoint_path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions predating ``weights_only``.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint {!r} is not a supported MLP checkpoint".format(path))
    if tuple(payload.get("feature_names", ())) != POSE12_FEATURES:
        raise ValueError("checkpoint feature order does not match required pose12 order")
    config = payload.get("model_config", {})
    if int(config.get("input_dim", -1)) != POSE12_DIM or int(
        config.get("output_dim", -1)
    ) != 1:
        raise ValueError("checkpoint model dimensions are incompatible with pose12")
    model = build_mlp(config["hidden_dims"], config.get("activation", "relu"))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def predict_mlp(
    model: Any, raw_features: Any, normalizer: FeatureNormalizer
) -> np.ndarray:
    """Predict metre-valued bias from raw pose12 features using an eager MLP."""

    torch = require_torch()
    normalized = normalizer.transform(raw_features).astype(np.float32)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(normalized)).detach().cpu().numpy()
    return np.asarray(output, dtype=np.float64).reshape(-1)
