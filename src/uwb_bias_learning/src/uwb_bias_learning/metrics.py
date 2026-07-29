"""Numerically explicit metrics for bias prediction and range correction."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np


def _paired_vectors(reference: Any, estimate: Any) -> tuple:
    truth = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(estimate, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError(
            "reference and estimate shapes differ: {} versus {}".format(
                truth.shape, prediction.shape
            )
        )
    if truth.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("metrics inputs contain NaN or infinite values")
    return truth, prediction


def error_metrics(reference: Any, estimate: Any, include_r2: bool = True) -> Dict[str, float]:
    """Compute metrics for ``error = estimate - reference``.

    ``error_std`` is the population standard deviation (``ddof=0``), which is
    well-defined for a single evaluation sample.  R2 is reported as NaN when
    the reference has zero total variation rather than inventing a score.
    """

    truth, prediction = _paired_vectors(reference, estimate)
    error = prediction - truth
    absolute = np.abs(error)
    metrics = {
        "count": int(error.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error": float(np.mean(error)),
        "error_std": float(np.std(error, ddof=0)),
        "p50_abs_error": float(np.percentile(absolute, 50.0)),
        "p90_abs_error": float(np.percentile(absolute, 90.0)),
        "p95_abs_error": float(np.percentile(absolute, 95.0)),
    }
    if include_r2:
        denominator = float(np.sum(np.square(truth - np.mean(truth))))
        numerator = float(np.sum(np.square(error)))
        metrics["r2"] = (
            float("nan") if denominator == 0.0 else float(1.0 - numerator / denominator)
        )
    return metrics


def bias_prediction_metrics(true_bias_m: Any, predicted_bias_m: Any) -> Dict[str, float]:
    """Metrics on per-pose signed mean bias, all in metres where applicable."""

    return error_metrics(true_bias_m, predicted_bias_m, include_r2=True)


def correct_ranges(range_m: Any, predicted_bias_m: Any) -> np.ndarray:
    """Apply the mandated sign convention: ``corrected = raw - predicted_bias``."""

    measured = np.asarray(range_m, dtype=np.float64)
    bias = np.asarray(predicted_bias_m, dtype=np.float64)
    if measured.size == 0:
        raise ValueError("range correction requires at least one range")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(bias)):
        raise ValueError("range correction input contains NaN or infinite values")
    if bias.ndim == 0:
        corrected = measured - float(bias)
    else:
        if measured.shape != bias.shape:
            raise ValueError(
                "range and bias shapes must match exactly unless bias is scalar: "
                "{} versus {}".format(measured.shape, bias.shape)
            )
        corrected = measured - bias
    return corrected


def range_error_metrics(
    range_m: Any, distance_gt_m: Any, predicted_bias_m: Optional[Any] = None
) -> Dict[str, float]:
    """Evaluate raw or corrected range against geometric truth."""

    measured = np.asarray(range_m, dtype=np.float64).reshape(-1)
    truth = np.asarray(distance_gt_m, dtype=np.float64).reshape(-1)
    if measured.shape != truth.shape:
        raise ValueError("range_m and distance_gt_m have different row counts")
    if predicted_bias_m is None:
        estimate = measured
    else:
        bias = np.asarray(predicted_bias_m, dtype=np.float64)
        if bias.ndim == 0:
            bias = np.full(measured.shape, float(bias), dtype=np.float64)
        bias = bias.reshape(-1)
        if bias.shape != measured.shape:
            raise ValueError("predicted_bias_m has a different row count")
        estimate = correct_ranges(measured, bias)
    return error_metrics(truth, estimate, include_r2=False)


def bootstrap_confidence_intervals(
    reference: Any,
    estimate: Any,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap intervals for MAE, RMSE, and mean error.

    Bootstrap output is deterministic for a recorded seed.  It is optional and
    is never substituted for the directly computed metrics.
    """

    truth, prediction = _paired_vectors(reference, estimate)
    repetitions = int(repetitions)
    confidence = float(confidence)
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    random = np.random.RandomState(int(seed))
    values = {"mae": [], "rmse": [], "mean_error": []}
    for _ in range(repetitions):
        indices = random.randint(0, truth.size, size=truth.size)
        sampled_error = prediction[indices] - truth[indices]
        values["mae"].append(float(np.mean(np.abs(sampled_error))))
        values["rmse"].append(float(np.sqrt(np.mean(np.square(sampled_error)))))
        values["mean_error"].append(float(np.mean(sampled_error)))
    tail = (1.0 - confidence) * 50.0
    output: Dict[str, Dict[str, float]] = {}
    for name, samples in values.items():
        output[name] = {
            "lower": float(np.percentile(samples, tail)),
            "upper": float(np.percentile(samples, 100.0 - tail)),
            "confidence": confidence,
            "repetitions": repetitions,
            "seed": int(seed),
        }
    return output


def metrics_by_group(
    reference: Any, estimate: Any, groups: Sequence[Any], include_r2: bool = False
) -> Dict[str, Dict[str, float]]:
    """Compute the same metrics independently for each supplied group label."""

    truth, prediction = _paired_vectors(reference, estimate)
    labels = np.asarray(groups, dtype=object).reshape(-1)
    if labels.shape[0] != truth.shape[0]:
        raise ValueError("group labels have a different row count")
    output: Dict[str, Dict[str, float]] = {}
    # Sorting stringified labels makes JSON/CSV output deterministic even when
    # source CSV row order changes.
    for label in sorted({str(value) for value in labels}):
        mask = np.asarray([str(value) == label for value in labels], dtype=bool)
        output[label] = error_metrics(
            truth[mask], prediction[mask], include_r2=include_r2
        )
    return output


# Small compatibility aliases for callers that use generic metric terminology.
regression_metrics = error_metrics
compute_metrics = error_metrics
apply_bias_correction = correct_ranges
