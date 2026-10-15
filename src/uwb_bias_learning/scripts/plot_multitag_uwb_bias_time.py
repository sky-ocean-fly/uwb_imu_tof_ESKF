#!/usr/bin/env python3
"""Plot per-link UWB bias versus elapsed bag time."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REQUIRED_COLUMNS = ("bag_timestamp_s", "tag_id", "anchor_id", "bias_m")
TAG_IDS = (5, 6, 7, 8)
ANCHOR_IDS = (0, 1, 2, 3, 4, 9)


class PlotError(ValueError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot UWB bias over elapsed bag time")
    parser.add_argument("--input", required=True, help="Per-sample bias CSV")
    parser.add_argument("--overview-output", required=True, help="New four-panel PNG")
    parser.add_argument("--detail-output", required=True, help="New 4x6-panel PNG")
    parser.add_argument(
        "--smooth-window-s",
        type=float,
        default=1.0,
        help="Centered moving-average width in seconds (default: %(default)s)",
    )
    return parser


def _paths(args) -> Tuple[Path, Path, Path]:
    source = Path(args.input).expanduser().resolve()
    overview = Path(args.overview_output).expanduser().resolve()
    detail = Path(args.detail_output).expanduser().resolve()
    if not source.is_file():
        raise PlotError("input CSV does not exist: {}".format(source))
    if overview == detail:
        raise PlotError("overview and detail outputs must differ")
    for output in (overview, detail):
        if output.exists():
            raise PlotError("refusing to overwrite existing plot: {}".format(output))
    if not math.isfinite(args.smooth_window_s) or args.smooth_window_s <= 0.0:
        raise PlotError("smooth-window-s must be finite and positive")
    return source, overview, detail


def _read(path: Path) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    grouped = defaultdict(lambda: ([], []))
    all_timestamps = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise PlotError("input CSV has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise PlotError("input CSV is missing columns: {}".format(", ".join(missing)))
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["bag_timestamp_s"])
                tag_id = int(row["tag_id"])
                anchor_id = int(row["anchor_id"])
                bias = float(row["bias_m"])
            except (TypeError, ValueError) as exc:
                raise PlotError("invalid numeric value on row {}".format(row_number)) from exc
            if not all(math.isfinite(value) for value in (timestamp, bias)):
                raise PlotError("non-finite timestamp or bias on row {}".format(row_number))
            if tag_id not in TAG_IDS or anchor_id not in ANCHOR_IDS:
                continue
            grouped[(tag_id, anchor_id)][0].append(timestamp)
            grouped[(tag_id, anchor_id)][1].append(bias)
            all_timestamps.append(timestamp)
    if not all_timestamps:
        raise PlotError("no configured tag-anchor rows found")
    origin = min(all_timestamps)
    result = {}
    for key in ((tag, anchor) for tag in TAG_IDS for anchor in ANCHOR_IDS):
        timestamps, biases = grouped.get(key, ([], []))
        if not timestamps:
            raise PlotError("no samples for tag {} anchor {}".format(*key))
        order = np.argsort(timestamps)
        result[key] = (
            np.asarray(timestamps, dtype=float)[order] - origin,
            np.asarray(biases, dtype=float)[order],
        )
    return result


def _time_average(time_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    half = 0.5 * window_s
    left = np.searchsorted(time_s, time_s - half, side="left")
    right = np.searchsorted(time_s, time_s + half, side="right")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return (cumulative[right] - cumulative[left]) / (right - left)


def _plot_overview(data, output: Path, window_s: float) -> None:
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, sharey=True)
    for row, tag_id in enumerate(TAG_IDS):
        axis = axes[row]
        for index, anchor_id in enumerate(ANCHOR_IDS):
            time_s, bias = data[(tag_id, anchor_id)]
            smooth = _time_average(time_s, bias, window_s)
            axis.plot(
                time_s,
                100.0 * smooth,
                color=colors(index),
                linewidth=1.45,
                label="anchor {}".format(anchor_id),
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
        axis.set_ylabel("Tag {} bias (cm)".format(tag_id))
        axis.grid(True, linewidth=0.45, alpha=0.32)
    axes[0].legend(ncol=6, loc="upper center", fontsize=9)
    axes[-1].set_xlabel("Elapsed bag time (s)")
    figure.suptitle(
        "test_2.bag UWB bias over time ({} s centered moving average)".format(window_s),
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output), dpi=180)
    plt.close(figure)


def _plot_detail(data, output: Path, window_s: float) -> None:
    figure, axes = plt.subplots(4, 6, figsize=(25, 14), sharex=True)
    for row, tag_id in enumerate(TAG_IDS):
        for column, anchor_id in enumerate(ANCHOR_IDS):
            axis = axes[row, column]
            time_s, bias = data[(tag_id, anchor_id)]
            smooth = _time_average(time_s, bias, window_s)
            axis.plot(time_s, 100.0 * bias, color="#7aa6d8", linewidth=0.45, alpha=0.40)
            axis.plot(time_s, 100.0 * smooth, color="#d62728", linewidth=1.35)
            mean_cm = 100.0 * float(np.mean(bias))
            axis.axhline(0.0, color="black", linewidth=0.65, alpha=0.5)
            axis.axhline(mean_cm, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.85)
            axis.set_title("T{}-A{}  mean={:.1f} cm".format(tag_id, anchor_id, mean_cm), fontsize=10)
            axis.grid(True, linewidth=0.4, alpha=0.25)
            if column == 0:
                axis.set_ylabel("Bias (cm)")
            if row == len(TAG_IDS) - 1:
                axis.set_xlabel("Time (s)")
    figure.suptitle(
        "test_2.bag per-link bias: raw (blue), {} s mean (red), link mean (green)".format(
            window_s
        ),
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output), dpi=170)
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    try:
        source, overview, detail = _paths(args)
        data = _read(source)
        _plot_overview(data, overview, args.smooth_window_s)
        _plot_detail(data, detail, args.smooth_window_s)
    except (OSError, PlotError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    print("overview: {}".format(overview))
    print("detail: {}".format(detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
