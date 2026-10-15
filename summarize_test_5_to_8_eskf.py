#!/usr/bin/env python3
"""Summarize and plot the eight CSV outputs of run_test_5_to_8_eskf.py."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eskf_test_5_to_8_results"
NODES = {
    "uwb_imu_tof_eskf_real_add_imu_bias_node": "With IMU bias",
    "uwb_imu_tof_eskf_real_node": "Without IMU bias",
}


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


summary = []
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
for ax, bag in zip(axes.flat, range(5, 9)):
    for node, label in NODES.items():
        csv_path = RESULTS / f"test_{bag}_{node}.csv"
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        times = [float(row["time"]) for row in rows]
        positions = [float(row["err_pos_norm"]) for row in rows]
        rotations = [float(row["err_rot_deg"]) for row in rows]
        axes_in_bounds = []
        for row in rows:
            axes_in_bounds.append(all(
                float(row[f"pos_3sigma_lower_{axis}_m"])
                <= float(row[f"consistency_err_{axis}"])
                <= float(row[f"pos_3sigma_upper_{axis}_m"])
                for axis in "xyz"))
        summary.append({
            "bag": f"test_{bag}.bag", "node": node,
            "samples": len(rows), "duration_s": f"{times[-1] - times[0]:.3f}",
            "position_mean_m": f"{sum(positions) / len(positions):.4f}",
            "position_rmse_m": f"{rms(positions):.4f}",
            "position_max_m": f"{max(positions):.4f}",
            "rotation_rmse_deg": f"{rms(rotations):.3f}",
            "position_3sigma_all_axes_fraction":
                f"{sum(axes_in_bounds) / len(axes_in_bounds):.4f}",
        })
        ax.plot(times, positions, linewidth=1, label=label)
    ax.set_title(f"test_{bag}.bag")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(alpha=0.25)
axes.flat[0].legend()
fig.tight_layout()
fig.savefig(RESULTS / "position_error_comparison.png", dpi=170)
plt.close(fig)

summary_path = RESULTS / "summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
