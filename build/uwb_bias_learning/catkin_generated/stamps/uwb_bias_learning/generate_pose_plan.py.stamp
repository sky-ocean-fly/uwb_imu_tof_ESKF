#!/usr/bin/env python3
"""Generate a space-filling physical pose plan; never generate training truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.config import (  # noqa: E402
    ConfigError,
    load_pose_domain,
    open_exclusive,
    without_private_metadata,
)


AXES = (
    "distance_m",
    "azimuth_deg",
    "elevation_deg",
    "relative_roll_deg",
    "relative_pitch_deg",
    "relative_yaw_deg",
)


def latin_hypercube(count, dimensions, rng):
    if count <= 0:
        return np.empty((0, dimensions), dtype=np.float64)
    result = np.empty((count, dimensions), dtype=np.float64)
    for column in range(dimensions):
        permutation = rng.permutation(count)
        result[:, column] = (permutation + rng.uniform(size=count)) / float(count)
    return result


def scale_unit_samples(unit_samples, bounds):
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    return lower + unit_samples * (upper - lower)


def nominal_translation(distance, azimuth_deg, elevation_deg):
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = distance * math.cos(elevation)
    return (
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        distance * math.sin(elevation),
    )


def build_plan(domain, count_override=None):
    planner = domain.get("planner") or {}
    method = planner.get("method", "latin_hypercube")
    if method != "latin_hypercube":
        raise ConfigError(
            "unsupported planner method %r; this environment provides latin_hypercube"
            % method
        )
    seed = int(planner.get("seed", 2026))
    count = int(
        count_override
        if count_override is not None
        else planner.get("interior_count", 40)
    )
    if count < 1:
        raise ConfigError("interior pose count must be positive")
    bounds = [
        (float(domain[axis]["min"]), float(domain[axis]["max"])) for axis in AXES
    ]
    center = np.asarray([(lo + hi) * 0.5 for lo, hi in bounds])
    rng = np.random.RandomState(seed)
    points = []
    for sample in scale_unit_samples(latin_hypercube(count, len(AXES), rng), bounds):
        points.append(("interior", sample, ""))
    if bool(planner.get("include_center", True)):
        points.append(("center", center.copy(), ""))
    if bool(planner.get("include_axis_boundaries", True)):
        for index, axis in enumerate(AXES):
            for side, value in (("min", bounds[index][0]), ("max", bounds[index][1])):
                sample = center.copy()
                sample[index] = value
                points.append(("boundary_%s_%s" % (axis, side), sample, ""))

    for item in domain.get("typical_points", []) or []:
        if not isinstance(item, dict):
            raise ConfigError("each typical_points entry must be a mapping")
        values = np.asarray([float(item[axis]) for axis in AXES], dtype=np.float64)
        for index, value in enumerate(values):
            lo, hi = bounds[index]
            if value < lo or value > hi:
                raise ConfigError("typical point lies outside %s" % AXES[index])
        points.append(("typical", values, str(item.get("note", ""))))

    base_count = len(points)
    repeat_count = int(planner.get("duplicate_pose_count", 0))
    repeat_indices = []
    if repeat_count > 0:
        repeat_indices = list(
            rng.choice(base_count, size=min(repeat_count, base_count), replace=False)
        )

    rows = []
    for index, (kind, values, note) in enumerate(points, start=1):
        rows.append(make_row(index, kind, values, note, ""))
    for repeat_index in repeat_indices:
        original = rows[repeat_index]
        values = np.asarray([float(original[axis]) for axis in AXES])
        rows.append(
            make_row(
                len(rows) + 1,
                "repeat",
                values,
                "planned repeat for cross-session/repeatability check",
                original["pose_id"],
            )
        )
    return rows, seed


def make_row(index, kind, values, note, repeat_of):
    distance, azimuth, elevation, roll, pitch, yaw = [float(v) for v in values]
    tx, ty, tz = nominal_translation(distance, azimuth, elevation)
    return {
        "pose_id": "P%04d" % index,
        "plan_kind": kind,
        "repeat_of_pose_id": repeat_of,
        "distance_m": distance,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "relative_roll_deg": roll,
        "relative_pitch_deg": pitch,
        "relative_yaw_deg": yaw,
        "nominal_rel_tx_m": tx,
        "nominal_rel_ty_m": ty,
        "nominal_rel_tz_m": tz,
        "collection_status": "PENDING",
        "operator_note": note,
    }


def parse_args():
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(package / "configs" / "pose_domain.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        domain = load_pose_domain(args.config, require_complete=True)
        rows, seed = build_plan(domain, args.count)
        output = Path(args.output).expanduser().resolve()
        with open_exclusive(output, "x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        payload = {
            "schema_version": 1,
            "purpose": "physical_pose_guidance_only",
            "warning": (
                "Nominal plan values must never replace measured mocap relative poses "
                "in a training dataset."
            ),
            "seed": seed,
            "pose_count": len(rows),
            "domain": without_private_metadata(domain),
            "csv": str(output),
        }
        with open_exclusive(manifest, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        print("created pose guidance plan: %s (%d poses)" % (output, len(rows)))
        print("created plan manifest: %s" % manifest)
        return 0
    except (ConfigError, FileExistsError, ValueError) as exc:
        print("pose plan not created: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
