#!/usr/bin/env python3
"""Show planned, completed, attempted, and pending static poses for one session."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.config import write_json_exclusive  # noqa: E402


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _read_plan(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "pose_id" not in reader.fieldnames:
            raise ValueError("pose plan must contain a pose_id column")
        rows = list(reader)
    if not rows:
        raise ValueError("pose plan contains no rows")
    pose_ids = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        pose_id = str(row.get("pose_id", "")).strip()
        if not SAFE_ID_RE.match(pose_id):
            raise ValueError("invalid pose_id on plan row %d: %r" % (row_number, pose_id))
        if pose_id in seen:
            raise ValueError("duplicate pose_id in plan: %s" % pose_id)
        seen.add(pose_id)
        pose_ids.append(pose_id)
    return pose_ids


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def collect_status(plan_path: Path, session_dir: Path):
    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("session manifest does not exist: %s" % manifest_path)
    manifest = _read_json(manifest_path)
    session_id = str(manifest.get("session_id", ""))
    if not session_id:
        raise ValueError("session manifest has no session_id")

    planned = _read_plan(plan_path)
    rows = []
    for pose_id in planned:
        pose_root = session_dir / "poses" / pose_id
        complete_path = pose_root / "COMPLETE.json"
        attempt_dirs = sorted(
            path for path in (pose_root / "attempts").glob("*") if path.is_dir()
        ) if (pose_root / "attempts").is_dir() else []
        latest_summary = None
        if attempt_dirs:
            summary_path = attempt_dirs[-1] / "summary.json"
            if summary_path.is_file():
                latest_summary = _read_json(summary_path)
        if complete_path.is_file():
            complete = _read_json(complete_path)
            state = "COMPLETE"
            valid_count = complete.get("valid_range_count")
            latest_attempt = complete.get("attempt_id", "")
        elif latest_summary:
            state = str(latest_summary.get("final_state", "ATTEMPTED"))
            valid_count = latest_summary.get("valid_range_count")
            latest_attempt = latest_summary.get("attempt_id", "")
        elif attempt_dirs:
            state = "INCOMPLETE_ATTEMPT"
            valid_count = None
            latest_attempt = attempt_dirs[-1].name
        else:
            state = "PENDING"
            valid_count = None
            latest_attempt = ""
        rows.append(
            {
                "pose_id": pose_id,
                "state": state,
                "attempt_count": len(attempt_dirs),
                "latest_attempt_id": latest_attempt,
                "valid_range_count": valid_count,
            }
        )

    planned_set = set(planned)
    unplanned = []
    poses_root = session_dir / "poses"
    if poses_root.is_dir():
        unplanned = sorted(
            path.name
            for path in poses_root.iterdir()
            if path.is_dir() and path.name not in planned_set
        )
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {
        "schema_version": 1,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "plan_csv": str(plan_path),
        "counts": dict(sorted(counts.items())),
        "poses": rows,
        "unplanned_pose_directories": unplanned,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="optional new JSON file; an existing file is never overwritten",
    )
    parser.add_argument("--json", action="store_true", help="print full JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        plan = args.plan.expanduser().resolve()
        session = args.session_dir.expanduser().resolve()
        if not plan.is_file():
            raise ValueError("pose plan does not exist: %s" % plan)
        if not session.is_dir():
            raise ValueError("session directory does not exist: %s" % session)
        payload = collect_status(plan, session)
        if args.output_json:
            write_json_exclusive(args.output_json, payload)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("pose_id\tstate\tattempts\tvalid_ranges\tlatest_attempt")
            for row in payload["poses"]:
                print(
                    "%s\t%s\t%s\t%s\t%s"
                    % (
                        row["pose_id"],
                        row["state"],
                        row["attempt_count"],
                        "" if row["valid_range_count"] is None else row["valid_range_count"],
                        row["latest_attempt_id"],
                    )
                )
            print("counts: " + json.dumps(payload["counts"], ensure_ascii=False, sort_keys=True))
            if payload["unplanned_pose_directories"]:
                print(
                    "warning: unplanned pose directories: "
                    + ", ".join(payload["unplanned_pose_directories"]),
                    file=sys.stderr,
                )
        return 0
    except (OSError, ValueError, json.JSONDecodeError, FileExistsError) as exc:
        print("pose status failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
