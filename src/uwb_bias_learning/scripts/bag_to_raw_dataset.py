#!/usr/bin/env python3
"""Rebuild immutable raw pose rows from rosbag topics and JSON capture states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.capture import (  # noqa: E402
    CAPTURE_STATES,
    CaptureError,
    CaptureSummary,
    CaptureThresholds,
    ExclusiveRawCsvWriter,
    ProcessedMeasurement,
    StatusAssignment,
    StatusAssignmentTracker,
    process_uwb_observation,
    validate_capture_interfaces,
)
from uwb_bias_learning.config import (  # noqa: E402
    ConfigError,
    load_collection_config,
    resolve_config_path,
    write_json_exclusive,
)
from uwb_bias_learning.ros_adapters import (  # noqa: E402
    RosAdapterError,
    extract_pose_stamped,
    get_uwb_adapter,
)
from uwb_bias_learning.synchronization import TimedPose  # noqa: E402
from uwb_bias_learning.session import validate_manifest_metadata  # noqa: E402


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class TimelineItem:
    timestamp: float
    ordinal: int
    kind: str
    value: Any


@dataclass(frozen=True)
class AssignedRange:
    observation: Any
    assignment: StatusAssignment


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise CaptureError("cannot read %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise CaptureError("JSON manifest must contain an object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise CaptureError("cannot hash config file %s: %s" % (path, exc))
    return digest.hexdigest()


def _session_manifest(collection, session_id: str, session_dir: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    directory = (
        Path(session_dir).expanduser().resolve()
        if session_dir
        else resolve_config_path(collection, collection["output"]["raw_root"]) / session_id
    )
    if not directory.is_dir():
        raise CaptureError("session directory does not exist: %s" % directory)
    manifest = _read_json(directory / "session_manifest.json")
    if manifest.get("session_id") != session_id:
        raise CaptureError("session manifest ID does not match --session-id")
    metadata = validate_manifest_metadata(directory, manifest)
    if not isinstance(metadata, dict) or metadata.get("los") is not True:
        raise CaptureError("session manifest must explicitly confirm metadata.los: true")
    snapshot_dir = directory / "config_snapshot"
    comparisons = (
        (Path(collection["_config_path"]).resolve(), snapshot_dir / "collection.yaml"),
        (Path(collection["_ros_topics_path"]).resolve(), snapshot_dir / "ros_topics.yaml"),
    )
    for current, snapshot in comparisons:
        if not snapshot.is_file() or _sha256(current) != _sha256(snapshot):
            raise CaptureError(
                "current config does not byte-match the immutable session snapshot: %s" % snapshot
            )
    topics = collection["ros_topics"]
    expected_topics = {name: topics[name]["topic"] for name in ("uwb", "mocap_a", "mocap_b", "status")}
    if manifest.get("topics") != expected_topics:
        raise CaptureError("current topic configuration differs from the immutable session manifest")
    expected_pair = {
        "source_id": topics["uwb"]["source_id"],
        "peer_id": topics["uwb"]["peer_id"],
        "direction": topics["uwb"]["direction"],
    }
    if manifest.get("uwb_pair") != expected_pair:
        raise CaptureError("current directed UWB pair differs from the immutable session manifest")
    return directory.resolve(), manifest


def _topic_types_from_bag(bag) -> Dict[str, str]:
    info = bag.get_type_and_topic_info()
    result = {}
    for topic, value in info.topics.items():
        result[str(topic)] = str(value.msg_type)
    return result


def _validate_bag_topic_types(observed_types: Dict[str, set], topics) -> None:
    reasons = []
    for role in ("uwb", "mocap_a", "mocap_b", "status"):
        topic = topics[role]["topic"]
        expected = topics[role]["message_type"]
        actual = observed_types.get(topic, set())
        if not actual:
            reasons.append("configured %s topic absent from all bags: %s" % (role, topic))
        elif actual != {expected}:
            reasons.append(
                "%s type mismatch for %s: configured %s, bag has %s"
                % (role, topic, expected, ",".join(sorted(actual)))
            )
    if reasons:
        raise CaptureError("bag interface validation failed: " + "; ".join(reasons))


def _normalize_pose_samples(
    samples: Sequence[TimedPose],
) -> Tuple[List[TimedPose], int, int]:
    """Mirror live handling: retain the first sample and reject later <= stamps."""

    normalized: List[TimedPose] = []
    duplicate_count = 0
    reversed_count = 0
    for sample in samples:
        if normalized and sample.timestamp <= normalized[-1].timestamp:
            if sample.timestamp == normalized[-1].timestamp:
                duplicate_count += 1
            else:
                reversed_count += 1
            continue
        normalized.append(sample)
    return normalized, duplicate_count, reversed_count


def _parse_status_payload(raw: Any) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _assign_timeline(
    timeline: Sequence[TimelineItem], session_id: str
) -> Tuple[List[AssignedRange], StatusAssignmentTracker, Dict[str, Dict[str, Any]], int]:
    tracker = StatusAssignmentTracker(session_id)
    assigned = []
    outcomes: Dict[str, Dict[str, Any]] = {}
    unassigned = 0
    for item in sorted(timeline, key=lambda value: (value.timestamp, value.ordinal)):
        if item.kind == "status":
            payload = _parse_status_payload(item.value)
            invalid_before = tracker.invalid_event_count
            tracker.feed(payload if payload is not None else item.value)
            if (
                payload is not None
                and tracker.invalid_event_count == invalid_before
                and str(payload.get("session_id", "")) == session_id
                and payload.get("state") in ("COMPLETE", "ABORTED")
                and payload.get("attempt_id") not in (None, "")
                and tracker.terminal_outcomes.get(str(payload["attempt_id"]))
                == str(payload["state"])
            ):
                outcomes[str(payload["attempt_id"])] = dict(payload)
        else:
            if tracker.active is None:
                unassigned += 1
            else:
                assigned.append(AssignedRange(item.value, tracker.active))
    return assigned, tracker, outcomes, unassigned


def _invalidate(processed: ProcessedMeasurement, reason: str) -> ProcessedMeasurement:
    existing = str(processed.row.get("invalid_reason", ""))
    reasons = [value for value in existing.split(";") if value]
    if reason not in reasons:
        reasons.append(reason)
    processed.row["invalid_reason"] = ";".join(reasons)
    processed.row["is_valid"] = 0
    return ProcessedMeasurement(
        row=processed.row,
        is_valid=False,
        stability=processed.stability,
        sync_a=processed.sync_a,
        sync_b=processed.sync_b,
    )


def convert_bags(
    *,
    collection,
    session_id: str,
    environment_id: str,
    data_origin: str,
    is_synthetic: bool,
    bag_paths: Sequence[Path],
    output_csv: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    try:
        import rosbag
    except ImportError as exc:
        raise CaptureError("ROS rosbag Python environment is not sourced: %s" % exc)

    topics = collection["ros_topics"]
    thresholds = CaptureThresholds.from_collection(collection)
    adapter = get_uwb_adapter(topics["uwb"]["adapter"])
    relevant_topics = [topics[name]["topic"] for name in ("uwb", "mocap_a", "mocap_b", "status")]
    observed_types: Dict[str, set] = {}
    pose_a = []
    pose_b = []
    timeline = []
    counts = Counter()
    status_envelope_delays = []
    ordinal = 0

    for bag_path in bag_paths:
        if not bag_path.is_file():
            raise CaptureError("bag file does not exist: %s" % bag_path)
        try:
            bag = rosbag.Bag(str(bag_path), "r")
        except Exception as exc:
            raise CaptureError("cannot open bag %s: %s" % (bag_path, exc))
        try:
            for topic, msg_type in _topic_types_from_bag(bag).items():
                observed_types.setdefault(topic, set()).add(msg_type)
            for topic, message, bag_stamp in bag.read_messages(topics=relevant_topics):
                ordinal += 1
                receive_timestamp = float(bag_stamp.to_sec())
                counts["messages_total"] += 1
                if topic == topics["mocap_a"]["topic"] or topic == topics["mocap_b"]["topic"]:
                    role = "mocap_a" if topic == topics["mocap_a"]["topic"] else "mocap_b"
                    try:
                        observation = extract_pose_stamped(
                            message,
                            receive_timestamp,
                            require_nonzero_stamp=bool(
                                topics.get("time", {}).get("require_nonzero_mocap_header_stamp", True)
                            ),
                        )
                        if observation.frame_id != topics[role]["expected_frame_id"]:
                            raise RosAdapterError(
                                "frame_id %r does not match configured %r"
                                % (observation.frame_id, topics[role]["expected_frame_id"])
                            )
                        sample = TimedPose(
                            observation.timestamp,
                            observation.position,
                            observation.quaternion_xyzw,
                        )
                        (pose_a if role == "mocap_a" else pose_b).append(sample)
                        counts[role + "_valid"] += 1
                    except (RosAdapterError, ValueError) as exc:
                        counts[role + "_invalid:" + str(exc)] += 1
                elif topic == topics["status"]["topic"]:
                    raw = getattr(message, "data", None)
                    if raw is None:
                        counts["status_missing_data_field"] += 1
                    else:
                        payload = _parse_status_payload(raw)
                        try:
                            effective_timestamp = float(payload["ros_timestamp"])
                        except (KeyError, TypeError, ValueError):
                            counts["status_invalid_or_missing_ros_timestamp"] += 1
                            continue
                        if not math.isfinite(effective_timestamp) or effective_timestamp <= 0.0:
                            counts["status_invalid_or_missing_ros_timestamp"] += 1
                            continue
                        status_envelope_delays.append(receive_timestamp - effective_timestamp)
                        timeline.append(
                            TimelineItem(effective_timestamp, ordinal, "status", payload)
                        )
                        counts["status_messages"] += 1
                else:
                    try:
                        extraction = adapter(
                            message,
                            source_id=int(topics["uwb"]["source_id"]),
                            peer_id=int(topics["uwb"]["peer_id"]),
                            receive_timestamp=receive_timestamp,
                            timestamp_source=topics["uwb"]["timestamp_source"],
                        )
                    except (RosAdapterError, ValueError) as exc:
                        counts["uwb_adapter_error:" + str(exc)] += 1
                        continue
                    if extraction.accepted:
                        timeline.append(
                            TimelineItem(receive_timestamp, ordinal, "uwb", extraction.observation)
                        )
                        counts["uwb_target_pair"] += 1
                    else:
                        counts["uwb_rejected:" + extraction.invalid_reason] += 1
        finally:
            bag.close()

    _validate_bag_topic_types(observed_types, topics)
    pose_a, duplicates_a, reversed_a = _normalize_pose_samples(pose_a)
    pose_b, duplicates_b, reversed_b = _normalize_pose_samples(pose_b)
    if len(pose_a) < 2 or len(pose_b) < 2:
        raise CaptureError("bags do not contain two unambiguous samples for each mocap stream")
    assigned, tracker, outcomes, unassigned = _assign_timeline(timeline, session_id)
    if tracker.event_count == 0:
        raise CaptureError("no valid status JSON events found for session %s" % session_id)
    if not assigned:
        raise CaptureError("no target-pair UWB ranges fall inside reconstructed pose segments")

    if output_csv.exists():
        raise FileExistsError("output CSV already exists: %s" % output_csv)
    if summary_json.exists():
        raise FileExistsError("summary JSON already exists: %s" % summary_json)

    summary = CaptureSummary()
    pose_sample_indices = Counter()
    pose_counts = Counter()
    valid_counts_by_attempt = Counter()
    last_uwb_timestamp_by_attempt: Dict[str, float] = {}
    with ExclusiveRawCsvWriter(
        output_csv,
        float_precision=int(collection["output"].get("float_precision", 17)),
    ) as writer:
        for record in assigned:
            assignment = record.assignment
            key = (assignment.session_id, assignment.pose_id, assignment.attempt_id)
            pose_sample_indices[key] += 1
            previous_uwb_timestamp = last_uwb_timestamp_by_attempt.get(assignment.attempt_id)
            timestamp_reason = ""
            if (
                previous_uwb_timestamp is not None
                and record.observation.timestamp <= previous_uwb_timestamp
            ):
                timestamp_reason = "uwb_timestamp_non_increasing"
            last_uwb_timestamp_by_attempt[assignment.attempt_id] = record.observation.timestamp
            processed = process_uwb_observation(
                record.observation,
                pose_a,
                pose_b,
                thresholds,
                session_id=assignment.session_id,
                environment_id=environment_id,
                pose_id=assignment.pose_id,
                attempt_id=assignment.attempt_id,
                sample_index=pose_sample_indices[key],
                capture_state=assignment.state,
                mocap_a_frame_id=topics["mocap_a"]["expected_frame_id"],
                mocap_b_frame_id=topics["mocap_b"]["expected_frame_id"],
                is_los=True,
                data_origin=data_origin,
                is_synthetic=is_synthetic,
                ingestion_path="rosbag_replay",
                operator_note=assignment.operator_note,
                upstream_invalid_reason=timestamp_reason,
            )
            outcome = outcomes.get(assignment.attempt_id)
            outcome_state = outcome.get("state") if outcome is not None else None
            if outcome_state != "COMPLETE":
                processed = _invalidate(
                    processed,
                    "attempt_not_complete"
                    if outcome_state is None
                    else "attempt_" + str(outcome_state).lower(),
                )
            if assignment.attempt_id in tracker.invalid_attempt_reasons:
                processed = _invalidate(
                    processed,
                    "status_lifecycle_invalid:"
                    + tracker.invalid_attempt_reasons[assignment.attempt_id],
                )
            if outcome_state == "COMPLETE" and processed.is_valid:
                try:
                    target_count = int(outcome["target_valid_ranges"])
                except (KeyError, TypeError, ValueError):
                    target_count = 0
                if target_count <= 0:
                    processed = _invalidate(processed, "terminal_missing_target_valid_ranges")
                elif valid_counts_by_attempt[assignment.attempt_id] >= target_count:
                    processed = _invalidate(processed, "after_target_valid_count")
                else:
                    valid_counts_by_attempt[assignment.attempt_id] += 1
            writer.write(processed.row)
            summary.add(processed)
            pose_counts[assignment.pose_id + ":rows"] += 1
            if processed.is_valid:
                pose_counts[assignment.pose_id + ":valid"] += 1

    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "environment_id": environment_id,
        "data_origin": data_origin,
        "is_synthetic": is_synthetic,
        "ingestion_path": "rosbag_replay",
        "training_eligible": True,
        "input_bags": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in bag_paths
        ],
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha256(output_csv),
        "configured_topics": {name: topics[name]["topic"] for name in ("uwb", "mocap_a", "mocap_b", "status")},
        "directed_uwb_pair": {
            "source_id": topics["uwb"]["source_id"],
            "peer_id": topics["uwb"]["peer_id"],
            "direction": topics["uwb"]["direction"],
        },
        "bag_message_counts": dict(sorted(counts.items())),
        "status_envelope_delay_s": {
            "minimum": min(status_envelope_delays) if status_envelope_delays else None,
            "maximum": max(status_envelope_delays) if status_envelope_delays else None,
        },
        "mocap_timestamp_rows_omitted": {
            "mocap_a": {
                "duplicate": duplicates_a,
                "reversed": reversed_a,
            },
            "mocap_b": {
                "duplicate": duplicates_b,
                "reversed": reversed_b,
            },
        },
        "status_audit": {
            "valid_session_event_count": tracker.event_count,
            "invalid_event_count": tracker.invalid_event_count,
            "ignored_other_session_count": tracker.ignored_other_session,
            "unassigned_target_pair_ranges": unassigned,
            "attempt_outcomes": {
                key: value.get("state") for key, value in sorted(outcomes.items())
            },
            "invalid_attempt_reasons": dict(sorted(tracker.invalid_attempt_reasons.items())),
        },
        "pose_counts": dict(sorted(pose_counts.items())),
        "is_los": True,
        **summary.payload(),
    }
    write_json_exclusive(summary_json, payload)
    return payload


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "collection.yaml"),
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-dir")
    parser.add_argument("--input-bag", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly allow a session whose immutable metadata declares synthetic_test.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        if not SAFE_ID_RE.match(args.session_id):
            raise CaptureError(
                "session_id must start with an alphanumeric character and contain only "
                "letters, digits, '.', '_' or '-'"
            )
        collection = load_collection_config(args.config, live=True)
        validate_capture_interfaces(collection)
        _, session_manifest = _session_manifest(
            collection, args.session_id, args.session_dir
        )
        environment_id = str(
            session_manifest.get("metadata", {}).get("environment_id", "")
        ).strip()
        if not environment_id:
            raise CaptureError("session manifest must contain a non-empty environment_id")
        metadata = session_manifest["metadata"]
        data_origin = str(metadata["data_origin"])
        is_synthetic = bool(metadata["is_synthetic"])
        if is_synthetic and not args.allow_synthetic:
            raise CaptureError("synthetic session requires explicit --allow-synthetic")
        output_csv = Path(args.output_csv).expanduser().resolve()
        summary_json = (
            Path(args.summary_json).expanduser().resolve()
            if args.summary_json
            else output_csv.with_suffix(".summary.json")
        )
        if output_csv == summary_json:
            raise CaptureError("output CSV and summary JSON paths must differ")
        bag_paths = [Path(value).expanduser().resolve() for value in args.input_bag]
        if len(set(bag_paths)) != len(bag_paths):
            raise CaptureError("duplicate --input-bag paths are not allowed")
        payload = convert_bags(
            collection=collection,
            session_id=args.session_id,
            environment_id=environment_id,
            data_origin=data_origin,
            is_synthetic=is_synthetic,
            bag_paths=bag_paths,
            output_csv=output_csv,
            summary_json=summary_json,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CaptureError, ConfigError, RosAdapterError, FileExistsError, OSError, ValueError) as exc:
        print("bag conversion failed: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
