#!/usr/bin/env python3
"""Capture one static pose from a confirmed UWB stream and two mocap streams."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from uwb_bias_learning.capture import (  # noqa: E402
    CAPTURE_STATES,
    CaptureError,
    CaptureSummary,
    CaptureThresholds,
    ExclusiveRawCsvWriter,
    assess_static_window,
    process_uwb_observation,
    validate_capture_interfaces,
)
from uwb_bias_learning.config import (  # noqa: E402
    ConfigError,
    load_collection_config,
    open_exclusive,
    resolve_config_path,
    write_json_exclusive,
)
from uwb_bias_learning.ros_adapters import (  # noqa: E402
    RosAdapterError,
    extract_pose_stamped,
    get_uwb_adapter,
    load_ros_message_class,
)
from uwb_bias_learning.session import append_event, validate_manifest_metadata  # noqa: E402
from uwb_bias_learning.synchronization import TimedPose  # noqa: E402


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _attempt_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-p%d" % os.getpid()


def _validate_id(value: str, name: str) -> str:
    if not SAFE_ID_RE.match(value):
        raise CaptureError(
            "%s must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'" % name
        )
    return value


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


def _resolve_session_dir(collection, session_id: str, session_dir, output_root):
    if session_dir is not None and output_root is not None:
        raise CaptureError("use only one of --session-dir and --output-root")
    if session_dir is not None:
        path = Path(session_dir).expanduser().resolve()
    else:
        root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else resolve_config_path(collection, collection["output"]["raw_root"])
        )
        path = root / session_id
    if not path.is_dir():
        raise CaptureError(
            "session directory does not exist: %s; start record_session.sh first" % path
        )
    manifest = _read_json(path / "session_manifest.json")
    if manifest.get("session_id") != session_id:
        raise CaptureError("session manifest ID does not match --session-id")
    metadata = validate_manifest_metadata(path, manifest)
    if not isinstance(metadata, dict) or metadata.get("los") is not True:
        raise CaptureError("session manifest must explicitly confirm metadata.los: true")
    if not str(metadata.get("environment_id", "")).strip():
        raise CaptureError("session manifest must contain a non-empty environment_id")
    snapshot_dir = path / "config_snapshot"
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
    pair = manifest.get("uwb_pair", {})
    expected_pair = {
        "source_id": topics["uwb"]["source_id"],
        "peer_id": topics["uwb"]["peer_id"],
        "direction": topics["uwb"]["direction"],
    }
    if pair != expected_pair:
        raise CaptureError("current directed UWB pair differs from the immutable session manifest")
    return path.resolve(), manifest


@dataclass(frozen=True)
class AttemptPaths:
    pose_root: Path
    attempt_dir: Path
    attempt_id: str
    raw_csv: Path
    events_jsonl: Path
    summary_json: Path
    complete_marker: Path


def _prepare_attempt(session_dir: Path, pose_id: str) -> AttemptPaths:
    pose_root = session_dir / "poses" / pose_id
    pose_root.mkdir(parents=True, exist_ok=True)
    complete_marker = pose_root / "COMPLETE.json"
    if complete_marker.exists():
        raise CaptureError(
            "pose_id %s is already complete; existing data will not be overwritten" % pose_id
        )
    attempts = pose_root / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt_id = _attempt_id()
    attempt_dir = attempts / attempt_id
    attempt_dir.mkdir(exist_ok=False)
    events = attempt_dir / "capture_events.jsonl"
    with open_exclusive(events, "x", encoding="utf-8"):
        pass
    return AttemptPaths(
        pose_root=pose_root,
        attempt_dir=attempt_dir,
        attempt_id=attempt_id,
        raw_csv=attempt_dir / "raw_ranges.csv",
        events_jsonl=events,
        summary_json=attempt_dir / "summary.json",
        complete_marker=complete_marker,
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class PendingRange:
    observation: Any
    enqueued_monotonic: float


class LiveCapture:
    def __init__(
        self,
        *,
        rospy,
        collection,
        session_dir: Path,
        session_id: str,
        environment_id: str,
        pose_id: str,
        paths: AttemptPaths,
        target_valid_ranges: int,
        operator_note: str,
        health_timeout_s: float,
        max_duration_s: Optional[float],
    ):
        self.rospy = rospy
        self.collection = collection
        self.topics = collection["ros_topics"]
        self.thresholds = CaptureThresholds.from_collection(collection)
        self.session_dir = session_dir
        self.session_id = session_id
        self.environment_id = str(environment_id)
        metadata = validate_manifest_metadata(session_dir, _read_json(session_dir / "session_manifest.json"))
        self.data_origin = str(metadata["data_origin"])
        self.is_synthetic = bool(metadata["is_synthetic"])
        self.pose_id = pose_id
        self.paths = paths
        self.target_valid_ranges = int(target_valid_ranges)
        self.operator_note = operator_note
        self.health_timeout_s = float(health_timeout_s)
        self.max_duration_s = max_duration_s
        self.capture_deadline_monotonic: Optional[float] = None
        self.stream_timeout_s = float(collection["stream_timeout_s"])
        self.settling_time_s = float(collection["settling_time_s"])
        self.status_period_s = 1.0 / float(collection["status_publish_rate_hz"])
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.pose_a = deque()
        self.pose_b = deque()
        self.pending = deque()
        self.receive_monotonic = {name: deque(maxlen=2000) for name in ("uwb", "mocap_a", "mocap_b")}
        self.valid_receive_monotonic = {
            name: deque(maxlen=2000) for name in ("uwb", "mocap_a", "mocap_b")
        }
        self.receive_ros = {name: None for name in ("uwb", "mocap_a", "mocap_b")}
        self.valid_messages = Counter()
        self.invalid_messages = Counter()
        self.uwb_extraction_reasons = Counter()
        self.recording_active = False
        self.recording_start_ros_timestamp: Optional[float] = None
        self.stream_faults = deque()
        self.state: Optional[str] = None
        self.state_reason = ""
        self.confirmed_uwb_time: Optional[float] = None
        self.stable_since_uwb_time: Optional[float] = None
        self.last_status_monotonic = 0.0
        self.sample_index = 0
        self.last_processed_uwb_timestamp: Optional[float] = None
        self.summary = CaptureSummary()
        self._summary_written = False
        self._subscribers = []
        self._status_publisher = None
        self._status_class = None
        self._writer = ExclusiveRawCsvWriter(
            paths.raw_csv,
            float_precision=int(collection["output"].get("float_precision", 17)),
        )
        self._uwb_adapter = get_uwb_adapter(self.topics["uwb"]["adapter"])

    def install_ros_interfaces(self) -> None:
        uwb_cls = load_ros_message_class(self.topics["uwb"]["message_type"])
        pose_a_cls = load_ros_message_class(self.topics["mocap_a"]["message_type"])
        pose_b_cls = load_ros_message_class(self.topics["mocap_b"]["message_type"])
        self._status_class = load_ros_message_class(self.topics["status"]["message_type"])
        probe = self._status_class()
        if not hasattr(probe, "data"):
            raise CaptureError("configured status message type has no data field for JSON")
        self._status_publisher = self.rospy.Publisher(
            self.topics["status"]["topic"], self._status_class, queue_size=20, latch=True
        )
        self._subscribers = [
            self.rospy.Subscriber(self.topics["uwb"]["topic"], uwb_cls, self._uwb_callback, queue_size=2000),
            self.rospy.Subscriber(
                self.topics["mocap_a"]["topic"], pose_a_cls, self._pose_a_callback, queue_size=2000
            ),
            self.rospy.Subscriber(
                self.topics["mocap_b"]["topic"], pose_b_cls, self._pose_b_callback, queue_size=2000
            ),
        ]

    def _mark_received(self, role: str, receive_ros: float) -> float:
        now = time.monotonic()
        self.receive_monotonic[role].append(now)
        self.receive_ros[role] = float(receive_ros)
        return now

    def _append_pose(self, role: str, message) -> None:
        receive_ros = float(self.rospy.Time.now().to_sec())
        with self.lock:
            received_monotonic = self._mark_received(role, receive_ros)
        try:
            observation = extract_pose_stamped(
                message,
                receive_ros,
                require_nonzero_stamp=bool(
                    self.topics.get("time", {}).get("require_nonzero_mocap_header_stamp", True)
                ),
            )
            expected_frame = self.topics[role]["expected_frame_id"]
            if observation.frame_id != expected_frame:
                raise RosAdapterError(
                    "frame_id %r does not match configured %r" % (observation.frame_id, expected_frame)
                )
            sample = TimedPose(observation.timestamp, observation.position, observation.quaternion_xyzw)
        except (RosAdapterError, ValueError) as exc:
            with self.lock:
                reason = role + ":" + str(exc)
                self.invalid_messages[reason] += 1
                if self.recording_active:
                    self.stream_faults.append(reason)
            return

        target = self.pose_a if role == "mocap_a" else self.pose_b
        with self.lock:
            if target and sample.timestamp <= target[-1].timestamp:
                reason = role + ":non_increasing_header_timestamp"
                self.invalid_messages[reason] += 1
                if self.recording_active:
                    self.stream_faults.append(reason)
                return
            target.append(sample)
            self.valid_messages[role] += 1
            self.valid_receive_monotonic[role].append(received_monotonic)
            cutoff = self._pose_retention_cutoff(sample.timestamp)
            while len(target) > 2 and target[1].timestamp < cutoff:
                target.popleft()

    def _pose_retention_cutoff(self, newest_timestamp: float) -> float:
        retain_s = max(
            5.0,
            3.0 * self.thresholds.stability_window_s + self.stream_timeout_s,
        )
        cutoff = float(newest_timestamp) - retain_s
        if self.pending:
            # A queued UWB observation needs mocap coverage back through the
            # full stability window. Never prune that history while it waits.
            pending_start = (
                float(self.pending[0].observation.timestamp)
                - self.thresholds.stability_window_s
                - self.thresholds.max_sync_gap_s
            )
            cutoff = min(cutoff, pending_start)
        return cutoff

    def _pose_a_callback(self, message) -> None:
        self._append_pose("mocap_a", message)

    def _pose_b_callback(self, message) -> None:
        self._append_pose("mocap_b", message)

    def _uwb_callback(self, message) -> None:
        receive_ros = float(self.rospy.Time.now().to_sec())
        with self.lock:
            enqueued = self._mark_received("uwb", receive_ros)
        try:
            extraction = self._uwb_adapter(
                message,
                source_id=int(self.topics["uwb"]["source_id"]),
                peer_id=int(self.topics["uwb"]["peer_id"]),
                receive_timestamp=receive_ros,
                timestamp_source=self.topics["uwb"]["timestamp_source"],
            )
        except (RosAdapterError, ValueError) as exc:
            with self.lock:
                self.invalid_messages["uwb:" + str(exc)] += 1
            return
        with self.lock:
            if not extraction.accepted:
                self.uwb_extraction_reasons[extraction.invalid_reason] += 1
                return
            self.valid_messages["uwb_target_pair"] += 1
            self.valid_receive_monotonic["uwb"].append(enqueued)
            if (
                self.recording_active
                and self.recording_start_ros_timestamp is not None
                and extraction.observation.timestamp >= self.recording_start_ros_timestamp
            ):
                self.pending.append(PendingRange(extraction.observation, enqueued))

    @staticmethod
    def _rate(times) -> Optional[float]:
        if len(times) < 2:
            return None
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0.0 else None

    def health_report(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            streams = {}
            for role in ("uwb", "mocap_a", "mocap_b"):
                receives = list(self.receive_monotonic[role])
                valid_receives = list(self.valid_receive_monotonic[role])
                streams[role] = {
                    "received_count": len(receives),
                    "valid_count": (
                        self.valid_messages["uwb_target_pair"]
                        if role == "uwb"
                        else self.valid_messages[role]
                    ),
                    "observed_receive_rate_hz": self._rate(receives),
                    "valid_observed_receive_rate_hz": self._rate(valid_receives),
                    "configured_expected_rate_hz": self.topics[role].get("expected_rate_hz"),
                    "latest_receive_timestamp": self.receive_ros[role],
                    "silence_s": (now - receives[-1]) if receives else None,
                    "valid_silence_s": (now - valid_receives[-1]) if valid_receives else None,
                }
            return {
                "streams": streams,
                "invalid_message_counts": dict(sorted(self.invalid_messages.items())),
                "uwb_non_target_or_missing_counts": dict(sorted(self.uwb_extraction_reasons.items())),
            }

    def _healthy(self) -> bool:
        now = time.monotonic()
        with self.lock:
            if len(self.pose_a) < 2 or len(self.pose_b) < 2 or self.valid_messages["uwb_target_pair"] < 1:
                return False
            return all(
                self.valid_receive_monotonic[role]
                and now - self.valid_receive_monotonic[role][-1] <= self.stream_timeout_s
                for role in ("uwb", "mocap_a", "mocap_b")
            )

    def wait_for_health(self) -> None:
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline and not self.stop_requested.is_set() and not self.rospy.is_shutdown():
            if self._healthy():
                print(json.dumps(self.health_report(), ensure_ascii=False, indent=2, sort_keys=True))
                return
            time.sleep(0.05)
        raise CaptureError(
            "three-stream health check did not pass within %.3f s: %s"
            % (self.health_timeout_s, json.dumps(self.health_report(), ensure_ascii=False, sort_keys=True))
        )

    def _event_payload(self, state: str, reason: str) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "environment_id": self.environment_id,
            "pose_id": self.pose_id,
            "attempt_id": self.paths.attempt_id,
            "state": state,
            "ros_timestamp": float(self.rospy.Time.now().to_sec()),
            "timestamp_utc": _utc_now(),
            "valid_ranges": self.summary.valid_rows,
            "target_valid_ranges": self.target_valid_ranges,
            "reason": reason,
            "operator_note": self.operator_note,
        }

    def _publish_payload(self, payload: Dict[str, Any]) -> None:
        if self._status_publisher is None or self._status_class is None:
            return
        message = self._status_class()
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._status_publisher.publish(message)
        self.last_status_monotonic = time.monotonic()

    def transition(self, state: str, reason: str = "") -> Optional[Dict[str, Any]]:
        if state not in CAPTURE_STATES:
            raise CaptureError("unknown capture state: %s" % state)
        if self.state == state and self.state_reason == reason:
            return None
        self.state = state
        self.state_reason = reason
        payload = self._event_payload(state, reason)
        _append_jsonl(self.paths.events_jsonl, payload)
        append_event(
            self.session_dir,
            "POSE_CAPTURE_STATE",
            session_id=self.session_id,
            environment_id=self.environment_id,
            pose_id=self.pose_id,
            attempt_id=self.paths.attempt_id,
            state=state,
            valid_ranges=self.summary.valid_rows,
            reason=reason,
        )
        self._publish_payload(payload)
        print(
            "[%s] valid %d/%d%s"
            % (state, self.summary.valid_rows, self.target_valid_ranges, (" - " + reason) if reason else "")
        )
        return payload

    def publish_periodic_status(self) -> None:
        if self.state is None:
            return
        if time.monotonic() - self.last_status_monotonic >= self.status_period_s:
            self._publish_payload(self._event_payload(self.state, self.state_reason))

    def _streams_stale(self) -> Optional[str]:
        now = time.monotonic()
        with self.lock:
            stale = [
                role
                for role in ("uwb", "mocap_a", "mocap_b")
                if not self.valid_receive_monotonic[role]
                or now - self.valid_receive_monotonic[role][-1] > self.stream_timeout_s
            ]
        return ",".join(stale) if stale else None

    def _pop_processable(self) -> Optional[PendingRange]:
        with self.lock:
            if not self.pending:
                return None
            pending = self.pending[0]
            bracket_ready = (
                self.pose_a
                and self.pose_b
                and self.pose_a[-1].timestamp >= pending.observation.timestamp
                and self.pose_b[-1].timestamp >= pending.observation.timestamp
            )
            wait_expired = time.monotonic() - pending.enqueued_monotonic >= self.stream_timeout_s
            if not bracket_ready and not wait_expired:
                return None
            return self.pending.popleft()

    def _update_state_for_stability(self, timestamp: float, is_stable: bool, reason: str) -> None:
        if self.state == "SETTLING":
            if self.confirmed_uwb_time is None:
                self.confirmed_uwb_time = timestamp
            elapsed = timestamp - self.confirmed_uwb_time
            required_history_s = max(
                self.settling_time_s, self.thresholds.stability_window_s
            )
            if elapsed >= required_history_s:
                if is_stable:
                    self.stable_since_uwb_time = timestamp
                    self.transition("COLLECTING")
                else:
                    self.stable_since_uwb_time = None
                    self.transition("PAUSED", reason or "not_stable_after_settling")
        elif self.state == "COLLECTING":
            if not is_stable:
                self.stable_since_uwb_time = None
                self.transition("PAUSED", reason or "movement_detected")
        elif self.state == "PAUSED":
            if not is_stable:
                self.stable_since_uwb_time = None
                if reason and self.state_reason != reason:
                    self.transition("PAUSED", reason)
            else:
                if self.stable_since_uwb_time is None:
                    self.stable_since_uwb_time = timestamp
                if timestamp - self.stable_since_uwb_time >= self.settling_time_s:
                    self.transition("COLLECTING", "stability_restored")

    def _process_one(
        self,
        pending: PendingRange,
        forced_invalid_reason: str = "",
        update_state: bool = True,
    ) -> None:
        with self.lock:
            samples_a = list(self.pose_a)
            samples_b = list(self.pose_b)
        stability = assess_static_window(
            samples_a, samples_b, pending.observation.timestamp, self.thresholds
        )
        # A rosbag records this UWB envelope before any state transition that
        # the envelope itself triggers.  Use the pre-transition state for this
        # row, then publish the transition for subsequent measurements so live
        # CSV and bag replay have identical segment boundaries.
        state_for_row = (
            self.state
            if self.state in ("SETTLING", "COLLECTING", "PAUSED")
            else "PAUSED"
        )
        invalid_reasons = []
        if (
            self.last_processed_uwb_timestamp is not None
            and pending.observation.timestamp <= self.last_processed_uwb_timestamp
        ):
            invalid_reasons.append("uwb_timestamp_non_increasing")
            self.stable_since_uwb_time = None
        if forced_invalid_reason:
            invalid_reasons.append(str(forced_invalid_reason))
        self.last_processed_uwb_timestamp = pending.observation.timestamp
        self.sample_index += 1
        processed = process_uwb_observation(
            pending.observation,
            samples_a,
            samples_b,
            self.thresholds,
            session_id=self.session_id,
            environment_id=self.environment_id,
            pose_id=self.pose_id,
            attempt_id=self.paths.attempt_id,
            sample_index=self.sample_index,
            capture_state=state_for_row,
            mocap_a_frame_id=self.topics["mocap_a"]["expected_frame_id"],
            mocap_b_frame_id=self.topics["mocap_b"]["expected_frame_id"],
            is_los=True,
            data_origin=self.data_origin,
            is_synthetic=self.is_synthetic,
            ingestion_path="ros_live",
            operator_note=self.operator_note,
            upstream_invalid_reason=";".join(invalid_reasons),
        )
        self._writer.write(processed.row)
        self.summary.add(processed)
        if update_state and "uwb_timestamp_non_increasing" in invalid_reasons:
            self.transition("PAUSED", "uwb_timestamp_non_increasing")
        elif update_state:
            self._update_state_for_stability(
                pending.observation.timestamp,
                stability.is_stable,
                stability.invalid_reason,
            )

    def request_stop(self, *_unused) -> None:
        self.stop_requested.set()

    def _write_summary(self, final_state: str, reason: str) -> Dict[str, Any]:
        if self._summary_written:
            return _read_json(self.paths.summary_json)
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "environment_id": self.environment_id,
            "pose_id": self.pose_id,
            "attempt_id": self.paths.attempt_id,
            "final_state": final_state,
            "reason": reason,
            "target_valid_ranges": self.target_valid_ranges,
            "finished_utc": _utc_now(),
            "raw_csv": str(self.paths.raw_csv),
            "raw_csv_size_bytes": self.paths.raw_csv.stat().st_size,
            "raw_csv_sha256": _sha256(self.paths.raw_csv),
            "operator_note": self.operator_note,
            "is_los": True,
            "data_origin": self.data_origin,
            "is_synthetic": self.is_synthetic,
            "ingestion_path": "ros_live",
            "training_eligible": False,
            "stream_health": self.health_report(),
            **self.summary.payload(),
        }
        write_json_exclusive(self.paths.summary_json, payload)
        self._summary_written = True
        return payload

    def finish(self, final_state: str, reason: str = "") -> Dict[str, Any]:
        if self._summary_written:
            return _read_json(self.paths.summary_json)
        with self.lock:
            self.recording_active = False
            remaining = list(self.pending)
            self.pending.clear()
        for pending in remaining:
            self._process_one(
                pending,
                forced_invalid_reason="capture_finalization_pending",
                update_state=False,
            )
        self._writer.close()
        payload = self._write_summary(final_state, reason)
        if final_state == "COMPLETE":
            write_json_exclusive(
                self.paths.complete_marker,
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "pose_id": self.pose_id,
                    "attempt_id": self.paths.attempt_id,
                    "completed_utc": _utc_now(),
                    "valid_range_count": self.summary.valid_rows,
                    "summary_json": str(self.paths.summary_json),
                },
            )
        self.transition(final_state, reason)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload

    def abort_after_error(self, reason: str) -> None:
        try:
            self.finish("ABORTED", reason)
        except Exception as final_exc:
            print("failed to finalize aborted attempt: %s" % final_exc, file=sys.stderr)

    def _drain_processable_until_idle_or_stop(self):
        """Drain queued ranges without starving signal-driven shutdown."""

        processed_any = False
        while not self.stop_requested.is_set() and not self.rospy.is_shutdown():
            if (
                self.capture_deadline_monotonic is not None
                and time.monotonic() >= self.capture_deadline_monotonic
            ):
                break
            pending = self._pop_processable()
            if pending is None:
                break
            processed_any = True
            self._process_one(pending)
            if self.summary.valid_rows >= self.target_valid_ranges:
                self.finish("COMPLETE", "target_valid_ranges_reached")
                return processed_any, True
        return processed_any, False

    def run(self, assume_yes: bool, skip: bool) -> bool:
        self.wait_for_health()
        if skip:
            self.finish("ABORTED", "operator_skipped")
            return False
        if not assume_yes:
            response = input(
                "Place the devices at pose %s. Press Enter to begin, or type 'skip': "
                % self.pose_id
            ).strip().lower()
            if response == "skip":
                self.finish("ABORTED", "operator_skipped")
                return False
            if response:
                raise CaptureError("unrecognized response; use Enter or 'skip'")

        now_ros = float(self.rospy.Time.now().to_sec())
        self.confirmed_uwb_time = now_ros
        # Only post-confirmation mocap samples may establish staticity for this
        # pose.  Raw stream health counters remain intact for the audit report.
        with self.lock:
            self.pose_a.clear()
            self.pose_b.clear()
            self.pending.clear()
        with self.lock:
            payload = self.transition("SETTLING", "operator_confirmed_pose")
            if payload is None:
                raise CaptureError("failed to establish a unique SETTLING boundary")
            self.recording_start_ros_timestamp = float(payload["ros_timestamp"])
            self.recording_active = True
        started = time.monotonic()
        self.capture_deadline_monotonic = (
            started + self.max_duration_s
            if self.max_duration_s is not None
            else None
        )
        while not self.stop_requested.is_set() and not self.rospy.is_shutdown():
            with self.lock:
                stream_fault = self.stream_faults.popleft() if self.stream_faults else None
            if stream_fault and self.state not in ("COMPLETE", "ABORTED"):
                self.stable_since_uwb_time = None
                self.transition("PAUSED", "stream_fault:" + stream_fault)
            stale = self._streams_stale()
            if stale and self.state not in ("PAUSED", "COMPLETE", "ABORTED"):
                self.stable_since_uwb_time = None
                self.transition("PAUSED", "stream_timeout:" + stale)
            processed_any, completed = self._drain_processable_until_idle_or_stop()
            if completed:
                return True
            self.publish_periodic_status()
            if self.max_duration_s is not None and time.monotonic() - started >= self.max_duration_s:
                self.finish("ABORTED", "max_duration_reached")
                return False
            if not processed_any:
                time.sleep(0.005)
        self.finish("ABORTED", "operator_interrupt_or_ros_shutdown")
        return False

    def close(self) -> None:
        self._writer.close()
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception:
                pass


def _ros_preflight(rospy, rosgraph, collection) -> None:
    topics = collection["ros_topics"]
    master = rosgraph.Master(rospy.get_name())
    published = dict(master.getPublishedTopics("/"))
    failures = []
    for role in ("uwb", "mocap_a", "mocap_b"):
        topic = topics[role]["topic"]
        configured_type = topics[role]["message_type"]
        actual_type = published.get(topic)
        if actual_type is None:
            failures.append("%s topic is not published: %s" % (role, topic))
        elif actual_type != configured_type:
            failures.append(
                "%s type mismatch on %s: configured %s, published %s"
                % (role, topic, configured_type, actual_type)
            )
    status_topic = topics["status"]["topic"]
    status_actual_type = published.get(status_topic)
    if status_actual_type is not None and status_actual_type != topics["status"]["message_type"]:
        failures.append(
            "status type mismatch on %s: configured %s, published %s"
            % (status_topic, topics["status"]["message_type"], status_actual_type)
        )
    if topics.get("time", {}).get("require_use_sim_time_false", True):
        if bool(rospy.get_param("/use_sim_time", False)):
            failures.append("/use_sim_time is true")
    if failures:
        raise CaptureError("ROS preflight failed: " + "; ".join(failures))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "collection.yaml"),
        help="Collection YAML; its ros_topics_config supplies every ROS interface value.",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--pose-id", required=True)
    parser.add_argument("--target-valid-ranges", type=int)
    parser.add_argument("--session-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--note", default="")
    parser.add_argument("--health-timeout-s", type=float, default=15.0)
    parser.add_argument("--max-duration-s", type=float)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip only the positioning prompt; all health/stability checks remain active.",
    )
    parser.add_argument("--skip", action="store_true", help="Record this attempt as operator-skipped.")
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Explicitly allow a session whose immutable metadata declares synthetic_test.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    runtime = None
    try:
        session_id = _validate_id(args.session_id, "session_id")
        pose_id = _validate_id(args.pose_id, "pose_id")
        if not math_is_positive(args.health_timeout_s):
            raise CaptureError("health_timeout_s must be finite and positive")
        if args.max_duration_s is not None and not math_is_positive(args.max_duration_s):
            raise CaptureError("max_duration_s must be finite and positive when provided")
        collection = load_collection_config(args.config, live=True)
        validate_capture_interfaces(collection)
        CaptureThresholds.from_collection(collection)
        if not math_is_positive(collection.get("stream_timeout_s")):
            raise CaptureError("stream_timeout_s must be finite and positive")
        if not math_is_positive(collection.get("status_publish_rate_hz")):
            raise CaptureError("status_publish_rate_hz must be finite and positive")
        settling_time_s = float(collection["settling_time_s"])
        if not (settling_time_s >= 0.0 and settling_time_s < float("inf")):
            raise CaptureError("settling_time_s must be finite and non-negative")
        float_precision = int(collection["output"].get("float_precision", 17))
        if float_precision < 6 or float_precision > 17:
            raise CaptureError("output.float_precision must be within [6, 17]")
        target = (
            int(args.target_valid_ranges)
            if args.target_valid_ranges is not None
            else int(collection["target_valid_ranges"])
        )
        if target <= 0:
            raise CaptureError("target-valid-ranges must be positive")
        session_dir, session_manifest = _resolve_session_dir(
            collection, session_id, args.session_dir, args.output_root
        )
        environment_id = str(session_manifest["metadata"]["environment_id"])
        if bool(session_manifest["metadata"]["is_synthetic"]) and not args.allow_synthetic:
            raise CaptureError("synthetic session requires explicit --allow-synthetic")

        try:
            import rosgraph
            import rospy
        except ImportError as exc:
            raise CaptureError("ROS Python environment is not sourced: %s" % exc)
        rospy.init_node("uwb_bias_static_pose_capture", anonymous=True, disable_signals=True)
        _ros_preflight(rospy, rosgraph, collection)
        paths = _prepare_attempt(session_dir, pose_id)
        print("capture attempt output: %s" % paths.attempt_dir)
        runtime = LiveCapture(
            rospy=rospy,
            collection=collection,
            session_dir=session_dir,
            session_id=session_id,
            environment_id=environment_id,
            pose_id=pose_id,
            paths=paths,
            target_valid_ranges=target,
            operator_note=args.note,
            health_timeout_s=args.health_timeout_s,
            max_duration_s=args.max_duration_s,
        )
        runtime.install_ros_interfaces()
        signal.signal(signal.SIGINT, runtime.request_stop)
        signal.signal(signal.SIGTERM, runtime.request_stop)
        completed = runtime.run(args.yes, args.skip)
        return 0 if completed else 4
    except (CaptureError, ConfigError, RosAdapterError, FileExistsError, OSError, ValueError) as exc:
        print("static pose capture failed: %s" % exc, file=sys.stderr)
        if runtime is not None and not runtime._summary_written:
            runtime.abort_after_error("error:" + str(exc))
        return 2
    except KeyboardInterrupt:
        if runtime is not None and not runtime._summary_written:
            runtime.abort_after_error("operator_keyboard_interrupt")
        return 130
    except Exception as exc:
        print("unexpected static pose capture error: %s" % exc, file=sys.stderr)
        if runtime is not None and not runtime._summary_written:
            runtime.abort_after_error("unexpected_error:" + str(exc))
        return 3
    finally:
        if runtime is not None:
            runtime.close()


def math_is_positive(value: Any) -> bool:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return converted > 0.0 and converted < float("inf")


if __name__ == "__main__":
    sys.exit(main())
