import csv
import importlib.util
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.capture import (  # noqa: E402
    CaptureError,
    CaptureThresholds,
    ExclusiveRawCsvWriter,
    StatusAssignmentTracker,
    assess_static_window,
    process_uwb_observation,
    validate_capture_interfaces,
)
from uwb_bias_learning.datasets import AggregationConfig, validate_raw_rows  # noqa: E402
from uwb_bias_learning.ros_adapters import UwbObservation  # noqa: E402
from uwb_bias_learning.synchronization import TimedPose  # noqa: E402


CAPTURE_SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "capture_static_pose.py"
CAPTURE_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "capture_static_pose_test_module", str(CAPTURE_SCRIPT_PATH)
)
CAPTURE_SCRIPT_MODULE = importlib.util.module_from_spec(CAPTURE_SCRIPT_SPEC)
sys.modules[CAPTURE_SCRIPT_SPEC.name] = CAPTURE_SCRIPT_MODULE
CAPTURE_SCRIPT_SPEC.loader.exec_module(CAPTURE_SCRIPT_MODULE)


def thresholds(position=0.005):
    return CaptureThresholds(
        stability_window_s=1.0,
        max_position_std_m=position,
        max_rotation_std_deg=0.5,
        max_sync_gap_s=0.051,
        max_mocap_sample_gap_s=0.11,
        range_min_m=None,
        range_max_m=None,
        reject_nonpositive=True,
    )


def static_stream(offset):
    result = []
    for index in range(21):
        timestamp = index * 0.1
        result.append(
            TimedPose(
                timestamp,
                np.asarray(offset, dtype=float),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            )
        )
    return result


def observation(timestamp=1.55, measured=2.1):
    return UwbObservation(
        receive_timestamp=timestamp,
        source_id=4,
        peer_id=9,
        range_m=measured,
        source_role=0,
        peer_role=1,
        local_time_raw=123,
        system_time_raw=456,
        fp_rssi=-80.0,
        rx_rssi=-77.0,
        timestamp=timestamp,
        timestamp_source="receive_time",
    )


def process(is_los=True, upstream_invalid_reason=""):
    return process_uwb_observation(
        observation(),
        static_stream([0.0, 0.0, 0.0]),
        static_stream([2.0, 0.0, 0.0]),
        thresholds(),
        session_id="synthetic_session",
        environment_id="synthetic_environment",
        pose_id="synthetic_pose",
        attempt_id="synthetic_attempt",
        sample_index=1,
        capture_state="COLLECTING",
        mocap_a_frame_id="synthetic_world",
        mocap_b_frame_id="synthetic_world",
        is_los=is_los,
        data_origin="synthetic_unit_test",
        is_synthetic=True,
        ingestion_path="synthetic_unit_test",
        operator_note="synthetic unit test only",
        upstream_invalid_reason=upstream_invalid_reason,
    )


class CapturePipelineTest(unittest.TestCase):
    def test_pending_drain_honors_signal_stop_even_when_queue_never_idles(self):
        runtime = CAPTURE_SCRIPT_MODULE.LiveCapture.__new__(
            CAPTURE_SCRIPT_MODULE.LiveCapture
        )
        runtime.stop_requested = threading.Event()
        runtime.rospy = SimpleNamespace(is_shutdown=lambda: False)
        runtime.summary = SimpleNamespace(valid_rows=0)
        runtime.target_valid_ranges = 200
        runtime.capture_deadline_monotonic = None
        processed = []
        runtime._pop_processable = lambda: object()

        def process_one(pending):
            processed.append(pending)
            runtime.stop_requested.set()

        runtime._process_one = process_one
        runtime.finish = lambda *_args: self.fail("shutdown must not mark COMPLETE")

        processed_any, completed = runtime._drain_processable_until_idle_or_stop()
        self.assertTrue(processed_any)
        self.assertFalse(completed)
        self.assertEqual(len(processed), 1)

    def test_pending_drain_honors_deadline_even_when_queue_never_idles(self):
        runtime = CAPTURE_SCRIPT_MODULE.LiveCapture.__new__(
            CAPTURE_SCRIPT_MODULE.LiveCapture
        )
        runtime.stop_requested = threading.Event()
        runtime.rospy = SimpleNamespace(is_shutdown=lambda: False)
        runtime.summary = SimpleNamespace(valid_rows=0)
        runtime.target_valid_ranges = 200
        runtime.capture_deadline_monotonic = time.monotonic() - 1.0
        runtime._pop_processable = lambda: self.fail("expired capture must not drain")

        processed_any, completed = runtime._drain_processable_until_idle_or_stop()
        self.assertFalse(processed_any)
        self.assertFalse(completed)

    def test_pose_buffer_retains_history_required_by_oldest_pending_range(self):
        runtime = CAPTURE_SCRIPT_MODULE.LiveCapture.__new__(
            CAPTURE_SCRIPT_MODULE.LiveCapture
        )
        runtime.thresholds = SimpleNamespace(
            stability_window_s=1.0, max_sync_gap_s=0.04
        )
        runtime.stream_timeout_s = 2.0
        runtime.pending = [
            SimpleNamespace(observation=SimpleNamespace(timestamp=100.0))
        ]

        cutoff = runtime._pose_retention_cutoff(110.0)
        self.assertAlmostEqual(cutoff, 98.96)

    def test_synced_row_has_directed_ids_and_signed_bias(self):
        processed = process()
        row = processed.row
        self.assertTrue(processed.is_valid)
        self.assertEqual(row["anchor_id"], "")
        self.assertEqual(row["node_id"], "")
        self.assertEqual(row["uwb_source_id"], 4)
        self.assertEqual(row["uwb_peer_id"], 9)
        self.assertEqual(row["is_mocap_valid"], 1)
        self.assertEqual(row["is_los"], 1)
        self.assertEqual(row["data_origin"], "synthetic_unit_test")
        self.assertEqual(row["is_synthetic"], 1)
        self.assertEqual(row["ingestion_path"], "synthetic_unit_test")
        self.assertEqual(row["environment_id"], "synthetic_environment")
        self.assertAlmostEqual(row["A_px"], 0.0)
        self.assertAlmostEqual(row["B_px"], 2.0)
        self.assertAlmostEqual(row["distance_gt_m"], 2.0)
        self.assertAlmostEqual(row["range_error_m"], 0.1)
        self.assertAlmostEqual(row["mocap_a_interpolation_alpha"], 0.5)

    def test_non_los_is_never_counted_valid(self):
        processed = process(is_los=False)
        self.assertFalse(processed.is_valid)
        self.assertEqual(processed.row["is_los"], 0)
        self.assertIn("not_los", processed.row["invalid_reason"])

    def test_upstream_timestamp_anomaly_is_never_counted_valid(self):
        processed = process(upstream_invalid_reason="uwb_timestamp_non_increasing")
        self.assertFalse(processed.is_valid)
        self.assertIn("uwb_timestamp_non_increasing", processed.row["invalid_reason"])

    def test_relative_motion_is_checked_separately(self):
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        stream_a = []
        stream_b = []
        for index in range(11):
            sign = -1.0 if index % 2 else 1.0
            timestamp = index * 0.1
            stream_a.append(TimedPose(timestamp, [sign * 0.004, 0.0, 0.0], identity))
            stream_b.append(TimedPose(timestamp, [2.0 - sign * 0.004, 0.0, 0.0], identity))
        result = assess_static_window(stream_a, stream_b, 1.0, thresholds())
        self.assertFalse(result.is_stable)
        self.assertIsNotNone(result.mocap_a)
        self.assertIsNotNone(result.mocap_b)
        self.assertLess(result.mocap_a.position_stability_m, 0.005)
        self.assertLess(result.mocap_b.position_stability_m, 0.005)
        self.assertGreater(result.relative.position_stability_m, 0.005)
        self.assertIn("relative:position_unstable", result.invalid_reason)

    def test_static_window_rejects_unobserved_internal_gap(self):
        identity = np.asarray([0.0, 0.0, 0.0, 1.0])
        sparse = [
            TimedPose(0.0, [0.0, 0.0, 0.0], identity),
            TimedPose(1.0, [0.0, 0.0, 0.0], identity),
        ]
        result = assess_static_window(sparse, sparse, 1.0, thresholds())
        self.assertFalse(result.is_stable)
        self.assertIn("stability_window_internal_gap_exceeded", result.invalid_reason)

    def test_mapped_device_time_fails_closed(self):
        collection = {
            "ros_topics": {
                "uwb": {
                    "topic": "/synthetic/uwb",
                    "message_type": "nlink_parser/LinktrackNodeframe2",
                    "adapter": "nodeframe2",
                    "source_id": 4,
                    "peer_id": 9,
                    "direction": "source_to_peer",
                    "timestamp_source": "mapped_device_time",
                },
                "mocap_a": {
                    "topic": "/synthetic/a",
                    "message_type": "geometry_msgs/PoseStamped",
                    "expected_frame_id": "synthetic_world",
                },
                "mocap_b": {
                    "topic": "/synthetic/b",
                    "message_type": "geometry_msgs/PoseStamped",
                    "expected_frame_id": "synthetic_world",
                },
                "status": {
                    "topic": "/synthetic/status",
                    "message_type": "std_msgs/String",
                },
                "time": {"require_same_mocap_frame": True},
            }
        }
        with self.assertRaisesRegex(CaptureError, "no explicit clock mapping"):
            validate_capture_interfaces(collection)

    def test_status_tracker_only_assigns_active_segment(self):
        tracker = StatusAssignmentTracker("synthetic_session")
        self.assertIsNone(tracker.active)

    def test_mismatched_terminal_event_does_not_close_active_segment(self):
        tracker = StatusAssignmentTracker("synthetic_session")
        tracker.feed(
            '{"session_id":"synthetic_session","pose_id":"p1",'
            '"attempt_id":"a1","state":"SETTLING"}'
        )
        tracker.feed(
            '{"session_id":"synthetic_session","pose_id":"p2",'
            '"attempt_id":"a2","state":"COMPLETE"}'
        )
        self.assertEqual(tracker.active.attempt_id, "a1")
        self.assertEqual(tracker.invalid_event_count, 1)
        self.assertEqual(tracker.active.pose_id, "p1")
        tracker.feed(
            '{"session_id":"synthetic_session","pose_id":"p1",'
            '"attempt_id":"a1","state":"COLLECTING"}'
        )
        self.assertEqual(tracker.active.state, "COLLECTING")
        tracker.feed(
            '{"session_id":"synthetic_session","pose_id":"p1",'
            '"attempt_id":"a1","state":"COMPLETE"}'
        )
        self.assertIsNone(tracker.active)

    def test_status_tracker_rejects_terminal_before_start_and_reactivation(self):
        tracker = StatusAssignmentTracker("synthetic_session")
        terminal = {
            "session_id": "synthetic_session",
            "pose_id": "p1",
            "attempt_id": "a1",
            "state": "COMPLETE",
        }
        tracker.feed(terminal)
        tracker.feed({**terminal, "state": "SETTLING"})
        self.assertIsNone(tracker.active)
        self.assertGreaterEqual(tracker.invalid_event_count, 2)

    def test_status_tracker_rejects_attempt_replacement_without_terminal(self):
        tracker = StatusAssignmentTracker("synthetic_session")
        tracker.feed({"session_id": "synthetic_session", "pose_id": "p1", "attempt_id": "a1", "state": "SETTLING"})
        tracker.feed({"session_id": "synthetic_session", "pose_id": "p2", "attempt_id": "a2", "state": "SETTLING"})
        self.assertEqual(tracker.active.attempt_id, "a1")
        self.assertIn("a1", tracker.invalid_attempt_reasons)
        self.assertIn("a2", tracker.invalid_attempt_reasons)

    def test_raw_writer_is_exclusive_and_keeps_required_flags(self):
        with tempfile.TemporaryDirectory(prefix="uwb_capture_synthetic_") as directory:
            path = Path(directory) / "raw.csv"
            with ExclusiveRawCsvWriter(path) as writer:
                writer.write(process().row)
            with path.open("r", encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["is_los"], "1")
            self.assertEqual(row["is_mocap_valid"], "1")
            self.assertEqual(row["is_synthetic"], "1")
            self.assertEqual(row["ingestion_path"], "synthetic_unit_test")
            with self.assertRaises(FileExistsError):
                ExclusiveRawCsvWriter(path)

    def test_valid_capture_row_satisfies_raw_dataset_contract(self):
        result = validate_raw_rows(
            [process().row],
            config=AggregationConfig(min_valid_ranges=1),
            require_complete_raw_schema=True,
            fail_incomplete_poses=True,
        )
        self.assertTrue(result.report["passed"], result.report)
        self.assertEqual(result.report["data_provenance"], "synthetic")


if __name__ == "__main__":
    unittest.main()
