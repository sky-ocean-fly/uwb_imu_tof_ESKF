import importlib.util
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
SCRIPT = PACKAGE_ROOT / "scripts" / "bag_to_raw_dataset.py"
SPEC = importlib.util.spec_from_file_location("synthetic_bag_to_raw_dataset", SCRIPT)
bag_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bag_module
SPEC.loader.exec_module(bag_module)

from uwb_bias_learning.ros_adapters import UwbObservation  # noqa: E402
from uwb_bias_learning.synchronization import TimedPose  # noqa: E402


def synthetic_observation(timestamp):
    return UwbObservation(
        receive_timestamp=timestamp,
        source_id=1,
        peer_id=2,
        range_m=1.1,
        source_role=None,
        peer_role=None,
        local_time_raw=None,
        system_time_raw=None,
        fp_rssi=None,
        rx_rssi=None,
        timestamp=timestamp,
        timestamp_source="receive_time",
    )


class BagReconstructionTest(unittest.TestCase):
    def test_timeline_uses_status_state_and_records_terminal_outcome(self):
        timeline = [
            bag_module.TimelineItem(
                1.0,
                1,
                "status",
                '{"session_id":"s","pose_id":"p","attempt_id":"a",'
                '"state":"SETTLING"}',
            ),
            bag_module.TimelineItem(
                1.05,
                2,
                "status",
                '{"session_id":"s","pose_id":"p","attempt_id":"a",'
                '"state":"COLLECTING"}',
            ),
            bag_module.TimelineItem(1.1, 3, "uwb", synthetic_observation(1.1)),
            bag_module.TimelineItem(
                1.2,
                4,
                "status",
                '{"session_id":"s","pose_id":"p","attempt_id":"a",'
                '"state":"COMPLETE","target_valid_ranges":1}',
            ),
            bag_module.TimelineItem(1.3, 5, "uwb", synthetic_observation(1.3)),
        ]
        assigned, tracker, outcomes, unassigned = bag_module._assign_timeline(timeline, "s")
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0].assignment.state, "COLLECTING")
        self.assertEqual(outcomes["a"]["state"], "COMPLETE")
        self.assertEqual(unassigned, 1)
        self.assertIsNone(tracker.active)

    def test_non_increasing_header_timestamps_match_live_first_sample_policy(self):
        identity = [0.0, 0.0, 0.0, 1.0]
        samples = [
            TimedPose(1.0, [0.0, 0.0, 0.0], identity),
            TimedPose(1.0, [1.0, 0.0, 0.0], identity),
            TimedPose(0.5, [1.0, 0.0, 0.0], identity),
            TimedPose(2.0, [2.0, 0.0, 0.0], identity),
        ]
        normalized, duplicates, reversed_count = bag_module._normalize_pose_samples(samples)
        self.assertEqual(duplicates, 1)
        self.assertEqual(reversed_count, 1)
        self.assertEqual([sample.timestamp for sample in normalized], [1.0, 2.0])
        self.assertAlmostEqual(normalized[0].position[0], 0.0)


if __name__ == "__main__":
    unittest.main()
