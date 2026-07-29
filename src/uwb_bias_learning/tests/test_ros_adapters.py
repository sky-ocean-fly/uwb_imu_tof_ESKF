import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.ros_adapters import (  # noqa: E402
    extract_nodeframe2_pair,
    extract_pose_stamped,
)


class Stamp:
    def __init__(self, value):
        self.value = value

    def to_sec(self):
        return self.value


def pose_message():
    return SimpleNamespace(
        header=SimpleNamespace(seq=7, stamp=Stamp(12.5), frame_id="world"),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=2.0),
        ),
    )


def uwb_message(source=4, peers=(9,)):
    return SimpleNamespace(
        role=0,
        id=source,
        local_time=100,
        system_time=200,
        nodes=[
            SimpleNamespace(
                role=0, id=peer, dis=5.25, fp_rssi=-81.5, rx_rssi=-78.0
            )
            for peer in peers
        ],
    )


class RosAdaptersTest(unittest.TestCase):
    def test_pose_stamped_xyzw_is_normalized(self):
        pose = extract_pose_stamped(pose_message(), receive_timestamp=12.501)
        self.assertEqual(pose.frame_id, "world")
        self.assertEqual(pose.sequence, 7)
        self.assertAlmostEqual(float(pose.quaternion_xyzw[3]), 1.0)

    def test_directed_pair_never_falls_back_to_reverse(self):
        forward = extract_nodeframe2_pair(
            uwb_message(source=4, peers=(9,)), 4, 9, 20.0
        )
        self.assertTrue(forward.accepted)
        self.assertAlmostEqual(forward.observation.range_m, 5.25)
        reverse = extract_nodeframe2_pair(
            uwb_message(source=4, peers=(9,)), 9, 4, 20.0
        )
        self.assertFalse(reverse.accepted)
        self.assertEqual(reverse.invalid_reason, "not_target_source")

    def test_missing_and_duplicate_peer_are_explicit(self):
        missing = extract_nodeframe2_pair(uwb_message(peers=(8,)), 4, 9, 20.0)
        self.assertEqual(missing.invalid_reason, "target_peer_missing")
        duplicate = extract_nodeframe2_pair(
            uwb_message(peers=(9, 9)), 4, 9, 20.0
        )
        self.assertEqual(duplicate.invalid_reason, "duplicate_target_peer")


if __name__ == "__main__":
    unittest.main()
