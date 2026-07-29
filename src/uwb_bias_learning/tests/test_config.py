import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.config import (  # noqa: E402
    ConfigError,
    load_collection_config,
    load_yaml,
    validate_ros_topics_config,
    write_json_exclusive,
)


class ConfigTest(unittest.TestCase):
    def test_unresolved_live_values_fail(self):
        config = load_yaml(PACKAGE_ROOT / "configs" / "ros_topics.yaml")
        config["uwb"]["source_id"] = None
        with self.assertRaises(ConfigError):
            validate_ros_topics_config(config, live=True)

    def test_confirmed_project_live_values_pass(self):
        config = load_yaml(PACKAGE_ROOT / "configs" / "ros_topics.yaml")
        validate_ros_topics_config(config, live=True)

    def test_exclusive_json_never_overwrites(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            path = Path(directory) / "manifest.json"
            write_json_exclusive(path, {"kind": "synthetic_test"})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(path, {"kind": "replacement"})

    def test_complete_collection_loads_relative_topics(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            root = Path(directory)
            topics = {
                "uwb": {
                    "topic": "/u",
                    "message_type": "nlink_parser/LinktrackNodeframe2",
                    "adapter": "nodeframe2",
                    "source_id": 10,
                    "peer_id": 11,
                    "direction": "source_to_peer",
                    "timestamp_source": "receive_time",
                },
                "mocap_a": {"topic": "/a", "message_type": "geometry_msgs/PoseStamped"},
                "mocap_b": {"topic": "/b", "message_type": "geometry_msgs/PoseStamped"},
                "status": {
                    "topic": "/uwb_bias_learning/capture_state",
                    "message_type": "std_msgs/String",
                },
                "time": {
                    "require_use_sim_time_false": True,
                    "require_nonzero_mocap_header_stamp": True,
                    "require_same_mocap_frame": True,
                },
                "extrinsics": {
                    "rigid_body_origins_at_antenna_centers": True,
                    "rigid_body_axes_aligned_with_antennas": True,
                },
            }
            (root / "topics.yaml").write_text(yaml.safe_dump(topics), encoding="utf-8")
            collection = {
                "ros_topics_config": "topics.yaml",
                "target_valid_ranges": 200,
                "min_valid_ranges": 180,
                "settling_time_s": 1.5,
                "stability_window_s": 1.0,
                "max_position_std_m": 0.005,
                "max_rotation_std_deg": 0.5,
                "max_sync_gap_s": 0.02,
                "max_mocap_sample_gap_s": 0.05,
                "output": {"raw_root": "raw"},
            }
            (root / "collection.yaml").write_text(
                yaml.safe_dump(collection), encoding="utf-8"
            )
            loaded = load_collection_config(root / "collection.yaml", live=True)
            self.assertEqual(loaded["ros_topics"]["uwb"]["source_id"], 10)


if __name__ == "__main__":
    unittest.main()
