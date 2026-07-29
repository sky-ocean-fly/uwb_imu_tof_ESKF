import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.session import prepare_session  # noqa: E402


class SessionTests(unittest.TestCase):
    def _write_fixture(self, root):
        topics = {
            "uwb": {
                "topic": "/confirmed_uwb",
                "message_type": "nlink_parser/LinktrackNodeframe2",
                "adapter": "nodeframe2",
                "source_id": 10,
                "peer_id": 11,
                "direction": "source_to_peer",
                "timestamp_source": "receive_time",
            },
            "mocap_a": {
                "topic": "/confirmed_a",
                "message_type": "geometry_msgs/PoseStamped",
            },
            "mocap_b": {
                "topic": "/confirmed_b",
                "message_type": "geometry_msgs/PoseStamped",
            },
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
        metadata = {
            "data_origin": "synthetic_test",
            "is_synthetic": True,
            "environment_id": "synthetic_test_environment",
            "location": "temporary_directory",
            "operator": "unit_test",
            "los": True,
            "uwb": {
                "model": "test_only",
                "firmware": "test_only",
                "ranging_mode": "test_only",
            },
            "mocap": {
                "rigid_body_a": "test_a",
                "rigid_body_b": "test_b",
                "antenna_mount_description": "test-only unit extrinsics",
            },
        }
        (root / "topics.yaml").write_text(
            yaml.safe_dump(topics), encoding="utf-8"
        )
        (root / "collection.yaml").write_text(
            yaml.safe_dump(collection), encoding="utf-8"
        )
        (root / "metadata.yaml").write_text(
            yaml.safe_dump(metadata), encoding="utf-8"
        )

    def test_session_is_snapshotted_and_never_reused(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            root = Path(directory)
            self._write_fixture(root)
            result = prepare_session(
                root / "collection.yaml",
                root / "metadata.yaml",
                "synthetic_session",
                output_root=root / "raw",
            )
            session_dir = Path(result["session_dir"])
            self.assertTrue((session_dir / "config_snapshot" / "collection.yaml").is_file())
            manifest = json.loads(
                (session_dir / "session_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["session_id"], "synthetic_session")
            self.assertEqual(manifest["uwb_pair"]["source_id"], 10)
            self.assertEqual(manifest["metadata"]["data_origin"], "synthetic_test")
            self.assertTrue(manifest["metadata"]["is_synthetic"])
            self.assertTrue(manifest["metadata_snapshot_sha256"])
            self.assertTrue((session_dir / "session_events.jsonl").is_file())

            with self.assertRaises(FileExistsError):
                prepare_session(
                    root / "collection.yaml",
                    root / "metadata.yaml",
                    "synthetic_session",
                    output_root=root / "raw",
                )


if __name__ == "__main__":
    unittest.main()
