import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "list_pose_status.py"
SPEC = importlib.util.spec_from_file_location("list_pose_status", str(SCRIPT_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PoseStatusTests(unittest.TestCase):
    def test_completed_attempted_pending_and_unplanned_are_distinct(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            root = Path(directory)
            plan = root / "plan.csv"
            with plan.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["pose_id"])
                writer.writeheader()
                writer.writerows([{"pose_id": "P1"}, {"pose_id": "P2"}, {"pose_id": "P3"}])
            session = root / "session"
            session.mkdir()
            (session / "session_manifest.json").write_text(
                json.dumps({"session_id": "synthetic_session"}), encoding="utf-8"
            )

            complete_root = session / "poses" / "P1"
            (complete_root / "attempts" / "A1").mkdir(parents=True)
            (complete_root / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "attempt_id": "A1",
                        "valid_range_count": 200,
                    }
                ),
                encoding="utf-8",
            )
            aborted = session / "poses" / "P2" / "attempts" / "A2"
            aborted.mkdir(parents=True)
            (aborted / "summary.json").write_text(
                json.dumps(
                    {
                        "attempt_id": "A2",
                        "final_state": "ABORTED",
                        "valid_range_count": 12,
                    }
                ),
                encoding="utf-8",
            )
            (session / "poses" / "UNPLANNED").mkdir(parents=True)

            payload = MODULE.collect_status(plan, session)
            states = {row["pose_id"]: row["state"] for row in payload["poses"]}
            self.assertEqual(states, {"P1": "COMPLETE", "P2": "ABORTED", "P3": "PENDING"})
            self.assertEqual(payload["unplanned_pose_directories"], ["UNPLANNED"])


if __name__ == "__main__":
    unittest.main()
