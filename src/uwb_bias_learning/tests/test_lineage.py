import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.lineage import (  # noqa: E402
    LineageError,
    SplitContract,
    file_sha256,
    require_lineage_match,
)


class LineageTests(unittest.TestCase):
    def _fixture(self, root, provenance="real"):
        files = {}
        for split, pose in (("train", "P1"), ("val", "P2"), ("test", "P3")):
            path = root / (split + ".csv")
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["session_id", "pose_id", "value"])
                writer.writeheader()
                writer.writerow({"session_id": "S1", "pose_id": pose, "value": split})
            files[split] = path
        normalization = root / "normalization.json"
        normalization.write_text("{}\n", encoding="utf-8")
        files["normalization"] = normalization
        payload = {
            "data_provenance": provenance,
            "splits": {
                split: {"pose_keys": [{"session_id": "S1", "pose_id": pose}]}
                for split, pose in (("train", "P1"), ("val", "P2"), ("test", "P3"))
            },
            "artifacts": {
                name: {"file": path.name, "sha256": file_sha256(path)}
                for name, path in files.items()
            },
        }
        manifest = root / "splits.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        return manifest, files

    def test_hash_and_pose_keys_bind_each_split(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            root = Path(directory)
            manifest, files = self._fixture(root)
            contract = SplitContract.load(manifest)
            contract.verify_pose_csv("train", files["train"])
            files["train"].write_text(
                "session_id,pose_id,value\nS1,P1,replaced\n", encoding="utf-8"
            )
            with self.assertRaises(LineageError):
                contract.verify_pose_csv("train", files["train"])

    def test_synthetic_requires_explicit_software_test_opt_in(self):
        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            manifest, _ = self._fixture(Path(directory), provenance="synthetic")
            with self.assertRaises(LineageError):
                SplitContract.load(manifest)
            self.assertEqual(
                SplitContract.load(manifest, allow_synthetic=True).data_provenance,
                "synthetic",
            )

    def test_lineage_mismatch_is_not_silently_accepted(self):
        with self.assertRaises(LineageError):
            require_lineage_match(
                {"split_manifest_sha256": "a"},
                {"split_manifest_sha256": "b"},
                ("split_manifest_sha256",),
                "model",
            )


if __name__ == "__main__":
    unittest.main()
