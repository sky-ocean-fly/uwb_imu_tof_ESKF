import importlib.util
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "generate_pose_plan.py"
SPEC = importlib.util.spec_from_file_location("generate_pose_plan", str(SCRIPT_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PosePlanTests(unittest.TestCase):
    def _domain(self):
        return {
            "distance_m": {"min": 1.0, "max": 3.0},
            "azimuth_deg": {"min": -60.0, "max": 60.0},
            "elevation_deg": {"min": -20.0, "max": 20.0},
            "relative_roll_deg": {"min": -30.0, "max": 30.0},
            "relative_pitch_deg": {"min": -40.0, "max": 40.0},
            "relative_yaw_deg": {"min": -90.0, "max": 90.0},
            "planner": {
                "method": "latin_hypercube",
                "seed": 17,
                "interior_count": 8,
                "include_center": True,
                "include_axis_boundaries": True,
                "duplicate_pose_count": 2,
            },
        }

    def test_plan_is_deterministic_bounded_and_has_explicit_repeats(self):
        first, seed = MODULE.build_plan(self._domain())
        second, _ = MODULE.build_plan(self._domain())
        self.assertEqual(first, second)
        self.assertEqual(seed, 17)
        self.assertEqual(len(first), 8 + 1 + 12 + 2)
        self.assertEqual(sum(row["plan_kind"] == "repeat" for row in first), 2)
        for row in first:
            for axis in MODULE.AXES:
                self.assertGreaterEqual(row[axis], self._domain()[axis]["min"])
                self.assertLessEqual(row[axis], self._domain()[axis]["max"])

    def test_nominal_translation_is_guidance_geometry_only(self):
        x, y, z = MODULE.nominal_translation(2.0, 90.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 2.0, places=12)
        self.assertAlmostEqual(z, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
