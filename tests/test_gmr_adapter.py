from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "retarget_lafan1_gmr.py"
SPEC = importlib.util.spec_from_file_location("retarget_lafan1_gmr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GMRAdapterTests(unittest.TestCase):
    def test_qpos_converts_wxyz_to_xyzw_without_losing_frames(self) -> None:
        qpos = np.zeros((2, 36), dtype=np.float64)
        qpos[:, 3:7] = ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        qpos[:, 7:] = np.arange(58).reshape(2, 29)
        rows = MODULE.qpos_to_stablemimic_rows(qpos)
        np.testing.assert_array_equal(rows[:, 3:7], ([2.0, 3.0, 4.0, 1.0], [6.0, 7.0, 8.0, 5.0]))
        np.testing.assert_array_equal(rows[:, 7:], qpos[:, 7:])

    def test_qpos_rejects_wrong_width_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected qpos shape"):
            MODULE.qpos_to_stablemimic_rows(np.zeros((2, 35)))
        bad = np.zeros((2, 36))
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.qpos_to_stablemimic_rows(bad)

    def test_source_discovery_is_sorted_unique_and_pattern_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dance2.bvh", "dance1.bvh", "walk1.bvh"):
                (root / name).touch()
            sources = MODULE.discover_sources(root, ("dance*.bvh", "dance1.bvh"))
            self.assertEqual([path.name for path in sources], ["dance1.bvh", "dance2.bvh"])

    def test_symmetric_rate_limit_bounds_true_ik_jump(self) -> None:
        values = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
        result = MODULE.symmetric_rate_limit(values, max_rate=3.0, fps=10.0)
        self.assertLessEqual(float(np.max(np.abs(np.diff(result[:, 0])))), 0.3 + 1.0e-12)
        self.assertGreater(result[1, 0], 0.0)
        self.assertLess(result[2, 0], 1.0)

    def test_ground_envelope_dominates_required_and_is_rate_limited(self) -> None:
        required = np.array([0.0, 0.0, 0.12, 0.0, 0.0], dtype=np.float64)
        result = MODULE.dominating_rate_limited_envelope(required, max_rate=0.4, fps=10.0)
        self.assertTrue(np.all(result >= required))
        self.assertLessEqual(float(np.max(np.abs(np.diff(result)))), 0.04 + 1.0e-12)
        self.assertEqual(result[2], required[2])


if __name__ == "__main__":
    unittest.main()
