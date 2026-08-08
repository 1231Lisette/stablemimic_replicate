from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stablemimic.motion.lafan1 import (
    LAFAN1_G1_JOINT_NAMES,
    discover_motion_libraries,
    load_lafan1_csv,
)


def valid_rows() -> np.ndarray:
    rows = np.zeros((2, 36), dtype=np.float64)
    rows[:, 2] = 0.8
    rows[:, 6] = 1.0
    return rows


class LAFAN1LoaderTests(unittest.TestCase):
    def test_load_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dance_test.csv"
            np.savetxt(path, valid_rows(), delimiter=",")
            motion = load_lafan1_csv(path)
            self.assertEqual(motion.num_frames, 2)
            self.assertEqual(motion.fps, 30.0)
            self.assertEqual(motion.joint_names, LAFAN1_G1_JOINT_NAMES)

    def test_reject_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dance_bad.csv"
            np.savetxt(path, np.zeros((2, 35)), delimiter=",")
            with self.assertRaisesRegex(ValueError, "Expected 36 columns"):
                load_lafan1_csv(path)

    def test_discover_sequence_disjoint_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "dance1.csv", valid_rows(), delimiter=",")
            np.savetxt(root / "fallAndGetUp1.csv", valid_rows(), delimiter=",")
            np.savetxt(root / "walk1.csv", valid_rows(), delimiter=",")
            libraries = discover_motion_libraries(root)
            self.assertEqual([path.name for path in libraries.tracking], ["dance1.csv"])
            self.assertEqual([path.name for path in libraries.recovery], ["fallAndGetUp1.csv"])

    def test_real_dataset_when_configured(self) -> None:
        data_root = os.environ.get("LAFAN1_G1_ROOT")
        if not data_root:
            self.skipTest("LAFAN1_G1_ROOT is not configured")
        libraries = discover_motion_libraries(data_root)
        self.assertEqual(len(libraries.tracking), 8)
        self.assertEqual(len(libraries.recovery), 6)
        tracking_frames = sum(load_lafan1_csv(path).num_frames for path in libraries.tracking)
        recovery_frames = sum(load_lafan1_csv(path).num_frames for path in libraries.recovery)
        self.assertEqual(tracking_frames, 45_690)
        self.assertEqual(recovery_frames, 28_043)


if __name__ == "__main__":
    unittest.main()
