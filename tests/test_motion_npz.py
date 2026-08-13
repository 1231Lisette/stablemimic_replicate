from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES
from stablemimic.motion.npz import load_npz_arrays, resample_motion, validate_standard_npz
from stablemimic.motion.reference import MotionReference


class MotionNpzTests(unittest.TestCase):
    def test_resampling_matches_half_open_beyondmimic_convention(self) -> None:
        frames = 4
        root = np.zeros((frames, 3), dtype=np.float64)
        root[:, 0] = np.arange(frames)
        quaternion = np.zeros((frames, 4), dtype=np.float64)
        quaternion[:, 3] = 1.0
        joints = np.zeros((frames, len(LAFAN1_G1_JOINT_NAMES)), dtype=np.float64)
        joints[:, 0] = np.arange(frames)
        motion = MotionReference(
            "linear", 2.0, LAFAN1_G1_JOINT_NAMES, root, quaternion, joints
        )
        result = resample_motion(motion, 4.0)
        self.assertEqual(result.num_frames, 6)
        np.testing.assert_allclose(result.times, np.arange(6) / 4.0)
        np.testing.assert_allclose(result.root_pos[:, 0], np.arange(6) / 2.0)
        np.testing.assert_allclose(result.joint_pos[:, 0], np.arange(6) / 2.0)
        np.testing.assert_allclose(result.root_quat_wxyz[:, 0], 1.0)
        np.testing.assert_allclose(result.root_ang_vel_w, 0.0, atol=1.0e-7)

    def test_standard_schema_round_trip_without_pickle(self) -> None:
        frames, bodies = 5, 30
        arrays = {
            "fps": np.asarray([50.0], dtype=np.float32),
            "joint_pos": np.zeros((frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((frames, 29), dtype=np.float32),
            "body_pos_w": np.zeros((frames, bodies, 3), dtype=np.float32),
            "body_quat_w": np.zeros((frames, bodies, 4), dtype=np.float32),
            "body_lin_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
            "body_ang_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
            "body_names": np.asarray([f"body_{index}" for index in range(bodies)]),
        }
        self.assertEqual(validate_standard_npz(arrays), frames)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez_compressed(path, **arrays)
            loaded = load_npz_arrays(path)
        self.assertEqual(validate_standard_npz(loaded), frames)
        np.testing.assert_array_equal(loaded["body_names"], arrays["body_names"])

    def test_schema_rejects_body_count_mismatch(self) -> None:
        arrays = {
            "fps": np.asarray([50.0]),
            "joint_pos": np.zeros((3, 29)),
            "joint_vel": np.zeros((3, 29)),
            "body_pos_w": np.zeros((3, 30, 3)),
            "body_quat_w": np.zeros((3, 29, 4)),
            "body_lin_vel_w": np.zeros((3, 30, 3)),
            "body_ang_vel_w": np.zeros((3, 30, 3)),
        }
        with self.assertRaisesRegex(ValueError, "same body count"):
            validate_standard_npz(arrays)


if __name__ == "__main__":
    unittest.main()
