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
    load_lafan1_npz,
    load_segmented_recovery_motions,
    segment_recovery_motion,
)
from stablemimic.config import RecoverySegmentationCfg
from stablemimic.motion.reference import MotionReference


def valid_rows() -> np.ndarray:
    rows = np.zeros((2, 36), dtype=np.float64)
    rows[:, 2] = 0.8
    rows[:, 6] = 1.0
    return rows


def write_valid_npz(path: Path, frames: int = 3) -> None:
    body_quat = np.zeros((frames, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    source_quat = np.zeros((frames, 4), dtype=np.float32)
    source_quat[:, 0] = 1.0
    np.savez_compressed(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((frames, 29), dtype=np.float32),
        joint_vel=np.zeros((frames, 29), dtype=np.float32),
        body_pos_w=np.zeros((frames, 2, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
        joint_names=np.asarray(LAFAN1_G1_JOINT_NAMES),
        source_root_pos=np.zeros((frames, 3), dtype=np.float32),
        source_root_quat_wxyz=source_quat,
    )


class LAFAN1LoaderTests(unittest.TestCase):
    def test_segment_repeated_recovery_recording_into_atomic_clips(self) -> None:
        frames = 120
        root_pos = np.zeros((frames, 3), dtype=np.float64)
        root_pos[:, 2] = 0.8
        root_pos[10:36, 2] = 0.3
        root_pos[65:85, 2] = 0.3
        quaternions = np.zeros((frames, 4), dtype=np.float64)
        quaternions[:, 3] = 1.0
        quaternions[10:36, 1] = np.sin(np.pi / 4)
        quaternions[10:36, 3] = np.cos(np.pi / 4)
        quaternions[65:85, 1] = np.sin(np.pi / 4)
        quaternions[65:85, 3] = np.cos(np.pi / 4)
        joints = np.zeros((frames, len(LAFAN1_G1_JOINT_NAMES)), dtype=np.float64)
        joints[:, 0] = np.arange(frames)
        motion = MotionReference(
            "repeated", 30.0, LAFAN1_G1_JOINT_NAMES, root_pos, quaternions, joints
        )
        clips = segment_recovery_motion(motion, RecoverySegmentationCfg())
        self.assertEqual(len(clips), 2)
        self.assertEqual([clip.name for clip in clips], [
            "repeated__recovery_000", "repeated__recovery_001"
        ])
        self.assertLess(clips[0].root_pos[0, 2], 0.5)
        self.assertGreaterEqual(clips[0].root_pos[-1, 2], 0.7)
        self.assertEqual(clips[0].joint_pos[0, 0], 10.0)
        self.assertEqual(clips[0].joint_pos[-1, 0], 50.0)
        self.assertEqual(clips[1].joint_pos[0, 0], 65.0)
        self.assertEqual(clips[1].joint_pos[-1, 0], 99.0)

    def test_recovery_segmentation_rejects_recording_without_get_up(self) -> None:
        rows = valid_rows()
        rows[:, 2] = 0.2
        motion = MotionReference(
            "never_upright", 30.0, LAFAN1_G1_JOINT_NAMES,
            rows[:, :3], rows[:, 3:7], rows[:, 7:],
        )
        with self.assertRaisesRegex(ValueError, "No valid fall-to-upright clips"):
            segment_recovery_motion(
                motion, RecoverySegmentationCfg(hold_time_s=1.0 / 30.0)
            )

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

    def test_load_valid_npz_and_convert_wxyz_to_xyzw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dance_test.npz"
            write_valid_npz(path)
            motion = load_lafan1_npz(path)
            self.assertEqual(motion.num_frames, 3)
            self.assertEqual(motion.fps, 50.0)
            np.testing.assert_allclose(motion.root_quat_xyzw[:, 3], 1.0)
            np.testing.assert_allclose(motion.root_quat_xyzw[:, :3], 0.0)

    def test_explicit_file_lists_select_one_tracking_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dance1.npz", "dance2.npz", "fallAndGetUp1.npz"):
                write_valid_npz(root / name)
            libraries = discover_motion_libraries(
                root,
                tracking_files=("dance2.npz",),
                recovery_files=("fallAndGetUp1.npz",),
            )
            self.assertEqual([path.name for path in libraries.tracking], ["dance2.npz"])
            self.assertEqual(
                [path.name for path in libraries.recovery], ["fallAndGetUp1.npz"]
            )

    def test_file_lists_must_be_configured_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be set together"):
                discover_motion_libraries(
                    directory, tracking_files=("dance1.npz",)
                )

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
        recovery_clips = load_segmented_recovery_motions(
            libraries.recovery, RecoverySegmentationCfg()
        )
        # The exact count depends on the retargeting geometry because the
        # fall/upright thresholds define atomic clip boundaries.  The legacy
        # public CSVs produce 80 clips while the corrected GMR references
        # produce 85 from the same six recordings.  The integration contract
        # is that every recovery recording yields at least one bounded clip.
        self.assertGreaterEqual(len(recovery_clips), len(libraries.recovery))
        self.assertLessEqual(max(motion.duration for motion in recovery_clips), 20.0)


if __name__ == "__main__":
    unittest.main()
