from __future__ import annotations

import math
import unittest

import numpy as np

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES
from stablemimic.motion.reference import MotionReference


class MotionReferenceTests(unittest.TestCase):
    def make_motion(self, quaternion_end: np.ndarray | None = None) -> MotionReference:
        if quaternion_end is None:
            quaternion_end = np.array([0.0, 0.0, 0.0, 1.0])
        return MotionReference(
            name="synthetic",
            fps=30.0,
            joint_names=LAFAN1_G1_JOINT_NAMES,
            root_pos=np.array([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8]]),
            root_quat_xyzw=np.array([[0.0, 0.0, 0.0, 1.0], quaternion_end]),
            joint_pos=np.stack([np.zeros(29), np.ones(29)]),
        )

    def test_half_source_frame_interpolation(self) -> None:
        motion = self.make_motion(np.array([0.0, 0.0, 1.0, 0.0]))
        sample = motion.sample(1.0 / 60.0)
        np.testing.assert_allclose(sample.root_pos, [0.5, 0.0, 0.8], atol=1.0e-9)
        np.testing.assert_allclose(sample.joint_pos, 0.5, atol=1.0e-9)
        expected_quaternion = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
        np.testing.assert_allclose(sample.root_quat_xyzw, expected_quaternion, atol=1.0e-7)
        np.testing.assert_allclose(sample.root_ang_vel_world, [0.0, 0.0, 30.0 * math.pi], atol=1.0e-7)

    def test_30_fps_reference_on_50_hz_policy_clock(self) -> None:
        motion = MotionReference(
            name="clock",
            fps=30.0,
            joint_names=LAFAN1_G1_JOINT_NAMES,
            root_pos=np.column_stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)]),
            root_quat_xyzw=np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)),
            joint_pos=np.zeros((5, 29)),
        )
        sampled_x = [motion.sample(step / 50.0).root_pos[0] for step in range(5)]
        np.testing.assert_allclose(sampled_x, [0.0, 0.6, 1.2, 1.8, 2.4], atol=1.0e-9)

    def test_quaternion_sign_is_canonicalized(self) -> None:
        motion = self.make_motion(np.array([0.0, 0.0, 0.0, -1.0]))
        sample = motion.sample(1.0 / 60.0)
        np.testing.assert_allclose(sample.root_quat_xyzw, [0.0, 0.0, 0.0, 1.0], atol=1.0e-9)
        np.testing.assert_allclose(sample.root_ang_vel_world, 0.0, atol=1.0e-9)

    def test_clamp_and_loop(self) -> None:
        motion = self.make_motion()
        np.testing.assert_allclose(motion.sample(-1.0).root_pos, motion.root_pos[0])
        np.testing.assert_allclose(motion.sample(100.0).root_pos, motion.root_pos[-1])
        looped = motion.sample(motion.duration + 0.5 * motion.duration, loop=True)
        np.testing.assert_allclose(looped.root_pos, [0.5, 0.0, 0.8], atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
