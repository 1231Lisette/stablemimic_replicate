from __future__ import annotations

import unittest

from stablemimic.sim import close_simulation_app
from stablemimic.sim.g1_mapping import build_lafan1_g1_joint_mapping
from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES


class FakeApplication:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ExitingApplication:
    def close(self) -> None:
        raise SystemExit(0)


class SimulationShutdownTests(unittest.TestCase):
    def test_normal_close_returns_without_forcing_exit(self) -> None:
        application = FakeApplication()
        close_simulation_app(application, timeout_seconds=0.1)
        self.assertTrue(application.closed)

    def test_timeout_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            close_simulation_app(FakeApplication(), timeout_seconds=0.0)

    def test_close_cannot_mask_prior_failure(self) -> None:
        with self.assertRaises(SystemExit) as context:
            close_simulation_app(
                ExitingApplication(), timeout_seconds=0.1, forced_exit_code=7
            )
        self.assertEqual(context.exception.code, 7)

    def test_lafan1_mapping_accepts_only_hand_extras(self) -> None:
        simulator_names = [
            LAFAN1_G1_JOINT_NAMES[1],
            "left_hand_index_0_joint",
            LAFAN1_G1_JOINT_NAMES[0],
            *LAFAN1_G1_JOINT_NAMES[2:],
        ]
        mapping = build_lafan1_g1_joint_mapping(simulator_names)
        self.assertEqual(mapping.csv_to_sim[0:2], (2, 0))
        self.assertEqual(mapping.extra_hand_joints, ("left_hand_index_0_joint",))

    def test_lafan1_mapping_rejects_unknown_extra(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported_extra"):
            build_lafan1_g1_joint_mapping([*LAFAN1_G1_JOINT_NAMES, "mystery_joint"])


if __name__ == "__main__":
    unittest.main()
