from pathlib import Path
import unittest

from stablemimic.config import load_config


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_50_hz(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "stablemimic_g1.yaml"
        config = load_config(path)
        self.assertAlmostEqual(config.environment.physics_dt * config.environment.decimation, 0.02)
        self.assertEqual(config.environment.action_clip, 100.0)
        self.assertEqual(config.model.expert_hidden_dims, (512, 256, 128))
        self.assertEqual(config.model.initial_std, 0.2)
        self.assertEqual(config.ppo.rollout_steps, 24)
        self.assertTrue(config.recovery_segmentation.enabled)
        self.assertEqual(config.recovery_segmentation.hold_time_s, 0.5)
        self.assertTrue(config.environment.tracking_fall_recovery_enabled)
        self.assertEqual(config.environment.tracking_fall_height_threshold, 0.5)
