from pathlib import Path
import unittest

from stablemimic.config import fall_recovery_curriculum_probability, load_config


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
        self.assertEqual(config.training.fall_recovery_warmup_iterations, 100)
        self.assertEqual(config.training.fall_recovery_ramp_iterations, 100)

    def test_fall_recovery_curriculum_warmup_and_ramp(self) -> None:
        probability = fall_recovery_curriculum_probability
        self.assertEqual(probability(1, 100, 100), 0.0)
        self.assertEqual(probability(100, 100, 100), 0.0)
        self.assertAlmostEqual(probability(101, 100, 100), 0.01)
        self.assertAlmostEqual(probability(150, 100, 100), 0.5)
        self.assertEqual(probability(200, 100, 100), 1.0)
        self.assertEqual(probability(250, 100, 100), 1.0)
        self.assertEqual(probability(1, 0, 0), 1.0)
        with self.assertRaises(ValueError):
            probability(0, 100, 100)
