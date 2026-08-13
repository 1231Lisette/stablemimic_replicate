from pathlib import Path
import unittest

from stablemimic.config import fall_recovery_curriculum_probability, load_config


class ConfigTests(unittest.TestCase):
    def test_repository_config_is_50_hz(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "stablemimic_g1.yaml"
        config = load_config(path)
        self.assertAlmostEqual(config.environment.physics_dt * config.environment.decimation, 0.02)
        self.assertAlmostEqual(config.environment.recovery_static_reset_probability, 0.25)
        self.assertAlmostEqual(config.environment.recovery_phase_reset_min, 0.40)
        self.assertAlmostEqual(config.environment.recovery_phase_reset_max, 0.75)
        self.assertAlmostEqual(config.reward.recovery_progress_bonus, 1.0)
        self.assertAlmostEqual(
            config.reward.recovery_progress_height_weight
            + config.reward.recovery_progress_upright_weight,
            1.0,
        )
        self.assertTrue(config.recovery_segmentation.trim_to_tilted_nadir)
        self.assertEqual(config.environment.action_clip, 100.0)
        self.assertEqual(config.model.expert_hidden_dims, (512, 256, 128))
        self.assertEqual(config.model.initial_std, 0.2)
        self.assertEqual(config.ppo.rollout_steps, 24)
        self.assertTrue(config.recovery_segmentation.enabled)
        self.assertEqual(config.recovery_segmentation.hold_time_s, 0.5)
        self.assertTrue(config.environment.tracking_fall_recovery_enabled)
        self.assertEqual(config.environment.tracking_fall_height_threshold, 0.5)
        self.assertEqual(config.training.fall_recovery_warmup_iterations, 5000)
        self.assertEqual(config.training.fall_recovery_ramp_iterations, 100)

    def test_clean_single_tracking_npz_baseline(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "stablemimic_g1_gmr_single_baseline.yaml"
        config = load_config(path)
        self.assertEqual(config.tracking_motion_files, ("dance1_subject2.npz",))
        self.assertEqual(len(config.recovery_motion_files), 6)
        self.assertFalse(config.environment.tracking_fall_recovery_enabled)
        self.assertEqual(config.environment.recovery_static_reset_probability, 0.0)
        self.assertEqual(config.environment.recovery_phase_reset_min, -1.0)
        self.assertEqual(config.reward.recovery_progress_bonus, 0.0)
        self.assertEqual(config.model.initial_std, 1.0)
        self.assertEqual(config.ppo.learning_rate, 1.0e-3)

    def test_upstream_v7_has_attributed_reward_boundary(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "stablemimic_g1_upstream_v7.yaml"
        config = load_config(path)
        self.assertEqual(len(config.reward.tracking_body_names), 14)
        self.assertEqual(config.reward.tracking_body_names[0], "pelvis")
        self.assertAlmostEqual(config.reward.root_position.weight, 0.5)
        self.assertAlmostEqual(config.reward.body_position.sigma, 0.3)
        self.assertAlmostEqual(config.reward.recovery_base_height_weight, 5.0)
        self.assertAlmostEqual(config.reward.recovery_upright_weight, 0.25)
        self.assertAlmostEqual(config.reward.recovery_double_support_weight, 2.5)
        self.assertEqual(config.environment.action_scale, 0.5)
        self.assertFalse(config.environment.tracking_fall_recovery_enabled)

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
