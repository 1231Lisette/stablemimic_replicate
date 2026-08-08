from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class StableMimicTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        from stablemimic.config import load_config

        self.config = load_config(Path(__file__).parents[1] / "configs" / "stablemimic_g1.yaml")

    def test_observation_dimensions_and_no_recovery_leak(self) -> None:
        from stablemimic.core.observations import (
            ACTOR_OBS_DIM, CRITIC_OBS_DIM, GATE_OBS_DIM, ObservationHistory,
            build_motion_command, build_proprioception,
        )

        batch = 3
        proprio = build_proprioception(
            torch.zeros(batch, 3), torch.zeros(batch, 3), torch.zeros(batch, 29),
            torch.zeros(batch, 29), torch.zeros(batch, 29),
        )
        command = build_motion_command(
            torch.zeros(batch, 29), torch.zeros(batch, 29), torch.zeros(batch, 3),
            torch.zeros(batch, 3), torch.zeros(batch, 1), torch.zeros(batch, 3),
            torch.zeros(batch, 29), torch.zeros(batch, 29), torch.zeros(batch, 2),
        )
        history_a = ObservationHistory(batch, "cpu")
        history_b = ObservationHistory(batch, "cpu")
        ids = torch.arange(batch)
        history_a.reset(ids, proprio, command, proprio, torch.zeros(batch, 43))
        history_b.reset(ids, proprio, command, proprio, torch.ones(batch, 43))
        a, b = history_a.batch(), history_b.batch()
        self.assertEqual(a.actor.shape, (batch, ACTOR_OBS_DIM))
        self.assertEqual(a.gate.shape, (batch, GATE_OBS_DIM))
        self.assertEqual(a.critic.shape, (batch, CRITIC_OBS_DIM))
        self.assertTrue(torch.equal(a.actor, b.actor))
        self.assertTrue(torch.equal(a.gate, b.gate))
        self.assertFalse(torch.equal(a.critic, b.critic))

    def test_phase_transition_targets(self) -> None:
        from stablemimic.core.phases import MotionPhase, PhaseState

        state = PhaseState.create(2, "cpu", transition_duration=1.5)
        ids = torch.arange(2)
        state.reset(ids, torch.tensor([False, True]))
        self.assertEqual(state.phase.tolist(), [MotionPhase.TRACKING, MotionPhase.RECOVERY])
        state.update(torch.tensor([0.0, 1.0]), torch.zeros(2), 0.02, 0.8)
        self.assertEqual(int(state.phase[1]), MotionPhase.TRANSITION)
        state.update(torch.zeros(2), torch.zeros(2), 0.75, 0.8)
        self.assertTrue(torch.allclose(state.gate_target()[1], torch.tensor([0.5, 0.5])))

    def test_perfect_reward_exceeds_perturbed_and_recovery_ignores_xy(self) -> None:
        from stablemimic.rewards import KinematicState, whole_body_reward

        def state(root_x: float, body_delta: float = 0.0):
            root = torch.tensor([[root_x, 0.0, 0.8]])
            quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
            body = root[:, None, :] + torch.tensor([[[0.0, 0.0, body_delta], [0.1, 0.0, 0.5]]])
            body_quat = quat[:, None, :].expand(-1, 2, -1).clone()
            velocity = torch.zeros(1, 2, 3)
            return KinematicState(root, quat, body, body_quat, velocity, velocity)

        target = state(0.0)
        perfect = whole_body_reward(target, target, torch.tensor([False]), self.config.reward)
        bad = whole_body_reward(state(0.0, 0.5), target, torch.tensor([False]), self.config.reward)
        recovery_origin = whole_body_reward(target, target, torch.tensor([True]), self.config.reward)
        shifted = whole_body_reward(state(5.0), target, torch.tensor([True]), self.config.reward)
        self.assertGreater(float(perfect.total), float(bad.total))
        self.assertAlmostEqual(float(recovery_origin.total), float(shifted.total), places=5)

    def test_actor_and_one_ppo_update_are_finite(self) -> None:
        from stablemimic.core.observations import ACTOR_OBS_DIM, CRITIC_OBS_DIM, GATE_OBS_DIM
        from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic
        from stablemimic.rl import PPO, RolloutStorage

        actor = StableMimicActor((32, 16), (16,), initial_std=0.5)
        critic = StableMimicCritic((32, 16))
        agent = StableMimicAgent(actor, critic)
        ppo_config = self.config.ppo.__class__(
            **{**self.config.ppo.__dict__, "epochs": 1, "minibatches": 2}
        )
        ppo = PPO(agent, ppo_config)
        storage = RolloutStorage(2, 4, "cpu")
        for _ in range(2):
            actor_obs = torch.randn(4, ACTOR_OBS_DIM)
            gate_obs = torch.randn(4, GATE_OBS_DIM)
            critic_obs = torch.randn(4, CRITIC_OBS_DIM)
            with torch.no_grad():
                action, log_prob, _, output = actor.act(actor_obs, gate_obs)
                value = critic(critic_obs)
            self.assertTrue(torch.allclose(output.gate_weights.sum(-1), torch.ones(4)))
            explicit = (
                output.gate_weights[:, 0:1] * output.tracking_mean
                + output.gate_weights[:, 1:2] * output.recovery_mean
            )
            self.assertTrue(torch.allclose(output.mean, explicit))
            storage.add(
                actor_obs, gate_obs, critic_obs, action, log_prob, value,
                torch.randn(4), torch.zeros(4, dtype=torch.bool),
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
            )
        storage.compute_returns(torch.zeros(4), 0.99, 0.95)
        metrics = ppo.update(storage)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.__dict__.values()))
