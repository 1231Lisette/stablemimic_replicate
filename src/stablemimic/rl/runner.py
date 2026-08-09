"""End-to-end on-policy runner for the custom StableMimic environment."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from stablemimic.config import StableMimicCfg
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic

from .ppo import PPO
from .storage import RolloutStorage


class StableMimicRunner:
    def __init__(self, env, config: StableMimicCfg, run_dir: str | Path):
        self.env, self.config = env, config
        self.device = torch.device(env.unwrapped.device)
        actor = StableMimicActor(
            config.model.expert_hidden_dims,
            config.model.gate_hidden_dims,
            config.model.activation,
            config.model.initial_std,
        )
        critic = StableMimicCritic(config.model.critic_hidden_dims, config.model.activation)
        self.agent = StableMimicAgent(actor, critic).to(self.device)
        self.ppo = PPO(self.agent, config.ppo)
        self.storage = RolloutStorage(
            config.ppo.rollout_steps, env.unwrapped.num_envs, self.device
        )
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.iteration = 0

    def load(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(payload["agent"])
        self.ppo.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        if "environment_training_state" in payload:
            self.env.unwrapped.load_training_state(payload["environment_training_state"])

    def save(self, name: str | None = None) -> Path:
        checkpoint = self.run_dir / (name or f"checkpoint_{self.iteration:06d}.pt")
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "iteration": self.iteration,
                "agent": self.agent.state_dict(),
                "optimizer": self.ppo.optimizer.state_dict(),
                "environment_training_state": self.env.unwrapped.training_state(),
                "config": asdict(self.config),
            },
            temporary,
        )
        temporary.replace(checkpoint)
        return checkpoint

    def learn(self, iterations: int) -> None:
        observation, _ = self.env.reset()
        actor_raw, critic_raw = observation["policy"], observation["critic"]
        gate_raw = self.env.unwrapped.gate_observations
        for _ in range(iterations):
            started = time.perf_counter()
            episode_reward = 0.0
            tracking_reward_sum = 0.0
            recovery_reward_sum = 0.0
            tracking_samples = 0
            recovery_samples = 0
            transition_samples = 0
            terminated_samples = 0
            truncated_samples = 0
            clipped_action_elements = 0
            unit_action_exceed_elements = 0
            action_elements = 0
            gate_weight_sum = torch.zeros(2, device=self.device)
            event_counts = {
                name: 0
                for name in (
                    "recovery_success",
                    "recovery_failure",
                    "transition_completed",
                    "sequence_termination",
                    "unrecoverable_fall_termination",
                    "timeout",
                )
            }
            for _step in range(self.config.ppo.rollout_steps):
                self.agent.update_normalizers(actor_raw, gate_raw, critic_raw)
                actor_obs, gate_obs, critic_obs = self.agent.normalized(actor_raw, gate_raw, critic_raw)
                gate_target = self.env.unwrapped.gate_targets.clone()
                with torch.no_grad():
                    actions, log_probability, _, policy_output = self.agent.actor.act(
                        actor_obs, gate_obs
                    )
                    values = self.agent.critic(critic_obs)
                next_observation, reward, terminated, truncated, _ = self.env.step(actions)
                done = terminated | truncated
                self.storage.add(
                    actor_obs,
                    gate_obs,
                    critic_obs,
                    actions,
                    log_probability,
                    values,
                    reward,
                    done,
                    gate_target,
                )
                episode_reward += float(reward.mean())
                tracking_mask = gate_target[:, 0] == 1.0
                recovery_mask = gate_target[:, 1] == 1.0
                transition_mask = ~(tracking_mask | recovery_mask)
                tracking_reward_sum += float(reward[tracking_mask].sum())
                recovery_reward_sum += float(reward[recovery_mask].sum())
                tracking_samples += int(tracking_mask.sum())
                recovery_samples += int(recovery_mask.sum())
                transition_samples += int(transition_mask.sum())
                terminated_samples += int(terminated.sum())
                truncated_samples += int(truncated.sum())
                clipped_action_elements += int(
                    (actions.abs() > self.config.environment.action_clip).sum()
                )
                unit_action_exceed_elements += int((actions.abs() > 1.0).sum())
                action_elements += actions.numel()
                gate_weight_sum += policy_output.gate_weights.sum(0)
                for name in event_counts:
                    event_counts[name] += int(self.env.unwrapped.latest_events[name].sum())
                actor_raw, critic_raw = next_observation["policy"], next_observation["critic"]
                gate_raw = self.env.unwrapped.gate_observations
            with torch.no_grad():
                _, _, normalized_critic = self.agent.normalized(actor_raw, gate_raw, critic_raw)
                last_value = self.agent.critic(normalized_critic)
            self.storage.compute_returns(
                last_value, self.config.ppo.gamma, self.config.ppo.gae_lambda
            )
            metrics = self.ppo.update(self.storage)
            self.iteration += 1
            record = {
                "iteration": self.iteration,
                "mean_step_reward": episode_reward / self.config.ppo.rollout_steps,
                "policy_std": float(self.agent.actor.log_std.exp()),
                "tracking_mean_reward": tracking_reward_sum / max(tracking_samples, 1),
                "recovery_mean_reward": recovery_reward_sum / max(recovery_samples, 1),
                "tracking_sample_fraction": tracking_samples / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "recovery_sample_fraction": recovery_samples / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "transition_sample_fraction": transition_samples / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "termination_fraction": terminated_samples / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "truncation_fraction": truncated_samples / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "action_clip_fraction": clipped_action_elements / max(action_elements, 1),
                "unit_action_exceed_fraction": unit_action_exceed_elements / max(
                    action_elements, 1
                ),
                "mean_tracking_gate_weight": float(gate_weight_sum[0]) / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                "mean_recovery_gate_weight": float(gate_weight_sum[1]) / max(
                    self.config.ppo.rollout_steps * self.env.unwrapped.num_envs, 1
                ),
                **{f"{name}_count": count for name, count in event_counts.items()},
                "wall_seconds": time.perf_counter() - started,
                **asdict(metrics),
            }
            if not all(math_is_finite(value) for value in record.values() if isinstance(value, float)):
                raise FloatingPointError(f"non-finite training metric: {record}")
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if self.iteration % self.config.training.log_interval == 0 or iterations == 1:
                print("[TRAIN] " + " ".join(f"{key}={value:.6g}" for key, value in record.items() if isinstance(value, float)), flush=True)
            if self.iteration % self.config.training.checkpoint_interval == 0:
                print(f"[CHECKPOINT] {self.save()}", flush=True)
        print(f"[CHECKPOINT] {self.save('latest.pt')}", flush=True)


def math_is_finite(value: float) -> bool:
    return not (value != value or value in (float("inf"), float("-inf")))
