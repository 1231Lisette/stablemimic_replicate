"""Fixed-horizon on-policy rollout storage and GAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stablemimic.core.observations import ACTION_DIM, ACTOR_OBS_DIM, CRITIC_OBS_DIM, GATE_OBS_DIM


@dataclass(frozen=True)
class RolloutBatch:
    actor_observation: torch.Tensor
    gate_observation: torch.Tensor
    critic_observation: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    old_value: torch.Tensor
    advantage: torch.Tensor
    return_: torch.Tensor
    gate_target: torch.Tensor


class RolloutStorage:
    def __init__(self, steps: int, num_envs: int, device: torch.device | str):
        shape = (steps, num_envs)
        self.steps, self.num_envs = int(steps), int(num_envs)
        self.device = torch.device(device)
        self.actor_observation = torch.zeros(*shape, ACTOR_OBS_DIM, device=device)
        self.gate_observation = torch.zeros(*shape, GATE_OBS_DIM, device=device)
        self.critic_observation = torch.zeros(*shape, CRITIC_OBS_DIM, device=device)
        self.actions = torch.zeros(*shape, ACTION_DIM, device=device)
        self.log_probabilities = torch.zeros(*shape, device=device)
        self.values = torch.zeros(*shape, device=device)
        self.rewards = torch.zeros(*shape, device=device)
        self.dones = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.gate_targets = torch.zeros(*shape, 2, device=device)
        self.advantages = torch.zeros(*shape, device=device)
        self.returns = torch.zeros(*shape, device=device)
        self.cursor = 0

    def add(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor,
        critic_observation: torch.Tensor, action: torch.Tensor, log_probability: torch.Tensor,
        value: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
        gate_target: torch.Tensor,
    ) -> None:
        if self.cursor >= self.steps:
            raise RuntimeError("rollout storage is full")
        values = (
            (self.actor_observation, actor_observation),
            (self.gate_observation, gate_observation),
            (self.critic_observation, critic_observation),
            (self.actions, action),
            (self.log_probabilities, log_probability),
            (self.values, value),
            (self.rewards, reward),
            (self.dones, done),
            (self.gate_targets, gate_target),
        )
        for destination, source in values:
            destination[self.cursor].copy_(source)
        self.cursor += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        if self.cursor != self.steps:
            raise RuntimeError("rollout must be full before computing returns")
        gae = torch.zeros(self.num_envs, device=self.device)
        for step in reversed(range(self.steps)):
            next_value = last_value if step == self.steps - 1 else self.values[step + 1]
            not_done = (~self.dones[step]).float()
            delta = self.rewards[step] + gamma * next_value * not_done - self.values[step]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[step] = gae
        self.returns.copy_(self.advantages + self.values)

    def minibatches(self, count: int):
        total = self.steps * self.num_envs
        if total % count != 0:
            raise ValueError(f"{total} samples are not divisible by {count} minibatches")
        permutation = torch.randperm(total, device=self.device)
        size = total // count
        flattened = {
            "actor_observation": self.actor_observation.flatten(0, 1),
            "gate_observation": self.gate_observation.flatten(0, 1),
            "critic_observation": self.critic_observation.flatten(0, 1),
            "action": self.actions.flatten(0, 1),
            "old_log_probability": self.log_probabilities.flatten(0, 1),
            "old_value": self.values.flatten(0, 1),
            "advantage": self.advantages.flatten(0, 1),
            "return_": self.returns.flatten(0, 1),
            "gate_target": self.gate_targets.flatten(0, 1),
        }
        for start in range(0, total, size):
            index = permutation[start : start + size]
            yield RolloutBatch(**{name: value[index] for name, value in flattened.items()})

    def clear(self) -> None:
        self.cursor = 0
