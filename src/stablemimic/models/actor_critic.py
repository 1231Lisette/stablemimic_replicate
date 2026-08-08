"""Dual-expert Actor, proprioceptive Gate, and privileged Critic."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from stablemimic.core.observations import ACTION_DIM, ACTOR_OBS_DIM, CRITIC_OBS_DIM, GATE_OBS_DIM


def _activation(name: str) -> type[nn.Module]:
    choices = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    try:
        return choices[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    activation_type = _activation(activation)
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), activation_type()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    network = nn.Sequential(*layers)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(network[-1].weight, gain=0.01)
    return network


@dataclass(frozen=True)
class PolicyOutput:
    mean: torch.Tensor
    gate_weights: torch.Tensor
    tracking_mean: torch.Tensor
    recovery_mean: torch.Tensor


class StableMimicActor(nn.Module):
    """Two 29-D expert means fused continuously by a two-way softmax Gate."""

    def __init__(
        self,
        expert_hidden_dims: tuple[int, ...] = (512, 256, 128),
        gate_hidden_dims: tuple[int, ...] = (256, 128),
        activation: str = "elu",
        initial_std: float = 1.0,
    ):
        super().__init__()
        if initial_std <= 0.0:
            raise ValueError("initial_std must be positive")
        self.tracking_expert = _mlp(ACTOR_OBS_DIM, expert_hidden_dims, ACTION_DIM, activation)
        self.recovery_expert = _mlp(ACTOR_OBS_DIM, expert_hidden_dims, ACTION_DIM, activation)
        self.gate = _mlp(GATE_OBS_DIM, gate_hidden_dims, 2, activation)
        self.log_std = nn.Parameter(torch.tensor(math.log(initial_std)))

    def forward(self, actor_observation: torch.Tensor, gate_observation: torch.Tensor) -> PolicyOutput:
        if not torch.onnx.is_in_onnx_export() and actor_observation.shape[-1] != ACTOR_OBS_DIM:
            raise ValueError(f"Actor input must be {ACTOR_OBS_DIM}-D")
        if not torch.onnx.is_in_onnx_export() and gate_observation.shape[-1] != GATE_OBS_DIM:
            raise ValueError(f"Gate input must be {GATE_OBS_DIM}-D")
        tracking_mean = self.tracking_expert(actor_observation)
        recovery_mean = self.recovery_expert(actor_observation)
        gate_weights = torch.softmax(self.gate(gate_observation), dim=-1)
        mean = gate_weights[:, 0:1] * tracking_mean + gate_weights[:, 1:2] * recovery_mean
        return PolicyOutput(mean, gate_weights, tracking_mean, recovery_mean)

    def distribution_parameters(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor
    ) -> tuple[PolicyOutput, torch.Tensor]:
        output = self(actor_observation, gate_observation)
        std = self.log_std.exp().expand_as(output.mean)
        return output, std

    def act(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PolicyOutput]:
        output, std = self.distribution_parameters(actor_observation, gate_observation)
        action = output.mean if deterministic else output.mean + std * torch.randn_like(output.mean)
        log_prob = self._log_prob(action, output.mean, std)
        entropy = self._entropy(std)
        return action, log_prob, entropy, output

    def evaluate_actions(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PolicyOutput]:
        output, std = self.distribution_parameters(actor_observation, gate_observation)
        return self._log_prob(actions, output.mean, std), self._entropy(std), output

    @staticmethod
    def _log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        value = -0.5 * (torch.square((action - mean) / std) + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
        return value.sum(-1)

    @staticmethod
    def _entropy(std: torch.Tensor) -> torch.Tensor:
        return (torch.log(std) + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(-1)


class StableMimicCritic(nn.Module):
    def __init__(
        self, hidden_dims: tuple[int, ...] = (512, 256, 128), activation: str = "elu"
    ):
        super().__init__()
        self.network = _mlp(CRITIC_OBS_DIM, hidden_dims, 1, activation)

    def forward(self, critic_observation: torch.Tensor) -> torch.Tensor:
        if critic_observation.shape[-1] != CRITIC_OBS_DIM:
            raise ValueError(f"Critic input must be {CRITIC_OBS_DIM}-D")
        return self.network(critic_observation).squeeze(-1)
