"""Actor/Critic plus training-time observation normalization."""

from __future__ import annotations

import torch
from torch import nn

from stablemimic.core.observations import ACTOR_OBS_DIM, CRITIC_OBS_DIM, GATE_OBS_DIM

from .actor_critic import StableMimicActor, StableMimicCritic
from .normalization import RunningMeanStd


class StableMimicAgent(nn.Module):
    def __init__(self, actor: StableMimicActor, critic: StableMimicCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.actor_normalizer = RunningMeanStd(ACTOR_OBS_DIM)
        self.gate_normalizer = RunningMeanStd(GATE_OBS_DIM)
        self.critic_normalizer = RunningMeanStd(CRITIC_OBS_DIM)

    @torch.no_grad()
    def update_normalizers(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor,
        critic_observation: torch.Tensor,
    ) -> None:
        self.actor_normalizer.update(actor_observation)
        self.gate_normalizer.update(gate_observation)
        self.critic_normalizer.update(critic_observation)

    def normalized(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor,
        critic_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.actor_normalizer(actor_observation),
            self.gate_normalizer(gate_observation),
            self.critic_normalizer(critic_observation),
        )
