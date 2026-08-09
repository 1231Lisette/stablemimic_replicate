"""PPO with StableMimic Gate and transition auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from stablemimic.config import PpoCfg
from stablemimic.models import StableMimicAgent

from .storage import RolloutStorage


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    gate_ce: float
    transition_loss: float
    consistency_loss: float
    alignment_loss: float
    approximate_kl: float
    learning_rate: float
    gradient_norm: float


class PPO:
    def __init__(self, agent: StableMimicAgent, config: PpoCfg):
        self.agent, self.config = agent, config
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    def update(self, storage: RolloutStorage) -> PPOMetrics:
        totals = {key: 0.0 for key in (
            "policy_loss", "value_loss", "entropy", "gate_ce", "transition_loss",
            "consistency_loss", "alignment_loss", "approximate_kl", "gradient_norm",
        )}
        updates = 0
        for _ in range(self.config.epochs):
            for batch in storage.minibatches(self.config.minibatches):
                advantage = batch.advantage
                if self.config.normalize_advantage:
                    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1.0e-8)
                new_log_prob, entropy, output = self.agent.actor.evaluate_actions(
                    batch.actor_observation, batch.gate_observation, batch.action
                )
                value = self.agent.critic(batch.critic_observation)
                ratio = torch.exp(new_log_prob - batch.old_log_probability)
                unclipped = -advantage * ratio
                clipped = -advantage * ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                policy_loss = torch.maximum(unclipped, clipped).mean()

                value_clipped = batch.old_value + (value - batch.old_value).clamp(
                    -self.config.value_clip, self.config.value_clip
                )
                value_loss = 0.5 * torch.maximum(
                    torch.square(value - batch.return_), torch.square(value_clipped - batch.return_)
                ).mean()

                per_sample_gate_ce = -(
                    batch.gate_target * torch.log(output.gate_weights.clamp_min(1.0e-8))
                ).sum(-1)
                mixed = (batch.gate_target > 0.0).all(-1)
                gate_sample_weight = torch.ones_like(per_sample_gate_ce)
                gate_sample_weight[mixed] = self.config.transition_coefficient
                gate_ce = (per_sample_gate_ce * gate_sample_weight).sum() / gate_sample_weight.sum()
                transition_loss = (
                    per_sample_gate_ce[mixed].mean()
                    if mixed.any() else output.mean.sum() * 0.0
                )
                previous_gate_weights = self.agent.actor.gate_weights(
                    batch.previous_gate_observation
                )
                consistency_loss = (
                    F.mse_loss(
                        output.gate_weights[batch.same_regime],
                        previous_gate_weights[batch.same_regime],
                    )
                    if batch.same_regime.any() else output.mean.sum() * 0.0
                )
                alignment_loss = (
                    F.mse_loss(output.tracking_mean[mixed], output.recovery_mean[mixed])
                    if mixed.any() else output.mean.sum() * 0.0
                )
                # The paper gives 0.05 but not the action-dimension reduction.
                # Mean reduction prevents the coefficient from scaling 29-fold.
                entropy_mean = entropy.mean() / batch.action.shape[-1]
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                    + self.config.gate_ce_coefficient * gate_ce
                    + self.config.consistency_coefficient * consistency_loss
                    + self.config.alignment_coefficient * alignment_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - (new_log_prob - batch.old_log_probability)).mean()
                metrics = {
                    "policy_loss": policy_loss, "value_loss": value_loss,
                    "entropy": entropy_mean, "gate_ce": gate_ce,
                    "transition_loss": transition_loss, "consistency_loss": consistency_loss,
                    "alignment_loss": alignment_loss, "approximate_kl": approximate_kl,
                    "gradient_norm": gradient_norm,
                }
                for key, value_metric in metrics.items():
                    totals[key] += float(value_metric.detach())
                updates += 1
        self._adapt_learning_rate(totals["approximate_kl"] / max(updates, 1))
        storage.clear()
        averaged = {key: value / max(updates, 1) for key, value in totals.items()}
        return PPOMetrics(**averaged, learning_rate=self.optimizer.param_groups[0]["lr"])

    def _adapt_learning_rate(self, approximate_kl: float) -> None:
        current = self.optimizer.param_groups[0]["lr"]
        if approximate_kl > 2.0 * self.config.target_kl:
            current = max(1.0e-5, current / 1.5)
        elif 0.0 < approximate_kl < 0.5 * self.config.target_kl:
            current = min(1.0e-2, current * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = current
