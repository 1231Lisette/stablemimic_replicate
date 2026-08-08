"""Numerically stable streaming observation normalization."""

from __future__ import annotations

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1.0e-4, clip: float = 10.0):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))
        self.register_buffer("count", torch.tensor(float(epsilon)))
        self.clip = float(clip)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        if value.numel() == 0:
            return
        batch_mean = value.mean(dim=0)
        batch_variance = value.var(dim=0, unbiased=False)
        batch_count = value.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.variance * self.count
        m_b = batch_variance * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.variance.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = (value - self.mean) / torch.sqrt(self.variance + 1.0e-8)
        return normalized.clamp(-self.clip, self.clip)
