"""Tracking/recovery/transition phase state without policy information leaks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class MotionPhase(IntEnum):
    TRACKING = 0
    RECOVERY = 1
    TRANSITION = 2


@dataclass
class PhaseState:
    phase: torch.Tensor
    transition_time: torch.Tensor
    recovery_error_time: torch.Tensor
    transition_duration: float = 1.5
    error_timeout: float = 2.0

    @classmethod
    def create(
        cls, num_envs: int, device: torch.device | str, *, transition_duration: float = 1.5,
        error_timeout: float = 2.0,
    ) -> "PhaseState":
        return cls(
            phase=torch.full((num_envs,), int(MotionPhase.TRACKING), dtype=torch.long, device=device),
            transition_time=torch.zeros(num_envs, device=device),
            recovery_error_time=torch.zeros(num_envs, device=device),
            transition_duration=float(transition_duration),
            error_timeout=float(error_timeout),
        )

    def reset(self, env_ids: torch.Tensor, recovery: torch.Tensor) -> None:
        if recovery.shape != env_ids.shape:
            raise ValueError("recovery mask must match env_ids")
        self.phase[env_ids] = torch.where(
            recovery,
            torch.full_like(env_ids, int(MotionPhase.RECOVERY)),
            torch.full_like(env_ids, int(MotionPhase.TRACKING)),
        )
        self.transition_time[env_ids] = 0.0
        self.recovery_error_time[env_ids] = 0.0

    def update(
        self, recovery_similarity: torch.Tensor, recovery_error: torch.Tensor, dt: float,
        success_threshold: float,
    ) -> torch.Tensor:
        """Advance phases and return recovery-failure terminations."""
        recovery_mask = self.phase == int(MotionPhase.RECOVERY)
        transition_mask = self.phase == int(MotionPhase.TRANSITION)
        success = recovery_mask & (recovery_similarity >= success_threshold)
        self.phase[success] = int(MotionPhase.TRANSITION)
        self.transition_time[success] = 0.0

        bad = recovery_mask & (recovery_error > (1.0 - success_threshold))
        self.recovery_error_time[bad] += dt
        self.recovery_error_time[recovery_mask & ~bad] = 0.0
        failed = recovery_mask & (self.recovery_error_time >= self.error_timeout)

        self.transition_time[transition_mask] += dt
        completed = transition_mask & (self.transition_time >= self.transition_duration)
        self.phase[completed] = int(MotionPhase.TRACKING)
        self.transition_time[completed] = self.transition_duration
        return failed

    def gate_target(self) -> torch.Tensor:
        target = torch.zeros(self.phase.shape[0], 2, device=self.phase.device)
        tracking = self.phase == int(MotionPhase.TRACKING)
        recovery = self.phase == int(MotionPhase.RECOVERY)
        transition = self.phase == int(MotionPhase.TRANSITION)
        target[tracking, 0] = 1.0
        target[recovery, 1] = 1.0
        if transition.any():
            alpha = (self.transition_time[transition] / self.transition_duration).clamp(0.0, 1.0)
            target[transition, 0] = alpha
            target[transition, 1] = 1.0 - alpha
        return target

    def reward_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.gate_target()
        return target[:, 0], target[:, 1]
