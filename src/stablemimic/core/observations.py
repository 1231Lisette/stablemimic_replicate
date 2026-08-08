"""Explicit Actor, Gate, and Critic observation schemas.

The paper publishes total dimensions and information boundaries, but not the
element-wise order. This file is the reproduction's single source of truth for
that otherwise-unpublished order.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACTION_DIM = 29
HISTORY_LENGTH = 4
GATE_FRAME_DIM = 93
COMMAND_DIM = 128
ACTOR_FRAME_DIM = 221
PRIVILEGED_DIM = 136
CRITIC_FRAME_DIM = 357
GATE_OBS_DIM = HISTORY_LENGTH * GATE_FRAME_DIM
ACTOR_OBS_DIM = HISTORY_LENGTH * ACTOR_FRAME_DIM
CRITIC_OBS_DIM = HISTORY_LENGTH * CRITIC_FRAME_DIM
RECOVERY_REFERENCE_DIM = 43


def _assert_matrix(name: str, value: torch.Tensor, rows: int, columns: int) -> None:
    if value.ndim != 2 or value.shape != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns}), got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def build_proprioception(
    base_angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_position_offset: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Build the deployable 93-D Gate frame in its frozen order."""
    batch = base_angular_velocity.shape[0]
    pieces = (
        ("base_angular_velocity", base_angular_velocity, 3),
        ("projected_gravity", projected_gravity, 3),
        ("joint_position_offset", joint_position_offset, 29),
        ("joint_velocity", joint_velocity, 29),
        ("previous_action", previous_action, 29),
    )
    for name, value, width in pieces:
        _assert_matrix(name, value, batch, width)
    result = torch.cat([value for _, value, _ in pieces], dim=-1)
    _assert_matrix("proprioception", result, batch, GATE_FRAME_DIM)
    return result


def build_motion_command(
    reference_joint_position: torch.Tensor,
    reference_joint_velocity: torch.Tensor,
    reference_root_linear_velocity: torch.Tensor,
    reference_root_angular_velocity: torch.Tensor,
    reference_root_height: torch.Tensor,
    reference_projected_gravity: torch.Tensor,
    future_joint_position_1: torch.Tensor,
    future_joint_position_2: torch.Tensor,
    phase_sin_cos: torch.Tensor,
) -> torch.Tensor:
    """Build the 128-D live nominal-motion command available at deployment.

    This command remains present during recovery. It never contains a get-up
    sequence id, frame, phase label, or hidden recovery successor.
    """
    batch = reference_joint_position.shape[0]
    pieces = (
        ("reference_joint_position", reference_joint_position, 29),
        ("reference_joint_velocity", reference_joint_velocity, 29),
        ("reference_root_linear_velocity", reference_root_linear_velocity, 3),
        ("reference_root_angular_velocity", reference_root_angular_velocity, 3),
        ("reference_root_height", reference_root_height, 1),
        ("reference_projected_gravity", reference_projected_gravity, 3),
        ("future_joint_position_1", future_joint_position_1, 29),
        ("future_joint_position_2", future_joint_position_2, 29),
        ("phase_sin_cos", phase_sin_cos, 2),
    )
    for name, value, width in pieces:
        _assert_matrix(name, value, batch, width)
    result = torch.cat([value for _, value, _ in pieces], dim=-1)
    _assert_matrix("motion_command", result, batch, COMMAND_DIM)
    return result


def build_recovery_reference(
    root_position: torch.Tensor,
    root_quaternion_xyzw: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    normalized_progress: torch.Tensor,
) -> torch.Tensor:
    """Build the Critic-only 43-D hidden recovery successor."""
    batch = root_position.shape[0]
    pieces = (
        ("root_position", root_position, 3),
        ("root_quaternion_xyzw", root_quaternion_xyzw, 4),
        ("root_linear_velocity", root_linear_velocity, 3),
        ("root_angular_velocity", root_angular_velocity, 3),
        ("joint_position", joint_position, 29),
        ("normalized_progress", normalized_progress, 1),
    )
    for name, value, width in pieces:
        _assert_matrix(name, value, batch, width)
    result = torch.cat([value for _, value, _ in pieces], dim=-1)
    _assert_matrix("recovery_reference", result, batch, RECOVERY_REFERENCE_DIM)
    return result


@dataclass(frozen=True)
class ObservationBatch:
    actor: torch.Tensor
    gate: torch.Tensor
    critic: torch.Tensor

    def validate(self) -> None:
        batch = self.actor.shape[0]
        _assert_matrix("actor observation", self.actor, batch, ACTOR_OBS_DIM)
        _assert_matrix("gate observation", self.gate, batch, GATE_OBS_DIM)
        _assert_matrix("critic observation", self.critic, batch, CRITIC_OBS_DIM)


class ObservationHistory:
    """Four-frame history with explicit privileged-information separation."""

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._gate = torch.zeros(num_envs, HISTORY_LENGTH, GATE_FRAME_DIM, device=device)
        self._actor = torch.zeros(num_envs, HISTORY_LENGTH, ACTOR_FRAME_DIM, device=device)
        self._critic = torch.zeros(num_envs, HISTORY_LENGTH, CRITIC_FRAME_DIM, device=device)

    def reset(
        self,
        env_ids: torch.Tensor,
        proprioception: torch.Tensor,
        command: torch.Tensor,
        uncorrupted_proprioception: torch.Tensor,
        recovery_reference: torch.Tensor,
    ) -> None:
        actor_frame, critic_frame = self._frames(
            proprioception, command, uncorrupted_proprioception, recovery_reference
        )
        self._gate[env_ids] = proprioception[:, None, :].expand(-1, HISTORY_LENGTH, -1)
        self._actor[env_ids] = actor_frame[:, None, :].expand(-1, HISTORY_LENGTH, -1)
        self._critic[env_ids] = critic_frame[:, None, :].expand(-1, HISTORY_LENGTH, -1)

    def append(
        self,
        proprioception: torch.Tensor,
        command: torch.Tensor,
        uncorrupted_proprioception: torch.Tensor,
        recovery_reference: torch.Tensor,
    ) -> ObservationBatch:
        actor_frame, critic_frame = self._frames(
            proprioception, command, uncorrupted_proprioception, recovery_reference
        )
        self._gate = torch.roll(self._gate, shifts=-1, dims=1)
        self._actor = torch.roll(self._actor, shifts=-1, dims=1)
        self._critic = torch.roll(self._critic, shifts=-1, dims=1)
        self._gate[:, -1] = proprioception
        self._actor[:, -1] = actor_frame
        self._critic[:, -1] = critic_frame
        return self.batch()

    def batch(self) -> ObservationBatch:
        result = ObservationBatch(
            actor=self._actor.reshape(self.num_envs, ACTOR_OBS_DIM),
            gate=self._gate.reshape(self.num_envs, GATE_OBS_DIM),
            critic=self._critic.reshape(self.num_envs, CRITIC_OBS_DIM),
        )
        result.validate()
        return result

    @staticmethod
    def _frames(
        proprioception: torch.Tensor,
        command: torch.Tensor,
        uncorrupted_proprioception: torch.Tensor,
        recovery_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = proprioception.shape[0]
        _assert_matrix("proprioception", proprioception, batch, GATE_FRAME_DIM)
        _assert_matrix("command", command, batch, COMMAND_DIM)
        _assert_matrix("uncorrupted_proprioception", uncorrupted_proprioception, batch, GATE_FRAME_DIM)
        _assert_matrix("recovery_reference", recovery_reference, batch, RECOVERY_REFERENCE_DIM)
        actor = torch.cat((proprioception, command), dim=-1)
        privileged = torch.cat((uncorrupted_proprioception, recovery_reference), dim=-1)
        critic = torch.cat((actor, privileged), dim=-1)
        _assert_matrix("actor frame", actor, batch, ACTOR_FRAME_DIM)
        _assert_matrix("critic frame", critic, batch, CRITIC_FRAME_DIM)
        return actor, critic
