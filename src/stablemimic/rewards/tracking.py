"""Six-family whole-body tracking/recovery Gaussian kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stablemimic.config import RewardCfg


@dataclass(frozen=True)
class KinematicState:
    root_position: torch.Tensor
    root_quaternion_xyzw: torch.Tensor
    body_position: torch.Tensor
    body_quaternion_xyzw: torch.Tensor
    body_linear_velocity: torch.Tensor
    body_angular_velocity: torch.Tensor


@dataclass(frozen=True)
class RewardBreakdown:
    total: torch.Tensor
    tracking: torch.Tensor
    root_position: torch.Tensor
    root_orientation: torch.Tensor
    body_position: torch.Tensor
    body_orientation: torch.Tensor
    body_linear_velocity: torch.Tensor
    body_angular_velocity: torch.Tensor
    regularization: torch.Tensor


@dataclass(frozen=True)
class RecoveryShapingBreakdown:
    total: torch.Tensor
    base_height: torch.Tensor
    upright: torch.Tensor
    double_support: torch.Tensor


def _kernel(error: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("kernel sigma must be positive")
    return torch.exp(-error / (sigma * sigma))


def _quaternion_angle_squared(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    dot = (current * target).sum(-1).abs().clamp(0.0, 1.0)
    return torch.square(2.0 * torch.acos(dot))


def recovery_shaping_reward(
    root_height: torch.Tensor,
    projected_gravity: torch.Tensor,
    recovery_mask: torch.Tensor,
    config: RewardCfg,
    *,
    foot_contact_forces: torch.Tensor | None = None,
    foot_heights: torch.Tensor | None = None,
) -> RecoveryShapingBreakdown:
    """HumanUP Stage-I shaping, restricted to StableMimic Recovery samples.

    The public HumanUP code rewards clipped exponential base height, body-up
    orientation, and simultaneous support on both ankle-roll links.  Keeping
    these terms separate from the six tracking kernels makes the attribution
    explicit and leaves them disabled in old configs through zero weights.
    """
    if root_height.ndim != 1 or recovery_mask.shape != root_height.shape:
        raise ValueError("root_height/recovery_mask shape mismatch")
    if projected_gravity.shape != (root_height.shape[0], 3):
        raise ValueError("projected_gravity shape mismatch")
    base_height = config.recovery_base_height_weight * (
        torch.exp(root_height.clamp(0.0, config.recovery_base_height_target)) - 1.0
    )
    upright = config.recovery_upright_weight * torch.exp(-projected_gravity[:, 2])
    double_support = torch.zeros_like(root_height)
    if config.recovery_double_support_weight > 0.0:
        if foot_contact_forces is None or foot_heights is None:
            raise ValueError("double-support reward requires foot contact forces and heights")
        if foot_contact_forces.shape != (root_height.shape[0], 2):
            raise ValueError("foot_contact_forces must have shape (N, 2)")
        if foot_heights.shape != (root_height.shape[0], 2):
            raise ValueError("foot_heights must have shape (N, 2)")
        supported = (
            (foot_contact_forces > config.recovery_support_force_threshold).all(dim=1)
            & (foot_heights < config.recovery_support_height_threshold).all(dim=1)
        )
        double_support = config.recovery_double_support_weight * supported.float()
    base_height = torch.where(recovery_mask, base_height, torch.zeros_like(base_height))
    upright = torch.where(recovery_mask, upright, torch.zeros_like(upright))
    double_support = torch.where(
        recovery_mask, double_support, torch.zeros_like(double_support)
    )
    return RecoveryShapingBreakdown(
        total=base_height + upright + double_support,
        base_height=base_height,
        upright=upright,
        double_support=double_support,
    )


def whole_body_reward(
    current: KinematicState,
    target: KinematicState,
    recovery_mask: torch.Tensor,
    config: RewardCfg,
    *,
    actions: torch.Tensor | None = None,
    previous_actions: torch.Tensor | None = None,
    torques: torch.Tensor | None = None,
    joint_velocities: torch.Tensor | None = None,
    joint_positions: torch.Tensor | None = None,
    soft_joint_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> RewardBreakdown:
    """Compute published error families with configured reproduction weights.

    Recovery substitutes root height for full root position and compares body
    positions relative to the root, so it never pulls the robot to old world XY.
    """
    batch = current.root_position.shape[0]
    if recovery_mask.shape != (batch,):
        raise ValueError("recovery_mask shape mismatch")
    full_root_error = torch.square(current.root_position - target.root_position).sum(-1)
    height_error = torch.square(current.root_position[:, 2] - target.root_position[:, 2])
    root_position_error = torch.where(recovery_mask, height_error, full_root_error)
    root_orientation_error = _quaternion_angle_squared(
        current.root_quaternion_xyzw, target.root_quaternion_xyzw
    )

    full_body_position_error = torch.square(current.body_position - target.body_position).sum(-1).mean(-1)
    current_relative = current.body_position - current.root_position[:, None, :]
    target_relative = target.body_position - target.root_position[:, None, :]
    recovery_body_error = torch.square(current_relative - target_relative).sum(-1).mean(-1)
    body_position_error = torch.where(recovery_mask, recovery_body_error, full_body_position_error)
    body_orientation_error = _quaternion_angle_squared(
        current.body_quaternion_xyzw, target.body_quaternion_xyzw
    ).mean(-1)
    body_linear_velocity_error = torch.square(
        current.body_linear_velocity - target.body_linear_velocity
    ).sum(-1).mean(-1)
    body_angular_velocity_error = torch.square(
        current.body_angular_velocity - target.body_angular_velocity
    ).sum(-1).mean(-1)

    components = {
        "root_position": config.root_position.weight * _kernel(root_position_error, config.root_position.sigma),
        "root_orientation": config.root_orientation.weight * _kernel(root_orientation_error, config.root_orientation.sigma),
        "body_position": config.body_position.weight * _kernel(body_position_error, config.body_position.sigma),
        "body_orientation": config.body_orientation.weight * _kernel(body_orientation_error, config.body_orientation.sigma),
        "body_linear_velocity": config.body_linear_velocity.weight * _kernel(
            body_linear_velocity_error, config.body_linear_velocity.sigma
        ),
        "body_angular_velocity": config.body_angular_velocity.weight * _kernel(
            body_angular_velocity_error, config.body_angular_velocity.sigma
        ),
    }
    tracking = torch.stack(tuple(components.values())).sum(0)
    tracking = torch.where(recovery_mask, tracking * config.recovery_multiplier, tracking)

    regularization = torch.zeros(batch, device=current.root_position.device)
    if actions is not None and previous_actions is not None:
        regularization += config.action_rate_penalty * torch.square(actions - previous_actions).mean(-1)
    if torques is not None:
        regularization += config.torque_penalty * torch.square(torques).mean(-1)
    if torques is not None and joint_velocities is not None:
        regularization += config.power_penalty * torch.abs(torques * joint_velocities).mean(-1)
    if joint_positions is not None and soft_joint_limits is not None:
        lower, upper = soft_joint_limits
        violation = torch.relu(lower - joint_positions) + torch.relu(joint_positions - upper)
        regularization += config.joint_limit_penalty * torch.square(violation).mean(-1)

    return RewardBreakdown(
        total=tracking + regularization,
        tracking=tracking,
        regularization=regularization,
        **components,
    )
