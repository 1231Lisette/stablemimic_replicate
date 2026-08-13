"""Typed configuration loading for the StableMimic reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KernelCfg:
    weight: float
    sigma: float


@dataclass(frozen=True)
class RewardCfg:
    root_position: KernelCfg
    root_orientation: KernelCfg
    body_position: KernelCfg
    body_orientation: KernelCfg
    body_linear_velocity: KernelCfg
    body_angular_velocity: KernelCfg
    recovery_multiplier: float = 2.5
    success_bonus: float = 1.0
    action_rate_penalty: float = -0.01
    torque_penalty: float = -2.0e-5
    power_penalty: float = -1.0e-5
    joint_limit_penalty: float = -1.0
    recovery_progress_bonus: float = 0.0
    recovery_progress_height_min: float = 0.25
    recovery_progress_height_max: float = 0.70
    recovery_progress_tilt_max_degrees: float = 90.0
    recovery_progress_height_weight: float = 0.6
    recovery_progress_upright_weight: float = 0.4
    tracking_body_names: tuple[str, ...] = ()
    recovery_base_height_weight: float = 0.0
    recovery_base_height_target: float = 0.728
    recovery_upright_weight: float = 0.0
    recovery_double_support_weight: float = 0.0
    recovery_support_force_threshold: float = 2.0
    recovery_support_height_threshold: float = 0.1


@dataclass(frozen=True)
class EnvironmentCfg:
    num_envs: int = 4096
    episode_length_s: float = 20.0
    physics_dt: float = 0.005
    decimation: int = 4
    action_scale: float = 0.5
    action_clip: float = 100.0
    tracking_reset_probability: float = 0.5
    transition_duration_s: float = 1.5
    recovery_error_timeout_s: float = 2.0
    recovery_failure_similarity_threshold: float = 0.05
    recovery_terminal_similarity_threshold: float = 0.70
    tracking_resumption_similarity_threshold: float = 0.70
    recovered_like_height_ratio: float = 0.8
    tracking_fall_recovery_enabled: bool = True
    tracking_fall_height_threshold: float = 0.5
    tracking_fall_tilt_degrees: float = 60.0
    recovery_match_joint_weight: float = 1.0
    recovery_match_height_weight: float = 4.0
    recovery_match_gravity_weight: float = 2.0
    recovery_static_reset_probability: float = 0.25
    recovery_phase_reset_min: float = -1.0
    recovery_phase_reset_max: float = -1.0
    observation_noise_std: float = 0.0


@dataclass(frozen=True)
class RecoverySegmentationCfg:
    enabled: bool = True
    fallen_height_threshold: float = 0.5
    fallen_tilt_degrees: float = 60.0
    upright_height_threshold: float = 0.7
    upright_tilt_degrees: float = 30.0
    hold_time_s: float = 0.5
    maximum_clip_duration_s: float = 20.0
    trim_to_tilted_nadir: bool = True


@dataclass(frozen=True)
class ModelCfg:
    history_length: int = 4
    expert_hidden_dims: tuple[int, ...] = (512, 256, 128)
    gate_hidden_dims: tuple[int, ...] = (256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    initial_std: float = 1.0


@dataclass(frozen=True)
class PpoCfg:
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 1.0e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_loss_coefficient: float = 1.0
    entropy_coefficient: float = 0.05
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    gate_ce_coefficient: float = 0.1
    transition_coefficient: float = 4.0
    consistency_coefficient: float = 0.01
    alignment_coefficient: float = 0.01
    normalize_advantage: bool = True


@dataclass(frozen=True)
class TrainingCfg:
    max_iterations: int = 10_000
    checkpoint_interval: int = 100
    log_interval: int = 10
    fall_recovery_warmup_iterations: int = 100
    fall_recovery_ramp_iterations: int = 100


@dataclass(frozen=True)
class StableMimicCfg:
    seed: int
    data_root: Path
    output_root: Path
    tracking_motion_files: tuple[str, ...]
    recovery_motion_files: tuple[str, ...]
    environment: EnvironmentCfg
    recovery_segmentation: RecoverySegmentationCfg
    reward: RewardCfg
    model: ModelCfg
    ppo: PpoCfg
    training: TrainingCfg
    reset_noise: dict[str, tuple[float, float]] = field(default_factory=dict)


def _kernel(value: dict[str, Any]) -> KernelCfg:
    return KernelCfg(weight=float(value["weight"]), sigma=float(value["sigma"]))


def fall_recovery_curriculum_probability(
    iteration: int, warmup_iterations: int, ramp_iterations: int
) -> float:
    """Return the fall-switch probability for a one-indexed training iteration."""
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    if warmup_iterations < 0 or ramp_iterations < 0:
        raise ValueError("fall-recovery curriculum iterations must be non-negative")
    if iteration <= warmup_iterations:
        return 0.0
    if ramp_iterations == 0:
        return 1.0
    return min((iteration - warmup_iterations) / ramp_iterations, 1.0)


def load_config(path: str | Path) -> StableMimicCfg:
    """Load the repository YAML and reject inconsistent control timing."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    env = EnvironmentCfg(**raw["environment"])
    if abs(env.physics_dt * env.decimation - 0.02) > 1.0e-12:
        raise ValueError("StableMimic policy step must be 0.02 s (50 Hz)")
    if env.action_clip <= 0.0:
        raise ValueError("environment.action_clip must be positive")
    if env.recovered_like_height_ratio <= 0.0:
        raise ValueError("environment.recovered_like_height_ratio must be positive")
    if not 0.0 <= env.recovery_failure_similarity_threshold < 1.0:
        raise ValueError("environment.recovery_failure_similarity_threshold must be in [0, 1)")
    if not 0.0 < env.recovery_terminal_similarity_threshold <= 1.0:
        raise ValueError("environment.recovery_terminal_similarity_threshold must be in (0, 1]")
    if not 0.0 < env.tracking_resumption_similarity_threshold <= 1.0:
        raise ValueError("environment.tracking_resumption_similarity_threshold must be in (0, 1]")
    if env.recovery_failure_similarity_threshold >= env.recovery_terminal_similarity_threshold:
        raise ValueError("recovery failure similarity threshold must be below terminal threshold")
    if env.tracking_fall_height_threshold <= 0.0:
        raise ValueError("environment.tracking_fall_height_threshold must be positive")
    if not 0.0 < env.tracking_fall_tilt_degrees < 180.0:
        raise ValueError("environment.tracking_fall_tilt_degrees must be in (0, 180)")
    if min(
        env.recovery_match_joint_weight,
        env.recovery_match_height_weight,
        env.recovery_match_gravity_weight,
    ) < 0.0:
        raise ValueError("recovery matching weights must be non-negative")
    if not 0.0 <= env.recovery_static_reset_probability <= 1.0:
        raise ValueError("recovery_static_reset_probability must be in [0, 1]")
    phase_range_disabled = env.recovery_phase_reset_min < 0.0 and env.recovery_phase_reset_max < 0.0
    phase_range_valid = (
        0.0 <= env.recovery_phase_reset_min < env.recovery_phase_reset_max <= 1.0
    )
    if not (phase_range_disabled or phase_range_valid):
        raise ValueError("recovery phase reset range must be disabled or satisfy 0 <= min < max <= 1")
    segmentation = RecoverySegmentationCfg(**raw.get("recovery_segmentation", {}))
    if segmentation.fallen_height_threshold <= 0.0:
        raise ValueError("recovery_segmentation.fallen_height_threshold must be positive")
    if segmentation.upright_height_threshold <= segmentation.fallen_height_threshold:
        raise ValueError("upright height threshold must exceed fallen height threshold")
    if not 0.0 < segmentation.upright_tilt_degrees < segmentation.fallen_tilt_degrees < 180.0:
        raise ValueError("recovery segmentation tilt thresholds must satisfy 0 < upright < fallen < 180")
    if segmentation.hold_time_s <= 0.0 or segmentation.maximum_clip_duration_s <= 0.0:
        raise ValueError("recovery segmentation durations must be positive")
    reward_raw = raw["reward"]
    reward = RewardCfg(
        **{name: _kernel(reward_raw[name]) for name in (
            "root_position", "root_orientation", "body_position",
            "body_orientation", "body_linear_velocity", "body_angular_velocity",
        )},
        **{
            key: tuple(value) if key == "tracking_body_names" else value
            for key, value in reward_raw.items() if not isinstance(value, dict)
        },
    )
    if reward.success_bonus < 0.0:
        raise ValueError("reward.success_bonus must be non-negative")
    if reward.recovery_progress_bonus < 0.0:
        raise ValueError("reward.recovery_progress_bonus must be non-negative")
    if reward.recovery_progress_height_max <= reward.recovery_progress_height_min:
        raise ValueError("recovery progress height max must exceed min")
    if not 0.0 < reward.recovery_progress_tilt_max_degrees < 180.0:
        raise ValueError("recovery progress tilt max must be in (0, 180)")
    if min(
        reward.recovery_progress_height_weight,
        reward.recovery_progress_upright_weight,
    ) < 0.0:
        raise ValueError("recovery progress weights must be non-negative")
    if abs(
        reward.recovery_progress_height_weight
        + reward.recovery_progress_upright_weight
        - 1.0
    ) > 1.0e-6:
        raise ValueError("recovery progress weights must sum to one")
    if len(set(reward.tracking_body_names)) != len(reward.tracking_body_names):
        raise ValueError("reward.tracking_body_names contains duplicates")
    if min(
        reward.recovery_base_height_weight,
        reward.recovery_upright_weight,
        reward.recovery_double_support_weight,
    ) < 0.0:
        raise ValueError("recovery shaping weights must be non-negative")
    if reward.recovery_base_height_target <= 0.0:
        raise ValueError("reward.recovery_base_height_target must be positive")
    if reward.recovery_support_force_threshold < 0.0:
        raise ValueError("reward.recovery_support_force_threshold must be non-negative")
    if reward.recovery_support_height_threshold <= 0.0:
        raise ValueError("reward.recovery_support_height_threshold must be positive")
    model_raw = dict(raw["model"])
    for key in ("expert_hidden_dims", "gate_hidden_dims", "critic_hidden_dims"):
        model_raw[key] = tuple(int(value) for value in model_raw[key])
    reset_noise = {
        key: (float(value[0]), float(value[1])) for key, value in raw["reset_noise"].items()
    }
    training = TrainingCfg(**raw["training"])
    if training.fall_recovery_warmup_iterations < 0:
        raise ValueError("training.fall_recovery_warmup_iterations must be non-negative")
    if training.fall_recovery_ramp_iterations < 0:
        raise ValueError("training.fall_recovery_ramp_iterations must be non-negative")
    tracking_motion_files = tuple(str(value) for value in raw.get("tracking_motion_files", ()))
    recovery_motion_files = tuple(str(value) for value in raw.get("recovery_motion_files", ()))
    if bool(tracking_motion_files) != bool(recovery_motion_files):
        raise ValueError("tracking_motion_files and recovery_motion_files must be set together")
    if len(set(tracking_motion_files)) != len(tracking_motion_files):
        raise ValueError("tracking_motion_files contains duplicates")
    if len(set(recovery_motion_files)) != len(recovery_motion_files):
        raise ValueError("recovery_motion_files contains duplicates")
    overlap = set(tracking_motion_files).intersection(recovery_motion_files)
    if overlap:
        raise ValueError(f"tracking/recovery motion file lists overlap: {sorted(overlap)}")
    return StableMimicCfg(
        seed=int(raw["seed"]),
        data_root=Path(raw["data_root"]),
        output_root=Path(raw["output_root"]),
        tracking_motion_files=tracking_motion_files,
        recovery_motion_files=recovery_motion_files,
        environment=env,
        recovery_segmentation=segmentation,
        reward=reward,
        model=ModelCfg(**model_raw),
        ppo=PpoCfg(**raw["ppo"]),
        training=training,
        reset_noise=reset_noise,
    )
