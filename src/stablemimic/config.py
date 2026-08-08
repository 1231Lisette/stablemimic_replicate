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
    action_rate_penalty: float = -0.01
    torque_penalty: float = -2.0e-5
    power_penalty: float = -1.0e-5
    joint_limit_penalty: float = -1.0


@dataclass(frozen=True)
class EnvironmentCfg:
    num_envs: int = 4096
    episode_length_s: float = 20.0
    physics_dt: float = 0.005
    decimation: int = 4
    action_scale: float = 0.5
    tracking_reset_probability: float = 0.5
    transition_duration_s: float = 1.5
    recovery_error_timeout_s: float = 2.0
    recovery_success_threshold: float = 0.82
    observation_noise_std: float = 0.0


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


@dataclass(frozen=True)
class StableMimicCfg:
    seed: int
    data_root: Path
    output_root: Path
    environment: EnvironmentCfg
    reward: RewardCfg
    model: ModelCfg
    ppo: PpoCfg
    training: TrainingCfg
    reset_noise: dict[str, tuple[float, float]] = field(default_factory=dict)


def _kernel(value: dict[str, Any]) -> KernelCfg:
    return KernelCfg(weight=float(value["weight"]), sigma=float(value["sigma"]))


def load_config(path: str | Path) -> StableMimicCfg:
    """Load the repository YAML and reject inconsistent control timing."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    env = EnvironmentCfg(**raw["environment"])
    if abs(env.physics_dt * env.decimation - 0.02) > 1.0e-12:
        raise ValueError("StableMimic policy step must be 0.02 s (50 Hz)")
    reward_raw = raw["reward"]
    reward = RewardCfg(
        **{name: _kernel(reward_raw[name]) for name in (
            "root_position", "root_orientation", "body_position",
            "body_orientation", "body_linear_velocity", "body_angular_velocity",
        )},
        **{key: value for key, value in reward_raw.items() if not isinstance(value, dict)},
    )
    model_raw = dict(raw["model"])
    for key in ("expert_hidden_dims", "gate_hidden_dims", "critic_hidden_dims"):
        model_raw[key] = tuple(int(value) for value in model_raw[key])
    reset_noise = {
        key: (float(value[0]), float(value[1])) for key, value in raw["reset_noise"].items()
    }
    return StableMimicCfg(
        seed=int(raw["seed"]),
        data_root=Path(raw["data_root"]),
        output_root=Path(raw["output_root"]),
        environment=env,
        reward=reward,
        model=ModelCfg(**model_raw),
        ppo=PpoCfg(**raw["ppo"]),
        training=TrainingCfg(**raw["training"]),
        reset_noise=reset_noise,
    )
