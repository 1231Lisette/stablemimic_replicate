#!/usr/bin/env python3
"""Train or smoke-test the complete StableMimic G1 policy."""

from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/stablemimic_g1.yaml")
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--run-dir", default=None)
parser.add_argument("--resume", default=None)
parser.add_argument("--mode", choices=("tracking", "recovery", "joint"), default="joint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from stablemimic.config import load_config

repository_config = load_config(args_cli.config)
if args_cli.num_envs is not None and args_cli.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args_cli.iterations is not None and args_cli.iterations <= 0:
    parser.error("--iterations must be positive")

faulthandler.enable()
print("[STAGE] Launching Isaac Sim...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from stablemimic.envs.isaac_g1_env import StableMimicG1Env
from stablemimic.envs.isaac_g1_env_cfg import StableMimicG1EnvCfg
from stablemimic.rl import StableMimicRunner
from stablemimic.sim import close_simulation_app


def main() -> None:
    torch.manual_seed(repository_config.seed)
    env_cfg = StableMimicG1EnvCfg()
    env_cfg.stablemimic_config_path = str(Path(args_cli.config).resolve())
    env_cfg.data_root = str(repository_config.data_root)
    env_cfg.scene.num_envs = args_cli.num_envs or repository_config.environment.num_envs
    env_cfg.episode_length_s = repository_config.environment.episode_length_s
    env_cfg.sim.dt = repository_config.environment.physics_dt
    env_cfg.decimation = repository_config.environment.decimation
    env_cfg.action_scale = repository_config.environment.action_scale
    env_cfg.action_clip = repository_config.environment.action_clip
    env_cfg.tracking_reset_probability = {
        "tracking": 1.0,
        "recovery": 0.0,
        "joint": repository_config.environment.tracking_reset_probability,
    }[args_cli.mode]
    env_cfg.transition_duration_s = repository_config.environment.transition_duration_s
    env_cfg.recovery_error_timeout_s = repository_config.environment.recovery_error_timeout_s
    env_cfg.recovery_failure_similarity_threshold = (
        repository_config.environment.recovery_failure_similarity_threshold
    )
    env_cfg.recovery_terminal_similarity_threshold = (
        repository_config.environment.recovery_terminal_similarity_threshold
    )
    env_cfg.recovered_like_height_ratio = (
        repository_config.environment.recovered_like_height_ratio
    )
    env_cfg.tracking_fall_recovery_enabled = (
        repository_config.environment.tracking_fall_recovery_enabled
    )
    env_cfg.tracking_fall_height_threshold = (
        repository_config.environment.tracking_fall_height_threshold
    )
    env_cfg.tracking_fall_tilt_degrees = (
        repository_config.environment.tracking_fall_tilt_degrees
    )
    env_cfg.recovery_match_joint_weight = (
        repository_config.environment.recovery_match_joint_weight
    )
    env_cfg.recovery_match_height_weight = (
        repository_config.environment.recovery_match_height_weight
    )
    env_cfg.recovery_match_gravity_weight = (
        repository_config.environment.recovery_match_gravity_weight
    )
    env_cfg.observation_noise_std = repository_config.environment.observation_noise_std
    env = StableMimicG1Env(env_cfg)
    run_dir = args_cli.run_dir or repository_config.output_root / "stablemimic_g1"
    runner = StableMimicRunner(env, repository_config, run_dir)
    if args_cli.resume:
        runner.load(args_cli.resume)
    iterations = args_cli.iterations or repository_config.training.max_iterations
    print(
        f"[STAGE] Training code ready: mode={args_cli.mode}, envs={env.num_envs}, "
        f"iterations={iterations}, device={env.device}",
        flush=True,
    )
    runner.learn(iterations)
    env.close()
    print("[PASS] Training run completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Isaac's close path may directly force exit 0 and mask the exception.
        os._exit(1)
    else:
        print("[STAGE] Closing Isaac Sim...", flush=True)
        close_simulation_app(simulation_app, timeout_seconds=15.0, forced_exit_code=0)
