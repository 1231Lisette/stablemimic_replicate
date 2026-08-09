#!/usr/bin/env python3
"""Run deterministic Isaac evaluation for a StableMimic checkpoint."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/stablemimic_g1.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", required=True)
parser.add_argument("--matched-pushes", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from stablemimic.config import load_config

config = load_config(args_cli.config)
if args_cli.num_envs <= 0 or args_cli.steps <= 0:
    parser.error("--num-envs and --steps must be positive")

faulthandler.enable()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from stablemimic.envs.isaac_g1_env import StableMimicG1Env
from stablemimic.envs.isaac_g1_env_cfg import StableMimicG1EnvCfg
from stablemimic.eval import matched_push_protocol
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic
from stablemimic.sim import close_simulation_app


def main() -> None:
    env_cfg = StableMimicG1EnvCfg()
    env_cfg.stablemimic_config_path = str(Path(args_cli.config).resolve())
    env_cfg.data_root = str(config.data_root)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = config.environment.episode_length_s
    env_cfg.sim.dt = config.environment.physics_dt
    env_cfg.decimation = config.environment.decimation
    env_cfg.action_scale = config.environment.action_scale
    env_cfg.action_clip = config.environment.action_clip
    env_cfg.tracking_reset_probability = (
        1.0 if args_cli.matched_pushes else config.environment.tracking_reset_probability
    )
    env_cfg.transition_duration_s = config.environment.transition_duration_s
    env_cfg.recovery_error_timeout_s = config.environment.recovery_error_timeout_s
    env_cfg.recovery_success_threshold = config.environment.recovery_success_threshold
    env_cfg.observation_noise_std = 0.0
    env = StableMimicG1Env(env_cfg)
    if args_cli.matched_pushes and env.num_envs < 100:
        raise ValueError("--matched-pushes requires --num-envs >= 100")
    agent = StableMimicAgent(
        StableMimicActor(
            config.model.expert_hidden_dims, config.model.gate_hidden_dims,
            config.model.activation, config.model.initial_std,
        ),
        StableMimicCritic(config.model.critic_hidden_dims, config.model.activation),
    ).to(env.device)
    payload = torch.load(args_cli.checkpoint, map_location=env.device, weights_only=False)
    agent.load_state_dict(payload["agent"])
    agent.eval()
    observation, _ = env.reset()
    total_reward = torch.zeros(env.num_envs, device=env.device)
    terminations = torch.zeros(env.num_envs, device=env.device)
    gate_sum = torch.zeros(env.num_envs, 2, device=env.device)
    target_sum = torch.zeros(2, device=env.device)
    tracking_reward_sum = torch.zeros((), device=env.device)
    recovery_reward_sum = torch.zeros((), device=env.device)
    tracking_samples = torch.zeros((), device=env.device)
    recovery_samples = torch.zeros((), device=env.device)
    transition_samples = torch.zeros((), device=env.device)
    clipped_action_elements = torch.zeros((), device=env.device)
    unit_action_exceed_elements = torch.zeros((), device=env.device)
    push_start, push_steps = 50, int(round(0.2 / env.step_dt))
    push_forces = torch.zeros(env.num_envs, 3, device=env.device)
    if args_cli.matched_pushes:
        for event in matched_push_protocol():
            push_forces[event.index, 0] = event.direction_xy[0] * event.force_newtons
            push_forces[event.index, 1] = event.direction_xy[1] * event.force_newtons
    for step in range(args_cli.steps):
        if args_cli.matched_pushes and step == push_start:
            env.set_push_forces(push_forces)
        if args_cli.matched_pushes and step == push_start + push_steps:
            env.set_push_forces(torch.zeros_like(push_forces))
        actor, gate, _critic = agent.normalized(
            observation["policy"], env.gate_observations, observation["critic"]
        )
        gate_target = env.gate_targets.clone()
        with torch.no_grad():
            action, _, _, policy = agent.actor.act(actor, gate, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        tracking_mask = gate_target[:, 0] == 1.0
        recovery_mask = gate_target[:, 1] == 1.0
        transition_mask = ~(tracking_mask | recovery_mask)
        total_reward += reward
        terminations += (terminated | truncated).float()
        gate_sum += policy.gate_weights
        target_sum += gate_target.sum(0)
        tracking_reward_sum += reward[tracking_mask].sum()
        recovery_reward_sum += reward[recovery_mask].sum()
        tracking_samples += tracking_mask.sum()
        recovery_samples += recovery_mask.sum()
        transition_samples += transition_mask.sum()
        clipped_action_elements += (action.abs() > config.environment.action_clip).sum()
        unit_action_exceed_elements += (action.abs() > 1.0).sum()
    metrics = {
        "steps": args_cli.steps,
        "num_envs": env.num_envs,
        "mean_total_reward": float(total_reward.mean()),
        "mean_terminations": float(terminations.mean()),
        "mean_tracking_gate_weight": float((gate_sum[:, 0] / args_cli.steps).mean()),
        "mean_recovery_gate_weight": float((gate_sum[:, 1] / args_cli.steps).mean()),
        "mean_tracking_gate_target": float(target_sum[0] / (args_cli.steps * env.num_envs)),
        "mean_recovery_gate_target": float(target_sum[1] / (args_cli.steps * env.num_envs)),
        "tracking_mean_reward": float(tracking_reward_sum / tracking_samples.clamp_min(1.0)),
        "recovery_mean_reward": float(recovery_reward_sum / recovery_samples.clamp_min(1.0)),
        "tracking_sample_fraction": float(tracking_samples / (args_cli.steps * env.num_envs)),
        "recovery_sample_fraction": float(recovery_samples / (args_cli.steps * env.num_envs)),
        "transition_sample_fraction": float(transition_samples / (args_cli.steps * env.num_envs)),
        "action_clip_fraction": float(
            clipped_action_elements / (args_cli.steps * env.num_envs * 29)
        ),
        "unit_action_exceed_fraction": float(
            unit_action_exceed_elements / (args_cli.steps * env.num_envs * 29)
        ),
        "matched_push_protocol": bool(args_cli.matched_pushes),
        "push_force_range_newtons": [525.0, 575.0] if args_cli.matched_pushes else None,
        "push_duration_seconds": 0.2 if args_cli.matched_pushes else None,
    }
    output = Path(args_cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] Evaluation metrics: {output}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    else:
        close_simulation_app(simulation_app, timeout_seconds=15.0, forced_exit_code=0)
