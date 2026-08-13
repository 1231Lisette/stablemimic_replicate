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
parser.add_argument(
    "--data-root", default=None,
    help="Optional evaluation-only motion root override; does not modify the training config.",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", required=True)
parser.add_argument("--matched-pushes", action="store_true")
parser.add_argument(
    "--standard-recovery", action="store_true",
    help=("Evaluate one no-push trial per environment from a representative fallen "
          "state in an atomic Recovery clip, with zero reset velocity and no reset noise."),
)
parser.add_argument(
    "--reference-reset-velocity", action="store_true",
    help="With --standard-recovery, retain the source motion velocity as a diagnostic control.",
)
parser.add_argument(
    "--recovery-phase-bin", choices=("early", "middle", "late"), default=None,
    help="Evaluate Recovery resets sampled only from one normalized reference-phase third.",
)
parser.add_argument(
    "--enable-early-termination", action="store_true",
    help="Use training-time fall/failure resets (paper evaluation leaves them disabled).",
)
parser.add_argument(
    "--physical-diagnostics", action="store_true",
    help=("Record support contacts, torque/limit utilization, reward components, and "
          "0.5/1/2-second progress for a Recovery reset diagnostic."),
)
parser.add_argument(
    "--reference-actions", action="store_true",
    help=("Use privileged next-reference joint-position targets instead of the learned "
          "policy, to test PD/contact/retargeting feasibility."),
)
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
from stablemimic.eval import (
    classify_fallen_orientation,
    matched_push_protocol,
    support_body_groups,
)
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic
from stablemimic.sim import close_simulation_app


def main() -> None:
    if args_cli.standard_recovery and args_cli.matched_pushes:
        parser.error("--standard-recovery and --matched-pushes are mutually exclusive")
    if args_cli.recovery_phase_bin and (args_cli.standard_recovery or args_cli.matched_pushes):
        parser.error("--recovery-phase-bin is mutually exclusive with other recovery protocols")
    if args_cli.reference_reset_velocity and not args_cli.standard_recovery:
        parser.error("--reference-reset-velocity requires --standard-recovery")
    if args_cli.physical_diagnostics and not (
        args_cli.standard_recovery or args_cli.recovery_phase_bin
    ):
        parser.error("--physical-diagnostics requires a Recovery reset diagnostic")
    if args_cli.reference_actions and not (
        args_cli.standard_recovery or args_cli.recovery_phase_bin
    ):
        parser.error("--reference-actions requires a Recovery reset diagnostic")
    env_cfg = StableMimicG1EnvCfg()
    env_cfg.stablemimic_config_path = str(Path(args_cli.config).resolve())
    env_cfg.data_root = str(
        Path(args_cli.data_root).expanduser().resolve()
        if args_cli.data_root is not None else config.data_root
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = config.environment.episode_length_s
    env_cfg.sim.dt = config.environment.physics_dt
    env_cfg.decimation = config.environment.decimation
    env_cfg.action_scale = config.environment.action_scale
    env_cfg.action_clip = config.environment.action_clip
    recovery_diagnostic = args_cli.standard_recovery or args_cli.recovery_phase_bin is not None
    env_cfg.tracking_reset_probability = 0.0 if recovery_diagnostic else (
        1.0 if args_cli.matched_pushes else config.environment.tracking_reset_probability
    )
    env_cfg.transition_duration_s = config.environment.transition_duration_s
    env_cfg.recovery_error_timeout_s = config.environment.recovery_error_timeout_s
    env_cfg.recovery_failure_similarity_threshold = (
        config.environment.recovery_failure_similarity_threshold
    )
    env_cfg.recovery_terminal_similarity_threshold = (
        config.environment.recovery_terminal_similarity_threshold
    )
    env_cfg.recovered_like_height_ratio = config.environment.recovered_like_height_ratio
    env_cfg.tracking_fall_recovery_enabled = config.environment.tracking_fall_recovery_enabled
    env_cfg.tracking_fall_height_threshold = config.environment.tracking_fall_height_threshold
    env_cfg.tracking_fall_tilt_degrees = config.environment.tracking_fall_tilt_degrees
    env_cfg.recovery_match_joint_weight = config.environment.recovery_match_joint_weight
    env_cfg.recovery_match_height_weight = config.environment.recovery_match_height_weight
    env_cfg.recovery_match_gravity_weight = config.environment.recovery_match_gravity_weight
    env_cfg.fall_recovery_probability = 1.0
    env_cfg.observation_noise_std = 0.0
    env_cfg.enable_early_termination = args_cli.enable_early_termination
    env_cfg.recovery_reset_at_fallen_state = args_cli.standard_recovery
    env_cfg.recovery_reset_zero_velocity = (
        args_cli.standard_recovery and not args_cli.reference_reset_velocity
    )
    env_cfg.reset_noise_enabled = not args_cli.standard_recovery
    env_cfg.enable_physical_diagnostics = args_cli.physical_diagnostics
    env_cfg.robot.spawn.activate_contact_sensors = args_cli.physical_diagnostics
    if args_cli.recovery_phase_bin:
        phase_ranges = {
            "early": (0.0, 1.0 / 3.0),
            "middle": (1.0 / 3.0, 2.0 / 3.0),
            "late": (2.0 / 3.0, 1.0),
        }
        env_cfg.recovery_phase_reset_min, env_cfg.recovery_phase_reset_max = phase_ranges[
            args_cli.recovery_phase_bin
        ]
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
    standard_recovery_labels = None
    standard_terminal_success = None
    standard_physical_success = None
    standard_upright_hold_steps = None
    standard_max_upright_hold_steps = None
    standard_max_height = None
    standard_min_tilt = None
    if recovery_diagnostic:
        standard_recovery_labels = classify_fallen_orientation(
            env.recovery_evaluation_state()["projected_gravity"].clone()
        )
        standard_terminal_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        standard_physical_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        standard_upright_hold_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        standard_max_upright_hold_steps = torch.zeros_like(standard_upright_hold_steps)
        initial_state = env.recovery_evaluation_state()
        standard_max_height = initial_state["root_height"].clone()
        standard_min_tilt = initial_state["root_tilt_radians"].clone()
    physical_initial_height = None
    physical_initial_tilt = None
    physical_progress = {}
    physical_contact_names = None
    physical_joint_names = None
    physical_articulation_body_names = None
    physical_articulation_group_indices = None
    physical_group_indices = None
    physical_group_contact_steps = None
    physical_group_max_force = None
    physical_actual_group_near_ground_steps = None
    physical_reference_group_near_ground_steps = None
    physical_torque_saturation_elements = None
    physical_max_torque_utilization = None
    physical_near_limit_elements = None
    physical_target_limit_elements = None
    physical_reward_sums = None
    physical_joint_torque_saturation_steps = None
    physical_joint_max_torque_utilization = None
    physical_joint_near_limit_steps = None
    physical_joint_target_limit_steps = None
    physical_horizon_steps = min(args_cli.steps, int(round(2.0 / env.step_dt)))
    if args_cli.physical_diagnostics:
        initial = env.recovery_evaluation_state()
        physical_initial_height = initial["root_height"].clone()
        physical_initial_tilt = initial["root_tilt_radians"].clone()
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
    ever_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    recovery_eligible = torch.zeros_like(ever_fallen)
    recovered_after_fall = torch.zeros_like(ever_fallen)
    recovery_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fall_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    resume_step = torch.full_like(fall_step, -1)
    event_counts = {
        name: torch.zeros((), device=env.device)
        for name in (
            "recovery_success",
            "recovery_failure",
            "transition_completed",
            "tracking_fall_candidate",
            "tracking_fall_entered_recovery",
            "sequence_termination",
            "unrecoverable_fall_termination",
            "timeout",
        )
    }
    push_start, push_steps = 50, int(round(0.2 / env.step_dt))
    fall_tilt_radians = torch.deg2rad(torch.tensor(60.0, device=env.device))
    resume_tilt_radians = torch.deg2rad(torch.tensor(30.0, device=env.device))
    required_resume_steps = int(round(0.5 / env.step_dt))
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
            policy_action, _, _, policy = agent.actor.act(actor, gate, deterministic=True)
        action = (
            env.next_reference_action_diagnostic()
            if args_cli.reference_actions
            else policy_action
        )
        observation, reward, terminated, truncated, _ = env.step(action)
        if args_cli.physical_diagnostics and step < physical_horizon_steps:
            physical = env.physical_diagnostic_state()
            contact_names = physical["contact_body_names"]
            if physical_contact_names is None:
                physical_contact_names = contact_names
                physical_joint_names = physical["controlled_joint_names"]
                physical_articulation_body_names = physical["articulation_body_names"]
                physical_articulation_group_indices = support_body_groups(
                    physical_articulation_body_names
                )
                physical_group_indices = support_body_groups(contact_names)
                physical_group_contact_steps = {
                    name: torch.zeros(env.num_envs, device=env.device)
                    for name in physical_group_indices
                }
                physical_group_max_force = {
                    name: torch.zeros(env.num_envs, device=env.device)
                    for name in physical_group_indices
                }
                physical_actual_group_near_ground_steps = {
                    name: torch.zeros(env.num_envs, device=env.device)
                    for name in physical_articulation_group_indices
                }
                physical_reference_group_near_ground_steps = {
                    name: torch.zeros(env.num_envs, device=env.device)
                    for name in physical_articulation_group_indices
                }
                physical_torque_saturation_elements = torch.zeros(
                    env.num_envs, device=env.device
                )
                physical_max_torque_utilization = torch.zeros(
                    env.num_envs, device=env.device
                )
                physical_near_limit_elements = torch.zeros(env.num_envs, device=env.device)
                physical_target_limit_elements = torch.zeros(env.num_envs, device=env.device)
                physical_reward_sums = {
                    key.removeprefix("reward_"): torch.zeros(env.num_envs, device=env.device)
                    for key in physical if key.startswith("reward_")
                }
                physical_joint_torque_saturation_steps = torch.zeros(
                    env.num_envs, 29, device=env.device
                )
                physical_joint_max_torque_utilization = torch.zeros(
                    env.num_envs, 29, device=env.device
                )
                physical_joint_near_limit_steps = torch.zeros(
                    env.num_envs, 29, device=env.device
                )
                physical_joint_target_limit_steps = torch.zeros(
                    env.num_envs, 29, device=env.device
                )
            forces = physical["contact_force_magnitudes"]
            for group_name, body_ids in physical_group_indices.items():
                if body_ids:
                    group_force = forces[:, body_ids].amax(dim=1)
                    physical_group_contact_steps[group_name] += (group_force >= 5.0).float()
                    physical_group_max_force[group_name] = torch.maximum(
                        physical_group_max_force[group_name], group_force
                    )
            for group_name, body_ids in physical_articulation_group_indices.items():
                if body_ids:
                    actual_near = (
                        physical["body_link_origin_heights"][:, body_ids].amin(dim=1) <= 0.12
                    )
                    reference_near = (
                        physical["reference_body_link_origin_heights"][:, body_ids].amin(dim=1)
                        <= 0.12
                    )
                    physical_actual_group_near_ground_steps[group_name] += actual_near.float()
                    physical_reference_group_near_ground_steps[group_name] += reference_near.float()
            torque_utilization = physical["torque_utilization"]
            physical_torque_saturation_elements += (torque_utilization >= 0.95).sum(dim=1)
            physical_joint_torque_saturation_steps += (torque_utilization >= 0.95).float()
            physical_joint_max_torque_utilization = torch.maximum(
                physical_joint_max_torque_utilization, torque_utilization
            )
            physical_max_torque_utilization = torch.maximum(
                physical_max_torque_utilization, torque_utilization.amax(dim=1)
            )
            physical_near_limit_elements += physical["near_soft_joint_limit"].sum(dim=1)
            physical_target_limit_elements += physical["target_outside_soft_joint_limit"].sum(dim=1)
            physical_joint_near_limit_steps += physical["near_soft_joint_limit"].float()
            physical_joint_target_limit_steps += physical["target_outside_soft_joint_limit"].float()
            for reward_name in physical_reward_sums:
                physical_reward_sums[reward_name] += physical[f"reward_{reward_name}"]
            elapsed = round((step + 1) * env.step_dt, 6)
            if elapsed in (0.5, 1.0, 2.0):
                state = env.recovery_evaluation_state()
                physical_progress[f"{elapsed:.1f}s"] = {
                    "height_gain": state["root_height"].clone() - physical_initial_height,
                    "tilt_reduction_degrees": torch.rad2deg(
                        physical_initial_tilt - state["root_tilt_radians"]
                    ),
                    "reference_height_gain": (
                        physical["reference_root_height"] - physical_initial_height
                    ).clone(),
                }
        if standard_terminal_success is not None:
            standard_terminal_success |= env.latest_events["recovery_success"]
        recovery_state = env.recovery_evaluation_state()
        if standard_physical_success is not None and standard_upright_hold_steps is not None:
            physically_upright = (
                (recovery_state["root_height"] >= config.recovery_segmentation.upright_height_threshold)
                & (
                    recovery_state["root_tilt_radians"]
                    <= torch.deg2rad(torch.tensor(
                        config.recovery_segmentation.upright_tilt_degrees,
                        device=env.device,
                    ))
                )
            )
            standard_upright_hold_steps = torch.where(
                physically_upright,
                standard_upright_hold_steps + 1,
                torch.zeros_like(standard_upright_hold_steps),
            )
            standard_max_upright_hold_steps = torch.maximum(
                standard_max_upright_hold_steps, standard_upright_hold_steps
            )
            standard_max_height = torch.maximum(
                standard_max_height, recovery_state["root_height"]
            )
            standard_min_tilt = torch.minimum(
                standard_min_tilt, recovery_state["root_tilt_radians"]
            )
            standard_physical_success |= standard_upright_hold_steps >= int(round(
                config.recovery_segmentation.hold_time_s / env.step_dt
            ))
        fallen_now = (
            (recovery_state["root_height"] < 0.5)
            | (recovery_state["root_tilt_radians"] > fall_tilt_radians)
        )
        episode_ended = terminated | truncated
        new_fall = fallen_now & ~ever_fallen & ~episode_ended
        fall_step[new_fall] = step
        ever_fallen |= fallen_now
        recovery_eligible |= new_fall
        recovery_eligible &= ~episode_ended
        # The paper publishes the fall threshold but not its exact tracking-resumption
        # threshold. This explicit reproduction criterion requires 0.5 s of upright,
        # command-height, high-similarity behavior routed back to the Tracking Expert.
        resume_candidate = (
            recovery_eligible
            & ~recovered_after_fall
            & (recovery_state["root_height"] >= 0.8 * recovery_state["command_height"])
            & (recovery_state["root_tilt_radians"] <= resume_tilt_radians)
            & (
                recovery_state["similarity"]
                >= config.environment.tracking_resumption_similarity_threshold
            )
            & (policy.gate_weights[:, 0] >= 0.5)
        )
        recovery_hold_steps = torch.where(
            resume_candidate, recovery_hold_steps + 1, torch.zeros_like(recovery_hold_steps)
        )
        newly_recovered = recovery_hold_steps >= required_resume_steps
        recovered_after_fall |= newly_recovered
        resume_step[newly_recovered & (resume_step < 0)] = step
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
        for name in event_counts:
            event_counts[name] += env.latest_events[name].sum()
    valid_resume = resume_step >= 0
    recovery_latency = (resume_step[valid_resume] - fall_step[valid_resume]).float() * env.step_dt
    metrics = {
        "steps": args_cli.steps,
        "num_envs": env.num_envs,
        "controller": "reference_joint_targets" if args_cli.reference_actions else "checkpoint_policy",
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
        "early_termination_enabled": bool(args_cli.enable_early_termination),
        "paper_fall_count": int(ever_fallen.sum()),
        "tracking_resumption_count": int(recovered_after_fall.sum()),
        "tracking_resumption_rate_after_fall": float(
            recovered_after_fall.sum() / ever_fallen.sum().clamp_min(1)
        ),
        "tracking_resumption_rate_per_trial": float(
            recovered_after_fall.sum() / env.num_envs
        ),
        "mean_tracking_resumption_seconds": (
            float(recovery_latency.mean()) if recovery_latency.numel() else None
        ),
        "fall_definition": "root_height<0.5m or root_tilt>60deg (paper)",
        "tracking_resumption_definition": (
            "0.5s sustained: height>=0.8*command_height, tilt<=30deg, "
            "similarity>=tracking_resumption_similarity_threshold, tracking_gate>=0.5 "
            "(reproduction choice)"
        ),
        **{f"{name}_count": int(value) for name, value in event_counts.items()},
        "recovery_successes_per_1000_recovery_steps": float(
            1000.0 * event_counts["recovery_success"] / recovery_samples.clamp_min(1.0)
        ),
        "recovery_failures_per_1000_recovery_steps": float(
            1000.0 * event_counts["recovery_failure"] / recovery_samples.clamp_min(1.0)
        ),
        "matched_push_protocol": bool(args_cli.matched_pushes),
        "push_force_range_newtons": [525.0, 575.0] if args_cli.matched_pushes else None,
        "push_duration_seconds": 0.2 if args_cli.matched_pushes else None,
    }
    if (
        standard_recovery_labels is not None
        and standard_terminal_success is not None
        and standard_physical_success is not None
        and standard_max_upright_hold_steps is not None
        and standard_max_height is not None
        and standard_min_tilt is not None
    ):
        label_names = ("supine", "prone", "left_side", "right_side", "other")
        by_orientation = {}
        for label_id, label_name in enumerate(label_names):
            mask = standard_recovery_labels == label_id
            trials = int(mask.sum())
            successes = int((standard_physical_success & mask).sum())
            terminal_successes = int((standard_terminal_success & mask).sum())
            reached_height = int((mask & (
                standard_max_height >= config.recovery_segmentation.upright_height_threshold
            )).sum())
            reached_tilt = int((mask & (
                standard_min_tilt <= torch.deg2rad(torch.tensor(
                    config.recovery_segmentation.upright_tilt_degrees, device=env.device
                ))
            )).sum())
            transient_upright = int((mask & (standard_max_upright_hold_steps > 0)).sum())
            by_orientation[label_name] = {
                "trials": trials,
                "physical_successes": successes,
                "physical_success_rate": successes / trials if trials else None,
                "terminal_reference_successes": terminal_successes,
                "terminal_reference_success_rate": terminal_successes / trials if trials else None,
                "reached_upright_height": reached_height,
                "reached_upright_tilt": reached_tilt,
                "transiently_upright": transient_upright,
            }
        hold_seconds = standard_max_upright_hold_steps.float() * env.step_dt
        diagnostic = {
            "profile": "static" if args_cli.standard_recovery else args_cli.recovery_phase_bin,
            "definition": (
                "lowest root-height frame satisfying height<=0.5m and tilt>=60deg in "
                "each eligible atomic clip; reset noise disabled, no push; "
                + ("source motion velocity" if args_cli.reference_reset_velocity else "zero velocity")
                if args_cli.standard_recovery
                else "normalized reference phase third; training reset noise and source velocity; no push"
            ),
            "trials": env.num_envs,
            "physical_success_definition": (
                "root height>=0.7m and root tilt<=30deg sustained for 0.5s"
            ),
            "physical_successes": int(standard_physical_success.sum()),
            "physical_success_rate": float(standard_physical_success.float().mean()),
            "terminal_reference_successes": int(standard_terminal_success.sum()),
            "terminal_reference_success_rate": float(standard_terminal_success.float().mean()),
            "reached_upright_height": int((
                standard_max_height >= config.recovery_segmentation.upright_height_threshold
            ).sum()),
            "reached_upright_tilt": int((
                standard_min_tilt <= torch.deg2rad(torch.tensor(
                    config.recovery_segmentation.upright_tilt_degrees, device=env.device
                ))
            ).sum()),
            "transiently_upright": int((standard_max_upright_hold_steps > 0).sum()),
            "median_max_root_height": float(standard_max_height.median()),
            "p90_max_root_height": float(torch.quantile(standard_max_height, 0.9)),
            "median_min_root_tilt_degrees": float(torch.rad2deg(standard_min_tilt).median()),
            "p10_min_root_tilt_degrees": float(torch.quantile(
                torch.rad2deg(standard_min_tilt), 0.1
            )),
            "median_max_upright_hold_seconds": float(hold_seconds.median()),
            "p90_max_upright_hold_seconds": float(torch.quantile(hold_seconds, 0.9)),
            "by_initial_orientation": by_orientation,
        }
        metrics["recovery_reset_diagnostic"] = diagnostic
        if args_cli.standard_recovery:
            metrics["standard_recovery"] = diagnostic
    if args_cli.physical_diagnostics:
        label_names = ("supine", "prone", "left_side", "right_side", "other")
        joint_elements_per_trial = physical_horizon_steps * 29

        def summarize_physics(mask):
            trials = int(mask.sum())
            if not trials:
                return {"trials": 0}
            progress = {}
            for timestamp, values in physical_progress.items():
                height_gain = values["height_gain"][mask]
                tilt_reduction = values["tilt_reduction_degrees"][mask]
                progress[timestamp] = {
                    "median_height_gain_m": float(height_gain.median()),
                    "p90_height_gain_m": float(torch.quantile(height_gain, 0.9)),
                    "positive_height_gain_trials": int((height_gain > 0.05).sum()),
                    "median_tilt_reduction_degrees": float(tilt_reduction.median()),
                    "median_reference_height_gain_m": float(
                        values["reference_height_gain"][mask].median()
                    ),
                }
            contacts = {}
            for group_name in physical_group_indices:
                contact_steps = physical_group_contact_steps[group_name][mask]
                max_force = physical_group_max_force[group_name][mask]
                contacts[group_name] = {
                    "body_names": [
                        physical_contact_names[index]
                        for index in physical_group_indices[group_name]
                    ],
                    "contact_step_fraction": float(
                        contact_steps.sum() / (trials * physical_horizon_steps)
                    ),
                    "trials_with_contact": int((contact_steps > 0).sum()),
                    "median_max_force_n": float(max_force.median()),
                    "p90_max_force_n": float(torch.quantile(max_force, 0.9)),
                    "actual_link_origin_near_ground_step_fraction": float(
                        physical_actual_group_near_ground_steps[group_name][mask].sum()
                        / (trials * physical_horizon_steps)
                    ),
                    "reference_link_origin_near_ground_step_fraction": float(
                        physical_reference_group_near_ground_steps[group_name][mask].sum()
                        / (trials * physical_horizon_steps)
                    ),
                }
            return {
                "trials": trials,
                "progress": progress,
                "contacts": contacts,
                "torque_saturation_element_fraction": float(
                    physical_torque_saturation_elements[mask].sum()
                    / (trials * joint_elements_per_trial)
                ),
                "trials_reaching_torque_limit": int(
                    (physical_max_torque_utilization[mask] >= 0.95).sum()
                ),
                "median_max_torque_utilization": float(
                    physical_max_torque_utilization[mask].median()
                ),
                "p90_max_torque_utilization": float(torch.quantile(
                    physical_max_torque_utilization[mask], 0.9
                )),
                "near_soft_joint_limit_element_fraction": float(
                    physical_near_limit_elements[mask].sum()
                    / (trials * joint_elements_per_trial)
                ),
                "target_outside_soft_limit_element_fraction": float(
                    physical_target_limit_elements[mask].sum()
                    / (trials * joint_elements_per_trial)
                ),
                "mean_reward_components": {
                    name: float(values[mask].sum() / (trials * physical_horizon_steps))
                    for name, values in physical_reward_sums.items()
                },
                "by_joint": {
                    joint_name: {
                        "torque_saturation_step_fraction": float(
                            physical_joint_torque_saturation_steps[mask, joint_id].sum()
                            / (trials * physical_horizon_steps)
                        ),
                        "median_max_torque_utilization": float(
                            physical_joint_max_torque_utilization[mask, joint_id].median()
                        ),
                        "near_soft_limit_step_fraction": float(
                            physical_joint_near_limit_steps[mask, joint_id].sum()
                            / (trials * physical_horizon_steps)
                        ),
                        "target_outside_soft_limit_step_fraction": float(
                            physical_joint_target_limit_steps[mask, joint_id].sum()
                            / (trials * physical_horizon_steps)
                        ),
                    }
                    for joint_id, joint_name in enumerate(physical_joint_names)
                },
            }

        metrics["physical_diagnostics"] = {
            "contact_threshold_newtons": 5.0,
            "torque_limit_threshold": 0.95,
            "near_soft_joint_limit_threshold_fraction": 0.02,
            "link_origin_near_ground_threshold_m": 0.12,
            "diagnostic_horizon_seconds": physical_horizon_steps * env.step_dt,
            "all": summarize_physics(torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device
            )),
            "by_initial_orientation": {
                label_name: summarize_physics(standard_recovery_labels == label_id)
                for label_id, label_name in enumerate(label_names)
            },
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
