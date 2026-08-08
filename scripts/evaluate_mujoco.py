#!/usr/bin/env python3
"""Evaluate a checkpoint in a user-supplied, joint-compatible G1 MuJoCo MJCF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from stablemimic.config import load_config
from stablemimic.core.geometry import projected_gravity_from_xyzw, wxyz_to_xyzw, xyzw_to_wxyz
from stablemimic.core.observations import (
    ObservationHistory,
    build_motion_command,
    build_proprioception,
)
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic
from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES, load_lafan1_csv
from stablemimic.motion.torch_library import TorchMotionLibrary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/stablemimic_g1.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mjcf", required=True)
parser.add_argument("--motion", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max-policy-steps", type=int, default=None)
args = parser.parse_args()


def _name(model: mujoco.MjModel, object_type, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    return value or f"unnamed_{index}"


def main() -> None:
    config = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser().resolve()))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError(f"MJCF must contain exactly one free root joint, got {len(free_joints)}")
    root_joint = int(free_joints[0])
    root_qpos = int(model.jnt_qposadr[root_joint])
    root_dof = int(model.jnt_dofadr[root_joint])
    root_body = int(model.jnt_bodyid[root_joint])

    joint_ids = {
        _name(model, mujoco.mjtObj.mjOBJ_JOINT, index): index
        for index in range(model.njnt)
    }
    missing = sorted(set(LAFAN1_G1_JOINT_NAMES) - set(joint_ids))
    if missing:
        raise ValueError(f"MJCF is missing required LAFAN1 joints: {missing}")
    body_joint_ids = [joint_ids[name] for name in LAFAN1_G1_JOINT_NAMES]
    qpos_addresses = np.array([model.jnt_qposadr[index] for index in body_joint_ids])
    dof_addresses = np.array([model.jnt_dofadr[index] for index in body_joint_ids])
    actuator_by_joint = {int(model.actuator_trnid[index, 0]): index for index in range(model.nu)}
    missing_actuators = [LAFAN1_G1_JOINT_NAMES[i] for i, joint in enumerate(body_joint_ids) if joint not in actuator_by_joint]
    if missing_actuators:
        raise ValueError(f"MJCF lacks position actuators for: {missing_actuators}")
    actuator_ids = np.array([actuator_by_joint[joint] for joint in body_joint_ids])

    reference = load_lafan1_csv(args.motion)
    library = TorchMotionLibrary((reference,), "cpu")
    motion_ids = torch.zeros(1, dtype=torch.long)
    sample = library.sample(motion_ids, torch.zeros(1))
    data.qpos[root_qpos : root_qpos + 3] = sample.root_pos[0].numpy()
    data.qpos[root_qpos + 3 : root_qpos + 7] = xyzw_to_wxyz(sample.root_quat_xyzw)[0].numpy()
    data.qpos[qpos_addresses] = sample.joint_pos[0].numpy()
    data.qvel[root_dof : root_dof + 3] = sample.root_lin_vel_world[0].numpy()
    data.qvel[root_dof + 3 : root_dof + 6] = sample.root_ang_vel_world[0].numpy()
    data.qvel[dof_addresses] = sample.joint_vel[0].numpy()
    mujoco.mj_forward(model, data)

    agent = StableMimicAgent(
        StableMimicActor(
            config.model.expert_hidden_dims, config.model.gate_hidden_dims,
            config.model.activation, config.model.initial_std,
        ),
        StableMimicCritic(config.model.critic_hidden_dims, config.model.activation),
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload["agent"])
    agent.eval()
    history = ObservationHistory(1, "cpu")
    previous_action = torch.zeros(1, 29)
    default_joint_position = torch.as_tensor(model.qpos0[qpos_addresses], dtype=torch.float32)[None]
    time_seconds = 0.0
    max_steps = args.max_policy_steps or int(math.ceil(reference.duration / 0.02))
    root_errors, joint_errors, gate_weights = [], [], []

    for step in range(max_steps):
        current = library.sample(motion_ids, torch.tensor([time_seconds]))
        future_1 = library.sample(motion_ids, torch.tensor([min(time_seconds + 0.10, reference.duration)]))
        future_2 = library.sample(motion_ids, torch.tensor([min(time_seconds + 0.20, reference.duration)]))
        root_quaternion = torch.as_tensor(
            data.qpos[root_qpos + 3 : root_qpos + 7], dtype=torch.float32
        )[None]
        root_xyzw = wxyz_to_xyzw(root_quaternion)
        proprio = build_proprioception(
            torch.as_tensor(data.qvel[root_dof + 3 : root_dof + 6], dtype=torch.float32)[None],
            projected_gravity_from_xyzw(root_xyzw),
            torch.as_tensor(data.qpos[qpos_addresses], dtype=torch.float32)[None] - default_joint_position,
            torch.as_tensor(data.qvel[dof_addresses], dtype=torch.float32)[None],
            previous_action,
        )
        phase_angle = 2.0 * math.pi * current.normalized_phase
        command = build_motion_command(
            current.joint_pos, current.joint_vel, current.root_lin_vel_world,
            current.root_ang_vel_world, current.root_pos[:, 2:3],
            projected_gravity_from_xyzw(current.root_quat_xyzw), future_1.joint_pos,
            future_2.joint_pos, torch.cat((torch.sin(phase_angle), torch.cos(phase_angle)), -1),
        )
        hidden = torch.zeros(1, 43)
        if step == 0:
            history.reset(torch.tensor([0]), proprio, command, proprio, hidden)
        observations = history.append(proprio, command, proprio, hidden)
        actor_obs, gate_obs, _ = agent.normalized(observations.actor, observations.gate, observations.critic)
        with torch.no_grad():
            action, _, _, policy = agent.actor.act(actor_obs, gate_obs, deterministic=True)
        target = default_joint_position + config.environment.action_scale * action
        data.ctrl[actuator_ids] = target[0].numpy()
        for _ in range(4):
            mujoco.mj_step(model, data)
        previous_action = action
        root_errors.append(float(np.linalg.norm(data.qpos[root_qpos : root_qpos + 3] - current.root_pos[0].numpy())))
        joint_errors.append(float(np.sqrt(np.mean(np.square(data.qpos[qpos_addresses] - current.joint_pos[0].numpy())))))
        gate_weights.append(policy.gate_weights[0].numpy())
        time_seconds = min(time_seconds + 0.02, reference.duration)
        if time_seconds >= reference.duration:
            break

    output = {
        "mjcf": str(Path(args.mjcf).resolve()),
        "motion": reference.name,
        "physics_dt": model.opt.timestep,
        "policy_hz": 50.0,
        "steps": len(root_errors),
        "mean_root_position_error": float(np.mean(root_errors)),
        "mean_joint_rms_error": float(np.mean(joint_errors)),
        "mean_gate_weights": np.mean(gate_weights, axis=0).tolist(),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] MuJoCo evaluation: {output_path}")


if __name__ == "__main__":
    main()
