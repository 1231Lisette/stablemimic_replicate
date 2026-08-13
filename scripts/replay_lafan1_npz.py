#!/usr/bin/env python3
"""Replay a BeyondMimic-style G1 NPZ by writing reference states kinematically."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True)
parser.add_argument("--rate", type=float, default=1.0)
parser.add_argument("--start-time", type=float, default=0.0)
parser.add_argument("--end-time", type=float, default=None)
parser.add_argument("--loop", action="store_true")
parser.add_argument("--max-steps", type=int, default=None, help="Optional headless smoke limit.")
parser.add_argument("--follow-camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.rate <= 0.0:
    parser.error("--rate must be positive")

faulthandler.enable()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import G1_29DOF_CFG

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES
from stablemimic.motion.npz import load_npz_arrays, validate_standard_npz
from stablemimic.sim import build_lafan1_g1_joint_mapping, close_simulation_app


def main() -> None:
    path = Path(args_cli.file).expanduser().resolve()
    arrays = load_npz_arrays(path)
    frames = validate_standard_npz(arrays)
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    duration = (frames - 1) / fps
    end_time = duration if args_cli.end_time is None else args_cli.end_time
    if not 0.0 <= args_cli.start_time <= end_time <= duration + 1.0e-6:
        raise ValueError(
            f"time window must satisfy 0 <= start <= end <= {duration:.3f}s"
        )
    if "joint_names" in arrays:
        names = tuple(str(name) for name in arrays["joint_names"].tolist())
        if names != LAFAN1_G1_JOINT_NAMES:
            raise ValueError("NPZ joint_names do not match the frozen LAFAN1 G1 order")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / fps, device=args_cli.device))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(robot_cfg)
    sim.reset()
    mapping = list(build_lafan1_g1_joint_mapping(robot.joint_names).csv_to_sim)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    start_frame = min(int(round(args_cli.start_time * fps)), frames - 1)
    end_frame = min(int(round(end_time * fps)), frames - 1)
    frame = start_frame
    displayed = 0
    next_deadline = time.perf_counter()
    print(
        f"[INFO] Kinematic NPZ replay: {path.name}, fps={fps:g}, "
        f"frames={start_frame}..{end_frame}, physics_step=False",
        flush=True,
    )
    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        root_state[0, :3] = torch.as_tensor(arrays["body_pos_w"][frame, 0], device=sim.device)
        root_state[0, 3:7] = torch.as_tensor(arrays["body_quat_w"][frame, 0], device=sim.device)
        root_state[0, 7:10] = torch.as_tensor(
            arrays["body_lin_vel_w"][frame, 0], device=sim.device
        )
        root_state[0, 10:13] = torch.as_tensor(
            arrays["body_ang_vel_w"][frame, 0], device=sim.device
        )
        joint_pos = default_joint_pos.clone()
        joint_vel = default_joint_vel.clone()
        joint_pos[0, mapping] = torch.as_tensor(arrays["joint_pos"][frame], device=sim.device)
        joint_vel[0, mapping] = torch.as_tensor(arrays["joint_vel"][frame], device=sim.device)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        robot.update(sim.get_physics_dt())
        if args_cli.follow_camera:
            root = arrays["body_pos_w"][frame, 0]
            sim.set_camera_view(
                (root + np.asarray([2.8, 2.8, 1.5])).tolist(),
                (root + np.asarray([0.0, 0.0, 0.7])).tolist(),
            )
        displayed += 1
        if args_cli.max_steps is not None and displayed >= args_cli.max_steps:
            break
        frame += 1
        if frame > end_frame:
            if not args_cli.loop:
                break
            frame = start_frame
        next_deadline += 1.0 / (fps * args_cli.rate)
        delay = next_deadline - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
    print(f"[PASS] Displayed {displayed} NPZ frames", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        raise
    finally:
        close_simulation_app(
            simulation_app,
            timeout_seconds=15.0,
            forced_exit_code=exit_code,
        )
