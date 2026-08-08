#!/usr/bin/env python3
"""Play a LAFAN1 reference on Isaac Lab's G1 29-DoF articulation.

This viewer writes reference root/joint states directly for kinematic validation.
It is not a physics-tracking policy and does not train anything.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True, help="Path to one headerless LAFAN1 G1 CSV file.")
parser.add_argument("--control-hz", type=float, default=50.0)
parser.add_argument("--rate", type=float, default=1.0, help="Reference playback speed multiplier.")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--max-steps", type=int, default=None, help="Optional deterministic smoke-test limit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

faulthandler.enable()
faulthandler.dump_traceback_later(120.0, repeat=False)
print("[STAGE] Launching Isaac Sim application...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[STAGE] Isaac Sim application launched.", flush=True)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import G1_29DOF_CFG

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES, load_lafan1_csv
from stablemimic.sim import close_simulation_app


def xyzw_to_wxyz(value) -> list[float]:
    return [float(value[3]), float(value[0]), float(value[1]), float(value[2])]


def validate_joint_mapping(robot: Articulation) -> list[int]:
    simulator_names = list(robot.joint_names)
    expected = list(LAFAN1_G1_JOINT_NAMES)
    missing = sorted(set(expected) - set(simulator_names))
    extra = sorted(set(simulator_names) - set(expected))
    duplicates = sorted(name for name in expected if simulator_names.count(name) != 1)
    unsupported_extra = sorted(
        name for name in extra if not (name.startswith("left_hand_") or name.startswith("right_hand_"))
    )
    if missing or duplicates or unsupported_extra:
        raise RuntimeError(
            "Isaac Lab G1 asset does not contain an unambiguous LAFAN1 body-joint subset. "
            f"missing={missing}, duplicates={duplicates}, unsupported_extra={unsupported_extra}, "
            f"simulator_count={len(simulator_names)}"
        )
    mapping = [simulator_names.index(name) for name in expected]
    print(f"[INFO] LAFAN1 29-joint subset validated in {len(simulator_names)}-joint Isaac articulation.")
    print(f"[INFO] Extra joints remain at their default state: {extra}")
    print(f"[INFO] CSV-to-simulator joint permutation: {mapping}")
    return mapping


def main() -> None:
    if args_cli.control_hz <= 0.0 or args_cli.rate <= 0.0:
        raise ValueError("--control-hz and --rate must be positive.")
    motion = load_lafan1_csv(args_cli.file)
    print(f"[STAGE] Loaded motion {motion.name} with {motion.num_frames} frames.", flush=True)
    sim_dt = 1.0 / args_cli.control_hz
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device))
    print("[STAGE] SimulationContext created.", flush=True)
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot = Articulation(robot_cfg)
    print("[STAGE] G1 articulation requested; entering sim.reset().", flush=True)
    sim.reset()
    print("[STAGE] sim.reset() completed.", flush=True)
    mapping = validate_joint_mapping(robot)
    robot.reset()

    step = 0
    while simulation_app.is_running():
        reference_time = step * sim_dt * args_cli.rate
        if not args_cli.loop and reference_time > motion.duration:
            break
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            break
        sample = motion.sample(reference_time, loop=args_cli.loop)

        root_pose = torch.tensor(
            [list(sample.root_pos) + xyzw_to_wxyz(sample.root_quat_xyzw)],
            dtype=torch.float32,
            device=sim.device,
        )
        root_velocity = torch.tensor(
            [list(sample.root_lin_vel_world) + list(sample.root_ang_vel_world)],
            dtype=torch.float32,
            device=sim.device,
        )
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        source_joint_pos = torch.as_tensor(sample.joint_pos, dtype=torch.float32, device=sim.device)
        source_joint_vel = torch.as_tensor(sample.joint_vel, dtype=torch.float32, device=sim.device)
        joint_pos[0, mapping] = source_joint_pos
        joint_vel[0, mapping] = source_joint_vel

        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_velocity)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        step += 1
        print(f"[STAGE] Completed simulation step {step}.", flush=True)

    print(
        f"[PASS] Playback complete: motion={motion.name}, steps={step}, "
        f"control_hz={args_cli.control_hz}, source_fps={motion.fps}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        print("[STAGE] Closing Isaac Sim application...", flush=True)
        close_simulation_app(
            simulation_app,
            timeout_seconds=15.0,
            forced_exit_code=1 if sys.exc_info()[0] is not None else 0,
        )
