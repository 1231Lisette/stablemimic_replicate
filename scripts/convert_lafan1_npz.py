#!/usr/bin/env python3
"""Convert corrected LAFAN1 G1 CSVs to local BeyondMimic-style NPZ files."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--input-fps", type=float, default=30.0)
parser.add_argument("--output-fps", type=float, default=50.0)
parser.add_argument("--pattern", action="append", default=[])
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--max-files", type=int, default=None, help="Optional conversion smoke limit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


input_dir = Path(args_cli.input_dir).expanduser().resolve()
output_dir = Path(args_cli.output_dir).expanduser().resolve()
patterns = tuple(args_cli.pattern or ("dance*.csv", "fallAndGetUp*.csv"))
sources = tuple(sorted({path for pattern in patterns for path in input_dir.glob(pattern)}))
if not input_dir.is_dir():
    parser.error(f"input directory does not exist: {input_dir}")
if not sources:
    parser.error(f"no CSV files matched {patterns} under {input_dir}")
if args_cli.max_files is not None:
    if args_cli.max_files <= 0:
        parser.error("--max-files must be positive")
    sources = sources[: args_cli.max_files]
if args_cli.input_fps <= 0.0 or args_cli.output_fps <= 0.0:
    parser.error("FPS values must be positive")

faulthandler.enable()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab_assets import G1_29DOF_CFG

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES, load_lafan1_csv
from stablemimic.motion.npz import resample_motion, validate_standard_npz
from stablemimic.sim import build_lafan1_g1_joint_mapping, close_simulation_app


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def convert_one(
    source: Path,
    destination: Path,
    robot: Articulation,
    sim: SimulationContext,
    joint_ids: list[int],
) -> dict[str, object]:
    motion = load_lafan1_csv(source, fps=args_cli.input_fps)
    sampled = resample_motion(motion, args_cli.output_fps)
    body_pos: list[np.ndarray] = []
    body_quat: list[np.ndarray] = []
    body_lin_vel: list[np.ndarray] = []
    body_ang_vel: list[np.ndarray] = []
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    for index in range(sampled.num_frames):
        root_state = robot.data.default_root_state.clone()
        root_state[0, :3] = torch.as_tensor(sampled.root_pos[index], device=sim.device)
        root_state[0, 3:7] = torch.as_tensor(sampled.root_quat_wxyz[index], device=sim.device)
        root_state[0, 7:10] = torch.as_tensor(sampled.root_lin_vel_w[index], device=sim.device)
        root_state[0, 10:13] = torch.as_tensor(sampled.root_ang_vel_w[index], device=sim.device)
        joint_pos = default_joint_pos.clone()
        joint_vel = default_joint_vel.clone()
        joint_pos[0, joint_ids] = torch.as_tensor(sampled.joint_pos[index], device=sim.device)
        joint_vel[0, joint_ids] = torch.as_tensor(sampled.joint_vel[index], device=sim.device)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        robot.update(sim.get_physics_dt())
        body_pos.append(robot.data.body_pos_w[0].detach().cpu().numpy().copy())
        body_quat.append(robot.data.body_quat_w[0].detach().cpu().numpy().copy())
        body_lin_vel.append(robot.data.body_lin_vel_w[0].detach().cpu().numpy().copy())
        body_ang_vel.append(robot.data.body_ang_vel_w[0].detach().cpu().numpy().copy())
    arrays = {
        "fps": np.asarray([args_cli.output_fps], dtype=np.float32),
        "joint_pos": sampled.joint_pos,
        "joint_vel": sampled.joint_vel,
        "body_pos_w": np.stack(body_pos).astype(np.float32),
        "body_quat_w": np.stack(body_quat).astype(np.float32),
        "body_lin_vel_w": np.stack(body_lin_vel).astype(np.float32),
        "body_ang_vel_w": np.stack(body_ang_vel).astype(np.float32),
        # Extra self-describing fields are ignored by standard loaders.
        "joint_names": np.asarray(LAFAN1_G1_JOINT_NAMES),
        "body_names": np.asarray(robot.body_names),
        "source_root_pos": sampled.root_pos,
        "source_root_quat_wxyz": sampled.root_quat_wxyz,
        "sample_times": sampled.times,
    }
    frame_count = validate_standard_npz(arrays)
    pelvis_error = float(np.max(np.abs(arrays["body_pos_w"][:, 0] - sampled.root_pos)))
    if pelvis_error > 2.0e-4:
        raise ValueError(
            f"Isaac pelvis/root FK parity failed for {source.name}: max error {pelvis_error}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(destination, arrays)
    return {
        "source": str(source),
        "output": str(destination),
        "source_sha256": sha256(source),
        "output_sha256": sha256(destination),
        "input_frames": motion.num_frames,
        "output_frames": frame_count,
        "duration_seconds": motion.duration,
        "body_count": int(arrays["body_pos_w"].shape[1]),
        "pelvis_root_max_abs_error_m": pelvis_error,
        "output_bytes": destination.stat().st_size,
    }


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / args_cli.output_fps, device=args_cli.device)
    )
    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(robot_cfg)
    sim.reset()
    mapping = build_lafan1_g1_joint_mapping(robot.joint_names)
    joint_ids = list(mapping.csv_to_sim)
    records = []
    for index, source in enumerate(sources, start=1):
        destination = output_dir / f"{source.stem}.npz"
        if destination.exists() and not args_cli.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {destination}")
        print(f"[CONVERT] {index}/{len(sources)} {source.name}", flush=True)
        records.append(convert_one(source, destination, robot, sim, joint_ids))
        print(
            f"[PASS] {destination.name}: {records[-1]['output_frames']} frames, "
            f"{records[-1]['output_bytes']} bytes",
            flush=True,
        )
    manifest = {
        "schema": "beyondmimic-motion-npz-v1",
        "generator": "stablemimic/scripts/convert_lafan1_npz.py",
        "robot_asset": "isaaclab_assets.G1_29DOF_CFG",
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "input_fps": args_cli.input_fps,
        "output_fps": args_cli.output_fps,
        "standard_fields": [
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        ],
        "quaternion_convention": {
            "source_csv_root": "xyzw",
            "npz_body_quat_w": "wxyz",
            "npz_source_root_quat_wxyz": "wxyz",
        },
        "joint_names": list(LAFAN1_G1_JOINT_NAMES),
        "body_names": list(robot.body_names),
        "files": records,
    }
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"[PASS] Wrote {len(records)} NPZ files and {manifest_path}", flush=True)


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
