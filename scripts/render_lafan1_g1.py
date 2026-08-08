#!/usr/bin/env python3
"""Render deterministic multi-view stills for LAFAN1 G1 visual validation."""

from __future__ import annotations

import argparse
import faulthandler
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True)
parser.add_argument("--times", required=True, help="Comma-separated timestamps in seconds.")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=640)
parser.add_argument("--warmup-steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.enable_cameras:
    parser.error("Offscreen rendering requires --enable_cameras")


def parse_times(value: str, duration: float) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("--times must contain at least one timestamp")
    # Accept harmless decimal rounding of a displayed sequence endpoint.
    tolerance = 1.0e-3
    for timestamp in result:
        if not math.isfinite(timestamp) or not -tolerance <= timestamp <= duration + tolerance:
            raise ValueError(f"Timestamp {timestamp} is outside [0, {duration}]")
    return [min(max(timestamp, 0.0), duration) for timestamp in result]


# Validate data and user input before launching Isaac. SimulationApp.close() can
# terminate the process directly in this container and must not mask input errors.
from stablemimic.motion.lafan1 import load_lafan1_csv

try:
    requested_motion = load_lafan1_csv(args_cli.file)
    requested_times = parse_times(args_cli.times, requested_motion.duration)
except (OSError, ValueError) as error:
    parser.error(str(error))

faulthandler.enable()
faulthandler.dump_traceback_later(180.0, repeat=False)
print("[STAGE] Launching Isaac Sim with camera rendering...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[STAGE] Isaac Sim application launched.", flush=True)

import numpy as np
import torch
from PIL import Image, ImageDraw
print("[STAGE] Core numeric/image modules imported.", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import G1_29DOF_CFG
print("[STAGE] Isaac Lab simulation modules imported.", flush=True)

from stablemimic.sim import build_lafan1_g1_joint_mapping, close_simulation_app
print(f"[STAGE] StableMimic modules imported; module name={__name__!r}.", flush=True)

VIEW_NAMES = ("front_left", "left_side")


def yaw_from_xyzw(quaternion) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_z(offset: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    x, y, z = offset
    return np.array([cosine * x - sine * y, sine * x + cosine * y, z], dtype=np.float64)


def xyzw_to_wxyz(value) -> list[float]:
    return [float(value[3]), float(value[0]), float(value[1]), float(value[2])]


def create_camera() -> Camera:
    sim_utils.create_prim("/World/View_00", "Xform")
    sim_utils.create_prim("/World/View_01", "Xform")
    cfg = CameraCfg(
        prim_path="/World/View_.*/CameraSensor",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=32.0,
            focus_distance=4.0,
            horizontal_aperture=28.0,
            clipping_range=(0.05, 100.0),
        ),
    )
    return Camera(cfg=cfg)


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    array = tensor[..., :3].detach().cpu().numpy()
    if array.dtype != np.uint8:
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def save_contact_sheet(sequence: str, rows: list[tuple[float, float, list[Image.Image]]], output: Path) -> None:
    label_height = 34
    title_height = 44
    sheet = Image.new(
        "RGB",
        (args_cli.width * len(VIEW_NAMES), title_height + len(rows) * (args_cli.height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), f"{sequence} | source 30 FPS | sampled/rendered at requested times", fill="black")
    for row_index, (timestamp, source_frame, images) in enumerate(rows):
        top = title_height + row_index * (args_cli.height + label_height)
        draw.text(
            (12, top + 8),
            f"t={timestamp:.3f}s | source_frame={source_frame:.3f}",
            fill="black",
        )
        for column, (view_name, image) in enumerate(zip(VIEW_NAMES, images)):
            x = column * args_cli.width
            draw.text((x + args_cli.width - 110, top + 8), view_name, fill="black")
            sheet.paste(image, (x, top + label_height))
    sheet.save(output)


def main() -> None:
    print(f"[STAGE] Entered main; file={args_cli.file}", flush=True)
    motion = requested_motion
    print(f"[STAGE] Loaded {motion.name} ({motion.root_pos.shape[0]} frames).", flush=True)
    timestamps = requested_times
    output_dir = Path(args_cli.output_dir).expanduser().resolve() / motion.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STAGE] Rendering {motion.name} at {timestamps} into {output_dir}", flush=True)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 50.0, device=args_cli.device))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = G1_29DOF_CFG.copy()
    robot_cfg.prim_path = "/World/G1"
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot = Articulation(robot_cfg)
    camera = create_camera()
    sim.reset()
    mapping = build_lafan1_g1_joint_mapping(robot.joint_names)
    print(f"[INFO] CSV-to-simulator joint permutation: {list(mapping.csv_to_sim)}", flush=True)

    captured_rows: list[tuple[float, float, list[Image.Image]]] = []
    for capture_index, timestamp in enumerate(timestamps):
        sample = motion.sample(timestamp)
        yaw = yaw_from_xyzw(sample.root_quat_xyzw)
        root = np.asarray(sample.root_pos, dtype=np.float64)
        target = root + np.array([0.0, 0.0, 0.75])
        local_offsets = (np.array([2.8, 2.8, 1.8]), np.array([0.0, 3.8, 1.4]))
        camera_positions = np.stack([root + rotate_z(offset, yaw) for offset in local_offsets])
        camera_targets = np.stack([target, target])
        camera.set_world_poses_from_view(
            torch.as_tensor(camera_positions, dtype=torch.float32, device=sim.device),
            torch.as_tensor(camera_targets, dtype=torch.float32, device=sim.device),
        )

        root_pose = torch.tensor(
            [list(sample.root_pos) + xyzw_to_wxyz(sample.root_quat_xyzw)],
            dtype=torch.float32,
            device=sim.device,
        )
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[0, list(mapping.csv_to_sim)] = torch.as_tensor(
            sample.joint_pos, dtype=torch.float32, device=sim.device
        )
        zero_root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
        zero_joint_velocity = torch.zeros_like(robot.data.default_joint_vel)

        for _ in range(args_cli.warmup_steps):
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(zero_root_velocity)
            robot.write_joint_state_to_sim(joint_pos, zero_joint_velocity)
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            camera.update(dt=sim.get_physics_dt())

        images = [image_from_tensor(camera.data.output["rgb"][index]) for index in range(len(VIEW_NAMES))]
        source_frame = timestamp * motion.fps
        for view_name, image in zip(VIEW_NAMES, images):
            image.save(output_dir / f"{capture_index:02d}_t{timestamp:08.3f}_{view_name}.png")
        captured_rows.append((timestamp, source_frame, images))
        print(f"[STAGE] Captured t={timestamp:.3f}s source_frame={source_frame:.3f}", flush=True)

    contact_sheet = output_dir / f"{motion.name}_contact_sheet.png"
    save_contact_sheet(motion.name, captured_rows, contact_sheet)
    print(f"[PASS] Saved contact sheet: {contact_sheet}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        exit_code = 1
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        print("[STAGE] Closing Isaac Sim application...", flush=True)
        close_simulation_app(
            simulation_app,
            timeout_seconds=15.0,
            forced_exit_code=exit_code,
        )
