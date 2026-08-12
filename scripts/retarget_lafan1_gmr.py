#!/usr/bin/env python3
"""Headless, reproducible raw LAFAN1 BVH -> GMR Unitree G1 conversion.

This adapter intentionally leaves a frozen GMR checkout untouched.  It works
around the upstream ``bvh_to_robot_dataset.py`` import/source-name mismatch at
revision bb1bbe4 and emits both GMR-compatible pickle files and StableMimic's
36-column CSV contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES


DEFAULT_PATTERNS = ("dance*.bvh", "fallAndGetUp*.bvh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qpos_to_stablemimic_rows(qpos: np.ndarray) -> np.ndarray:
    """Convert MuJoCo free-joint qpos (xyz,wxyz,joints) to xyz,xyzw,joints."""
    values = np.asarray(qpos, dtype=np.float64)
    expected = 7 + len(LAFAN1_G1_JOINT_NAMES)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"Expected qpos shape (N, {expected}), got {values.shape}")
    if values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("GMR qpos must contain at least two finite frames")
    rows = np.empty_like(values)
    rows[:, :3] = values[:, :3]
    rows[:, 3:7] = values[:, [4, 5, 6, 3]]
    rows[:, 7:] = values[:, 7:]
    return rows


def symmetric_rate_limit(values: np.ndarray, max_rate: float, fps: float) -> np.ndarray:
    """Bound frame differences while sharing a jump across past/future frames.

    A causal forward clamp and an anti-causal backward clamp are both feasible
    solutions to the same convex rate constraint. Their average is therefore
    also feasible and avoids assigning the full correction to only one side of
    an IK discontinuity.
    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or len(source) < 2 or not np.all(np.isfinite(source)):
        raise ValueError("Rate-limited values must be a finite (N>=2, D) array")
    if max_rate <= 0.0 or fps <= 0.0:
        raise ValueError("max_rate and fps must be positive")
    maximum_step = max_rate / fps
    forward = source.copy()
    for index in range(1, len(forward)):
        forward[index] = np.clip(
            forward[index], forward[index - 1] - maximum_step, forward[index - 1] + maximum_step
        )
    backward = source.copy()
    for index in range(len(backward) - 2, -1, -1):
        backward[index] = np.clip(
            backward[index], backward[index + 1] - maximum_step, backward[index + 1] + maximum_step
        )
    result = 0.5 * (forward + backward)
    if np.max(np.abs(np.diff(result, axis=0))) > maximum_step + 1.0e-12:
        raise RuntimeError("Internal error: symmetric rate limiter violated its bound")
    return result


def dominating_rate_limited_envelope(
    required: np.ndarray, max_rate: float, fps: float
) -> np.ndarray:
    """Smallest two-pass envelope that stays above required and changes smoothly."""
    values = np.asarray(required, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Required ground offsets must be a finite N>=2 vector")
    if np.any(values < 0.0) or max_rate <= 0.0 or fps <= 0.0:
        raise ValueError("Offsets must be nonnegative and rate/fps must be positive")
    maximum_step = max_rate / fps
    result = values.copy()
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1] - maximum_step)
    for index in range(len(result) - 2, -1, -1):
        result[index] = max(result[index], result[index + 1] - maximum_step)
    if np.any(result + 1.0e-12 < values):
        raise RuntimeError("Internal error: ground envelope fell below required offset")
    if np.max(np.abs(np.diff(result))) > maximum_step + 1.0e-12:
        raise RuntimeError("Internal error: ground envelope violated its rate bound")
    return result


def floor_contact_distances(qpos: np.ndarray, model: object, mujoco: object) -> np.ndarray:
    """Return the most negative floor collision distance in every pose."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("GMR MuJoCo model has no named floor geom")
    data = mujoco.MjData(model)
    result = np.zeros(len(qpos), dtype=np.float64)
    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        distances = [
            float(data.contact[index].dist)
            for index in range(data.ncon)
            if data.contact[index].geom1 == floor_id or data.contact[index].geom2 == floor_id
        ]
        result[frame_index] = min(distances, default=0.0)
    return result


def discover_sources(folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    sources = sorted({path.resolve() for pattern in patterns for path in folder.glob(pattern)})
    if not sources:
        raise FileNotFoundError(f"No BVH files matched {patterns} under {folder}")
    return sources


def validate_joint_order(model: object, mujoco: object) -> tuple[str, ...]:
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(1, model.njnt)
    )
    if names != LAFAN1_G1_JOINT_NAMES:
        raise ValueError(
            "GMR MuJoCo joint order does not match StableMimic's 29-joint contract:\n"
            f"GMR={names}\nStableMimic={LAFAN1_G1_JOINT_NAMES}"
        )
    return names


def retarget_one(
    source: Path,
    gmr_root: Path,
    fps: float,
    use_velocity_limit: bool,
    joint_velocity_limit: float,
    ground_clearance: float,
    ground_offset_speed_limit: float,
) -> tuple[dict[str, object], dict[str, object]]:
    sys.path.insert(0, str(gmr_root))
    import mujoco as mj
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(str(source), format="lafan1")
    retargeter = GMR(
        src_human="bvh_lafan1",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        use_velocity_limit=use_velocity_limit,
        verbose=False,
    )
    model = mj.MjModel.from_xml_path(retargeter.xml_file)
    joint_names = validate_joint_order(model, mj)
    raw_qpos = np.asarray([retargeter.retarget(frame).copy() for frame in frames])
    raw_rows = qpos_to_stablemimic_rows(raw_qpos)
    raw_ground_distances = floor_contact_distances(raw_qpos, model, mj)
    qpos = raw_qpos.copy()
    qpos[:, 7:] = symmetric_rate_limit(qpos[:, 7:], joint_velocity_limit, fps)
    pre_ground_distances = floor_contact_distances(qpos, model, mj)
    required_ground_offset = np.maximum(0.0, ground_clearance - pre_ground_distances)
    ground_offset = dominating_rate_limited_envelope(
        required_ground_offset, ground_offset_speed_limit, fps
    )
    qpos[:, 2] += ground_offset
    rows = qpos_to_stablemimic_rows(qpos)
    final_ground_distances = floor_contact_distances(qpos, model, mj)

    # Match the useful fields in GMR's dataset pickle without requiring its
    # CUDA-only KinematicsModel. Local-body positions come from MuJoCo FK with
    # an identity floating root, just like the upstream batch export.
    data = mj.MjData(model)
    local_body_pos = np.empty((len(qpos), model.nbody - 1, 3), dtype=np.float64)
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mj.mj_forward(model, data)
        local_pose = pose.copy()
        local_pose[:3] = 0.0
        local_pose[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[:] = local_pose
        mj.mj_forward(model, data)
        local_body_pos[index] = data.xpos[1:]

    raw_joint_velocity = np.diff(raw_rows[:, 7:], axis=0) * fps
    joint_velocity = np.diff(rows[:, 7:], axis=0) * fps
    joint_error = rows[:, 7:] - raw_rows[:, 7:]
    quaternion_norm = np.linalg.norm(rows[:, 3:7], axis=1)
    motion = {
        "fps": float(fps),
        "root_pos": rows[:, :3],
        "root_rot": rows[:, 3:7],
        "dof_pos": rows[:, 7:],
        "local_body_pos": local_body_pos,
        "link_body_list": tuple(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(1, model.nbody)
        ),
    }
    qa = {
        "frames": int(len(rows)),
        "fps": float(fps),
        "duration_s": float((len(rows) - 1) / fps),
        "human_height_m": float(human_height),
        "gmr_use_velocity_limit": bool(use_velocity_limit),
        "postprocess_joint_velocity_limit_rad_s": float(joint_velocity_limit),
        "ground_clearance_m": float(ground_clearance),
        "ground_offset_speed_limit_m_s": float(ground_offset_speed_limit),
        "joint_names": joint_names,
        "quaternion_norm_min": float(np.min(quaternion_norm)),
        "quaternion_norm_max": float(np.max(quaternion_norm)),
        "raw_maximum_abs_joint_velocity_rad_s": float(np.max(np.abs(raw_joint_velocity))),
        "maximum_abs_joint_velocity_rad_s": float(np.max(np.abs(joint_velocity))),
        "velocity_elements_over_limit": int(
            np.count_nonzero(np.abs(joint_velocity) > joint_velocity_limit + 1.0e-10)
        ),
        "joint_correction_rmse_rad": float(np.sqrt(np.mean(joint_error * joint_error))),
        "joint_correction_max_abs_rad": float(np.max(np.abs(joint_error))),
        "raw_gmr_maximum_floor_penetration_m": float(
            max(0.0, -np.min(raw_ground_distances))
        ),
        "post_joint_limit_maximum_floor_penetration_m": float(
            max(0.0, -np.min(pre_ground_distances))
        ),
        "maximum_floor_penetration_m": float(max(0.0, -np.min(final_ground_distances))),
        "ground_offset_min_m": float(np.min(ground_offset)),
        "ground_offset_max_m": float(np.max(ground_offset)),
        "ground_offset_max_speed_m_s": float(np.max(np.abs(np.diff(ground_offset))) * fps),
    }
    return motion, qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-folder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--gmr-revision", required=True)
    parser.add_argument("--pattern", action="append", dest="patterns")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--velocity-limit",
        action="store_true",
        help="Explicitly enable frozen GMR's 3*pi motor VelocityLimit.",
    )
    parser.add_argument("--joint-velocity-limit", type=float, default=3.0 * np.pi)
    parser.add_argument("--ground-clearance", type=float, default=0.002)
    parser.add_argument("--ground-offset-speed-limit", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = tuple(args.patterns or DEFAULT_PATTERNS)
    sources = discover_sources(args.src_folder.resolve(), patterns)
    output_root = args.output_root.resolve()
    pkl_root, csv_root = output_root / "pkl", output_root / "csv"
    pkl_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        pkl_path, csv_path = pkl_root / f"{source.stem}.pkl", csv_root / f"{source.stem}.csv"
        if not args.overwrite and (pkl_path.exists() or csv_path.exists()):
            raise FileExistsError(f"Refusing to overwrite {pkl_path} or {csv_path}")
        print(f"[{index}/{len(sources)}] retargeting {source.name}", flush=True)
        motion, qa = retarget_one(
            source,
            args.gmr_root.resolve(),
            args.fps,
            args.velocity_limit,
            args.joint_velocity_limit,
            args.ground_clearance,
            args.ground_offset_speed_limit,
        )
        with pkl_path.open("wb") as stream:
            pickle.dump(motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
        rows = np.column_stack((motion["root_pos"], motion["root_rot"], motion["dof_pos"]))
        np.savetxt(csv_path, rows, delimiter=",", fmt="%.10g")
        records.append({
            "source": str(source),
            "source_sha256": sha256(source),
            "pickle": str(pkl_path),
            "pickle_sha256": sha256(pkl_path),
            "csv": str(csv_path),
            "csv_sha256": sha256(csv_path),
            **qa,
        })
    manifest = {
        "generator": Path(__file__).name,
        "gmr_repository": "https://github.com/YanjieZe/GMR",
        "gmr_revision": args.gmr_revision,
        "gmr_root": str(args.gmr_root.resolve()),
        "source_root": str(args.src_folder.resolve()),
        "patterns": patterns,
        "gmr_use_velocity_limit": bool(args.velocity_limit),
        "postprocess_joint_velocity_limit_rad_s": float(args.joint_velocity_limit),
        "ground_clearance_m": float(args.ground_clearance),
        "ground_offset_speed_limit_m_s": float(args.ground_offset_speed_limit),
        "environment": {"python": sys.version, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        "motions": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} motions and {manifest_path}")


if __name__ == "__main__":
    main()
