"""Strict loader for the public retargeted LAFAN1 Unitree G1 CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stablemimic.config import RecoverySegmentationCfg

from .reference import MotionReference

LAFAN1_G1_FPS = 30.0
LAFAN1_G1_COLUMN_COUNT = 36
LAFAN1_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class MotionLibraries:
    """Sequence-disjoint first-stage tracking and recovery file collections."""

    tracking: tuple[Path, ...]
    recovery: tuple[Path, ...]


def discover_motion_libraries(data_root: str | Path) -> MotionLibraries:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LAFAN1 G1 data directory does not exist: {root}")
    tracking = tuple(sorted(root.glob("dance*.csv")))
    recovery = tuple(sorted(root.glob("fallAndGetUp*.csv")))
    if not tracking:
        raise FileNotFoundError(f"No dance*.csv files found under {root}")
    if not recovery:
        raise FileNotFoundError(f"No fallAndGetUp*.csv files found under {root}")
    overlap = set(tracking).intersection(recovery)
    if overlap:
        raise ValueError(f"Tracking/recovery libraries overlap: {sorted(overlap)}")
    return MotionLibraries(tracking=tracking, recovery=recovery)


def load_lafan1_csv(path: str | Path, *, fps: float = LAFAN1_G1_FPS) -> MotionReference:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Motion CSV does not exist: {csv_path}")
    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Failed to parse numeric, headerless CSV {csv_path}: {error}") from error
    if data.shape[1] != LAFAN1_G1_COLUMN_COUNT:
        raise ValueError(
            f"Expected {LAFAN1_G1_COLUMN_COUNT} columns in {csv_path}, got {data.shape[1]}"
        )
    if data.shape[0] < 2:
        raise ValueError(f"Expected at least two frames in {csv_path}, got {data.shape[0]}")
    if not np.all(np.isfinite(data)):
        bad_count = int(np.size(data) - np.count_nonzero(np.isfinite(data)))
        raise ValueError(f"CSV contains {bad_count} non-finite values: {csv_path}")
    return MotionReference(
        name=csv_path.stem,
        fps=float(fps),
        joint_names=LAFAN1_G1_JOINT_NAMES,
        root_pos=data[:, 0:3],
        root_quat_xyzw=data[:, 3:7],
        joint_pos=data[:, 7:36],
    )


def _root_tilt_degrees(root_quat_xyzw: np.ndarray) -> np.ndarray:
    quaternion = root_quat_xyzw / np.linalg.norm(root_quat_xyzw, axis=1, keepdims=True)
    x, y = quaternion[:, 0], quaternion[:, 1]
    upright_cosine = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.rad2deg(np.arccos(upright_cosine))


def _first_sustained(mask: np.ndarray, start: int, frames: int) -> int | None:
    """Return the first run start at or after ``start`` that lasts ``frames``."""
    run = 0
    for index in range(start, len(mask)):
        run = run + 1 if bool(mask[index]) else 0
        if run >= frames:
            return index - frames + 1
    return None


def segment_recovery_motion(
    motion: MotionReference, config: RecoverySegmentationCfg
) -> tuple[MotionReference, ...]:
    """Split a repeated recording into sustained fallen-to-upright clips.

    The public LAFAN1 recovery CSVs are session recordings, not atomic
    trajectories.  This deterministic state machine is an explicit
    reproduction choice; it never interpolates or joins disjoint clips.
    """
    if not config.enabled:
        return (motion,)
    hold_frames = max(1, int(round(config.hold_time_s * motion.fps)))
    tilt = _root_tilt_degrees(motion.root_quat_xyzw)
    fallen = (motion.root_pos[:, 2] < config.fallen_height_threshold) | (
        tilt > config.fallen_tilt_degrees
    )
    upright = (motion.root_pos[:, 2] >= config.upright_height_threshold) & (
        tilt <= config.upright_tilt_degrees
    )
    clips: list[MotionReference] = []
    cursor = 0
    while cursor < motion.num_frames:
        fall_start = _first_sustained(fallen, cursor, hold_frames)
        if fall_start is None:
            break
        upright_start = _first_sustained(upright, fall_start + hold_frames, hold_frames)
        if upright_start is None:
            break
        end = upright_start + hold_frames - 1
        recovery_start = fall_start
        if config.trim_to_tilted_nadir:
            candidate_mask = (
                (motion.root_pos[fall_start:upright_start, 2] < config.fallen_height_threshold)
                & (tilt[fall_start:upright_start] > config.fallen_tilt_degrees)
            )
            candidates = np.flatnonzero(candidate_mask) + fall_start
            if candidates.size == 0:
                cursor = end + 1
                continue
            recovery_start = int(candidates[
                np.argmin(motion.root_pos[candidates, 2])
            ])
        duration = (end - recovery_start) / motion.fps
        if duration <= config.maximum_clip_duration_s:
            clip_index = len(clips)
            frame_slice = slice(recovery_start, end + 1)
            clips.append(MotionReference(
                name=f"{motion.name}__recovery_{clip_index:03d}",
                fps=motion.fps,
                joint_names=motion.joint_names,
                root_pos=motion.root_pos[frame_slice],
                root_quat_xyzw=motion.root_quat_xyzw[frame_slice],
                joint_pos=motion.joint_pos[frame_slice],
            ))
        cursor = end + 1
    if not clips:
        raise ValueError(f"No valid fall-to-upright clips detected in recovery motion {motion.name!r}")
    return tuple(clips)


def load_segmented_recovery_motions(
    paths: tuple[Path, ...], config: RecoverySegmentationCfg
) -> tuple[MotionReference, ...]:
    clips: list[MotionReference] = []
    for path in paths:
        clips.extend(segment_recovery_motion(load_lafan1_csv(path), config))
    return tuple(clips)
