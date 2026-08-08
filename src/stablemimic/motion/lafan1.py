"""Strict loader for the public retargeted LAFAN1 Unitree G1 CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
