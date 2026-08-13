"""Strict loader for the public retargeted LAFAN1 Unitree G1 CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stablemimic.config import RecoverySegmentationCfg

from .npz import validate_standard_npz
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


def _resolve_selected_files(root: Path, names: tuple[str, ...], label: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in names:
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} motion escapes data_root: {name}") from error
        if path.suffix.lower() not in (".csv", ".npz"):
            raise ValueError(f"Unsupported {label} motion extension: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Selected {label} motion does not exist: {path}")
        paths.append(path)
    return tuple(paths)


def _discover_by_prefix(root: Path, prefix: str) -> tuple[Path, ...]:
    paths = tuple(sorted((*root.glob(f"{prefix}*.csv"), *root.glob(f"{prefix}*.npz"))))
    stems = [path.stem for path in paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"Multiple motion formats found for the same stems: {duplicates}")
    return paths


def discover_motion_libraries(
    data_root: str | Path,
    *,
    tracking_files: tuple[str, ...] = (),
    recovery_files: tuple[str, ...] = (),
) -> MotionLibraries:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LAFAN1 G1 data directory does not exist: {root}")
    if bool(tracking_files) != bool(recovery_files):
        raise ValueError("tracking_files and recovery_files must be set together")
    tracking = (
        _resolve_selected_files(root, tracking_files, "tracking")
        if tracking_files else _discover_by_prefix(root, "dance")
    )
    recovery = (
        _resolve_selected_files(root, recovery_files, "recovery")
        if recovery_files else _discover_by_prefix(root, "fallAndGetUp")
    )
    if not tracking:
        raise FileNotFoundError(f"No dance motion files found under {root}")
    if not recovery:
        raise FileNotFoundError(f"No fallAndGetUp motion files found under {root}")
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


def load_lafan1_npz(path: str | Path) -> MotionReference:
    """Load a validated BeyondMimic-style G1 NPZ as a training reference."""
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Motion NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    frames = validate_standard_npz(arrays, expected_joint_count=len(LAFAN1_G1_JOINT_NAMES))
    if "joint_names" in arrays:
        joint_names = tuple(str(value) for value in np.asarray(arrays["joint_names"]).tolist())
        if joint_names != LAFAN1_G1_JOINT_NAMES:
            raise ValueError(f"NPZ joint order does not match G1-29DoF contract: {npz_path}")
    root_pos = np.asarray(
        arrays.get("source_root_pos", np.asarray(arrays["body_pos_w"])[:, 0]),
        dtype=np.float64,
    )
    root_quat_wxyz = np.asarray(
        arrays.get("source_root_quat_wxyz", np.asarray(arrays["body_quat_w"])[:, 0]),
        dtype=np.float64,
    )
    if root_pos.shape != (frames, 3) or root_quat_wxyz.shape != (frames, 4):
        raise ValueError("NPZ root fields must have shapes (T, 3) and (T, 4)")
    if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat_wxyz)):
        raise ValueError(f"NPZ root fields contain NaN or Inf: {npz_path}")
    norms = np.linalg.norm(root_quat_wxyz, axis=1)
    if np.any(norms <= 1.0e-12) or not np.allclose(norms, 1.0, atol=1.0e-4):
        raise ValueError(f"NPZ root quaternions are not normalized: {npz_path}")
    root_quat_wxyz = root_quat_wxyz / norms[:, None]
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    return MotionReference(
        name=npz_path.stem,
        fps=fps,
        joint_names=LAFAN1_G1_JOINT_NAMES,
        root_pos=root_pos,
        root_quat_xyzw=root_quat_wxyz[:, (1, 2, 3, 0)],
        joint_pos=np.asarray(arrays["joint_pos"], dtype=np.float64),
    )


def load_lafan1_motion(path: str | Path) -> MotionReference:
    motion_path = Path(path)
    if motion_path.suffix.lower() == ".csv":
        return load_lafan1_csv(motion_path)
    if motion_path.suffix.lower() == ".npz":
        return load_lafan1_npz(motion_path)
    raise ValueError(f"Unsupported motion extension: {motion_path}")


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
        clips.extend(segment_recovery_motion(load_lafan1_motion(path), config))
    return tuple(clips)
