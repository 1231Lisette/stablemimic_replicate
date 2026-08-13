"""BeyondMimic-style NPZ reference preprocessing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .reference import MotionReference


STANDARD_NPZ_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class ResampledMotion:
    times: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.times.shape[0])


def _normalize_quaternions_xyzw(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("motion contains a zero-norm root quaternion")
    return value / norm


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    left = _normalize_quaternions_xyzw(left)
    right = _normalize_quaternions_xyzw(right)
    dot = np.sum(left * right, axis=-1)
    right = np.where((dot < 0.0)[:, None], -right, right)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    linear = sine < 1.0e-7
    left_weight = np.empty_like(alpha)
    right_weight = np.empty_like(alpha)
    left_weight[linear] = 1.0 - alpha[linear]
    right_weight[linear] = alpha[linear]
    left_weight[~linear] = np.sin((1.0 - alpha[~linear]) * theta[~linear]) / sine[~linear]
    right_weight[~linear] = np.sin(alpha[~linear] * theta[~linear]) / sine[~linear]
    result = left_weight[:, None] * left + right_weight[:, None] * right
    return _normalize_quaternions_xyzw(result)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _angular_velocity_wxyz(quaternions: np.ndarray, dt: float) -> np.ndarray:
    if quaternions.shape[0] < 3:
        raise ValueError("at least three output frames are required to compute angular velocity")
    previous = quaternions[:-2]
    following = quaternions[2:]
    conjugate = previous.copy()
    conjugate[:, 1:] *= -1.0
    relative = _quat_multiply_wxyz(following, conjugate)
    relative = np.where((relative[:, :1] < 0.0), -relative, relative)
    vector = relative[:, 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, relative[:, 0].clip(-1.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1.0e-8,
    )
    center = vector * scale[:, None] / (2.0 * dt)
    return np.concatenate((center[:1], center, center[-1:]), axis=0)


def resample_motion(motion: MotionReference, output_fps: float = 50.0) -> ResampledMotion:
    """Match BeyondMimic's half-open 50 Hz interpolation convention."""
    if output_fps <= 0.0 or not np.isfinite(output_fps):
        raise ValueError("output_fps must be finite and positive")
    output_dt = 1.0 / float(output_fps)
    times = np.arange(0.0, motion.duration, output_dt, dtype=np.float64)
    if times.shape[0] < 3:
        raise ValueError("motion is too short to produce three output frames")
    coordinate = times * motion.fps
    lower = np.floor(coordinate).astype(np.int64)
    upper = np.minimum(lower + 1, motion.num_frames - 1)
    alpha = coordinate - lower
    root_pos = (
        (1.0 - alpha[:, None]) * motion.root_pos[lower]
        + alpha[:, None] * motion.root_pos[upper]
    )
    joint_pos = (
        (1.0 - alpha[:, None]) * motion.joint_pos[lower]
        + alpha[:, None] * motion.joint_pos[upper]
    )
    root_quat_xyzw = _slerp_xyzw(
        motion.root_quat_xyzw[lower], motion.root_quat_xyzw[upper], alpha
    )
    root_quat_wxyz = root_quat_xyzw[:, (3, 0, 1, 2)]
    root_lin_vel = np.gradient(root_pos, output_dt, axis=0)
    joint_vel = np.gradient(joint_pos, output_dt, axis=0)
    root_ang_vel = _angular_velocity_wxyz(root_quat_wxyz, output_dt)
    result = ResampledMotion(
        times=times.astype(np.float32),
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=root_quat_wxyz.astype(np.float32),
        root_lin_vel_w=root_lin_vel.astype(np.float32),
        root_ang_vel_w=root_ang_vel.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
    )
    for name, value in result.__dict__.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"resampled field {name} contains NaN or Inf")
    return result


def validate_standard_npz(
    arrays: Mapping[str, np.ndarray], *, expected_joint_count: int = 29
) -> int:
    """Validate the standard fields and return the common frame count."""
    missing = [name for name in STANDARD_NPZ_FIELDS if name not in arrays]
    if missing:
        raise ValueError(f"NPZ is missing standard fields: {missing}")
    fps = np.asarray(arrays["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0.0:
        raise ValueError("fps must contain one finite positive value")
    joint_pos = np.asarray(arrays["joint_pos"])
    if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_joint_count:
        raise ValueError(f"joint_pos must have shape (T, {expected_joint_count})")
    frames = int(joint_pos.shape[0])
    expected_shapes = {
        "joint_vel": (frames, expected_joint_count),
        "body_pos_w": (frames, None, 3),
        "body_quat_w": (frames, None, 4),
        "body_lin_vel_w": (frames, None, 3),
        "body_ang_vel_w": (frames, None, 3),
    }
    body_count = None
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if shape[1] is None:
            if value.ndim != 3 or value.shape[0] != frames or value.shape[2] != shape[2]:
                raise ValueError(f"{name} must have shape (T, B, {shape[2]})")
            body_count = value.shape[1] if body_count is None else body_count
            if value.shape[1] != body_count:
                raise ValueError("all body fields must use the same body count")
        elif value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")
    if frames < 3:
        raise ValueError("NPZ must contain at least three frames")
    return frames


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
