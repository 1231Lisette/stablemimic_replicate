"""Unified continuous-time motion representation.

The raw public LAFAN1 G1 CSV files contain only root pose and joint
positions.  This module derives velocities from the continuous piecewise
interpolation used by the reproduction.  That derivation is an engineering
choice; it is not claimed to be the StableMimic authors' storage format.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _normalize_quaternion_xyzw(quaternion: FloatArray) -> FloatArray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("Quaternion norm must be non-zero.")
    return quaternion / norm


def _quaternion_conjugate_xyzw(quaternion: FloatArray) -> FloatArray:
    result = np.array(quaternion, dtype=np.float64, copy=True)
    result[..., :3] *= -1.0
    return result


def _quaternion_multiply_xyzw(left: FloatArray, right: FloatArray) -> FloatArray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _slerp_xyzw(start: FloatArray, end: FloatArray, alpha: float) -> FloatArray:
    q0 = _normalize_quaternion_xyzw(np.asarray(start, dtype=np.float64))
    q1 = _normalize_quaternion_xyzw(np.asarray(end, dtype=np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return _normalize_quaternion_xyzw(
        np.sin((1.0 - alpha) * theta) / sin_theta * q0 + np.sin(alpha * theta) / sin_theta * q1
    )


def _world_angular_velocity(q0_xyzw: FloatArray, q1_xyzw: FloatArray, dt: float) -> FloatArray:
    """Return the shortest-arc world-frame angular velocity from q0 to q1."""
    q0 = _normalize_quaternion_xyzw(np.asarray(q0_xyzw, dtype=np.float64))
    q1 = _normalize_quaternion_xyzw(np.asarray(q1_xyzw, dtype=np.float64))
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    delta = _normalize_quaternion_xyzw(_quaternion_multiply_xyzw(q1, _quaternion_conjugate_xyzw(q0)))
    vector = delta[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(delta[3]))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return vector / vector_norm * (angle / dt)


@dataclass(frozen=True)
class MotionSample:
    """One continuously sampled reference state."""

    time: float
    root_pos: FloatArray
    root_quat_xyzw: FloatArray
    joint_pos: FloatArray
    root_lin_vel_world: FloatArray
    root_ang_vel_world: FloatArray
    joint_vel: FloatArray


@dataclass(frozen=True)
class MotionReference:
    """A validated, uniformly sampled root-and-joint reference sequence."""

    name: str
    fps: float
    joint_names: tuple[str, ...]
    root_pos: FloatArray
    root_quat_xyzw: FloatArray
    joint_pos: FloatArray

    def __post_init__(self) -> None:
        root_pos = np.asarray(self.root_pos, dtype=np.float64)
        root_quat = np.asarray(self.root_quat_xyzw, dtype=np.float64)
        joint_pos = np.asarray(self.joint_pos, dtype=np.float64)
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {self.fps!r}")
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must have shape (N, 3), got {root_pos.shape}")
        if root_quat.shape != (root_pos.shape[0], 4):
            raise ValueError(f"root_quat_xyzw must have shape (N, 4), got {root_quat.shape}")
        if joint_pos.shape != (root_pos.shape[0], len(self.joint_names)):
            raise ValueError(
                f"joint_pos must have shape (N, {len(self.joint_names)}), got {joint_pos.shape}"
            )
        if root_pos.shape[0] < 2:
            raise ValueError("A motion must contain at least two frames.")
        if not all(np.all(np.isfinite(value)) for value in (root_pos, root_quat, joint_pos)):
            raise ValueError("Motion arrays must contain only finite values.")

        normalized = _normalize_quaternion_xyzw(root_quat)
        continuous = normalized.copy()
        for index in range(1, len(continuous)):
            if float(np.dot(continuous[index - 1], continuous[index])) < 0.0:
                continuous[index] *= -1.0

        object.__setattr__(self, "root_pos", root_pos)
        object.__setattr__(self, "root_quat_xyzw", continuous)
        object.__setattr__(self, "joint_pos", joint_pos)

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration(self) -> float:
        """Time between the first and last samples, in seconds."""
        return (self.num_frames - 1) / self.fps

    def sample(self, time_seconds: float, *, loop: bool = False) -> MotionSample:
        """Sample the reference at an arbitrary time.

        Root and joint positions use linear interpolation. Root orientation uses
        shortest-path SLERP. Velocities are the derivative of the active linear
        segment and the shortest-arc quaternion delta over one source interval.
        """
        if not np.isfinite(time_seconds):
            raise ValueError(f"time_seconds must be finite, got {time_seconds!r}")
        requested_time = float(time_seconds)
        if loop:
            sampled_time = requested_time % self.duration
        else:
            sampled_time = float(np.clip(requested_time, 0.0, self.duration))

        frame_coordinate = sampled_time * self.fps
        if frame_coordinate >= self.num_frames - 1:
            lower = self.num_frames - 2
            alpha = 1.0
        else:
            lower = int(np.floor(frame_coordinate))
            alpha = float(frame_coordinate - lower)
        upper = lower + 1
        source_dt = 1.0 / self.fps

        root_pos = (1.0 - alpha) * self.root_pos[lower] + alpha * self.root_pos[upper]
        joint_pos = (1.0 - alpha) * self.joint_pos[lower] + alpha * self.joint_pos[upper]
        root_quat = _slerp_xyzw(self.root_quat_xyzw[lower], self.root_quat_xyzw[upper], alpha)
        root_lin_vel = (self.root_pos[upper] - self.root_pos[lower]) / source_dt
        joint_vel = (self.joint_pos[upper] - self.joint_pos[lower]) / source_dt
        root_ang_vel = _world_angular_velocity(
            self.root_quat_xyzw[lower], self.root_quat_xyzw[upper], source_dt
        )
        return MotionSample(
            time=sampled_time,
            root_pos=root_pos,
            root_quat_xyzw=root_quat,
            joint_pos=joint_pos,
            root_lin_vel_world=root_lin_vel,
            root_ang_vel_world=root_ang_vel,
            joint_vel=joint_vel,
        )
