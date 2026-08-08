"""GPU-resident, sequence-safe continuous-time motion library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .lafan1 import MotionLibraries, discover_motion_libraries, load_lafan1_csv
from .reference import MotionReference


@dataclass(frozen=True)
class TorchMotionSample:
    root_pos: torch.Tensor
    root_quat_xyzw: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel_world: torch.Tensor
    root_ang_vel_world: torch.Tensor
    joint_vel: torch.Tensor
    normalized_phase: torch.Tensor


def _normalize(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dim=-1)


def _slerp(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    q0, q1 = _normalize(start), _normalize(end)
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    close = dot > 0.9995
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1.0e-8)
    spherical = torch.sin((1.0 - alpha) * theta) / sin_theta * q0 + torch.sin(alpha * theta) / sin_theta * q1
    linear = (1.0 - alpha) * q0 + alpha * q1
    return _normalize(torch.where(close, linear, spherical))


def _angular_velocity(q0: torch.Tensor, q1: torch.Tensor, dt: float) -> torch.Tensor:
    q0, q1 = _normalize(q0), _normalize(q1)
    q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0.0, -q1, q1)
    conjugate = torch.cat((-q0[..., :3], q0[..., 3:]), dim=-1)
    delta = _normalize(_quat_mul(q1, conjugate))
    vector = delta[..., :3]
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, delta[..., 3:])
    angle = torch.where(angle > torch.pi, angle - 2.0 * torch.pi, angle)
    return torch.where(norm > 1.0e-8, vector / norm.clamp_min(1.0e-8) * angle / dt, torch.zeros_like(vector))


class TorchMotionLibrary:
    """A collection that never interpolates across sequence boundaries."""

    def __init__(self, motions: tuple[MotionReference, ...], device: torch.device | str):
        if not motions:
            raise ValueError("motion library cannot be empty")
        self.device = torch.device(device)
        self.names = tuple(motion.name for motion in motions)
        self.fps = tuple(float(motion.fps) for motion in motions)
        self.root_pos = tuple(torch.as_tensor(m.root_pos, dtype=torch.float32, device=device) for m in motions)
        self.root_quat = tuple(torch.as_tensor(m.root_quat_xyzw, dtype=torch.float32, device=device) for m in motions)
        self.joint_pos = tuple(torch.as_tensor(m.joint_pos, dtype=torch.float32, device=device) for m in motions)
        self.lengths = torch.tensor([m.num_frames for m in motions], dtype=torch.long, device=device)
        self.durations = torch.tensor([m.duration for m in motions], dtype=torch.float32, device=device)

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...], device: torch.device | str) -> "TorchMotionLibrary":
        return cls(tuple(load_lafan1_csv(path) for path in paths), device)

    @property
    def num_motions(self) -> int:
        return len(self.names)

    def random_ids(self, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(self.num_motions, (count,), device=self.device, generator=generator)

    def random_times(self, motion_ids: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.rand(motion_ids.shape, device=self.device, generator=generator) * self.durations[motion_ids]

    def sample(self, motion_ids: torch.Tensor, times: torch.Tensor) -> TorchMotionSample:
        if motion_ids.shape != times.shape or motion_ids.ndim != 1:
            raise ValueError("motion_ids and times must be matching rank-one tensors")
        count = motion_ids.numel()
        result = {
            "root_pos": torch.empty(count, 3, device=self.device),
            "root_quat_xyzw": torch.empty(count, 4, device=self.device),
            "joint_pos": torch.empty(count, 29, device=self.device),
            "root_lin_vel_world": torch.empty(count, 3, device=self.device),
            "root_ang_vel_world": torch.empty(count, 3, device=self.device),
            "joint_vel": torch.empty(count, 29, device=self.device),
            "normalized_phase": torch.empty(count, 1, device=self.device),
        }
        for motion_id in motion_ids.unique(sorted=True).tolist():
            mask = motion_ids == motion_id
            duration = self.durations[motion_id]
            sampled_time = times[mask].clamp(0.0, duration)
            coordinate = sampled_time * self.fps[motion_id]
            lower = torch.floor(coordinate).long().clamp(max=int(self.lengths[motion_id]) - 2)
            upper = lower + 1
            alpha = (coordinate - lower.float()).unsqueeze(-1)
            positions, quaternions, joints = self.root_pos[motion_id], self.root_quat[motion_id], self.joint_pos[motion_id]
            dt = 1.0 / self.fps[motion_id]
            result["root_pos"][mask] = torch.lerp(positions[lower], positions[upper], alpha)
            result["root_quat_xyzw"][mask] = _slerp(quaternions[lower], quaternions[upper], alpha)
            result["joint_pos"][mask] = torch.lerp(joints[lower], joints[upper], alpha)
            result["root_lin_vel_world"][mask] = (positions[upper] - positions[lower]) / dt
            result["root_ang_vel_world"][mask] = _angular_velocity(quaternions[lower], quaternions[upper], dt)
            result["joint_vel"][mask] = (joints[upper] - joints[lower]) / dt
            result["normalized_phase"][mask] = (sampled_time / duration.clamp_min(1.0e-6)).unsqueeze(-1)
        return TorchMotionSample(**result)


@dataclass(frozen=True)
class TorchMotionLibraries:
    tracking: TorchMotionLibrary
    recovery: TorchMotionLibrary


def load_torch_motion_libraries(data_root: str | Path, device: torch.device | str) -> TorchMotionLibraries:
    paths: MotionLibraries = discover_motion_libraries(data_root)
    return TorchMotionLibraries(
        tracking=TorchMotionLibrary.from_paths(paths.tracking, device),
        recovery=TorchMotionLibrary.from_paths(paths.recovery, device),
    )


class FailureAdaptiveSampler:
    """Half uniform, half failure-weighted recovery frame sampling."""

    def __init__(self, library: TorchMotionLibrary, bins_per_motion: int = 64):
        self.library = library
        self.bins_per_motion = int(bins_per_motion)
        self.failures = torch.ones(library.num_motions, self.bins_per_motion, device=library.device)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        uniform_count = count // 2
        hard_count = count - uniform_count
        uniform_ids = self.library.random_ids(uniform_count)
        uniform_times = self.library.random_times(uniform_ids)
        hard_flat = torch.multinomial(self.failures.flatten(), hard_count, replacement=True)
        hard_ids, bins = hard_flat // self.bins_per_motion, hard_flat % self.bins_per_motion
        hard_times = (bins.float() + torch.rand(hard_count, device=self.library.device)) / self.bins_per_motion
        hard_times *= self.library.durations[hard_ids]
        return torch.cat((uniform_ids, hard_ids)), torch.cat((uniform_times, hard_times))

    def record_failures(self, motion_ids: torch.Tensor, times: torch.Tensor, failed: torch.Tensor) -> None:
        if failed.any():
            ids = motion_ids[failed]
            bins = ((times[failed] / self.library.durations[ids]) * self.bins_per_motion).long()
            bins.clamp_(0, self.bins_per_motion - 1)
            self.failures[ids, bins] += 1.0
