"""GPU-resident, sequence-safe continuous-time motion library."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from stablemimic.config import RecoverySegmentationCfg
from stablemimic.core.geometry import projected_gravity_from_xyzw

from .lafan1 import (
    MotionLibraries,
    discover_motion_libraries,
    load_lafan1_motion,
    load_segmented_recovery_motions,
)
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
        self._fps_tensor = torch.tensor(self.fps, dtype=torch.float32, device=device)
        self._root_pos_padded = torch.nn.utils.rnn.pad_sequence(
            self.root_pos, batch_first=True
        )
        self._root_quat_padded = torch.nn.utils.rnn.pad_sequence(
            self.root_quat, batch_first=True
        )
        self._joint_pos_padded = torch.nn.utils.rnn.pad_sequence(
            self.joint_pos, batch_first=True
        )
        self._frame_motion_ids = torch.cat([
            torch.full((m.num_frames,), index, dtype=torch.long, device=device)
            for index, m in enumerate(motions)
        ])
        self._frame_times = torch.cat([
            torch.arange(m.num_frames, dtype=torch.float32, device=device) / m.fps
            for m in motions
        ])
        self._frame_root_height = torch.cat([values[:, 2] for values in self.root_pos])
        self._frame_projected_gravity = projected_gravity_from_xyzw(torch.cat(self.root_quat))
        self._frame_joint_pos = torch.cat(self.joint_pos)

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...], device: torch.device | str) -> "TorchMotionLibrary":
        return cls(tuple(load_lafan1_motion(path) for path in paths), device)

    @property
    def num_motions(self) -> int:
        return len(self.names)

    def random_ids(self, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randint(self.num_motions, (count,), device=self.device, generator=generator)

    def random_times(self, motion_ids: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.rand(motion_ids.shape, device=self.device, generator=generator) * self.durations[motion_ids]

    def sample_representative_fallen_states(
        self,
        count: int,
        *,
        height_threshold: float = 0.5,
        tilt_threshold_degrees: float = 60.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the lowest clearly tilted frame from each eligible motion.

        Atomic clips begin when the fall criterion first persists, which can be
        a low crouch or a still-falling frame. This evaluation helper instead
        finds a reproducible, physically fallen state inside each clip.
        """
        tilt_threshold = torch.deg2rad(torch.tensor(
            tilt_threshold_degrees, device=self.device
        ))
        candidates_ids = []
        candidates_times = []
        for motion_id in range(self.num_motions):
            gravity = projected_gravity_from_xyzw(self.root_quat[motion_id])
            tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
            eligible = (tilt >= tilt_threshold) & (
                self.root_pos[motion_id][:, 2] <= height_threshold
            )
            frame_ids = torch.nonzero(eligible, as_tuple=False).flatten()
            if frame_ids.numel() == 0:
                continue
            local = self.root_pos[motion_id][frame_ids, 2].argmin()
            frame_id = frame_ids[local]
            candidates_ids.append(motion_id)
            candidates_times.append(frame_id / self.fps[motion_id])
        if not candidates_ids:
            raise ValueError("recovery library contains no clearly fallen states")
        bank_ids = torch.tensor(candidates_ids, dtype=torch.long, device=self.device)
        bank_times = torch.stack(candidates_times)
        choices = torch.randint(bank_ids.numel(), (count,), device=self.device)
        return bank_ids[choices], bank_times[choices]

    def sample(self, motion_ids: torch.Tensor, times: torch.Tensor) -> TorchMotionSample:
        if motion_ids.shape != times.shape or motion_ids.ndim != 1:
            raise ValueError("motion_ids and times must be matching rank-one tensors")
        duration = self.durations[motion_ids]
        sampled_time = torch.minimum(times.clamp_min(0.0), duration)
        fps = self._fps_tensor[motion_ids]
        coordinate = sampled_time * fps
        lower = torch.minimum(
            torch.floor(coordinate).long(), self.lengths[motion_ids] - 2
        )
        upper = lower + 1
        alpha = (coordinate - lower.float()).unsqueeze(-1)
        positions_lower = self._root_pos_padded[motion_ids, lower]
        positions_upper = self._root_pos_padded[motion_ids, upper]
        quaternions_lower = self._root_quat_padded[motion_ids, lower]
        quaternions_upper = self._root_quat_padded[motion_ids, upper]
        joints_lower = self._joint_pos_padded[motion_ids, lower]
        joints_upper = self._joint_pos_padded[motion_ids, upper]
        dt = (1.0 / fps).unsqueeze(-1)
        return TorchMotionSample(
            root_pos=torch.lerp(positions_lower, positions_upper, alpha),
            root_quat_xyzw=_slerp(quaternions_lower, quaternions_upper, alpha),
            joint_pos=torch.lerp(joints_lower, joints_upper, alpha),
            root_lin_vel_world=(positions_upper - positions_lower) / dt,
            root_ang_vel_world=_angular_velocity(
                quaternions_lower, quaternions_upper, dt
            ),
            joint_vel=(joints_upper - joints_lower) / dt,
            normalized_phase=(sampled_time / duration.clamp_min(1.0e-6)).unsqueeze(-1),
        )

    def nearest_frame(
        self,
        root_height: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_pos: torch.Tensor,
        *,
        joint_weight: float = 1.0,
        height_weight: float = 4.0,
        gravity_weight: float = 2.0,
        query_chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match physical states to source frames without materializing N×M×29."""
        count = root_height.shape[0]
        if projected_gravity.shape != (count, 3) or joint_pos.shape != (count, 29):
            raise ValueError("nearest-frame state shapes must be (N,), (N, 3), and (N, 29)")
        if min(joint_weight, height_weight, gravity_weight) < 0.0:
            raise ValueError("nearest-frame weights must be non-negative")
        matches: list[torch.Tensor] = []
        bank_joint_norm = self._frame_joint_pos.square().mean(dim=1)
        bank_gravity_norm = self._frame_projected_gravity.square().mean(dim=1)
        for start in range(0, count, query_chunk_size):
            stop = min(start + query_chunk_size, count)
            query_joint = joint_pos[start:stop]
            joint_distance = (
                query_joint.square().mean(dim=1, keepdim=True)
                + bank_joint_norm.unsqueeze(0)
                - (2.0 / query_joint.shape[1]) * query_joint @ self._frame_joint_pos.T
            )
            query_gravity = projected_gravity[start:stop]
            gravity_distance = (
                query_gravity.square().mean(dim=1, keepdim=True)
                + bank_gravity_norm.unsqueeze(0)
                - (2.0 / query_gravity.shape[1])
                * query_gravity @ self._frame_projected_gravity.T
            )
            height_distance = (
                root_height[start:stop, None] - self._frame_root_height[None, :]
            ).square()
            score = (
                joint_weight * joint_distance
                + height_weight * height_distance
                + gravity_weight * gravity_distance
            )
            matches.append(score.argmin(dim=1))
        frame_ids = torch.cat(matches) if matches else torch.empty(0, dtype=torch.long, device=self.device)
        return self._frame_motion_ids[frame_ids], self._frame_times[frame_ids]


@dataclass(frozen=True)
class TorchMotionLibraries:
    tracking: TorchMotionLibrary
    recovery: TorchMotionLibrary


def load_torch_motion_libraries(
    data_root: str | Path,
    device: torch.device | str,
    recovery_segmentation: RecoverySegmentationCfg | None = None,
    tracking_files: tuple[str, ...] = (),
    recovery_files: tuple[str, ...] = (),
) -> TorchMotionLibraries:
    paths: MotionLibraries = discover_motion_libraries(
        data_root,
        tracking_files=tracking_files,
        recovery_files=recovery_files,
    )
    segmentation = recovery_segmentation or RecoverySegmentationCfg()
    return TorchMotionLibraries(
        tracking=TorchMotionLibrary.from_paths(paths.tracking, device),
        recovery=TorchMotionLibrary(
            load_segmented_recovery_motions(paths.recovery, segmentation), device
        ),
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
