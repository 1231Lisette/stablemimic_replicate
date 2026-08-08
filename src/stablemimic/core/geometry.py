"""Torch quaternion helpers with explicit convention names."""

from __future__ import annotations

import torch


def xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion[..., (3, 0, 1, 2)]


def wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion[..., (1, 2, 3, 0)]


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def euler_xyz_to_quaternion_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dim=-1)


def rotate_inverse_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from world into the quaternion's local frame."""
    q_vector = quaternion[..., 1:]
    q_scalar = quaternion[..., :1]
    return (
        vector * (2.0 * torch.square(q_scalar) - 1.0)
        - 2.0 * q_scalar * torch.cross(q_vector, vector, dim=-1)
        + 2.0 * q_vector * (q_vector * vector).sum(-1, keepdim=True)
    )


def projected_gravity_from_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros(quaternion.shape[0], 3, device=quaternion.device, dtype=quaternion.dtype)
    gravity[:, 2] = -1.0
    return rotate_inverse_wxyz(xyzw_to_wxyz(quaternion), gravity)
