"""Paper-matched deterministic horizontal push schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PushEvent:
    index: int
    direction_xy: tuple[float, float]
    force_newtons: float
    duration_seconds: float = 0.2


def matched_push_protocol() -> tuple[PushEvent, ...]:
    """Return 100 pushes: 25 in each +/-x and +/-y direction, 525--575 N."""
    directions = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    events = []
    for direction_index, direction in enumerate(directions):
        for within_direction in range(25):
            force = 525.0 + 50.0 * within_direction / 24.0
            events.append(PushEvent(direction_index * 25 + within_direction, direction, force))
    return tuple(events)


def classify_fallen_orientation(projected_gravity):
    """Classify a fallen torso from gravity expressed in the G1 body frame.

    Labels are 0=supine, 1=prone, 2=left-side, 3=right-side, 4=other.
    G1 uses x-forward, y-left, z-up. Near-upright/upside-down states are
    reported as ``other`` rather than forced into a lying class.
    """
    import torch

    if projected_gravity.ndim != 2 or projected_gravity.shape[1] != 3:
        raise ValueError("projected_gravity must have shape (N, 3)")
    dominant = projected_gravity.abs().argmax(dim=1)
    result = torch.full(
        (projected_gravity.shape[0],), 4, dtype=torch.long, device=projected_gravity.device
    )
    x_dominant = dominant == 0
    y_dominant = dominant == 1
    result[x_dominant & (projected_gravity[:, 0] < 0.0)] = 0
    result[x_dominant & (projected_gravity[:, 0] >= 0.0)] = 1
    result[y_dominant & (projected_gravity[:, 1] >= 0.0)] = 2
    result[y_dominant & (projected_gravity[:, 1] < 0.0)] = 3
    return result


def support_body_groups(body_names: list[str]) -> dict[str, list[int]]:
    """Group G1 rigid bodies that can reveal get-up support contacts."""
    patterns = {
        "hands": ("wrist", "hand"),
        "elbows": ("elbow",),
        "knees": ("knee",),
        "feet": ("ankle", "foot"),
        "trunk": ("pelvis", "torso"),
    }
    lowered = [name.lower() for name in body_names]
    return {
        group: [
            index for index, name in enumerate(lowered)
            if any(pattern in name for pattern in group_patterns)
        ]
        for group, group_patterns in patterns.items()
    }
