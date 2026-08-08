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
