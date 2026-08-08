"""Pure joint-name mapping helpers for the Isaac Lab G1 articulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES


@dataclass(frozen=True)
class G1JointMapping:
    csv_to_sim: tuple[int, ...]
    extra_hand_joints: tuple[str, ...]


def build_lafan1_g1_joint_mapping(simulator_names: Sequence[str]) -> G1JointMapping:
    """Map the fixed LAFAN1 29-joint order into an Isaac articulation.

    The audited Isaac Lab ``G1_29DOF_CFG`` USD has 43 joints: the required
    29 body joints plus 14 hand joints. Only recognized hand extras are allowed.
    """
    names = list(simulator_names)
    expected = list(LAFAN1_G1_JOINT_NAMES)
    missing = sorted(set(expected) - set(names))
    duplicates = sorted(name for name in expected if names.count(name) != 1)
    extra = sorted(set(names) - set(expected))
    unsupported_extra = sorted(
        name for name in extra if not (name.startswith("left_hand_") or name.startswith("right_hand_"))
    )
    if missing or duplicates or unsupported_extra:
        raise ValueError(
            "Isaac G1 articulation does not contain an unambiguous LAFAN1 body-joint subset: "
            f"missing={missing}, duplicates={duplicates}, unsupported_extra={unsupported_extra}, "
            f"simulator_count={len(names)}"
        )
    return G1JointMapping(
        csv_to_sim=tuple(names.index(name) for name in expected),
        extra_hand_joints=tuple(extra),
    )
