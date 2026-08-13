"""Motion data loading and continuous-time sampling."""

from .lafan1 import (
    LAFAN1_G1_JOINT_NAMES,
    MotionLibraries,
    discover_motion_libraries,
    load_lafan1_csv,
    load_lafan1_motion,
    load_lafan1_npz,
    load_segmented_recovery_motions,
    segment_recovery_motion,
)
from .reference import MotionReference, MotionSample

try:
    from .torch_library import (
        FailureAdaptiveSampler,
        TorchMotionLibraries,
        TorchMotionLibrary,
        TorchMotionSample,
        load_torch_motion_libraries,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise

__all__ = [
    "LAFAN1_G1_JOINT_NAMES",
    "MotionLibraries",
    "MotionReference",
    "MotionSample",
    "discover_motion_libraries",
    "load_lafan1_csv",
    "load_lafan1_motion",
    "load_lafan1_npz",
    "load_segmented_recovery_motions",
    "segment_recovery_motion",
]

if "TorchMotionLibrary" in globals():
    __all__ += [
        "FailureAdaptiveSampler",
        "TorchMotionLibraries",
        "TorchMotionLibrary",
        "TorchMotionSample",
        "load_torch_motion_libraries",
    ]
