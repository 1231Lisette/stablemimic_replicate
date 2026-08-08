"""Motion data loading and continuous-time sampling."""

from .lafan1 import LAFAN1_G1_JOINT_NAMES, MotionLibraries, discover_motion_libraries, load_lafan1_csv
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
]

if "TorchMotionLibrary" in globals():
    __all__ += [
        "FailureAdaptiveSampler",
        "TorchMotionLibraries",
        "TorchMotionLibrary",
        "TorchMotionSample",
        "load_torch_motion_libraries",
    ]
