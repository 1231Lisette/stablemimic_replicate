#!/usr/bin/env python3
"""Audit the first-stage LAFAN1 G1 tracking and recovery libraries."""

from __future__ import annotations

import argparse
import json

import numpy as np

from stablemimic.config import RecoverySegmentationCfg
from stablemimic.motion.lafan1 import (
    discover_motion_libraries,
    load_lafan1_csv,
    load_segmented_recovery_motions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    libraries = discover_motion_libraries(args.data_root)
    result: dict[str, object] = {"data_root": args.data_root, "tracking": [], "recovery": []}
    for category, paths in (("tracking", libraries.tracking), ("recovery", libraries.recovery)):
        records: list[dict[str, object]] = []
        for path in paths:
            motion = load_lafan1_csv(path)
            quaternion_norm = np.linalg.norm(motion.root_quat_xyzw, axis=1)
            records.append(
                {
                    "file": path.name,
                    "frames": motion.num_frames,
                    "duration_seconds": motion.duration,
                    "fps": motion.fps,
                    "quaternion_norm_min": float(quaternion_norm.min()),
                    "quaternion_norm_max": float(quaternion_norm.max()),
                }
            )
        result[category] = records
    result["tracking_files"] = len(libraries.tracking)
    result["recovery_files"] = len(libraries.recovery)
    result["tracking_frames"] = sum(record["frames"] for record in result["tracking"])
    result["recovery_frames"] = sum(record["frames"] for record in result["recovery"])
    recovery_clips = load_segmented_recovery_motions(
        libraries.recovery, RecoverySegmentationCfg()
    )
    result["recovery_atomic_clips"] = len(recovery_clips)
    result["recovery_atomic_clip_duration_seconds"] = {
        "minimum": min(motion.duration for motion in recovery_clips),
        "median": float(np.median([motion.duration for motion in recovery_clips])),
        "maximum": max(motion.duration for motion in recovery_clips),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
