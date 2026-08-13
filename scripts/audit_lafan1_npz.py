#!/usr/bin/env python3
"""Audit a converted G1 NPZ dataset against its source CSVs and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES, load_lafan1_csv
from stablemimic.motion.npz import load_npz_arrays, resample_motion, validate_standard_npz


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--npz-dir", required=True)
args = parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(args.npz_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", [])
    outputs = sorted(root.glob("*.npz"))
    if len(records) != len(outputs) or len(records) != 14:
        raise ValueError(
            f"expected 14 manifest records and NPZs, got {len(records)} and {len(outputs)}"
        )
    total_input = total_output = total_bytes = 0
    for record in records:
        source = Path(record["source"])
        output = Path(record["output"])
        if output.parent.resolve() != root:
            raise ValueError(f"manifest output escapes audited directory: {output}")
        if sha256(source) != record["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        if sha256(output) != record["output_sha256"]:
            raise ValueError(f"output hash mismatch: {output}")
        arrays = load_npz_arrays(output)
        frames = validate_standard_npz(arrays)
        motion = load_lafan1_csv(source, fps=float(manifest["input_fps"]))
        expected = resample_motion(motion, float(manifest["output_fps"]))
        if frames != expected.num_frames or frames != int(record["output_frames"]):
            raise ValueError(f"frame count mismatch: {output}")
        if tuple(arrays["joint_names"].tolist()) != LAFAN1_G1_JOINT_NAMES:
            raise ValueError(f"joint-name order mismatch: {output}")
        np.testing.assert_allclose(arrays["joint_pos"], expected.joint_pos, atol=2.0e-6)
        np.testing.assert_allclose(arrays["joint_vel"], expected.joint_vel, atol=2.0e-5)
        np.testing.assert_allclose(arrays["source_root_pos"], expected.root_pos, atol=2.0e-6)
        np.testing.assert_allclose(
            arrays["source_root_quat_wxyz"], expected.root_quat_wxyz, atol=2.0e-6
        )
        np.testing.assert_allclose(
            arrays["body_pos_w"][:, 0], arrays["source_root_pos"], atol=2.0e-4
        )
        quaternion_norm_error = float(
            np.max(np.abs(np.linalg.norm(arrays["body_quat_w"], axis=-1) - 1.0))
        )
        if quaternion_norm_error > 2.0e-4:
            raise ValueError(f"body quaternion norm error {quaternion_norm_error}: {output}")
        total_input += motion.num_frames
        total_output += frames
        total_bytes += output.stat().st_size
        print(
            f"[PASS] {output.name}: input={motion.num_frames}, output={frames}, "
            f"bodies={arrays['body_pos_w'].shape[1]}",
            flush=True,
        )
    print(
        f"[PASS] NPZ dataset: files={len(records)}, input_frames={total_input}, "
        f"output_frames={total_output}, bytes={total_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
