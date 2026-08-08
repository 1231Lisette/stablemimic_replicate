#!/usr/bin/env python3
"""Export a trained StableMimic checkpoint to a deployable ONNX Actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stablemimic.config import load_config
from stablemimic.export import export_actor_onnx
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/stablemimic_g1.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def main() -> None:
    config = load_config(args.config)
    agent = StableMimicAgent(
        StableMimicActor(
            config.model.expert_hidden_dims,
            config.model.gate_hidden_dims,
            config.model.activation,
            config.model.initial_std,
        ),
        StableMimicCritic(config.model.critic_hidden_dims, config.model.activation),
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload["agent"])
    output = export_actor_onnx(agent, args.output, config_path=Path(args.config))
    print(f"[PASS] Exported deployable Actor: {output}")


if __name__ == "__main__":
    main()
