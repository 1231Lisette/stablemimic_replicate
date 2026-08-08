#!/usr/bin/env python3
"""Check ONNX structure and compare it numerically with the PyTorch Actor."""

from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from stablemimic.config import load_config
from stablemimic.core.observations import ACTOR_OBS_DIM, GATE_OBS_DIM
from stablemimic.export.onnx import DeploymentActor
from stablemimic.models import StableMimicActor, StableMimicAgent, StableMimicCritic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/stablemimic_g1.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--onnx", required=True)
parser.add_argument("--batch", type=int, default=3)
args = parser.parse_args()


def main() -> None:
    if args.batch <= 0:
        parser.error("--batch must be positive")
    config = load_config(args.config)
    agent = StableMimicAgent(
        StableMimicActor(
            config.model.expert_hidden_dims, config.model.gate_hidden_dims,
            config.model.activation, config.model.initial_std,
        ),
        StableMimicCritic(config.model.critic_hidden_dims, config.model.activation),
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent.load_state_dict(payload["agent"])
    wrapper = DeploymentActor(agent.eval())
    generator = torch.Generator().manual_seed(1234)
    actor_input = torch.randn(args.batch, ACTOR_OBS_DIM, generator=generator)
    gate_input = torch.randn(args.batch, GATE_OBS_DIM, generator=generator)
    with torch.no_grad():
        torch_mean, torch_gate = wrapper(actor_input, gate_input)

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    if [value.name for value in model.graph.input] != ["actor_observation", "gate_observation"]:
        raise AssertionError("ONNX has unexpected deployment inputs")
    if any("critic" in value.name.lower() for value in model.graph.initializer):
        raise AssertionError("ONNX unexpectedly contains Critic parameters")
    evaluator = ReferenceEvaluator(model)
    onnx_mean, onnx_gate = evaluator.run(
        None,
        {
            "actor_observation": actor_input.numpy(),
            "gate_observation": gate_input.numpy(),
        },
    )
    mean_error = float(np.max(np.abs(onnx_mean - torch_mean.numpy())))
    gate_error = float(np.max(np.abs(onnx_gate - torch_gate.numpy())))
    if mean_error > 1.0e-4 or gate_error > 1.0e-5:
        raise AssertionError(f"ONNX parity failed: mean={mean_error}, gate={gate_error}")
    if not np.allclose(onnx_gate.sum(-1), 1.0, atol=1.0e-6):
        raise AssertionError("ONNX gate weights do not sum to one")
    print(f"[PASS] ONNX parity: max_mean_error={mean_error:.3e}, max_gate_error={gate_error:.3e}")


if __name__ == "__main__":
    main()
