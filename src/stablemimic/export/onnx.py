"""Export exactly one deployable Actor with normalization and soft Gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from stablemimic.core.observations import ACTOR_OBS_DIM, GATE_OBS_DIM
from stablemimic.config import load_config
from stablemimic.models import StableMimicAgent
from stablemimic.motion.lafan1 import LAFAN1_G1_JOINT_NAMES


class DeploymentActor(nn.Module):
    def __init__(self, agent: StableMimicAgent):
        super().__init__()
        self.agent = agent

    def forward(
        self, actor_observation: torch.Tensor, gate_observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor = self.agent.actor_normalizer(actor_observation)
        gate = self.agent.gate_normalizer(gate_observation)
        output = self.agent.actor(actor, gate)
        return output.mean, output.gate_weights


def export_actor_onnx(
    agent: StableMimicAgent,
    output_path: str | Path,
    *,
    config_path: str | Path,
    opset_version: int = 17,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DeploymentActor(agent.eval()).cpu()
    actor_example = torch.zeros(1, ACTOR_OBS_DIM)
    gate_example = torch.zeros(1, GATE_OBS_DIM)
    torch.onnx.export(
        wrapper,
        (actor_example, gate_example),
        output,
        input_names=["actor_observation", "gate_observation"],
        output_names=["joint_target_mean", "gate_weights"],
        dynamic_axes={
            "actor_observation": {0: "batch"}, "gate_observation": {0: "batch"},
            "joint_target_mean": {0: "batch"}, "gate_weights": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    config_bytes = Path(config_path).read_bytes()
    config = load_config(config_path)
    metadata = {
        "actor_observation_dim": ACTOR_OBS_DIM,
        "gate_observation_dim": GATE_OBS_DIM,
        "action_dim": 29,
        "history_length": 4,
        "joint_names": list(LAFAN1_G1_JOINT_NAMES),
        "quaternion_convention": "xyzw at dataset/observation boundary",
        "policy_frequency_hz": 50.0,
        "action_scale": config.environment.action_scale,
        "action_interface": "default_joint_position + action_scale * clip(joint_target_mean, -1, 1)",
        "use_default_joint_offset": True,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "contains_critic": False,
        "contains_recovery_reference": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output
