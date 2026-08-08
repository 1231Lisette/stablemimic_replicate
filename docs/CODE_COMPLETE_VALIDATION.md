# Code-complete validation record

Date: 2026-08-09

This record validates software integration only. The generated checkpoint has
two PPO updates and is not a trained StableMimic policy.

## Passed checks

- 19/19 remote unit/integration tests with all 14 real CSV files.
- Actor/Gate/Critic history dimensions: 884 / 372 / 1428.
- Recovery successor changes Critic observation but leaves Actor and Gate bit-identical.
- Perfect-state reward exceeds perturbed-state reward; recovery reward is invariant to world XY translation.
- Two 29-D Expert means, finite two-way softmax weights summing to one, and explicit soft-fusion equality.
- Finite synthetic rollout, GAE, PPO, Gate CE, transition, consistency, and alignment losses.
- Isaac CUDA fresh smoke: 4 environments x 24 steps, one five-epoch PPO update, checkpoint saved.
- Isaac CUDA resume smoke: restored Actor/Critic/Gate/normalizers/optimizer/failure histogram and advanced from iteration 1 to 2.
- Deterministic Isaac checkpoint evaluation completed.
- 100-environment matched-push smoke completed with 25 pushes in each +/-x and +/-y direction, 525--575 N for 0.2 s.
- ONNX structure check and PyTorch parity passed; maximum mean error was 3.576e-7 and maximum Gate error was 5.960e-8.

## Runtime artifacts

Artifacts intentionally remain on the data disk rather than in Git:

```text
/root/gpufree-data/stablemimic_replicate/runs/final_code_smoke/
├── latest.pt
├── metrics.jsonl
├── actor.onnx
├── actor.json
└── push_smoke_tracking_only.json
```

## Not validated yet

- Long-running PPO convergence and paper-level performance.
- Maximum feasible parallel environment count on the 24 GB RTX 4090.
- MuJoCo numerical results. The adapter is implemented, but the server has no
  verified G1 MJCF/actuator/PD package and no installed `mujoco` Python extra.
- Real-robot safety, latency, actuator mapping, and deployment.

These remaining items require training time, an externally selected MuJoCo
asset, or hardware authority; they are not missing code paths silently treated
as passed results.
