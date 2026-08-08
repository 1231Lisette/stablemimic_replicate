# stablemimic_replicate

Reproduction of the Unitree G1 + retargeted LAFAN1 StableMimic baseline described in arXiv:2608.02385.

## Status

The repository is currently at **Gate 0 / Phase 1**:

- repository and public dataset audit completed;
- simulator-independent LAFAN1 CSV loading and continuous-time sampling foundation;
- Isaac Lab G1 29-DoF reference visualizer;
- no PPO training, recovery policy, mixture of experts, or deployment code yet.

## Fact boundary

The paper states that MuJoCo is the common evaluation simulator, but it does not identify the training simulator. This repository uses the server's Isaac Lab installation as a **reproduction engineering choice**, not as a claim about the authors' training stack.

## Server layout

```text
Code: /root/gpufree-share/stablemimic_replicate
Data: /root/gpufree-data/stablemimic_replicate
```

The first-stage dataset is expected at:

```text
/root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
```

Only these files are used initially:

- tracking: `dance*.csv`
- recovery references: `fallAndGetUp*.csv`

## Quick checks

Run the simulator-independent unit and integration tests with Isaac Lab's Python:

```bash
cd /root/gpufree-share/stablemimic_replicate
PYTHONPATH=src LAFAN1_G1_ROOT=/root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1 \
  /workspace/isaaclab/isaaclab.sh -p -m unittest discover -s tests -v
```

Audit the real CSV library:

```bash
cd /root/gpufree-share/stablemimic_replicate
PYTHONPATH=src /workspace/isaaclab/isaaclab.sh -p scripts/audit_lafan1.py \
  --data-root /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1
```

Run a short headless G1 asset/mapping smoke test:

```bash
cd /root/gpufree-share/stablemimic_replicate
PYTHONPATH=src /workspace/isaaclab/isaaclab.sh -p scripts/visualize_lafan1_g1.py \
  --file /root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1/dance1_subject1.csv \
  --headless --max-steps 5
```

The audited Isaac Sim 5.1 container can block during application shutdown after successful simulation. CLI utilities include a documented 15-second shutdown watchdog so completed runs terminate and release the GPU.

For interactive viewing, omit `--headless` and use the server's supported display/livestream method.

## Data format

Each raw CSV row has no header and contains 36 values:

```text
root position XYZ + root quaternion QX QY QZ QW + 29 joint positions
```

The source data is 30 FPS. Policy/reference access is time-based through `MotionReference.sample(t)`; it is never advanced with a simple one-frame-per-policy-step rule.
