# Gate 0 Remote Environment Audit

Date: 2026-08-09

## Reproduction environment

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Isaac Sim: `5.1.0-rc.19+release.26219.9c81211b.gl`
- Isaac Lab package: `0.54.2`
- PyTorch: `2.7.0+cu128`
- NumPy: `1.26.0`
- CUDA available: `True`
- Isaac Lab path: `/workspace/isaaclab`
- Isaac Sim path: `/isaac-sim`

The installed Isaac Lab tree has no Git metadata. Package and simulator versions are recorded, but an exact upstream Isaac Lab commit cannot be recovered from this installation.

## Simulator fact boundary

Isaac Lab is a reproduction engineering choice. StableMimic identifies MuJoCo as its unified evaluation environment but does not state which simulator was used for training.

## G1 asset choice

The existing Isaac Lab velocity task uses `G1_MINIMAL_CFG`. Its observed naming patterns include `torso_joint` and elbow pitch/roll joints, so it must not be assumed to match the public LAFAN1 29-joint schema.

Phase 1 uses `G1_29DOF_CFG` as the candidate asset. Runtime inspection shows that its `g1.usd` articulation contains the 29 LAFAN1 body joints plus 14 hand joints (43 total), despite the configuration name. The viewer requires every LAFAN1 joint exactly once, rejects any unknown non-hand extra joint, builds an explicit 29-to-43 permutation, and leaves the 14 hand joints at their default state. The policy/action space must remain 29-D.

## Existing reusable settings

The generic velocity environment already demonstrates:

- simulation timestep: 0.005 s;
- action decimation: 4;
- policy/control frequency: 50 Hz;
- horizon: 20 s;
- 4096 environments;
- joint-position action support;
- contact sensing and common regularization rewards;
- RSL-RL PPO integration.

These are useful implementation references. They are not an existing StableMimic task and are not treated as proof of the paper's training simulator.

## Headless runtime note

A minimal CUDA headless run successfully completed application launch, `SimulationContext.reset()`, and one physics step. In this container, Isaac Sim 5.1 can remain blocked in `simulation_app.close()` after the successful body of a script. Command-line tools therefore retain the normal close call but add a documented 15-second watchdog that forces process teardown only if close does not return. The watchdog preserves a non-zero exit when an exception is active. This container-specific workaround must be revisited before long-running automation.
