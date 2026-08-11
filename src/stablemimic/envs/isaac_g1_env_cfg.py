"""Isaac Lab DirectRLEnv configuration for StableMimic G1."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import G1_29DOF_CFG


@configclass
class StableMimicG1EnvCfg(DirectRLEnvCfg):
    seed = 42
    episode_length_s = 20.0
    decimation = 4
    action_space = 29
    observation_space = 884
    state_space = 1428
    stablemimic_config_path: str = "configs/stablemimic_g1.yaml"
    data_root: str = "/root/gpufree-data/stablemimic_replicate/datasets/lafan1/g1"
    action_scale: float = 0.5
    action_clip: float = 100.0
    tracking_reset_probability: float = 0.5
    transition_duration_s: float = 1.5
    recovery_error_timeout_s: float = 2.0
    recovery_failure_similarity_threshold: float = 0.05
    recovery_terminal_similarity_threshold: float = 0.70
    recovered_like_height_ratio: float = 0.8
    tracking_fall_recovery_enabled: bool = True
    tracking_fall_height_threshold: float = 0.5
    tracking_fall_tilt_degrees: float = 60.0
    recovery_match_joint_weight: float = 1.0
    recovery_match_height_weight: float = 4.0
    recovery_match_gravity_weight: float = 2.0
    recovery_static_reset_probability: float = 0.0
    fall_recovery_probability: float = 1.0
    observation_noise_std: float = 0.0
    enable_early_termination: bool = True
    # Evaluation-only reset controls. Training keeps all three defaults.
    recovery_reset_at_fallen_state: bool = False
    recovery_reset_zero_velocity: bool = False
    reset_noise_enabled: bool = True

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim.physics_material,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    reference_robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/ReferenceRobot")
    reference_robot.spawn.rigid_props.disable_gravity = True
    reference_robot.spawn.activate_contact_sensors = False
    reference_robot.init_state.pos = (0.0, 0.0, 100.0)
    terminal_reference_robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/TerminalReferenceRobot"
    )
    terminal_reference_robot.spawn.rigid_props.disable_gravity = True
    terminal_reference_robot.spawn.activate_contact_sensors = False
    terminal_reference_robot.init_state.pos = (0.0, 0.0, 200.0)
