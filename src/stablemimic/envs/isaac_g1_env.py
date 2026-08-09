"""Complete Isaac Lab tracking/recovery environment for StableMimic G1."""

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from stablemimic.config import load_config
from stablemimic.core.geometry import (
    euler_xyz_to_quaternion_wxyz,
    projected_gravity_from_xyzw,
    quaternion_multiply_wxyz,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from stablemimic.core.observations import (
    ObservationHistory,
    build_motion_command,
    build_proprioception,
    build_recovery_reference,
)
from stablemimic.core.phases import MotionPhase, PhaseState
from stablemimic.motion.torch_library import (
    FailureAdaptiveSampler,
    TorchMotionSample,
    load_torch_motion_libraries,
)
from stablemimic.rewards import KinematicState, whole_body_reward
from stablemimic.sim import build_lafan1_g1_joint_mapping

from .isaac_g1_env_cfg import StableMimicG1EnvCfg


class StableMimicG1Env(DirectRLEnv):
    cfg: StableMimicG1EnvCfg

    def __init__(self, cfg: StableMimicG1EnvCfg, render_mode: str | None = None, **kwargs):
        self._repository_config = load_config(Path(cfg.stablemimic_config_path))
        super().__init__(cfg, render_mode, **kwargs)
        mapping = build_lafan1_g1_joint_mapping(self._robot.joint_names)
        self._body_joint_ids = torch.tensor(mapping.csv_to_sim, dtype=torch.long, device=self.device)
        if self._robot.joint_names != self._reference_robot.joint_names:
            raise ValueError("controlled and reference G1 articulations have different joint orders")
        torso_ids, _ = self._robot.find_bodies("torso_link")
        if len(torso_ids) != 1:
            raise ValueError(f"expected one torso_link, got {torso_ids}")
        self._torso_body_ids = torso_ids
        self._motions = load_torch_motion_libraries(cfg.data_root, self.device)
        # The USD uses instanced collision prims that cannot be disabled through
        # CollisionPropertiesCfg. Keep the FK articulation far above the scene
        # and subtract this deterministic offset at the reward boundary.
        self._reference_z_offset = 100.0
        self._recovery_sampler = FailureAdaptiveSampler(self._motions.recovery)
        self._phase_state = PhaseState.create(
            self.num_envs,
            self.device,
            transition_duration=cfg.transition_duration_s,
            error_timeout=cfg.recovery_error_timeout_s,
        )
        self._history = ObservationHistory(self.num_envs, self.device)
        self._actions = torch.zeros(self.num_envs, 29, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._tracking_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._recovery_motion_ids = torch.zeros_like(self._tracking_motion_ids)
        self._tracking_times = torch.zeros(self.num_envs, device=self.device)
        self._recovery_times = torch.zeros(self.num_envs, device=self.device)
        self._tracking_xy_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self._recovery_xy_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self._history_needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sequence_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._phase_failed = torch.zeros_like(self._sequence_done)
        self._latest_gate_observation = torch.zeros(self.num_envs, 372, device=self.device)
        self._latest_gate_target = torch.zeros(self.num_envs, 2, device=self.device)
        self._latest_events = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in (
                "recovery_success",
                "recovery_failure",
                "transition_completed",
                "sequence_termination",
                "unrecoverable_fall_termination",
                "timeout",
            )
        }
        self._tracking_sample = self._motions.tracking.sample(self._tracking_motion_ids, self._tracking_times)
        self._recovery_sample = self._motions.recovery.sample(self._recovery_motion_ids, self._recovery_times)
        self._active_sample = self._tracking_sample
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "total", "tracking", "root_position", "root_orientation", "body_position",
                "body_orientation", "body_linear_velocity", "body_angular_velocity", "regularization",
            )
        }

    @property
    def gate_observations(self) -> torch.Tensor:
        return self._latest_gate_observation

    @property
    def gate_targets(self) -> torch.Tensor:
        return self._latest_gate_target

    @property
    def phases(self) -> torch.Tensor:
        return self._phase_state.phase

    @property
    def latest_events(self) -> dict[str, torch.Tensor]:
        """Per-environment events produced by the most recent policy step."""
        return self._latest_events

    def training_state(self) -> dict[str, torch.Tensor]:
        return {"recovery_failure_histogram": self._recovery_sampler.failures.detach().cpu()}

    def load_training_state(self, state: dict[str, torch.Tensor]) -> None:
        histogram = state.get("recovery_failure_histogram")
        if histogram is not None:
            if histogram.shape != self._recovery_sampler.failures.shape:
                raise ValueError("recovery failure histogram shape mismatch")
            self._recovery_sampler.failures.copy_(histogram.to(self.device))

    def set_push_forces(self, forces_world: torch.Tensor) -> None:
        """Set persistent world-frame torso forces for evaluation."""
        if forces_world.shape != (self.num_envs, 3):
            raise ValueError(f"forces_world must have shape ({self.num_envs}, 3)")
        forces = forces_world[:, None, :]
        self._robot.set_external_force_and_torque(
            forces,
            torch.zeros_like(forces),
            body_ids=self._torso_body_ids,
            is_global=True,
        )

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._reference_robot = Articulation(self.cfg.reference_robot)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["reference_robot"] = self._reference_robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-self.cfg.action_clip, self.cfg.action_clip))
        self._processed_actions = (
            self._robot.data.default_joint_pos[:, self._body_joint_ids]
            + self.cfg.action_scale * self._actions
        )
        self._write_reference_at_current_time()
        next_tracking = self._tracking_times + self.step_dt
        next_recovery = self._recovery_times + self.step_dt
        self._sequence_done = torch.where(
            self._phase_state.phase == int(MotionPhase.RECOVERY),
            next_recovery >= self._motions.recovery.durations[self._recovery_motion_ids],
            next_tracking >= self._motions.tracking.durations[self._tracking_motion_ids],
        )
        self._tracking_times = torch.minimum(
            next_tracking, self._motions.tracking.durations[self._tracking_motion_ids]
        )
        self._recovery_times = torch.minimum(
            next_recovery, self._motions.recovery.durations[self._recovery_motion_ids]
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._body_joint_ids)

    def _write_reference_at_current_time(self) -> None:
        self._tracking_sample = self._motions.tracking.sample(self._tracking_motion_ids, self._tracking_times)
        self._recovery_sample = self._motions.recovery.sample(self._recovery_motion_ids, self._recovery_times)
        recovery = self._phase_state.phase == int(MotionPhase.RECOVERY)
        self._active_sample = self._select_sample(self._tracking_sample, self._recovery_sample, recovery)
        xy_offset = torch.where(recovery[:, None], self._recovery_xy_offset, self._tracking_xy_offset)
        root_position = self._active_sample.root_pos.clone()
        root_position[:, :2] += xy_offset
        root_position += self._terrain.env_origins
        root_position[:, 2] += self._reference_z_offset
        root_pose = torch.cat((root_position, xyzw_to_wxyz(self._active_sample.root_quat_xyzw)), dim=-1)
        root_velocity = torch.cat(
            (self._active_sample.root_lin_vel_world, self._active_sample.root_ang_vel_world), dim=-1
        )
        joint_position = self._reference_robot.data.default_joint_pos.clone()
        joint_velocity = torch.zeros_like(self._reference_robot.data.default_joint_vel)
        joint_position[:, self._body_joint_ids] = self._active_sample.joint_pos
        joint_velocity[:, self._body_joint_ids] = self._active_sample.joint_vel
        self._reference_robot.write_root_pose_to_sim(root_pose)
        self._reference_robot.write_root_velocity_to_sim(root_velocity)
        self._reference_robot.write_joint_state_to_sim(joint_position, joint_velocity)

    @staticmethod
    def _select_sample(
        tracking: TorchMotionSample, recovery: TorchMotionSample, recovery_mask: torch.Tensor
    ) -> TorchMotionSample:
        return TorchMotionSample(**{
            name: torch.where(recovery_mask[:, None], getattr(recovery, name), getattr(tracking, name))
            for name in tracking.__dataclass_fields__
        })

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._tracking_sample = self._motions.tracking.sample(self._tracking_motion_ids, self._tracking_times)
        self._recovery_sample = self._motions.recovery.sample(self._recovery_motion_ids, self._recovery_times)
        proprio = build_proprioception(
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos[:, self._body_joint_ids]
            - self._robot.data.default_joint_pos[:, self._body_joint_ids],
            self._robot.data.joint_vel[:, self._body_joint_ids],
            self._actions,
        )
        uncorrupted = proprio
        if self.cfg.observation_noise_std > 0.0:
            proprio = proprio + self.cfg.observation_noise_std * torch.randn_like(proprio)
        future_1 = self._motions.tracking.sample(
            self._tracking_motion_ids,
            torch.minimum(
                self._tracking_times + 0.10,
                self._motions.tracking.durations[self._tracking_motion_ids],
            ),
        )
        future_2 = self._motions.tracking.sample(
            self._tracking_motion_ids,
            torch.minimum(
                self._tracking_times + 0.20,
                self._motions.tracking.durations[self._tracking_motion_ids],
            ),
        )
        phase_angle = 2.0 * math.pi * self._tracking_sample.normalized_phase
        command = build_motion_command(
            self._tracking_sample.joint_pos,
            self._tracking_sample.joint_vel,
            self._tracking_sample.root_lin_vel_world,
            self._tracking_sample.root_ang_vel_world,
            self._tracking_sample.root_pos[:, 2:3],
            projected_gravity_from_xyzw(self._tracking_sample.root_quat_xyzw),
            future_1.joint_pos,
            future_2.joint_pos,
            torch.cat((torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1),
        )
        successor_time = torch.minimum(
            self._recovery_times + self.step_dt,
            self._motions.recovery.durations[self._recovery_motion_ids],
        )
        successor = self._motions.recovery.sample(self._recovery_motion_ids, successor_time)
        hidden = build_recovery_reference(
            successor.root_pos,
            successor.root_quat_xyzw,
            successor.root_lin_vel_world,
            successor.root_ang_vel_world,
            successor.joint_pos,
            successor.normalized_phase,
        )
        recovery_mask = self._phase_state.phase == int(MotionPhase.RECOVERY)
        hidden = torch.where(recovery_mask[:, None], hidden, torch.zeros_like(hidden))
        reset_ids = self._history_needs_reset.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self._history.reset(
                reset_ids,
                proprio[reset_ids],
                command[reset_ids],
                uncorrupted[reset_ids],
                hidden[reset_ids],
            )
            self._history_needs_reset[reset_ids] = False
        observations = self._history.append(proprio, command, uncorrupted, hidden)
        self._latest_gate_observation = observations.gate
        self._latest_gate_target = self._phase_state.gate_target()
        return {"policy": observations.actor, "critic": observations.critic}

    def _get_rewards(self) -> torch.Tensor:
        current = self._kinematic_state(self._robot)
        target = self._kinematic_state(self._reference_robot, reference=True)
        recovery = self._phase_state.phase == int(MotionPhase.RECOVERY)
        breakdown = whole_body_reward(
            current,
            target,
            recovery,
            self._repository_config.reward,
            actions=self._actions,
            previous_actions=self._previous_actions,
            torques=self._robot.data.applied_torque[:, self._body_joint_ids],
            joint_velocities=self._robot.data.joint_vel[:, self._body_joint_ids],
            joint_positions=self._robot.data.joint_pos[:, self._body_joint_ids],
            soft_joint_limits=(
                self._robot.data.soft_joint_pos_limits[:, self._body_joint_ids, 0],
                self._robot.data.soft_joint_pos_limits[:, self._body_joint_ids, 1],
            ),
        )
        maximum = sum(
            getattr(self._repository_config.reward, name).weight
            for name in (
                "root_position", "root_orientation", "body_position", "body_orientation",
                "body_linear_velocity", "body_angular_velocity",
            )
        )
        similarity = breakdown.tracking / maximum
        similarity = torch.where(
            recovery,
            similarity / self._repository_config.reward.recovery_multiplier,
            similarity,
        ).clamp(0.0, 1.0)
        old_phase = self._phase_state.phase.clone()
        self._phase_failed = self._phase_state.update(
            similarity,
            1.0 - similarity,
            self.step_dt,
            self.cfg.recovery_success_threshold,
        )
        self._recovery_sampler.record_failures(
            self._recovery_motion_ids, self._recovery_times, self._phase_failed
        )
        began_transition = (old_phase == int(MotionPhase.RECOVERY)) & (
            self._phase_state.phase == int(MotionPhase.TRANSITION)
        )
        completed_transition = (old_phase == int(MotionPhase.TRANSITION)) & (
            self._phase_state.phase == int(MotionPhase.TRACKING)
        )
        self._latest_events["recovery_success"].copy_(began_transition)
        self._latest_events["transition_completed"].copy_(completed_transition)
        if began_transition.any():
            tracking_root = self._tracking_sample.root_pos[began_transition, :2]
            robot_xy = self._robot.data.root_pos_w[began_transition, :2]
            env_xy = self._terrain.env_origins[began_transition, :2]
            self._tracking_xy_offset[began_transition] = robot_xy - env_xy - tracking_root
        tracking_weight, recovery_weight = self._phase_state.reward_weights()
        phase_weight = torch.where(recovery, recovery_weight, tracking_weight)
        reward = (breakdown.tracking * phase_weight + breakdown.regularization) * self.step_dt
        for name in self._episode_sums:
            value = reward if name == "total" else getattr(breakdown, name) * self.step_dt
            self._episode_sums[name] += value
        return reward

    def _kinematic_state(self, robot: Articulation, reference: bool = False) -> KinematicState:
        root_position = robot.data.root_pos_w
        body_position = robot.data.body_pos_w
        if reference:
            offset = torch.zeros_like(root_position)
            offset[:, 2] = self._reference_z_offset
            root_position = root_position - offset
            body_position = body_position - offset[:, None, :]
        return KinematicState(
            root_position=root_position,
            root_quaternion_xyzw=wxyz_to_xyzw(robot.data.root_quat_w),
            body_position=body_position,
            body_quaternion_xyzw=wxyz_to_xyzw(robot.data.body_quat_w),
            body_linear_velocity=robot.data.body_lin_vel_w,
            body_angular_velocity=robot.data.body_ang_vel_w,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tracking = self._phase_state.phase == int(MotionPhase.TRACKING)
        unrecoverable_fall = tracking & (self._robot.data.root_pos_w[:, 2] < 0.18)
        terminated = self._phase_failed | self._sequence_done | unrecoverable_fall
        self._latest_events["recovery_failure"].copy_(self._phase_failed)
        self._latest_events["sequence_termination"].copy_(self._sequence_done)
        self._latest_events["unrecoverable_fall_termination"].copy_(unrecoverable_fall)
        self._latest_events["timeout"].copy_(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        self._reference_robot.reset(env_ids)
        super()._reset_idx(env_ids)
        count = env_ids.numel()
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._sequence_done[env_ids] = False
        self._phase_failed[env_ids] = False
        self._tracking_motion_ids[env_ids] = self._motions.tracking.random_ids(count)
        self._tracking_times[env_ids] = self._motions.tracking.random_times(
            self._tracking_motion_ids[env_ids]
        )
        recovery_ids, recovery_times = self._recovery_sampler.sample(count)
        self._recovery_motion_ids[env_ids] = recovery_ids
        self._recovery_times[env_ids] = recovery_times
        recovery_reset = torch.rand(count, device=self.device) >= self.cfg.tracking_reset_probability
        self._phase_state.reset(env_ids, recovery_reset)
        tracking_sample = self._motions.tracking.sample(
            self._tracking_motion_ids[env_ids], self._tracking_times[env_ids]
        )
        recovery_sample = self._motions.recovery.sample(recovery_ids, recovery_times)
        reset_sample = self._select_sample(tracking_sample, recovery_sample, recovery_reset)
        origins = self._terrain.env_origins[env_ids]
        self._tracking_xy_offset[env_ids] = -tracking_sample.root_pos[:, :2]
        self._recovery_xy_offset[env_ids] = -recovery_sample.root_pos[:, :2]

        root_position = reset_sample.root_pos.clone()
        root_position[:, :2] += torch.where(
            recovery_reset[:, None], self._recovery_xy_offset[env_ids], self._tracking_xy_offset[env_ids]
        )
        root_position += origins
        root_position[:, :2] += self._uniform_noise("root_xy", (count, 2))
        root_position[:, 2:3] += self._uniform_noise("root_z", (count, 1))
        orientation_noise = torch.cat((
            self._uniform_noise("roll_pitch", (count, 2)),
            self._uniform_noise("yaw", (count, 1)),
        ), dim=-1)
        delta_quaternion = euler_xyz_to_quaternion_wxyz(
            orientation_noise[:, 0], orientation_noise[:, 1], orientation_noise[:, 2]
        )
        root_quaternion = quaternion_multiply_wxyz(
            delta_quaternion, xyzw_to_wxyz(reset_sample.root_quat_xyzw)
        )
        root_pose = torch.cat((root_position, root_quaternion), dim=-1)
        root_velocity = torch.cat((reset_sample.root_lin_vel_world, reset_sample.root_ang_vel_world), dim=-1)
        root_velocity[:, :2] += self._uniform_noise("linear_velocity_xy", (count, 2))
        root_velocity[:, 2:3] += self._uniform_noise("linear_velocity_z", (count, 1))
        root_velocity[:, 3:5] += self._uniform_noise("angular_velocity_roll_pitch", (count, 2))
        root_velocity[:, 5:6] += self._uniform_noise("angular_velocity_yaw", (count, 1))
        joint_position = self._robot.data.default_joint_pos[env_ids].clone()
        joint_velocity = self._robot.data.default_joint_vel[env_ids].clone()
        joint_position[:, self._body_joint_ids] = reset_sample.joint_pos + self._uniform_noise(
            "joint_position", (count, 29)
        )
        joint_velocity[:, self._body_joint_ids] = reset_sample.joint_vel
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self._robot.write_joint_state_to_sim(joint_position, joint_velocity, None, env_ids)
        self._history_needs_reset[env_ids] = True

        log = {}
        for name, values in self._episode_sums.items():
            log[f"Episode_Reward/{name}"] = values[env_ids].mean() / self.max_episode_length_s
            values[env_ids] = 0.0
        log["Episode_Phase/recovery_reset_fraction"] = recovery_reset.float().mean()
        self.extras["log"] = log

    def _uniform_noise(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        lower, upper = self._repository_config.reset_noise[name]
        return torch.empty(shape, device=self.device).uniform_(lower, upper)
