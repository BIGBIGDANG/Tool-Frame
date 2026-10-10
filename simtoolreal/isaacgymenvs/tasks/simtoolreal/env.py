# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from copy import copy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgym import gymapi, gymtorch, gymutil
import torch
from torch import Tensor
# isort: on

import json

from isaacgymenvs.utils.torch_jit_utils import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.simtoolreal.utils import (
    populate_dof_properties,
)
from isaacgymenvs.utils.torch_jit_utils import (
    get_axis_params,
    quat_diff_rad,
    quat_rotate,
    scale,
    tensor_clamp,
    to_torch,
    torch_rand_float,
    unscale,
)

def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"

class SimToolReal(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        # Goal related variables
        self.goal_object_indices = []
        self.goal_assets = []
        self.scene_actor_specs = list(
            cfg["env"].get("sceneActors", []) or []
        )
        self.scene_assets = []
        self.scene_actor_indices = []
        self.scene_actor_init_states = []
        self.hide_goal_object_visual = bool(
            cfg["env"].get("hideGoalObjectVisual", False)
        )
        self.goal_object_visual_color = self._goal_object_visual_color_from_cfg()
        self.disable_default_ground_plane = bool(
            cfg["env"].get("disableDefaultGroundPlane", False)
        )
        self.goal_sampling_type = cfg["env"]["goalSamplingType"]
        self.target_volume_region_scale = cfg["env"]["targetVolumeRegionScale"]
        self.delta_goal_distance = cfg["env"]["deltaGoalDistance"]
        self.delta_rotation_degrees = cfg["env"]["deltaRotationDegrees"]

        self.frame_since_restart: int = (
            0  # number of control steps since last restart across all actors
        )

        self.robot_asset_file: str = self.cfg["env"]["asset"]["robot"]

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.privileged_actions = self.cfg["env"]["privilegedActions"]
        self.privileged_actions_torque = self.cfg["env"]["privilegedActionsTorque"]

        # 4 joints for index, middle, ring, and thumb and 7 for kuka arm
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_fingertips

        if self.use_sharpa:
            self.num_fingertips = 5
            self.num_hand_dofs = 22
            self.num_finger_dofs = None  # Different per finger
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_robot_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            self.num_robot_actions += 3

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.hand_actions_penalty_scale = self.cfg["env"]["handActionsPenaltyScale"]
        self.object_lin_vel_penalty_scale = self.cfg["env"]["objectLinVelPenaltyScale"]
        self.object_ang_vel_penalty_scale = self.cfg["env"]["objectAngVelPenaltyScale"]
        self.toolframe_orientation_rew_scale = float(
            self.cfg["env"].get("toolFrameOrientationRewScale", 0.0)
        )
        self.toolframe_orientation_rew_sharpness = float(
            self.cfg["env"].get("toolFrameOrientationRewSharpness", 2.0)
        )
        self.use_toolframe_orientation_success = bool(
            self.cfg["env"].get("useToolFrameOrientationSuccess", False)
        )
        self.toolframe_orientation_success_stage_families = list(
            self.cfg["env"].get("toolFrameOrientationSuccessStageFamilies", []) or []
        )
        self.toolframe_orientation_success_tolerance_rad = float(
            self.cfg["env"].get("toolFrameOrientationSuccessToleranceRad", 0.30)
        )

        self.initial_tolerance = self.cfg["env"]["successTolerance"]

        self.success_tolerance = float(self.cfg["env"].get("goalTolerance", self.initial_tolerance))

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.randomize_object_rotation = self.cfg["env"]["randomizeObjectRotation"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"][
            "resetDofPosRandomIntervalFingers"
        ]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        self.force_only_when_lifted = self.cfg["env"].get("forceOnlyWhenLifted", False)

        self.torque_scale = self.cfg["env"].get("torqueScale", 0.0)
        self.torque_prob_range = self.cfg["env"].get("torqueProbRange", [0.001, 0.1])
        self.torque_decay = self.cfg["env"].get("torqueDecay", 0.99)
        self.torque_decay_interval = self.cfg["env"].get("torqueDecayInterval", 0.08)
        self.torque_only_when_lifted = self.cfg["env"].get(
            "torqueOnlyWhenLifted", False
        )

        self.lin_vel_impulse_prob_range = self.cfg["env"].get(
            "linVelImpulseProbRange", [0.001, 0.1]
        )
        self.lin_vel_impulse_scale = self.cfg["env"].get("linVelImpulseScale", 0.0)
        self.lin_vel_impulse_only_when_lifted = self.cfg["env"].get(
            "linVelImpulseOnlyWhenLifted", False
        )

        self.ang_vel_impulse_prob_range = self.cfg["env"].get(
            "angVelImpulseProbRange", [0.001, 0.1]
        )
        self.ang_vel_impulse_scale = self.cfg["env"].get("angVelImpulseScale", 0.0)
        self.ang_vel_impulse_only_when_lifted = self.cfg["env"].get(
            "angVelImpulseOnlyWhenLifted", False
        )

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]

        # 1.0 means keypoints correspond to the corners of the object
        # larger values help the agent to prioritize rotation matching
        self.keypoint_scale = self.cfg["env"]["keypointScale"]

        # size of the object (i.e. cube) before scaling
        self.object_base_size = self.cfg["env"]["objectBaseSize"]

        # whether to sample random object dimensions
        self.with_dof_force_sensors = False
        # create fingertip force-torque sensors
        self.with_fingertip_force_sensors = False
        self.with_table_force_sensor = self.cfg["env"]["withTableForceSensor"]

        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (self.control_freq_inv * self.sim_params.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.asset_files_dict = {
            "table": self.cfg["env"]["asset"]["table"],
        }

        self.keypoints_offsets = self._object_keypoint_offsets()

        self.num_keypoints = len(self.keypoints_offsets)

        self.fingertips = [
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
            "thumb_link_3",
        ]
        self.fingertip_offsets = np.array(
            [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]],
            dtype=np.float32,
        )

        if self.use_sharpa:
            if self.use_right_sharpa:
                self.fingertips = [
                    "right_index_DP",
                    "right_middle_DP",
                    "right_ring_DP",
                    "right_thumb_DP",
                    "right_pinky_DP",
                ]
            elif self.use_left_sharpa:
                self.fingertips = [
                    "left_index_DP",
                    "left_middle_DP",
                    "left_ring_DP",
                    "left_thumb_DP",
                    "left_pinky_DP",
                ]
            else:
                raise ValueError(f"Unknown sharpa type: {self.use_sharpa}")
            self.fingertip_offsets = np.array(
                [
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                    [0.02, 0.002, 0],
                ],
                dtype=np.float32,
                # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
            )
        self.palm_offset = np.array([-0.00, -0.02, 0.16], dtype=np.float32)

        assert self.num_fingertips == len(self.fingertips)

        # can be only "full_state" or "asymmetric"
        self.obs_type = self.cfg["env"]["observationType"]

        if self.obs_type not in ["full_state", "asymmetric"]:
            raise Exception("Unknown type of observations!")

        print("Obs type:", self.obs_type)

        self.obs_type_size_dict = {
            "joint_pos": self.num_hand_arm_dofs,
            "joint_vel": self.num_hand_arm_dofs,
            "prev_action_targets": self.num_hand_arm_dofs,
            "palm_pos": 3,
            "palm_rot": 4,
            "palm_vel": 6,
            "object_rot": 4,
            "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * self.num_fingertips,
            "keypoints_rel_palm": 3 * self.num_keypoints,
            "keypoints_rel_goal": 3 * self.num_keypoints,
            "object_scales": 3,
            "closest_keypoint_max_dist": 1,
            "closest_fingertip_dist": self.num_fingertips,
            "lifted_object": 1,
            "progress": 1,
            "successes": 1,
            "reward": 1,
        }

        self.state_list = self.cfg["env"]["stateList"]
        self.obs_list = self.cfg["env"]["obsList"]

        # assert that all obs in state_list and obs_list are keys of self.obs_type_size_dict
        for obs_type in self.state_list + self.obs_list:
            assert obs_type in self.obs_type_size_dict, (
                f"Obs type {obs_type} not found in obs_type_size_dict"
            )

        # assert that all obs in obs_list are also in state_list
        for obs_type in self.obs_list:
            assert obs_type in self.state_list, (
                f"Obs type {obs_type} not found in state_list but is in obs_list"
            )

        self.full_state_size = sum(
            [self.obs_type_size_dict[obs_type] for obs_type in self.state_list]
        )
        self.full_obs_size = sum(
            [self.obs_type_size_dict[obs_type] for obs_type in self.obs_list]
        )

        self.up_axis = "z"

        self.fingertip_obs = True

        self.cfg["env"]["numStates"] = self.full_state_size
        self.cfg["env"]["numObservations"] = self.full_obs_size
        self.cfg["env"]["numActions"] = self.num_robot_actions

        self.cfg["device_type"] = (
            sim_device.split(":")[0] if sim_device.find(":") != -1 else sim_device
        )
        self.cfg["device_id"] = (
            int(sim_device.split(":")[1]) if sim_device.find(":") != -1 else 0
        )
        self.cfg["headless"] = headless

        # Must subscribe to keyboard events before calling super().__init__()
        self._subscribe_to_keyboard_events()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Index of environment to view in viewer and camera
        self.index_to_view = 0

        # Camera position and target for viewer
        cam_target = gymapi.Vec3(0.0, 0.0, 0.70)
        cam_pos = gymapi.Vec3(0.0, -0.70, 0.80)
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[self.index_to_view], cam_pos, cam_target
            )

        # Initialize the RGB-D observation camera
        self._initialize_camera_sensor(cam_pos=cam_pos, cam_target=cam_target)
        self._modify_render_settings_if_headless()

        # volume to sample target position from
        target_volume_origin = np.array([0, 0.05, 0.8], dtype=np.float32)
        target_volume_extent = np.array(
            [[-0.4, 0.4], [-0.05, 0.3], [-0.12, 0.25]], dtype=np.float32
        )
        if (
            self.cfg["env"]["targetVolumeMins"] is not None
            and self.cfg["env"]["targetVolumeMaxs"] is not None
        ):
            assert (
                len(self.cfg["env"]["targetVolumeMins"])
                == len(self.cfg["env"]["targetVolumeMaxs"])
                == 3
            ), "targetVolumeMins and targetVolumeMaxs must be 3-element lists"
            mins = np.array(self.cfg["env"]["targetVolumeMins"], dtype=np.float32)
            maxs = np.array(self.cfg["env"]["targetVolumeMaxs"], dtype=np.float32)
            new_target_volume_origin = (mins + maxs) / 2
            new_target_volume_range = maxs - mins
            new_target_volume_extent = np.stack(
                [-new_target_volume_range / 2, new_target_volume_range / 2], axis=1
            )
            assert (
                target_volume_origin.shape == new_target_volume_origin.shape == (3,)
            ), (
                f"target_volume_origin.shape: {target_volume_origin.shape}, new_target_volume_origin.shape: {new_target_volume_origin.shape}"
            )
            assert (
                target_volume_extent.shape == new_target_volume_extent.shape == (3, 2)
            ), (
                f"target_volume_extent.shape: {target_volume_extent.shape}, new_target_volume_extent.shape: {new_target_volume_extent.shape}"
            )
            target_volume_origin = new_target_volume_origin
            target_volume_extent = new_target_volume_extent

        self.target_volume_origin = (
            torch.from_numpy(target_volume_origin).to(self.device).float()
        )
        self.target_volume_extent = (
            torch.from_numpy(target_volume_extent).to(self.device).float()
        )

        # Scale the target volume extent by the target volume region scale
        self.target_volume_extent = (
            self.target_volume_extent * self.target_volume_region_scale
        )

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.with_fingertip_force_sensors or self.with_table_force_sensor:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                # self.num_envs, self.num_fingertips * 6
                self.num_envs,
                -1,
                6,
            )
            print(f"force_sensor_tensor: {self.force_sensor_tensor.shape}")
            self.gym.refresh_force_sensor_tensor(self.sim)
            if self.device == "cpu":
                raise ValueError(
                    "Force sensors not supported on CPU, they only gives 0s on CPU"
                )

        if self.with_dof_force_sensors:
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_hand_arm_dofs
            )
            self.gym.refresh_dof_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.hand_arm_default_dof_pos = torch.zeros(
            self.num_hand_arm_dofs, dtype=torch.float, device=self.device
        )

        desired_kuka_pos = torch.tensor(
            [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358]
        )  # pose v1
        if self.use_sharpa:
            desired_kuka_pos = torch.tensor(
                [-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.308]
            )  # same as above but 60 deg offset for the mount

        START_HIGHER = self.cfg["env"]["startArmHigher"]
        if START_HIGHER:
            desired_kuka_pos[1] -= np.deg2rad(10)
            desired_kuka_pos[3] += np.deg2rad(10)

        self.hand_arm_default_dof_pos[:7] = desired_kuka_pos

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_hand_arm_dofs
        ]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_arm_hand_dof_state = self.dof_state.view(
                self.num_envs, -1, 2
            )[:, self.num_hand_arm_dofs :]
            self.blue_robot_arm_hand_dof_pos = self.blue_robot_arm_hand_dof_state[
                ..., 0
            ]
            self.blue_robot_arm_hand_dof_vel = self.blue_robot_arm_hand_dof_state[
                ..., 1
            ]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        self.set_actor_root_state_object_indices: List[Tensor] = []
        self.set_dof_state_object_indices: List[Tensor] = []

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.global_indices = torch.arange(
            self.num_envs * 3, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # true objective value for the whole episode, plus saving values for the previous episode

        # object apply random forces parameters
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )
        self.force_prob_range = to_torch(
            self.force_prob_range, dtype=torch.float, device=self.device
        )
        self.random_force_prob = self._sample_log_uniform(
            min_value=self.force_prob_range[0],
            max_value=self.force_prob_range[1],
            num_samples=self.num_envs,
        )
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        self.torque_decay = to_torch(
            self.torque_decay, dtype=torch.float, device=self.device
        )
        self.torque_prob_range = to_torch(
            self.torque_prob_range, dtype=torch.float, device=self.device
        )
        self.random_torque_prob = self._sample_log_uniform(
            min_value=self.torque_prob_range[0],
            max_value=self.torque_prob_range[1],
            num_samples=self.num_envs,
        )
        self.rb_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.action_torques = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )

        # Random velocity impulses applied to the object
        self.lin_vel_impulse_prob_range = to_torch(
            self.lin_vel_impulse_prob_range, dtype=torch.float, device=self.device
        )
        self.random_lin_vel_impulse_prob = self._sample_log_uniform(
            min_value=self.lin_vel_impulse_prob_range[0],
            max_value=self.lin_vel_impulse_prob_range[1],
            num_samples=self.num_envs,
        )

        self.ang_vel_impulse_prob_range = to_torch(
            self.ang_vel_impulse_prob_range, dtype=torch.float, device=self.device
        )
        self.random_ang_vel_impulse_prob = self._sample_log_uniform(
            min_value=self.ang_vel_impulse_prob_range[0],
            max_value=self.ang_vel_impulse_prob_range[1],
            num_samples=self.num_envs,
        )

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.observed_obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )

        self.obj_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )
        self.goal_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3),
            dtype=torch.float,
            device=self.device,
        )

        self.object_scale_noise_multiplier = torch.ones(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )

        self.lifted_object = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_fingertips], dtype=torch.float, device=self.device
        )
        self.furthest_hand_dist = -torch.ones(
            [self.num_envs], dtype=torch.float, device=self.device
        )

        self.finger_rew_coeffs = torch.ones(
            [self.num_envs, self.num_fingertips], dtype=torch.float, device=self.device
        )

        if self.with_table_force_sensor:
            self.table_sensor_forces_raw = torch.zeros(
                (self.num_envs, 6), dtype=torch.float, device=self.device
            )
            self.table_sensor_forces_smoothed = torch.zeros(
                (self.num_envs, 6), dtype=torch.float, device=self.device
            )
            self.max_table_sensor_force_norm_smoothed = torch.zeros(
                (self.num_envs), dtype=torch.float, device=self.device
            )

        self._init_obs_action_queue()

    def _sample_log_uniform(
        self, min_value: float, max_value: float, num_samples: int
    ) -> torch.Tensor:
        """
        Sample values log-uniformly between min_value and max_value.

        Args:
            min_value: Lower bound (must be > 0).
            max_value: Upper bound (must be > 0).

        Returns:
            A tensor of shape (self.num_envs,) sampled log-uniformly.
        """
        assert min_value > 0, f"min_value must be > 0, got {min_value}"
        assert max_value > 0, f"max_value must be > 0, got {max_value}"
        assert min_value <= max_value, (
            f"min_value must be <= max_value, got {min_value} <= {max_value}"
        )

        return torch.exp(
            torch.log(min_value)
            + (torch.log(max_value) - torch.log(min_value))
            * torch.rand(num_samples, device=self.device)
        )

    ##### KEYBOARD START #####
    def _subscribe_to_keyboard_events(self) -> None:
        from dataclasses import dataclass
        from typing import Callable

        @dataclass
        class KeyboardShortcut:
            name: str
            key: int
            function: Callable

        keyboard_shortcuts = [

            KeyboardShortcut(
                name="reset",
                key=gymapi.KEY_E,
                function=self._reset_callback,
            ),
            KeyboardShortcut(
                name="toggle_do_not_move",
                key=gymapi.KEY_T,
                function=self._toggle_do_not_move_callback,
            ),
            KeyboardShortcut(
                name="toggle_debug_viz",
                key=gymapi.KEY_D,
                function=self._toggle_debug_viz_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_x_plus",
                key=gymapi.KEY_RIGHT,
                function=self._force_or_torque_x_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_x_minus",
                key=gymapi.KEY_LEFT,
                function=self._force_or_torque_x_minus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_y_plus",
                key=gymapi.KEY_UP,
                function=self._force_or_torque_y_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_y_minus",
                key=gymapi.KEY_DOWN,
                function=self._force_or_torque_y_minus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_z_plus",
                key=gymapi.KEY_PAGE_UP,
                function=self._force_or_torque_z_plus_callback,
            ),
            KeyboardShortcut(
                name="force_or_torque_z_minus",
                key=gymapi.KEY_PAGE_DOWN,
                function=self._force_or_torque_z_minus_callback,
            ),
            KeyboardShortcut(
                name="toggle_force_or_torque",
                key=gymapi.KEY_F,
                function=self._toggle_force_or_torque_callback,
            ),
            KeyboardShortcut(
                name="teleport_object",
                key=gymapi.KEY_P,
                function=self._teleport_object_callback,
            ),
        ]
        self.name_to_keyboard_shortcut_dict = {
            keyboard_shortcut.name: keyboard_shortcut
            for keyboard_shortcut in keyboard_shortcuts
        }
        self._DO_NOT_MOVE = False

        # Keyboard applied force or torque or impulse
        # Press arrow keys to apply force or torque or impulse in XY plane, PAGE_UP and PAGE_DOWN for Z direction
        # Press F to toggle between force, torque, lin_vel_impulse, and ang_vel_impulse
        #
        # Control flow is currently
        # - pre_physics_step
        #   - apply_rigid_body_force_tensors
        #   - set_actor_root_state_tensor_indexed
        # - render
        #   - keyboard callbacks
        # - sim step
        #   - root_state_tensor is updated
        # - post_physics_step
        #
        # force and torque is currently stateful
        # you set it once in self.rb_forces and self.rb_torques and it doesn't automatically get reset
        # right now, they just decay over time or get overwritten by random forces/torques
        # doing this from keyboard callbacks is easy because you can set the forces/torques directly
        #
        # velocity impulses are not stateful
        # you are directly setting the root_state_tensor
        # doing this from keyboard callbacks is not easy because keyboard callbacks are called
        # after set_actor_root_state_tensor_indexed and before sim_step
        # directly modifying the root_state_tensor doesn't work because it is overwritten
        # thus, we instead set a flag and store the impulse in a buffer and apply it in pre_physics_step
        # after this, the flag and buffer are reset

        self.force_or_torque_mode = "force"
        self.need_apply_vel_impulse_from_keyboard = False
        self.vel_impulse_from_keyboard = np.array([0.0] * 6)

        self.need_teleport_object_from_keyboard = False

    def _reset_callback(self) -> None:
        print("Resetting...")
        # Easiest way to reset without other issues (reset being overwritten)
        # Is to set the progress buffer to the max episode length - 2 so it will reset shortly
        self.progress_buf[:] = self.max_episode_length - 2

    def _toggle_do_not_move_callback(self) -> None:
        print("Toggling do not move...")
        self._DO_NOT_MOVE = not self._DO_NOT_MOVE
        print(f"Do not move is now {self._DO_NOT_MOVE}")

    def _toggle_debug_viz_callback(self) -> None:
        print("Toggling debug viz...")
        self.debug_viz = not self.debug_viz
        print(f"Debug viz is now {self.debug_viz}")
        if not self.debug_viz:
            self.gym.clear_lines(self.viewer)

    def _force_or_torque_x_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 0] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force x plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 0] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque x plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[0] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse x plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[3] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse x plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_x_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 0] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force x minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 0] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque x minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[0] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse x minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[3] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse x minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_y_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 1] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force y plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 1] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque y plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[1] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse y plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[4] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse y plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_y_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 1] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force y minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 1] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque y minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[1] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse y minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[4] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse y minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_z_plus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 2] += (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force z plus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 2] += (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque z plus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[2] += self.lin_vel_impulse_scale
            print(f"Lin vel impulse z plus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[5] += self.ang_vel_impulse_scale
            print(f"Ang vel impulse z plus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _force_or_torque_z_minus_callback(self) -> None:
        if self.force_or_torque_mode == "force":
            self.rb_forces[:, self.object_rb_handles, 2] -= (
                self.force_scale * self.object_rb_masses
            )
            print(f"Force z minus: {self.rb_forces[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "torque":
            self.rb_torques[:, self.object_rb_handles, 2] -= (
                self.torque_scale * self.object_rb_masses
            )
            print(f"Torque z minus: {self.rb_torques[:, self.object_rb_handles]}")
        elif self.force_or_torque_mode == "lin_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[2] -= self.lin_vel_impulse_scale
            print(f"Lin vel impulse z minus: {self.vel_impulse_from_keyboard}")
        elif self.force_or_torque_mode == "ang_vel_impulse":
            self.need_apply_vel_impulse_from_keyboard = True
            self.vel_impulse_from_keyboard[5] -= self.ang_vel_impulse_scale
            print(f"Ang vel impulse z minus: {self.vel_impulse_from_keyboard}")
        else:
            raise ValueError(
                f"Invalid force or torque mode: {self.force_or_torque_mode}"
            )

    def _toggle_force_or_torque_callback(self) -> None:
        print("Toggling force or torque...")
        MODES = ["force", "torque", "lin_vel_impulse", "ang_vel_impulse"]
        assert self.force_or_torque_mode in MODES, (
            f"Invalid force or torque mode: {self.force_or_torque_mode}"
        )
        idx = MODES.index(self.force_or_torque_mode)
        idx = (idx + 1) % len(MODES)
        self.force_or_torque_mode = MODES[idx]
        print(f"Force or torque mode is now {self.force_or_torque_mode}")

    def _teleport_object_callback(self) -> None:
        print("Teleporting object...")
        self.need_teleport_object_from_keyboard = True
        print(f"Object is now teleported: {self.need_teleport_object_from_keyboard}")

    ##### KEYBOARD END #####

    # SimToolReal abstract interface - to be overriden in derived classes

    def _object_keypoint_offsets(self):
        return [
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]

    def _object_start_pose(self, robot_pose, table_pose_dy, table_pose_dz):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = robot_pose.p.x

        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25

        object_start_pose.p.y = robot_pose.p.y + pose_dy
        object_start_pose.p.z = robot_pose.p.z + pose_dz

        # HACK: Overwrite
        if self.cfg["env"]["objectStartPose"] is not None:
            assert len(self.cfg["env"]["objectStartPose"]) == 7, (
                f"objectStartPose must be a 7-element list, got {len(self.cfg['env']['objectStartPose'])}"
            )
            # Assumes [x, y, z, qx, qy, qz, qw]
            object_start_pose.p.x = self.cfg["env"]["objectStartPose"][0]
            object_start_pose.p.y = self.cfg["env"]["objectStartPose"][1]
            object_start_pose.p.z = self.cfg["env"]["objectStartPose"][2]
            object_start_pose.r.x = self.cfg["env"]["objectStartPose"][3]
            object_start_pose.r.y = self.cfg["env"]["objectStartPose"][4]
            object_start_pose.r.z = self.cfg["env"]["objectStartPose"][5]
            object_start_pose.r.w = self.cfg["env"]["objectStartPose"][6]

        return object_start_pose

    def _main_object_assets_and_scales(self):
        object_name = self.cfg["env"]["objectName"]
        known_object_names = set(NAME_TO_OBJECT.keys())
        if object_name in known_object_names:
            # One of known objects
            obj = NAME_TO_OBJECT[object_name]
            object_asset_files = [obj.urdf_path]
            object_asset_scales = [obj.scale]
            need_vhacds = [obj.need_vhacd]

        else:
            raise ValueError(f"Unknown object name: {object_name}")

        USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
        if USE_FIXED_GOAL_STATES:
            FIXED_GOAL_STATES = self.cfg["env"]["fixedGoalStates"]
            FIXED_GOAL_STATES_JSON_PATH = self.cfg["env"]["fixedGoalStatesJsonPath"]
            fixed_goal_payload = None
            assert (FIXED_GOAL_STATES is None) != (
                FIXED_GOAL_STATES_JSON_PATH is None
            ), (
                "Exactly one of fixedGoalStates or fixedGoalStatesJsonPath must be set when useFixedGoalStates is True"
            )

            if FIXED_GOAL_STATES is not None:
                self.trajectory_states = torch.tensor(
                    FIXED_GOAL_STATES, device=self.device
                )
            if FIXED_GOAL_STATES_JSON_PATH is not None:
                with open(FIXED_GOAL_STATES_JSON_PATH, "r") as f:
                    fixed_goal_payload = json.load(f)
                    self.trajectory_states = torch.tensor(
                        fixed_goal_payload["goals"], device=self.device
                    )

            # Set max consecutive successes to the length of the trajectory so we don't run out of goal states
            self.max_consecutive_successes = len(self.trajectory_states)
            self.toolframe_orientation_success_mask_by_goal = (
                self._toolframe_orientation_success_mask_from_fixed_goals(
                    fixed_goal_payload, len(self.trajectory_states)
                )
            )

        return object_asset_files, object_asset_scales, need_vhacds

    def _toolframe_orientation_success_mask_from_fixed_goals(
        self, fixed_goal_payload, num_goals: int
    ) -> Tensor:
        stage_families = self.toolframe_orientation_success_stage_families
        mask = torch.ones(num_goals, dtype=torch.bool, device=self.device)
        if not stage_families:
            return mask
        mask[:] = False
        if not isinstance(fixed_goal_payload, dict):
            return mask
        goal_index_map = fixed_goal_payload.get("metadata", {}).get("goal_index_map", [])
        for row in goal_index_map:
            if not isinstance(row, dict):
                continue
            goal_index = row.get("fixed_goal_index")
            stage = str(row.get("stage", ""))
            if not isinstance(goal_index, int) or not (0 <= goal_index < num_goals):
                continue
            if any(stage.startswith(str(family)) for family in stage_families):
                mask[goal_index] = True
        return mask

    def _load_main_object_asset(self):
        """Load manipulated object and goal assets."""
        object_assets = []
        for object_asset_file, need_vhacd in zip(
            self.object_asset_files, self.object_need_vhacds
        ):
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_enabled = need_vhacd

            # WARNING: This should not be done if trying to set different densities for different parts of the object, unless handled appropriately in the URDF
            object_asset_options.collapse_fixed_joints = True

            # This should speed up things and make physics better
            # But self-collision may be an issue
            object_asset_options.replace_cylinder_with_capsule = True
            # object_asset_options.replace_cylinder_with_capsule = False

            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            object_asset_ = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            object_assets.append(object_asset_)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        object_shapes_count = self.gym.get_asset_rigid_shape_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        return object_assets, object_rb_count, object_shapes_count

    def _load_additional_assets(self, object_asset_root, arm_pose):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True

        # WARNING: This should not be done if trying to set different densities for different parts of the object, unless handled appropriately in the URDF
        object_asset_options.collapse_fixed_joints = True

        # This should speed up things and make physics better
        # But self-collision may be an issue
        object_asset_options.replace_cylinder_with_capsule = True
        # object_asset_options.replace_cylinder_with_capsule = False

        self.goal_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            goal_asset_ = self.gym.load_asset(
                self.sim, object_asset_dir, object_asset_fname, object_asset_options
            )
            self.goal_assets.append(goal_asset_)
        goal_rb_count = self.gym.get_asset_rigid_body_count(
            self.goal_assets[0]
        )  # assuming all of them have the same rb count
        goal_shapes_count = self.gym.get_asset_rigid_shape_count(
            self.goal_assets[0]
        )  # assuming all of them have the same rb count

        extra_rb_count = 0
        extra_shapes_count = 0
        self.scene_assets = []
        for spec in self.scene_actor_specs:
            if not hasattr(spec, "get"):
                raise ValueError(
                    "sceneActors entries must be mappings"
                )
            asset_file = spec.get("asset_file") or spec.get("asset")
            if not asset_file:
                raise ValueError(
                    "sceneActors entries require asset_file"
                )
            asset_path = Path(str(asset_file))
            if not asset_path.is_absolute():
                asset_path = Path(object_asset_root) / asset_path
            extra_asset_options = gymapi.AssetOptions()
            extra_asset_options.disable_gravity = bool(spec.get("disable_gravity", False))
            extra_asset_options.fix_base_link = bool(spec.get("fix_base_link", False))
            extra_asset_options.collapse_fixed_joints = bool(
                spec.get("collapse_fixed_joints", True)
            )
            extra_asset_options.vhacd_enabled = bool(spec.get("vhacd_enabled", False))
            extra_asset_options.replace_cylinder_with_capsule = False
            extra_asset = self.gym.load_asset(
                self.sim,
                str(asset_path.parent),
                asset_path.name,
                extra_asset_options,
            )
            self.scene_assets.append(extra_asset)
            extra_rb_count += self.gym.get_asset_rigid_body_count(extra_asset)
            extra_shapes_count += self.gym.get_asset_rigid_shape_count(extra_asset)

        return goal_rb_count + extra_rb_count, goal_shapes_count + extra_shapes_count

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = self.object_start_pose.p + self.goal_displacement
        goal_start_pose.p.z -= 0.04
        if self.hide_goal_object_visual:
            goal_start_pose.p = gymapi.Vec3(100.0, 100.0, 100.0)

        goal_asset = self.goal_assets[object_asset_idx]
        goal_handle = self.gym.create_actor(
            env_ptr,
            goal_asset,
            goal_start_pose,
            "goal_object",
            env_idx + self.num_envs,
            0,
            0,
        )
        goal_object_idx = self.gym.get_actor_index(
            env_ptr, goal_handle, gymapi.DOMAIN_SIM
        )
        self.goal_object_indices.append(goal_object_idx)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            self.rigid_body_name_to_idx["goal/" + name] = (
                self.gym.find_actor_rigid_body_index(
                    env_ptr, goal_handle, name, gymapi.DOMAIN_ENV
                )
            )

        if self.hide_goal_object_visual:
            # The goal actor is still needed as a SimToolReal state carrier, but
            # it should not appear in planning/evidence cameras.
            self._set_actor_color(env_ptr, goal_handle, (0.78, 0.66, 0.52))
        else:
            self._set_actor_color(env_ptr, goal_handle, self.goal_object_visual_color)

        for spec_idx, spec in enumerate(self.scene_actor_specs):
            pose_values = spec.get("pose_xyzw") or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            if not hasattr(pose_values, "__len__") or len(pose_values) != 7:
                raise ValueError(
                    "sceneActors pose_xyzw must be a 7-element list"
                )
            pose_values = [float(value) for value in pose_values]
            extra_pose = gymapi.Transform()
            extra_pose.p = gymapi.Vec3(
                float(pose_values[0]),
                float(pose_values[1]),
                float(pose_values[2]),
            )
            extra_pose.r = gymapi.Quat(
                float(pose_values[3]),
                float(pose_values[4]),
                float(pose_values[5]),
                float(pose_values[6]),
            )
            actor_name = str(spec.get("name") or f"parttoolgen_extra_actor_{spec_idx}")
            extra_handle = self.gym.create_actor(
                env_ptr,
                self.scene_assets[spec_idx],
                extra_pose,
                actor_name,
                env_idx,
                int(spec.get("collision_filter", 0)),
                int(spec.get("segmentation_id", 0)),
            )
            extra_idx = self.gym.get_actor_index(
                env_ptr, extra_handle, gymapi.DOMAIN_SIM
            )
            self.scene_actor_indices.append(extra_idx)
            self.scene_actor_init_states.append(
                [
                    float(pose_values[0]),
                    float(pose_values[1]),
                    float(pose_values[2]),
                    float(pose_values[3]),
                    float(pose_values[4]),
                    float(pose_values[5]),
                    float(pose_values[6]),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            for name in self.gym.get_actor_rigid_body_names(env_ptr, extra_handle):
                self.rigid_body_name_to_idx[f"parttoolgen_extra/{actor_name}/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, extra_handle, name, gymapi.DOMAIN_ENV
                    )
                )
            color = spec.get("color_rgb") or spec.get("color_rgba")
            if hasattr(color, "__len__") and len(color) >= 3:
                self._set_actor_color(
                    env_ptr,
                    extra_handle,
                    (float(color[0]), float(color[1]), float(color[2])),
                )

    def _after_envs_created(self):
        self.goal_object_indices = to_torch(
            self.goal_object_indices, dtype=torch.long, device=self.device
        )
        if self.scene_actor_indices:
            num_extra = len(self.scene_actor_specs)
            self.scene_actor_indices = to_torch(
                self.scene_actor_indices,
                dtype=torch.long,
                device=self.device,
            ).view(self.num_envs, num_extra)
            self.scene_actor_init_states = to_torch(
                self.scene_actor_init_states,
                dtype=torch.float,
                device=self.device,
            ).view(self.num_envs, num_extra, 13)

    def _extra_reset_rules(self, resets):
        return resets

    def _sample_delta_goal(
        self, goal_states, delta_goal_distance, delta_rotation_degrees
    ):
        # get the target volume origin and extent
        target_volume_origin = self.target_volume_origin
        target_volume_extent = self.target_volume_extent
        target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
        target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]

        last_goal = goal_states.clone()
        last_goal_pos = last_goal[:, :3]
        last_goal_quat_xyzw = last_goal[:, 3:7]

        new_goal_pos = last_goal_pos + torch_rand_float(
            -delta_goal_distance,
            delta_goal_distance,
            (goal_states.shape[0], 3),
            device=self.device,
        )
        new_goal_pos = torch.clamp(
            new_goal_pos, target_volume_min_coord, target_volume_max_coord
        )

        new_goal_quat_xyzw = self.sample_delta_quat_xyzw(
            input_quat_xyzw=last_goal_quat_xyzw,
            delta_rotation_degrees=delta_rotation_degrees,
        )
        goal_states[:, 0:3] = new_goal_pos
        goal_states[:, 3:7] = new_goal_quat_xyzw
        return goal_states

    def _clip_goal_z(self, env_ids: Tensor):
        min_z = (
            self.object_init_state[env_ids, 2:3] - 0.05 + self.lifting_bonus_threshold
        )
        self.goal_states[env_ids, 2:3] = torch.max(
            min_z, self.goal_states[env_ids, 2:3]
        )

    def _reset_target(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset=True,
        is_first_goal=True,
    ) -> None:
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            USE_FIXED_GOAL_STATES = self.cfg["env"]["useFixedGoalStates"]
            if USE_FIXED_GOAL_STATES:
                trajectory_state = self.trajectory_states.repeat(len(env_ids), 1, 1)
                current_subgoal_idx = (
                    self.successes[env_ids] % len(self.trajectory_states)
                ).long()
                batch_indices = torch.arange(len(env_ids), device=self.device)
                self.goal_states[env_ids, 0:7] = trajectory_state[
                    batch_indices, current_subgoal_idx, 0:7
                ]
            elif not is_first_goal and self.goal_sampling_type == "delta":
                self.goal_states[env_ids, 0:7] = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7],
                    self.delta_goal_distance,
                    self.delta_rotation_degrees,
                )
                # Actually don't clip goal z for delta poses to allow it to go below the lifted z
                # self._clip_goal_z(env_ids)
            elif not is_first_goal and self.goal_sampling_type == "coin_flip":
                # flip a coin. 50% of envs only get delta translation, 0 rotation and 50% get delta rotation, 0 translation
                coin_flips = torch_rand_float(
                    0.0, 1.0, (len(env_ids), 1), device=self.device
                )
                translation_only_goal_states = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7], self.delta_goal_distance, 0.0
                )
                rotation_only_goal_states = self._sample_delta_goal(
                    self.goal_states[env_ids, 0:7], 0.0, self.delta_rotation_degrees
                )
                self.goal_states[env_ids, 0:7] = torch.where(
                    coin_flips < 0.5,
                    translation_only_goal_states,
                    rotation_only_goal_states,
                )
                # Actually don't clip goal z for delta poses to allow it to go below the lifted z
                # self._clip_goal_z(env_ids)
            else:
                # Randomly sample a target pose
                target_volume_origin = self.target_volume_origin
                target_volume_extent = self.target_volume_extent

                target_volume_min_coord = (
                    target_volume_origin + target_volume_extent[:, 0]
                )
                target_volume_max_coord = (
                    target_volume_origin + target_volume_extent[:, 1]
                )
                target_volume_size = target_volume_max_coord - target_volume_min_coord

                rand_pos_floats = torch_rand_float(
                    0.0, 1.0, (len(env_ids), 3), device=self.device
                )
                target_coords = (
                    target_volume_min_coord + rand_pos_floats * target_volume_size
                )

                self.goal_states[env_ids, 0:3] = target_coords
                new_rot = self.get_random_quat(env_ids)
                self.goal_states[env_ids, 3:7] = new_rot
                self._clip_goal_z(env_ids)
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                self.goal_states[env_ids, 0:7]
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
                torch.zeros_like(
                    self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
                )
            )
        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            # TODO: Check if last 6 indices are 0
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[self.goal_object_indices[env_ids], :] = (
                self.root_state_resets[
                    reset_buf_idxs[env_ids].cpu(),
                    self.goal_object_indices[env_ids].cpu() % rs_ofs,
                    :,
                ].to(self.device)
            )
            self.goal_states[env_ids, 0:7] = self.root_state_tensor[
                self.goal_object_indices[env_ids], 0:7
            ]

        self.deferred_set_actor_root_state_tensor_indexed(
            [self.goal_object_indices[env_ids]]
        )

        # HACK: Force the goal object pose to be the specified value
        if self.cfg["env"]["goalObjectPose"] is not None:
            desired_goal_object_pose = self.cfg["env"]["goalObjectPose"]
            assert len(desired_goal_object_pose) == 7, (
                f"desired_goal_object_pose must be a 7-element list, got {len(desired_goal_object_pose)}"
            )
            for i in range(7):
                self.goal_states[env_ids, i] = desired_goal_object_pose[i]
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                self.goal_states[env_ids, 0:7]
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.goal_object_indices]
            )
        if (
            self.hide_goal_object_visual
            and len(env_ids) > 0
            and tensor_reset
        ):
            hidden_goal_states = self.goal_states[env_ids, 0:7].clone()
            hidden_goal_states[:, 0:3] = torch.tensor(
                [100.0, 100.0, 100.0],
                dtype=hidden_goal_states.dtype,
                device=hidden_goal_states.device,
            )
            hidden_goal_states[:, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0],
                dtype=hidden_goal_states.dtype,
                device=hidden_goal_states.device,
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:7] = (
                hidden_goal_states
            )
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = (
                torch.zeros_like(
                    self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
                )
            )
            self.deferred_set_actor_root_state_tensor_indexed(
                [self.goal_object_indices[env_ids]]
            )

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        indices = [self.goal_object_indices[env_ids]]
        if (
            isinstance(self.scene_actor_indices, torch.Tensor)
            and self.scene_actor_indices.numel() > 0
        ):
            indices.append(
                self.scene_actor_indices[env_ids].reshape(-1)
            )
        return indices

    # SimToolReal implementation

    def set_env_state(self, env_state):
        """Restore compatible physical tensors from the supplied policy checkpoint."""
        if not env_state:
            return
        fields = ('dof_state', 'root_state_tensor', 'rigid_body_states', 'successes', 'near_goal_steps', 'lifted_object', 'closest_keypoint_max_dist', 'closest_keypoint_max_dist_fixed_size', 'closest_fingertip_dist', 'furthest_hand_dist', 'prev_targets', 'cur_targets', 'reset_buf', 'progress_buf', 'reset_goal_buf', 'obj_keypoint_pos', 'goal_keypoint_pos', 'obj_keypoint_pos_fixed_size', 'goal_keypoint_pos_fixed_size', 'rb_forces', 'rb_torques', 'random_force_prob', 'random_torque_prob', 'random_lin_vel_impulse_prob', 'random_ang_vel_impulse_prob', 'goal_states', 'goal_init_state', 'object_init_state')
        for key in fields:
            value = env_state.get(key)
            current = getattr(self, key, None)
            if value is None or current is None:
                continue
            if isinstance(current, torch.Tensor):
                if isinstance(value, torch.Tensor) and current.shape == value.shape:
                    current.copy_(value.to(self.device))
            else:
                setattr(self, key, value)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device), tensor_reset=False)
        self.set_actor_root_state_tensor_indexed()

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.control_dt = self.dt * self.control_freq_inv
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

        # If randomizing, apply once immediately on startup before the first sim step
        # Necessary for setup_only=True properties
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        if self.disable_default_ground_plane:
            return
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        pass

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../assets"
        )

        object_asset_root = asset_root
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = (
            self._main_object_assets_and_scales()
        )

        asset_options = gymapi.AssetOptions()
        # asset_options.vhacd_enabled = True  # Should be False so the robot is not complicated to model, but can test
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.robot_asset_file} from {asset_root}")
        robot_asset = self.gym.load_asset(
            self.sim, asset_root, self.robot_asset_file, asset_options
        )
        print(f"Loaded asset {robot_asset}")

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(robot_asset)
        assert self.num_hand_arm_dofs == num_hand_arm_dofs, (
            f"Number of DOFs in asset {robot_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"
        )

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        robot_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(robot_asset, i)
            for i in range(self.num_hand_arm_bodies)
        ]
        print(f"Robot num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Robot rigid bodies: {robot_rigid_body_names}")

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []

        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(robot_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(robot_dof_props["upper"][i])

        self.arm_hand_dof_lower_limits = to_torch(
            self.arm_hand_dof_lower_limits, device=self.device
        )
        self.arm_hand_dof_upper_limits = to_torch(
            self.arm_hand_dof_upper_limits, device=self.device
        )

        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(
            *get_axis_params(0.0, self.up_axis_idx)
        ) + gymapi.Vec3(0.0, 0.8, 0)
        robot_pose.r = gymapi.Quat(0, 0, 0, 1)

        object_assets, object_rb_count, object_shapes_count = (
            self._load_main_object_asset()
        )
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(
            self.sim, asset_root, self.asset_files_dict["table"], table_asset_options
        )

        if self.with_table_force_sensor:
            table_sensor_pose = gymapi.Transform()
            table_sensor_props = gymapi.ForceSensorProperties()
            # If both enable_constraint_solver_forces=False and enable_forward_dynamics_forces=False, always will have a force of 0.0
            table_sensor_props.enable_constraint_solver_forces = True  # Defaults True, keep True to get constraint forces to get contact forces
            table_sensor_props.enable_forward_dynamics_forces = False  # Defaults True, but set to False to avoid gravity being part of this force. Can be True if have disable_gravity=True for the table
            table_sensor_props.use_world_frame = True
            self.table_sensor_idx = self.gym.create_asset_force_sensor(
                asset=table_asset,
                body_idx=0,
                local_pose=table_sensor_pose,
                props=table_sensor_props,
            )
            if self.table_sensor_idx == -1:
                raise ValueError("Failed to create table force sensor")

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = robot_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, self.cfg["env"]["tableResetZ"]
        table_pose.p.y = robot_pose.p.y + table_pose_dy
        table_pose.p.z = robot_pose.p.z + table_pose_dz

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        additional_rb, additional_shapes = self._load_additional_assets(
            object_asset_root, robot_pose
        )
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(
            robot_pose, table_pose_dy, table_pose_dz
        )

        self.robots = []
        self.envs = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robots = []
        self.objects = []

        object_init_state = []
        table_init_state = []

        self.rigid_body_name_to_idx = {}

        self.robot_indices = []
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = []
        object_indices = []
        table_indices = []
        object_scales = []
        object_keypoint_offsets = []
        object_keypoint_offsets_fixed_size = []

        # Sanity checks
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        for name in self.fingertips:
            assert name in body_names, f"Finger {name} not found in asset {robot_asset}"
        has_iiwa14 = "iiwa14_link_7" in body_names
        assert has_iiwa14, f"iiwa14_link_7 not found in asset {robot_asset}"

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.fingertips
        ]

        if self.with_fingertip_force_sensors:
            finger_sensor_pose = gymapi.Transform()
            self.finger_sensor_idxs = [
                self.gym.create_asset_force_sensor(
                    asset=robot_asset, body_idx=ft_handle, local_pose=finger_sensor_pose
                )
                for ft_handle in self.fingertip_handles
            ]

        if has_iiwa14:
            self.robot_name = "iiwa14"
            self.palm_handle = self.gym.find_asset_rigid_body_index(
                robot_asset, "iiwa14_link_7"
            )
        else:
            raise ValueError(f"iiwa14_link_7 not found in asset {robot_asset}")

        # this rely on the fact that objects are added right after the arms in terms of create_actor()
        self.object_rb_handles = list(
            range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count)
        )
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            # Account for the blue robot's additional rigid bodies
            self.object_rb_handles = list(
                range(
                    2 * self.num_hand_arm_bodies,
                    2 * self.num_hand_arm_bodies + object_rb_count,
                )
            )

        # Set asset rigid shape properties (friction)
        MODIFY_ASSET_FRICTIONS = self.cfg["env"]["modifyAssetFrictions"]

        if MODIFY_ASSET_FRICTIONS:
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=self.cfg["env"]["robotFriction"],
                fingertip_friction=self.cfg["env"]["fingerTipFriction"],
            )
            self.set_table_asset_rigid_shape_properties(
                table_asset=table_asset,
                friction=self.cfg["env"]["tableFriction"],
            )
            for object_asset_idx_to_modify in range(len(object_assets)):
                self.set_object_asset_rigid_shape_properties(
                    object_asset=object_assets[object_asset_idx_to_modify],
                    friction=self.cfg["env"]["objectFriction"],
                )
            for extra_asset_idx, extra_asset in enumerate(
                self.scene_assets
            ):
                spec = self.scene_actor_specs[extra_asset_idx]
                self.set_object_asset_rigid_shape_properties(
                    object_asset=extra_asset,
                    friction=float(spec.get("friction", self.cfg["env"]["objectFriction"])),
                )
        else:
            # Still run this because it sets the collision filters to avoid self-collisions between adjacent links
            self.set_robot_asset_rigid_shape_properties(
                robot_asset=robot_asset,
                friction=None,
                fingertip_friction=None,
            )

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            collision_group = i  # Each env has a different collision group so they don't collide with each other
            collision_filter = -1  # -1 = use asset collision filters set in mjcf loader (for URDF, enable all self-collisions)
            # 0 = enable all self-collisions
            # >0 = disable all self-collisions
            segmentation_id = 0
            robot_actor = self.gym.create_actor(
                env_ptr,
                robot_asset,
                robot_pose,
                "robot",
                collision_group,
                collision_filter,
                segmentation_id,
            )
            populate_dof_properties(
                robot_dof_props, self.num_arm_dofs, self.num_hand_dofs
            )

            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)
            robot_idx = self.gym.get_actor_index(
                env_ptr, robot_actor, gymapi.DOMAIN_SIM
            )
            self.robot_indices.append(robot_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, robot_actor):
                self.rigid_body_name_to_idx["robot/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, robot_actor, name, gymapi.DOMAIN_ENV
                    )
                )

            if self.with_dof_force_sensors:
                self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                blue_robot_actor = self.gym.create_actor(
                    env_ptr,
                    robot_asset,
                    robot_pose,
                    "blue_robot",
                    i + self.num_envs * 2,
                    -1,
                    0,
                )
                self.gym.set_actor_dof_properties(
                    env_ptr,
                    blue_robot_actor,
                    robot_dof_props,
                )
                self.blue_robots.append(blue_robot_actor)
                BLUE = (0, 0, 1)
                self._set_actor_color(env_ptr, blue_robot_actor, BLUE)

                blue_robot_idx = self.gym.get_actor_index(
                    env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM
                )
                self.blue_robot_indices.append(blue_robot_idx)

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, self.object_start_pose, "object", i, 0, 0
            )
            object_init_state.append(
                [
                    self.object_start_pose.p.x,
                    self.object_start_pose.p.y,
                    self.object_start_pose.p.z,
                    self.object_start_pose.r.x,
                    self.object_start_pose.r.y,
                    self.object_start_pose.r.z,
                    self.object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, object_handle, name, gymapi.DOMAIN_ENV
                    )
                )

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= (
                        object_scale[coord_idx]
                        * self.object_base_size
                        * self.keypoint_scale
                        / 2
                    )
                object_offsets.append(keypoint)
            object_keypoint_offsets.append(object_offsets)

            # We make a version of keypoint offsets that are a fixed size for all objects
            object_scale_fixed_size = self.cfg["env"]["fixedSize"]
            assert len(object_scale_fixed_size) == 3, (
                f"object_scale_fixed_size must be a 3-element list, got {len(object_scale_fixed_size)}"
            )
            object_offsets_fixed_size = []
            for keypoint in self.keypoints_offsets:
                keypoint_fixed_size = copy(keypoint)
                for coord_idx in range(3):
                    keypoint_fixed_size[coord_idx] *= (
                        object_scale_fixed_size[coord_idx] * self.keypoint_scale / 2
                    )  # Don't multiply by object_base_size here since it's already metric scale
                object_offsets_fixed_size.append(keypoint_fixed_size)
            object_keypoint_offsets_fixed_size.append(object_offsets_fixed_size)

            # table object
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table_object", i, 0, 0
            )
            table_init_state.append(
                [
                    table_pose.p.x,
                    table_pose.p.y,
                    table_pose.p.z,
                    table_pose.r.x,
                    table_pose.r.y,
                    table_pose.r.z,
                    table_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            table_object_idx = self.gym.get_actor_index(
                env_ptr, table_handle, gymapi.DOMAIN_SIM
            )
            table_indices.append(table_object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = (
                    self.gym.find_actor_rigid_body_index(
                        env_ptr, table_handle, name, gymapi.DOMAIN_ENV
                    )
                )

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(
                env_ptr, env_idx=i, object_asset_idx=object_asset_idx
            )

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(object_handle)

        # Default false because this is slow

        # Get mass and inertia of object

        # Set mass and inertia of object

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(
            self.envs[0], object_handle
        )
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(
            object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.table_init_state = to_torch(
            table_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device
        )
        self.object_rb_masses = to_torch(
            self.object_rb_masses, dtype=torch.float, device=self.device
        )

        self.robot_indices = to_torch(
            self.robot_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            object_indices, dtype=torch.long, device=self.device
        )
        self.table_indices = to_torch(
            table_indices, dtype=torch.long, device=self.device
        )
        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = to_torch(
                self.blue_robot_indices, dtype=torch.long, device=self.device
            )

        self.object_scales = to_torch(
            object_scales, dtype=torch.float, device=self.device
        )
        self.object_keypoint_offsets = to_torch(
            object_keypoint_offsets, dtype=torch.float, device=self.device
        )
        self.object_keypoint_offsets_fixed_size = to_torch(
            object_keypoint_offsets_fixed_size, dtype=torch.float, device=self.device
        )

        self.joint_names = self.gym.get_actor_joint_names(env_ptr, robot_actor)
        props = self.gym.get_actor_dof_properties(env_ptr, robot_actor)
        self.joint_lower_limits = props["lower"]
        self.joint_upper_limits = props["upper"]

        print(f"Robot joint names: {self.joint_names}")

        self._after_envs_created()


    def _get_original_object_masses_and_inertias_and_coms(
        self,
    ) -> Tuple[
        List[float], List[Tuple[float, float, float]], List[Tuple[float, float, float]]
    ]:
        original_masses, original_inertias, original_coms = [], [], []
        for env, object in zip(self.envs, self.objects):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            assert len(object_rb_props) == 1, (
                f"Expected 1 rigid body, got {len(object_rb_props)}"
            )
            object_rb_prop = object_rb_props[0]
            original_mass = object_rb_prop.mass
            original_inertia = (
                object_rb_prop.inertia.x.x,
                object_rb_prop.inertia.y.y,
                object_rb_prop.inertia.z.z,
            )
            original_com = (
                object_rb_prop.com.x,
                object_rb_prop.com.y,
                object_rb_prop.com.z,
            )
            original_masses.append(original_mass)
            original_inertias.append(original_inertia)
            original_coms.append(original_com)
        return original_masses, original_inertias, original_coms

    def _set_actor_color(self, env, actor, color: Tuple[float, float, float]) -> None:
        for rigid_body_idx in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_color(
                env,
                actor,
                rigid_body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

    def _goal_object_visual_color_from_cfg(self) -> Tuple[float, float, float]:
        def _color_triplet(value, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return tuple(max(0.0, min(1.0, float(value[i]))) for i in range(3))
                except (TypeError, ValueError):
                    return default
            return default

        base = _color_triplet(self.cfg["env"].get("goalObjectVisualColor"), (0.0, 1.0, 0.0))
        blend = _color_triplet(
            self.cfg["env"].get("goalObjectVisualBlendColor"),
            (1.0, 1.0, 1.0),
        )
        try:
            opacity = float(self.cfg["env"].get("goalObjectVisualOpacity", 1.0))
        except (TypeError, ValueError):
            opacity = 1.0
        opacity = max(0.0, min(1.0, opacity))
        return tuple(opacity * base[i] + (1.0 - opacity) * blend[i] for i in range(3))

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = (
            self.closest_fingertip_dist - self.curr_fingertip_distances
        )
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(
            self.closest_fingertip_dist, self.curr_fingertip_distances
        )

        # again, positive is closer, negative is further away
        # here we use index of the 1st finger, when the distance is large it doesn't matter which one we use
        hand_deltas_furthest = (
            self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        )
        # update the values if finger tips got further away from the object
        self.furthest_hand_dist = torch.maximum(
            self.furthest_hand_dist, self.curr_fingertip_distances[:, 0]
        )

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        fingertip_delta_rew *= ~lifted_object

        # clip between zero and -inf to turn deltas into penalties
        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= ~lifted_object
        # multiply by the number of fingers so two rewards are on the same scale
        hand_delta_penalty *= self.num_fingertips

        return fingertip_delta_rew, hand_delta_penalty

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        max_keypoint_deltas_fixed_size = (
            self.closest_keypoint_max_dist_fixed_size
            - self.keypoints_max_dist_fixed_size
        )

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(
            self.closest_keypoint_max_dist, self.keypoints_max_dist
        )
        self.closest_keypoint_max_dist_fixed_size = torch.minimum(
            self.closest_keypoint_max_dist_fixed_size,
            self.keypoints_max_dist_fixed_size,
        )

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)
        max_keypoint_deltas_fixed_size = torch.clip(
            max_keypoint_deltas_fixed_size, 0, 100
        )

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object
        keypoint_rew_fixed_size = max_keypoint_deltas_fixed_size * lifted_object

        return keypoint_rew, keypoint_rew_fixed_size

    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        kuka_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 0:7]), dim=-1)
            * self.kuka_actions_penalty_scale
        )
        hand_actions_penalty = (
            torch.sum(
                torch.abs(self.arm_hand_dof_vel[..., 7 : self.num_hand_arm_dofs]),
                dim=-1,
            )
            * self.hand_actions_penalty_scale
        )

        return -1 * kuka_actions_penalty, -1 * hand_actions_penalty

    def _compute_resets(self, is_success):
        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        object_z_low = torch.where(self.object_pos[:, 2] < 0.1, ones, zeros)  # fall
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            self.progress_buf = torch.where(
                is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf
            )
            max_consecutive_successes_reached = torch.where(
                self.successes >= self.max_consecutive_successes, ones, zeros
            )
        else:
            max_consecutive_successes_reached = zeros

        max_episode_length_reached = torch.where(
            self.progress_buf >= self.max_episode_length - 1, ones, zeros
        )

        if self.with_table_force_sensor:
            TABLE_FORCE_THRESHOLD = 100.0
            table_force_too_high = torch.where(
                self.max_table_sensor_force_norm_smoothed > TABLE_FORCE_THRESHOLD,
                ones,
                zeros,
            )
        else:
            table_force_too_high = zeros

        # hand far from the object
        hand_far_from_object = torch.where(
            self.curr_fingertip_distances.max(dim=-1).values > 1.5, ones, zeros
        )

        # Reset when dropped
        # Dropped means the object was lifted and then dropped back down to the table
        if self.cfg["env"]["resetWhenDropped"]:
            # As of right now:
            # - table center is at 0.38m and is 0.3m tall, so its surface is at 0.38m + 0.15m = 0.53m
            # - object init state is at 0.63m (table height 0.38m + object relative to table height 0.25m) with +/-0.02m variation
            # - The goal states are sampled in 0.8m + [-0.12m, 0.25m] = [0.68m, 1.05m]
            # - Right now, liftedBonusThreshold is 0.15m, so the object is lifted if it is at least 0.15m - 0.05m above init state, which is 0.73m
            # - We don't want it to flicker between first lifted then suddenly dropped if it slightly goes down, so we need hysteresis here
            # - Thus, we choose a threshold of object init state at 0.63m, so there is 0.1m gap below lifted threshold and 0.05m gap below goal states
            # - And there is 0.1 gap above the table surface, so dropping the object on the table will be detected as dropped
            dropped_z = self.object_init_state[:, 2]
            dropped = (
                torch.where(self.object_pos[:, 2] < dropped_z, ones, zeros)
                * self.lifted_object
            )
        else:
            dropped = zeros

        resets = (
            self.reset_buf
            | object_z_low
            | max_consecutive_successes_reached
            | max_episode_length_reached
            | table_force_too_high
            | hand_far_from_object
            | dropped
        )
        resets = self._extra_reset_rules(resets)

        return resets

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(
            lifted_object
        )
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        # noinspection PyTypeChecker
        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        near_goal_fixed_size: Tensor = (
            self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance
        )
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            near_goal = near_goal_fixed_size

        toolframe_orientation_error = quat_diff_rad(self.object_rot, self.goal_rot)
        orientation_near_goal = (
            toolframe_orientation_error
            <= self.toolframe_orientation_success_tolerance_rad
        )
        if self.use_toolframe_orientation_success:
            if hasattr(self, "toolframe_orientation_success_mask_by_goal"):
                current_subgoal_idx = (
                    self.successes % len(self.toolframe_orientation_success_mask_by_goal)
                ).long()
                orientation_gate_active = (
                    self.toolframe_orientation_success_mask_by_goal[current_subgoal_idx]
                )
            else:
                orientation_gate_active = torch.ones_like(near_goal, dtype=torch.bool)
            near_goal = torch.where(
                orientation_gate_active,
                near_goal & orientation_near_goal,
                near_goal,
            )

        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            # If near_goal is True (1): (steps + 1) * 1 = Increment
            # If near_goal is False (0): (steps + 0) * 0 = Reset to 0
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success

        self.reset_goal_buf[:] = goal_resets

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        toolframe_orientation_rew = torch.exp(
            -self.toolframe_orientation_rew_sharpness
            * torch.square(toolframe_orientation_error)
        ) * lifted_object

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        toolframe_orientation_rew *= self.toolframe_orientation_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_rew = is_success * self.reach_goal_bonus

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty  # + sign here because hand_delta_penalty is negative
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + toolframe_orientation_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        # Keep the current fixed-goal episode active
        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        return self.rew_buf, is_success

    def populate_sim_buffers(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.with_fingertip_force_sensors or self.with_table_force_sensor:
            self.gym.refresh_force_sensor_tensor(self.sim)
        if self.with_dof_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.with_table_force_sensor:
            self.table_sensor_forces_raw = self.force_sensor_tensor[
                :, self.table_sensor_idx, :
            ]

            # Smooth the force because the signal can be spikey, and we don't want to make decisions based on spikey signals
            TABLE_SENSOR_FORCE_SMOOTHING_ALPHA = 0.1  # 1 = no smoothing, 0 = no updates
            self.table_sensor_forces_smoothed = self.interpolate(
                init=self.table_sensor_forces_smoothed,
                final=self.table_sensor_forces_raw,
                alpha=TABLE_SENSOR_FORCE_SMOOTHING_ALPHA,
            )
            table_sensor_force_norm_smoothed = self.table_sensor_forces_smoothed[
                :, :3
            ].norm(dim=-1)
            self.max_table_sensor_force_norm_smoothed = torch.where(
                table_sensor_force_norm_smoothed
                > self.max_table_sensor_force_norm_smoothed,
                table_sensor_force_norm_smoothed,
                self.max_table_sensor_force_norm_smoothed,
            )

        if self.with_fingertip_force_sensors:
            raise NotImplementedError(
                "Fingertip force sensors are not implemented yet, be careful about indexing"
            )
            self.finger_sensor_forces = self.force_sensor_tensor[
                :, self.finger_sensor_idxs, :
            ]

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]

        # Ultra hack: Move the perceived object position up by 0.1 meters
        # from copy import deepcopy
        # D = [0.00, 0.02, -0.02]
        # D = [0.0, 0.0, 0.015]
        # D = [0.0, 0.0, -0.015]
        # D = [0.0, 0.01, 0.0]
        # DX = D[0]
        # DY = D[1]
        # DZ = D[2]
        # self.object_state = deepcopy(self.object_state)
        # self.object_state[:, 0] += DX
        # self.object_state[:, 1] += DY
        # self.object_state[:, 2] += DZ
        # self.object_pose = deepcopy(self.object_pose)
        # self.object_pose[:, 0] += DX
        # self.object_pose[:, 1] += DY
        # self.object_pose[:, 2] += DZ
        # self.object_pos = deepcopy(self.object_pos)
        # self.object_pos[:, 0] += DX
        # self.object_pos[:, 1] += DY
        # self.object_pos[:, 2] += DZ

        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        # Update object state queue
        self.object_state_queue = self.update_queue(
            queue=self.object_state_queue, current_values=self.object_state
        )

        # Observed object state
        self.observed_object_state = self.object_state.clone()
        use_object_state_delay_noise = self.cfg["env"]["useObjectStateDelayNoise"]
        if use_object_state_delay_noise:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0,
                self.object_state_queue.shape[1],
                (self.num_envs,),
                device=self.device,
            )
            self.observed_object_state[:] = self.object_state_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

            # Add noise to the observed object state
            xyz_noise_std = self.cfg["env"]["objectStateXyzNoiseStd"]
            rotation_noise_degrees = self.cfg["env"]["objectStateRotationNoiseDegrees"]
            self.observed_object_state[:, 0:3] += (
                torch.randn_like(self.observed_object_state[:, 0:3]) * xyz_noise_std
            )
            self.observed_object_state[:, 3:7] = self.sample_delta_quat_xyzw(
                input_quat_xyzw=self.observed_object_state[:, 3:7],
                delta_rotation_degrees=rotation_noise_degrees,
            )

        self.observed_object_pose = self.observed_object_state[:, 0:7]
        self.observed_object_pos = self.observed_object_state[:, 0:3]
        self.observed_object_rot = self.observed_object_state[:, 3:7]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # HACK: Move offsets down by X along x axis to grab hammer on bottom of handle
        use_hack_object_pos_offset = self.cfg["env"]["use_hack_object_pos_offset"]
        hack_object_pos_offset = self.cfg["env"]["hack_object_pos_offset"]
        if use_hack_object_pos_offset:
            self.object_pos_offset = torch.tensor(
                [-hack_object_pos_offset, 0.0, 0.0], device=self.device
            )[None].repeat_interleave(self.num_envs, dim=0)
            self.object_pos = self.object_pos + quat_rotate(
                self.object_rot, self.object_pos_offset
            )
            self.goal_pos = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_pos_offset
            )

        self.palm_center_offset = (
            torch.from_numpy(self.palm_offset)
            .to(self.device)
            .repeat((self.num_envs, 1))
        )
        self._palm_state = self.rigid_body_states[:, self.palm_handle][:, 0:13]
        self._palm_pos = self.rigid_body_states[:, self.palm_handle][:, 0:3]
        self._palm_rot = self.rigid_body_states[:, self.palm_handle][:, 3:7]
        self.palm_center_pos = self._palm_pos + quat_rotate(
            self._palm_rot, self.palm_center_offset
        )

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:13
        ]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 0:3
        ]
        self.fingertip_rot = self.rigid_body_states[:, self.fingertip_handles][
            :, :, 3:7
        ]

        if not isinstance(self.fingertip_offsets, torch.Tensor):
            self.fingertip_offsets = (
                torch.from_numpy(self.fingertip_offsets)
                .to(self.device)
                .repeat((self.num_envs, 1, 1))
            )

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for i in range(self.num_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets[:, i]
            )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(
            self.fingertip_pos_rel_object, dim=-1
        )

        # when episode ends or target changes we reset this to -1, this will initialize it to the actual distance on the 1st frame of the episode
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0,
            self.curr_fingertip_distances,
            self.closest_fingertip_dist,
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0,
            self.curr_fingertip_distances[:, 0],
            self.furthest_hand_dist,
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(
            1, self.num_fingertips, 1
        )
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.observed_obj_keypoint_pos[:, i] = (
                self.observed_object_pos
                + quat_rotate(
                    self.observed_object_rot,
                    self.object_keypoint_offsets[:, i]
                    * self.object_scale_noise_multiplier,
                )
            )

            self.obj_keypoint_pos_fixed_size[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )
            self.goal_keypoint_pos_fixed_size[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.observed_keypoints_rel_goal = (
            self.observed_obj_keypoint_pos - self.goal_keypoint_pos
        )
        self.keypoints_rel_goal_fixed_size = (
            self.obj_keypoint_pos_fixed_size - self.goal_keypoint_pos_fixed_size
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(
            1, self.num_keypoints, 1
        )
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat
        self.observed_keypoints_rel_palm = (
            self.observed_obj_keypoint_pos - palm_center_repeat
        )

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoint_distances_l2_fixed_size = torch.norm(
            self.keypoints_rel_goal_fixed_size, dim=-1
        )

        # furthest keypoint from the goal
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values
        self.keypoints_max_dist_fixed_size = self.keypoint_distances_l2_fixed_size.max(
            dim=-1
        ).values

        # this is the closest the keypoint had been to the target in the current episode (for the furthest keypoint of all)
        # make sure we initialize this value before using it for obs or rewards
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0,
            self.keypoints_max_dist,
            self.closest_keypoint_max_dist,
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )

    def populate_obs_and_states_buffers(self) -> None:
        num_dofs = self.num_hand_arm_dofs
        obs_dict = {}

        # We first fill in the obs_dict with the values that should be given to the critic, which are clean
        # Then after creating state_buf, we add the noisy delayed object state observations and other observation changes to the policy's obs_buf

        ## POLICY OBSERVATIONS ##
        # dof positions
        obs_dict["joint_pos"] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        # dof velocities
        obs_dict["joint_vel"] = self.arm_hand_dof_vel[:, :num_dofs]
        # prev action targets
        obs_dict["prev_action_targets"] = self.prev_targets.clone()
        # palm pos
        obs_dict["palm_pos"] = self.palm_center_pos
        # palm rot
        obs_dict["palm_rot"] = self._palm_state[:, 3:7]
        # object rot
        obs_dict["object_rot"] = self.object_state[:, 3:7]
        # keypoint distances relative to the palm of the hand
        keypoint_rel_pos_size = 3 * self.num_keypoints
        obs_dict["keypoints_rel_palm"] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        # keypoint distances relative to the goal
        obs_dict["keypoints_rel_goal"] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        # fingertip pos relative to the palm of the hand
        fingertip_rel_pos_size = 3 * self.num_fingertips
        obs_dict["fingertip_pos_rel_palm"] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        # object scales
        obs_dict["object_scales"] = (
            self.object_scales * self.object_scale_noise_multiplier
        )

        ## CRITIC OBSERVATIONS ##
        # palm linvel, ang vel
        obs_dict["palm_vel"] = self._palm_state[:, 7:13]
        # object vel
        obs_dict["object_vel"] = self.object_state[:, 7:13]
        # closest distance to the furthest keypoint, achieved so far in this episode
        obs_dict["closest_keypoint_max_dist"] = (
            self.closest_keypoint_max_dist.unsqueeze(-1)
        )
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            obs_dict["closest_keypoint_max_dist"] = (
                self.closest_keypoint_max_dist_fixed_size.unsqueeze(-1)
            )

        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        obs_dict["closest_fingertip_dist"] = self.closest_fingertip_dist.unsqueeze(-1)
        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        obs_dict["lifted_object"] = self.lifted_object.unsqueeze(-1)
        # this should help the critic predict the future rewards better and anticipate the episode termination
        obs_dict["progress"] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
        obs_dict["successes"] = torch.log(self.successes + 1).unsqueeze(-1)
        # this is where we will add the reward observation
        reward_obs_scale = 0.01
        obs_dict["reward"] = reward_obs_scale * self.rew_buf

        # ##############################################################################################################
        # Create state_buf
        # ##############################################################################################################
        self.states_buf = torch.cat(
            [obs_dict[k].reshape(self.num_envs, -1) for k in self.state_list], dim=-1
        )

        # Policy observations
        # Add noisy delayed object state observations
        use_object_state_delay_noise = self.cfg["env"]["useObjectStateDelayNoise"]
        if use_object_state_delay_noise:
            # Add noise
            obs_dict["object_rot"] = self.observed_object_state[:, 3:7]
            obs_dict["object_vel"] = (
                self.observed_object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
            )
            keypoint_rel_pos_size = 3 * self.num_keypoints
            obs_dict["keypoints_rel_palm"] = self.observed_keypoints_rel_palm.reshape(
                self.num_envs, keypoint_rel_pos_size
            )
            obs_dict["keypoints_rel_goal"] = self.observed_keypoints_rel_goal.reshape(
                self.num_envs, keypoint_rel_pos_size
            )

        # Add noise to joint velocities
        obs_dict["joint_vel"] += (
            torch.randn_like(obs_dict["joint_vel"])
            * self.cfg["env"]["jointVelocityObsNoiseStd"]
        )
        # palm linvel, ang vel
        obs_dict["palm_vel"] = (
            self._palm_state[:, 7:13] * self.turn_off_palm_vel_obs_scale
        )
        # object vel
        obs_dict["object_vel"] = (
            self.object_state[:, 7:13] * self.turn_off_object_vel_obs_scale
        )
        # closest distance to the furthest keypoint, achieved so far in this episode
        obs_dict["closest_keypoint_max_dist"] = (
            self.closest_keypoint_max_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        obs_dict["closest_fingertip_dist"] = (
            self.closest_fingertip_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        obs_dict["lifted_object"] = (
            self.lifted_object.unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # this should help the critic predict the future rewards better and anticipate the episode termination
        obs_dict["progress"] = (
            torch.log(self.progress_buf / 10 + 1).unsqueeze(-1)
            * self.turn_off_extra_obs_scale
        )
        obs_dict["successes"] = (
            torch.log(self.successes + 1).unsqueeze(-1) * self.turn_off_extra_obs_scale
        )
        # this is where we will add the reward observation
        reward_obs_scale = 0.01
        obs_dict["reward"] = (
            reward_obs_scale * self.rew_buf * self.turn_off_extra_obs_scale
        )

        # ##############################################################################################################
        # Create obs_buf
        # ##############################################################################################################
        self.obs_buf = torch.cat(
            [obs_dict[k].reshape(self.num_envs, -1) for k in self.obs_list], dim=-1
        )

        # Update obs queue
        self.obs_queue = self.update_queue(
            queue=self.obs_queue, current_values=self.obs_buf
        )

        # Modify obs to be delayed
        use_obs_delay = self.cfg["env"]["useObsDelay"]
        if use_obs_delay:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0, self.obs_queue.shape[1], (self.num_envs,), device=self.device
            )
            self.obs_buf[:] = self.obs_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

    @property
    def turn_off_palm_vel_obs_scale(self) -> float:
        return 0.0 if self.cfg["env"].get("turn_off_palm_vel_obs", False) else 1.0

    @property
    def turn_off_object_vel_obs_scale(self) -> float:
        return 0.0 if self.cfg["env"].get("turn_off_object_vel_obs", False) else 1.0

    @property
    def turn_off_extra_obs_scale(self) -> float:
        return 0.0 if self.cfg["env"].get("turn_off_extra_obs", False) else 1.0

    def clamp_obs(self) -> None:
        if self.clamp_abs_observations > 0:
            self.obs_buf.clamp_(
                -self.clamp_abs_observations, self.clamp_abs_observations
            )
            self.states_buf.clamp_(
                -self.clamp_abs_observations, self.clamp_abs_observations
            )

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat(
            (
                q_x.unsqueeze(-1),
                q_y.unsqueeze(-1),
                q_z.unsqueeze(-1),
                q_w.unsqueeze(-1),
            ),
            dim=-1,
        )

        return new_rot

    def reset_target_pose(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        tensor_reset=True,
        is_first_goal=True,
    ) -> None:
        self._reset_target(
            env_ids,
            reset_buf_idxs,
            tensor_reset=tensor_reset,
            is_first_goal=is_first_goal,
        )

        if tensor_reset:
            self.reset_goal_buf[env_ids] = 0
            self.near_goal_steps[env_ids] = 0

            self.closest_keypoint_max_dist[env_ids] = -1
            self.closest_keypoint_max_dist_fixed_size[env_ids] = -1

    def reset_object_pose(
        self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True
    ):
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            table_indices = self.table_indices[env_ids]

            # decide table reset z
            table_reset_z = (
                torch_rand_float(
                    -self.cfg["env"]["tableResetZRange"],
                    self.cfg["env"]["tableResetZRange"],
                    (len(env_ids), 1),
                    device=self.device,
                )
                + self.cfg["env"]["tableResetZ"]
            )
            self.table_init_state[env_ids, 2:3] = table_reset_z
            USE_FIXED_INIT_OBJECT_POSE = self.cfg["env"]["useFixedInitObjectPose"]
            HAS_EXPLICIT_OBJECT_START_POSE = self.cfg["env"]["objectStartPose"] is not None
            if not (USE_FIXED_INIT_OBJECT_POSE and HAS_EXPLICIT_OBJECT_START_POSE):
                self.object_init_state[env_ids, 2:3] = (
                    table_reset_z + self.cfg["env"]["tableObjectZOffset"]
                )

            # reset object
            rand_pos_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), 3), device=self.device
            )
            if USE_FIXED_INIT_OBJECT_POSE:
                rand_pos_floats[:] = 0.0
            self.root_state_tensor[obj_indices] = self.object_init_state[
                env_ids
            ].clone()
            self.root_state_tensor[table_indices] = self.table_init_state[
                env_ids
            ].clone()

            # indices 0..2 correspond to the object position
            self.root_state_tensor[obj_indices, 0:1] = (
                self.object_init_state[env_ids, 0:1]
                + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
            )
            self.root_state_tensor[obj_indices, 1:2] = (
                self.object_init_state[env_ids, 1:2]
                + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
            )
            self.root_state_tensor[obj_indices, 2:3] = (
                self.object_init_state[env_ids, 2:3]
                + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
            )

            if self.randomize_object_rotation:
                new_object_rot = self.get_random_quat(env_ids)
                if USE_FIXED_INIT_OBJECT_POSE:
                    new_object_rot[:] = 0.0
                    new_object_rot[:, -1] = 1.0  # xyzw

                    # HACK: Rotate the object by 180 degrees around the z-axis to go from right handed to left handed robot
                    from scipy.spatial.transform import Rotation as R

                    new_object_rot[:] = (
                        torch.from_numpy(R.from_euler("z", 180, degrees=True).as_quat())
                        .float()
                        .to(self.device)[None]
                    )

                # indices 3,4,5,6 correspond to the rotation quaternion
                self.root_state_tensor[obj_indices, 3:7] = new_object_rot

            self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(
                self.root_state_tensor[obj_indices, 7:13]
            )
            if (
                isinstance(self.scene_actor_indices, torch.Tensor)
                and self.scene_actor_indices.numel() > 0
            ):
                extra_indices = self.scene_actor_indices[
                    env_ids
                ].reshape(-1)
                extra_states = self.scene_actor_init_states[
                    env_ids
                ].reshape(-1, 13)
                self.root_state_tensor[extra_indices, :] = extra_states.clone()
                self.deferred_set_actor_root_state_tensor_indexed([extra_indices])

            noise_min, noise_max = self.cfg["env"]["objectScaleNoiseMultiplierRange"]
            self.object_scale_noise_multiplier[env_ids] = torch_rand_float(
                noise_min, noise_max, (len(env_ids), 3), device=self.device
            )

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            # TODO: Check if last 6 indices are 0
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[obj_indices, :] = self.root_state_resets[
                reset_buf_idxs[env_ids].cpu(), obj_indices.cpu() % rs_ofs, :
            ].to(self.device)

        # since we reset the object, we also should update distances between fingers and the object
        if tensor_reset:
            self.closest_fingertip_dist[env_ids] = -1
            self.furthest_hand_dist[env_ids] = -1
            self.lifted_object[env_ids] = False
        self.deferred_set_actor_root_state_tensor_indexed(
            [self.object_indices[env_ids]]
        )
        self.deferred_set_actor_root_state_tensor_indexed([self.table_indices[env_ids]])

    def deferred_set_actor_root_state_tensor_indexed(
        self, obj_indices: List[Tensor]
    ) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def deferred_set_dof_state_tensor_indexed(self, dof_indices: List[Tensor]) -> None:
        self.set_dof_state_object_indices.extend(dof_indices)

    def set_dof_state_tensor_indexed(self) -> None:
        dof_indices: List[Tensor] = self.set_dof_state_object_indices
        if not dof_indices:
            # nothing to set
            return

        unique_dof_indices = torch.unique(torch.cat(dof_indices).to(torch.int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(unique_dof_indices),
            len(unique_dof_indices),
        )

        self.set_dof_state_object_indices = []

    def reset_idx(
        self,
        env_ids: Tensor,
        reset_buf_idxs=None,
        episode_reset=True,
        tensor_reset=True,
    ) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if len(env_ids) == 0:
            return

        if self.randomize and episode_reset:
            self.apply_randomizations(self.randomization_params)

        # Do this before reset_target_pose so that the successes is 0 for the new episode
        if episode_reset and tensor_reset:
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

            self.successes[env_ids] = 0

            if self.with_table_force_sensor:
                self.table_sensor_forces_raw[env_ids, :] = 0
                self.table_sensor_forces_smoothed[env_ids, :] = 0
                self.max_table_sensor_force_norm_smoothed[env_ids] = 0

        # reset rigid body forces
        if tensor_reset:
            self.rb_forces[env_ids, :, :] = 0.0
            self.rb_torques[env_ids, :, :] = 0.0

        # reset object
        self.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        # randomize start object poses
        self.reset_target_pose(
            env_ids, reset_buf_idxs, tensor_reset=tensor_reset, is_first_goal=True
        )

        robot_indices = self.robot_indices[env_ids].to(torch.int32)

        # reset random force probabilities
        if tensor_reset:
            self.random_force_prob[env_ids] = self._sample_log_uniform(
                min_value=self.force_prob_range[0],
                max_value=self.force_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_torque_prob[env_ids] = self._sample_log_uniform(
                min_value=self.torque_prob_range[0],
                max_value=self.torque_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_lin_vel_impulse_prob[env_ids] = self._sample_log_uniform(
                min_value=self.lin_vel_impulse_prob_range[0],
                max_value=self.lin_vel_impulse_prob_range[1],
                num_samples=len(env_ids),
            )
            self.random_ang_vel_impulse_prob[env_ids] = self._sample_log_uniform(
                min_value=self.ang_vel_impulse_prob_range[0],
                max_value=self.ang_vel_impulse_prob_range[1],
                num_samples=len(env_ids),
            )

        # reset robot
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
            delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

            rand_dof_floats = torch_rand_float(
                0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device
            )

            rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

            noise_coeff = torch.zeros_like(
                self.hand_arm_default_dof_pos, device=self.device
            )

            noise_coeff[0:7] = self.reset_dof_pos_noise_arm
            noise_coeff[7 : self.num_hand_arm_dofs] = self.reset_dof_pos_noise_fingers

            robot_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta
            robot_pos = tensor_clamp(
                robot_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            )

            self.arm_hand_dof_pos[env_ids, :] = robot_pos
            if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                self.blue_robot_arm_hand_dof_pos[env_ids, :] = robot_pos.clone()
                self.blue_robot_arm_hand_dof_vel[env_ids, :] = 0.0

            rand_vel_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device
            )
            self.arm_hand_dof_vel[env_ids, :] = (
                self.reset_dof_vel_noise * rand_vel_floats
            )
            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos

        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            self.arm_hand_dof_pos[env_ids, :] = self.dof_resets[
                reset_buf_idxs[env_ids].cpu(), :, 0
            ].to(self.device)
            self.arm_hand_dof_vel[env_ids, :] = self.dof_resets[
                reset_buf_idxs[env_ids].cpu(), :, 1
            ].to(self.device)
            robot_pos = self.arm_hand_dof_pos[env_ids, : self.num_hand_arm_dofs]
            robot_pos = tensor_clamp(
                robot_pos,
                self.arm_hand_dof_lower_limits,
                self.arm_hand_dof_upper_limits,
            )

            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = robot_pos

        pass

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(env_ids),
        )

        self.deferred_set_dof_state_tensor_indexed([robot_indices])
        self.deferred_set_actor_root_state_tensor_indexed(
            self._extra_object_indices(env_ids)
        )

    def update_queue(
        self, queue: torch.Tensor, current_values: torch.Tensor
    ) -> torch.Tensor:
        N, T, D = queue.shape
        assert current_values.shape == (N, D), (
            f"current_values.shape: {current_values.shape}, expected: ({N}, {D})"
        )

        # Update queue
        # queue is reset on episode start
        # This means we need to fill the entire queue with the current values
        # queue shape is (N, T, D), where N is num envs, T is queue length, D is dimension of the buffer
        is_episode_start = self.progress_buf == 1
        assert is_episode_start.shape == (N,), (
            f"is_episode_start.shape: {is_episode_start.shape}, expected: ({N})"
        )

        queue[:] = torch.where(
            is_episode_start.unsqueeze(1).unsqueeze(1),
            current_values.unsqueeze(1).repeat_interleave(T, dim=1),
            queue,
        )

        # Roll the queue down by 1, then update index=0 with the new action
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values.clone()
        return queue

    def pre_physics_step(
        self, actions, joint_pos_targets: Optional[torch.Tensor] = None
    ):

        actions = actions.to(self.device)

        # Update actions queue
        self.action_queue = self.update_queue(
            queue=self.action_queue, current_values=actions
        )

        # Modify actions to be delayed
        use_action_delay = self.cfg["env"]["useActionDelay"]
        if use_action_delay:
            # Sample a delay index from the queue
            delay_index = torch.randint(
                0, self.action_queue.shape[1], (self.num_envs,), device=self.device
            )
            actions = self.action_queue[
                torch.arange(self.num_envs), delay_index
            ].clone()

        self.actions = actions.clone()

        if self.privileged_actions:
            torque_actions = actions[:, :3]
            actions = actions[:, 3:]

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        combined_random_env_ids = torch.cat(
            [reset_env_ids, reset_goal_env_ids, reset_goal_env_ids]
        )
        uniques, counts = combined_random_env_ids.unique(return_counts=True)
        reset_goal_env_ids = uniques[counts == 2]
        self.reset_target_pose(reset_goal_env_ids, None, is_first_goal=False)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids, None)

        if self.use_relative_control:
            # arm relative to current position
            targets = (
                self.arm_hand_dof_pos[:, :7]
                + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            )
            self.cur_targets[:, :7] = tensor_clamp(
                targets,
                self.arm_hand_dof_lower_limits[:7],
                self.arm_hand_dof_upper_limits[:7],
            )
        else:
            # arm relative to previous target
            targets = (
                self.prev_targets[:, :7]
                + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            )
            self.cur_targets[:, :7] = tensor_clamp(
                targets,
                self.arm_hand_dof_lower_limits[:7],
                self.arm_hand_dof_upper_limits[:7],
            )

        # Smooth arm
        self.cur_targets[:, :7] = (
            self.arm_moving_average * self.cur_targets[:, :7]
            + (1.0 - self.arm_moving_average) * self.prev_targets[:, :7]
        )

        # hand
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
            actions[:, 7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
        )
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
            self.hand_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
            + (1.0 - self.hand_moving_average)
            * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
        )
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
            self.cur_targets[:, 7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
        )

        # Default CHECK_WITH_COMPUTED_JOINT_POS_TARGETS = False
        # Set to True to check if the computed joint pos targets are correct

        if joint_pos_targets is not None:

            self.cur_targets[:, : self.num_hand_arm_dofs] = joint_pos_targets.clone()

        # print(f"self.cur_targets: {self.cur_targets[0, 7:]}")
        # print(f"self.arm_dof_pos: {self.arm_hand_dof_pos[0, 7:]}")
        # print()

        if self._DO_NOT_MOVE:
            self.cur_targets[:, :] = self.prev_targets[:, :]

        self.prev_targets[:, :] = self.cur_targets[:, :]

        if self.VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.cur_targets[:, self.num_hand_arm_dofs :] = self.cur_targets[
                :, : self.num_hand_arm_dofs
            ].clone()
            self.blue_robot_arm_hand_dof_pos[:] = self.cur_targets[
                :, self.num_hand_arm_dofs :
            ].clone()
            self.blue_robot_arm_hand_dof_vel[:] = 0.0

            blue_robot_indices = self.blue_robot_indices.to(torch.int32)
            self.deferred_set_dof_state_tensor_indexed([blue_robot_indices])

        self.set_dof_state_tensor_indexed()
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

        # Random forces
        if self.force_scale > 0.0 or self.torque_scale > 0.0:
            if self.force_scale > 0.0:
                self.rb_forces *= torch.pow(
                    self.force_decay, self.dt / self.force_decay_interval
                )

                # apply new forces
                force_indices = (
                    torch.rand(self.num_envs, device=self.device)
                    < self.random_force_prob
                ).nonzero()
                self.rb_forces[force_indices, self.object_rb_handles, :] = (
                    torch.randn(
                        self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                        device=self.device,
                    )
                    * self.object_rb_masses
                    * self.force_scale
                )

                if self.force_only_when_lifted:
                    # self.rb_forces is (N, R, 3), assuming there are R rigid bodies per env
                    # self.lifted_object is (N,), True if the object is lifted
                    self.rb_forces[:, self.object_rb_handles, :] *= (
                        self.lifted_object.unsqueeze(1).unsqueeze(2)
                    )

            if self.torque_scale > 0.0:
                self.rb_torques *= torch.pow(
                    self.torque_decay, self.dt / self.torque_decay_interval
                )

                # apply new torques
                torque_indices = (
                    torch.rand(self.num_envs, device=self.device)
                    < self.random_torque_prob
                ).nonzero()
                self.rb_torques[torque_indices, self.object_rb_handles, :] = (
                    torch.randn(
                        self.rb_torques[
                            torque_indices, self.object_rb_handles, :
                        ].shape,
                        device=self.device,
                    )
                    * self.object_rb_masses  # in theory should do inertia, but harder to do this, so just use mass for now
                    * self.torque_scale
                )

                if self.torque_only_when_lifted:
                    # self.rb_torques is (N, R, 3), assuming there are R rigid bodies per env
                    # self.lifted_object is (N,), True if the object is lifted
                    self.rb_torques[:, self.object_rb_handles, :] *= (
                        self.lifted_object.unsqueeze(1).unsqueeze(2)
                    )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                gymtorch.unwrap_tensor(self.rb_torques),
                gymapi.ENV_SPACE,
            )

        # Random velocity impulses
        if self.lin_vel_impulse_scale > 0.0 or self.ang_vel_impulse_scale > 0.0:
            if self.lin_vel_impulse_scale > 0.0:
                if self.lin_vel_impulse_only_when_lifted:
                    lin_vel_impulse_env_ids = (
                        (
                            (
                                torch.rand(self.num_envs, device=self.device)
                                < self.random_lin_vel_impulse_prob
                            )
                            * self.lifted_object
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                else:
                    lin_vel_impulse_env_ids = (
                        (
                            torch.rand(self.num_envs, device=self.device)
                            < self.random_lin_vel_impulse_prob
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                random_lin_vel_impulses = (
                    torch.randn(self.num_envs, 3, device=self.device)
                    * self.lin_vel_impulse_scale
                )
                self.root_state_tensor[
                    self.object_indices[lin_vel_impulse_env_ids], 7:10
                ] = random_lin_vel_impulses[lin_vel_impulse_env_ids, :]
                self.deferred_set_actor_root_state_tensor_indexed(
                    [self.object_indices[lin_vel_impulse_env_ids]]
                )

            if self.ang_vel_impulse_scale > 0.0:
                if self.ang_vel_impulse_only_when_lifted:
                    ang_vel_impulse_env_ids = (
                        (
                            (
                                torch.rand(self.num_envs, device=self.device)
                                < self.random_ang_vel_impulse_prob
                            )
                            * self.lifted_object
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                else:
                    ang_vel_impulse_env_ids = (
                        (
                            torch.rand(self.num_envs, device=self.device)
                            < self.random_ang_vel_impulse_prob
                        )
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                random_ang_vel_impulses = (
                    torch.randn(self.num_envs, 3, device=self.device)
                    * self.ang_vel_impulse_scale
                )
                self.root_state_tensor[
                    self.object_indices[ang_vel_impulse_env_ids], 10:13
                ] = random_ang_vel_impulses[ang_vel_impulse_env_ids, :]
                self.deferred_set_actor_root_state_tensor_indexed(
                    [self.object_indices[ang_vel_impulse_env_ids]]
                )

        # Keyboard applied velocity impulses
        # See keyboard callbacks for more details
        if self.need_apply_vel_impulse_from_keyboard:
            self.root_state_tensor[self.object_indices, 7:13] += (
                torch.from_numpy(self.vel_impulse_from_keyboard)
                .float()
                .to(self.device)[None, :]
            )
            self.deferred_set_actor_root_state_tensor_indexed([self.object_indices])
            self.need_apply_vel_impulse_from_keyboard = False
            self.vel_impulse_from_keyboard[:] = 0.0

        # Teleport object
        if self.need_teleport_object_from_keyboard:
            self.root_state_tensor[self.object_indices, :] = (
                self.object_init_state.clone()
            )
            self.deferred_set_actor_root_state_tensor_indexed([self.object_indices])
            self.need_teleport_object_from_keyboard = False

        self.set_actor_root_state_tensor_indexed()

        # apply torques
        if self.privileged_actions:
            torque_actions = torque_actions.unsqueeze(1)
            torque_amount = self.privileged_actions_torque
            torque_actions *= torque_amount
            self.action_torques[:, self.object_rb_handles, :] = torque_actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                None,
                gymtorch.unwrap_tensor(self.action_torques),
                gymapi.ENV_SPACE,
            )

        pass

    @property
    def use_sharpa(self) -> bool:
        return "sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def use_right_sharpa(self) -> bool:
        return "right_sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def use_left_sharpa(self) -> bool:
        return "left_sharpa" in self.cfg["env"]["asset"]["robot"].lower()

    @property
    def hand_moving_average(self) -> float:
        return self.cfg["env"]["handMovingAverage"]

    @property
    def arm_moving_average(self) -> float:
        return self.cfg["env"]["armMovingAverage"]

    @property
    def hand_dof_speed_scale(self) -> float:
        return self.cfg["env"]["dofSpeedScale"]

    @staticmethod
    def interpolate(init, final, alpha: float) -> float:
        assert 0 <= alpha <= 1, f"alpha must be between 0 and 1, got {alpha}"
        return init + (final - init) * alpha

    def post_physics_step(self):
        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self.populate_sim_buffers()
        rewards, is_success = self.compute_kuka_reward()
        self.populate_obs_and_states_buffers()

        self.clamp_obs()

        pass

        pass

        pass

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            YELLOW = (1, 1, 0)
            WHITE = (1, 1, 1)
            BLACK = (0, 0, 0)
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.01, 8, 8, sphere_pose, color=YELLOW
            )
            sphere_geom_white = gymutil.WireframeSphereGeometry(
                0.02, 8, 8, sphere_pose, color=WHITE
            )
            sphere_geom_black = gymutil.WireframeSphereGeometry(
                0.01, 8, 8, sphere_pose, color=BLACK
            )

            palm_center_pos_cpu = self.palm_center_pos.cpu().numpy()
            palm_rot_cpu = self._palm_rot.cpu().numpy()

            for i in range(self.num_envs):
                palm_center_transform = gymapi.Transform()
                palm_center_transform.p = gymapi.Vec3(*palm_center_pos_cpu[i])
                palm_center_transform.r = gymapi.Quat(*palm_rot_cpu[i])
                gymutil.draw_lines(
                    sphere_geom_white,
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    palm_center_transform,
                )

            for j in range(self.num_fingertips):
                fingertip_pos_cpu = self.fingertip_pos_offset[:, j].cpu().numpy()
                fingertip_rot_cpu = self.fingertip_rot[:, j].cpu().numpy()

                for i in range(self.num_envs):
                    fingertip_transform = gymapi.Transform()
                    fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])
                    fingertip_transform.r = gymapi.Quat(*fingertip_rot_cpu[i])

                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        fingertip_transform,
                    )

            for i in range(self.num_envs):
                rb_forces_cpu = (
                    self.rb_forces[i, self.object_rb_handles, :]
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )
                assert rb_forces_cpu.shape == (3,), (
                    f"rb_forces_cpu.shape: {rb_forces_cpu.shape}"
                )
                object_pos_cpu = self.object_pos[i].cpu().numpy()
                assert object_pos_cpu.shape == (3,), (
                    f"object_pos_cpu.shape: {object_pos_cpu.shape}"
                )
                start_pos = gymapi.Vec3(*object_pos_cpu)
                MAX_FORCE_NORM = self.force_scale * 0.1  # Often mass is about 0.1 kg
                MAX_VECTOR_LENGTH = 0.3
                force_norm = np.linalg.norm(rb_forces_cpu)
                if force_norm > MAX_FORCE_NORM:
                    rb_forces_cpu = rb_forces_cpu / force_norm * MAX_FORCE_NORM
                vector = rb_forces_cpu * MAX_VECTOR_LENGTH / (MAX_FORCE_NORM + 1e-6)
                end_pos = start_pos + gymapi.Vec3(*vector)
                PURPLE = (1, 0, 1)
                self._draw_debug_line_of_spheres(
                    env=self.envs[i],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    color=PURPLE,
                )

            for i in range(self.num_envs):
                rb_torques_cpu = (
                    self.rb_torques[i, self.object_rb_handles, :]
                    .cpu()
                    .numpy()
                    .squeeze(axis=0)
                )
                assert rb_torques_cpu.shape == (3,), (
                    f"rb_torques_cpu.shape: {rb_torques_cpu.shape}"
                )
                object_pos_cpu = self.object_pos[i].cpu().numpy()
                assert object_pos_cpu.shape == (3,), (
                    f"object_pos_cpu.shape: {object_pos_cpu.shape}"
                )
                start_pos = gymapi.Vec3(*object_pos_cpu)
                MAX_TORQUE_NORM = self.torque_scale * 0.1  # Often mass is about 0.1 kg
                MAX_VECTOR_LENGTH = 0.3
                torque_norm = np.linalg.norm(rb_torques_cpu)
                if torque_norm > MAX_TORQUE_NORM:
                    rb_torques_cpu = rb_torques_cpu / torque_norm * MAX_TORQUE_NORM
                vector = rb_torques_cpu * MAX_VECTOR_LENGTH / (MAX_TORQUE_NORM + 1e-6)
                end_pos = start_pos + gymapi.Vec3(*vector)
                CYAN = (0, 1, 1)
                self._draw_debug_line_of_spheres(
                    env=self.envs[i],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    color=CYAN,
                )

            for j in range(self.num_keypoints):
                keypoint_pos_cpu = self.obj_keypoint_pos[:, j].cpu().numpy()
                goal_keypoint_pos_cpu = self.goal_keypoint_pos[:, j].cpu().numpy()
                keypoint_pos_fixed_size_cpu = (
                    self.obj_keypoint_pos_fixed_size[:, j].cpu().numpy()
                )
                goal_keypoint_pos_fixed_size_cpu = (
                    self.goal_keypoint_pos_fixed_size[:, j].cpu().numpy()
                )

                for i in range(self.num_envs):
                    keypoint_transform = gymapi.Transform()
                    keypoint_transform.p = gymapi.Vec3(*keypoint_pos_cpu[i])
                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        keypoint_transform,
                    )

                    goal_keypoint_transform = gymapi.Transform()
                    goal_keypoint_transform.p = gymapi.Vec3(*goal_keypoint_pos_cpu[i])
                    gymutil.draw_lines(
                        sphere_geom,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        goal_keypoint_transform,
                    )

                    keypoint_transform_fixed_size = gymapi.Transform()
                    keypoint_transform_fixed_size.p = gymapi.Vec3(
                        *keypoint_pos_fixed_size_cpu[i]
                    )
                    gymutil.draw_lines(
                        sphere_geom_black,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        keypoint_transform_fixed_size,
                    )

                    goal_keypoint_transform_fixed_size = gymapi.Transform()
                    goal_keypoint_transform_fixed_size.p = gymapi.Vec3(
                        *goal_keypoint_pos_fixed_size_cpu[i]
                    )
                    gymutil.draw_lines(
                        sphere_geom_black,
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        goal_keypoint_transform_fixed_size,
                    )

            # Visualize object and goal pose
            for i in range(self.num_envs):
                object_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.object_pos[i]),
                    r=gymapi.Quat(*self.object_rot[i]),
                )
                goal_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.goal_pos[i]),
                    r=gymapi.Quat(*self.goal_rot[i]),
                )
                self._draw_transform(transform=object_transform, env_idx=i)
                self._draw_transform(transform=goal_transform, env_idx=i)

    def _init_obs_action_queue(self):
        obs_queue_length = self.cfg["env"]["obsDelayMax"]
        action_queue_length = self.cfg["env"]["actionDelayMax"]
        object_state_queue_length = self.cfg["env"]["objectStateDelayMax"]
        self.obs_queue = torch.zeros(
            self.num_envs,
            obs_queue_length,
            self.num_observations,
            dtype=torch.float,
            device=self.device,
        )
        self.action_queue = torch.zeros(
            self.num_envs,
            action_queue_length,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
        )

        # Along QUEUE_LENGTH dimension, index=0 is the most recent observation/action
        # At each step, we update the queue by shifting the queue down, then updating index=0 with the new observation/action
        # If use index=0 for all time, then there is no delay
        # But we can sample delay index to simulate stochastic delay
        # Need to be careful on reset, need to flush the queue with the first obs/action

        # The object state queue is similar, but we want to have more delay on object state than the rest of the observations because it is
        # More noise and more latency/delay with FoundationPose and more likely to be incorrect in the real world
        # So we want to have more delay on object state than the rest of the observations
        # These will store the object state WITHOUT any noise
        # If we want to add noise to these, we will add noise on top of it after the object state is sampled from the queue
        self.object_state_queue = torch.zeros(
            self.num_envs,
            object_state_queue_length,
            13,
            dtype=torch.float,
            device=self.device,
        )

    def _initialize_camera_sensor(self, cam_pos, cam_target) -> None:
        self.camera_properties = gymapi.CameraProperties()
        resolution_reduction_factor_to_save_space = max(
            1,
            int(self.cfg["env"].get("cameraResolutionReductionFactor", 4)),
        )
        self.camera_properties.width = int(
            self.camera_properties.width / resolution_reduction_factor_to_save_space
        )
        self.camera_properties.height = int(
            self.camera_properties.height / resolution_reduction_factor_to_save_space
        )
        self.camera_handle = self.gym.create_camera_sensor(
            self.envs[self.index_to_view],
            self.camera_properties,
        )

        self.gym.set_camera_location(
            self.camera_handle, self.envs[self.index_to_view], cam_pos, cam_target
        )

    def _modify_render_settings_if_headless(self) -> None:
        # If not headless, leave things as they are
        if self.viewer is not None:
            return

        # If headless, we should default to having self.enable_viewer_sync=False to speed up env stepping
        self.enable_viewer_sync = False

    def _draw_transform(
        self, transform: gymapi.Transform, line_length: float = 0.2, env_idx: int = 0
    ) -> None:
        env = self.envs[env_idx]

        origin = transform.transform_point(gymapi.Vec3(0, 0, 0))
        x_dir = transform.transform_point(gymapi.Vec3(line_length, 0, 0))
        y_dir = transform.transform_point(gymapi.Vec3(0, line_length, 0))
        z_dir = transform.transform_point(gymapi.Vec3(0, 0, line_length))

        RED = (1, 0, 0)
        GREEN = (0, 1, 0)
        BLUE = (0, 0, 1)

        for color, dir in zip([RED, GREEN, BLUE], [x_dir, y_dir, z_dir]):
            gymutil.draw_line(
                p1=origin,
                p2=dir,
                color=gymapi.Vec3(*color),  # type: ignore
                gym=self.gym,
                viewer=self.viewer,
                env=env,
            )
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=origin,
                end_pos=dir,
                color=color,
            )

    def _draw_debug_line_of_spheres(
        self,
        env,
        start_pos: gymapi.Vec3,
        end_pos: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.01,
        num_spheres: int = 10,
    ) -> None:
        for i in range(num_spheres):
            fraction = (i + 1) / (num_spheres + 1)
            pos = start_pos + ((end_pos - start_pos) * fraction)
            self._draw_debug_sphere(
                env=env,
                position=pos,
                color=color,
                radius=radius,
            )

    def _draw_debug_sphere(
        self,
        env,
        position: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.005,
        num_lats: int = 10,
        num_lons: int = 10,
    ) -> None:
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius, num_lats, num_lons, color=color
        )
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, env, gymapi.Transform(p=position)
        )

    def set_robot_asset_rigid_shape_properties(
        self,
        robot_asset: gymapi.Asset,
        friction: Optional[float],
        fingertip_friction: Optional[float],
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(robot_asset),
        )

        # Different friction for normal links (low friction) and fingertips (high friction)
        for i in range(len(rigid_shape_props)):
            if friction is not None:
                rigid_shape_props[i].friction = friction

        # Rigid bodies (links) are not the same as rigid shapes (collision geometries)
        # Each rigid body can have >=1 rigid shapes
        rb_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(f"rb_names = {rb_names}")
        rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        assert_equals(len(rb_names), len(rb_shape_indices))
        rb_name_to_shape_indices = {
            name: (x.start, x.count) for name, x in zip(rb_names, rb_shape_indices)
        }
        fingertip_names = self.fingertips
        for name in fingertip_names:
            start, count = rb_name_to_shape_indices[name]
            for i in range(start, start + count):
                if fingertip_friction is not None:
                    rigid_shape_props[i].friction = fingertip_friction

        # Turn off self-collisions for adjacent links
        from isaacgymenvs.tasks.simtoolreal.adjacent_links import (
            LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
            RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS,
        )

        if self.use_sharpa:
            if self.use_right_sharpa:
                link_to_adjacent_links = RIGHT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
            elif self.use_left_sharpa:
                link_to_adjacent_links = LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS
            else:
                raise ValueError(f"Invalid use_sharpa: {self.use_sharpa}")
        else:
            raise ValueError(f"Invalid use_sharpa: {self.use_sharpa}")

        assert set(link_to_adjacent_links.keys()).issubset(rb_names), (
            f"Some links are not in the asset {robot_asset}, rb_names: {rb_names}, link_to_adjacent_links: {link_to_adjacent_links}, only in link_to_adjacent_links: {set(link_to_adjacent_links.keys()) - set(rb_names)}, only in rb_names: {set(rb_names) - set(link_to_adjacent_links.keys())}"
        )
        assert set(sum(link_to_adjacent_links.values(), [])).issubset(rb_names), (
            f"Some links are not in the asset {robot_asset}, rb_names: {rb_names}, link_to_adjacent_links: {link_to_adjacent_links}, only in link_to_adjacent_links: {set(sum(link_to_adjacent_links.values(), [])) - set(rb_names)}, only in rb_names: {set(rb_names) - set(sum(link_to_adjacent_links.values(), []))}"
        )

        no_collision_pairs = set()
        for link, adjacent_links in link_to_adjacent_links.items():
            for adjacent_link in adjacent_links:
                no_collision_pairs.add(tuple(sorted((link, adjacent_link))))
        no_collision_pairs = sorted(list(no_collision_pairs))

        # Set collision_filters
        # collision if (filterA & filterB) == 0
        # Assign unique bit per link (up to 32 bits)
        link_bitmask = {name: 1 << i for i, name in enumerate(rb_names)}

        # For each no-collision pair, share bits so they don't collide
        for a, b in no_collision_pairs:
            bit_a = link_bitmask[a]
            bit_b = link_bitmask[b]
            # Add b's bit to a, and a's bit to b → ensures (filterA & filterB) != 0
            link_bitmask[a] |= bit_b
            link_bitmask[b] |= bit_a

        # Update filters on shapes again
        for name, (start, count) in rb_name_to_shape_indices.items():
            bit = link_bitmask[name]
            for i in range(start, start + count):
                rigid_shape_props[i].filter = bit

        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

    def set_table_asset_rigid_shape_properties(
        self, table_asset: gymapi.Asset, friction: float
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(table_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = friction
        self.gym.set_asset_rigid_shape_properties(table_asset, rigid_shape_props)

    def set_object_asset_rigid_shape_properties(
        self, object_asset: gymapi.Asset, friction: float
    ):
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        assert_equals(
            len(rigid_shape_props),
            self.gym.get_asset_rigid_shape_count(object_asset),
        )
        for i in range(len(rigid_shape_props)):
            rigid_shape_props[i].friction = friction
        self.gym.set_asset_rigid_shape_properties(object_asset, rigid_shape_props)

    def set_object_masses_and_inertias(
        self,
        envs: List[gymapi.Env],
        objects: List[int],
        masses: List[float],
        inertias: List[Tuple[float, float, float]],
    ):
        for env, object, mass, inertia in zip(envs, objects, masses, inertias):
            object_rb_props = self.gym.get_actor_rigid_body_properties(env, object)
            OBJECT_NUM_RIGID_BODIES = 1
            assert_equals(len(object_rb_props), OBJECT_NUM_RIGID_BODIES)
            for i in range(OBJECT_NUM_RIGID_BODIES):
                object_rb_props[i].mass = mass
                object_rb_props[i].inertia.x.x = inertia[0]
                object_rb_props[i].inertia.y.y = inertia[1]
                object_rb_props[i].inertia.z.z = inertia[2]
            self.gym.set_actor_rigid_body_properties(env, object, object_rb_props)

    @property
    def VISUALIZE_PD_TARGET_AS_BLUE_ROBOT(self) -> bool:
        if "VISUALIZE_PD_TARGET_AS_BLUE_ROBOT" in self.cfg["env"]:
            return self.cfg["env"]["VISUALIZE_PD_TARGET_AS_BLUE_ROBOT"]
        return False

    def sample_random_unit_axis(self, shape) -> Tensor:
        v = torch_rand_float(0.0, 1.0, shape, device=self.device)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        return v

    def sample_delta_quat_xyzw(
        self, input_quat_xyzw: Tensor, delta_rotation_degrees: float
    ) -> Tensor:
        N, D = input_quat_xyzw.shape
        assert D == 4, (
            f"input_quat_xyzw.shape: {input_quat_xyzw.shape}, expected: (N, 4)"
        )

        quat_wxyz = torch.cat((input_quat_xyzw[:, 3:], input_quat_xyzw[:, :3]), dim=1)
        quat_matrix = quaternion_to_matrix(quat_wxyz)

        delta_rotation_radians = delta_rotation_degrees * np.pi / 180.0
        random_direction = self.sample_random_unit_axis(shape=(N, 3))
        sampled_rotation_magnitude = torch_rand_float(
            -delta_rotation_radians, delta_rotation_radians, (N, 1), device=self.device
        )
        sampled_rotation_axis_angles = random_direction * sampled_rotation_magnitude
        sampled_rotation_matrix = axis_angle_to_matrix(sampled_rotation_axis_angles)
        new_matrix = quat_matrix @ sampled_rotation_matrix

        new_quat_wxyz = matrix_to_quaternion(new_matrix)
        new_quat_xyzw = torch.cat(
            (new_quat_wxyz[:, 1:], new_quat_wxyz[:, 0:1]), dim=1
        ).clone()
        assert new_quat_xyzw.shape == (N, 4), (
            f"new_quat_xyzw.shape: {new_quat_xyzw.shape}, expected: (N, 4)"
        )
        return new_quat_xyzw
