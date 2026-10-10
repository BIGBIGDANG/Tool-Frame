# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
import torch
# isort: on

from typing import Tuple

import yourdfpy

from isaacgymenvs.utils.observation_action_utils_sharpa import compute_joint_pos_targets, compute_observation

N_OBS = 140
N_ACT = 29

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.05
HAND_DOF_SPEED_SCALE = 2.5


class IsaacEnvNoRos:
    def __init__(
        self,
        env: SimToolReal,
        control_dt: float,
        device: str,
        urdf: yourdfpy.URDF,
    ):
        self.env = env
        self.control_dt = control_dt
        self.device = device
        self.urdf = urdf

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(
            torch.zeros((self.env.num_envs, N_ACT), device=self.device)
        )
        return obs["obs"]

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=self.env.prev_targets.cpu().numpy(),
            hand_moving_average=HAND_MOVING_AVERAGE,
            arm_moving_average=ARM_MOVING_AVERAGE,
            hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
            dt=self.control_dt,
        )
        joint_pos_targets = torch.from_numpy(joint_pos_targets).float().to(self.device)

        obs, reward, done, info = self.env.step(
            action, joint_pos_targets=joint_pos_targets
        )
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        object_pose = self.env.object_pose.clone()
        goal_object_pose = self.env.goal_pose.clone()
        # SimToolReal can intentionally bias the observed object/goal positions
        # toward a grasp proxy (for example the handle side of a tool).  The
        # native env observation applies that bias through object_pos/goal_pos;
        # mirror it here when recomputing the no-ROS observation so the wrapper
        # does not silently revert to the tool root after the first frame.
        object_pose[:, 0:3] = self.env.object_pos
        goal_object_pose[:, 0:3] = self.env.goal_pos
        object_scales = self.env.object_scales


        new_obs = compute_observation(
            q=q.cpu().numpy(),
            qd=qd.cpu().numpy(),
            prev_action_targets=self.env.prev_targets.cpu().numpy(),
            object_pose=object_pose.cpu().numpy(),
            goal_object_pose=goal_object_pose.cpu().numpy(),
            object_scales=object_scales.cpu().numpy(),
            urdf=self.urdf,
            obs_list=self.env.obs_list,
        )
        new_obs = torch.from_numpy(new_obs).float().to(self.device)

        return new_obs, reward, done, info
