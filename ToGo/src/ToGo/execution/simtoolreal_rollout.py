#!/usr/bin/env python3
"""Execute fixed goals with the supplied SimToolReal policy; return runtime state."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ToGo/src"))
for path in (ROOT / "simtoolreal", ROOT / "simtoolreal/rl_games",
             Path(os.environ.get("TOGO_ISAACGYM", str(ROOT / "isaacgym/python")))):
    sys.path.insert(0, str(path))

from ToGo.execution.runtime import _install_numpy_legacy_aliases, scene_actor_specs
from ToGo.pipeline.handoff import read_json, write_json
from ToGo.validation.validate_parttoolgen import validate_simtoolreal_fixed_goals


def environment_overrides(task: Mapping[str, Any], goals: Mapping[str, Any],
                          settings: Mapping[str, Any]) -> dict[str, Any]:
    state = task["execution"]["state_bundle"]
    values = {
        "resetPositionNoiseX": 0.0, "resetPositionNoiseY": 0.0, "resetPositionNoiseZ": 0.0,
        "randomizeObjectRotation": False, "resetDofPosRandomIntervalFingers": 0.0,
        "resetDofPosRandomIntervalArm": 0.0, "resetDofVelRandomInterval": 0.0,
        "tableResetZRange": 0.0, "objectName": state["object_name"], "numEnvs": 1,
        "envSpacing": 0.4,
        "useFixedGoalStates": True, "fixedGoalStates": goals["goals"],
        "useActionDelay": False, "useObsDelay": False, "useObjectStateDelayNoise": False,
        "objectScaleNoiseMultiplierRange": [1.0, 1.0], "resetWhenDropped": False,
        "armMovingAverage": 0.1, "goalTolerance": float(settings["goal_tolerance_m"]),
        "successSteps": int(settings["goal_hold_steps"]), "forceConsecutiveNearGoalSteps": True,
        "useToolFrameOrientationSuccess": False, "fixedSizeKeypointReward": True,
        "useFixedInitObjectPose": True, "objectStartPose": goals["start_pose"],
        "startArmHigher": False, "forceScale": 0.0, "torqueScale": 0.0,
        "linVelImpulseScale": 0.0, "angVelImpulseScale": 0.0, "forceNoReset": True,
        "use_hack_object_pos_offset": False, "hideGoalObjectVisual": True,
        "asset.table": state["table_asset"], "tableResetZ": float(state["table_pose_xyzw"][2]),
        "sceneActors": scene_actor_specs(dict(state)),
    }
    return {"task.env." + key: value for key, value in values.items()}


def execute(task: dict[str, Any], goals: dict[str, Any], settings: dict[str, Any],
            device: str) -> dict[str, Any]:
    validate_simtoolreal_fixed_goals(goals)
    policy_root = Path(os.environ.get("TOGO_PRETRAINED_POLICY_ROOT", str(ROOT / "pretrained_policy")))
    config, checkpoint = policy_root / "config.yaml", policy_root / "model.pth"
    for path in (config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"missing policy asset: {path}; set TOGO_PRETRAINED_POLICY_ROOT")
    _install_numpy_legacy_aliases()
    # IsaacGym enforces this import order.
    from deployment.isaac.isaac_env import create_env
    from deployment.isaac.isaac_env_no_ros import IsaacEnvNoRos, N_OBS, N_ACT, compute_observation
    from deployment.rl_player import RlPlayer
    from isaacgymenvs.utils.observation_action_utils_sharpa import create_urdf_object
    import torch

    env = create_env(config_path=str(config), headless=True, device=device,
                     episode_length=int(settings["max_steps"]) + 5,
                     overrides=environment_overrides(task, goals, settings))
    try:
        policy = RlPlayer(N_OBS, N_ACT, str(config), str(checkpoint), device, env.num_envs)
        if policy.environment_state is not None:
            env.set_env_state(policy.environment_state)
        urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")
        wrapper = IsaacEnvNoRos(env, 1.0 / float(settings["control_hz"]), device, urdf)

        def observation() -> Any:
            actual, target = env.object_pose.clone(), env.goal_pose.clone()
            actual[:, :3], target[:, :3] = env.object_pos, env.goal_pos
            result = compute_observation(q=env.arm_hand_dof_pos.cpu().numpy(),
                qd=env.arm_hand_dof_vel.cpu().numpy(), prev_action_targets=env.prev_targets.cpu().numpy(),
                object_pose=actual.cpu().numpy(), goal_object_pose=target.cpu().numpy(),
                object_scales=env.object_scales.cpu().numpy(), urdf=urdf, obs_list=env.obs_list)
            return torch.from_numpy(result).float().to(device)

        wrapper.reset()
        obs = observation()
        steps = 0
        for steps in range(1, int(settings["max_steps"]) + 1):
            if int(env.successes[0].item()) >= len(goals["goals"]):
                steps -= 1
                break
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            obs, _, _, _ = wrapper.step(action)
        progress = min(int(env.successes[0].item()), len(goals["goals"]))
        return {"status": "completed" if progress == len(goals["goals"]) else "timeout",
                "steps": steps, "goal_index": progress, "num_goals": len(goals["goals"]),
                "final_tool_pose_xyzw": env.object_pose[0].detach().cpu().tolist()}
    finally:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--fixed-goals", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        state = execute(read_json(args.task), read_json(args.fixed_goals), read_json(args.settings), args.device)
    except Exception as error:
        write_json(args.output / "execution_state.json", {"status": "error", "error": str(error)})
        raise
    write_json(args.output / "execution_state.json", state)
    return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
