#!/usr/bin/env python3
"""Run fixed SimToolReal policy on PartToolGen fixed-goal artifacts.

This runner is deliberately thin: it loads SimToolReal's existing
policy/config/checkpoint, executes fixed goal states through SimToolReal's
native fixed-goal config path, and writes rollout evidence plus an execution
report. It does not train the policy, alter the observation/action space, or
fork SimToolReal actor/goal-object visualization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence


PARTTOOLGEN_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PARTTOOLGEN_ROOT.parents[0]
SRC_ROOT = PARTTOOLGEN_ROOT / "src"
SIMTOOLREAL_ROOT = PROJECT_ROOT / "simtoolreal"
ISAACGYM_PYTHON_ROOT = PROJECT_ROOT / "artifacts" / "third_party" / "isaacgym" / "python"
for path in (
    str(SRC_ROOT),
    str(PARTTOOLGEN_ROOT),
    str(SIMTOOLREAL_ROOT),
    str(ISAACGYM_PYTHON_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    load_structured_file,
    validate_simtoolreal_fixed_goals,
    validate_tool_pose_subgoals,
)
from parttoolgen.adapters.simtoolreal_policy_priors import (  # noqa: E402
    DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
    fixed_goal_floor_z,
    table_top_z_from_state_bundle,
)
from parttoolgen.execution.reporting import build_execution_report  # noqa: E402


DEFAULT_PRETRAINED_POLICY_ROOT = Path(
    os.environ.get("PARTTOOLGEN_PRETRAINED_POLICY_ROOT", PROJECT_ROOT / "pretrained_policy")
)
DEFAULT_CONFIG = DEFAULT_PRETRAINED_POLICY_ROOT / "config.yaml"
DEFAULT_CHECKPOINT = DEFAULT_PRETRAINED_POLICY_ROOT / "model.pth"
DEFAULT_CONTROL_HZ = 60.0
DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M = 0.03
DEFAULT_SIMTOOLREAL_KEYPOINT_SCALE = 1.5
DEFAULT_SUCCESS_TOLERANCE_SAFETY_FACTOR = 0.9
DEFAULT_SUCCESS_STEPS = 3
DEFAULT_FORCE_CONSECUTIVE_NEAR_GOAL_STEPS = True
DEFAULT_RENDER_LIGHT_PRESET = "soft"
DEFAULT_SOFT_LIGHT_INDICES = (0, 1)
DEFAULT_SOFT_LIGHT_INTENSITY = (0.45, 0.45, 0.45)
DEFAULT_SOFT_LIGHT_AMBIENT = (0.78, 0.78, 0.78)
DEFAULT_SOFT_LIGHT_DIRECTION = (0.5, -0.8, 1.0)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of light indices")
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated integer list: {value!r}"
        ) from exc


def apply_simtoolreal_render_lighting(
    env: Any,
    preset: str = DEFAULT_RENDER_LIGHT_PRESET,
    *,
    soft_light_indices: Sequence[int] = DEFAULT_SOFT_LIGHT_INDICES,
) -> dict[str, Any]:
    """Apply the PartToolGen render lighting preset to a SimToolReal env."""
    if preset == "native":
        return {"preset": "native", "uses_simtoolreal_default": True}
    if preset != "soft":
        raise ValueError(f"unsupported light preset {preset!r}")
    from isaacgym import gymapi  # noqa: WPS433

    light_indices = tuple(int(item) for item in soft_light_indices)
    for light_index in light_indices:
        env.gym.set_light_parameters(
            env.sim,
            light_index,
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_INTENSITY),
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_AMBIENT),
            gymapi.Vec3(*DEFAULT_SOFT_LIGHT_DIRECTION),
        )
    return {
        "preset": "soft",
        "name": (
            "soft01"
            if light_indices == DEFAULT_SOFT_LIGHT_INDICES
            else ("soft02" if light_indices == (0, 2) else "soft_custom")
        ),
        "light_indices": list(light_indices),
        "intensity": list(DEFAULT_SOFT_LIGHT_INTENSITY),
        "ambient": list(DEFAULT_SOFT_LIGHT_AMBIENT),
        "direction": list(DEFAULT_SOFT_LIGHT_DIRECTION),
        "applied_to": "simtoolreal_env",
        "note": "local IsaacGym binding exposes intensity, ambient, and direction",
    }


def _simtoolreal_original_camera_spec() -> dict[str, Any]:
    return {
        "source": "SimToolReal original env.py camera initialization",
        "modified_by_parttoolgen": False,
        "position_world": [0.0, -0.42, 0.72],
        "look_at_world": [0.0, 0.02, 0.50],
        "capture_source": "env.camera_handle",
        "resolution_reduction_factor_to_save_space": 4,
    }


def _write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _quat_angle_rad(a: list[float], b: list[float]) -> float:
    def normalize(q: list[float]) -> list[float]:
        mag = math.sqrt(sum(float(x) * float(x) for x in q))
        if mag <= 1e-12:
            return [0.0, 0.0, 0.0, 1.0]
        return [float(x) / mag for x in q]

    qa = normalize(a)
    qb = normalize(b)
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _normalize_quat_xyzw(q: list[float]) -> list[float]:
    mag = math.sqrt(sum(float(x) * float(x) for x in q))
    if mag <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(x) / mag for x in q[:4]]


def _quat_conjugate_xyzw(q: list[float]) -> list[float]:
    x, y, z, w = _normalize_quat_xyzw(q)
    return [-x, -y, -z, w]


def _quat_multiply_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = [float(item) for item in a[:4]]
    bx, by, bz, bw = [float(item) for item in b[:4]]
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_rotate_xyzw(q: list[float], v: list[float]) -> list[float]:
    normalized_q = _normalize_quat_xyzw(q)
    rotated = _quat_multiply_xyzw(
        _quat_multiply_xyzw(normalized_q, [float(v[0]), float(v[1]), float(v[2]), 0.0]),
        _quat_conjugate_xyzw(normalized_q),
    )
    return [float(rotated[0]), float(rotated[1]), float(rotated[2])]


def _add3(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a[:3], b[:3])]


def _sub3(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a[:3], b[:3])]


def _dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def _torch_pose_to_list(tensor: Any) -> list[float]:
    values = tensor.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in values[:7]]


def _benchmark_extra_actor_specs(state_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    specs = state_bundle.get("parttoolgen_benchmark_extra_actors")
    if not isinstance(specs, list):
        return []
    return [item for item in specs if isinstance(item, dict)]


def _benchmark_extra_actor_states(env: Any, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not specs or not hasattr(env, "parttoolgen_benchmark_extra_actor_indices"):
        return []
    indices = getattr(env, "parttoolgen_benchmark_extra_actor_indices")
    if not hasattr(indices, "numel") or indices.numel() == 0:
        return []
    root_states = env.root_state_tensor[indices[0].reshape(-1), :13].detach().cpu().tolist()
    records: list[dict[str, Any]] = []
    for idx, state in enumerate(root_states[: len(specs)]):
        spec = specs[idx]
        records.append(
            {
                "name": str(spec.get("name") or f"parttoolgen_extra_actor_{idx}"),
                "pose_xyzw": [float(item) for item in state[:7]],
                "linear_velocity_world": [float(item) for item in state[7:10]],
                "angular_velocity_world": [float(item) for item in state[10:13]],
            }
        )
    return records


def _benchmark_extra_actor_motion_summary(
    *,
    initial: list[dict[str, Any]],
    final: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_by_name = {item["name"]: item for item in initial if isinstance(item, dict)}
    actors = []
    for item in final:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        start = initial_by_name.get(name)
        if not isinstance(start, dict):
            continue
        start_pose = start.get("pose_xyzw")
        final_pose = item.get("pose_xyzw")
        if not isinstance(start_pose, list) or not isinstance(final_pose, list):
            continue
        displacement = _dist3(final_pose, start_pose)
        actors.append(
            {
                "name": name,
                "initial_pose_xyzw": start_pose,
                "final_pose_xyzw": final_pose,
                "displacement_m": displacement,
                "xy_displacement_m": _dist3(
                    [start_pose[0], start_pose[1], 0.0],
                    [final_pose[0], final_pose[1], 0.0],
                ),
                "z_delta_m": float(final_pose[2]) - float(start_pose[2]),
            }
        )
    return {
        "checked": bool(actors),
        "actors": actors,
        "max_displacement_m": max((float(item["displacement_m"]) for item in actors), default=0.0),
        "max_xy_displacement_m": max((float(item["xy_displacement_m"]) for item in actors), default=0.0),
    }


def _source_goal_for_fixed_goal(
    tool_pose_subgoals: dict[str, Any] | None,
    goal_index: int,
    goal_index_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not tool_pose_subgoals:
        return None
    source_index = goal_index
    if goal_index_map:
        mapped = next(
            (
                item
                for item in goal_index_map
                if int(item.get("fixed_goal_index", -1)) == goal_index
            ),
            None,
        )
        if mapped:
            if mapped.get("source_goal_index") is None:
                return None
            source_index = int(mapped["source_goal_index"])
    goals = tool_pose_subgoals.get("goals", [])
    if source_index < 0 or source_index >= len(goals):
        return None
    goal = goals[source_index]
    return goal if isinstance(goal, dict) else None


def _scene_world_point(scene_graph: dict[str, Any] | None, ref: str | None) -> list[float] | None:
    if not scene_graph or not ref:
        return None
    keypoint_name = None
    if ref.startswith("scene.keypoints."):
        keypoint_name = ref[len("scene.keypoints.") :]
    elif ref.startswith("keypoints."):
        keypoint_name = ref[len("keypoints.") :]
    elif ref in scene_graph.get("keypoints", {}):
        keypoint_name = ref
    if not keypoint_name:
        return None
    keypoint = scene_graph.get("keypoints", {}).get(keypoint_name)
    if not isinstance(keypoint, dict):
        return None
    xyz = (
        keypoint.get("xyz_world")
        or keypoint.get("position_world")
        or keypoint.get("point_xyz_world")
    )
    if isinstance(xyz, list) and len(xyz) >= 3:
        return [float(item) for item in xyz[:3]]
    return None


def _functional_observation_spec(
    *,
    tool_pose_subgoals: dict[str, Any] | None,
    goal_index: int,
    scene_graph: dict[str, Any] | None,
    goal_index_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    goal = _source_goal_for_fixed_goal(tool_pose_subgoals, goal_index, goal_index_map)
    if not goal:
        return None
    contact_point_goal = goal.get("contact_point_goal")
    pose_world_tool = goal.get("pose_world_tool")
    if not isinstance(contact_point_goal, dict) or not isinstance(pose_world_tool, dict):
        return None
    contact_position = contact_point_goal.get("position_world")
    pose_position = pose_world_tool.get("position")
    pose_orientation = pose_world_tool.get("orientation_xyzw")
    if (
        not isinstance(contact_position, list)
        or len(contact_position) < 3
        or not isinstance(pose_position, list)
        or len(pose_position) < 3
        or not isinstance(pose_orientation, list)
        or len(pose_orientation) < 4
    ):
        return None
    local_point = _quat_rotate_xyzw(
        _quat_conjugate_xyzw([float(item) for item in pose_orientation[:4]]),
        _sub3(contact_position, pose_position),
    )
    expected_contact = goal.get("expected_contact", {})
    target_ref = (
        expected_contact.get("scene_target")
        if isinstance(expected_contact, dict)
        else None
    )
    stage = str(goal.get("stage", ""))
    use_planned_target = _goal_uses_planned_functional_target(goal, stage)
    target_world = None if use_planned_target else _scene_world_point(scene_graph, target_ref)
    unchecked_reason = None
    if target_ref and target_world is None and not use_planned_target:
        unchecked_reason = "scene_target_ref_unresolved"
    target_source = "scene_ref" if target_world is not None else "planned_contact_point_goal"
    if target_world is None:
        target_world = [float(item) for item in contact_position[:3]]
    return {
        "tool_ref": contact_point_goal.get("part_ref")
        or (
            expected_contact.get("tool_part")
            if isinstance(expected_contact, dict)
            else None
        ),
        "target_ref": target_ref,
        "point_tool": local_point,
        "planned_point_world": [float(item) for item in contact_position[:3]],
        "target_world": target_world,
        "target_source": target_source,
        "unchecked_reason": unchecked_reason,
        "semantic_gate": _goal_semantic_gate(goal, stage),
    }


def _goal_uses_planned_functional_target(goal: dict[str, Any], stage: str) -> bool:
    if not _goal_semantic_gate(goal, stage):
        return True
    return "_transition_" in stage


def _goal_semantic_gate(goal: dict[str, Any] | None, stage: str) -> bool:
    if isinstance(goal, dict):
        explicit_gate = goal.get("semantic_gate")
        if isinstance(explicit_gate, bool):
            return explicit_gate
        semantic_role = goal.get("semantic_role")
        if isinstance(semantic_role, str) and semantic_role == "generated_transition_waypoint":
            return False
    return not _stage_is_generated_path_waypoint(stage)


def _stage_is_generated_path_waypoint(stage: str) -> bool:
    return "_transition_" in stage


def _functional_observation_from_pose(
    *,
    actual_pose: list[float],
    spec: dict[str, Any] | None,
    max_functional_target_error_m: float | None = None,
    max_contact_point_target_error_m: float | None = None,
) -> dict[str, Any] | None:
    if not spec:
        return None
    if max_functional_target_error_m is None:
        max_functional_target_error_m = (
            DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M
            if max_contact_point_target_error_m is None
            else float(max_contact_point_target_error_m)
        )
    semantic_gate = bool(spec.get("semantic_gate", True))
    if spec.get("unchecked_reason"):
        observation = {
            "contact_point_tool_ref": spec.get("tool_ref"),
            "contact_target_ref": spec.get("target_ref"),
            "semantic_gate": semantic_gate,
            "contact_target_source": spec.get("target_source"),
            "contact_point_target_source": spec.get("target_source"),
        }
        if semantic_gate:
            observation["semantic_contact_check"] = {
                "checked": False,
                "passed": False,
                "unchecked_reason": spec.get("unchecked_reason"),
                "tool_part_ref": spec.get("tool_ref"),
                "scene_target_ref": spec.get("target_ref"),
                "source": "rollout_object_pose_x_tool_pose_subgoals_expected_contact",
                "provenance": {
                    "policy_modified": False,
                    "simtoolreal_low_level_modified": False,
                    "used_for_verification_only": True,
                },
            }
        return observation
    point_tool = spec.get("point_tool")
    if not isinstance(point_tool, list) or len(point_tool) < 3:
        return None
    actual_point = _add3(
        actual_pose[:3],
        _quat_rotate_xyzw(actual_pose[3:7], [float(item) for item in point_tool[:3]]),
    )
    planned_point = spec.get("planned_point_world")
    target_point = spec.get("target_world")
    if not isinstance(planned_point, list) or len(planned_point) < 3:
        return None
    if not isinstance(target_point, list) or len(target_point) < 3:
        return None
    contact_target_error_m = _dist3(actual_point, target_point)
    observation = {
        "contact_point_tool_ref": spec.get("tool_ref"),
        "contact_target_ref": spec.get("target_ref"),
        "semantic_gate": semantic_gate,
        "contact_point_tool": [float(item) for item in point_tool[:3]],
        "actual_contact_point_world": actual_point,
        "planned_contact_point_world": [float(item) for item in planned_point[:3]],
        "expected_contact_target_world": [float(item) for item in target_point[:3]],
        "contact_target_source": spec.get("target_source"),
        "contact_point_target_source": spec.get("target_source"),
        "contact_point_tracking_error_m": _dist3(actual_point, planned_point),
        "contact_target_error_m": contact_target_error_m,
        "functional_target_error_m": contact_target_error_m,
        "contact_target_success": contact_target_error_m
        <= float(max_functional_target_error_m),
        "functional_target_success": contact_target_error_m
        <= float(max_functional_target_error_m),
    }
    if semantic_gate:
        observation["semantic_contact_check"] = {
            "checked": True,
            "tool_part_ref": spec.get("tool_ref"),
            "scene_target_ref": spec.get("target_ref"),
            "contact_point_tool": [float(item) for item in point_tool[:3]],
            "planned_contact_point_world": [float(item) for item in planned_point[:3]],
            "actual_contact_point_world": actual_point,
            "target_position_world": [float(item) for item in target_point[:3]],
            "semantic_contact_error_m": contact_target_error_m,
            "threshold_m": float(max_functional_target_error_m),
            "passed": contact_target_error_m <= float(max_functional_target_error_m),
            "source": "rollout_object_pose_x_tool_pose_subgoals_expected_contact",
            "target_source": spec.get("target_source"),
            "provenance": {
                "policy_modified": False,
                "simtoolreal_low_level_modified": False,
                "used_for_verification_only": True,
            },
        }
    return observation


def _record_sample(
    *,
    step: int,
    goal_index: int,
    stage: str,
    actual_pose: list[float],
    goal_pose: list[float],
    successes: int,
    lifted: bool | None,
    closest_fingertip_dist: float | None,
    closest_fingertip_distances: list[float] | None = None,
    functional_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    translation_error = math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(actual_pose[:3], goal_pose[:3]))
    )
    sample = {
        "step": int(step),
        "goal_index": int(goal_index),
        "stage": stage,
        "actual_pose_xyzw": actual_pose,
        "goal_pose_xyzw": goal_pose,
        "tracking_error_m": translation_error,
        "rotation_error_rad": _quat_angle_rad(actual_pose[3:7], goal_pose[3:7]),
        "successes": int(successes),
        "lifted_object": lifted,
        "closest_fingertip_dist": closest_fingertip_dist,
        "closest_fingertip_distances": closest_fingertip_distances,
    }
    if functional_observation:
        sample.update(functional_observation)
    return sample


def _stage_error_summary(stage_reports: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        row
        for row in stage_reports
        if isinstance(row, dict) and row.get("tracking_gate_required") is True
    ]

    def values(key: str) -> list[float]:
        result = []
        for row in required:
            value = row.get(key)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    translation_errors = values("tracking_error_m")
    rotation_errors = values("rotation_error_rad")
    failed = [row for row in required if row.get("tracking_gate_success") is False]
    by_family: dict[str, dict[str, Any]] = {}
    for row in required:
        stage = str(row.get("stage") or "unknown")
        if stage.startswith("use_orientation"):
            family = "use_orientation"
        elif stage.startswith("grasp_lift"):
            family = "grasp_lift"
        elif stage.startswith("execution_start"):
            family = "execution_start"
        elif stage.startswith("approach"):
            family = "approach"
        elif stage.startswith("action"):
            family = "action"
        else:
            family = stage.split("_", 1)[0]
        bucket = by_family.setdefault(
            family,
            {
                "stage_count": 0,
                "failed_count": 0,
                "max_tracking_error_m": None,
                "max_rotation_error_rad": None,
                "first_failed_stage": None,
            },
        )
        bucket["stage_count"] += 1
        if row.get("tracking_gate_success") is False:
            bucket["failed_count"] += 1
            bucket["first_failed_stage"] = bucket["first_failed_stage"] or stage
        if isinstance(row.get("tracking_error_m"), (int, float)):
            value = float(row["tracking_error_m"])
            bucket["max_tracking_error_m"] = max(
                value,
                bucket["max_tracking_error_m"]
                if bucket["max_tracking_error_m"] is not None
                else value,
            )
        if isinstance(row.get("rotation_error_rad"), (int, float)):
            value = float(row["rotation_error_rad"])
            bucket["max_rotation_error_rad"] = max(
                value,
                bucket["max_rotation_error_rad"]
                if bucket["max_rotation_error_rad"] is not None
                else value,
            )
    return {
        "tracking_gate_required_stage_count": len(required),
        "tracking_gate_failed_count": len(failed),
        "max_tracking_error_m": max(translation_errors) if translation_errors else None,
        "max_rotation_error_rad": max(rotation_errors) if rotation_errors else None,
        "failed_goal_indices": [
            int(row["goal_index"])
            for row in failed
            if isinstance(row.get("goal_index"), int)
        ],
        "first_failed_stage": failed[0].get("stage") if failed else None,
        "by_stage_family": by_family,
    }


def _functional_specs_by_goal(
    *,
    fixed_goals: dict[str, Any],
    tool_pose_subgoals: dict[str, Any] | None,
    scene_graph: dict[str, Any] | None,
    goal_index_map: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        goal_index: spec
        for goal_index in range(len(fixed_goals.get("goals", [])))
        if (
            spec := _functional_observation_spec(
                tool_pose_subgoals=tool_pose_subgoals,
                goal_index=goal_index,
                scene_graph=scene_graph,
                goal_index_map=goal_index_map,
            )
        )
        is not None
    }


def _auto_observed_scene_graph_path(
    *,
    explicit_scene_path: str | Path | None,
    subgoals_path: str | Path | None,
) -> Path | None:
    if explicit_scene_path:
        return Path(explicit_scene_path)
    if not subgoals_path:
        return None
    candidate = Path(subgoals_path).parent / "observed_scene_graph.json"
    if candidate.exists():
        return candidate
    return None


def _pose7_close(a: list[float], b: list[float], *, tol: float = 1e-9) -> bool:
    return len(a) == 7 and len(b) == 7 and all(
        abs(float(x) - float(y)) <= tol for x, y in zip(a, b)
    )


def _derive_goal_index_map(
    fixed_goals: dict[str, Any],
    tool_pose_subgoals: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    explicit_map = _explicit_goal_index_map(fixed_goals)
    if explicit_map:
        return explicit_map
    if not tool_pose_subgoals:
        return []
    source_goals = tool_pose_subgoals.get("goals", [])
    fixed_goal_poses = fixed_goals.get("goals", [])
    if (
        isinstance(source_goals, list)
        and isinstance(fixed_goal_poses, list)
        and len(fixed_goal_poses) == len(source_goals) + 1
        and _pose7_close(fixed_goal_poses[0], fixed_goals.get("start_pose", []))
    ):
        goal_index_map = [
            {
                "fixed_goal_index": 0,
                "source_goal_index": None,
                "stage": "start_hold",
                "source": "simtoolreal_fixed_goals.goals[0]",
            }
        ]
        goal_index_map.extend(
            {
                "fixed_goal_index": index + 1,
                "source_goal_index": index,
                "stage": str(goal.get("stage", f"source_goal_{index}"))
                if isinstance(goal, dict)
                else f"source_goal_{index}",
                "source": "tool_pose_subgoals.goals",
            }
            for index, goal in enumerate(source_goals)
        )
        return goal_index_map
    return []


def _explicit_goal_index_map(fixed_goals: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = fixed_goals.get("metadata")
    candidates = []
    if isinstance(metadata, dict):
        candidates.append(metadata.get("goal_index_map"))
    candidates.append(fixed_goals.get("goal_index_map"))
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _stage_for_goal(
    tool_pose_subgoals: dict[str, Any] | None,
    goal_index: int,
    goal_index_map: list[dict[str, Any]] | None = None,
) -> str:
    if goal_index_map:
        mapped = next(
            (
                item
                for item in goal_index_map
                if int(item.get("fixed_goal_index", -1)) == goal_index
            ),
            None,
        )
        if mapped:
            source_index = mapped.get("source_goal_index")
            if source_index is None:
                return str(mapped.get("stage", f"goal_{goal_index}"))
            goal_index = int(source_index)
    if not tool_pose_subgoals:
        return f"goal_{goal_index}"
    goals = tool_pose_subgoals.get("goals", [])
    if goal_index >= len(goals) or not isinstance(goals[goal_index], dict):
        return f"goal_{goal_index}"
    return str(goals[goal_index].get("stage", f"goal_{goal_index}"))


def _event_from_info(info: dict[str, Any], key: str) -> bool:
    value = info.get(key)
    if hasattr(value, "sum"):
        return bool(value.sum().item())
    if isinstance(value, (int, float, bool)):
        return bool(value)
    return False


def _capture_camera_frame(env: Any) -> Any | None:
    """Capture one RGBA camera frame from the SimToolReal IsaacGym env."""

    if not all(hasattr(env, name) for name in ("gym", "sim", "envs", "camera_handle")):
        return None
    if not hasattr(env, "camera_properties"):
        return None
    from isaacgym import gymapi  # noqa: WPS433

    previous_sync = getattr(env, "enable_viewer_sync", None)
    try:
        if previous_sync is not None:
            env.enable_viewer_sync = True
        if hasattr(env, "render"):
            env.render()
        elif hasattr(env.gym, "step_graphics"):
            env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        color_image = env.gym.get_camera_image(
            env.sim,
            env.envs[getattr(env, "index_to_view", 0)],
            env.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        if getattr(color_image, "size", 0) == 0:
            return None
        return color_image.reshape(
            env.camera_properties.height,
            env.camera_properties.width,
            4,
        ).copy()
    finally:
        if previous_sync is not None:
            env.enable_viewer_sync = previous_sync


def _capture_camera_sensor_frame(env: Any, *, camera_handle: Any, camera_properties: Any) -> Any | None:
    """Capture one RGBA frame from an explicitly created IsaacGym camera."""

    if not all(hasattr(env, name) for name in ("gym", "sim", "envs")):
        return None
    from isaacgym import gymapi  # noqa: WPS433

    previous_sync = getattr(env, "enable_viewer_sync", None)
    try:
        if previous_sync is not None:
            env.enable_viewer_sync = True
        if hasattr(env, "render"):
            env.render()
        elif hasattr(env.gym, "step_graphics"):
            env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        color_image = env.gym.get_camera_image(
            env.sim,
            env.envs[getattr(env, "index_to_view", 0)],
            camera_handle,
            gymapi.IMAGE_COLOR,
        )
        if getattr(color_image, "size", 0) == 0:
            return None
        return color_image.reshape(
            int(camera_properties.height),
            int(camera_properties.width),
            4,
        ).copy()
    finally:
        if previous_sync is not None:
            env.enable_viewer_sync = previous_sync


def _create_rollout_topdown_camera(env: Any) -> tuple[Any, Any] | None:
    """Create a robot-forward top-down camera for rollout inspection."""

    if not all(hasattr(env, name) for name in ("gym", "sim", "envs")):
        return None
    try:
        from isaacgym import gymapi  # noqa: WPS433

        env_handle = env.envs[getattr(env, "index_to_view", 0)]
        props = gymapi.CameraProperties()
        props.width = 900
        props.height = 700
        props.horizontal_fov = 55.0
        handle = env.gym.create_camera_sensor(env_handle, props)
        env.gym.set_camera_location(
            handle,
            env_handle,
            gymapi.Vec3(0.0, -0.001, 1.42),
            gymapi.Vec3(0.0, 0.0, 0.53),
        )
        return handle, props
    except Exception:
        return None


def _create_side_board_front_camera(env: Any) -> tuple[Any, Any] | None:
    """Create a side-board front camera isolated from tabletop/topdown views."""

    if not all(hasattr(env, name) for name in ("gym", "sim", "envs")):
        return None
    try:
        from isaacgym import gymapi  # noqa: WPS433

        env_handle = env.envs[getattr(env, "index_to_view", 0)]
        props = gymapi.CameraProperties()
        props.width = 900
        props.height = 700
        props.horizontal_fov = 64.0
        handle = env.gym.create_camera_sensor(env_handle, props)
        env.gym.set_camera_location(
            handle,
            env_handle,
            gymapi.Vec3(-0.24, 0.0, 0.68),
            gymapi.Vec3(0.38, 0.0, 0.62),
        )
        return handle, props
    except Exception:
        return None


def _write_rollout_video(path: Path, frames: list[Any], *, fps: int) -> None:
    if not frames:
        return
    import imageio  # noqa: WPS433

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=int(fps))


def _can_capture_camera_frames(*, device: str, headless: bool) -> tuple[bool, str]:
    if str(device) == "cpu" and headless and "DISPLAY" not in os.environ:
        return (
            False,
            "IsaacGym camera rendering is not available for CPU headless runs "
            "without a DISPLAY; use a CUDA device or run under Xvfb/viewer mode.",
        )
    return True, ""


def _install_cpu_checkpoint_loader(device: str) -> None:
    """Force rl_games checkpoint restore onto CPU when no CUDA is available."""

    if str(device) != "cpu":
        return
    import torch  # noqa: WPS433
    from rl_games.algos_torch import torch_ext  # noqa: WPS433

    def safe_load_cpu(filename: str) -> Any:
        return torch_ext.safe_filesystem_op(
            torch.load,
            filename,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    torch_ext.safe_load = safe_load_cpu


def _install_pytorch3d_transform_fallback() -> None:
    """Provide the tiny transforms subset SimToolReal imports when unavailable.

    The fallback is registered only when the real ``pytorch3d.transforms`` cannot
    be found.  The functions import torch lazily so IsaacGym still gets imported
    before torch in the rollout path.
    """

    if "pytorch3d.transforms" in sys.modules:
        return
    try:
        if importlib.util.find_spec("pytorch3d.transforms") is not None:
            return
    except (ModuleNotFoundError, ValueError):
        pass

    package = types.ModuleType("pytorch3d")
    transforms = types.ModuleType("pytorch3d.transforms")

    def axis_angle_to_matrix(axis_angle: Any) -> Any:
        import torch  # noqa: WPS433

        angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
        axis = axis_angle / torch.clamp(angle, min=1e-12)
        x, y, z = axis.unbind(-1)
        zeros = torch.zeros_like(x)
        k = torch.stack(
            (
                zeros,
                -z,
                y,
                z,
                zeros,
                -x,
                -y,
                x,
                zeros,
            ),
            dim=-1,
        ).reshape(axis_angle.shape[:-1] + (3, 3))
        eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        eye = eye.expand(axis_angle.shape[:-1] + (3, 3))
        sin = torch.sin(angle)[..., None]
        cos = torch.cos(angle)[..., None]
        return eye + sin * k + (1.0 - cos) * (k @ k)

    def quaternion_to_matrix(quaternions: Any) -> Any:
        import torch  # noqa: WPS433

        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / torch.clamp((quaternions * quaternions).sum(-1), min=1e-12)
        matrix = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            dim=-1,
        )
        return matrix.reshape(quaternions.shape[:-1] + (3, 3))

    def matrix_to_quaternion(matrix: Any) -> Any:
        import torch  # noqa: WPS433

        m00 = matrix[..., 0, 0]
        m01 = matrix[..., 0, 1]
        m02 = matrix[..., 0, 2]
        m10 = matrix[..., 1, 0]
        m11 = matrix[..., 1, 1]
        m12 = matrix[..., 1, 2]
        m20 = matrix[..., 2, 0]
        m21 = matrix[..., 2, 1]
        m22 = matrix[..., 2, 2]

        def signed_sqrt(value: Any, sign_source: Any) -> Any:
            sign = torch.where(sign_source < 0.0, -1.0, 1.0)
            return sign * torch.sqrt(torch.clamp(value, min=0.0)) * 0.5

        qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0)) * 0.5
        qx = signed_sqrt(1.0 + m00 - m11 - m22, m21 - m12)
        qy = signed_sqrt(1.0 - m00 + m11 - m22, m02 - m20)
        qz = signed_sqrt(1.0 - m00 - m11 + m22, m10 - m01)
        quat = torch.stack((qw, qx, qy, qz), dim=-1)
        return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-12)

    transforms.axis_angle_to_matrix = axis_angle_to_matrix
    transforms.quaternion_to_matrix = quaternion_to_matrix
    transforms.matrix_to_quaternion = matrix_to_quaternion
    package.transforms = transforms
    sys.modules.setdefault("pytorch3d", package)
    sys.modules.setdefault("pytorch3d.transforms", transforms)


def _install_numpy_legacy_aliases() -> None:
    """Restore NumPy aliases expected by older robotics dependencies."""

    import numpy as np  # noqa: WPS433
    import warnings  # noqa: WPS433

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
    }.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            has_alias = hasattr(np, name)
        if not has_alias:
            setattr(np, name, value)


def _resolve_eval_success_tolerance_m(
    *,
    explicit_eval_success_tolerance_m: float | None,
    max_tracking_error_m: float,
    max_functional_target_error_m: float,
    simtoolreal_keypoint_scale: float = DEFAULT_SIMTOOLREAL_KEYPOINT_SCALE,
    safety_factor: float = DEFAULT_SUCCESS_TOLERANCE_SAFETY_FACTOR,
) -> dict[str, Any]:
    if explicit_eval_success_tolerance_m is not None:
        return {
            "value_m": float(explicit_eval_success_tolerance_m),
            "source": "explicit_cli_or_api",
            "max_tracking_error_m": float(max_tracking_error_m),
            "max_functional_target_error_m": float(max_functional_target_error_m),
            "simtoolreal_keypoint_scale": float(simtoolreal_keypoint_scale),
            "safety_factor": float(safety_factor),
        }
    scale = max(1e-9, float(simtoolreal_keypoint_scale))
    semantic_limited = (
        float(max_functional_target_error_m) / scale * float(safety_factor)
    )
    return {
        "value_m": min(float(max_tracking_error_m), semantic_limited),
        "source": "parttoolgen_semantic_threshold",
        "max_tracking_error_m": float(max_tracking_error_m),
        "max_functional_target_error_m": float(max_functional_target_error_m),
        "simtoolreal_keypoint_scale": scale,
        "safety_factor": float(safety_factor),
    }


def run_fixed_policy_rollout(
    *,
    fixed_goals_path: str | Path,
    output_dir: str | Path,
    observed_scene_graph_path: str | Path | None = None,
    subgoals_path: str | Path | None = None,
    state_bundle_path: str | Path | None = None,
    tool_mesh_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    object_name: str | None = None,
    max_steps: int = 900,
    control_hz: float = DEFAULT_CONTROL_HZ,
    device: str | None = None,
    headless: bool = True,
    require_contact: bool = False,
    max_tracking_error_m: float = 0.035,
    max_rotation_error_rad: float = 0.30,
    max_functional_target_error_m: float = DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M,
    eval_success_tolerance_m: float | None = None,
    success_steps: int = DEFAULT_SUCCESS_STEPS,
    force_consecutive_near_goal_steps: bool = DEFAULT_FORCE_CONSECUTIVE_NEAR_GOAL_STEPS,
    use_tool_frame_orientation_success: bool = False,
    tool_frame_orientation_success_tolerance_rad: float | None = None,
    tool_frame_orientation_success_stage_families: Sequence[str] | None = None,
    simtoolreal_keypoint_scale: float = DEFAULT_SIMTOOLREAL_KEYPOINT_SCALE,
    success_tolerance_safety_factor: float = DEFAULT_SUCCESS_TOLERANCE_SAFETY_FACTOR,
    force_no_reset: bool = False,
    grasp_observation_offset_m: float = 0.0,
    initial_grasp_hold_steps: int = 0,
    post_success_steps: int = 0,
    hide_goal_object_visual: bool = False,
    goal_object_visual_opacity: float = 0.32,
    capture_video: bool = False,
    capture_topdown_video: bool = False,
    capture_side_board_video: bool = False,
    video_fps: int = 30,
    video_stride: int = 2,
    light_preset: str = DEFAULT_RENDER_LIGHT_PRESET,
    soft_light_indices: Sequence[int] = DEFAULT_SOFT_LIGHT_INDICES,
) -> dict[str, Any]:
    fixed_goals = load_structured_file(fixed_goals_path)
    validate_simtoolreal_fixed_goals(fixed_goals)
    orientation_stage_families = [
        str(item)
        for item in (tool_frame_orientation_success_stage_families or [])
        if str(item).strip()
    ]
    tool_pose_subgoals = load_structured_file(subgoals_path) if subgoals_path else None
    if tool_pose_subgoals is not None:
        validate_tool_pose_subgoals(tool_pose_subgoals)
    goal_index_map = _derive_goal_index_map(fixed_goals, tool_pose_subgoals)
    resolved_observed_scene_graph_path = _auto_observed_scene_graph_path(
        explicit_scene_path=observed_scene_graph_path,
        subgoals_path=subgoals_path,
    )
    observed_scene_graph = (
        load_structured_file(resolved_observed_scene_graph_path)
        if resolved_observed_scene_graph_path
        else None
    )
    state_bundle = load_structured_file(state_bundle_path) if state_bundle_path else {}
    benchmark_extra_actors = _benchmark_extra_actor_specs(state_bundle)
    candidate_tool_id = object_name or state_bundle.get("object_name")
    if not candidate_tool_id:
        raise RuntimeError(
            "SimToolReal rollout requires --object-name or state_bundle.object_name; "
            "PartToolGen does not use a default tool fallback."
        )
    resolved_object_name = str(candidate_tool_id)
    resolved_table_asset = state_bundle.get("table_asset")
    table_pose = state_bundle.get("table_pose_xyzw")
    resolved_table_pose_z = (
        float(table_pose[2])
        if isinstance(table_pose, list) and len(table_pose) >= 3
        else None
    )
    table_clearance_check = {
        "checked": False,
        "disabled": True,
        "reason": "tabletop_clearance_preflight_removed",
        "note": (
            "Fixed-goal rollout does not apply an additional tabletop clearance "
            "guard. Contact geometry is defined by the compiler constraints, and "
            "physical feasibility is left to the SimToolReal rollout."
        ),
    }
    functional_specs = _functional_specs_by_goal(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=tool_pose_subgoals,
        scene_graph=observed_scene_graph,
        goal_index_map=goal_index_map,
    )
    success_tolerance_report = _resolve_eval_success_tolerance_m(
        explicit_eval_success_tolerance_m=eval_success_tolerance_m,
        max_tracking_error_m=max_tracking_error_m,
        max_functional_target_error_m=max_functional_target_error_m,
        simtoolreal_keypoint_scale=simtoolreal_keypoint_scale,
        safety_factor=success_tolerance_safety_factor,
    )
    resolved_eval_success_tolerance_m = float(success_tolerance_report["value_m"])
    resolved_success_steps = max(1, int(success_steps))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_spec_artifact = {
        "simtoolreal_environment": (
            "benchmark_extended" if benchmark_extra_actors else "original"
        ),
        "object_start_pose_xyzw": fixed_goals["start_pose"],
        "goal_poses_xyzw": fixed_goals["goals"],
        "goal_source": "SimToolReal native useFixedGoalStates/fixedGoalStates",
        "object_start_source": "SimToolReal native useFixedInitObjectPose/objectStartPose",
        "extra_actors": benchmark_extra_actors,
        "visual_overrides": [],
        "table_asset": resolved_table_asset or "SimToolReal config default",
        "table_asset_source": (
            "state_bundle SimToolReal task table asset"
            if resolved_table_asset
            else "SimToolReal config default"
        ),
        "table_reset_z": (
            resolved_table_pose_z
            if resolved_table_pose_z is not None
            else "SimToolReal config default"
        ),
        "table_clearance_check": table_clearance_check,
        "goal_index_map": goal_index_map,
        "start_goal_in_fixed_goals": bool(
            fixed_goals.get("metadata", {}).get("start_goal_in_fixed_goals")
        )
        if isinstance(fixed_goals.get("metadata"), dict)
        else bool(
            goal_index_map
            and goal_index_map[0].get("source_goal_index") is None
        ),
        "functional_verification": {
            "checked": bool(functional_specs),
            "num_goals_with_contact_point_goal": len(functional_specs),
            "max_functional_target_error_m": float(max_functional_target_error_m),
            "scene_graph_source": (
                str(resolved_observed_scene_graph_path)
                if resolved_observed_scene_graph_path
                else None
            ),
            "scene_graph_source_auto_discovered": bool(
                resolved_observed_scene_graph_path and not observed_scene_graph_path
            ),
            "uses_simtoolreal_actors": False,
            "uses_visual_overrides": False,
        },
        "success_gate": {
            "eval_success_tolerance_m": resolved_eval_success_tolerance_m,
            "success_steps": resolved_success_steps,
            "force_consecutive_near_goal_steps": bool(force_consecutive_near_goal_steps),
            "use_tool_frame_orientation_success": bool(use_tool_frame_orientation_success),
            "tool_frame_orientation_success_tolerance_rad": (
                float(tool_frame_orientation_success_tolerance_rad)
                if tool_frame_orientation_success_tolerance_rad is not None
                else None
            ),
            "tool_frame_orientation_success_stage_families": list(orientation_stage_families),
            "initial_grasp_hold_steps": max(0, int(initial_grasp_hold_steps)),
            "source": success_tolerance_report,
        },
        "grasp_observation_bias": {
            "enabled": bool(float(grasp_observation_offset_m) > 0.0),
            "offset_m": float(grasp_observation_offset_m),
            "simtoolreal_config": "use_hack_object_pos_offset / hack_object_pos_offset",
            "meaning": (
                "Bias the low-level policy's observed tool reference toward "
                "the tool-local -x side, which is the grasp/handle side for "
                "the current SimToolReal handle-head convention. This does "
                "not modify the compiled whole-tool goals."
            ),
        },
        "camera": _simtoolreal_original_camera_spec(),
        "metadata": {
            "generated_by": "parttoolgen.execution.simtoolreal_rollout",
            "policy_modified": False,
            "observation_action_space_modified": bool(float(grasp_observation_offset_m) > 0.0),
            "simtoolreal_actor_visualization_modified": bool(benchmark_extra_actors),
            "start_pose_inserted_by_rollout": False,
        },
    }

    _install_numpy_legacy_aliases()
    _install_pytorch3d_transform_fallback()
    # Import SimToolReal/IsaacGym before torch; IsaacGym enforces this order.
    from deployment.isaac.isaac_env import create_env  # noqa: WPS433
    from deployment.isaac.isaac_env_no_ros import (  # noqa: WPS433
        IsaacEnvNoRos,
        N_ACT,
        N_OBS,
        compute_observation,
    )
    from deployment.rl_player import RlPlayer  # noqa: WPS433
    from isaacgymenvs.utils.observation_action_utils_sharpa import create_urdf_object  # noqa: WPS433
    import torch  # noqa: WPS433

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _install_cpu_checkpoint_loader(resolved_device)
    can_capture_video, capture_video_reason = _can_capture_camera_frames(
        device=resolved_device,
        headless=headless,
    )
    if capture_video and not can_capture_video:
        raise RuntimeError(capture_video_reason)
    control_dt = 1.0 / float(control_hz)
    overrides = {
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.objectName": str(resolved_object_name),
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.useFixedGoalStates": True,
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.resetWhenDropped": False,
        "task.env.armMovingAverage": 0.1,
        "task.env.evalSuccessTolerance": resolved_eval_success_tolerance_m,
        "task.env.successSteps": resolved_success_steps,
        "task.env.forceConsecutiveNearGoalSteps": bool(force_consecutive_near_goal_steps),
        "task.env.useToolFrameOrientationSuccess": bool(use_tool_frame_orientation_success),
        "task.env.toolFrameOrientationSuccessStageFamilies": list(orientation_stage_families),
        "task.env.fixedSizeKeypointReward": True,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": fixed_goals["start_pose"],
        "task.env.startArmHigher": False,
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.forceOnlyWhenLifted": True,
        "task.env.torqueOnlyWhenLifted": True,
        "task.env.linVelImpulseOnlyWhenLifted": True,
        "task.env.angVelImpulseOnlyWhenLifted": True,
        "task.env.forceProbRange": [0.0001, 0.0001],
        "task.env.torqueProbRange": [0.0001, 0.0001],
        "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.forceNoReset": bool(force_no_reset),
        "task.env.use_hack_object_pos_offset": bool(float(grasp_observation_offset_m) > 0.0),
        "task.env.hack_object_pos_offset": float(grasp_observation_offset_m),
        "task.env.hideGoalObjectVisual": bool(hide_goal_object_visual),
        "task.env.goalObjectVisualColor": [0.0, 1.0, 0.0],
        "task.env.goalObjectVisualOpacity": float(max(0.0, min(1.0, goal_object_visual_opacity))),
        "task.env.goalObjectVisualBlendColor": [1.0, 1.0, 1.0],
    }
    if use_tool_frame_orientation_success and orientation_stage_families:
        overrides["task.env.fixedGoalStatesJsonPath"] = str(fixed_goals_path)
    else:
        overrides["task.env.fixedGoalStates"] = fixed_goals["goals"]
    if benchmark_extra_actors:
        overrides["task.env.parttoolgenBenchmarkExtraActors"] = benchmark_extra_actors
    if tool_frame_orientation_success_tolerance_rad is not None:
        overrides["task.env.toolFrameOrientationSuccessToleranceRad"] = float(
            tool_frame_orientation_success_tolerance_rad
        )
    if resolved_table_asset:
        overrides["task.env.asset.table"] = str(resolved_table_asset)
    if resolved_table_pose_z is not None:
        overrides["task.env.tableResetZ"] = resolved_table_pose_z
    env = create_env(
        config_path=str(config_path),
        headless=headless,
        device=resolved_device,
        episode_length=max_steps + 5,
        overrides=overrides,
    )
    lighting = apply_simtoolreal_render_lighting(
        env,
        light_preset,
        soft_light_indices=soft_light_indices,
    )
    scene_spec_artifact["lighting"] = lighting
    scene_spec_artifact["metadata"]["render_lighting_preset"] = lighting.get(
        "name", lighting["preset"]
    )
    _write_json(output_dir / "scene_spec.json", scene_spec_artifact)
    checkpoint = torch.load(str(checkpoint_path), map_location=resolved_device)
    env_state = checkpoint[0].get("env_state") if isinstance(checkpoint, list) else None
    if env_state is not None:
        env.set_env_state(env_state)
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=resolved_device,
        num_envs=env.num_envs,
    )
    urdf = create_urdf_object(robot_name="iiwa14_left_sharpa_adjusted_restricted")
    isaac_env = IsaacEnvNoRos(
        env=env,
        control_dt=control_dt,
        device=resolved_device,
        urdf=urdf,
    )
    observation = isaac_env.reset()
    benchmark_extra_actor_initial = _benchmark_extra_actor_states(env, benchmark_extra_actors)
    benchmark_extra_actor_samples: list[dict[str, Any]] = []
    topdown_camera = _create_rollout_topdown_camera(env) if capture_video and capture_topdown_video else None
    side_board_camera = (
        _create_side_board_front_camera(env)
        if capture_video and capture_side_board_video
        else None
    )

    tracking_samples: list[dict[str, Any]] = []
    contact_events: list[dict[str, Any]] = []
    collision_events: list[dict[str, Any]] = []
    drop_events: list[dict[str, Any]] = []
    grasp_hold_samples: list[dict[str, Any]] = []
    video_frames: list[Any] = []
    topdown_video_frames: list[Any] = []
    side_board_video_frames: list[Any] = []

    def _force_fixed_goal_index(goal_index: int) -> None:
        env.successes[0] = int(goal_index)
        if hasattr(env, "near_goal_steps"):
            env.near_goal_steps[0] = 0
        if hasattr(env, "reset_goal_buf"):
            env.reset_goal_buf[0] = 0
        env._reset_target(
            torch.tensor([0], device=env.device),
            reset_buf_idxs=None,
            tensor_reset=True,
        )

    def _current_observation() -> Any:
        object_pose = env.object_pose.clone()
        goal_object_pose = env.goal_pose.clone()
        object_pose[:, 0:3] = env.object_pos
        goal_object_pose[:, 0:3] = env.goal_pos
        obs = compute_observation(
            q=env.arm_hand_dof_pos.cpu().numpy(),
            qd=env.arm_hand_dof_vel.cpu().numpy(),
            prev_action_targets=env.prev_targets.cpu().numpy(),
            object_pose=object_pose.cpu().numpy(),
            goal_object_pose=goal_object_pose.cpu().numpy(),
            object_scales=env.object_scales.cpu().numpy(),
            urdf=urdf,
            obs_list=env.obs_list,
        )
        return torch.from_numpy(obs).float().to(resolved_device)

    resolved_grasp_hold_steps = max(0, int(initial_grasp_hold_steps))
    if resolved_grasp_hold_steps > 0:
        _force_fixed_goal_index(0)
        observation = _current_observation()
        hold_stage = _stage_for_goal(tool_pose_subgoals, 0, goal_index_map)
        for hold_step in range(resolved_grasp_hold_steps):
            action = policy.get_normalized_action(observation, deterministic_actions=True)
            observation, _, _, info = isaac_env.step(action)
            actual_pose = _torch_pose_to_list(env.object_pose[0])
            functional_observation = _functional_observation_from_pose(
                actual_pose=actual_pose,
                spec=functional_specs.get(0),
                max_functional_target_error_m=max_functional_target_error_m,
            )
            grasp_hold_samples.append(
                _record_sample(
                    step=hold_step,
                    goal_index=0,
                    stage=hold_stage,
                    actual_pose=actual_pose,
                    goal_pose=fixed_goals["goals"][0],
                    successes=int(env.successes[0].detach().cpu().item()),
                    lifted=bool(env.lifted_object[0].detach().cpu().item())
                    if hasattr(env, "lifted_object")
                    else None,
                    closest_fingertip_dist=None,
                    closest_fingertip_distances=None,
                    functional_observation=functional_observation,
                )
            )
            if _event_from_info(info, "dropped"):
                drop_events.append(
                    {
                        "step": -resolved_grasp_hold_steps + hold_step,
                        "goal_index": 0,
                        "stage": hold_stage,
                        "phase": "initial_grasp_hold",
                    }
                )
            if _event_from_info(info, "table_force_too_high"):
                collision_events.append(
                    {
                        "step": -resolved_grasp_hold_steps + hold_step,
                        "goal_index": 0,
                        "stage": hold_stage,
                        "phase": "initial_grasp_hold",
                        "type": "table_force_too_high",
                    }
                )
            if capture_video and hold_step % max(1, int(video_stride)) == 0:
                frame = _capture_camera_frame(env)
                if frame is not None:
                    video_frames.append(frame)
                if topdown_camera is not None:
                    topdown_frame = _capture_camera_sensor_frame(
                        env,
                        camera_handle=topdown_camera[0],
                        camera_properties=topdown_camera[1],
                    )
                    if topdown_frame is not None:
                        topdown_video_frames.append(topdown_frame)
                if side_board_camera is not None:
                    side_board_frame = _capture_camera_sensor_frame(
                        env,
                        camera_handle=side_board_camera[0],
                        camera_properties=side_board_camera[1],
                    )
                    if side_board_frame is not None:
                        side_board_video_frames.append(side_board_frame)
            _force_fixed_goal_index(0)
            observation = _current_observation()

    initial_successes = min(
        int(env.successes[0].detach().cpu().item()),
        len(fixed_goals["goals"]),
    )
    reached_goal_count = initial_successes
    final_successes = initial_successes
    resolved_post_success_steps = max(0, int(post_success_steps))
    post_success_remaining: int | None = None
    for initial_goal_index in range(initial_successes):
        initial_stage = _stage_for_goal(
            tool_pose_subgoals,
            initial_goal_index,
            goal_index_map,
        )
        contact_events.append(
            {
                "step": -1,
                "goal_index": int(initial_goal_index),
                "stage": initial_stage,
                "type": "simtoolreal_goal_success",
                "source": "fixed_policy_success_signal",
                "note": (
                    "Goal was already satisfied immediately after reset by the "
                    "unchanged SimToolReal policy success counter."
                ),
            }
        )
    for step in range(int(max_steps)):
        successes_before = int(env.successes[0].detach().cpu().item())
        goal_index = min(successes_before, len(fixed_goals["goals"]) - 1)
        stage = _stage_for_goal(tool_pose_subgoals, goal_index, goal_index_map)
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, _, info = isaac_env.step(action)
        final_successes = int(env.successes[0].detach().cpu().item())
        actual_pose = _torch_pose_to_list(env.object_pose[0])
        lifted = bool(env.lifted_object[0].detach().cpu().item()) if hasattr(env, "lifted_object") else None
        closest = None
        closest_values = None
        if hasattr(env, "closest_fingertip_dist"):
            closest_tensor = env.closest_fingertip_dist[0].detach().cpu().reshape(-1)
            closest_values = [float(item) for item in closest_tensor.tolist()]
            valid_closest = [item for item in closest_values if item >= 0.0]
            if valid_closest:
                closest = min(valid_closest)
        functional_observation = _functional_observation_from_pose(
            actual_pose=actual_pose,
            spec=functional_specs.get(goal_index),
            max_functional_target_error_m=max_functional_target_error_m,
        )
        tracking_samples.append(
            _record_sample(
                step=step,
                goal_index=goal_index,
                stage=stage,
                actual_pose=actual_pose,
                goal_pose=fixed_goals["goals"][goal_index],
                successes=final_successes,
                lifted=lifted,
                closest_fingertip_dist=closest,
                closest_fingertip_distances=closest_values,
                functional_observation=functional_observation,
            )
        )
        if final_successes > successes_before:
            reached_goal_count = final_successes
            contact_events.append(
                {
                    "step": int(step),
                    "goal_index": int(goal_index),
                    "stage": stage,
                    "type": "simtoolreal_goal_success",
                    "source": "fixed_policy_success_signal",
                    "note": "Goal tolerance reached by unchanged SimToolReal policy; not a raw contact sensor.",
                }
            )
        if _event_from_info(info, "dropped"):
            drop_events.append({"step": int(step), "goal_index": int(goal_index), "stage": stage})
        if _event_from_info(info, "table_force_too_high"):
            collision_events.append(
                {
                    "step": int(step),
                    "goal_index": int(goal_index),
                    "stage": stage,
                    "type": "table_force_too_high",
                }
            )
        if benchmark_extra_actors:
            benchmark_extra_actor_samples.append(
                {
                    "step": int(step),
                    "goal_index": int(goal_index),
                    "stage": stage,
                    "actors": _benchmark_extra_actor_states(env, benchmark_extra_actors),
                }
            )
        if capture_video and step % max(1, int(video_stride)) == 0:
            frame = _capture_camera_frame(env)
            if frame is not None:
                video_frames.append(frame)
            if topdown_camera is not None:
                topdown_frame = _capture_camera_sensor_frame(
                    env,
                    camera_handle=topdown_camera[0],
                    camera_properties=topdown_camera[1],
                )
                if topdown_frame is not None:
                    topdown_video_frames.append(topdown_frame)
            if side_board_camera is not None:
                side_board_frame = _capture_camera_sensor_frame(
                    env,
                    camera_handle=side_board_camera[0],
                    camera_properties=side_board_camera[1],
                )
                if side_board_frame is not None:
                    side_board_video_frames.append(side_board_frame)
        if final_successes >= len(fixed_goals["goals"]):
            if post_success_remaining is None:
                post_success_remaining = resolved_post_success_steps
                contact_events.append(
                    {
                        "step": int(step),
                        "goal_index": int(goal_index),
                        "stage": stage,
                        "type": "post_success_settle_start",
                        "source": "rollout_post_success_steps",
                        "post_success_steps": resolved_post_success_steps,
                    }
                )
            if post_success_remaining <= 0:
                break
            post_success_remaining -= 1
        sleep_time = control_dt - (time.time() - start_time)
        if sleep_time > 0 and not headless:
            time.sleep(sleep_time)

    _write_json(output_dir / "tracking_samples.json", tracking_samples)
    _write_json(output_dir / "initial_grasp_hold_samples.json", grasp_hold_samples)
    _write_json(output_dir / "contact_events.json", contact_events)
    _write_json(output_dir / "collision_events.json", collision_events)
    _write_json(output_dir / "drop_events.json", drop_events)
    benchmark_extra_actor_final = _benchmark_extra_actor_states(env, benchmark_extra_actors)
    benchmark_extra_actor_summary = _benchmark_extra_actor_motion_summary(
        initial=benchmark_extra_actor_initial,
        final=benchmark_extra_actor_final,
    )
    if benchmark_extra_actors:
        _write_json(
            output_dir / "benchmark_extra_actor_samples.json",
            benchmark_extra_actor_samples,
        )
        _write_json(
            output_dir / "benchmark_extra_actor_summary.json",
            benchmark_extra_actor_summary,
        )
    video_path = output_dir / "rollout_render.mp4"
    topdown_video_path = output_dir / "rollout_topdown_render.mp4"
    side_board_video_path = output_dir / "rollout_side_board_front_render.mp4"
    if capture_video:
        _write_rollout_video(video_path, video_frames, fps=video_fps)
        if topdown_video_frames:
            _write_rollout_video(topdown_video_path, topdown_video_frames, fps=video_fps)
        if side_board_video_frames:
            _write_rollout_video(side_board_video_path, side_board_video_frames, fps=video_fps)
    execution_report = build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=tool_pose_subgoals,
        tracking_samples=tracking_samples,
        contact_events=contact_events,
        collision_events=collision_events,
        drop_events=drop_events,
        max_tracking_error_m=max_tracking_error_m,
        max_rotation_error_rad=max_rotation_error_rad,
        max_functional_target_error_m=max_functional_target_error_m,
        require_contact=require_contact,
        goal_index_map=goal_index_map,
    )
    tracking_execution_passed = bool(execution_report.get("tracking_execution_passed"))
    semantic_execution_passed = bool(execution_report.get("semantic_execution_passed"))
    policy_goal_passed = final_successes >= len(fixed_goals["goals"])
    tracking_error_summary = _stage_error_summary(execution_report.get("stage_reports", []))
    rollout_summary = {
        "passed": policy_goal_passed and tracking_execution_passed and semantic_execution_passed,
        "policy_goal_passed": policy_goal_passed,
        "tracking_execution_passed": tracking_execution_passed,
        "semantic_execution_passed": semantic_execution_passed,
        "final_successes": final_successes,
        "num_goals": len(fixed_goals["goals"]),
        "start_goal_in_fixed_goals": bool(
            fixed_goals.get("metadata", {}).get("start_goal_in_fixed_goals")
        )
        if isinstance(fixed_goals.get("metadata"), dict)
        else bool(
            goal_index_map
            and goal_index_map[0].get("source_goal_index") is None
        ),
        "reached_goal_count": reached_goal_count,
        "num_steps": len(tracking_samples),
        "device": resolved_device,
        "headless": headless,
        "eval_success_tolerance_m": resolved_eval_success_tolerance_m,
        "success_tolerance_source": success_tolerance_report,
        "success_steps": resolved_success_steps,
        "post_success_steps": resolved_post_success_steps,
        "initial_grasp_hold_steps": resolved_grasp_hold_steps,
        "initial_grasp_hold_samples": len(grasp_hold_samples),
        "force_consecutive_near_goal_steps": bool(force_consecutive_near_goal_steps),
        "max_tracking_error_m": tracking_error_summary["max_tracking_error_m"],
        "max_rotation_error_rad": tracking_error_summary["max_rotation_error_rad"],
        "tracking_error_summary": tracking_error_summary,
        "max_functional_target_error_m": float(max_functional_target_error_m),
        "object_name": resolved_object_name,
        "render_video_path": str(video_path) if video_path.exists() else None,
        "topdown_render_video_path": str(topdown_video_path) if topdown_video_path.exists() else None,
        "side_board_front_render_video_path": (
            str(side_board_video_path) if side_board_video_path.exists() else None
        ),
        "render_video_frames": len(video_frames),
        "topdown_render_video_frames": len(topdown_video_frames),
        "side_board_front_render_video_frames": len(side_board_video_frames),
        "benchmark_extra_actor_summary": benchmark_extra_actor_summary,
        "grasp_observation_bias": scene_spec_artifact["grasp_observation_bias"],
        "camera": scene_spec_artifact["camera"],
        "lighting": scene_spec_artifact["lighting"],
        "metadata": {
            "generated_by": "parttoolgen.execution.simtoolreal_rollout",
            "policy_modified": False,
            "observation_action_space_modified": bool(float(grasp_observation_offset_m) > 0.0),
            "simtoolreal_actor_visualization_modified": bool(benchmark_extra_actors),
            "goal_object_visual_hidden": bool(hide_goal_object_visual),
            "goal_object_visual_opacity": float(goal_object_visual_opacity),
            "render_lighting_preset": lighting.get("name", lighting["preset"]),
            "start_pose_inserted_by_rollout": False,
        },
    }
    _write_json(output_dir / "rollout_summary.json", rollout_summary)
    _write_json(output_dir / "execution_report.json", execution_report)
    return {
        "rollout_summary": rollout_summary,
        "execution_report": execution_report,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-goals", required=True)
    parser.add_argument(
        "--scene",
        help="Optional legacy input accepted for compatibility; it is not used to create SimToolReal actors.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subgoals")
    parser.add_argument("--state-bundle")
    parser.add_argument("--tool-mesh")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--object-name")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--device")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--require-contact", action="store_true")
    parser.add_argument("--max-tracking-error-m", type=float, default=0.035)
    parser.add_argument("--max-rotation-error-rad", type=float, default=0.30)
    parser.add_argument(
        "--max-functional-target-error-m",
        type=float,
        default=DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M,
        help=(
            "Maximum allowed distance between the executed functional point "
            "and the expected scene target when subgoals provide functional "
            "contact metadata."
        ),
    )
    parser.add_argument("--eval-success-tolerance-m", type=float)
    parser.add_argument(
        "--success-steps",
        type=int,
        default=DEFAULT_SUCCESS_STEPS,
        help=(
            "Number of consecutive near-goal frames required before SimToolReal "
            "advances to the next fixed goal."
        ),
    )
    parser.add_argument(
        "--allow-cumulative-near-goal-steps",
        action="store_true",
        help=(
            "Debug only: keep SimToolReal's cumulative near-goal step counter "
            "instead of requiring consecutive near-goal frames per fixed goal."
        ),
    )
    parser.add_argument(
        "--use-tool-frame-orientation-success",
        action="store_true",
        help=(
            "Require SimToolReal's tool-frame orientation gate before advancing "
            "fixed goals. This uses public fixed-goal orientation and does not "
            "change the compiled PartToolGen subgoals."
        ),
    )
    parser.add_argument(
        "--tool-frame-orientation-success-tolerance-rad",
        type=float,
        help=(
            "Optional SimToolReal orientation-success tolerance in radians. "
            "Only used when --use-tool-frame-orientation-success is enabled."
        ),
    )
    parser.add_argument(
        "--tool-frame-orientation-success-stage-family",
        action="append",
        default=[],
        help=(
            "Restrict orientation-success gating to fixed-goal stages whose "
            "metadata stage name starts with this prefix. Can be repeated; "
            "omit to gate every fixed goal."
        ),
    )
    parser.add_argument(
        "--simtoolreal-keypoint-scale",
        type=float,
        default=DEFAULT_SIMTOOLREAL_KEYPOINT_SCALE,
        help="SimToolReal keypointScale used to derive default eval success tolerance.",
    )
    parser.add_argument(
        "--success-tolerance-safety-factor",
        type=float,
        default=DEFAULT_SUCCESS_TOLERANCE_SAFETY_FACTOR,
        help=(
            "Multiplier applied when deriving eval success tolerance from the "
            "PartToolGen functional target threshold."
        ),
    )
    parser.add_argument(
        "--force-no-reset",
        action="store_true",
        help=(
            "Disable SimToolReal episode resets during long fixed-goal "
            "inspection rollouts. Goal advancement still uses the normal "
            "fixed-goal success counter."
        ),
    )
    parser.add_argument(
        "--grasp-observation-offset-m",
        type=float,
        default=0.0,
        help=(
            "Bias the low-level SimToolReal policy's observed tool reference "
            "toward the tool-local -x grasp/handle side before and during "
            "tracking. This is a lightweight grasp-side hint for inspection "
            "rollouts; it does not alter PartToolGen's compiled tool goals."
        ),
    )
    parser.add_argument(
        "--initial-grasp-hold-steps",
        type=int,
        default=0,
        help=(
            "Run this many policy steps on the first fixed goal while keeping "
            "SimToolReal's success counter at goal 0. This lets the hand settle "
            "into the grasp before semantic subgoals begin, without changing "
            "the compiled tool pose."
        ),
    )
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=0,
        help=(
            "Continue stepping after the final fixed-goal success counter is "
            "reached. This is useful for dynamic-effect tasks whose measured "
            "target motion needs a short physics settle window."
        ),
    )
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument(
        "--light-preset",
        choices=("soft", "native"),
        default=DEFAULT_RENDER_LIGHT_PRESET,
        help=(
            "Render lighting for rollout evidence. The default soft preset "
            "softens IsaacGym light indices 0 and 1."
        ),
    )
    parser.add_argument(
        "--soft-light-indices",
        type=_parse_int_csv,
        default=DEFAULT_SOFT_LIGHT_INDICES,
        help=(
            "Comma-separated IsaacGym light indices used by --light-preset soft. "
            "Side-board rollout evidence should use 0,2 to match the canonical soft02 view."
        ),
    )
    parser.add_argument(
        "--hide-goal-object-visual",
        action="store_true",
        help="Hide SimToolReal's green goal-object/subgoal visual in rollout videos.",
    )
    parser.add_argument(
        "--goal-object-visual-opacity",
        type=float,
        default=0.32,
        help="Blend opacity for optional visible goal-object debugging geometry.",
    )
    parser.add_argument(
        "--capture-topdown-video",
        action="store_true",
        help="Also capture a robot-forward top-down rollout video.",
    )
    parser.add_argument(
        "--capture-side-board-video",
        action="store_true",
        help=(
            "Also capture an isolated front-facing side-board rollout video. "
            "This is for vertical whiteboard/side tasks and does not replace tabletop views."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-stride", type=int, default=2)
    args = parser.parse_args()
    result = run_fixed_policy_rollout(
        fixed_goals_path=args.fixed_goals,
        output_dir=args.output_dir,
        observed_scene_graph_path=args.scene,
        subgoals_path=args.subgoals,
        state_bundle_path=args.state_bundle,
        tool_mesh_path=args.tool_mesh,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        object_name=args.object_name,
        max_steps=args.max_steps,
        control_hz=args.control_hz,
        device=args.device,
        headless=args.headless,
        require_contact=args.require_contact,
        max_tracking_error_m=args.max_tracking_error_m,
        max_rotation_error_rad=args.max_rotation_error_rad,
        max_functional_target_error_m=args.max_functional_target_error_m,
        eval_success_tolerance_m=args.eval_success_tolerance_m,
        success_steps=args.success_steps,
        force_consecutive_near_goal_steps=not args.allow_cumulative_near_goal_steps,
        use_tool_frame_orientation_success=args.use_tool_frame_orientation_success,
        tool_frame_orientation_success_tolerance_rad=(
            args.tool_frame_orientation_success_tolerance_rad
        ),
        tool_frame_orientation_success_stage_families=(
            args.tool_frame_orientation_success_stage_family
        ),
        simtoolreal_keypoint_scale=args.simtoolreal_keypoint_scale,
        success_tolerance_safety_factor=args.success_tolerance_safety_factor,
        force_no_reset=args.force_no_reset,
        grasp_observation_offset_m=args.grasp_observation_offset_m,
        initial_grasp_hold_steps=args.initial_grasp_hold_steps,
        post_success_steps=args.post_success_steps,
        hide_goal_object_visual=args.hide_goal_object_visual,
        goal_object_visual_opacity=args.goal_object_visual_opacity,
        capture_video=args.capture_video,
        capture_topdown_video=args.capture_topdown_video,
        capture_side_board_video=args.capture_side_board_video,
        video_fps=args.video_fps,
        video_stride=args.video_stride,
        light_preset=args.light_preset,
        soft_light_indices=args.soft_light_indices,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
