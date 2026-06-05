#!/usr/bin/env python3
"""Convert PartToolGen whole-tool subgoals into SimToolReal fixed-goal JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parttoolgen.validation.validate_parttoolgen import (
    load_structured_file,
    validate_simtoolreal_fixed_goals,
    validate_tool_pose_subgoals,
)


class SimToolRealAdapterError(ValueError):
    """Raised when ToolPoseSubgoals are not ready for SimToolReal conversion."""


def _pose_to_list(pose: dict[str, Any]) -> list[float]:
    return [*map(float, pose["position"]), *map(float, pose["orientation_xyzw"])]


def _quat_dot_abs(a: list[float], b: list[float]) -> float:
    return abs(sum(float(a[i]) * float(b[i]) for i in range(4)))


def _pose7_is_redundant_with_start(
    pose: list[float],
    start_pose: list[float],
    *,
    max_translation_m: float = 0.005,
    max_rotation_dot_error: float = 1e-4,
) -> bool:
    if len(pose) != 7 or len(start_pose) != 7:
        return False
    translation_sq = sum((float(pose[i]) - float(start_pose[i])) ** 2 for i in range(3))
    if translation_sq > float(max_translation_m) ** 2:
        return False
    quat_dot = _quat_dot_abs(pose[3:7], start_pose[3:7])
    return (1.0 - min(1.0, quat_dot)) <= float(max_rotation_dot_error)


def _deferred_path_objective_count(data: dict[str, Any]) -> int:
    sanity = data.get("sanity_checks", {})
    if not isinstance(sanity, dict):
        return 0
    raw = sanity.get("path_objectives_deferred", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _semantic_role(goal: dict[str, Any]) -> str:
    role = goal.get("semantic_role")
    if isinstance(role, str):
        return role
    source_constraints = goal.get("source_constraints")
    if isinstance(source_constraints, list):
        for constraint in source_constraints:
            if isinstance(constraint, str) and constraint.startswith("source_semantic_role:"):
                return constraint.split(":", 1)[1]
    return ""


def _is_generated_nonsemantic_waypoint(goal: dict[str, Any]) -> bool:
    stage = str(goal.get("stage", "")).lower()
    if "_transition_" in stage:
        return True
    return goal.get("semantic_role") == "generated_transition_waypoint"


def _is_task_terminal_goal(goal: dict[str, Any]) -> bool:
    if _is_generated_nonsemantic_waypoint(goal):
        return False
    if goal.get("semantic_gate") is not True:
        return False
    semantic_role = _semantic_role(goal).lower()
    return "terminal" in semantic_role


def _task_terminal_source_index(goals: list[Any]) -> int | None:
    for source_index, waypoint in enumerate(goals):
        if isinstance(waypoint, dict) and _is_task_terminal_goal(waypoint):
            return source_index
    return None


def _require_tool_path_solver_for_deferred_objectives(data: dict[str, Any]) -> None:
    sanity = data.get("sanity_checks", {})
    solver = sanity.get("tool_path_solver") if isinstance(sanity, dict) else None
    if isinstance(solver, dict) and solver.get("semantic_goal_preserved") is not True:
        raise SimToolRealAdapterError(
            "ToolPoseSubgoals tool_path_solver did not report semantic_goal_preserved=true; "
            "repair ProgramIR/refs before SimToolReal conversion"
        )
    deferred = _deferred_path_objective_count(data)
    if deferred <= 0:
        return
    if not isinstance(solver, dict):
        raise SimToolRealAdapterError(
            "ToolPoseSubgoals has deferred path objectives but no tool_path_solver gate report"
        )
    required = {
        "passed": True,
        "path_objectives_checked": True,
        "path_objectives_passed": True,
        "semantic_goal_preserved": True,
    }
    for key, expected in required.items():
        if solver.get(key) is not expected:
            raise SimToolRealAdapterError(
                "ToolPoseSubgoals deferred path objectives were not cleared by "
                f"tool_path_solver ({key}={solver.get(key)!r})"
            )


def convert_subgoals_to_simtoolreal(data: dict[str, Any]) -> dict[str, Any]:
    validate_tool_pose_subgoals(data)
    _require_tool_path_solver_for_deferred_objectives(data)
    start_goal = _pose_to_list(data["start_pose"])
    source_goals = data["goals"]
    terminal_source_index = _task_terminal_source_index(source_goals)
    exported_source_goals = source_goals
    truncated_post_terminal_goals: list[dict[str, Any]] = []
    converted_goals: list[list[float]] = []
    goal_index_map: list[dict[str, Any]] = []
    skipped_initial_goals: list[dict[str, Any]] = []
    for source_index, waypoint in enumerate(exported_source_goals):
        if not isinstance(waypoint, dict):
            continue
        pose = [
            *map(float, waypoint["pose_world_tool"]["position"]),
            *map(float, waypoint["pose_world_tool"]["orientation_xyzw"]),
        ]
        force_export = _semantic_role(waypoint) == "execution_start_settle"
        if (
            not force_export
            and not converted_goals
            and _pose7_is_redundant_with_start(pose, start_goal)
        ):
            skipped_initial_goals.append(
                {
                    "source_goal_index": source_index,
                    "stage": str(waypoint.get("stage", f"source_goal_{source_index}")),
                    "reason": "already_satisfied_by_objectStartPose",
                }
            )
            continue
        fixed_goal_index = len(converted_goals)
        converted_goals.append(pose)
        goal_index_map.append(
            {
                "fixed_goal_index": fixed_goal_index,
                "source_goal_index": source_index,
                "stage": str(waypoint.get("stage", f"source_goal_{source_index}")),
                "source": "tool_pose_subgoals.goals",
            }
        )
    if not converted_goals:
        raise SimToolRealAdapterError("ToolPoseSubgoals contained no non-initial fixed goals")
    return {
        "start_pose": start_goal,
        "goals": converted_goals,
        "metadata": {
            "schema_version": "v0.1",
            "source_subgoals_id": data["subgoals_id"],
            "target_api": "fixed_goal_states_json",
            "coordinate_frame": "world",
            "units": "meters",
            "quaternion_order": "xyzw",
            "generated_by": "parttoolgen_simtoolreal_adapter",
            "goal_index_map": goal_index_map,
            "start_goal_in_fixed_goals": bool(
                converted_goals
                and _pose7_is_redundant_with_start(
                    converted_goals[0],
                    start_goal,
                    max_translation_m=1e-6,
                    max_rotation_dot_error=1e-8,
                )
            ),
            "skipped_initial_goals": skipped_initial_goals,
            "task_terminal_source_goal_index": terminal_source_index,
            "task_terminal_stage": str(source_goals[terminal_source_index].get("stage"))
            if terminal_source_index is not None
            and isinstance(source_goals[terminal_source_index], dict)
            else None,
            "truncated_post_terminal_goals": truncated_post_terminal_goals,
        },
    }


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    data = load_structured_file(input_path)
    converted = convert_subgoals_to_simtoolreal(data)
    validate_simtoolreal_fixed_goals(converted)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2) + "\n", encoding="utf-8")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_file(
        args.input,
        args.output,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
