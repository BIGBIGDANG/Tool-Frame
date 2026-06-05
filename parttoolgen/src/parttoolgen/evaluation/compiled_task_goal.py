"""Compile-level task-goal checks for image-grounded ToolFrame tasks.

These checks answer a narrow question before rollout: if the robot reached the
compiler's semantic terminal subgoal exactly, would the planned tool contact
point satisfy the task contract?
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _vector(values: Any) -> list[float] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) < 3:
        return None
    try:
        return [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return None


def _unit(values: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(sum(float(item) * float(item) for item in values[:3]))
    if norm <= 1e-12:
        return None
    return [float(item) / norm for item in values[:3]]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[index]) - float(b[index]) for index in range(3)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in values[:3]))


def _predicate(
    name: str,
    passed: bool,
    *,
    value: Any = None,
    threshold: Any = None,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": bool(required),
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


def _terminal_semantic_contact_goal(subgoals: Mapping[str, Any]) -> dict[str, Any] | None:
    goals = subgoals.get("goals")
    if not isinstance(goals, list):
        return None
    semantic_goals = [
        goal
        for goal in goals
        if isinstance(goal, Mapping)
        and bool(goal.get("semantic_gate"))
        and isinstance(goal.get("contact_point_goal"), Mapping)
    ]
    if semantic_goals:
        return dict(semantic_goals[-1])
    action_goals = [
        goal
        for goal in goals
        if isinstance(goal, Mapping)
        and str(goal.get("stage") or "").startswith("action")
        and isinstance(goal.get("contact_point_goal"), Mapping)
    ]
    return dict(action_goals[-1]) if action_goals else None


def check_screwdriver_insert_compiled_task_goal(
    *,
    subgoals: Mapping[str, Any] | None,
    lifted_refs: Mapping[str, Any] | None,
    max_tip_radial_error_m: float,
    min_tip_inside_depth_m: float,
) -> dict[str, Any]:
    """Check whether compiled terminal subgoals place the tip inside the socket."""

    subgoals = subgoals if isinstance(subgoals, Mapping) else {}
    lifted_refs = lifted_refs if isinstance(lifted_refs, Mapping) else {}
    terminal_goal = _terminal_semantic_contact_goal(subgoals)
    terminal_contact = (
        _vector((terminal_goal or {}).get("contact_point_goal", {}).get("position_world"))
        if isinstance(terminal_goal, Mapping)
        else None
    )
    start_ref = lifted_refs.get("start_point") or lifted_refs.get("start")
    socket_center = (
        _vector(start_ref.get("position_world"))
        if isinstance(start_ref, Mapping)
        else None
    )
    axis = _unit(_vector(lifted_refs.get("direction")) or [])
    terminal_goal_found = terminal_contact is not None
    geometry_available = socket_center is not None and axis is not None

    radial_error = None
    inside_depth = None
    if terminal_contact is not None and socket_center is not None and axis is not None:
        offset = _sub(terminal_contact, socket_center)
        inside_depth = _dot(offset, axis)
        radial_component = [
            offset[index] - inside_depth * axis[index]
            for index in range(3)
        ]
        radial_error = _norm(radial_component)

    radial_passed = radial_error is not None and radial_error <= float(max_tip_radial_error_m)
    depth_passed = inside_depth is not None and inside_depth >= float(min_tip_inside_depth_m)
    predicates = [
        _predicate(
            "compiled_terminal_contact_goal_present",
            terminal_goal_found,
            value=(terminal_goal or {}).get("stage") if isinstance(terminal_goal, Mapping) else None,
        ),
        _predicate(
            "compiled_socket_geometry_available",
            geometry_available,
            value={
                "socket_center_world": socket_center,
                "insertion_axis_world": axis,
            },
        ),
        _predicate(
            "compiled_tip_radial_inside_socket",
            radial_passed,
            value=radial_error,
            threshold={"max_tip_radial_error_m": float(max_tip_radial_error_m)},
        ),
        _predicate(
            "compiled_tip_inside_depth",
            depth_passed,
            value=inside_depth,
            threshold={"min_tip_inside_depth_m": float(min_tip_inside_depth_m)},
        ),
    ]
    required = [row for row in predicates if row.get("required")]
    failure_category = None
    if required and not all(bool(row.get("passed")) for row in required):
        if not terminal_goal_found:
            failure_category = "compiler did not emit semantic terminal contact subgoal"
        elif not geometry_available:
            failure_category = "compiled task-goal geometry unavailable"
        elif not radial_passed:
            failure_category = "compiled terminal tip outside socket aperture"
        elif not depth_passed:
            failure_category = "compiled terminal tip not inside socket"
    return {
        "schema_version": "toolframe_compiled_task_goal_check_v1",
        "task_type": "screwdriver_insert",
        "policy": {
            "name": "compiled_screwdriver_tip_inside_socket",
            "task_type": "screwdriver_insert",
            "requires_terminal_contact_goal": True,
            "requires_tip_radial_inside_socket": True,
            "requires_tip_inside_depth": True,
        },
        "passed": bool(required) and all(bool(row.get("passed")) for row in required),
        "predicates": predicates,
        "diagnostics": {
            "terminal_stage": (
                terminal_goal.get("stage")
                if isinstance(terminal_goal, Mapping)
                else None
            ),
            "terminal_contact_point_world": terminal_contact,
            "socket_center_world": socket_center,
            "insertion_axis_world": axis,
            "compiled_tip_radial_error_m": radial_error,
            "compiled_tip_inside_depth_m": inside_depth,
            "failure_category": failure_category,
        },
    }

