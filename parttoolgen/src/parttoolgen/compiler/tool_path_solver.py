#!/usr/bin/env python3
"""Collision-aware path gate for whole-tool pose subgoals.

The subgoal compiler solves symbolic point/axis constraints. This module adds a
separate diagnostic path layer: it checks the whole tool mesh against scene
object AABBs and reports infeasible compiler outputs. The current mainline does
not insert corridor, tracking, recovery, or truncation waypoints here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.adapters.simtoolreal_policy_priors import (  # noqa: E402
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
)
from parttoolgen.checks.subgoal_scene_collision import (  # noqa: E402
    NEAR_CONTACT_EFFECT_GOAL_GRAZING_M,
    _allowed_contact_penetration_m,
    _expected_contact_allows_penetration,
    _object_ref_for_target,
    _pose_records,
    _scene_obstacles,
    _semantic_role,
    _transform_point,
    check_subgoal_scene_collision,
)
from parttoolgen.compiler.objective_runtime import (  # noqa: E402
    ObjectiveFunction,
    ObjectiveRuntimeError,
    load_objective_functions,
    pose_objective_residuals,
)
from parttoolgen.compiler.reference_resolver import ReferenceResolver  # noqa: E402
from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    load_structured_file,
    validate_functional_tool_parts,
    validate_grounding_candidates,
    validate_observed_scene_graph,
    validate_program_ir,
    validate_target_part_graph,
    validate_tool_part_graph,
    validate_tool_pose_subgoals,
)


def _point_aabb_clearance_and_escape(
    point: Sequence[float],
    center: Sequence[float],
    extent: Sequence[float],
) -> tuple[float, list[float]]:
    half = [0.5 * float(v) for v in extent]
    deltas = [float(p) - float(c) for p, c in zip(point, center)]
    outside = [0.0, 0.0, 0.0]
    inside_margins: list[tuple[float, int]] = []
    for axis, (delta, h) in enumerate(zip(deltas, half)):
        abs_delta = abs(delta)
        if abs_delta > h:
            outside[axis] = math.copysign(abs_delta - h, delta)
        else:
            inside_margins.append((h - abs_delta, axis))
    outside_norm = math.sqrt(sum(v * v for v in outside))
    if outside_norm > 0.0:
        return outside_norm, [v / outside_norm for v in outside]
    margin, axis = min(inside_margins, key=lambda item: item[0])
    sign = 1.0 if deltas[axis] >= 0.0 else -1.0
    direction = [0.0, 0.0, 0.0]
    direction[axis] = sign
    return -margin, direction


def _distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _goal_expected_target(
    goal: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    expected_contact = goal.get("expected_contact", {})
    target_ref = None
    if isinstance(expected_contact, Mapping):
        value = expected_contact.get("scene_target")
        if isinstance(value, str):
            target_ref = value
    object_ref, role = _object_ref_for_target(
        target_ref,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    return target_ref, object_ref, role


def _goal_allows_contact_penetration(goal: Mapping[str, Any]) -> bool:
    expected_contact = goal.get("expected_contact", {})
    return _expected_contact_allows_penetration(
        expected_contact if isinstance(expected_contact, Mapping) else None
    )


def _goal_target_object_ref(
    goal: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> str | None:
    _target_ref, object_ref, _target_role = _goal_expected_target(
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    return object_ref


def _goal_endpoint_allows_contact_penetration(goal: Mapping[str, Any]) -> bool:
    return (
        not _is_compiler_transition_waypoint(goal)
        and _goal_allows_contact_penetration(goal)
    )


def _contact_semantic_role(goal: Mapping[str, Any]) -> str:
    source_constraints = goal.get("source_constraints")
    if isinstance(source_constraints, Sequence) and not isinstance(source_constraints, str):
        for constraint in source_constraints:
            if not isinstance(constraint, str):
                continue
            prefix = "source_semantic_role:"
            if constraint.startswith(prefix):
                return constraint[len(prefix) :]
    role = goal.get("semantic_role")
    return role if isinstance(role, str) else ""


def _goal_is_terminal_contact_anchor(goal: Mapping[str, Any]) -> bool:
    explicit_gate = goal.get("semantic_gate")
    if isinstance(explicit_gate, bool) and explicit_gate:
        return True
    role = _contact_semantic_role(goal).lower()
    return "terminal" in role


def _goal_allows_task_direction_contact_penetration(
    goal: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> bool:
    _target_ref, object_ref, role = _goal_expected_target(
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    return (
        object_ref is not None
        and role == "task_direction_contact_goal_point"
        and _goal_allows_contact_penetration(goal)
        and _goal_is_terminal_contact_anchor(goal)
    )


def _goal_endpoint_allows_contact_penetration_for_scene(
    goal: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> bool:
    if _goal_endpoint_allows_contact_penetration(goal):
        return True
    return _goal_allows_task_direction_contact_penetration(
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )


def _goal_uses_near_contact_clearance(goal: Mapping[str, Any]) -> bool:
    return _goal_allows_contact_penetration(goal)


def _segment_interior_allows_contact_penetration(
    previous_goal: Mapping[str, Any] | None,
    goal: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> bool:
    """Allow swept contact only when both segment endpoints are contact anchors."""

    if not isinstance(previous_goal, Mapping):
        return False
    if not (
        _goal_endpoint_allows_contact_penetration_for_scene(
            previous_goal,
            scene=scene,
            target_part_graph=target_part_graph,
        )
        and _goal_endpoint_allows_contact_penetration_for_scene(
            goal,
            scene=scene,
            target_part_graph=target_part_graph,
        )
    ):
        return False
    previous_object_ref = _goal_target_object_ref(
        previous_goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    current_object_ref = _goal_target_object_ref(
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    return previous_object_ref is not None and previous_object_ref == current_object_ref


def _normalize_quat_xyzw(quat: Sequence[float]) -> list[float]:
    values = [float(v) for v in quat]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [v / norm for v in values]


def _quat_angle_rad(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _normalize_quat_xyzw(a)
    qb = _normalize_quat_xyzw(b)
    dot = abs(sum(float(x) * float(y) for x, y in zip(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _lerp_pose(
    start_pose: Mapping[str, Any],
    end_pose: Mapping[str, Any],
    fraction: float,
) -> dict[str, Any]:
    t = max(0.0, min(1.0, float(fraction)))
    start_pos = [float(v) for v in start_pose["position"]]
    end_pos = [float(v) for v in end_pose["position"]]
    start_q = [float(v) for v in start_pose["orientation_xyzw"]]
    end_q = [float(v) for v in end_pose["orientation_xyzw"]]
    if sum(a * b for a, b in zip(start_q, end_q)) < 0.0:
        end_q = [-v for v in end_q]
    return {
        "position": [
            (1.0 - t) * a + t * b
            for a, b in zip(start_pos, end_pos)
        ],
        "orientation_xyzw": _normalize_quat_xyzw(
            (1.0 - t) * a + t * b
            for a, b in zip(start_q, end_q)
        ),
    }


def _goal_clearance_violations(
    *,
    goal: Mapping[str, Any],
    pose: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    vertices_tool: Sequence[Sequence[float]],
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
    min_clearance_m: float,
    allowed_contact_penetration_m: float,
    allow_contact_penetration: bool = True,
) -> tuple[bool, list[dict[str, Any]]]:
    stage = str(goal.get("stage", ""))
    target_ref, target_object_ref, target_role = _goal_expected_target(
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    violations: list[dict[str, Any]] = []
    for obstacle in obstacles:
        contact_allowed = (
            target_object_ref == obstacle["object_ref"]
            and allow_contact_penetration
            and _goal_allows_contact_penetration(goal)
        )
        near_contact_effect_goal_grazing = (
            target_object_ref == obstacle["object_ref"]
            and not contact_allowed
            and _goal_uses_near_contact_clearance(goal)
            and target_role == "task_direction_contact_goal_point"
        )
        expected_contact = goal.get("expected_contact", {})
        contact_penetration_m = _allowed_contact_penetration_m(
            expected_contact if isinstance(expected_contact, Mapping) else None,
            default_m=allowed_contact_penetration_m,
            target_role=target_role,
        )
        required = (
            -contact_penetration_m
            if contact_allowed
            else -NEAR_CONTACT_EFFECT_GOAL_GRAZING_M
            if near_contact_effect_goal_grazing
            else float(min_clearance_m)
        )
        min_clearance = float("inf")
        nearest_vertex = [0.0, 0.0, 0.0]
        for vertex in vertices_tool:
            world_point = _transform_point(pose, vertex)
            clearance, _escape = _point_aabb_clearance_and_escape(
                world_point,
                obstacle["center_xyz_world"],
                obstacle["extent_xyz"],
            )
            if clearance < min_clearance:
                min_clearance = clearance
                nearest_vertex = [float(v) for v in world_point]
        if min_clearance + 1e-9 < required:
            violations.append(
                {
                    "object_ref": obstacle["object_ref"],
                    "clearance_m": min_clearance,
                    "required_clearance_m": required,
                    "allowed_contact_penetration_m": contact_penetration_m
                    if contact_allowed
                    else NEAR_CONTACT_EFFECT_GOAL_GRAZING_M
                    if near_contact_effect_goal_grazing
                    else 0.0,
                    "nearest_vertex_world": nearest_vertex,
                    "contact_allowed": contact_allowed,
                    "contact_penetration_scope": (
                        "expected_target_contact"
                        if contact_allowed
                        else "near_contact_effect_goal_grazing"
                        if near_contact_effect_goal_grazing
                        else "strict_path_clearance"
                    ),
                    "expected_target_ref": target_ref,
                    "expected_target_role": target_role,
                }
            )
    return not violations, violations


def _line_search_safe_pose(
    *,
    start_pose: Mapping[str, Any],
    end_pose: Mapping[str, Any],
    goal: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    vertices_tool: Sequence[Sequence[float]],
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
    min_clearance_m: float,
    swept_min_clearance_m: float,
    allowed_contact_penetration_m: float,
    endpoint_allows_contact_penetration: bool | None = None,
    interior_allows_contact_penetration: bool = False,
    iterations: int = 32,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    endpoint_contact_allowed = (
        _goal_endpoint_allows_contact_penetration_for_scene(
            goal,
            scene=scene,
            target_part_graph=target_part_graph,
        )
        if endpoint_allows_contact_penetration is None
        else bool(endpoint_allows_contact_penetration)
    )
    start_safe, _start_violations = _goal_clearance_violations(
        goal=goal,
        pose=start_pose,
        obstacles=obstacles,
        vertices_tool=vertices_tool,
        scene=scene,
        target_part_graph=target_part_graph,
        min_clearance_m=swept_min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
        allow_contact_penetration=interior_allows_contact_penetration,
    )
    end_safe, end_violations = _goal_clearance_violations(
        goal=goal,
        pose=end_pose,
        obstacles=obstacles,
        vertices_tool=vertices_tool,
        scene=scene,
        target_part_graph=target_part_graph,
        min_clearance_m=min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
        allow_contact_penetration=endpoint_contact_allowed,
    )
    if end_safe:
        return dict(end_pose), 1.0, []
    if not start_safe:
        return dict(start_pose), 0.0, end_violations

    low, high = 0.0, 1.0
    low_pose = _lerp_pose(start_pose, end_pose, low)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        mid_pose = _lerp_pose(start_pose, end_pose, mid)
        mid_safe, _mid_violations = _goal_clearance_violations(
            goal=goal,
            pose=mid_pose,
            obstacles=obstacles,
            vertices_tool=vertices_tool,
            scene=scene,
            target_part_graph=target_part_graph,
            min_clearance_m=swept_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            allow_contact_penetration=interior_allows_contact_penetration,
        )
        if mid_safe:
            low = mid
            low_pose = mid_pose
        else:
            high = mid
    return low_pose, low, end_violations


def _is_compiler_transition_waypoint(goal: Mapping[str, Any]) -> bool:
    stage = str(goal.get("stage", ""))
    if "_transition_" in stage:
        return True
    return goal.get("semantic_role") == "generated_transition_waypoint"


def _goal_semantic_gate(goal: Mapping[str, Any]) -> bool:
    explicit_gate = goal.get("semantic_gate")
    if isinstance(explicit_gate, bool):
        return explicit_gate
    return not _is_compiler_transition_waypoint(goal)


def _segment_sample_check(
    *,
    start_pose: Mapping[str, Any],
    end_pose: Mapping[str, Any],
    goal: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    vertices_tool: Sequence[Sequence[float]],
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
    min_clearance_m: float,
    swept_min_clearance_m: float,
    allowed_contact_penetration_m: float,
    previous_goal: Mapping[str, Any] | None = None,
    endpoint_allows_contact_penetration: bool | None = None,
    samples: int = 9,
) -> tuple[bool, float, list[dict[str, Any]]]:
    endpoint_contact_allowed = (
        _goal_endpoint_allows_contact_penetration_for_scene(
            goal,
            scene=scene,
            target_part_graph=target_part_graph,
        )
        if endpoint_allows_contact_penetration is None
        else bool(endpoint_allows_contact_penetration)
    )
    interior_contact_allowed = _segment_interior_allows_contact_penetration(
        previous_goal,
        goal,
        scene=scene,
        target_part_graph=target_part_graph,
    )
    intervals = max(1, int(samples) - 1)
    for sample_index in range(1, intervals + 1):
        fraction = sample_index / float(intervals)
        pose = _lerp_pose(start_pose, end_pose, fraction)
        allow_contact_penetration = (
            endpoint_contact_allowed if sample_index == intervals else interior_contact_allowed
        )
        sample_min_clearance_m = (
            float(min_clearance_m)
            if allow_contact_penetration
            else 0.0
            if endpoint_contact_allowed or _goal_uses_near_contact_clearance(goal)
            else float(swept_min_clearance_m)
        )
        safe, violations = _goal_clearance_violations(
            goal=goal,
            pose=pose,
            obstacles=obstacles,
            vertices_tool=vertices_tool,
            scene=scene,
            target_part_graph=target_part_graph,
            min_clearance_m=sample_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            allow_contact_penetration=allow_contact_penetration,
        )
        if not safe:
            return False, fraction, violations
    return True, 1.0, []


def _check_subgoal_segments(
    *,
    subgoals: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    vertices_tool: Sequence[Sequence[float]],
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
    min_clearance_m: float,
    swept_min_clearance_m: float,
    allowed_contact_penetration_m: float,
    samples: int = 9,
) -> dict[str, Any]:
    previous_pose: Mapping[str, Any] = subgoals["start_pose"]
    previous_goal: Mapping[str, Any] | None = None
    violations: list[dict[str, Any]] = []
    checked_segments = 0
    for goal in subgoals.get("goals", []):
        if not isinstance(goal, Mapping):
            continue
        pose = goal.get("pose_world_tool")
        if not isinstance(pose, Mapping):
            continue
        checked_segments += 1
        safe, first_bad_fraction, segment_violations = _segment_sample_check(
            start_pose=previous_pose,
            end_pose=pose,
            goal=goal,
            obstacles=obstacles,
            vertices_tool=vertices_tool,
            scene=scene,
            target_part_graph=target_part_graph,
            min_clearance_m=min_clearance_m,
            swept_min_clearance_m=swept_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            previous_goal=previous_goal,
            samples=samples,
        )
        if not safe:
            violations.append(
                {
                    "goal_index": goal.get("index"),
                    "stage": goal.get("stage"),
                    "first_bad_fraction": first_bad_fraction,
                    "violations": segment_violations,
                }
            )
        previous_pose = pose
        previous_goal = goal
    return {
        "passed": not violations,
        "checked_segments": checked_segments,
        "samples_per_segment": int(samples),
        "violations": violations,
    }


def _reindex_goals(goals: Sequence[dict[str, Any]]) -> None:
    for index, goal in enumerate(goals):
        goal["index"] = index


def _path_objectives(program_ir: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(program_ir, Mapping):
        return []
    return [
        dict(objective)
        for objective in program_ir.get("objectives", [])
        if isinstance(objective, Mapping) and objective.get("kind") == "path"
    ]


def _goal_program_stage(goal_stage: str, program_ir: Mapping[str, Any] | None) -> str | None:
    if not isinstance(program_ir, Mapping):
        return None
    stage_names = [
        str(stage.get("name"))
        for stage in program_ir.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("name")
    ]
    for stage_name in sorted(stage_names, key=len, reverse=True):
        if goal_stage == stage_name or goal_stage.startswith(f"{stage_name}_"):
            return stage_name
    return None


def _path_objectives_for_stage(
    objectives: Sequence[Mapping[str, Any]],
    stage_name: str | None,
) -> list[dict[str, Any]]:
    if stage_name is None:
        return []
    return [
        dict(objective)
        for objective in objectives
        if objective.get("stage") == stage_name
    ]


def _path_objective_segment_check(
    *,
    start_pose: Mapping[str, Any],
    end_pose: Mapping[str, Any],
    goal: Mapping[str, Any],
    stage_name: str,
    objectives: list[dict[str, Any]],
    functions: dict[str, ObjectiveFunction],
    resolver: ReferenceResolver,
    samples: int,
    tolerance: float,
) -> tuple[dict[str, Any], float, bool, list[dict[str, Any]], list[dict[str, Any]]]:
    intervals = max(1, int(samples) - 1)
    results: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    last_safe_fraction = 0.0
    first_bad_fraction: float | None = None

    for sample_index in range(intervals + 1):
        fraction = sample_index / float(intervals)
        pose = _lerp_pose(start_pose, end_pose, fraction)
        residuals = pose_objective_residuals(
            pose=pose,
            objectives=objectives,
            functions=functions,
            resolver=resolver,
        )
        sample_bad = False
        for objective in objectives:
            objective_id = str(objective.get("id") or objective.get("function_name"))
            key = f"{objective_id}.objective_residual"
            residual = float(residuals[key])
            weighted_cost = float(objective.get("weight", 1.0)) * max(residual, 0.0) ** 2
            record = results.setdefault(
                objective_id,
                {
                    "objective_id": objective_id,
                    "stage": stage_name,
                    "goal_index": goal.get("index"),
                    "goal_stage": goal.get("stage"),
                    "max_residual": residual,
                    "weighted_cost": weighted_cost,
                    "passed": True,
                },
            )
            if residual > float(record["max_residual"]):
                record["max_residual"] = residual
            if weighted_cost > float(record["weighted_cost"]):
                record["weighted_cost"] = weighted_cost
            if residual > tolerance:
                sample_bad = True
                record["passed"] = False
                violations.append(
                    {
                        "objective_id": objective_id,
                        "stage": stage_name,
                        "goal_index": goal.get("index"),
                        "goal_stage": goal.get("stage"),
                        "sample_fraction": fraction,
                        "residual": residual,
                        "tolerance": float(tolerance),
                        "pose_world_tool": pose,
                    }
                )
        if sample_bad:
            first_bad_fraction = fraction
            break
        last_safe_fraction = fraction

    safe_pose = _lerp_pose(start_pose, end_pose, last_safe_fraction)
    unrepaired_failure = first_bad_fraction == 0.0
    return safe_pose, last_safe_fraction, unrepaired_failure, list(results.values()), violations


def solve_collision_safe_tool_path(
    subgoals: dict[str, Any],
    scene: dict[str, Any],
    *,
    program_ir: dict[str, Any] | None = None,
    planner_program: dict[str, Any] | str | None = None,
    functional_tool_parts: dict[str, Any] | None = None,
    grounding_candidates: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    tool_id: str | None = None,
    tool_mesh_path: str | Path | None = None,
    min_clearance_m: float = 0.003,
    swept_min_clearance_m: float = 0.006,
    allowed_contact_penetration_m: float = 0.006,
    max_iterations: int = 32,
    path_objective_samples_per_segment: int = 9,
    path_objective_tolerance: float = 1e-4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_tool_pose_subgoals(subgoals)
    validate_observed_scene_graph(scene)
    if program_ir:
        validate_program_ir(program_ir)
    if functional_tool_parts:
        validate_functional_tool_parts(functional_tool_parts)
    if grounding_candidates:
        validate_grounding_candidates(grounding_candidates)
    if target_part_graph:
        validate_target_part_graph(target_part_graph)
    if tool_part_graph:
        validate_tool_part_graph(tool_part_graph)
    swept_min_clearance_m = max(float(min_clearance_m), float(swept_min_clearance_m))

    mesh_path = resolve_simtoolreal_tool_mesh_path(
        tool_id or subgoals.get("tool_id"),
        explicit_path=tool_mesh_path,
    )
    path_objectives = _path_objectives(program_ir)
    declared_path_objectives = max(
        len(path_objectives),
        int(
            subgoals.get("sanity_checks", {}).get("path_objectives_declared", 0)
            if isinstance(subgoals.get("sanity_checks"), Mapping)
            else 0
        ),
    )
    path_objective_ids = [
        str(objective.get("id") or objective.get("function_name"))
        for objective in path_objectives
    ]
    path_objective_functions: dict[str, ObjectiveFunction] = {}
    path_objective_resolver: ReferenceResolver | None = None
    path_objective_warnings: list[str] = []
    path_objectives_checked = False
    path_objectives_passed = True
    if path_objectives:
        missing_inputs = []
        if planner_program is None:
            missing_inputs.append("planner_program")
        if functional_tool_parts is None:
            missing_inputs.append("functional_tool_parts")
        if missing_inputs:
            path_objectives_passed = False
            path_objective_warnings.append(
                "path objectives declared but required input(s) were not provided: "
                + ", ".join(missing_inputs)
            )
        else:
            try:
                path_objective_functions = load_objective_functions(
                    planner_program=planner_program,
                    program_ir={**program_ir, "objectives": path_objectives},
                )
                path_objective_resolver = ReferenceResolver(
                    observed_scene_graph=scene,
                    functional_tool_parts=functional_tool_parts or {},
                    grounding_candidates=grounding_candidates,
                    target_part_graph=target_part_graph,
                    tool_part_graph=tool_part_graph,
                )
                path_objectives_checked = True
            except ObjectiveRuntimeError as exc:
                path_objectives_passed = False
                path_objective_warnings.append(f"failed to load path objectives: {exc}")
    elif declared_path_objectives:
        path_objectives_passed = False
        path_objective_warnings.append(
            "ToolPoseSubgoals declares deferred path objectives but program_ir was not provided"
        )
    before = check_subgoal_scene_collision(
        subgoals,
        scene,
        target_part_graph=target_part_graph,
        tool_id=tool_id,
        tool_mesh_path=tool_mesh_path,
        min_clearance_m=min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
    )
    checked_subgoals = copy.deepcopy(subgoals)
    report: dict[str, Any] = {
        "schema_version": "v0.1",
        "passed": True,
        "changed": False,
        "tool_id": tool_id or subgoals.get("tool_id"),
        "tool_mesh_path": str(mesh_path) if mesh_path else None,
        "min_clearance_m": float(min_clearance_m),
        "swept_min_clearance_m": float(swept_min_clearance_m),
        "allowed_contact_penetration_m": float(allowed_contact_penetration_m),
        "path_solver_mode": "diagnostic_no_goal_repair_v1",
        "line_search_iterations": int(max_iterations),
        "semantic_goal_preserved": True,
        "path_objectives_checked": path_objectives_checked,
        "path_objectives_passed": path_objectives_passed,
        "path_objective_runtime": "restricted_path_cost_v0" if path_objectives_checked else "not_loaded",
        "path_objective_ids": path_objective_ids,
        "path_objective_samples_per_segment": int(path_objective_samples_per_segment),
        "path_objective_tolerance": float(path_objective_tolerance),
        "segment_contact_penetration_policy": (
            "no_intermediate_penetration; swept_clearance_for_non_contact_path; "
            "terminal_task_direction_contact_inherits_declared_penetration; "
            "nonterminal_task_direction_contact_allows_small_grazing; "
            "ordinary_generated_waypoints_do_not_inherit_contact_penetration"
        ),
        "path_objective_results": [],
        "path_objective_violations": [],
        "path_failures": [],
        "path_objective_failures": [],
        "repair_hints": [],
        "warnings": list(path_objective_warnings),
        "before": before,
    }
    if mesh_path is None:
        report["passed"] = False
        report["warnings"] = ["tool mesh not resolved; path solver did not run"]
        checked_subgoals.setdefault("sanity_checks", {})["tool_path_solver"] = {
            "checked": False,
            "passed": False,
            "changed": False,
            "semantic_goal_preserved": True,
            "collision_report_passed": False,
            "path_objectives_checked": path_objectives_checked,
            "path_objectives_passed": path_objectives_passed,
        }
        return checked_subgoals, report

    vertices = load_obj_vertices(mesh_path)
    obstacles = _scene_obstacles(scene)
    if not vertices and not path_objectives_checked:
        report["after"] = before
        report["passed"] = bool(before.get("passed")) and path_objectives_passed
        checked_subgoals.setdefault("sanity_checks", {})["tool_path_solver"] = {
            "checked": bool(vertices),
            "passed": report["passed"],
            "changed": report["changed"],
            "semantic_goal_preserved": True,
            "collision_report_passed": bool(before.get("passed")),
            "path_objectives_checked": path_objectives_checked,
            "path_objectives_passed": path_objectives_passed,
        }
        if report["changed"]:
            checked_subgoals["subgoals_id"] = f"{checked_subgoals['subgoals_id']}_path_solved"
            checked_subgoals.setdefault("metadata", {})["generated_by"] = (
                "parttoolgen.tool_path_solver"
            )
            checked_subgoals.setdefault("metadata", {})["source"] = (
                "ToolPoseSubgoals+ObservedSceneGraph+ToolMesh"
            )
        validate_tool_pose_subgoals(checked_subgoals)
        return checked_subgoals, report
    if not vertices:
        report["warnings"].append(
            "tool mesh had no vertices; collision path gate skipped mesh collision checks but path objectives were checked"
        )
        obstacles = []
    if not obstacles and not path_objectives_checked:
        report["after"] = before
        report["passed"] = bool(before.get("passed")) and path_objectives_passed
        checked_subgoals.setdefault("sanity_checks", {})["tool_path_solver"] = {
            "checked": bool(vertices),
            "passed": report["passed"],
            "changed": report["changed"],
            "semantic_goal_preserved": True,
            "collision_report_passed": bool(before.get("passed")),
            "path_objectives_checked": path_objectives_checked,
            "path_objectives_passed": path_objectives_passed,
        }
        if report["changed"]:
            checked_subgoals["subgoals_id"] = f"{checked_subgoals['subgoals_id']}_path_solved"
            checked_subgoals.setdefault("metadata", {})["generated_by"] = (
                "parttoolgen.tool_path_solver"
            )
            checked_subgoals.setdefault("metadata", {})["source"] = (
                "ToolPoseSubgoals+ObservedSceneGraph+ToolMesh"
            )
        validate_tool_pose_subgoals(checked_subgoals)
        return checked_subgoals, report

    previous_pose: Mapping[str, Any] = checked_subgoals["start_pose"]
    previous_goal: Mapping[str, Any] | None = None
    goals = checked_subgoals.get("goals", [])
    goal_cursor = 0
    while goal_cursor < len(goals):
        goal = goals[goal_cursor]
        if not isinstance(goal, dict):
            goal_cursor += 1
            continue
        pose = goal.get("pose_world_tool")
        if not isinstance(pose, dict):
            goal_cursor += 1
            continue
        segment_safe, segment_bad_fraction, segment_violations = _segment_sample_check(
            start_pose=previous_pose,
            end_pose=pose,
            goal=goal,
            obstacles=obstacles,
            vertices_tool=vertices,
            scene=scene,
            target_part_graph=target_part_graph,
            min_clearance_m=min_clearance_m,
            swept_min_clearance_m=swept_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            previous_goal=previous_goal,
        )
        if not segment_safe:
            report["semantic_goal_preserved"] = False
            report["path_failures"].append(
                {
                    "goal_index": goal.get("index"),
                    "stage": goal.get("stage"),
                    "segment_first_bad_fraction": segment_bad_fraction,
                    "blocked_by": segment_violations,
                }
            )
            report["warnings"].append(
                f"goal {goal.get('index')} segment is not collision clear; "
                "mainline diagnostic mode reports the failure instead of inserting corridor waypoints"
            )
            report["repair_hints"].append(
                {
                    "goal_index": goal.get("index"),
                    "stage": goal.get("stage"),
                    "hint": (
                        "Regenerate public scene refs, selected tool refs, or staged constraints; "
                        "the path gate does not alter compiler subgoals."
                    ),
                }
            )
        safe_pose, safe_fraction, end_violations = _line_search_safe_pose(
            start_pose=previous_pose,
            end_pose=pose,
            goal=goal,
            obstacles=obstacles,
            vertices_tool=vertices,
            scene=scene,
            target_part_graph=target_part_graph,
            min_clearance_m=min_clearance_m,
            swept_min_clearance_m=swept_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            interior_allows_contact_penetration=_segment_interior_allows_contact_penetration(
                previous_goal,
                goal,
                scene=scene,
                target_part_graph=target_part_graph,
            ),
            iterations=max_iterations,
        )
        if safe_fraction < 1.0:
            report["semantic_goal_preserved"] = False
            report["path_failures"].append(
                {
                    "goal_index": goal.get("index"),
                    "stage": goal.get("stage"),
                    "safe_fraction": safe_fraction,
                    "safe_pose_world_tool": safe_pose,
                    "blocked_by": end_violations,
                }
            )
        if path_objectives_checked and path_objective_resolver is not None:
            program_stage = _goal_program_stage(str(goal.get("stage", "")), program_ir)
            segment_objectives = _path_objectives_for_stage(path_objectives, program_stage)
            if segment_objectives:
                try:
                    (
                        objective_safe_pose,
                        objective_safe_fraction,
                        unrepaired_failure,
                        objective_results,
                        objective_violations,
                    ) = _path_objective_segment_check(
                        start_pose=previous_pose,
                        end_pose=goal["pose_world_tool"],
                        goal=goal,
                        stage_name=str(program_stage),
                        objectives=segment_objectives,
                        functions=path_objective_functions,
                        resolver=path_objective_resolver,
                        samples=path_objective_samples_per_segment,
                        tolerance=path_objective_tolerance,
                    )
                except ObjectiveRuntimeError as exc:
                    path_objectives_passed = False
                    report["warnings"].append(
                        f"path objective evaluation failed for goal {goal.get('index')}: {exc}"
                    )
                    objective_results = []
                    objective_violations = []
                    objective_safe_pose = dict(goal["pose_world_tool"])
                    objective_safe_fraction = 1.0
                    unrepaired_failure = True
                report["path_objective_results"].extend(objective_results)
                report["path_objective_violations"].extend(objective_violations)
                if unrepaired_failure:
                    path_objectives_passed = False
                    report["warnings"].append(
                        f"goal {goal.get('index')} path objective is already violated at segment start"
                    )
                    report["repair_hints"].append(
                        {
                            "goal_index": goal.get("index"),
                            "stage": goal.get("stage"),
                            "hint": (
                                "Repair the previous ProgramIR stage or selected refs; "
                                "the path objective is violated before this segment starts."
                            ),
                        }
                    )
                elif objective_safe_fraction < 1.0:
                    path_objectives_passed = False
                    failure_record = {
                        "goal_index": goal.get("index"),
                        "stage": goal.get("stage"),
                        "safe_fraction": objective_safe_fraction,
                        "safe_pose_world_tool": objective_safe_pose,
                        "path_objective_violations": objective_violations,
                    }
                    report["path_objective_failures"].append(failure_record)
                    report["warnings"].append(
                        f"goal {goal.get('index')} path objective would require truncation; "
                        "mainline diagnostic mode reports the failure instead of changing the goal"
                    )
                    report["repair_hints"].append(
                        {
                            "goal_index": goal.get("index"),
                            "stage": goal.get("stage"),
                            "hint": (
                                "Change ProgramIR refs/constraints or selected refs; "
                                "the path gate does not truncate compiler subgoals."
                            ),
                        }
                    )
        previous_pose = goal["pose_world_tool"]
        previous_goal = goal
        goal_cursor += 1

    _reindex_goals(checked_subgoals.get("goals", []))
    after = check_subgoal_scene_collision(
        checked_subgoals,
        scene,
        target_part_graph=target_part_graph,
        tool_id=tool_id,
        tool_mesh_path=tool_mesh_path,
        min_clearance_m=min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
    )
    after_segments = _check_subgoal_segments(
        subgoals=checked_subgoals,
        obstacles=obstacles,
        vertices_tool=vertices,
        scene=scene,
        target_part_graph=target_part_graph,
        min_clearance_m=min_clearance_m,
        swept_min_clearance_m=swept_min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
        samples=path_objective_samples_per_segment,
    )
    report["after"] = after
    report["after_segments"] = after_segments
    report["path_objectives_passed"] = path_objectives_passed
    report["passed"] = (
        bool(after.get("passed"))
        and bool(after_segments.get("passed"))
        and bool(path_objectives_passed)
        and bool(report["semantic_goal_preserved"])
    )
    checked_subgoals.setdefault("sanity_checks", {})["tool_path_solver"] = {
        "checked": True,
        "passed": report["passed"],
        "changed": report["changed"],
        "semantic_goal_preserved": report["semantic_goal_preserved"],
        "collision_report_passed": bool(after.get("passed")),
        "segment_collision_report_passed": bool(after_segments.get("passed")),
        "path_objectives_checked": path_objectives_checked,
        "path_objectives_passed": path_objectives_passed,
    }
    if report["changed"]:
        checked_subgoals["subgoals_id"] = f"{checked_subgoals['subgoals_id']}_path_solved"
        checked_subgoals.setdefault("metadata", {})["generated_by"] = (
            "parttoolgen.tool_path_solver"
        )
        checked_subgoals.setdefault("metadata", {})["source"] = (
            "ToolPoseSubgoals+ObservedSceneGraph+ToolMesh"
        )
    validate_tool_pose_subgoals(checked_subgoals)
    return checked_subgoals, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgoals", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--program-ir")
    parser.add_argument("--planner-program")
    parser.add_argument("--tool")
    parser.add_argument("--candidates")
    parser.add_argument("--target-parts")
    parser.add_argument("--tool-parts")
    parser.add_argument("--tool-id")
    parser.add_argument("--tool-mesh")
    parser.add_argument("--min-clearance-m", type=float, default=0.003)
    parser.add_argument("--swept-min-clearance-m", type=float, default=0.006)
    parser.add_argument("--allowed-contact-penetration-m", type=float, default=0.006)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--path-objective-samples-per-segment", type=int, default=9)
    parser.add_argument("--path-objective-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    checked_subgoals, report = solve_collision_safe_tool_path(
        load_structured_file(args.subgoals),
        load_structured_file(args.scene),
        program_ir=load_structured_file(args.program_ir) if args.program_ir else None,
        planner_program=load_structured_file(args.planner_program)
        if args.planner_program
        else None,
        functional_tool_parts=load_structured_file(args.tool) if args.tool else None,
        grounding_candidates=load_structured_file(args.candidates)
        if args.candidates
        else None,
        target_part_graph=(
            load_structured_file(args.target_parts) if args.target_parts else None
        ),
        tool_part_graph=load_structured_file(args.tool_parts) if args.tool_parts else None,
        tool_id=args.tool_id,
        tool_mesh_path=args.tool_mesh,
        min_clearance_m=args.min_clearance_m,
        swept_min_clearance_m=args.swept_min_clearance_m,
        allowed_contact_penetration_m=args.allowed_contact_penetration_m,
        max_iterations=args.max_iterations,
        path_objective_samples_per_segment=args.path_objective_samples_per_segment,
        path_objective_tolerance=args.path_objective_tolerance,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(checked_subgoals, indent=2) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
