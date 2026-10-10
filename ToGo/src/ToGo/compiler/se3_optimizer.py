#!/usr/bin/env python3
"""Stage-wise SE(3) optimizer for PartToolGen whole-tool pose subgoals."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple

from ToGo.compiler.constraint_canonicalizer import (
    CanonicalStageConstraints,
    ClearanceTarget,
    MoveAlongTarget,
)
from ToGo.compiler.geometric_solvers import quat_between_vectors, quat_mul, quat_rotate
from ToGo.compiler.residuals import (
    add,
    axis_alignment_cost,
    clearance_margin_m,
    dot,
    mul,
    norm,
    normalize_quat,
    point_tracking_error_m,
    pose_for_tool_point_goal,
    pose_point,
    quat_angle_rad,
    quat_slerp,
    stage_cost,
    sub,
    unit,
    weighted_position_average,
)


class SE3OptimizerError(ValueError):
    """Raised when a stage cannot be solved into a tool pose."""


_TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M = 1e-6


def _approach_roll_max_rad() -> float:
    value = os.environ.get("TOGO_COMPILER_APPROACH_ROLL_MAX_RAD", "1.2")
    try:
        limit = float(value)
    except (TypeError, ValueError):
        return 1.2
    if limit <= 0.0:
        return math.pi
    return min(limit, math.pi)


@dataclass(frozen=True)
class StageSolveResult:
    pose: dict[str, Any]
    primary_tool_ref: str
    primary_target_ref: str
    final_cost: float
    constraint_errors: dict[str, float]
    pose_selection: dict[str, Any]
    warnings: list[str]
    primary_tool_point: list[float] | None = None


class CompiledMoveWaypoint(NamedTuple):
    pose: dict[str, Any]
    tool_ref: str
    target_ref: str
    semantic_role: str
    semantic_gate: bool
    point_tool: list[float] | None = None


def _orientation_from_axes(
    *,
    current_orientation: list[float],
    stage: CanonicalStageConstraints,
    current_position: list[float],
    line_proximity_target: dict[str, Any] | None = None,
    table_clearance_context: dict[str, Any] | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    if len(stage.axis_targets) > 1:
        return _orientation_from_coupled_axes(
            current_orientation=current_orientation,
            stage=stage,
            current_position=current_position,
            line_proximity_target=line_proximity_target,
            table_clearance_context=table_clearance_context,
        )
    orientation = normalize_quat(current_orientation)
    resolutions: list[dict[str, Any]] = []
    for axis in stage.axis_targets:
        (
            best_orientation,
            best_polarity,
            best_rotation,
            best_cosine,
            selection_record,
        ) = _orientation_for_single_axis_with_geometry_cost(
            current_orientation=orientation,
            axis=axis,
            sample_points_tool=stage.orientation_sample_points_tool,
            current_position=current_position,
            line_proximity_target=line_proximity_target,
            stage=stage,
            table_clearance_context=table_clearance_context,
        )
        if best_orientation is None:
            raise SE3OptimizerError(f"{axis.constraint_id} has no feasible axis orientation candidate")
        orientation = best_orientation
        record = {
                "constraint_id": axis.constraint_id,
                "source_ref": axis.source_ref,
                "target_ref": axis.target_ref,
                "requested_polarity": axis.polarity,
                "resolved_polarity": best_polarity or axis.polarity,
                "axis_rotation_delta_rad": best_rotation,
                "axis_cosine_after_resolution": best_cosine,
        }
        record.update(selection_record)
        resolutions.append(record)
    return normalize_quat(orientation), resolutions


def _tool_geometry_displacement_cost(
    *,
    candidate_orientation: list[float],
    reference_orientation: list[float],
    sample_points_tool: list[list[float]],
) -> float:
    if not sample_points_tool:
        return quat_angle_rad(candidate_orientation, reference_orientation) ** 2
    total = 0.0
    for point in sample_points_tool:
        before = quat_rotate(reference_orientation, point)
        after = quat_rotate(candidate_orientation, point)
        delta = sub(after, before)
        total += dot(delta, delta)
    return total / float(len(sample_points_tool))


def _axis_line_proximity_cost(
    *,
    candidate_orientation: list[float],
    current_position: list[float],
    axis_world: list[float],
    line_proximity_target: dict[str, Any] | None,
) -> float:
    if not line_proximity_target:
        return 0.0
    point_tool = line_proximity_target.get("point_tool")
    point_world = line_proximity_target.get("point_world")
    if not isinstance(point_tool, list) or len(point_tool) != 3:
        return 0.0
    if not isinstance(point_world, list) or len(point_world) != 3:
        return 0.0
    contact_world = add(current_position, quat_rotate(candidate_orientation, point_tool))
    offset = sub(contact_world, point_world)
    axis = unit(axis_world)
    lateral = sub(offset, mul(axis, dot(offset, axis)))
    return dot(lateral, lateral)


def _support_clearance_plane(table_clearance_context: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    bbox_points_tool = table_clearance_context.get("bbox_points_tool") or []
    if not bbox_points_tool:
        return None
    plane_point = table_clearance_context.get("plane_point_world")
    plane_normal = table_clearance_context.get("plane_normal_world")
    if isinstance(plane_point, list) and len(plane_point) >= 3 and isinstance(plane_normal, list) and len(plane_normal) >= 3:
        return [float(item) for item in plane_point[:3]], unit([float(item) for item in plane_normal[:3]])
    table_top_z = table_clearance_context.get("table_top_z")
    if table_top_z is None:
        return None
    return [0.0, 0.0, float(table_top_z)], [0.0, 0.0, 1.0]


def _support_clearance_signed_distances(
    *,
    pose: dict[str, Any],
    bbox_points_tool: list[list[float]],
    plane_point_world: list[float],
    plane_normal_world: list[float],
) -> list[float]:
    orientation = normalize_quat(pose["orientation_xyzw"])
    origin = [float(item) for item in pose["position"]]
    plane_point = [float(item) for item in plane_point_world]
    plane_normal = unit([float(item) for item in plane_normal_world])
    return [
        dot(sub(add(origin, quat_rotate(orientation, point)), plane_point), plane_normal)
        for point in bbox_points_tool
    ]

def _orientation_table_clearance_violation_m(
    *,
    candidate_orientation: list[float],
    current_position: list[float],
    stage: CanonicalStageConstraints,
    line_proximity_target: dict[str, Any] | None,
    table_clearance_context: dict[str, Any] | None,
) -> float:
    if not table_clearance_context:
        return 0.0
    bbox_points_tool = table_clearance_context.get("bbox_points_tool") or []
    plane = _support_clearance_plane(table_clearance_context)
    if plane is None or not bbox_points_tool:
        return 0.0
    plane_point_world, plane_normal_world = plane
    clearance_m = float(table_clearance_context.get("clearance_m", 0.0))
    pose = {
        "position": list(current_position),
        "orientation_xyzw": normalize_quat(candidate_orientation),
    }
    if stage.point_targets:
        target = stage.point_targets[0]
        pose = pose_for_tool_point_goal(
            orientation_xyzw=candidate_orientation,
            point_tool=target.point_tool,
            point_world=target.point_world,
        )
    elif stage.move_along_targets:
        move = stage.move_along_targets[0]
        pose = pose_for_tool_point_goal(
            orientation_xyzw=candidate_orientation,
            point_tool=move.point_tool,
            point_world=move.start_world,
        )
    min_signed_distance = min(
        _support_clearance_signed_distances(
            pose=pose,
            bbox_points_tool=bbox_points_tool,
            plane_point_world=plane_point_world,
            plane_normal_world=plane_normal_world,
        )
    )
    return max(0.0, clearance_m - min_signed_distance)


def _orientation_stage_candidate_combo_cost(
    *,
    candidate_orientation: list[float],
    current_position: list[float],
    stage: CanonicalStageConstraints,
    line_proximity_target: dict[str, Any] | None,
    table_clearance_context: dict[str, Any] | None,
) -> tuple[float, float]:
    """Evaluate the best VLM-approved point/axis candidate combination.

    Axis candidates are enumerated by the caller. This helper makes the
    orientation score aware of all VLM-approved contact-point candidates too,
    instead of choosing a roll from only the primary point and selecting the
    point afterwards.
    """

    def pose_cost(pose: dict[str, Any]) -> tuple[float, float]:
        violation_m = _pose_table_clearance_violation_m(
            pose=pose,
            table_clearance_context=table_clearance_context,
        )
        translation_m = norm(sub(pose["position"], current_position))
        cost = stage_cost(
            pose=pose,
            point_targets=stage.point_targets,
            axis_targets=stage.axis_targets,
            clearance_targets=stage.clearance_targets,
        )
        return violation_m, cost + 1e-4 * translation_m * translation_m

    if stage.point_targets:
        pose = _pose_from_point_targets(
            orientation_xyzw=candidate_orientation,
            point_targets=stage.point_targets,
        )
        return pose_cost(pose)

    if stage.move_along_targets:
        move = stage.move_along_targets[0]
        start_pose = pose_for_tool_point_goal(
            orientation_xyzw=candidate_orientation,
            point_tool=move.point_tool,
            point_world=move.start_world,
        )
        goal_pose = pose_for_tool_point_goal(
            orientation_xyzw=candidate_orientation,
            point_tool=move.point_tool,
            point_world=move.goal_world,
        )
        start_violation = _pose_table_clearance_violation_m(
            pose=start_pose,
            table_clearance_context=table_clearance_context,
        )
        goal_violation = _pose_table_clearance_violation_m(
            pose=goal_pose,
            table_clearance_context=table_clearance_context,
        )
        start_translation_m = norm(sub(start_pose["position"], current_position))
        goal_translation_m = norm(sub(goal_pose["position"], current_position))
        return (
            max(start_violation, goal_violation),
            max(start_translation_m * start_translation_m, goal_translation_m * goal_translation_m),
        )

    violation_m = _orientation_table_clearance_violation_m(
        candidate_orientation=candidate_orientation,
        current_position=current_position,
        stage=stage,
        line_proximity_target=line_proximity_target,
        table_clearance_context=table_clearance_context,
    )
    return violation_m, 1e6 * violation_m * violation_m


def _stage_with_axis_candidate_vector(
    *,
    stage: CanonicalStageConstraints,
    constraint_id: str,
    vector_tool: list[float],
) -> CanonicalStageConstraints:
    axis_targets = [
        replace(axis, vector_tool=unit(vector_tool))
        if axis.constraint_id == constraint_id
        else axis
        for axis in stage.axis_targets
    ]
    return replace(stage, axis_targets=axis_targets)


def _pose_table_clearance_violation_m(
    *,
    pose: dict[str, Any],
    table_clearance_context: dict[str, Any] | None,
) -> float:
    if not table_clearance_context:
        return 0.0
    bbox_points_tool = table_clearance_context.get("bbox_points_tool") or []
    plane = _support_clearance_plane(table_clearance_context)
    if plane is None or not bbox_points_tool:
        return 0.0
    plane_point_world, plane_normal_world = plane
    clearance_m = float(table_clearance_context.get("clearance_m", 0.0))
    min_signed_distance = min(
        _support_clearance_signed_distances(
            pose=pose,
            bbox_points_tool=bbox_points_tool,
            plane_point_world=plane_point_world,
            plane_normal_world=plane_normal_world,
        )
    )
    return max(0.0, clearance_m - min_signed_distance)


def _tool_occupancy_clearance_margin_m(
    *,
    pose: dict[str, Any],
    target: Any,
) -> float:
    obstacle_world = getattr(target, "obstacle_world", None)
    sample_points_tool = getattr(target, "sample_points_tool", None) or []
    if not obstacle_world or not sample_points_tool:
        return float("inf")
    min_distance = float("inf")
    for point_tool in sample_points_tool:
        distance = norm(sub(pose_point(pose, point_tool), obstacle_world))
        min_distance = min(min_distance, distance)
    return float(min_distance - float(getattr(target, "min_distance_m", 0.0)))


def _tool_occupancy_clearance_violation_m(
    *,
    pose: dict[str, Any],
    targets: list[Any] | None,
) -> float:
    if not targets:
        return 0.0
    violation_m = 0.0
    for target in targets:
        margin = _tool_occupancy_clearance_margin_m(pose=pose, target=target)
        if margin < 0.0:
            violation_m = max(violation_m, -margin)
    return float(violation_m)


def _project_pose_to_table_feasible(
    *,
    pose: dict[str, Any],
    table_clearance_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Lift a nominal point/axis pose only as much as table feasibility needs."""

    violation_m = _pose_table_clearance_violation_m(
        pose=pose,
        table_clearance_context=table_clearance_context,
    )
    if violation_m <= _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M:
        return pose
    position = list(pose["position"])
    plane = _support_clearance_plane(table_clearance_context) if table_clearance_context else None
    if plane is None:
        position[2] = float(position[2]) + float(violation_m)
    else:
        _plane_point_world, plane_normal_world = plane
        position = add(position, mul(plane_normal_world, float(violation_m)))
    return {
        "position": position,
        "orientation_xyzw": list(pose["orientation_xyzw"]),
    }


def _require_pose_table_clearance(
    *,
    pose: dict[str, Any],
    table_clearance_context: dict[str, Any] | None,
    label: str,
) -> None:
    violation_m = _pose_table_clearance_violation_m(
        pose=pose,
        table_clearance_context=table_clearance_context,
    )
    if violation_m > _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M:
        raise SE3OptimizerError(
            f"{label} violates native support-plane bbox constraint by {violation_m:.6f}m"
        )


def _axis_source_vector_candidates(axis: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "primary_axis",
            "vector_tool": unit(axis.vector_tool),
            "source": "axis.vector_tool",
        }
    ]
    return candidates


def _selected_axis_candidate_record(
    *,
    axis_candidates: list[dict[str, Any]],
    selected_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_axis_candidate_count": len(axis_candidates),
    }
    if selected_candidate is None:
        return record
    record["selected_source_axis_tool"] = list(selected_candidate["vector_tool"])
    record["selected_source_axis_candidate_index"] = int(selected_candidate.get("candidate_index", 0))
    optional_fields = {
        "candidate_id": "selected_source_axis_candidate_id",
        "source_candidate_index": "selected_source_axis_source_candidate_index",
        "source": "selected_source_axis_candidate_source",
        "axis_marker": "selected_source_axis_candidate_marker",
        "axis_role": "selected_source_axis_candidate_role",
        "axis_ref": "selected_source_axis_candidate_ref",
        "surface_ref": "selected_source_axis_candidate_surface_ref",
        "patch_id": "selected_source_axis_candidate_patch_id",
    }
    for source_key, record_key in optional_fields.items():
        value = selected_candidate.get(source_key)
        if value is not None:
            record[record_key] = value
    return record


def _pose_from_point_targets(
    *,
    orientation_xyzw: list[float],
    point_targets: list[Any],
) -> dict[str, Any]:
    candidates = [
        (
            pose_for_tool_point_goal(
                orientation_xyzw=orientation_xyzw,
                point_tool=target.point_tool,
                point_world=target.point_world,
            )["position"],
            target.weight,
        )
        for target in point_targets
    ]
    return {
        "position": weighted_position_average(candidates),
        "orientation_xyzw": orientation_xyzw,
    }


def _select_point_candidates_for_stage(
    *,
    stage: CanonicalStageConstraints,
    orientation_xyzw: list[float],
    current_pose: dict[str, Any],
    objective_cost_fn: Callable[[dict[str, Any]], float] | None,
    table_clearance_context: dict[str, Any] | None = None,
) -> tuple[CanonicalStageConstraints, list[dict[str, Any]]]:
    # The selected tool contact point is the active geometry for the whole
    # stage. Candidate contacts are visual/evidence alternatives, not a late
    # optimizer degree of freedom: reselecting them here can satisfy the same
    # axis constraint with a different physical face and break tool-use
    # semantics. Keep the canonical point exactly as selected upstream.
    return stage, []


def _orientation_for_single_axis_with_geometry_cost(
    *,
    current_orientation: list[float],
    axis: Any,
    sample_points_tool: list[list[float]],
    current_position: list[float],
    line_proximity_target: dict[str, Any] | None,
    stage: CanonicalStageConstraints,
    table_clearance_context: dict[str, Any] | None,
) -> tuple[list[float] | None, str | None, float, float, dict[str, Any]]:
    """Align one axis and resolve free roll by minimizing tool geometry motion."""

    seed = normalize_quat(current_orientation)
    axis_candidates = _axis_source_vector_candidates(axis)
    best_orientation: list[float] | None = None
    best_polarity: str | None = None
    best_source_candidate: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    best_rotation = float("inf")
    best_cosine = -1.0
    best_roll = 0.0
    best_geometry_cost = float("inf")
    best_line_proximity_cost = 0.0
    best_table_clearance_violation_m = 0.0
    best_candidate_combo_cost = 0.0
    roll_step_rad = math.radians(2.0)
    for source_candidate in axis_candidates:
        source_vector = source_candidate["vector_tool"]
        current = unit(quat_rotate(seed, source_vector))
        for candidate_polarity, target in _axis_target_candidates(axis):
            delta = quat_between_vectors(current, target)
            base_orientation = normalize_quat(quat_mul(delta, seed))
            roll_axis = unit(target)
            for roll_index in range(-90, 91):
                roll_angle = roll_step_rad * roll_index
                roll_delta = _quat_from_rotvec(mul(roll_axis, roll_angle))
                candidate = normalize_quat(quat_mul(roll_delta, base_orientation))
                alignment_cost = _axis_orientation_cost_for_vector(
                    orientation=candidate,
                    axis=axis,
                    vector_tool=source_vector,
                )
                geometry_cost = _tool_geometry_displacement_cost(
                    candidate_orientation=candidate,
                    reference_orientation=seed,
                    sample_points_tool=sample_points_tool,
                )
                line_proximity_cost = _axis_line_proximity_cost(
                    candidate_orientation=candidate,
                    current_position=current_position,
                    axis_world=target,
                    line_proximity_target=line_proximity_target,
                )
                table_clearance_violation_m, candidate_combo_cost = _orientation_stage_candidate_combo_cost(
                    candidate_orientation=candidate,
                    current_position=current_position,
                    stage=_stage_with_axis_candidate_vector(
                        stage=stage,
                        constraint_id=axis.constraint_id,
                        vector_tool=source_vector,
                    ),
                    line_proximity_target=None,
                    table_clearance_context=table_clearance_context,
                )
                feasibility_rank = (
                    0
                    if table_clearance_violation_m <= _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M
                    else 1
                )
                rotation = quat_angle_rad(candidate, seed)
                # Prefer support-feasible rotations, but do not make the
                # orientation pre-selection itself a hard failure. The final
                # stage pose and move-along waypoints are still projected onto
                # support feasibility below; if every aligned-roll candidate is
                # infeasible at this pre-selection step, choose the
                # least-violating one instead of aborting compilation.
                if line_proximity_target:
                    score_key = (
                        float(feasibility_rank),
                        float(table_clearance_violation_m),
                        float(alignment_cost),
                        float(geometry_cost),
                        float(rotation * rotation),
                        float(line_proximity_cost),
                        float(candidate_combo_cost),
                    )
                else:
                    score_key = (
                        float(feasibility_rank),
                        float(table_clearance_violation_m),
                        float(alignment_cost),
                        float(geometry_cost),
                        float(rotation * rotation),
                        float(candidate_combo_cost),
                    )
                if best_score is None or score_key < best_score:
                    best_score = score_key
                    best_orientation = candidate
                    best_polarity = candidate_polarity
                    best_source_candidate = source_candidate
                    best_rotation = rotation
                    best_cosine = dot(unit(quat_rotate(candidate, source_vector)), target)
                    best_roll = roll_angle
                    best_geometry_cost = geometry_cost
                    best_line_proximity_cost = line_proximity_cost
                    best_table_clearance_violation_m = table_clearance_violation_m
                    best_candidate_combo_cost = candidate_combo_cost
    selection_record = _selected_axis_candidate_record(
        axis_candidates=axis_candidates,
        selected_candidate=best_source_candidate,
    )
    selection_record.update(
        {
            "axis_roll_resolution": "minimize_tool_geometry_displacement",
            "orientation_cost_sample_mode": "tool_geometry_sample_points"
            if sample_points_tool
            else "fallback_so3_angle",
            "orientation_cost_num_samples": len(sample_points_tool),
            "orientation_geometry_displacement_m2": best_geometry_cost,
            "axis_line_proximity_m2": best_line_proximity_cost,
            "orientation_table_clearance_violation_m": best_table_clearance_violation_m,
            "orientation_candidate_combo_cost": best_candidate_combo_cost,
            "candidate_selection_scope": "joint_axis_roll_and_vlm_contact_point_candidates",
            "axis_roll_resolution_priority": (
                "prefer_table_clearance_then_axis_alignment_tool_geometry_axis_line_distance_contact_target"
                if table_clearance_context and line_proximity_target
                else "prefer_table_clearance_then_axis_alignment_tool_geometry_contact_target"
                if table_clearance_context
                else "axis_alignment_tool_geometry_axis_line_distance_contact_target"
                if line_proximity_target
                else "axis_alignment_tool_geometry_contact_target"
            ),
            "axis_roll_rad": best_roll,
            "orientation_support_fallback_used": bool(
                best_table_clearance_violation_m > _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M
            ),
        }
    )
    return (
        best_orientation,
        best_polarity,
        best_rotation,
        best_cosine,
        selection_record,
    )


def _axis_target_candidates(axis: Any) -> list[tuple[str, list[float]]]:
    world = unit(axis.vector_world)
    if axis.polarity == "opposite":
        return [("opposite", [-x for x in world])]
    if axis.polarity == "auto":
        return [("same", world)]
    return [("same", world)]


def _axis_orientation_cost(
    *,
    orientation: list[float],
    axis: Any,
) -> float:
    return _axis_orientation_cost_for_vector(
        orientation=orientation,
        axis=axis,
        vector_tool=axis.vector_tool,
    )


def _axis_orientation_cost_for_vector(
    *,
    orientation: list[float],
    axis: Any,
    vector_tool: list[float],
) -> float:
    predicted = unit(quat_rotate(orientation, vector_tool))
    world = unit(axis.vector_world)
    cosine = dot(predicted, world)
    if axis.polarity == "opposite":
        return 1.0 + cosine
    if axis.polarity == "auto":
        return 1.0 - cosine
    return 1.0 - cosine


def _axis_final_resolution(
    *,
    orientation: list[float],
    axis: Any,
    rotation_delta_rad: float,
) -> dict[str, Any]:
    return _axis_final_resolution_for_vector(
        orientation=orientation,
        axis=axis,
        vector_tool=axis.vector_tool,
        rotation_delta_rad=rotation_delta_rad,
    )


def _axis_final_resolution_for_vector(
    *,
    orientation: list[float],
    axis: Any,
    vector_tool: list[float],
    rotation_delta_rad: float,
) -> dict[str, Any]:
    predicted = unit(quat_rotate(orientation, vector_tool))
    world = unit(axis.vector_world)
    raw_cosine = dot(predicted, world)
    if axis.polarity == "opposite":
        resolved_polarity = "opposite"
        cosine = -raw_cosine
    elif axis.polarity == "auto":
        resolved_polarity = "same"
        cosine = raw_cosine
    else:
        resolved_polarity = "same"
        cosine = raw_cosine
    return {
        "constraint_id": axis.constraint_id,
        "source_ref": axis.source_ref,
        "target_ref": axis.target_ref,
        "requested_polarity": axis.polarity,
        "resolved_polarity": resolved_polarity,
        "axis_rotation_delta_rad": rotation_delta_rad,
        "axis_cosine_after_resolution": cosine,
    }


def _orientation_from_coupled_axes(
    *,
    current_orientation: list[float],
    stage: CanonicalStageConstraints,
    current_position: list[float],
    line_proximity_target: dict[str, Any] | None = None,
    table_clearance_context: dict[str, Any] | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Resolve primary axis alignment while using secondary axes to choose roll.

    The first declared axis is treated as the task-critical functional axis.
    Roll about that aligned primary axis is still free, so secondary declared
    axes pick the most natural/table-compatible orientation without adding a
    tool-specific branch.
    """

    seed = normalize_quat(current_orientation)
    primary = stage.axis_targets[0]
    primary_candidates = _axis_source_vector_candidates(primary)
    best_orientation: list[float] | None = None
    best_primary_candidate: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    best_rotation = float("inf")
    best_roll = 0.0
    best_primary_polarity = primary.polarity
    best_line_proximity_cost = 0.0
    best_table_clearance_violation_m = 0.0
    best_candidate_combo_cost = 0.0
    for primary_candidate in primary_candidates:
        primary_vector = primary_candidate["vector_tool"]
        primary_current = unit(quat_rotate(seed, primary_vector))
        for primary_polarity, primary_target in _axis_target_candidates(primary):
            base_delta = quat_between_vectors(primary_current, primary_target)
            base_orientation = normalize_quat(quat_mul(base_delta, seed))
            roll_axis = unit(primary_target)
            for roll_index in range(-90, 91):
                roll_angle = math.radians(2.0 * roll_index)
                roll_delta = _quat_from_rotvec(mul(roll_axis, roll_angle))
                candidate = normalize_quat(quat_mul(roll_delta, base_orientation))
                axis_cost = 0.0
                for axis_index, axis in enumerate(stage.axis_targets):
                    source_vector = primary_vector if axis_index == 0 else axis.vector_tool
                    cost = _axis_orientation_cost_for_vector(
                        orientation=candidate,
                        axis=axis,
                        vector_tool=source_vector,
                    )
                    axis_cost += float(axis.weight) * cost * cost
                line_proximity_cost = _axis_line_proximity_cost(
                    candidate_orientation=candidate,
                    current_position=current_position,
                    axis_world=primary_target,
                    line_proximity_target=line_proximity_target,
                )
                table_clearance_violation_m, candidate_combo_cost = _orientation_stage_candidate_combo_cost(
                    candidate_orientation=candidate,
                    current_position=current_position,
                    stage=_stage_with_axis_candidate_vector(
                        stage=stage,
                        constraint_id=primary.constraint_id,
                        vector_tool=primary_vector,
                    ),
                    line_proximity_target=None,
                    table_clearance_context=table_clearance_context,
                )
                feasibility_rank = (
                    0
                    if table_clearance_violation_m <= _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M
                    else 1
                )
                rotation = quat_angle_rad(candidate, seed)
                if line_proximity_target:
                    score_key = (
                        float(feasibility_rank),
                        float(table_clearance_violation_m),
                        float(axis_cost),
                        float(line_proximity_cost),
                        float(candidate_combo_cost),
                        float(rotation * rotation),
                    )
                else:
                    score_key = (
                        float(feasibility_rank),
                        float(table_clearance_violation_m),
                        float(axis_cost),
                        float(rotation * rotation),
                        float(candidate_combo_cost),
                    )
                if best_score is None or score_key < best_score:
                    best_score = score_key
                    best_orientation = candidate
                    best_primary_candidate = primary_candidate
                    best_rotation = rotation
                    best_roll = roll_angle
                    best_primary_polarity = primary_polarity
                    best_line_proximity_cost = line_proximity_cost
                    best_table_clearance_violation_m = table_clearance_violation_m
                    best_candidate_combo_cost = candidate_combo_cost
    if best_orientation is None:
        raise SE3OptimizerError(f"{primary.constraint_id} has no feasible coupled-axis orientation candidate")
    resolutions = [
        _axis_final_resolution_for_vector(
            orientation=best_orientation,
            axis=axis,
            vector_tool=best_primary_candidate["vector_tool"] if index == 0 and best_primary_candidate else axis.vector_tool,
            rotation_delta_rad=best_rotation,
        )
        for index, axis in enumerate(stage.axis_targets)
    ]
    if resolutions:
        resolutions[0]["resolved_polarity"] = best_primary_polarity
        resolutions[0]["coupled_axis_roll_rad"] = best_roll
        resolutions[0]["coupled_axis_count"] = len(stage.axis_targets)
        resolutions[0]["coupled_axis_objective"] = "preserve primary axis while minimizing secondary axis alignment cost"
        if line_proximity_target:
            resolutions[0]["axis_line_proximity_objective"] = (
                "minimize_primary_axis_line_distance_to_next_contact_direction"
            )
        resolutions[0]["axis_line_proximity_m2"] = best_line_proximity_cost
        resolutions[0]["orientation_table_clearance_violation_m"] = best_table_clearance_violation_m
        resolutions[0]["orientation_candidate_combo_cost"] = best_candidate_combo_cost
        resolutions[0]["orientation_support_fallback_used"] = bool(
            best_table_clearance_violation_m > _TABLE_CLEARANCE_FEASIBILITY_TOLERANCE_M
        )
        resolutions[0]["candidate_selection_scope"] = "joint_axis_roll_and_vlm_contact_point_candidates"
        if table_clearance_context:
            resolutions[0]["coupled_axis_roll_resolution_priority"] = (
                "prefer_table_clearance_then_axis_alignment_rotation_axis_line_distance_candidate_combo"
            )
        elif line_proximity_target:
            resolutions[0]["coupled_axis_roll_resolution_priority"] = (
                "axis_alignment_rotation_axis_line_distance_candidate_combo"
            )
        resolutions[0].update(
            _selected_axis_candidate_record(
                axis_candidates=primary_candidates,
                selected_candidate=best_primary_candidate,
            )
        )
    return normalize_quat(best_orientation), resolutions


def apply_clearance_constraints(
    *,
    pose: dict[str, Any],
    clearance_targets: list[ClearanceTarget],
) -> dict[str, Any]:
    position = list(pose["position"])
    orientation = list(pose["orientation_xyzw"])
    for target in clearance_targets:
        moving_world = add(position, quat_rotate(orientation, target.point_tool))
        delta = sub(moving_world, target.obstacle_world)
        distance = norm(delta)
        if distance >= target.min_distance_m:
            continue
        direction = unit(delta) if distance > 1e-9 else [0.0, 0.0, 1.0]
        position = add(position, mul(direction, target.min_distance_m - distance))
    return {"position": position, "orientation_xyzw": orientation}


def stage_constraint_errors(
    *,
    pose: dict[str, Any],
    stage: CanonicalStageConstraints,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for target in stage.point_targets:
        errors[f"{target.constraint_id}.point_error_m"] = point_tracking_error_m(
            pose,
            target.point_tool,
            target.point_world,
        )
        if target.pre_contact_axis_world is not None and target.pre_contact_target_world is not None:
            offset = sub(target.point_world, target.pre_contact_target_world)
            axis = unit(target.pre_contact_axis_world)
            signed_offset = dot(offset, axis)
            errors[f"{target.constraint_id}.pre_contact_signed_offset_m"] = signed_offset
            errors[f"{target.constraint_id}.pre_contact_offset_norm_m"] = norm(offset)
    for target in stage.axis_targets:
        cost = axis_alignment_cost(
            pose,
            target.vector_tool,
            target.vector_world,
            polarity=target.polarity,
        )
        predicted = unit(quat_rotate(pose["orientation_xyzw"], target.vector_tool))
        world = unit(target.vector_world)
        cosine = dot(predicted, world)
        if target.polarity == "opposite":
            cosine = -cosine
        elif target.polarity == "auto":
            cosine = cosine
        errors[f"{target.constraint_id}.axis_cost"] = cost
        errors[f"{target.constraint_id}.axis_cosine"] = cosine
    for target in stage.clearance_targets:
        errors[f"{target.constraint_id}.clearance_margin_m"] = clearance_margin_m(
            pose,
            target.point_tool,
            target.obstacle_world,
            target.min_distance_m,
        )
    for target in stage.tool_occupancy_clearance_targets:
        errors[f"{target.constraint_id}.tool_occupancy_clearance_margin_m"] = (
            _tool_occupancy_clearance_margin_m(pose=pose, target=target)
        )
    return errors


def _warnings_from_errors(stage: CanonicalStageConstraints, errors: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    for target in stage.point_targets:
        error = errors.get(f"{target.constraint_id}.point_error_m", 0.0)
        if target.hard and target.tolerance_m > 0.0 and error > target.tolerance_m:
            warnings.append(
                f"{target.constraint_id} point error {error:.4f}m exceeds tolerance {target.tolerance_m:.4f}m"
            )
    for target in stage.axis_targets:
        cosine = errors.get(f"{target.constraint_id}.axis_cosine", 1.0)
        if target.hard and cosine < target.min_cosine:
            warnings.append(
                f"{target.constraint_id} axis cosine {cosine:.4f} below minimum {target.min_cosine:.4f}"
            )
    for target in stage.clearance_targets:
        margin = errors.get(f"{target.constraint_id}.clearance_margin_m", 0.0)
        if target.hard and margin < 0.0:
            warnings.append(f"{target.constraint_id} clearance margin {margin:.4f}m is negative")
    for target in stage.tool_occupancy_clearance_targets:
        margin = errors.get(f"{target.constraint_id}.tool_occupancy_clearance_margin_m", 0.0)
        if target.hard and margin < 0.0:
            warnings.append(
                f"{target.constraint_id} tool occupancy clearance margin {margin:.4f}m is negative"
            )
    return warnings


def _quat_from_rotvec(rotvec: list[float]) -> list[float]:
    angle = norm(rotvec)
    if angle <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    axis = mul(rotvec, 1.0 / angle)
    half = 0.5 * angle
    scale = math.sin(half)
    return normalize_quat([axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half)])


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _fallback_perpendicular(axis: list[float]) -> list[float]:
    candidate = _cross(axis, [1.0, 0.0, 0.0])
    if norm(candidate) <= 1e-9:
        candidate = _cross(axis, [0.0, 1.0, 0.0])
    return unit(candidate)


def _swing_arc_point(
    *,
    move: MoveAlongTarget,
    swing_axis_world: list[float] | None,
    alpha: float,
) -> list[float]:
    linear = [
        move.start_world[i] + alpha * (move.goal_world[i] - move.start_world[i])
        for i in range(3)
    ]
    if swing_axis_world is None:
        return linear
    chord = sub(move.start_world, move.goal_world)
    chord_len = norm(chord)
    if chord_len <= 1e-9:
        return linear
    swing_axis = unit(swing_axis_world)
    arc_direction = _cross(swing_axis, chord)
    if norm(arc_direction) <= 1e-9:
        arc_direction = _fallback_perpendicular(swing_axis)
    else:
        arc_direction = unit(arc_direction)
    bulge = min(float(move.swing_lift_m), 0.75 * chord_len) * math.sin(math.pi * alpha)
    return add(linear, mul(arc_direction, bulge))


def _resolved_swing_axis_world(
    *,
    move: MoveAlongTarget,
    orientation_xyzw: list[float],
) -> list[float] | None:
    if move.swing_axis_world is not None:
        return unit(move.swing_axis_world)
    if move.swing_axis_tool is not None:
        return unit(quat_rotate(orientation_xyzw, move.swing_axis_tool))
    return None


def _select_move_along_point_candidate(
    *,
    move: MoveAlongTarget,
    orientation_xyzw: list[float],
    current_pose: dict[str, Any],
    table_clearance_context: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, Any] | None]:
    # Action must continue with the same active contact point that reached the
    # pre-contact/start relation. Do not switch to another candidate during the
    # path, even if a symmetric candidate would reduce a local translation cost.
    return [float(value) for value in move.point_tool], None


def _move_along_axis_world(move: MoveAlongTarget) -> list[float] | None:
    path = sub(move.goal_world, move.start_world)
    if norm(path) <= 1e-9:
        return None
    return unit(path)


def _start_point_is_on_support_plane(
    *,
    move: MoveAlongTarget,
    table_clearance_context: dict[str, Any] | None,
    activation_distance_m: float = 0.025,
) -> bool:
    if not table_clearance_context:
        return False
    plane = _support_clearance_plane(table_clearance_context)
    if plane is None:
        return False
    plane_point_world, plane_normal_world = plane
    clearance_m = float(table_clearance_context.get("clearance_m", 0.0))
    signed = dot(sub(move.start_world, plane_point_world), plane_normal_world)
    return abs(float(signed) - clearance_m) <= activation_distance_m


def _axis_preservation_cost(
    *,
    orientation: list[float],
    axis_targets: list[Any] | None,
) -> float:
    if not axis_targets:
        return 0.0
    cost = 0.0
    for axis in axis_targets:
        predicted = unit(quat_rotate(orientation, axis.vector_tool))
        world = unit(axis.vector_world)
        cosine = dot(predicted, world)
        if axis.polarity == "opposite":
            cosine = -cosine
        cost += float(getattr(axis, "weight", 1.0)) * float((1.0 - cosine) * (1.0 - cosine))
    return float(cost)


def _axis_preservation_satisfied(
    *,
    orientation: list[float],
    axis_targets: list[Any] | None,
) -> bool:
    if not axis_targets:
        return True
    for axis in axis_targets:
        predicted = unit(quat_rotate(orientation, axis.vector_tool))
        world = unit(axis.vector_world)
        cosine = dot(predicted, world)
        if axis.polarity == "opposite":
            cosine = -cosine
        min_cosine = float(getattr(axis, "min_cosine", 0.995))
        if cosine < min_cosine:
            return False
    return True


def _roll_move_along_approach_orientation(
    *,
    move: MoveAlongTarget,
    point_tool: list[float],
    approach_point_world: list[float],
    base_orientation_xyzw: list[float],
    table_clearance_context: dict[str, Any] | None,
    axis_preservation_targets: list[Any] | None = None,
    tool_occupancy_clearance_targets: list[Any] | None = None,
) -> list[float]:
    """Roll a contact-follow approach around the selected action axis.

    The action axis is already aligned by the previous tool_axis_align stage.
    Rolling about the corresponding world direction preserves that alignment
    while allowing the approach waypoint to bring the selected contact geometry
    into a more contact-ready relation with the support surface.
    """

    if move.profile != "linear" or move.path_family not in {"contact_follow", "slide"}:
        return normalize_quat(base_orientation_xyzw)
    axis_world = _move_along_axis_world(move)
    if axis_world is None:
        return normalize_quat(base_orientation_xyzw)
    plane = _support_clearance_plane(table_clearance_context) if table_clearance_context else None
    bbox_points_tool = (table_clearance_context or {}).get("bbox_points_tool") or []
    if plane is None or not bbox_points_tool:
        return normalize_quat(base_orientation_xyzw)
    base_orientation = normalize_quat(base_orientation_xyzw)
    best_orientation = base_orientation
    best_score: tuple[float, ...] | None = None
    for roll_index in range(-90, 91):
        roll_angle = math.radians(2.0 * roll_index)
        roll_delta = _quat_from_rotvec(mul(axis_world, roll_angle))
        candidate_orientation = normalize_quat(quat_mul(roll_delta, base_orientation))
        rotation_delta_rad = quat_angle_rad(candidate_orientation, base_orientation)
        candidate_pose = pose_for_tool_point_goal(
            orientation_xyzw=candidate_orientation,
            point_tool=point_tool,
            point_world=approach_point_world,
        )
        violation_m = _pose_table_clearance_violation_m(
            pose=candidate_pose,
            table_clearance_context=table_clearance_context,
        )
        occupancy_violation_m = _tool_occupancy_clearance_violation_m(
            pose=candidate_pose,
            targets=tool_occupancy_clearance_targets,
        )
        geometry_cost = _tool_geometry_displacement_cost(
            candidate_orientation=candidate_orientation,
            reference_orientation=base_orientation,
            sample_points_tool=bbox_points_tool,
        )
        axis_cost = _axis_preservation_cost(
            orientation=candidate_orientation,
            axis_targets=axis_preservation_targets,
        )
        if axis_cost > 1e-8 and not _axis_preservation_satisfied(
            orientation=candidate_orientation,
            axis_targets=axis_preservation_targets,
        ):
            continue
        score = (
            float(axis_cost),
            float(1e6 * violation_m * violation_m),
            float(1e6 * occupancy_violation_m * occupancy_violation_m),
            float(rotation_delta_rad * rotation_delta_rad),
            float(geometry_cost),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_orientation = candidate_orientation
    return normalize_quat(best_orientation)


def optimize_move_along_approach_pose(
    *,
    move: MoveAlongTarget,
    pose: dict[str, Any],
    point_tool: list[float],
    approach_point_world: list[float],
    table_clearance_context: dict[str, Any] | None = None,
    axis_preservation_targets: list[Any] | None = None,
    tool_occupancy_clearance_targets: list[Any] | None = None,
) -> dict[str, Any]:
    """Optimize the pre-action approach pose by rolling around the action axis.

    This keeps the already selected action direction fixed while allowing the
    approach stage to rotate the tool into a contact-ready roll before the
    move_along action starts.
    """

    orientation = _roll_move_along_approach_orientation(
        move=move,
        point_tool=point_tool,
        approach_point_world=approach_point_world,
        base_orientation_xyzw=pose["orientation_xyzw"],
        table_clearance_context=table_clearance_context,
        axis_preservation_targets=axis_preservation_targets,
        tool_occupancy_clearance_targets=tool_occupancy_clearance_targets,
    )
    optimized = pose_for_tool_point_goal(
        orientation_xyzw=orientation,
        point_tool=point_tool,
        point_world=approach_point_world,
    )
    optimized = _project_pose_to_table_feasible(
        pose=optimized,
        table_clearance_context=table_clearance_context,
    )
    _require_pose_table_clearance(
        pose=optimized,
        table_clearance_context=table_clearance_context,
        label=f"{move.source_ref} move-along approach",
    )
    return optimized


def _move_waypoint_semantics(
    *,
    move: MoveAlongTarget,
    step: int,
    intervals: int,
) -> tuple[str, bool, str]:
    family = move.path_family
    if step >= intervals:
        return f"{family}_terminal", True, move.goal_ref
    if family == "swing":
        role = "motion_swing_windup" if step == 0 else "motion_swing_arc"
    elif family == "slide":
        role = "motion_slide_start" if step == 0 else "motion_slide_path"
    else:
        role = "contact_point_approach" if step == 0 else "contact_point_follow_path"
    target_ref = move.start_ref if step == 0 else move.goal_ref
    return role, False, target_ref


def _pose_from_variables(
    *,
    variables: list[float],
    base_orientation_xyzw: list[float],
) -> dict[str, Any]:
    delta = _quat_from_rotvec([variables[3], variables[4], variables[5]])
    return {
        "position": [float(variables[0]), float(variables[1]), float(variables[2])],
        "orientation_xyzw": normalize_quat(quat_mul(delta, base_orientation_xyzw)),
    }


def _objective_search_cost(
    *,
    pose: dict[str, Any],
    seed_pose: dict[str, Any],
    stage: CanonicalStageConstraints,
    objective_cost_fn: Callable[[dict[str, Any]], float],
) -> float:
    cost = stage_cost(
        pose=pose,
        point_targets=stage.point_targets,
        axis_targets=stage.axis_targets,
        clearance_targets=stage.clearance_targets,
    )
    cost += float(objective_cost_fn(pose))
    translation_from_seed = norm(sub(pose["position"], seed_pose["position"]))
    rotation_from_seed = quat_angle_rad(pose["orientation_xyzw"], seed_pose["orientation_xyzw"])
    cost += 1e-4 * translation_from_seed * translation_from_seed
    cost += 1e-4 * rotation_from_seed * rotation_from_seed
    return float(cost)


def _stage_with_selected_axis_vectors(
    *,
    stage: CanonicalStageConstraints,
    axis_resolutions: list[dict[str, Any]],
) -> CanonicalStageConstraints:
    selected_by_id: dict[str, list[float]] = {}
    for resolution in axis_resolutions:
        vector = resolution.get("selected_source_axis_tool")
        if isinstance(vector, list) and len(vector) == 3:
            selected_by_id[str(resolution.get("constraint_id"))] = unit([float(item) for item in vector])
    if not selected_by_id:
        return stage
    axis_targets = [
        replace(axis, vector_tool=selected_by_id.get(axis.constraint_id, axis.vector_tool))
        for axis in stage.axis_targets
    ]
    return replace(stage, axis_targets=axis_targets)


def _optimize_with_objectives(
    *,
    seed_pose: dict[str, Any],
    stage: CanonicalStageConstraints,
    objective_cost_fn: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    base_orientation = normalize_quat(seed_pose["orientation_xyzw"])
    variables = [float(x) for x in seed_pose["position"]] + [0.0, 0.0, 0.0]

    def to_pose(candidate: list[float]) -> dict[str, Any]:
        return _pose_from_variables(
            variables=candidate,
            base_orientation_xyzw=base_orientation,
        )

    best = list(variables)
    best_pose = to_pose(best)
    best_cost = _objective_search_cost(
        pose=best_pose,
        seed_pose=seed_pose,
        stage=stage,
        objective_cost_fn=objective_cost_fn,
    )
    steps = [0.08, 0.08, 0.08, 0.35, 0.35, 0.35]
    for _ in range(8):
        improved = True
        sweeps = 0
        while improved and sweeps < 16:
            improved = False
            sweeps += 1
            for index, step in enumerate(steps):
                for direction in (-1.0, 1.0):
                    candidate = list(best)
                    candidate[index] += direction * step
                    pose = to_pose(candidate)
                    cost = _objective_search_cost(
                        pose=pose,
                        seed_pose=seed_pose,
                        stage=stage,
                        objective_cost_fn=objective_cost_fn,
                    )
                    if cost + 1e-12 < best_cost:
                        best = candidate
                        best_pose = pose
                        best_cost = cost
                        improved = True
        steps = [step * 0.5 for step in steps]
    return best_pose


def _move_requires_orientation_change(move: MoveAlongTarget) -> bool:
    """Return whether a move-along path is itself an orientation-changing action."""

    return move.path_family == "swing" or move.profile == "swing_arc"


def _stage_preserves_execution_orientation(stage: CanonicalStageConstraints) -> bool:
    """Keep straight contact-follow actions in the already chosen execution pose.

    Axis alignment is responsible for choosing the execution orientation. Once the
    contact point is active, approach/action stages should translate that same
    tool pose unless the motion primitive explicitly encodes a rotating action
    such as a swing arc.
    """

    if not stage.point_targets or not stage.move_along_targets:
        return False
    return all(not _move_requires_orientation_change(move) for move in stage.move_along_targets)


def solve_stage_pose(
    *,
    stage: CanonicalStageConstraints,
    current_pose: dict[str, Any],
    default_tool_point_ref: str,
    objective_cost_fn: Callable[[dict[str, Any]], float] | None = None,
    objective_error_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    line_proximity_target: dict[str, Any] | None = None,
    table_clearance_context: dict[str, Any] | None = None,
) -> StageSolveResult:
    preserves_execution_orientation = _stage_preserves_execution_orientation(stage)
    if preserves_execution_orientation:
        orientation = normalize_quat(current_pose["orientation_xyzw"])
        axis_resolutions = [
            {
                "constraint_id": axis.constraint_id,
                "source_ref": axis.source_ref,
                "target_ref": axis.target_ref,
                "orientation_locked_to_execution_pose": True,
                "orientation_lock_reason": "straight_move_along_preserves_align_orientation",
            }
            for axis in stage.axis_targets
        ]
    else:
        orientation, axis_resolutions = _orientation_from_axes(
            current_orientation=current_pose["orientation_xyzw"],
            stage=stage,
            current_position=list(current_pose["position"]),
            line_proximity_target=line_proximity_target,
            table_clearance_context=table_clearance_context,
        )
    effective_stage = _stage_with_selected_axis_vectors(
        stage=stage,
        axis_resolutions=axis_resolutions,
    )
    effective_stage, point_resolutions = _select_point_candidates_for_stage(
        stage=effective_stage,
        orientation_xyzw=orientation,
        current_pose=current_pose,
        objective_cost_fn=objective_cost_fn,
        table_clearance_context=table_clearance_context,
    )
    primary_tool_point: list[float] | None = None
    if effective_stage.point_targets:
        pose = _pose_from_point_targets(
            orientation_xyzw=orientation,
            point_targets=effective_stage.point_targets,
        )
        primary_tool_ref = effective_stage.point_targets[0].source_ref
        primary_target_ref = effective_stage.point_targets[0].target_ref
        primary_tool_point = list(effective_stage.point_targets[0].point_tool)
    elif effective_stage.axis_targets:
        pose = {
            "position": list(current_pose["position"]),
            "orientation_xyzw": orientation,
        }
        primary_tool_ref = default_tool_point_ref
        primary_target_ref = effective_stage.axis_targets[0].target_ref
    else:
        pose = {
            "position": list(current_pose["position"]),
            "orientation_xyzw": list(current_pose["orientation_xyzw"]),
        }
        primary_tool_ref = default_tool_point_ref
        primary_target_ref = f"stage.{stage.stage_name}"

    pose = apply_clearance_constraints(
        pose=pose,
        clearance_targets=effective_stage.clearance_targets,
    )
    pose = _project_pose_to_table_feasible(
        pose=pose,
        table_clearance_context=table_clearance_context,
    )
    _require_pose_table_clearance(
        pose=pose,
        table_clearance_context=table_clearance_context,
        label=f"stage {stage.stage_name}",
    )
    if objective_cost_fn is not None:
        try:
            pose = _optimize_with_objectives(
                seed_pose=pose,
                stage=effective_stage,
                objective_cost_fn=objective_cost_fn,
            )
        except Exception as exc:  # noqa: BLE001 - surface objective failures as compiler errors.
            raise SE3OptimizerError(f"objective optimization failed for stage {stage.stage_name}: {exc}") from exc
        pose = apply_clearance_constraints(
            pose=pose,
            clearance_targets=effective_stage.clearance_targets,
        )
        pose = _project_pose_to_table_feasible(
            pose=pose,
            table_clearance_context=table_clearance_context,
        )
        _require_pose_table_clearance(
            pose=pose,
            table_clearance_context=table_clearance_context,
            label=f"stage {stage.stage_name} objective pose",
        )
    errors = stage_constraint_errors(pose=pose, stage=effective_stage)
    objective_cost = 0.0
    if objective_cost_fn is not None:
        objective_cost = float(objective_cost_fn(pose))
    if objective_error_fn is not None:
        errors.update(objective_error_fn(pose))
    return StageSolveResult(
        pose=pose,
        primary_tool_ref=primary_tool_ref,
        primary_target_ref=primary_target_ref,
        final_cost=stage_cost(
            pose=pose,
            point_targets=effective_stage.point_targets,
            axis_targets=effective_stage.axis_targets,
            clearance_targets=effective_stage.clearance_targets,
        )
        + objective_cost,
        constraint_errors=errors,
        pose_selection={
            "algorithm": "constrained_minimum_change_v0",
            "objective": "minimize pose change from previous subgoal subject to canonical point/axis/clearance constraints",
            "axis_resolutions": axis_resolutions,
            "point_resolutions": point_resolutions,
        },
        warnings=_warnings_from_errors(effective_stage, errors),
        primary_tool_point=primary_tool_point,
    )


def compile_move_along_waypoints(
    *,
    move: MoveAlongTarget,
    orientation_xyzw: list[float],
    current_pose: dict[str, Any],
    steps: int,
    table_clearance_context: dict[str, Any] | None = None,
) -> list[CompiledMoveWaypoint]:
    compiled: list[CompiledMoveWaypoint] = []
    if _move_requires_orientation_change(move):
        final_orientation = normalize_quat(orientation_xyzw)
    else:
        final_orientation = normalize_quat(current_pose["orientation_xyzw"])
    point_tool, _point_selection = _select_move_along_point_candidate(
        move=move,
        orientation_xyzw=final_orientation,
        current_pose=current_pose,
        table_clearance_context=table_clearance_context,
    )
    if norm(sub(move.goal_world, move.start_world)) <= 1e-9:
        terminal_pose = pose_for_tool_point_goal(
            orientation_xyzw=final_orientation,
            point_tool=point_tool,
            point_world=move.goal_world,
        )
        terminal_pose = _project_pose_to_table_feasible(
            pose=terminal_pose,
            table_clearance_context=table_clearance_context,
        )
        _require_pose_table_clearance(
            pose=terminal_pose,
            table_clearance_context=table_clearance_context,
            label=f"{move.source_ref} move-along terminal",
        )
        return [
            CompiledMoveWaypoint(
                pose=terminal_pose,
                tool_ref=move.source_ref,
                target_ref=move.goal_ref,
                semantic_role=f"{move.path_family}_terminal",
                semantic_gate=True,
                point_tool=point_tool,
            )
        ]
    intervals = max(steps, 1)
    if move.path_family == "swing":
        intervals = max(intervals, 4)
    elif move.path_family == "slide":
        intervals = max(intervals, 3)
    swing_axis_world = _resolved_swing_axis_world(move=move, orientation_xyzw=orientation_xyzw)
    start_orientation = final_orientation
    for step in range(0, intervals + 1):
        alpha = step / float(intervals)
        if move.profile == "swing_arc":
            point_world = _swing_arc_point(move=move, swing_axis_world=swing_axis_world, alpha=alpha)
        else:
            point_world = [
                move.start_world[i] + alpha * (move.goal_world[i] - move.start_world[i])
                for i in range(3)
            ]
        waypoint_orientation = quat_slerp(start_orientation, final_orientation, alpha)
        if move.profile == "swing_arc" and swing_axis_world is not None:
            swing_axis = unit(swing_axis_world)
            swing_angle = math.sin(math.pi * alpha) * float(move.swing_angle_rad)
            delta = _quat_from_rotvec(mul(swing_axis, swing_angle))
            waypoint_orientation = normalize_quat(quat_mul(delta, waypoint_orientation))
        pose = pose_for_tool_point_goal(
            orientation_xyzw=waypoint_orientation,
            point_tool=point_tool,
            point_world=point_world,
        )
        pose = _project_pose_to_table_feasible(
            pose=pose,
            table_clearance_context=table_clearance_context,
        )
        _require_pose_table_clearance(
            pose=pose,
            table_clearance_context=table_clearance_context,
            label=f"{move.source_ref} move-along waypoint {step}",
        )
        semantic_role, semantic_gate, target_ref = _move_waypoint_semantics(
            move=move,
            step=step,
            intervals=intervals,
        )
        compiled.append(
            CompiledMoveWaypoint(
                pose=pose,
                tool_ref=move.source_ref,
                target_ref=target_ref,
                semantic_role=semantic_role,
                semantic_gate=semantic_gate,
                point_tool=point_tool,
            )
        )
    if not compiled:
        return [
            CompiledMoveWaypoint(
                pose=current_pose,
                tool_ref=move.source_ref,
                target_ref=move.goal_ref,
                semantic_role=f"{move.path_family}_terminal",
                semantic_gate=True,
                point_tool=point_tool,
            )
        ]
    return compiled
