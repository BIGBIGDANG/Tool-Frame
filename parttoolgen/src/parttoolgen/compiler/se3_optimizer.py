#!/usr/bin/env python3
"""Stage-wise SE(3) optimizer for PartToolGen whole-tool pose subgoals."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, NamedTuple

from parttoolgen.compiler.constraint_canonicalizer import (
    CanonicalStageConstraints,
    ClearanceTarget,
    MoveAlongTarget,
)
from parttoolgen.compiler.geometric_solvers import quat_between_vectors, quat_mul, quat_rotate
from parttoolgen.compiler.residuals import (
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
) -> tuple[list[float], list[dict[str, Any]]]:
    if len(stage.axis_targets) > 1:
        return _orientation_from_coupled_axes(
            current_orientation=current_orientation,
            stage=stage,
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


def _axis_source_vector_candidates(axis: Any) -> list[dict[str, Any]]:
    raw_candidates = getattr(axis, "vector_tool_candidates", None) or []
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "primary_axis",
            "vector_tool": unit(axis.vector_tool),
            "source": "axis.vector_tool",
        }
    ]
    if isinstance(raw_candidates, list):
        for index, raw in enumerate(raw_candidates):
            if isinstance(raw, dict):
                vector = raw.get("vector_tool")
                if vector is None:
                    continue
                candidate = dict(raw)
                candidate["vector_tool"] = unit([float(item) for item in vector])
                candidate["source_candidate_index"] = index
            elif isinstance(raw, list) and len(raw) == 3:
                candidate = {
                    "candidate_id": f"axis_candidate_{index}",
                    "vector_tool": unit([float(item) for item in raw]),
                    "source": "axis.vector_tool_candidates",
                    "source_candidate_index": index,
                }
            else:
                continue
            candidates.append(candidate)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        vector = unit(candidate["vector_tool"])
        key = tuple(round(value, 6) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["vector_tool"] = list(key)
        row["candidate_index"] = len(deduped)
        deduped.append(row)
    return deduped


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


def _source_point_candidates(target: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "primary_contact_point",
            "point_tool": [float(item) for item in target.point_tool],
            "source": "target.point_tool",
        }
    ]
    raw_candidates = getattr(target, "point_tool_candidates", None) or []
    if isinstance(raw_candidates, list):
        for index, raw in enumerate(raw_candidates):
            if isinstance(raw, dict):
                point = (
                    raw.get("point_tool")
                    or raw.get("contact_point_tool")
                    or raw.get("xyz_tool")
                    or raw.get("center_xyz_tool")
                )
                if point is None:
                    continue
                candidate = dict(raw)
                candidate["point_tool"] = [float(item) for item in point]
                candidate["source_candidate_index"] = index
            elif isinstance(raw, list) and len(raw) == 3:
                candidate = {
                    "candidate_id": f"point_candidate_{index}",
                    "point_tool": [float(item) for item in raw],
                    "source": "target.point_tool_candidates",
                    "source_candidate_index": index,
                }
            else:
                continue
            candidates.append(candidate)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        point = [float(item) for item in candidate["point_tool"]]
        key = tuple(round(value, 6) for value in point)
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["point_tool"] = list(key)
        row["candidate_index"] = len(deduped)
        deduped.append(row)
    return deduped


def _selected_point_candidate_record(
    *,
    source_ref: str,
    point_candidates: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_ref": source_ref,
        "source_point_candidate_count": len(point_candidates),
        "selected_source_point_tool": list(selected_candidate["point_tool"]),
        "selected_source_point_candidate_index": int(selected_candidate.get("candidate_index", 0)),
    }
    optional_fields = {
        "candidate_id": "selected_source_point_candidate_id",
        "source_candidate_index": "selected_source_point_source_candidate_index",
        "source": "selected_source_point_candidate_source",
        "contact_point_face_marker": "selected_contact_point_face_marker",
        "contact_point_face_id": "selected_contact_point_face_id",
        "contact_point_source_ref": "selected_contact_point_source_ref",
        "contact_point_surface_ref": "selected_contact_point_surface_ref",
        "contact_point_part_ref": "selected_contact_point_part_ref",
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


def _stage_with_point_selection(
    *,
    stage: CanonicalStageConstraints,
    selections: dict[str, dict[str, Any]],
) -> CanonicalStageConstraints:
    point_targets = [
        replace(target, point_tool=list(selections[target.source_ref]["point_tool"]))
        if target.source_ref in selections
        else target
        for target in stage.point_targets
    ]
    return replace(stage, point_targets=point_targets)


def _select_point_candidates_for_stage(
    *,
    stage: CanonicalStageConstraints,
    orientation_xyzw: list[float],
    current_pose: dict[str, Any],
    objective_cost_fn: Callable[[dict[str, Any]], float] | None,
) -> tuple[CanonicalStageConstraints, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for target in stage.point_targets:
        candidates = _source_point_candidates(target)
        if len(candidates) > 1:
            groups.setdefault(target.source_ref, candidates)
    if not groups:
        return stage, []

    source_refs = sorted(groups)
    best_stage = stage
    best_selection: dict[str, dict[str, Any]] = {}
    best_cost = float("inf")
    for combination in product(*(groups[source_ref] for source_ref in source_refs)):
        selections = {source_ref: candidate for source_ref, candidate in zip(source_refs, combination)}
        candidate_stage = _stage_with_point_selection(stage=stage, selections=selections)
        pose = _pose_from_point_targets(
            orientation_xyzw=orientation_xyzw,
            point_targets=candidate_stage.point_targets,
        )
        pose = apply_clearance_constraints(
            pose=pose,
            clearance_targets=candidate_stage.clearance_targets,
        )
        translation_from_seed = norm(sub(pose["position"], current_pose["position"]))
        cost = stage_cost(
            pose=pose,
            point_targets=candidate_stage.point_targets,
            axis_targets=candidate_stage.axis_targets,
            clearance_targets=candidate_stage.clearance_targets,
        )
        cost += 1e-4 * translation_from_seed * translation_from_seed
        if objective_cost_fn is not None:
            cost += float(objective_cost_fn(pose))
        if cost + 1e-12 < best_cost:
            best_cost = float(cost)
            best_stage = candidate_stage
            best_selection = selections

    records = [
        _selected_point_candidate_record(
            source_ref=source_ref,
            point_candidates=groups[source_ref],
            selected_candidate=best_selection[source_ref],
        )
        for source_ref in source_refs
        if source_ref in best_selection
    ]
    return best_stage, records


def _orientation_for_single_axis_with_geometry_cost(
    *,
    current_orientation: list[float],
    axis: Any,
    sample_points_tool: list[list[float]],
) -> tuple[list[float] | None, str | None, float, float, dict[str, Any]]:
    """Align one axis and resolve free roll by minimizing tool geometry motion."""

    seed = normalize_quat(current_orientation)
    axis_candidates = _axis_source_vector_candidates(axis)
    best_orientation: list[float] | None = None
    best_polarity: str | None = None
    best_source_candidate: dict[str, Any] | None = None
    best_score = float("inf")
    best_rotation = float("inf")
    best_cosine = -1.0
    best_roll = 0.0
    best_geometry_cost = float("inf")
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
                rotation = quat_angle_rad(candidate, seed)
                # Alignment is the hard objective. Geometry displacement resolves
                # the residual roll ambiguity; SO(3) angle only breaks near ties.
                score = 1e3 * alignment_cost * alignment_cost + geometry_cost + 1e-8 * rotation * rotation
                if score + 1e-12 < best_score:
                    best_score = score
                    best_orientation = candidate
                    best_polarity = candidate_polarity
                    best_source_candidate = source_candidate
                    best_rotation = rotation
                    best_cosine = dot(unit(quat_rotate(candidate, source_vector)), target)
                    best_roll = roll_angle
                    best_geometry_cost = geometry_cost
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
            "axis_roll_rad": best_roll,
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
        return [("same", world), ("opposite", [-x for x in world])]
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
        return 1.0 - abs(cosine)
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
        resolved_polarity = "same" if raw_cosine >= 0.0 else "opposite"
        cosine = abs(raw_cosine)
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
    best_score = float("inf")
    best_rotation = float("inf")
    best_roll = 0.0
    best_primary_polarity = primary.polarity
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
                rotation = quat_angle_rad(candidate, seed)
                score = axis_cost + 1e-6 * rotation * rotation
                if score + 1e-12 < best_score:
                    best_score = score
                    best_orientation = candidate
                    best_primary_candidate = primary_candidate
                    best_rotation = rotation
                    best_roll = roll_angle
                    best_primary_polarity = primary_polarity
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
            cosine = abs(cosine)
        errors[f"{target.constraint_id}.axis_cost"] = cost
        errors[f"{target.constraint_id}.axis_cosine"] = cosine
    for target in stage.clearance_targets:
        errors[f"{target.constraint_id}.clearance_margin_m"] = clearance_margin_m(
            pose,
            target.point_tool,
            target.obstacle_world,
            target.min_distance_m,
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
) -> tuple[list[float], dict[str, Any] | None]:
    candidates = _source_point_candidates(move)
    if len(candidates) <= 1:
        return list(move.point_tool), None
    best_candidate = candidates[0]
    best_cost = float("inf")
    for candidate in candidates:
        point_tool = list(candidate["point_tool"])
        start_pose = pose_for_tool_point_goal(
            orientation_xyzw=orientation_xyzw,
            point_tool=point_tool,
            point_world=move.start_world,
        )
        goal_pose = pose_for_tool_point_goal(
            orientation_xyzw=orientation_xyzw,
            point_tool=point_tool,
            point_world=move.goal_world,
        )
        start_translation = norm(sub(start_pose["position"], current_pose["position"]))
        use_translation = norm(sub(goal_pose["position"], start_pose["position"]))
        # All candidates are VLM-valid; prefer the one that reaches the path
        # with the least free-space translation, then the shorter tool-base path.
        cost = start_translation * start_translation + 1e-6 * use_translation * use_translation
        if cost + 1e-12 < best_cost:
            best_candidate = candidate
            best_cost = float(cost)
    return (
        list(best_candidate["point_tool"]),
        _selected_point_candidate_record(
            source_ref=move.source_ref,
            point_candidates=candidates,
            selected_candidate=best_candidate,
        ),
    )


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


def solve_stage_pose(
    *,
    stage: CanonicalStageConstraints,
    current_pose: dict[str, Any],
    default_tool_point_ref: str,
    objective_cost_fn: Callable[[dict[str, Any]], float] | None = None,
    objective_error_fn: Callable[[dict[str, Any]], dict[str, float]] | None = None,
) -> StageSolveResult:
    orientation, axis_resolutions = _orientation_from_axes(
        current_orientation=current_pose["orientation_xyzw"],
        stage=stage,
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
) -> list[CompiledMoveWaypoint]:
    compiled: list[CompiledMoveWaypoint] = []
    final_orientation = normalize_quat(orientation_xyzw)
    point_tool, _point_selection = _select_move_along_point_candidate(
        move=move,
        orientation_xyzw=final_orientation,
        current_pose=current_pose,
    )
    if norm(sub(move.goal_world, move.start_world)) <= 1e-9:
        return [
            CompiledMoveWaypoint(
                pose=pose_for_tool_point_goal(
                    orientation_xyzw=final_orientation,
                    point_tool=point_tool,
                    point_world=move.goal_world,
                ),
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
    start_orientation = normalize_quat(current_pose["orientation_xyzw"])
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
