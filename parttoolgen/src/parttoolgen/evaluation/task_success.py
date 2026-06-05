"""Task-level success checks for image-grounded tool-use runners."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from parttoolgen.adapters.simtoolreal_policy_priors import (
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
)


HAMMER_DOWN_SUCCESS_POLICY = {
    "name": "hammer_down_semantic_contact",
    "task_type": "hammer_down_strike",
    "requires_support_down_direction": True,
    "requires_compile_path_solver": True,
    "requires_semantic_contact": True,
    "requires_tracking_execution_passed": False,
    "requires_policy_goal_passed": False,
    "requires_target_displacement": False,
}

BRUSH_SWEEP_SUCCESS_POLICY = {
    "name": "brush_sweep_directional_displacement",
    "task_type": "brush_sweep",
    "requires_target_effect": True,
    "requires_tracking_execution_passed": False,
    "requires_policy_goal_passed": False,
    "requires_semantic_execution_passed": False,
}

BRUSH_SWEEP_TARGET_ZONE_SUCCESS_POLICY = {
    "name": "brush_sweep_target_zone_occupancy",
    "task_type": "brush_sweep",
    "requires_target_region_occupancy": True,
    "requires_tracking_execution_passed": False,
    "requires_policy_goal_passed": False,
    "requires_semantic_execution_passed": False,
}

ERASER_WIPE_SUCCESS_POLICY = {
    "name": "eraser_wipe_contact_coverage",
    "task_type": "eraser_wipe",
    "requires_compile_path_solver": True,
    "requires_lifted_target_surface_path": True,
    "requires_contact_coverage": True,
    "min_contact_coverage_fraction": 0.30,
    "contact_coverage_is_task_success": True,
    "requires_policy_goal_passed": False,
    "requires_semantic_contact_path": False,
    "surface_contact_coverage_is_diagnostic": False,
    "requires_semantic_execution_passed": False,
    "requires_tracking_execution_passed": False,
    "requires_visual_patch_disappearance": False,
}

SCREWDRIVER_INSERT_SUCCESS_POLICY = {
    "name": "screwdriver_insert_tip_path",
    "task_type": "screwdriver_insert",
    "requires_compile_path_solver": True,
    "requires_semantic_execution_passed": True,
    "requires_tip_radial_alignment": True,
    "requires_tip_insertion_depth": True,
    "requires_tracking_execution_passed": False,
    "requires_policy_goal_passed": False,
}


def _predicate(
    name: str,
    passed: bool,
    *,
    required: bool = True,
    value: Any = None,
    threshold: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "required": bool(required),
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


def _semantic_execution_passed(
    rollout_summary: Mapping[str, Any] | None,
    execution_report: Mapping[str, Any] | None,
) -> bool | None:
    for report in (execution_report, rollout_summary):
        if isinstance(report, Mapping) and report.get("semantic_execution_passed") is not None:
            return bool(report.get("semantic_execution_passed"))
    return None


def _policy_goal_passed(rollout_summary: Mapping[str, Any] | None) -> bool | None:
    if isinstance(rollout_summary, Mapping) and rollout_summary.get("policy_goal_passed") is not None:
        return bool(rollout_summary.get("policy_goal_passed"))
    return None


def _point_from_lifted_refs(
    lifted_refs: Mapping[str, Any] | None,
    *keys: str,
) -> list[float] | None:
    if not isinstance(lifted_refs, Mapping):
        return None
    for key in keys:
        value = lifted_refs.get(key)
        if not isinstance(value, Mapping):
            continue
        point = value.get("position_world")
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 3:
            try:
                return [float(item) for item in point[:3]]
            except (TypeError, ValueError):
                continue
    return None


def _quat_xyzw_to_matrix(q: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = [float(item) for item in q[:4]]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w),
        (2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w),
        (2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y),
    )


def _yaw_from_xyzw(q: Sequence[float]) -> float:
    x, y, z, w = [float(item) for item in q[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _transform_points(
    pose_xyzw: Sequence[float],
    vertices: Sequence[Sequence[float]],
) -> list[tuple[float, float, float]]:
    rot = _quat_xyzw_to_matrix(pose_xyzw[3:7])
    px, py, pz = [float(item) for item in pose_xyzw[:3]]
    points = []
    for vx, vy, vz in vertices:
        x, y, z = float(vx), float(vy), float(vz)
        points.append(
            (
                px + rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
                py + rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
                pz + rot[2][0] * x + rot[2][1] * y + rot[2][2] * z,
            )
        )
    return points


def _patch_from_task(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(task, Mapping):
        return None
    metadata = task.get("metadata")
    if isinstance(metadata, Mapping):
        center = (
            metadata.get("tabletop_hdu_board_center_world")
            or metadata.get("tabletop_hdu_sticker_center_world")
            or metadata.get("tabletop_mark_center_world")
        )
        size = (
            metadata.get("tabletop_hdu_board_size_xyz")
            or metadata.get("tabletop_hdu_sticker_size_xyz")
            or metadata.get("tabletop_mark_size_xyz")
        )
        yaw_deg = metadata.get("tabletop_hdu_board_yaw_deg")
        if yaw_deg is None:
            yaw_deg = metadata.get("tabletop_hdu_sticker_yaw_deg")
        if yaw_deg is None:
            yaw_deg = metadata.get("tabletop_mark_yaw_deg")
        if (
            isinstance(center, Sequence)
            and len(center) >= 3
            and isinstance(size, Sequence)
            and len(size) >= 3
            and isinstance(yaw_deg, (int, float))
        ):
            return {
                "center_world": [float(item) for item in center[:3]],
                "size_xyz": [float(item) for item in size[:3]],
                "yaw_rad": math.radians(float(yaw_deg)),
                "source": (
                    "task.metadata.tabletop_hdu_board_*"
                    if metadata.get("tabletop_hdu_board_center_world") is not None
                    else "task.metadata.tabletop_hdu_sticker_*"
                    if metadata.get("tabletop_hdu_sticker_center_world") is not None
                    else "task.metadata.tabletop_mark_*"
                ),
            }
    state_bundle = (
        task.get("execution", {}).get("state_bundle", {})
        if isinstance(task.get("execution"), Mapping)
        else {}
    )
    actors = (
        state_bundle.get("parttoolgen_benchmark_extra_actors", [])
        if isinstance(state_bundle, Mapping)
        else []
    )
    for actor in actors if isinstance(actors, list) else []:
        if not isinstance(actor, Mapping):
            continue
        if actor.get("name") not in {
            "tabletop_hdu_board",
            "tabletop_hdu_sticker",
            "tabletop_wipe_mark",
        }:
            continue
        pose = actor.get("pose_xyzw")
        if not (isinstance(pose, Sequence) and len(pose) >= 7):
            continue
        size = None
        if isinstance(metadata, Mapping):
            candidate_size = (
                metadata.get("tabletop_hdu_board_size_xyz")
                or metadata.get("tabletop_hdu_sticker_size_xyz")
                or metadata.get("tabletop_mark_size_xyz")
            )
            if isinstance(candidate_size, Sequence) and len(candidate_size) >= 3:
                size = [float(item) for item in candidate_size[:3]]
        if size is None:
            return None
        return {
            "center_world": [float(item) for item in pose[:3]],
            "size_xyz": size,
            "yaw_rad": _yaw_from_xyzw([float(item) for item in pose[3:7]]),
            "source": "task.execution.state_bundle.parttoolgen_benchmark_extra_actors",
        }
    return None


def evaluate_eraser_wipe_lifted_target_path(
    *,
    task: Mapping[str, Any] | None,
    lifted_refs: Mapping[str, Any] | None,
    lateral_margin_m: float = 0.020,
    endpoint_margin_m: float = 0.020,
    max_abs_z_error_m: float = 0.020,
    min_span_fraction: float = 0.70,
    min_axis_cosine: float = 0.80,
) -> dict[str, Any]:
    patch = _patch_from_task(task)
    if patch is None:
        return {
            "schema_version": "toolframe_eraser_wipe_lifted_target_path_v1",
            "passed": False,
            "failure_reason": "target_patch_geometry_missing",
        }
    contact = _point_from_lifted_refs(
        lifted_refs,
        "start",
        "start_point",
    )
    goal = _point_from_lifted_refs(
        lifted_refs,
        "end",
        "end_point",
        "action_endpoint",
    )
    if contact is None or goal is None:
        return {
            "schema_version": "toolframe_eraser_wipe_lifted_target_path_v1",
            "passed": False,
            "failure_reason": "lifted_start_or_end_missing",
            "has_contact": contact is not None,
            "has_end": goal is not None,
        }

    center = [float(item) for item in patch["center_world"]]
    size = [float(item) for item in patch["size_xyz"]]
    yaw = float(patch["yaw_rad"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    long_axis_is_x = size[0] >= size[1]
    long_size = float(size[0] if long_axis_is_x else size[1])
    short_size = float(size[1] if long_axis_is_x else size[0])
    half_long = 0.5 * long_size
    half_short = 0.5 * short_size
    patch_top_z = center[2] + 0.5 * float(size[2])

    def local(point: Sequence[float]) -> tuple[float, float, float]:
        dx = float(point[0]) - center[0]
        dy = float(point[1]) - center[1]
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return (
            local_x if long_axis_is_x else local_y,
            local_y if long_axis_is_x else local_x,
            float(point[2]) - patch_top_z,
        )

    contact_local = local(contact)
    end_local = local(goal)
    span_m = abs(end_local[0] - contact_local[0])
    span_fraction = span_m / long_size if long_size > 0.0 else 0.0
    axis_vector = [goal[index] - contact[index] for index in range(3)]
    axis_xy_norm = math.hypot(axis_vector[0], axis_vector[1])
    axis_cosine = None
    if axis_xy_norm > 1e-9:
        long_axis_world = (
            (cos_yaw, sin_yaw, 0.0)
            if long_axis_is_x
            else (-sin_yaw, cos_yaw, 0.0)
        )
        axis_cosine = abs(
            (axis_vector[0] * long_axis_world[0] + axis_vector[1] * long_axis_world[1])
            / axis_xy_norm
        )
    xy_passed = all(
        -half_long - float(endpoint_margin_m) <= row[0] <= half_long + float(endpoint_margin_m)
        and abs(row[1]) <= half_short + float(lateral_margin_m)
        for row in (contact_local, end_local)
    )
    z_passed = all(abs(row[2]) <= float(max_abs_z_error_m) for row in (contact_local, end_local))
    span_passed = span_fraction >= float(min_span_fraction)
    axis_passed = axis_cosine is not None and axis_cosine >= float(min_axis_cosine)
    passed = xy_passed and z_passed and span_passed and axis_passed
    failure_reason = None
    if not passed:
        if not xy_passed:
            failure_reason = "lifted_points_outside_target_patch_xy"
        elif not z_passed:
            failure_reason = "lifted_points_wrong_surface_height"
        elif not span_passed:
            failure_reason = "lifted_start_end_span_too_short"
        elif not axis_passed:
            failure_reason = "lifted_direction_not_aligned_with_target"
    return {
        "schema_version": "toolframe_eraser_wipe_lifted_target_path_v1",
        "passed": passed,
        "failure_reason": failure_reason,
        "contact_point_world": contact,
        "end_point_world": goal,
        "contact_local_long_short_dz_m": list(contact_local),
        "end_local_long_short_dz_m": list(end_local),
        "start_end_span_m": span_m,
        "start_end_span_fraction": span_fraction,
        "axis_cosine_to_target_long_axis": axis_cosine,
        "target_patch": {
            "center_world": center,
            "size_xyz": size,
            "top_z": patch_top_z,
            "yaw_rad": yaw,
            "source": patch["source"],
        },
        "thresholds": {
            "lateral_margin_m": float(lateral_margin_m),
            "endpoint_margin_m": float(endpoint_margin_m),
            "max_abs_z_error_m": float(max_abs_z_error_m),
            "min_span_fraction": float(min_span_fraction),
            "min_axis_cosine": float(min_axis_cosine),
        },
    }


def _is_wipe_contact_stage(stage: str) -> bool:
    lowered = str(stage).lower()
    if any(token in lowered for token in ("approach", "grasp", "settle")):
        return False
    return lowered.startswith("action_waypoint") or any(
        token in lowered for token in ("wipe", "contact")
    )


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_total_length(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, float(end) - float(start)) for start, end in intervals)


def evaluate_eraser_wipe_surface_contact_path(
    *,
    task: Mapping[str, Any] | None,
    tracking_samples: Sequence[Mapping[str, Any]] | None,
    tool_mesh_path: str | Path | None = None,
    tool_id: str | None = "flat_eraser",
    max_surface_gap_m: float = 0.008,
    allowed_penetration_m: float = 0.003,
    bottom_band_m: float = 0.004,
    lateral_margin_m: float = 0.012,
    min_coverage_fraction: float = 0.30,
    min_contact_samples: int = 3,
) -> dict[str, Any]:
    patch = _patch_from_task(task)
    if patch is None:
        return {
            "schema_version": "toolframe_eraser_wipe_surface_contact_path_v1",
            "passed": False,
            "failure_reason": "target_patch_geometry_missing",
        }
    resolved_mesh = (
        Path(tool_mesh_path)
        if tool_mesh_path is not None
        else resolve_simtoolreal_tool_mesh_path(tool_id)
    )
    if resolved_mesh is not None and not resolved_mesh.is_absolute():
        resolved_mesh = Path.cwd() / resolved_mesh
    if resolved_mesh is None or not resolved_mesh.exists():
        return {
            "schema_version": "toolframe_eraser_wipe_surface_contact_path_v1",
            "passed": False,
            "failure_reason": "tool_mesh_missing",
            "tool_mesh_path": str(resolved_mesh) if resolved_mesh else None,
        }
    vertices = load_obj_vertices(resolved_mesh)
    if not vertices:
        return {
            "schema_version": "toolframe_eraser_wipe_surface_contact_path_v1",
            "passed": False,
            "failure_reason": "tool_mesh_has_no_vertices",
            "tool_mesh_path": str(resolved_mesh),
        }
    samples = [
        sample
        for sample in (tracking_samples or [])
        if isinstance(sample, Mapping)
        and _is_wipe_contact_stage(str(sample.get("stage", "")))
        and isinstance(sample.get("actual_pose_xyzw"), list)
        and len(sample.get("actual_pose_xyzw")) == 7
    ]
    center = [float(item) for item in patch["center_world"]]
    size = [float(item) for item in patch["size_xyz"]]
    yaw = float(patch["yaw_rad"])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    long_axis_is_x = size[0] >= size[1]
    long_size = float(size[0] if long_axis_is_x else size[1])
    short_size = float(size[1] if long_axis_is_x else size[0])
    half_long = 0.5 * long_size
    half_short = 0.5 * short_size
    patch_top_z = center[2] + 0.5 * float(size[2])

    contact_intervals: list[tuple[float, float]] = []
    contact_samples: list[dict[str, Any]] = []
    stage_sample_count = 0
    for sample in samples:
        pose = [float(item) for item in sample["actual_pose_xyzw"]]
        world_points = _transform_points(pose, vertices)
        min_z = min(point[2] for point in world_points)
        gap = min_z - patch_top_z
        stage_sample_count += 1
        if gap > float(max_surface_gap_m) or gap < -float(allowed_penetration_m):
            continue
        bottom_points = [
            point for point in world_points if point[2] <= min_z + float(bottom_band_m)
        ]
        local_long_values = []
        local_short_values = []
        for x, y, _ in bottom_points:
            dx = x - center[0]
            dy = y - center[1]
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            local_long_values.append(local_x if long_axis_is_x else local_y)
            local_short_values.append(local_y if long_axis_is_x else local_x)
        if not local_long_values or not local_short_values:
            continue
        long_min = min(local_long_values)
        long_max = max(local_long_values)
        short_min = min(local_short_values)
        short_max = max(local_short_values)
        short_overlap = (
            short_max >= -half_short - float(lateral_margin_m)
            and short_min <= half_short + float(lateral_margin_m)
        )
        if not short_overlap:
            continue
        overlap_start = max(long_min, -half_long)
        overlap_end = min(long_max, half_long)
        if overlap_end <= overlap_start:
            continue
        contact_intervals.append((overlap_start, overlap_end))
        contact_samples.append(
            {
                "step": sample.get("step"),
                "goal_index": sample.get("goal_index"),
                "stage": sample.get("stage"),
                "surface_gap_m": gap,
                "bottom_patch_long_interval_m": [overlap_start, overlap_end],
                "bottom_patch_short_interval_m": [short_min, short_max],
            }
        )

    merged = _merge_intervals(contact_intervals)
    covered_length = _interval_total_length(merged)
    coverage_fraction = covered_length / long_size if long_size > 0.0 else 0.0
    passed = (
        len(contact_samples) >= int(min_contact_samples)
        and coverage_fraction >= float(min_coverage_fraction)
    )
    failure_reason = None
    if not passed:
        if not contact_samples:
            failure_reason = "no_surface_contact_on_target_patch"
        elif len(contact_samples) < int(min_contact_samples):
            failure_reason = "insufficient_surface_contact_samples"
        else:
            failure_reason = "insufficient_target_patch_coverage"
    return {
        "schema_version": "toolframe_eraser_wipe_surface_contact_path_v1",
        "passed": passed,
        "failure_reason": failure_reason,
        "tool_mesh_path": str(resolved_mesh),
        "target_patch": {
            "center_world": center,
            "size_xyz": size,
            "top_z": patch_top_z,
            "yaw_rad": yaw,
            "source": patch["source"],
        },
        "thresholds": {
            "max_surface_gap_m": float(max_surface_gap_m),
            "allowed_penetration_m": float(allowed_penetration_m),
            "bottom_band_m": float(bottom_band_m),
            "lateral_margin_m": float(lateral_margin_m),
            "min_coverage_fraction": float(min_coverage_fraction),
            "min_contact_samples": int(min_contact_samples),
        },
        "stage_sample_count": stage_sample_count,
        "contact_sample_count": len(contact_samples),
        "covered_long_length_m": covered_length,
        "coverage_fraction": coverage_fraction,
        "covered_long_intervals_m": [[a, b] for a, b in merged],
        "contact_samples_preview": contact_samples[:12],
    }


def predicate_passed(success_check: Mapping[str, Any], name: str) -> bool | None:
    predicates = success_check.get("predicates")
    if not isinstance(predicates, list):
        return None
    for predicate in predicates:
        if isinstance(predicate, Mapping) and predicate.get("name") == name:
            return bool(predicate.get("passed"))
    return None


def check_hammer_down_strike_success(
    *,
    axis: Mapping[str, Any] | None,
    compile_summary: Mapping[str, Any] | None,
    rollout_summary: Mapping[str, Any] | None = None,
    execution_report: Mapping[str, Any] | None = None,
    skip_rollout: bool = False,
) -> dict[str, Any]:
    semantic_contact_passed = _semantic_execution_passed(rollout_summary, execution_report)
    actor_summary = (
        rollout_summary.get("benchmark_extra_actor_summary")
        if isinstance(rollout_summary, Mapping)
        else None
    )
    predicates = [
        _predicate(
            "support_down_direction",
            bool(axis and axis.get("passed")),
            value=axis,
            threshold={"min_down_cosine": 0.95},
        ),
        _predicate(
            "compile_path_solver",
            bool(compile_summary and compile_summary.get("path_solver_passed")),
            value=compile_summary.get("path_solver_passed") if isinstance(compile_summary, Mapping) else None,
        ),
    ]
    policy = dict(HAMMER_DOWN_SUCCESS_POLICY)
    predicates.append(
        _predicate(
            "semantic_contact",
            bool(semantic_contact_passed),
            value=semantic_contact_passed,
            threshold={"semantic_execution_passed": True},
        )
    )
    if skip_rollout:
        predicates.append(
            _predicate(
                "rollout_evidence_present",
                False,
                value="rollout_skipped",
                threshold={"skip_rollout": False},
            )
        )
    required = [row for row in predicates if row.get("required")]
    passed = bool(required) and all(bool(row.get("passed")) for row in required)
    failure_category = None if passed else "hammer_down_required_predicate_failure"
    return {
        "schema_version": "toolframe_task_success_check_v1",
        "task_type": "hammer_down_strike",
        "policy": policy,
        "passed": passed,
        "predicates": predicates,
        "diagnostics": {
            "policy_goal_passed": (
                bool(rollout_summary.get("policy_goal_passed"))
                if isinstance(rollout_summary, Mapping) and rollout_summary.get("policy_goal_passed") is not None
                else None
            ),
            "tracking_execution_passed": (
                bool(rollout_summary.get("tracking_execution_passed"))
                if isinstance(rollout_summary, Mapping)
                and rollout_summary.get("tracking_execution_passed") is not None
                else None
            ),
            "semantic_execution_passed": semantic_contact_passed,
            "target_displacement_diagnostic": actor_summary,
            "failure_category": failure_category,
        },
    }


def check_brush_sweep_success(
    *,
    effect: Mapping[str, Any] | None,
    rollout_summary: Mapping[str, Any] | None = None,
    execution_report: Mapping[str, Any] | None = None,
    skip_rollout: bool = False,
) -> dict[str, Any]:
    uses_target_region = bool(effect and effect.get("target_region"))
    effect_passed = bool(
        effect
        and (
            effect.get("target_region_passed")
            if uses_target_region
            else effect.get("target_effect_passed")
        )
    )
    policy = (
        dict(BRUSH_SWEEP_TARGET_ZONE_SUCCESS_POLICY)
        if uses_target_region
        else dict(BRUSH_SWEEP_SUCCESS_POLICY)
    )
    predicate_name = (
        "target_region_occupancy"
        if uses_target_region
        else "target_directional_displacement"
    )
    predicate_value = (
        {
            "target_final_xy_world": effect.get("target_final_xy_world")
            if isinstance(effect, Mapping)
            else None,
            "target_region": effect.get("target_region") if isinstance(effect, Mapping) else None,
        }
        if uses_target_region
        else {
            "target_xy_displacement_m": effect.get("target_xy_displacement_m")
            if isinstance(effect, Mapping)
            else None,
            "projected_displacement_m": effect.get("projected_displacement_m")
            if isinstance(effect, Mapping)
            else None,
            "direction_cosine": effect.get("direction_cosine") if isinstance(effect, Mapping) else None,
        }
    )
    predicate_threshold = (
        {"requires_final_target_center_inside_region": True}
        if uses_target_region
        else {"min_xy_displacement_m": 0.06, "min_direction_cosine": 0.6}
    )
    predicates = [
        _predicate(
            predicate_name,
            effect_passed,
            value=predicate_value,
            threshold=predicate_threshold,
        )
    ]
    if skip_rollout:
        predicates.append(
            _predicate(
                "rollout_evidence_present",
                False,
                value="rollout_skipped",
                threshold={"skip_rollout": False},
            )
        )
    semantic_passed = _semantic_execution_passed(rollout_summary, execution_report)
    passed = all(bool(row.get("passed")) for row in predicates)
    failure_category = None if passed else (
        "brush_target_region_failure"
        if uses_target_region
        else "brush_directional_displacement_failure"
    )
    return {
        "schema_version": "toolframe_task_success_check_v1",
        "task_type": "brush_sweep",
        "policy": policy,
        "passed": passed,
        "predicates": predicates,
        "diagnostics": {
            "policy_goal_passed": (
                bool(rollout_summary.get("policy_goal_passed"))
                if isinstance(rollout_summary, Mapping) and rollout_summary.get("policy_goal_passed") is not None
                else None
            ),
            "tracking_execution_passed": (
                bool(rollout_summary.get("tracking_execution_passed"))
                if isinstance(rollout_summary, Mapping)
                and rollout_summary.get("tracking_execution_passed") is not None
                else None
            ),
            "semantic_execution_passed": semantic_passed,
            "failure_category": failure_category,
        },
    }


def check_eraser_wipe_success(
    *,
    compile_summary: Mapping[str, Any] | None,
    rollout_summary: Mapping[str, Any] | None = None,
    execution_report: Mapping[str, Any] | None = None,
    wipe_contact: Mapping[str, Any] | None = None,
    lifted_refs: Mapping[str, Any] | None = None,
    task: Mapping[str, Any] | None = None,
    tracking_samples: Sequence[Mapping[str, Any]] | None = None,
    tool_mesh_path: str | Path | None = None,
    skip_rollout: bool = False,
) -> dict[str, Any]:
    semantic_passed = _semantic_execution_passed(rollout_summary, execution_report)
    policy_goal_passed = _policy_goal_passed(rollout_summary)
    lifted_path = evaluate_eraser_wipe_lifted_target_path(task=task, lifted_refs=lifted_refs)
    lifted_path_passed = bool(
        isinstance(lifted_path, Mapping) and lifted_path.get("passed")
    )
    if wipe_contact is None and not skip_rollout:
        wipe_contact = evaluate_eraser_wipe_surface_contact_path(
            task=task,
            tracking_samples=tracking_samples,
            tool_mesh_path=tool_mesh_path,
            tool_id="flat_eraser",
        )
    contact_coverage_passed = bool(
        isinstance(wipe_contact, Mapping) and wipe_contact.get("passed")
    )
    predicates = [
        _predicate(
            "compile_path_solver",
            bool(compile_summary and compile_summary.get("path_solver_passed")),
            value=compile_summary.get("path_solver_passed") if isinstance(compile_summary, Mapping) else None,
        )
    ]
    policy = dict(ERASER_WIPE_SUCCESS_POLICY)
    predicates.extend(
        [
            _predicate(
                "lifted_target_surface_path",
                lifted_path_passed,
                value=dict(lifted_path) if isinstance(lifted_path, Mapping) else None,
                threshold={
                    "requires_lifted_start_end_on_target_patch": True,
                    "requires_lifted_points_near_tabletop_surface": True,
                },
            ),
            _predicate(
                "contact_coverage",
                contact_coverage_passed,
                value=dict(wipe_contact) if isinstance(wipe_contact, Mapping) else None,
                threshold={
                    "requires_tool_surface_contact_on_target_patch": True,
                    "min_target_patch_coverage_fraction": policy["min_contact_coverage_fraction"],
                },
            ),
            _predicate(
                "policy_goal_path",
                bool(policy_goal_passed),
                required=False,
                value=policy_goal_passed,
                threshold={"policy_goal_passed": True},
            ),
            _predicate(
                "semantic_contact_path",
                bool(semantic_passed),
                required=False,
                value=semantic_passed,
                threshold={"semantic_execution_passed": True},
            ),
        ]
    )
    if skip_rollout:
        predicates.append(
            _predicate(
                "rollout_evidence_present",
                False,
                value="rollout_skipped",
                threshold={"skip_rollout": False},
            )
        )
    required = [row for row in predicates if row.get("required")]
    failure_category = None
    if required and not all(bool(row.get("passed")) for row in required):
        if not bool(predicates[0].get("passed")):
            failure_category = "compile/path infeasible"
        elif skip_rollout:
            failure_category = "rollout evidence skipped"
        elif not lifted_path_passed:
            failure_category = "rgbd lifting/scene path invalid"
        elif not contact_coverage_passed:
            failure_category = "insufficient eraser contact coverage"
    return {
        "schema_version": "toolframe_task_success_check_v1",
        "task_type": "eraser_wipe",
        "policy": policy,
        "passed": bool(required) and all(bool(row.get("passed")) for row in required),
        "predicates": predicates,
        "diagnostics": {
            "policy_goal_passed": policy_goal_passed,
            "tracking_execution_passed": (
                bool(rollout_summary.get("tracking_execution_passed"))
                if isinstance(rollout_summary, Mapping)
                and rollout_summary.get("tracking_execution_passed") is not None
                else None
            ),
            "semantic_execution_passed": semantic_passed,
            "failure_category": failure_category,
            "lifted_target_surface_path": dict(lifted_path) if isinstance(lifted_path, Mapping) else None,
            "wipe_contact": dict(wipe_contact) if isinstance(wipe_contact, Mapping) else None,
        },
    }


def _float_metric(source: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(source, Mapping) or source.get(key) is None:
        return None
    try:
        return float(source[key])
    except (TypeError, ValueError):
        return None


def check_screwdriver_insert_success(
    *,
    compile_summary: Mapping[str, Any] | None,
    insertion: Mapping[str, Any] | None = None,
    rollout_summary: Mapping[str, Any] | None = None,
    execution_report: Mapping[str, Any] | None = None,
    skip_rollout: bool = False,
    max_tip_radial_error_m: float = 0.012,
    min_tip_insertion_depth_m: float = 0.012,
    min_axis_cosine: float = 0.85,
) -> dict[str, Any]:
    semantic_passed = _semantic_execution_passed(rollout_summary, execution_report)
    if insertion is None:
        for report in (execution_report, rollout_summary):
            candidate = report.get("screwdriver_insert") if isinstance(report, Mapping) else None
            if isinstance(candidate, Mapping):
                insertion = candidate
                break
    tip_radial_error_m = _float_metric(insertion, "tip_xy_radial_error_m")
    if tip_radial_error_m is None:
        tip_radial_error_m = _float_metric(insertion, "tip_radial_error_m")
    tip_insertion_depth_m = _float_metric(insertion, "tip_insertion_depth_m")
    tip_end_3d_error_m = _float_metric(insertion, "tip_end_3d_error_m")
    axis_cosine = _float_metric(insertion, "axis_cosine")
    if axis_cosine is None:
        axis_cosine = _float_metric(insertion, "insertion_axis_cosine")
    radial_passed = (
        tip_radial_error_m is not None
        and tip_radial_error_m <= float(max_tip_radial_error_m)
    )
    depth_passed = (
        tip_insertion_depth_m is not None
        and tip_insertion_depth_m >= float(min_tip_insertion_depth_m)
    )
    axis_passed = axis_cosine is not None and axis_cosine >= float(min_axis_cosine)
    predicates = [
        _predicate(
            "compile_path_solver",
            bool(compile_summary and compile_summary.get("path_solver_passed")),
            value=compile_summary.get("path_solver_passed") if isinstance(compile_summary, Mapping) else None,
        )
    ]
    policy = dict(SCREWDRIVER_INSERT_SUCCESS_POLICY)
    predicates.extend(
        [
            _predicate(
                "semantic_insertion_path",
                bool(semantic_passed),
                value=semantic_passed,
                threshold={"semantic_execution_passed": True},
            ),
            _predicate(
                "tip_radial_alignment",
                radial_passed,
                value=tip_radial_error_m,
                threshold={
                    "max_tip_radial_error_m": float(max_tip_radial_error_m),
                    "metric": "tip_xy_radial_error_m",
                },
            ),
            _predicate(
                "tip_insertion_depth",
                depth_passed,
                value=tip_insertion_depth_m,
                threshold={"min_tip_insertion_depth_m": float(min_tip_insertion_depth_m)},
            ),
            _predicate(
                "insertion_axis_alignment",
                axis_passed,
                required=False,
                value=axis_cosine,
                threshold={"min_axis_cosine": float(min_axis_cosine)},
            ),
        ]
    )
    if skip_rollout:
        predicates.append(
            _predicate(
                "rollout_evidence_present",
                False,
                value="rollout_skipped",
                threshold={"skip_rollout": False},
            )
        )
    required = [row for row in predicates if row.get("required")]
    failure_category = None
    if required and not all(bool(row.get("passed")) for row in required):
        if not bool(predicates[0].get("passed")):
            failure_category = "compile/path infeasible"
        elif skip_rollout:
            failure_category = "rollout evidence skipped"
        elif not bool(semantic_passed):
            failure_category = "insertion semantic contact/path failure"
        elif not radial_passed:
            failure_category = "tip xy radial alignment failure"
        elif not depth_passed:
            failure_category = "tip insertion depth failure"
    return {
        "schema_version": "toolframe_task_success_check_v1",
        "task_type": "screwdriver_insert",
        "policy": policy,
        "passed": bool(required) and all(bool(row.get("passed")) for row in required),
        "predicates": predicates,
        "diagnostics": {
            "policy_goal_passed": (
                bool(rollout_summary.get("policy_goal_passed"))
                if isinstance(rollout_summary, Mapping) and rollout_summary.get("policy_goal_passed") is not None
                else None
            ),
            "tracking_execution_passed": (
                bool(rollout_summary.get("tracking_execution_passed"))
                if isinstance(rollout_summary, Mapping)
                and rollout_summary.get("tracking_execution_passed") is not None
                else None
            ),
            "semantic_execution_passed": semantic_passed,
            "failure_category": failure_category,
            "tip_end_3d_error_m": tip_end_3d_error_m,
            "insertion_metrics": dict(insertion) if isinstance(insertion, Mapping) else None,
        },
    }


def check_task_success(task_type: str, **kwargs: Any) -> dict[str, Any]:
    if task_type in {
        "hammer_down_strike",
        "hammer_strike_down",
        "image_grounded_hammer_down_strike_smoke",
    }:
        return check_hammer_down_strike_success(**kwargs)
    if task_type in {"brush_sweep", "image_grounded_brush_seed_eval"}:
        return check_brush_sweep_success(**kwargs)
    if task_type in {"eraser_wipe", "flat_eraser_tabletop_wipe"}:
        return check_eraser_wipe_success(**kwargs)
    if task_type in {"screwdriver_insert", "long_screwdriver_deep_socket_insert"}:
        return check_screwdriver_insert_success(**kwargs)
    raise ValueError(f"unsupported task success check: {task_type}")


__all__ = [
    "BRUSH_SWEEP_SUCCESS_POLICY",
    "BRUSH_SWEEP_TARGET_ZONE_SUCCESS_POLICY",
    "ERASER_WIPE_SUCCESS_POLICY",
    "HAMMER_DOWN_SUCCESS_POLICY",
    "SCREWDRIVER_INSERT_SUCCESS_POLICY",
    "check_brush_sweep_success",
    "check_eraser_wipe_success",
    "check_hammer_down_strike_success",
    "check_screwdriver_insert_success",
    "check_task_success",
    "evaluate_eraser_wipe_lifted_target_path",
    "evaluate_eraser_wipe_surface_contact_path",
    "predicate_passed",
]
