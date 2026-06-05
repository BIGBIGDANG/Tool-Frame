"""Compute quantitative task-effect metrics from PartToolGen artifacts.

Some helpers in this module read legacy ``private_eval`` fields for historical
reports. Those metrics are evaluator-side diagnostics only and are not active
method inputs for the 6.4 image-grounded tabletop benchmark.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from parttoolgen.compiler.reference_resolver import ReferenceResolutionError, ReferenceResolver
from parttoolgen.validation.validate_parttoolgen import validate_tool_pose_subgoals

from .task_spec import ToolUseTask


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | list[Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tool_pose_subgoals_if_exists(path: Path | None) -> dict[str, Any] | None:
    data = _load_json_if_exists(path)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"ToolPoseSubgoals artifact must be a JSON object: {path}")
    validate_tool_pose_subgoals(data)
    return data


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _unit(values: Iterable[float]) -> list[float] | None:
    vec = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 1e-12:
        return None
    return [value / norm for value in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _quat_xyzw_to_matrix(quat: Iterable[float]) -> list[list[float]]:
    x, y, z, w = [float(value) for value in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
    ]


def _rotate(quat_xyzw: Iterable[float], vector: Iterable[float]) -> list[float]:
    matrix = _quat_xyzw_to_matrix(quat_xyzw)
    x, y, z = [float(value) for value in vector]
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    ]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _selected_contact_point_ref(
    *,
    axis_binding: Mapping[str, Any] | None,
    affordance_binding: Mapping[str, Any] | None,
    program_ir: Mapping[str, Any] | None,
) -> str | None:
    if isinstance(axis_binding, Mapping):
        contact_point = axis_binding.get("contact_point")
        if isinstance(contact_point, Mapping):
            value = (
                contact_point.get("derived_contact_ref")
                or contact_point.get("point_ref")
                or contact_point.get("surface_ref")
            )
            if isinstance(value, str) and value:
                return value
    if isinstance(affordance_binding, Mapping):
        slot_bindings = affordance_binding.get("slot_bindings")
        if isinstance(slot_bindings, Mapping):
            value = slot_bindings.get("function_keypoint_ref")
            if isinstance(value, str) and value:
                return value
    if isinstance(program_ir, Mapping):
        for constraint in program_ir.get("constraints", []):
            if not isinstance(constraint, Mapping):
                continue
            refs = constraint.get("refs")
            if isinstance(refs, Mapping):
                value = refs.get("source")
                if isinstance(value, str) and value.startswith("candidate.points."):
                    return value
    return None


def _selected_motion_axis_ref(
    *,
    axis_binding: Mapping[str, Any] | None,
    affordance_binding: Mapping[str, Any] | None,
    program_ir: Mapping[str, Any] | None,
) -> str | None:
    if isinstance(axis_binding, Mapping):
        motion_axis = axis_binding.get("motion_axis")
        if isinstance(motion_axis, Mapping):
            value = motion_axis.get("axis_ref") or motion_axis.get("ref") or motion_axis.get("surface_ref")
            if isinstance(value, str) and value:
                return value
    if isinstance(affordance_binding, Mapping):
        slot_bindings = affordance_binding.get("slot_bindings")
        if isinstance(slot_bindings, Mapping):
            value = slot_bindings.get("motion_axis_ref")
            if isinstance(value, str) and value:
                return value
    if isinstance(program_ir, Mapping):
        for constraint in program_ir.get("constraints", []):
            if not isinstance(constraint, Mapping) or constraint.get("type") != "align_axis":
                continue
            refs = constraint.get("refs")
            if isinstance(refs, Mapping):
                value = refs.get("source_axis")
                if isinstance(value, str):
                    return value
    return None


def _selected_target_ref(target_selection: Mapping[str, Any] | None) -> str | None:
    if not isinstance(target_selection, Mapping):
        return None
    value = target_selection.get("selected_target_ref")
    return value if isinstance(value, str) and value else None


def _align_axis_constraint(program_ir: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(program_ir, Mapping):
        return None
    for constraint in program_ir.get("constraints", []):
        if isinstance(constraint, Mapping) and constraint.get("type") == "align_axis":
            return constraint
    return None


def _move_along_constraints(program_ir: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(program_ir, Mapping):
        return []
    return [
        constraint
        for constraint in program_ir.get("constraints", [])
        if isinstance(constraint, Mapping) and constraint.get("type") == "move_along"
    ]


def _goal_orientation(goal: Mapping[str, Any]) -> list[float] | None:
    contact_point_goal = goal.get("contact_point_goal")
    if isinstance(contact_point_goal, Mapping):
        orientation = contact_point_goal.get("orientation_xyzw")
        if isinstance(orientation, list) and len(orientation) == 4:
            return [float(value) for value in orientation]
    pose = goal.get("pose_world_tool")
    if isinstance(pose, Mapping):
        orientation = pose.get("orientation_xyzw")
        if isinstance(orientation, list) and len(orientation) == 4:
            return [float(value) for value in orientation]
    return None


def _contact_point_goal_position(goal: Mapping[str, Any]) -> list[float] | None:
    contact_point_goal = goal.get("contact_point_goal")
    if not isinstance(contact_point_goal, Mapping):
        return None
    position = contact_point_goal.get("position_world")
    if isinstance(position, list) and len(position) == 3:
        return [float(value) for value in position]
    return None


def _pre_contact_goal(subgoals: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(subgoals, Mapping):
        return None
    goals = [goal for goal in subgoals.get("goals", []) if isinstance(goal, Mapping)]
    pre_goals = []
    for goal in goals:
        expected_contact = goal.get("expected_contact")
        scene_target = expected_contact.get("scene_target") if isinstance(expected_contact, Mapping) else None
        stage = str(goal.get("stage") or "")
        if (isinstance(scene_target, str) and "pre_contact" in scene_target) or "pre" in stage:
            pre_goals.append(goal)
    if pre_goals:
        semantic_pre_goals = [goal for goal in pre_goals if goal.get("semantic_gate") is True]
        if semantic_pre_goals:
            return semantic_pre_goals[-1]
        return pre_goals[-1]
    return goals[0] if goals else None


def _metric_unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "reason": reason, **extra}


def _build_resolver(
    *,
    observed_scene_graph: Mapping[str, Any] | None,
    functional_tool_parts: Mapping[str, Any] | None,
    grounding_candidates: Mapping[str, Any] | None,
    target_part_graph: Mapping[str, Any] | None,
    tool_part_graph: Mapping[str, Any] | None,
) -> ReferenceResolver | None:
    if not isinstance(observed_scene_graph, Mapping):
        return None
    return ReferenceResolver(
        observed_scene_graph=dict(observed_scene_graph),
        functional_tool_parts=dict(functional_tool_parts or {}),
        grounding_candidates=dict(grounding_candidates or {}),
        target_part_graph=dict(target_part_graph or {}),
        tool_part_graph=dict(tool_part_graph or {}),
    )


def _contact_success_rate(execution_report: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(execution_report, Mapping):
        return _metric_unavailable("missing_execution_report")
    stage_reports = [row for row in execution_report.get("stage_reports", []) if isinstance(row, Mapping)]
    required = [row for row in stage_reports if row.get("semantic_contact_required") is True]
    if not required:
        return _metric_unavailable("no_semantic_contact_required_stages")
    passed = [row for row in required if row.get("semantic_contact_success") is True]
    errors = [_as_float(row.get("semantic_contact_error_m")) for row in required]
    errors = [value for value in errors if value is not None]
    return {
        "available": True,
        "rate": len(passed) / float(len(required)),
        "passed_count": len(passed),
        "required_count": len(required),
        "max_error_m": max(errors) if errors else None,
        "mean_error_m": sum(errors) / float(len(errors)) if errors else None,
    }


def _pre_contact_alignment_error(
    *,
    resolver: ReferenceResolver | None,
    program_ir: Mapping[str, Any] | None,
    subgoals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if resolver is None:
        return _metric_unavailable("missing_reference_resolver_inputs")
    constraint = _align_axis_constraint(program_ir)
    if constraint is None:
        return _metric_unavailable("missing_align_axis_constraint")
    refs = constraint.get("refs")
    if not isinstance(refs, Mapping):
        return _metric_unavailable("missing_align_axis_refs")
    source_axis_ref = refs.get("source_axis")
    target_axis_ref = refs.get("target_axis")
    if not isinstance(source_axis_ref, str) or not isinstance(target_axis_ref, str):
        return _metric_unavailable("invalid_align_axis_refs")
    goal = _pre_contact_goal(subgoals)
    if goal is None:
        return _metric_unavailable("missing_pre_contact_goal")
    orientation = _goal_orientation(goal)
    if orientation is None:
        return _metric_unavailable("missing_pre_contact_orientation")
    try:
        source = resolver.tool_vector(source_axis_ref)
        target = resolver.world_vector(target_axis_ref)
    except ReferenceResolutionError as exc:
        return _metric_unavailable("ref_resolution_failed", error=str(exc))
    if source.vector_tool is None or target.vector_world is None:
        return _metric_unavailable("axis_frame_mismatch")
    source_world = _unit(_rotate(orientation, source.vector_tool))
    target_world = _unit(target.vector_world)
    if source_world is None or target_world is None:
        return _metric_unavailable("zero_axis")
    params = constraint.get("params") if isinstance(constraint.get("params"), Mapping) else {}
    polarity = str(params.get("polarity") or "same")
    cosine = _dot(source_world, target_world)
    if polarity == "opposite":
        signed_cosine = -cosine
    elif polarity in {"either", "unsigned"}:
        signed_cosine = abs(cosine)
    else:
        signed_cosine = cosine
    error = math.acos(_clamp(signed_cosine))
    return {
        "available": True,
        "error_rad": error,
        "error_deg": math.degrees(error),
        "axis_cosine": signed_cosine,
        "raw_axis_cosine": cosine,
        "source_axis_ref": source_axis_ref,
        "target_axis_ref": target_axis_ref,
        "polarity": polarity,
        "goal_stage": goal.get("stage"),
        "goal_index": goal.get("index"),
    }


def _motion_axis_from_move(
    *,
    resolver: ReferenceResolver | None,
    move_constraints: list[Mapping[str, Any]],
) -> tuple[list[float] | None, dict[str, Any]]:
    if resolver is None:
        return None, {"reason": "missing_reference_resolver_inputs"}
    for constraint in move_constraints:
        refs = constraint.get("refs")
        if not isinstance(refs, Mapping):
            continue
        start_ref = refs.get("start")
        goal_ref = refs.get("end")
        if isinstance(start_ref, str) and isinstance(goal_ref, str):
            try:
                start = resolver.world_point(start_ref)
                goal = resolver.world_point(goal_ref)
            except ReferenceResolutionError:
                continue
            if start.point_world is not None and goal.point_world is not None:
                direction = _unit(goal.point_world[i] - start.point_world[i] for i in range(3))
                if direction is not None:
                    return direction, {"source": "move_along_start_to_goal", "constraint_id": constraint.get("id")}
        axis_ref = refs.get("motion_axis") or refs.get("axis") or refs.get("target_axis")
        if isinstance(axis_ref, str):
            try:
                axis = resolver.world_vector(axis_ref)
            except ReferenceResolutionError:
                continue
            if axis.vector_world is not None:
                return list(axis.vector_world), {"source": "move_along_axis_ref", "axis_ref": axis_ref}
    return None, {"reason": "missing_resolvable_move_direction"}


def _functional_direction_consistency(
    *,
    resolver: ReferenceResolver | None,
    program_ir: Mapping[str, Any] | None,
    subgoals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(subgoals, Mapping):
        return _metric_unavailable("missing_subgoals")
    move_constraints = _move_along_constraints(program_ir)
    if not move_constraints:
        return _metric_unavailable("missing_move_along_constraint")
    motion_axis, direction_info = _motion_axis_from_move(resolver=resolver, move_constraints=move_constraints)
    if motion_axis is None:
        return _metric_unavailable(direction_info.get("reason", "missing_motion_axis"), **direction_info)
    move_ids = {str(row.get("id")) for row in move_constraints if row.get("id") is not None}
    goals = []
    for goal in subgoals.get("goals", []):
        if not isinstance(goal, Mapping):
            continue
        source_constraints = {str(value) for value in goal.get("source_constraints", [])}
        if source_constraints & move_ids:
            goals.append(goal)
    if len(goals) < 2:
        return _metric_unavailable("not_enough_move_along_goals", move_constraint_ids=sorted(move_ids))
    cosines = []
    for previous, current in zip(goals, goals[1:]):
        p0 = _contact_point_goal_position(previous)
        p1 = _contact_point_goal_position(current)
        if p0 is None or p1 is None:
            continue
        step_axis = _unit(p1[i] - p0[i] for i in range(3))
        if step_axis is None:
            continue
        cosines.append(_dot(step_axis, motion_axis))
    if not cosines:
        return _metric_unavailable("no_nonzero_functional_motion_segments")
    return {
        "available": True,
        "cosine": sum(cosines) / float(len(cosines)),
        "min_segment_cosine": min(cosines),
        "max_segment_cosine": max(cosines),
        "segment_count": len(cosines),
        "motion_axis_world": motion_axis,
        **direction_info,
    }


def _tool_misuse_rate(task: ToolUseTask, selected_contact_point_ref: str | None) -> dict[str, Any]:
    allowed = task.private_eval.get("allowed_contact_point_source_refs", [])
    if not isinstance(allowed, list) or not allowed:
        return _metric_unavailable("missing_private_allowed_contact_point_source_refs")
    allowed_refs = {str(value) for value in allowed}
    passed = isinstance(selected_contact_point_ref, str) and selected_contact_point_ref in allowed_refs
    return {
        "available": True,
        "rate": 0.0 if passed else 1.0,
        "selected_contact_point_ref": selected_contact_point_ref,
        "allowed_contact_point_source_refs": sorted(allowed_refs),
    }


def _tracking_metrics(
    *,
    execution_report: Mapping[str, Any] | None,
    rollout_summary: Mapping[str, Any] | None,
    visual_diagnosis_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage_reports = (
        [row for row in execution_report.get("stage_reports", []) if isinstance(row, Mapping)]
        if isinstance(execution_report, Mapping)
        else []
    )
    tracking_errors = [_as_float(row.get("tracking_error_m")) for row in stage_reports]
    rotation_errors = [_as_float(row.get("rotation_error_rad")) for row in stage_reports]
    tracking_errors = [value for value in tracking_errors if value is not None]
    rotation_errors = [value for value in rotation_errors if value is not None]
    semantic_errors = [_as_float(row.get("semantic_contact_error_m")) for row in stage_reports]
    semantic_errors = [value for value in semantic_errors if value is not None]

    visual_metrics: Mapping[str, Any] = {}
    if isinstance(visual_diagnosis_report, Mapping):
        metadata = visual_diagnosis_report.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("metrics"), Mapping):
            visual_metrics = metadata.get("metrics")  # type: ignore[assignment]

    tracking_passed = None
    if isinstance(execution_report, Mapping) and execution_report.get("tracking_execution_passed") is not None:
        tracking_passed = bool(execution_report.get("tracking_execution_passed"))
    semantic_passed = None
    if isinstance(execution_report, Mapping) and execution_report.get("semantic_execution_passed") is not None:
        semantic_passed = bool(execution_report.get("semantic_execution_passed"))
    max_tracking_error = max(tracking_errors) if tracking_errors else _as_float(visual_metrics.get("max_tracking_error_m"))
    max_rotation_error = max(rotation_errors) if rotation_errors else _as_float(visual_metrics.get("max_rotation_error_rad"))
    max_semantic_error = max(semantic_errors) if semantic_errors else _as_float(visual_metrics.get("semantic_contact_error_m"))
    video_frames = (
        _as_float(rollout_summary.get("render_video_frames")) if isinstance(rollout_summary, Mapping) else None
    )
    return {
        "tracking_passed": {
            "available": tracking_passed is not None,
            "value": 1.0 if tracking_passed else 0.0,
            "passed": tracking_passed,
        }
        if tracking_passed is not None
        else _metric_unavailable("missing_tracking_execution_passed"),
        "semantic_execution_passed": {
            "available": semantic_passed is not None,
            "value": 1.0 if semantic_passed else 0.0,
            "passed": semantic_passed,
        }
        if semantic_passed is not None
        else _metric_unavailable("missing_semantic_execution_passed"),
        "max_tracking_error_m": max_tracking_error
        if max_tracking_error is not None
        else _metric_unavailable("missing_tracking_error"),
        "max_rotation_error_rad": max_rotation_error
        if max_rotation_error is not None
        else _metric_unavailable("missing_rotation_error"),
        "max_semantic_contact_error_m": max_semantic_error
        if max_semantic_error is not None
        else _metric_unavailable("missing_semantic_contact_error"),
        "rollout_video_frames": video_frames if video_frames is not None else _metric_unavailable("missing_video_frames"),
    }


def _target_displacement_metric(
    rollout_summary: Mapping[str, Any] | None,
    *,
    motion_axis_world: list[float] | None = None,
) -> dict[str, Any]:
    if not isinstance(rollout_summary, Mapping):
        return _metric_unavailable("missing_rollout_summary")
    summary = rollout_summary.get("benchmark_extra_actor_summary")
    if not isinstance(summary, Mapping) or not summary.get("checked"):
        return _metric_unavailable("no_dynamic_target_state_trace")
    displacement = _as_float(summary.get("max_displacement_m"))
    xy_displacement = _as_float(summary.get("max_xy_displacement_m"))
    actors = summary.get("actors")
    actor_rows = actors if isinstance(actors, list) else []
    enriched_actors = []
    for actor in actor_rows:
        if not isinstance(actor, Mapping):
            continue
        row = dict(actor)
        direction_cosine = _as_float(row.get("direction_cosine"))
        start_pose = row.get("initial_pose_xyzw")
        final_pose = row.get("final_pose_xyzw")
        if direction_cosine is None and motion_axis_world is not None and isinstance(start_pose, list) and isinstance(final_pose, list):
            delta_axis = _unit(float(final_pose[i]) - float(start_pose[i]) for i in range(3))
            if delta_axis is not None:
                direction_cosine = _clamp(_dot(delta_axis, motion_axis_world))
                row["direction_cosine"] = direction_cosine
                row["direction_axis_world"] = motion_axis_world
        enriched_actors.append(row)
    first_actor = enriched_actors[0] if enriched_actors else {}
    return {
        "available": displacement is not None,
        "distance_m": displacement,
        "xy_distance_m": xy_displacement,
        "actors": enriched_actors,
        "direction_cosine": _as_float(first_actor.get("direction_cosine"))
        if isinstance(first_actor, Mapping)
        else None,
    }


def _language_axis_binding_metric(
    task: ToolUseTask,
    affordance_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected = task.private_eval.get("expected_language_binding")
    if not isinstance(expected, Mapping):
        return _metric_unavailable("missing_expected_language_binding")
    if not isinstance(affordance_binding, Mapping):
        return _metric_unavailable("missing_affordance_binding")

    expected_axis = expected.get("motion_axis_ref")
    acceptable_axes = expected.get("acceptable_motion_axis_refs")
    if not isinstance(acceptable_axes, list) or not acceptable_axes:
        acceptable_axes = [expected_axis] if isinstance(expected_axis, str) else []
    acceptable_axis_set = {str(value) for value in acceptable_axes if isinstance(value, str)}
    selected_axis = (
        affordance_binding.get("motion_axis_ref")
        or affordance_binding.get("selected_target_axis_ref")
        or affordance_binding.get("target_axis_ref")
    )
    selected_contact = affordance_binding.get("target_contact_ref") or affordance_binding.get(
        "selected_target_contact_ref"
    )
    selected_pre = affordance_binding.get("target_pre_contact_ref")
    selected_post = affordance_binding.get("target_post_contact_ref")

    checks = {
        "motion_axis": isinstance(selected_axis, str) and selected_axis in acceptable_axis_set,
        "target_contact": selected_contact == expected.get("target_contact_ref"),
        "target_pre_contact": selected_pre == expected.get("target_pre_contact_ref"),
        "target_post_contact": selected_post == expected.get("target_post_contact_ref"),
    }
    passed = all(checks.values())
    return {
        "available": True,
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "checks": checks,
        "selected": {
            "motion_axis_ref": selected_axis,
            "target_contact_ref": selected_contact,
            "target_pre_contact_ref": selected_pre,
            "target_post_contact_ref": selected_post,
        },
        "expected": dict(expected),
    }


def _debug_metrics(repair_trace: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(repair_trace, Mapping):
        return {
            "debug_count": _metric_unavailable("missing_repair_trace"),
            "pcap_debug_count": _metric_unavailable("missing_repair_trace"),
            "rollout_attempt_count": _metric_unavailable("missing_repair_trace"),
        }
    debug_count = _as_float(repair_trace.get("total_debug_count"))
    if debug_count is None:
        debug_count = _as_float(repair_trace.get("debug_count"))
    return {
        "debug_count": debug_count if debug_count is not None else _metric_unavailable("missing_debug_count"),
        "pcap_debug_count": _as_float(repair_trace.get("pcap_debug_count"))
        if _as_float(repair_trace.get("pcap_debug_count")) is not None
        else _metric_unavailable("missing_pcap_debug_count"),
        "motion_axis_rebind_count": _as_float(repair_trace.get("motion_axis_rebind_count"))
        if _as_float(repair_trace.get("motion_axis_rebind_count")) is not None
        else _metric_unavailable("missing_motion_axis_rebind_count"),
        "region_rebind_count": _as_float(repair_trace.get("region_rebind_count"))
        if _as_float(repair_trace.get("region_rebind_count")) is not None
        else _metric_unavailable("missing_region_rebind_count"),
        "rollout_attempt_count": _as_float(repair_trace.get("rollout_attempt_count"))
        if _as_float(repair_trace.get("rollout_attempt_count")) is not None
        else _metric_unavailable("missing_rollout_attempt_count"),
        "pipeline_attempt_count": _as_float(repair_trace.get("pipeline_attempt_count"))
        if _as_float(repair_trace.get("pipeline_attempt_count")) is not None
        else _metric_unavailable("missing_pipeline_attempt_count"),
        "repair_trace_status": {"available": True, "status": repair_trace.get("status")},
    }


def build_task_effect_report(
    *,
    task: ToolUseTask,
    output_path: Path,
    target_selection_path: Path | None = None,
    axis_binding_path: Path | None = None,
    affordance_binding_path: Path | None = None,
    program_ir_path: Path | None = None,
    subgoals_path: Path | None = None,
    execution_report_path: Path | None = None,
    rollout_summary_path: Path | None = None,
    repair_trace_path: Path | None = None,
    visual_diagnosis_report_path: Path | None = None,
    observed_scene_graph_path: Path | None = None,
    target_part_graph_path: Path | None = None,
    functional_tool_parts_path: Path | None = None,
    grounding_candidates_path: Path | None = None,
    tool_part_graph_path: Path | None = None,
) -> dict[str, Any]:
    target_selection = _load_json_if_exists(target_selection_path)
    axis_binding = _load_json_if_exists(axis_binding_path)
    affordance_binding = _load_json_if_exists(affordance_binding_path)
    program_ir = _load_json_if_exists(program_ir_path)
    subgoals = _load_tool_pose_subgoals_if_exists(subgoals_path)
    execution_report = _load_json_if_exists(execution_report_path)
    rollout_summary = _load_json_if_exists(rollout_summary_path)
    repair_trace = _load_json_if_exists(repair_trace_path)
    visual_diagnosis_report = _load_json_if_exists(visual_diagnosis_report_path)
    observed_scene_graph = _load_json_if_exists(observed_scene_graph_path)
    target_part_graph = _load_json_if_exists(target_part_graph_path)
    functional_tool_parts = _load_json_if_exists(functional_tool_parts_path)
    grounding_candidates = _load_json_if_exists(grounding_candidates_path)
    tool_part_graph = _load_json_if_exists(tool_part_graph_path)

    target_selection_map = target_selection if isinstance(target_selection, Mapping) else None
    axis_binding_map = axis_binding if isinstance(axis_binding, Mapping) else None
    affordance_binding_map = affordance_binding if isinstance(affordance_binding, Mapping) else None
    program_ir_map = program_ir if isinstance(program_ir, Mapping) else None
    subgoals_map = subgoals if isinstance(subgoals, Mapping) else None
    execution_report_map = execution_report if isinstance(execution_report, Mapping) else None
    rollout_summary_map = rollout_summary if isinstance(rollout_summary, Mapping) else None
    repair_trace_map = repair_trace if isinstance(repair_trace, Mapping) else None
    visual_diagnosis_map = visual_diagnosis_report if isinstance(visual_diagnosis_report, Mapping) else None

    selected_contact_ref = _selected_contact_point_ref(
        axis_binding=axis_binding_map,
        affordance_binding=affordance_binding_map,
        program_ir=program_ir_map,
    )
    selected_motion_axis_ref = _selected_motion_axis_ref(
        axis_binding=axis_binding_map,
        affordance_binding=affordance_binding_map,
        program_ir=program_ir_map,
    )

    resolver = _build_resolver(
        observed_scene_graph=observed_scene_graph if isinstance(observed_scene_graph, Mapping) else None,
        functional_tool_parts=functional_tool_parts if isinstance(functional_tool_parts, Mapping) else None,
        grounding_candidates=grounding_candidates if isinstance(grounding_candidates, Mapping) else None,
        target_part_graph=target_part_graph if isinstance(target_part_graph, Mapping) else None,
        tool_part_graph=tool_part_graph if isinstance(tool_part_graph, Mapping) else None,
    )
    move_constraints = _move_along_constraints(program_ir_map)
    motion_axis_world, motion_axis_info = _motion_axis_from_move(
        resolver=resolver,
        move_constraints=move_constraints,
    )
    tracking = _tracking_metrics(
        execution_report=execution_report_map,
        rollout_summary=rollout_summary_map,
        visual_diagnosis_report=visual_diagnosis_map,
    )
    report = {
        "schema_version": "parttoolgen_task_effect_report_v0",
        "task_id": task.task_id,
        "selected_target_ref": _selected_target_ref(target_selection_map),
        "selected_contact_point_ref": selected_contact_ref,
        "selected_motion_axis_ref": selected_motion_axis_ref,
        "metrics": {
            "csr": _contact_success_rate(execution_report_map),
            "pae_rad": _pre_contact_alignment_error(
                resolver=resolver,
                program_ir=program_ir_map,
                subgoals=subgoals_map,
            ),
            "tmr": _tool_misuse_rate(task, selected_contact_ref),
            "fdc": _functional_direction_consistency(
                resolver=resolver,
                program_ir=program_ir_map,
                subgoals=subgoals_map,
            ),
            "target_displacement_m": _target_displacement_metric(
                rollout_summary_map,
                motion_axis_world=motion_axis_world,
            ),
            "language_axis_binding_accuracy": _language_axis_binding_metric(task, affordance_binding_map),
            "effect_direction": {
                "available": motion_axis_world is not None,
                "motion_axis_world": motion_axis_world,
                **motion_axis_info,
            },
            **tracking,
            **_debug_metrics(repair_trace_map),
        },
        "grounding": {
            "selected_contact_point_ref": selected_contact_ref,
            "selected_motion_axis_ref": selected_motion_axis_ref,
            "selected_target_ref": _selected_target_ref(target_selection_map),
        },
        "artifact_refs": {
            "target_selection": str(target_selection_path) if target_selection_path else None,
            "axis_binding": str(axis_binding_path) if axis_binding_path else None,
            "affordance_binding": str(affordance_binding_path) if affordance_binding_path else None,
            "program_ir": str(program_ir_path) if program_ir_path else None,
            "subgoals": str(subgoals_path) if subgoals_path else None,
            "execution_report": str(execution_report_path) if execution_report_path else None,
            "rollout_summary": str(rollout_summary_path) if rollout_summary_path else None,
            "repair_trace": str(repair_trace_path) if repair_trace_path else None,
            "visual_diagnosis_report": str(visual_diagnosis_report_path) if visual_diagnosis_report_path else None,
            "observed_scene_graph": str(observed_scene_graph_path) if observed_scene_graph_path else None,
            "target_part_graph": str(target_part_graph_path) if target_part_graph_path else None,
            "functional_tool_parts": str(functional_tool_parts_path) if functional_tool_parts_path else None,
            "grounding_candidates": str(grounding_candidates_path) if grounding_candidates_path else None,
            "tool_part_graph": str(tool_part_graph_path) if tool_part_graph_path else None,
        },
        "metadata": {
            "created_by": "parttoolgen.evaluation.task_effect",
            "uses_private_eval_for_metrics": True,
            "method_input_modified": False,
            "simtoolreal_policy_modified": False,
        },
    }
    _write_json(output_path, report)
    return report


__all__ = ["build_task_effect_report"]
