"""Failure attribution for image-grounded tool-use benchmark trials.

The attribution code is evaluator-side diagnostics. It may compare method
outputs against benchmark task geometry recorded in ``task.json``; those checks
must not be fed back into the method pipeline.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


MODULE_FAILURE_CATEGORIES = [
    "success",
    "vlm_tool_selection_error",
    "vlm_scene_marking_error",
    "rgbd_lifting_error",
    "compiler_subgoal_error",
    "policy_tracking_failure",
    "needs_review",
]


def _as_float_list(value: Any, *, min_len: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < min_len:
        return None
    try:
        return [float(item) for item in value[:min_len]]
    except (TypeError, ValueError):
        return None


def _get_nested(source: Mapping[str, Any] | None, keys: Sequence[str]) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _patch_frame_from_task(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
    metadata = task.get("metadata") if isinstance(task, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    center = _as_float_list(
        metadata.get("tabletop_hdu_sticker_center_world")
        or metadata.get("tabletop_mark_center_world"),
        min_len=3,
    )
    size = _as_float_list(
        metadata.get("tabletop_hdu_sticker_size_xyz")
        or metadata.get("tabletop_mark_size_xyz"),
        min_len=3,
    )
    yaw_deg = metadata.get("tabletop_hdu_sticker_yaw_deg")
    if yaw_deg is None:
        yaw_deg = metadata.get("tabletop_mark_yaw_deg")
    if center is None or size is None or not isinstance(yaw_deg, (int, float)):
        return None
    yaw = math.radians(float(yaw_deg))
    if size[0] >= size[1]:
        long_size = float(size[0])
        short_size = float(size[1])
        long_axis = [math.cos(yaw), math.sin(yaw), 0.0]
        short_axis = [-math.sin(yaw), math.cos(yaw), 0.0]
    else:
        long_size = float(size[1])
        short_size = float(size[0])
        long_axis = [-math.sin(yaw), math.cos(yaw), 0.0]
        short_axis = [math.cos(yaw), math.sin(yaw), 0.0]
    return {
        "center_world": center,
        "size_xyz": size,
        "top_z": center[2] + 0.5 * float(size[2]),
        "yaw_rad": yaw,
        "long_axis_world": long_axis,
        "short_axis_world": short_axis,
        "half_long_m": 0.5 * long_size,
        "half_short_m": 0.5 * short_size,
        "long_size_m": long_size,
        "short_size_m": short_size,
    }


def _local_on_patch(point_world: Sequence[float], patch: Mapping[str, Any]) -> tuple[float, float, float]:
    point = [float(item) for item in point_world[:3]]
    center = [float(item) for item in patch["center_world"][:3]]
    long_axis = [float(item) for item in patch["long_axis_world"][:3]]
    short_axis = [float(item) for item in patch["short_axis_world"][:3]]
    dx = [point[index] - center[index] for index in range(3)]
    long_value = sum(dx[index] * long_axis[index] for index in range(3))
    short_value = sum(dx[index] * short_axis[index] for index in range(3))
    dz = point[2] - float(patch["top_z"])
    return long_value, short_value, dz


def _point_from_lifted_refs(
    lifted_refs: Mapping[str, Any] | None,
    *keys: str,
) -> list[float] | None:
    if not isinstance(lifted_refs, Mapping):
        return None
    for key in keys:
        value = lifted_refs.get(key)
        if isinstance(value, Mapping):
            point = _as_float_list(value.get("position_world"), min_len=3)
            if point is not None:
                return point
    return None


def _scene_keypoint_has_required_refs(scene_keypoints: Mapping[str, Any] | None) -> bool:
    if not isinstance(scene_keypoints, Mapping):
        return False
    start_point = scene_keypoints.get("start_point")
    end_point = scene_keypoints.get("end_point")
    action_direction = scene_keypoints.get("action_direction")
    return (
        isinstance(start_point, Mapping)
        and isinstance(end_point, Mapping)
        and isinstance(action_direction, Mapping)
        and isinstance(action_direction.get("start"), Mapping)
        and isinstance(action_direction.get("end"), Mapping)
    )


def _label_matches_eraser_target(scene_keypoints: Mapping[str, Any] | None) -> bool | None:
    target_object = scene_keypoints.get("target_object") if isinstance(scene_keypoints, Mapping) else None
    if not isinstance(target_object, Mapping):
        return None
    text = " ".join(
        str(target_object.get(key, ""))
        for key in ("label", "description")
        if target_object.get(key) is not None
    ).lower()
    if not text.strip():
        return None
    target_tokens = (
        "hdu",
        "sticker",
        "decal",
        "mark",
        "patch",
        "stripe",
        "line",
        "rectangle",
    )
    surface_tokens = ("white", "blue", "table", "tabletop")
    return bool(
        any(token in text for token in target_tokens)
        and any(token in text for token in surface_tokens)
    )


def _lifted_geometry_report(
    *,
    task: Mapping[str, Any] | None,
    lifted_refs: Mapping[str, Any] | None,
    lateral_margin_m: float = 0.020,
    endpoint_margin_m: float = 0.020,
    max_abs_z_error_m: float = 0.020,
    min_span_fraction: float = 0.70,
    min_axis_cosine: float = 0.80,
) -> dict[str, Any]:
    patch = _patch_frame_from_task(task)
    if patch is None:
        return {
            "checked": False,
            "passed": False,
            "failure_reason": "target_patch_geometry_missing",
        }
    contact = _point_from_lifted_refs(
        lifted_refs,
        "start",
        "start_point",
        "target_contact_point",
    )
    goal = _point_from_lifted_refs(
        lifted_refs,
        "end",
        "end_point",
        "action_endpoint",
        "goal",
    )
    if contact is None or goal is None:
        return {
            "checked": True,
            "passed": False,
            "failure_reason": "lifted_start_or_end_missing",
            "has_contact": contact is not None,
            "has_end": goal is not None,
            "target_patch": patch,
        }
    contact_local = _local_on_patch(contact, patch)
    end_local = _local_on_patch(goal, patch)
    half_long = float(patch["half_long_m"])
    half_short = float(patch["half_short_m"])
    long_size = float(patch["long_size_m"])
    start_end_span_m = abs(end_local[0] - contact_local[0])
    coverage_fraction = start_end_span_m / long_size if long_size > 0.0 else 0.0
    axis_vector = [goal[index] - contact[index] for index in range(3)]
    axis_xy_norm = math.sqrt(axis_vector[0] * axis_vector[0] + axis_vector[1] * axis_vector[1])
    long_axis = [float(item) for item in patch["long_axis_world"][:3]]
    axis_cosine = None
    if axis_xy_norm > 1e-9:
        axis_cosine = abs(
            (axis_vector[0] * long_axis[0] + axis_vector[1] * long_axis[1])
            / axis_xy_norm
        )
    xy_passed = all(
        -half_long - endpoint_margin_m <= row[0] <= half_long + endpoint_margin_m
        and abs(row[1]) <= half_short + lateral_margin_m
        for row in (contact_local, end_local)
    )
    z_passed = all(abs(row[2]) <= max_abs_z_error_m for row in (contact_local, end_local))
    span_passed = coverage_fraction >= min_span_fraction
    axis_passed = axis_cosine is not None and axis_cosine >= min_axis_cosine
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
        "checked": True,
        "passed": passed,
        "failure_reason": failure_reason,
        "contact_point_world": contact,
        "end_point_world": goal,
        "contact_local_long_short_dz_m": list(contact_local),
        "end_local_long_short_dz_m": list(end_local),
        "start_end_span_m": start_end_span_m,
        "start_end_span_fraction": coverage_fraction,
        "axis_cosine_to_target_long_axis": axis_cosine,
        "target_patch": patch,
        "thresholds": {
            "lateral_margin_m": float(lateral_margin_m),
            "endpoint_margin_m": float(endpoint_margin_m),
            "max_abs_z_error_m": float(max_abs_z_error_m),
            "min_span_fraction": float(min_span_fraction),
            "min_axis_cosine": float(min_axis_cosine),
        },
    }


def _is_action_contact_goal(row: Mapping[str, Any]) -> bool:
    stage = str(row.get("stage", "")).lower()
    semantic_role = str(row.get("semantic_role", "")).lower()
    if any(token in stage for token in ("approach", "grasp", "settle")):
        return False
    if stage.startswith("action_waypoint"):
        return True
    return any(
        token in semantic_role
        for token in ("contact_point_approach", "contact_point_follow_path", "contact_follow_terminal")
    )


def _action_contact_goal_points(tool_pose_subgoals: Mapping[str, Any] | None) -> list[list[float]]:
    if not isinstance(tool_pose_subgoals, Mapping):
        return []
    subgoals = None
    for key in ("subgoals", "goals", "tool_pose_subgoals"):
        candidate = tool_pose_subgoals.get(key)
        if isinstance(candidate, list):
            subgoals = candidate
            break
    if subgoals is None:
        return []
    points: list[list[float]] = []
    for row in subgoals:
        if not isinstance(row, Mapping):
            continue
        if not _is_action_contact_goal(row):
            continue
        point = _as_float_list(_get_nested(row, ["contact_point_goal", "position_world"]), min_len=3)
        if point is not None:
            points.append(point)
    return points


def _subgoal_geometry_report(
    *,
    task: Mapping[str, Any] | None,
    tool_pose_subgoals: Mapping[str, Any] | None,
    min_span_fraction: float = 0.70,
    max_abs_z_error_m: float = 0.025,
) -> dict[str, Any]:
    patch = _patch_frame_from_task(task)
    if patch is None:
        return {
            "checked": False,
            "passed": False,
            "failure_reason": "target_patch_geometry_missing",
        }
    points = _action_contact_goal_points(tool_pose_subgoals)
    if not points:
        return {
            "checked": True,
            "passed": False,
            "failure_reason": "action_contact_point_goals_missing",
            "action_contact_point_goal_count": 0,
        }
    locals_ = [_local_on_patch(point, patch) for point in points]
    long_values = [row[0] for row in locals_]
    dz_values = [abs(row[2]) for row in locals_]
    span_m = max(long_values) - min(long_values)
    long_size = float(patch["long_size_m"])
    span_fraction = span_m / long_size if long_size > 0.0 else 0.0
    z_passed = bool(dz_values) and max(dz_values) <= max_abs_z_error_m
    span_passed = span_fraction >= min_span_fraction
    passed = z_passed and span_passed
    failure_reason = None
    if not passed:
        failure_reason = (
            "subgoal_action_contact_path_wrong_surface_height"
            if not z_passed
            else "subgoal_action_contact_path_span_too_short"
        )
    return {
        "checked": True,
        "passed": passed,
        "failure_reason": failure_reason,
        "action_contact_point_goal_count": len(points),
        "contact_point_goal_long_range_m": [min(long_values), max(long_values)],
        "contact_point_goal_span_m": span_m,
        "contact_point_goal_span_fraction": span_fraction,
        "max_abs_surface_height_error_m": max(dz_values) if dz_values else None,
        "thresholds": {
            "min_span_fraction": float(min_span_fraction),
            "max_abs_z_error_m": float(max_abs_z_error_m),
        },
    }


def _allowed_values(contract: Mapping[str, Any] | None, key: str) -> list[str]:
    if not isinstance(contract, Mapping):
        return []
    value = contract.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item is not None]


def _label_check(
    *,
    selected_tool_refs: Mapping[str, Any],
    contract: Mapping[str, Any],
    selected_key: str,
    allowed_key: str,
) -> dict[str, Any] | None:
    allowed = _allowed_values(contract, allowed_key)
    if not allowed:
        return None
    selected = selected_tool_refs.get(selected_key)
    return {
        "selected_key": selected_key,
        "selected": selected,
        "allowed_key": allowed_key,
        "allowed": allowed,
        "passed": selected is not None and str(selected) in allowed,
    }


def _tool_grounding_report(
    selected_tool_refs: Mapping[str, Any] | None,
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if selected_tool_refs is None:
        return {"checked": False, "passed": None, "failure_reason": None}
    contact_point_face = selected_tool_refs.get("contact_point_face_marker")
    contact_ref = selected_tool_refs.get("contact_point_source_ref")
    motion_axis_face = selected_tool_refs.get("motion_axis_face_marker")
    contact_point_part = selected_tool_refs.get("contact_point_part_ref")
    contact_point_surface = selected_tool_refs.get("contact_point_surface_ref")
    motion_axis_surface = selected_tool_refs.get("motion_axis_surface_ref")
    completeness_passed = bool(contact_point_face and contact_ref and motion_axis_face)
    label_checks = []
    contract_approved = (
        isinstance(contract, Mapping)
        and str(contract.get("manual_review_status", "")).lower() == "approved"
    )
    allowed_label_fields = (
        ("contact_point_part_ref", "allowed_contact_point_part_refs"),
        ("contact_point_face_marker", "allowed_contact_point_face_markers"),
        ("contact_point_source_ref", "allowed_contact_point_source_refs"),
        ("contact_point_surface_ref", "allowed_contact_point_surface_refs"),
        ("motion_axis_face_marker", "allowed_motion_axis_face_markers"),
        ("motion_axis_surface_ref", "allowed_motion_axis_surface_refs"),
    )
    if contract_approved:
        for selected_key, allowed_key in allowed_label_fields:
            check = _label_check(
                selected_tool_refs=selected_tool_refs,
                contract=contract,
                selected_key=selected_key,
                allowed_key=allowed_key,
            )
            if check is not None:
                label_checks.append(check)
    audit_label_passed = None
    if label_checks:
        audit_label_passed = all(bool(row["passed"]) for row in label_checks)
    passed = completeness_passed
    failure_reason = None
    if not completeness_passed:
        failure_reason = "selected_tool_refs_incomplete"
    return {
        "checked": True,
        "passed": passed,
        "failure_reason": failure_reason,
        "contract_checked": bool(label_checks),
        "audit_policy": "diagnostic_only_not_used_for_selection_or_task_success",
        "audit_label_passed": audit_label_passed,
        "manual_review_status": contract.get("manual_review_status")
        if isinstance(contract, Mapping)
        else None,
        "contact_point_face_marker": contact_point_face,
        "contact_point_part_ref": contact_point_part,
        "contact_point_source_ref": contact_ref,
        "contact_point_surface_ref": contact_point_surface,
        "motion_axis_face_marker": motion_axis_face,
        "motion_axis_surface_ref": motion_axis_surface,
        "label_checks": label_checks,
        "contract": dict(contract) if isinstance(contract, Mapping) else None,
    }


def _rollout_report(
    *,
    rollout_summary: Mapping[str, Any] | None,
    execution_report: Mapping[str, Any] | None,
    tracking_samples: Sequence[Mapping[str, Any]] | None,
    max_steps: int | None,
) -> dict[str, Any]:
    if not isinstance(rollout_summary, Mapping):
        return {"checked": False, "passed": None, "failure_reason": "rollout_summary_missing"}
    final_successes = rollout_summary.get("final_successes")
    num_goals = rollout_summary.get("num_goals")
    num_steps = rollout_summary.get("num_steps")
    policy_goal_passed = bool(rollout_summary.get("policy_goal_passed"))
    tracking_passed = bool(rollout_summary.get("tracking_execution_passed"))
    semantic_passed = bool(rollout_summary.get("semantic_execution_passed"))
    stage_family = None
    first_failed_stage = _get_nested(rollout_summary, ["tracking_error_summary", "first_failed_stage"])
    if isinstance(first_failed_stage, str):
        stage_family = first_failed_stage.split("_", 1)[0]
    elif isinstance(execution_report, Mapping):
        semantic_failed_stage = execution_report.get("semantic_failed_stage")
        if isinstance(semantic_failed_stage, str):
            stage_family = semantic_failed_stage.split("_", 1)[0]
    has_action_sample = any(
        isinstance(sample, Mapping)
        and str(sample.get("stage", "")).lower().startswith("action_waypoint")
        for sample in (tracking_samples or [])
    )
    exhausted_budget = (
        isinstance(max_steps, int)
        and isinstance(num_steps, int)
        and int(num_steps) >= int(max_steps)
        and not policy_goal_passed
    )
    failure_reason = None
    if exhausted_budget:
        failure_reason = (
            "step_budget_exhausted_during_action_path"
            if has_action_sample
            else "step_budget_exhausted_before_action_path"
        )
    elif not policy_goal_passed:
        failure_reason = "policy_goal_not_reached"
    elif not semantic_passed:
        failure_reason = "semantic_contact_path_failed"
    elif not tracking_passed:
        failure_reason = "tracking_gate_failed"
    return {
        "checked": True,
        "passed": policy_goal_passed,
        "failure_reason": failure_reason,
        "policy_goal_passed": policy_goal_passed,
        "tracking_execution_passed": tracking_passed,
        "semantic_execution_passed": semantic_passed,
        "final_successes": final_successes,
        "num_goals": num_goals,
        "num_steps": num_steps,
        "max_steps": max_steps,
        "step_budget_exhausted": exhausted_budget,
        "has_action_sample": has_action_sample,
        "first_failed_stage": first_failed_stage,
        "first_failed_stage_family": stage_family,
    }


def _surface_contact_report(success_check: Mapping[str, Any] | None) -> dict[str, Any]:
    wipe_contact = _get_nested(success_check, ["diagnostics", "wipe_contact"])
    if not isinstance(wipe_contact, Mapping):
        return {"checked": False, "passed": None, "failure_reason": "wipe_contact_metrics_missing"}
    return {
        "checked": True,
        "passed": bool(wipe_contact.get("passed")),
        "failure_reason": wipe_contact.get("failure_reason"),
        "coverage_fraction": wipe_contact.get("coverage_fraction"),
        "contact_sample_count": wipe_contact.get("contact_sample_count"),
        "stage_sample_count": wipe_contact.get("stage_sample_count"),
        "covered_long_intervals_m": wipe_contact.get("covered_long_intervals_m"),
    }


def attribute_eraser_wipe_failure(
    *,
    task: Mapping[str, Any] | None,
    scene_keypoints: Mapping[str, Any] | None = None,
    lifted_refs: Mapping[str, Any] | None = None,
    compile_summary: Mapping[str, Any] | None = None,
    tool_pose_subgoals: Mapping[str, Any] | None = None,
    rollout_summary: Mapping[str, Any] | None = None,
    execution_report: Mapping[str, Any] | None = None,
    success_check: Mapping[str, Any] | None = None,
    tracking_samples: Sequence[Mapping[str, Any]] | None = None,
    selected_tool_refs: Mapping[str, Any] | None = None,
    tool_grounding_contract: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Attribute the primary failure layer for an eraser wipe trial."""

    scene_refs_present = _scene_keypoint_has_required_refs(scene_keypoints)
    target_label_match = _label_matches_eraser_target(scene_keypoints)
    scene_report = {
        "checked": isinstance(scene_keypoints, Mapping),
        "passed": bool(scene_refs_present and target_label_match is not False),
        "failure_reason": None,
        "has_required_refs": scene_refs_present,
        "target_label_matches_expected": target_label_match,
        "target_object": scene_keypoints.get("target_object")
        if isinstance(scene_keypoints, Mapping)
        else None,
    }
    if not scene_refs_present:
        scene_report["failure_reason"] = "required_vlm_scene_refs_missing"
    elif target_label_match is False:
        scene_report["failure_reason"] = "target_object_label_mismatch"

    lifted_report = _lifted_geometry_report(task=task, lifted_refs=lifted_refs)
    tool_report = _tool_grounding_report(
        selected_tool_refs,
        contract=tool_grounding_contract,
    )
    compile_passed = bool(
        isinstance(compile_summary, Mapping) and compile_summary.get("path_solver_passed")
    )
    compile_report = {
        "checked": isinstance(compile_summary, Mapping),
        "passed": compile_passed,
        "failure_reason": None if compile_passed else "compile_path_solver_failed",
        "path_solver_passed": compile_summary.get("path_solver_passed")
        if isinstance(compile_summary, Mapping)
        else None,
    }
    subgoal_report = _subgoal_geometry_report(
        task=task,
        tool_pose_subgoals=tool_pose_subgoals,
    )
    rollout = _rollout_report(
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        tracking_samples=tracking_samples,
        max_steps=max_steps,
    )
    surface = _surface_contact_report(success_check)

    passed = bool(isinstance(success_check, Mapping) and success_check.get("passed"))
    primary_category = "success" if passed else "needs_review"
    primary_layer = None if passed else "needs_review"
    primary_reason = None

    if not passed:
        if not scene_report["checked"] or not scene_refs_present:
            primary_category = "vlm_scene_marking_error"
            primary_layer = "vlm_scene_grounding"
            primary_reason = scene_report["failure_reason"]
        elif target_label_match is False:
            primary_category = "vlm_scene_marking_error"
            primary_layer = "vlm_scene_grounding"
            primary_reason = scene_report["failure_reason"]
        elif lifted_report["checked"] and not lifted_report["passed"]:
            reason = lifted_report.get("failure_reason")
            if reason == "lifted_points_wrong_surface_height":
                primary_category = "rgbd_lifting_error"
                primary_layer = "rgbd_lifting"
            else:
                primary_category = "vlm_scene_marking_error"
                primary_layer = "vlm_scene_grounding"
            primary_reason = reason
        elif tool_report["checked"] and tool_report["passed"] is False:
            primary_category = "vlm_tool_selection_error"
            primary_layer = "tool_grounding"
            primary_reason = tool_report["failure_reason"]
        elif compile_report["checked"] and not compile_report["passed"]:
            primary_category = "compiler_subgoal_error"
            primary_layer = "constraint_compile"
            primary_reason = compile_report["failure_reason"]
        elif subgoal_report["checked"] and not subgoal_report["passed"]:
            primary_category = "compiler_subgoal_error"
            primary_layer = "constraint_compile"
            primary_reason = subgoal_report["failure_reason"]
        elif rollout["checked"] and rollout.get("step_budget_exhausted"):
            primary_category = "policy_tracking_failure"
            primary_layer = "rollout_policy"
            primary_reason = rollout["failure_reason"]
        elif rollout["checked"] and not rollout.get("policy_goal_passed"):
            primary_category = "policy_tracking_failure"
            primary_layer = "rollout_policy"
            primary_reason = rollout["failure_reason"]
        elif rollout["checked"] and not rollout.get("semantic_execution_passed"):
            primary_category = "policy_tracking_failure"
            primary_layer = "rollout_policy"
            primary_reason = rollout["failure_reason"] or "semantic_contact_path_failed"
        elif rollout["checked"] and not rollout.get("tracking_execution_passed"):
            primary_category = "policy_tracking_failure"
            primary_layer = "rollout_policy"
            primary_reason = rollout["failure_reason"] or "tracking_gate_failed"

    return {
        "schema_version": "toolframe_failure_attribution_v1",
        "task_type": "eraser_wipe",
        "label_taxonomy": list(MODULE_FAILURE_CATEGORIES),
        "passed": passed,
        "primary_category": primary_category,
        "primary_layer": primary_layer,
        "primary_reason": primary_reason,
        "layers": {
            "vlm_scene_grounding": scene_report,
            "rgbd_lifting": lifted_report,
            "tool_grounding": tool_report,
            "constraint_compile": compile_report,
            "subgoal_generation": subgoal_report,
            "rollout_policy": rollout,
            "diagnostic_surface_contact_coverage": surface,
        },
    }


def failure_category_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        attribution = row.get("failure_attribution")
        if isinstance(attribution, Mapping):
            counts[str(attribution.get("primary_category", "needs_review"))] += 1
        elif row.get("passed"):
            counts["success"] += 1
        else:
            counts["needs_review"] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "MODULE_FAILURE_CATEGORIES",
    "attribute_eraser_wipe_failure",
    "failure_category_counts",
]
