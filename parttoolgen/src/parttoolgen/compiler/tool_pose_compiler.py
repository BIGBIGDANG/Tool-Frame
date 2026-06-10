#!/usr/bin/env python3
"""Compile ProgramIR into sparse whole-tool pose subgoals.

This module is orchestration only: it validates artifacts, resolves/canonicalizes
stage constraints, calls the SE(3) optimizer, stitches subgoals, and writes the
ToolPoseSubgoals artifact. Constraint math lives in residuals.py and
se3_optimizer.py.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from parttoolgen.compiler.constraint_canonicalizer import (
    CanonicalStageConstraints,
    canonicalize_stage_constraints,
)
from parttoolgen.compiler.objective_runtime import (
    ObjectiveFunction,
    ObjectiveRuntimeError,
    load_objective_functions,
    pose_objective_cost,
    pose_objective_residuals,
)
from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.compiler.residuals import (
    add,
    dot,
    mul,
    norm,
    normalize_quat,
    pose_for_tool_point_goal,
    pose_point,
    quat_rotate,
    quat_angle_rad,
    quat_slerp,
    sub,
    unit,
)
from parttoolgen.compiler.se3_optimizer import (
    StageSolveResult,
    apply_clearance_constraints,
    compile_move_along_waypoints,
    solve_stage_pose,
)
from parttoolgen.validation.validate_parttoolgen import (
    load_structured_file,
    validate_functional_tool_parts,
    validate_grounding_candidates,
    validate_observed_scene_graph,
    validate_program_ir,
    validate_target_part_graph,
    validate_tool_part_graph,
    validate_tool_pose_subgoals,
)


class ProgramIRCompileError(ValueError):
    """Raised when ProgramIR cannot be compiled to tool poses."""


DEFAULT_ENABLE_ALL_TOOL_PART_OBB_TABLE_PROJECTION = True
DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M = 0.0
DEFAULT_SUBGOAL_TABLE_CLEARANCE_TOLERANCE_M = 1e-6


def _constraints_for_stage(program_ir: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    return [
        constraint
        for constraint in program_ir.get("constraints", [])
        if constraint.get("stage") == stage_name
    ]


def _objectives_for_stage(
    program_ir: dict[str, Any],
    stage_name: str,
    *,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    objectives: list[dict[str, Any]] = []
    for objective in program_ir.get("objectives", []):
        if objective.get("stage") != stage_name:
            continue
        if kind is not None and objective.get("kind") != kind:
            continue
        objectives.append(dict(objective))
    return objectives


def _stage_with_locked_tool_points(
    canonical_stage: CanonicalStageConstraints,
    locked_tool_points: dict[str, list[float]],
) -> CanonicalStageConstraints:
    if not locked_tool_points:
        return canonical_stage
    point_targets = [
        replace(target, point_tool=locked_tool_points[target.source_ref], point_tool_candidates=[])
        if target.source_ref in locked_tool_points
        else target
        for target in canonical_stage.point_targets
    ]
    move_along_targets = [
        replace(target, point_tool=locked_tool_points[target.source_ref], point_tool_candidates=[])
        if target.source_ref in locked_tool_points
        else target
        for target in canonical_stage.move_along_targets
    ]
    return replace(
        canonical_stage,
        point_targets=point_targets,
        move_along_targets=move_along_targets,
    )


def _goal_record(
    *,
    index: int,
    stage: str,
    pose: dict[str, Any],
    source_constraints: list[str],
    primary_tool_ref: str,
    primary_target_ref: str,
    resolver: ReferenceResolver,
    duration_sec: float = 0.45,
    allow_contact_penetration: bool = False,
    semantic_role: str | None = None,
    semantic_gate: bool | None = None,
    primary_tool_point_tool: list[float] | None = None,
) -> dict[str, Any]:
    tool_point = resolver.tool_point(primary_tool_ref)
    point_tool = (
        [float(item) for item in primary_tool_point_tool]
        if primary_tool_point_tool is not None
        else (tool_point.point_tool or [])
    )
    functional_position = pose_point(pose, point_tool)
    record = {
        "index": index,
        "stage": stage,
        "pose_world_tool": pose,
        "duration_sec": duration_sec,
        "tolerance": {
            "position_m": 0.02,
            "rotation_rad": 0.18,
        },
        "source_constraints": source_constraints or [f"stage:{stage}"],
        "expected_contact": {
            "tool_part": primary_tool_ref,
            "scene_target": primary_target_ref,
            "allow_contact_penetration": bool(allow_contact_penetration),
        },
        "contact_point_goal": {
            "part_ref": primary_tool_ref,
            "point_tool": point_tool,
            "position_world": functional_position,
            "orientation_xyzw": pose["orientation_xyzw"],
        },
    }
    if semantic_role is not None:
        record["semantic_role"] = semantic_role
    if semantic_gate is not None:
        record["semantic_gate"] = bool(semantic_gate)
    return record


def _pose_delta_records(
    *,
    start_pose: dict[str, Any],
    goals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_pose = {
        "position": [float(x) for x in start_pose["position"]],
        "orientation_xyzw": normalize_quat(start_pose["orientation_xyzw"]),
    }
    max_translation = 0.0
    max_rotation = 0.0
    total_translation = 0.0
    total_rotation = 0.0
    for goal in goals:
        pose = goal["pose_world_tool"]
        translation = norm(sub(pose["position"], previous_pose["position"]))
        rotation = quat_angle_rad(pose["orientation_xyzw"], previous_pose["orientation_xyzw"])
        max_translation = max(max_translation, translation)
        max_rotation = max(max_rotation, rotation)
        total_translation += translation
        total_rotation += rotation
        records.append(
            {
                "goal_index": int(goal["index"]),
                "stage": str(goal["stage"]),
                "translation_delta_m": float(translation),
                "rotation_delta_rad": float(rotation),
            }
        )
        previous_pose = pose
    final_pose = goals[-1]["pose_world_tool"] if goals else start_pose
    net_translation = norm(sub(final_pose["position"], start_pose["position"]))
    net_rotation = quat_angle_rad(final_pose["orientation_xyzw"], start_pose["orientation_xyzw"])
    rotation_denominator = max(net_rotation, 1e-6)
    translation_denominator = max(net_translation, 1e-6)
    return records, {
        "max_translation_delta_m": float(max_translation),
        "max_rotation_delta_rad": float(max_rotation),
        "total_translation_m": float(total_translation),
        "total_rotation_rad": float(total_rotation),
        "net_translation_m": float(net_translation),
        "net_rotation_rad": float(net_rotation),
        "excess_translation_m": float(max(0.0, total_translation - net_translation)),
        "excess_rotation_rad": float(max(0.0, total_rotation - net_rotation)),
        "translation_path_ratio": float(total_translation / translation_denominator),
        "rotation_path_ratio": float(total_rotation / rotation_denominator),
    }


def _support_table_top_z(observed_scene_graph: dict[str, Any]) -> float | None:
    regions = observed_scene_graph.get("regions", {})
    if isinstance(regions, dict):
        table = regions.get("support_table_bbox")
        if isinstance(table, dict):
            center = table.get("center_xyz_world") or table.get("bbox_center_xyz_world")
            extent = table.get("extent_xyz")
            if isinstance(center, list) and len(center) >= 3 and isinstance(extent, list) and len(extent) >= 3:
                return float(center[2]) + 0.5 * float(extent[2])
    objects = observed_scene_graph.get("objects", {})
    if isinstance(objects, dict):
        table = objects.get("support_table")
        if isinstance(table, dict):
            center = table.get("center_xyz_world") or table.get("bbox_center_xyz_world") or table.get("position_world")
            extent = table.get("extent_xyz")
            if isinstance(center, list) and len(center) >= 3 and isinstance(extent, list) and len(extent) >= 3:
                return float(center[2]) + 0.5 * float(extent[2])
    return None


def _project_pose_to_tool_bbox_table_clearance(
    pose: dict[str, Any],
    *,
    table_top_z: float | None,
    bbox_points_tool: list[list[float]],
    clearance_m: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if table_top_z is None or not bbox_points_tool:
        return pose, None
    min_corner_z = min(
        float(pose["position"][2]) + float(quat_rotate(pose["orientation_xyzw"], point)[2])
        for point in bbox_points_tool
    )
    required_min_z = float(table_top_z) + float(clearance_m)
    lift_m = max(0.0, required_min_z - min_corner_z)
    if lift_m <= 1e-9:
        return pose, {
            "applied": False,
            "min_bbox_corner_z_before": min_corner_z,
            "table_top_z": float(table_top_z),
            "required_min_z": required_min_z,
            "lift_m": 0.0,
        }
    projected = {
        "position": [
            float(pose["position"][0]),
            float(pose["position"][1]),
            float(pose["position"][2]) + lift_m,
        ],
        "orientation_xyzw": normalize_quat(pose["orientation_xyzw"]),
    }
    return projected, {
        "applied": True,
        "min_bbox_corner_z_before": min_corner_z,
        "min_bbox_corner_z_after": min_corner_z + lift_m,
        "table_top_z": float(table_top_z),
        "required_min_z": required_min_z,
        "lift_m": lift_m,
    }


def _pose_with_generation_time_table_clearance(
    pose: dict[str, Any],
    *,
    table_top_z: float | None,
    bbox_points_tool: list[list[float]],
    clearance_m: float,
    allow_contact_penetration: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _projected, record = _project_pose_to_tool_bbox_table_clearance(
        pose,
        table_top_z=table_top_z,
        bbox_points_tool=bbox_points_tool,
        clearance_m=clearance_m,
    )
    if record is None:
        return pose, None
    if record.get("applied"):
        lift_m = float(record.get("lift_m", 0.0))
        if lift_m > DEFAULT_SUBGOAL_TABLE_CLEARANCE_TOLERANCE_M:
            raise ProgramIRCompileError(
                "generated subgoal violates hard table bbox clearance invariant: "
                f"required lift {lift_m:.6f}m exceeds tolerance "
                f"{DEFAULT_SUBGOAL_TABLE_CLEARANCE_TOLERANCE_M:.6f}m"
            )
    return pose, record


def _validate_generated_goals_table_clearance(
    *,
    goals: list[dict[str, Any]],
    table_top_z: float | None,
    bbox_points_tool: list[list[float]],
    clearance_m: float,
) -> dict[str, Any]:
    if table_top_z is None or not bbox_points_tool:
        return {"enabled": False, "checked_goal_count": 0, "violation_count": 0, "violations": []}
    violations: list[dict[str, Any]] = []
    min_goal_bbox_z = float("inf")
    required_min_z = float(table_top_z) + float(clearance_m)
    for goal in goals:
        pose = goal["pose_world_tool"]
        _, record = _project_pose_to_tool_bbox_table_clearance(
            pose,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool,
            clearance_m=clearance_m,
        )
        if record is None:
            continue
        min_z = float(record.get("min_bbox_corner_z_before", required_min_z))
        min_goal_bbox_z = min(min_goal_bbox_z, min_z)
        violation_m = max(0.0, required_min_z - min_z)
        if violation_m > DEFAULT_SUBGOAL_TABLE_CLEARANCE_TOLERANCE_M:
            violations.append(
                {
                    "goal_index": int(goal["index"]),
                    "stage": str(goal["stage"]),
                    "min_bbox_corner_z": min_z,
                    "required_min_z": required_min_z,
                    "violation_m": violation_m,
                }
            )
    if violations:
        preview = ", ".join(
            f"{item['stage']}#{item['goal_index']}:{item['violation_m']:.6f}m"
            for item in violations[:5]
        )
        raise ProgramIRCompileError(
            "generated subgoals violate hard table bbox clearance invariant: "
            f"{preview}"
        )
    return {
        "enabled": True,
        "checked_goal_count": len(goals),
        "violation_count": 0,
        "violations": [],
        "min_goal_bbox_corner_z": None if min_goal_bbox_z == float("inf") else min_goal_bbox_z,
        "table_top_z": float(table_top_z),
        "required_min_z": required_min_z,
        "tolerance_m": DEFAULT_SUBGOAL_TABLE_CLEARANCE_TOLERANCE_M,
    }


def _append_transition_if_needed(
    *,
    goals: list[dict[str, Any]],
    current_pose: dict[str, Any],
    next_pose: dict[str, Any],
    stage_name: str,
    source_constraints: list[str],
    primary_tool_ref: str,
    primary_target_ref: str,
    resolver: ReferenceResolver,
    allow_contact_penetration: bool,
    max_translation_m: float,
    max_rotation_rad: float,
    clearance_axis_world: list[float] | None = None,
    min_axis_descent_m: float = 0.015,
    min_lateral_split_m: float = 0.025,
    primary_tool_point_tool: list[float] | None = None,
    table_top_z: float | None = None,
    bbox_points_tool: list[list[float]] | None = None,
    table_clearance_m: float = 0.0,
    table_clearance_records: list[dict[str, Any]] | None = None,
) -> None:
    def make_table_safe(pose: dict[str, Any], *, stage: str) -> dict[str, Any]:
        safe_pose, table_record = _pose_with_generation_time_table_clearance(
            pose,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool or [],
            clearance_m=table_clearance_m,
            allow_contact_penetration=allow_contact_penetration,
        )
        if table_record is not None and table_clearance_records is not None:
            table_clearance_records.append({"stage": stage, **table_record})
        return safe_pose

    def append_linear_segment(
        *,
        segment_start: dict[str, Any],
        segment_end: dict[str, Any],
        stage_prefix: str,
    ) -> None:
        translation = norm(sub(segment_end["position"], segment_start["position"]))
        rotation = quat_angle_rad(
            segment_end["orientation_xyzw"],
            segment_start["orientation_xyzw"],
        )
        segments = max(
            1,
            math.ceil(translation / max_translation_m),
            math.ceil(rotation / max_rotation_rad),
        )
        for segment in range(1, segments):
            alpha = segment / float(segments)
            pose = {
                "position": [
                    segment_start["position"][i]
                    + alpha * (segment_end["position"][i] - segment_start["position"][i])
                    for i in range(3)
                ],
                "orientation_xyzw": quat_slerp(
                    segment_start["orientation_xyzw"],
                    segment_end["orientation_xyzw"],
                    alpha,
                ),
            }
            transition_stage = f"{stage_prefix}_{segment}"
            pose = make_table_safe(pose, stage=transition_stage)
            goals.append(
                _goal_record(
                    index=len(goals),
                    stage=transition_stage,
                    pose=pose,
                    source_constraints=source_constraints,
                    primary_tool_ref=primary_tool_ref,
                    primary_target_ref=primary_target_ref,
                    resolver=resolver,
                    duration_sec=0.35,
                    allow_contact_penetration=allow_contact_penetration,
                    semantic_role="generated_transition_waypoint",
                    semantic_gate=False,
                    primary_tool_point_tool=primary_tool_point_tool,
                )
            )

    if clearance_axis_world is None:
        append_linear_segment(
            segment_start=current_pose,
            segment_end=next_pose,
            stage_prefix=f"{stage_name}_transition",
        )
        return

    clearance_axis = unit(clearance_axis_world)
    current_axis_coord = dot(current_pose["position"], clearance_axis)
    next_axis_coord = dot(next_pose["position"], clearance_axis)
    lateral_delta = _orthogonal_component(
        sub(next_pose["position"], current_pose["position"]),
        clearance_axis,
    )
    if (
        current_axis_coord - next_axis_coord >= min_axis_descent_m
        and norm(lateral_delta) >= min_lateral_split_m
    ):
        high_lateral_pose = {
            "position": add(
                next_pose["position"],
                mul(clearance_axis, current_axis_coord - next_axis_coord),
            ),
            "orientation_xyzw": normalize_quat(next_pose["orientation_xyzw"]),
        }
        high_lateral_pose = make_table_safe(
            high_lateral_pose,
            stage=f"{stage_name}_transition_high",
        )
        append_linear_segment(
            segment_start=current_pose,
            segment_end=high_lateral_pose,
            stage_prefix=f"{stage_name}_transition_high",
        )
        goals.append(
            _goal_record(
                index=len(goals),
                stage=f"{stage_name}_transition_high",
                pose=high_lateral_pose,
                source_constraints=source_constraints,
                primary_tool_ref=primary_tool_ref,
                primary_target_ref=primary_target_ref,
                resolver=resolver,
                duration_sec=0.35,
                allow_contact_penetration=allow_contact_penetration,
                semantic_role="generated_transition_waypoint",
                semantic_gate=False,
                primary_tool_point_tool=primary_tool_point_tool,
            )
        )
        append_linear_segment(
            segment_start=high_lateral_pose,
            segment_end=next_pose,
            stage_prefix=f"{stage_name}_transition_descent",
        )
        return

    append_linear_segment(
        segment_start=current_pose,
        segment_end=next_pose,
        stage_prefix=f"{stage_name}_transition",
    )


def _lifted_pose(
    pose: dict[str, Any],
    *,
    lift_m: float,
) -> dict[str, Any]:
    return {
        "position": [
            float(pose["position"][0]),
            float(pose["position"][1]),
            float(pose["position"][2]) + float(lift_m),
        ],
        "orientation_xyzw": normalize_quat(pose["orientation_xyzw"]),
    }


def _append_start_settle_subgoal(
    *,
    goals: list[dict[str, Any]],
    current_pose: dict[str, Any],
    resolver: ReferenceResolver,
    table_top_z: float | None = None,
    bbox_points_tool: list[list[float]] | None = None,
    table_clearance_m: float = 0.0,
    table_clearance_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    default_tool_ref = resolver.default_tool_point_ref()
    pose, table_record = _pose_with_generation_time_table_clearance(
        {
            "position": [float(value) for value in current_pose["position"]],
            "orientation_xyzw": normalize_quat(current_pose["orientation_xyzw"]),
        },
        table_top_z=table_top_z,
        bbox_points_tool=bbox_points_tool or [],
        clearance_m=table_clearance_m,
        allow_contact_penetration=False,
    )
    if table_record is not None and table_clearance_records is not None:
        table_clearance_records.append({"stage": "execution_start_settle", **table_record})
    record = _goal_record(
        index=len(goals),
        stage="execution_start_settle",
        pose=pose,
        source_constraints=["execution:start_settle"],
        primary_tool_ref=default_tool_ref,
        primary_target_ref="execution.start_settle",
        resolver=resolver,
        duration_sec=0.30,
        allow_contact_penetration=False,
        semantic_role="execution_start_settle",
        semantic_gate=False,
    )
    goals.append(record)
    return pose


def _append_grasp_lift_subgoal(
    *,
    goals: list[dict[str, Any]],
    current_pose: dict[str, Any],
    resolver: ReferenceResolver,
    lift_m: float,
    max_translation_m: float,
    max_rotation_rad: float,
    lift_steps: int = 3,
    table_top_z: float | None = None,
    bbox_points_tool: list[list[float]] | None = None,
    table_clearance_m: float = 0.0,
    table_clearance_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Optionally insert an execution-side grasp-lift phase before task contact."""

    if lift_m <= 1e-9:
        return current_pose
    lifted, lifted_table_record = _pose_with_generation_time_table_clearance(
        _lifted_pose(current_pose, lift_m=lift_m),
        table_top_z=table_top_z,
        bbox_points_tool=bbox_points_tool or [],
        clearance_m=table_clearance_m,
        allow_contact_penetration=False,
    )
    if lifted_table_record is not None and table_clearance_records is not None:
        table_clearance_records.append({"stage": "grasp_lift", **lifted_table_record})
    default_tool_ref = resolver.default_tool_point_ref()
    target_ref = "execution.grasp_lift"
    steps = max(
        1,
        int(lift_steps),
        math.ceil(float(lift_m) / max(float(max_translation_m), 1e-6)),
    )
    previous_pose = current_pose
    for step in range(1, steps):
        alpha = step / float(steps)
        waypoint = {
            "position": [
                float(current_pose["position"][0]),
                float(current_pose["position"][1]),
                float(current_pose["position"][2]) + alpha * float(lift_m),
            ],
            "orientation_xyzw": normalize_quat(current_pose["orientation_xyzw"]),
        }
        waypoint, waypoint_table_record = _pose_with_generation_time_table_clearance(
            waypoint,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool or [],
            clearance_m=table_clearance_m,
            allow_contact_penetration=False,
        )
        if waypoint_table_record is not None and table_clearance_records is not None:
            table_clearance_records.append({"stage": f"grasp_lift_ramp_{step}", **waypoint_table_record})
        goals.append(
            _goal_record(
                index=len(goals),
                stage=f"grasp_lift_ramp_{step}",
                pose=waypoint,
                source_constraints=["execution:grasp_lift"],
                primary_tool_ref=default_tool_ref,
                primary_target_ref=target_ref,
                resolver=resolver,
                duration_sec=0.25,
                allow_contact_penetration=False,
                semantic_role="grasp_lift_ramp",
                semantic_gate=False,
            )
        )
        previous_pose = waypoint
    if previous_pose is current_pose:
        _append_transition_if_needed(
            goals=goals,
            current_pose=current_pose,
            next_pose=lifted,
            stage_name="grasp_lift",
            source_constraints=["execution:grasp_lift"],
            primary_tool_ref=default_tool_ref,
            primary_target_ref=target_ref,
            resolver=resolver,
            allow_contact_penetration=False,
            max_translation_m=max_translation_m,
            max_rotation_rad=max_rotation_rad,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool,
            table_clearance_m=table_clearance_m,
            table_clearance_records=table_clearance_records,
        )
    goals.append(
        _goal_record(
            index=len(goals),
            stage="grasp_lift",
            pose=lifted,
            source_constraints=["execution:grasp_lift"],
            primary_tool_ref=default_tool_ref,
            primary_target_ref=target_ref,
            resolver=resolver,
            duration_sec=0.45,
            allow_contact_penetration=False,
            semantic_role="grasp_lift",
            semantic_gate=False,
        )
    )
    return lifted


def _append_tool_axis_align_subgoal(
    *,
    goals: list[dict[str, Any]],
    current_pose: dict[str, Any],
    next_pose: dict[str, Any],
    stage_name: str,
    canonical_stage: CanonicalStageConstraints,
    primary_tool_ref: str,
    primary_target_ref: str,
    resolver: ReferenceResolver,
    allow_contact_penetration: bool,
    max_translation_m: float,
    max_rotation_rad: float,
    min_rotation_rad: float = 0.035,
    clearance_lift_m: float = 0.0,
    primary_tool_point_tool: list[float] | None = None,
    table_top_z: float | None = None,
    bbox_points_tool: list[list[float]] | None = None,
    table_clearance_m: float = 0.0,
    table_clearance_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    """Insert the shared free-space tool-axis alignment phase.

    Every tool uses the same execution skeleton: lift first, satisfy declared
    tool-axis constraints while the tool is still in free space, then move
    the selected contact point into contact or along a path. This keeps
    interaction behavior in one compiler rule instead of per-tool wrappers.
    """

    if not canonical_stage.axis_targets:
        return current_pose, False, False
    rotation = quat_angle_rad(next_pose["orientation_xyzw"], current_pose["orientation_xyzw"])
    if rotation < min_rotation_rad:
        return current_pose, False, False
    working_pose = current_pose
    clearance_inserted = False
    if clearance_lift_m > 1e-9:
        clearance_pose = _lifted_pose(current_pose, lift_m=clearance_lift_m)
        clearance_pose, clearance_table_record = _pose_with_generation_time_table_clearance(
            clearance_pose,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool or [],
            clearance_m=table_clearance_m,
            allow_contact_penetration=False,
        )
        if clearance_table_record is not None and table_clearance_records is not None:
            table_clearance_records.append(
                {
                    "stage": f"{stage_name}_tool_axis_align_clearance",
                    **clearance_table_record,
                }
            )
        clearance_stage = f"{stage_name}_tool_axis_align_clearance"
        _append_transition_if_needed(
            goals=goals,
            current_pose=current_pose,
            next_pose=clearance_pose,
            stage_name=clearance_stage,
            source_constraints=canonical_stage.source_ids,
            primary_tool_ref=primary_tool_ref,
            primary_target_ref=primary_target_ref,
            resolver=resolver,
            allow_contact_penetration=False,
            max_translation_m=max_translation_m,
            max_rotation_rad=max_rotation_rad,
            primary_tool_point_tool=primary_tool_point_tool,
            table_top_z=table_top_z,
            bbox_points_tool=bbox_points_tool,
            table_clearance_m=table_clearance_m,
            table_clearance_records=table_clearance_records,
        )
        goals.append(
            _goal_record(
                index=len(goals),
                stage=clearance_stage,
                pose=clearance_pose,
                source_constraints=canonical_stage.source_ids,
                primary_tool_ref=primary_tool_ref,
                primary_target_ref=primary_target_ref,
                resolver=resolver,
                duration_sec=0.35,
                allow_contact_penetration=False,
                semantic_role="generated_transition_waypoint",
                semantic_gate=False,
                primary_tool_point_tool=primary_tool_point_tool,
            )
        )
        working_pose = clearance_pose
        clearance_inserted = True
    align_orientation = normalize_quat(next_pose["orientation_xyzw"])
    align_point_tool = primary_tool_point_tool
    align_pose = {
        "position": [float(value) for value in working_pose["position"]],
        "orientation_xyzw": align_orientation,
    }
    align_stage = f"{stage_name}_tool_axis_align"
    align_pose, table_clearance_record = _pose_with_generation_time_table_clearance(
        align_pose,
        table_top_z=table_top_z,
        bbox_points_tool=bbox_points_tool or [],
        clearance_m=table_clearance_m,
        allow_contact_penetration=False,
    )
    if table_clearance_record is not None and table_clearance_records is not None:
        table_clearance_records.append(
            {
                "stage": align_stage,
                **table_clearance_record,
            }
        )
    _append_transition_if_needed(
        goals=goals,
        current_pose=working_pose,
        next_pose=align_pose,
        stage_name=align_stage,
        source_constraints=canonical_stage.source_ids,
        primary_tool_ref=primary_tool_ref,
        primary_target_ref=primary_target_ref,
        resolver=resolver,
        allow_contact_penetration=allow_contact_penetration,
        max_translation_m=max_translation_m,
        max_rotation_rad=max_rotation_rad,
        primary_tool_point_tool=align_point_tool,
        table_top_z=table_top_z,
        bbox_points_tool=bbox_points_tool,
        table_clearance_m=table_clearance_m,
        table_clearance_records=table_clearance_records,
    )
    goals.append(
        _goal_record(
            index=len(goals),
            stage=align_stage,
            pose=align_pose,
            source_constraints=canonical_stage.source_ids,
            primary_tool_ref=primary_tool_ref,
            primary_target_ref=primary_target_ref,
            resolver=resolver,
            duration_sec=0.45,
            allow_contact_penetration=allow_contact_penetration,
            semantic_role="tool_axis_align",
            semantic_gate=False,
            primary_tool_point_tool=align_point_tool,
        )
    )
    return align_pose, True, clearance_inserted


def _stage_allows_contact_penetration(canonical_stage: CanonicalStageConstraints) -> bool:
    return any(
        bool(getattr(target, "allows_contact_penetration", False))
        for target in canonical_stage.point_targets
    )


def _pre_contact_point_target(
    *,
    canonical_stage: CanonicalStageConstraints,
    primary_tool_ref: str,
) -> Any | None:
    for target in canonical_stage.point_targets:
        if target.source_ref != primary_tool_ref:
            continue
        if target.pre_contact_axis_world is None:
            continue
        if target.pre_contact_target_world is None:
            continue
        return target
    return None


def _is_pre_contact_target_ref(target_ref: str | None) -> bool:
    if not isinstance(target_ref, str):
        return False
    lowered = target_ref.lower()
    return "pre_contact" in lowered or "precontact" in lowered


def _is_axis_target_ref(target_ref: str | None) -> bool:
    if not isinstance(target_ref, str):
        return False
    lowered = target_ref.lower()
    return lowered.startswith("scene.axes.") or lowered.startswith("target.axes.")


def _primary_refs_from_objectives(
    *,
    objectives: list[dict[str, Any]],
    resolver: ReferenceResolver,
) -> tuple[str | None, str | None]:
    """Infer display/report primary refs for objective-only stages.

    Symbolic constraints already carry explicit source/target slots. A
    ReKep-style stage may instead carry only scalar objectives. In that case we
    keep the optimizer behavior unchanged but use the first listed tool-frame
    point and first listed world-frame point from the objective whitelist as the
    expected functional contact in ToolPoseSubgoals evidence.
    """

    primary_tool_ref: str | None = None
    primary_target_ref: str | None = None
    for objective in objectives:
        refs = objective.get("refs") or []
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, str):
                continue
            try:
                resolved = resolver.point(ref)
            except Exception:  # noqa: BLE001 - non-point objective refs are valid.
                continue
            if primary_tool_ref is None and resolved.point_tool is not None:
                primary_tool_ref = ref
            elif primary_target_ref is None and resolved.point_world is not None:
                primary_target_ref = ref
            if primary_tool_ref is not None and primary_target_ref is not None:
                return primary_tool_ref, primary_target_ref
    return primary_tool_ref, primary_target_ref


def _orthogonal_component(vector: list[float], axis: list[float]) -> list[float]:
    return sub(vector, mul(axis, dot(vector, axis)))


def _compiler_stage_record(
    *,
    stage_name: str,
    canonical_stage: CanonicalStageConstraints,
    solve_result: StageSolveResult,
    source_objectives: list[str] | None = None,
) -> dict[str, Any]:
    path_constraints = {
        "tool_occupancy_clearance": [
            {
                "constraint_id": target.constraint_id,
                "tool_part_ref": target.tool_part_ref,
                "obstacle_ref": target.obstacle_ref,
                "samples": target.samples,
                "num_sample_points": len(target.sample_points_tool),
                "min_distance_m": target.min_distance_m,
            }
            for target in canonical_stage.tool_occupancy_clearance_targets
        ],
    }
    return {
        "name": stage_name,
        "success": not solve_result.warnings,
        "final_cost": solve_result.final_cost,
        "source_constraints": canonical_stage.source_ids,
        "source_objectives": source_objectives or [],
        "pose_selection": solve_result.pose_selection,
        "path_constraints": path_constraints,
        "motion_paths": [
            {
                "constraint_id": target.constraint_id,
                "source_ref": target.source_ref,
                "start_ref": target.start_ref,
                "goal_ref": target.goal_ref,
                "profile": target.profile,
                "path_family": target.path_family,
                "swing_axis_ref": target.swing_axis_ref,
                "swing_angle_rad": target.swing_angle_rad,
                "swing_lift_m": target.swing_lift_m,
            }
            for target in canonical_stage.move_along_targets
        ],
        "constraint_errors": solve_result.constraint_errors,
        "warnings": solve_result.warnings,
    }


def _lookahead_stage_contact_target(
    *,
    program_ir: dict[str, Any],
    stage_index: int,
    resolver: ReferenceResolver,
    current_pose: dict[str, Any],
    locked_tool_points: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Find the next functional point target for an axis-only alignment stage."""

    for next_stage in program_ir["stages"][stage_index + 1 :]:
        next_constraints = _constraints_for_stage(program_ir, str(next_stage["name"]))
        if not next_constraints:
            continue
        canonical_next = canonicalize_stage_constraints(
            stage_name=str(next_stage["name"]),
            constraints=next_constraints,
            resolver=resolver,
            current_pose=current_pose,
        )
        canonical_next = _stage_with_locked_tool_points(
            canonical_next,
            locked_tool_points,
        )
        if canonical_next.point_targets:
            target = canonical_next.point_targets[0]
            return {
                "source_ref": target.source_ref,
                "target_ref": target.target_ref,
                "point_tool": [float(value) for value in target.point_tool],
                "point_world": [float(value) for value in target.point_world],
                "lookahead_stage": str(next_stage["name"]),
                "lookahead_kind": "point_target",
            }
        if canonical_next.move_along_targets:
            target = canonical_next.move_along_targets[0]
            return {
                "source_ref": target.source_ref,
                "target_ref": target.start_ref,
                "point_tool": [float(value) for value in target.point_tool],
                "point_world": [float(value) for value in target.start_world],
                "lookahead_stage": str(next_stage["name"]),
                "lookahead_kind": "move_start",
            }
    return None


def _validate_inputs(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    program_ir: dict[str, Any],
    grounding_candidates: dict[str, Any] | None,
    target_part_graph: dict[str, Any] | None,
    tool_part_graph: dict[str, Any] | None,
) -> None:
    validate_observed_scene_graph(observed_scene_graph)
    validate_functional_tool_parts(functional_tool_parts)
    validate_program_ir(program_ir)
    if grounding_candidates:
        validate_grounding_candidates(grounding_candidates)
    if target_part_graph:
        validate_target_part_graph(target_part_graph)
    if tool_part_graph:
        validate_tool_part_graph(tool_part_graph)


def compile_program_ir_to_tool_pose_subgoals(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    program_ir: dict[str, Any],
    start_pose: dict[str, Any],
    planner_program: dict[str, Any] | str | None = None,
    grounding_candidates: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    subgoals_id: str | None = None,
    source_trace: dict[str, Any] | None = None,
    move_along_steps: int = 1,
    max_translation_step_m: float = 0.04,
    max_rotation_step_rad: float = 0.35,
    initial_grasp_lift_m: float = 0.08,
    initial_grasp_lift_steps: int = 3,
    tool_axis_align_clearance_lift_m: float = 0.0,
) -> dict[str, Any]:
    """Compile ProgramIR into a ToolPoseSubgoals artifact."""

    _validate_inputs(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        program_ir=program_ir,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
    )

    resolver = ReferenceResolver(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
    )
    current_pose = {
        "position": [float(x) for x in start_pose["position"]],
        "orientation_xyzw": normalize_quat(start_pose["orientation_xyzw"]),
    }
    goals: list[dict[str, Any]] = []
    compiler_stages: list[dict[str, Any]] = []
    objective_functions: dict[str, ObjectiveFunction] = {}
    objective_warnings: list[str] = []
    subgoal_objectives = [
        dict(objective)
        for objective in program_ir.get("objectives", [])
        if objective.get("kind") == "subgoal"
    ]
    path_objectives = [
        dict(objective)
        for objective in program_ir.get("objectives", [])
        if objective.get("kind") == "path"
    ]
    if subgoal_objectives and planner_program is not None:
        try:
            objective_functions = load_objective_functions(
                planner_program=planner_program,
                program_ir={**program_ir, "objectives": subgoal_objectives},
            )
        except ObjectiveRuntimeError as exc:
            raise ProgramIRCompileError(f"failed to load planner objectives: {exc}") from exc
    elif subgoal_objectives:
        raise ProgramIRCompileError(
            "ProgramIR contains subgoal objectives but planner_program source "
            "was not provided to the compiler"
        )
    compiled_objective_ids: set[str] = set()
    path_objective_ids = [
        str(objective.get("id") or objective.get("function_name"))
        for objective in path_objectives
    ]
    ignored_objectives = len(subgoal_objectives)
    tool_axis_align_waypoints = 0
    tool_axis_align_clearance_waypoints = 0
    locked_tool_points: dict[str, list[float]] = {}
    table_clearance_sample_points_tool = (
        resolver.table_clearance_sample_points_tool()
        if DEFAULT_ENABLE_ALL_TOOL_PART_OBB_TABLE_PROJECTION
        else []
    )
    table_top_z = _support_table_top_z(observed_scene_graph)
    table_clearance_context = (
        {
            "table_top_z": table_top_z,
            "bbox_points_tool": table_clearance_sample_points_tool,
            "clearance_m": DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
        }
        if table_top_z is not None and table_clearance_sample_points_tool
        else None
    )
    generation_table_clearance_records: list[dict[str, Any]] = []
    current_pose = _append_start_settle_subgoal(
        goals=goals,
        current_pose=current_pose,
        resolver=resolver,
        table_top_z=table_top_z,
        bbox_points_tool=table_clearance_sample_points_tool,
        table_clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
        table_clearance_records=generation_table_clearance_records,
    )
    current_pose = _append_grasp_lift_subgoal(
        goals=goals,
        current_pose=current_pose,
        resolver=resolver,
        lift_m=initial_grasp_lift_m,
        max_translation_m=max_translation_step_m,
        max_rotation_rad=max_rotation_step_rad,
        lift_steps=initial_grasp_lift_steps,
        table_top_z=table_top_z,
        bbox_points_tool=table_clearance_sample_points_tool,
        table_clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
        table_clearance_records=generation_table_clearance_records,
    )

    for stage_index, stage in enumerate(program_ir["stages"]):
        stage_name = stage["name"]
        repeat = int(stage.get("repeat", 1))
        constraints = _constraints_for_stage(program_ir, stage_name)
        stage_objectives = _objectives_for_stage(program_ir, stage_name, kind="subgoal")
        active_stage_objectives = stage_objectives if objective_functions else []
        for objective in active_stage_objectives:
            compiled_objective_ids.add(str(objective.get("id") or objective.get("function_name")))
        for repeat_index in range(repeat):
            repeat_suffix = "" if repeat == 1 else f"_{repeat_index + 1}"
            indexed_base_stage = f"{stage_name}{repeat_suffix}"
            canonical_stage = canonicalize_stage_constraints(
                stage_name=stage_name,
                constraints=constraints,
                resolver=resolver,
                current_pose=current_pose,
            )
            canonical_stage = _stage_with_locked_tool_points(
                canonical_stage,
                locked_tool_points,
            )
            objective_cost_fn = None
            objective_error_fn = None
            if active_stage_objectives:
                objective_cost_fn = lambda pose, objectives=active_stage_objectives: pose_objective_cost(
                    pose=pose,
                    objectives=objectives,
                    functions=objective_functions,
                    resolver=resolver,
                )
                objective_error_fn = (
                    lambda pose, objectives=active_stage_objectives: pose_objective_residuals(
                        pose=pose,
                        objectives=objectives,
                        functions=objective_functions,
                        resolver=resolver,
                    )
                )
            try:
                line_proximity_target = None
                if (
                    canonical_stage.axis_targets
                    and not canonical_stage.point_targets
                    and not canonical_stage.move_along_targets
                ):
                    line_proximity_target = _lookahead_stage_contact_target(
                        program_ir=program_ir,
                        stage_index=stage_index,
                        resolver=resolver,
                        current_pose=current_pose,
                        locked_tool_points=locked_tool_points,
                    )
                solve_result = solve_stage_pose(
                    stage=canonical_stage,
                    current_pose=current_pose,
                    default_tool_point_ref=resolver.default_tool_point_ref(),
                    objective_cost_fn=objective_cost_fn,
                    objective_error_fn=objective_error_fn,
                    line_proximity_target=line_proximity_target,
                    table_clearance_context=table_clearance_context,
                )
            except Exception as exc:  # noqa: BLE001 - normalize compiler-stage failures.
                raise ProgramIRCompileError(f"failed to solve stage {stage_name}: {exc}") from exc
            for point_resolution in solve_result.pose_selection.get("point_resolutions", []):
                if not isinstance(point_resolution, dict):
                    continue
                source_ref = point_resolution.get("source_ref")
                selected_point = point_resolution.get("selected_source_point_tool")
                if isinstance(source_ref, str) and isinstance(selected_point, list) and len(selected_point) == 3:
                    locked_tool_points[source_ref] = [float(item) for item in selected_point]
            if active_stage_objectives and not canonical_stage.point_targets:
                objective_tool_ref, objective_target_ref = _primary_refs_from_objectives(
                    objectives=active_stage_objectives,
                    resolver=resolver,
                )
                if objective_tool_ref is not None:
                    solve_result = replace(
                        solve_result,
                        primary_tool_ref=objective_tool_ref,
                        primary_target_ref=objective_target_ref
                        or solve_result.primary_target_ref,
                    )
            compiler_stages.append(
                _compiler_stage_record(
                    stage_name=indexed_base_stage,
                    canonical_stage=canonical_stage,
                    solve_result=solve_result,
                    source_objectives=[
                        str(objective.get("id") or objective.get("function_name"))
                        for objective in active_stage_objectives
                    ],
                )
            )

            allow_contact_penetration = _stage_allows_contact_penetration(canonical_stage)
            alignment_tool_ref = solve_result.primary_tool_ref
            alignment_target_ref = solve_result.primary_target_ref
            if canonical_stage.move_along_targets and not canonical_stage.point_targets:
                alignment_tool_ref = canonical_stage.move_along_targets[0].source_ref
                alignment_target_ref = canonical_stage.move_along_targets[0].start_ref
            (
                current_pose,
                axis_align_inserted,
                axis_align_clearance_inserted,
            ) = _append_tool_axis_align_subgoal(
                goals=goals,
                current_pose=current_pose,
                next_pose=solve_result.pose,
                stage_name=indexed_base_stage,
                canonical_stage=canonical_stage,
                primary_tool_ref=alignment_tool_ref,
                primary_target_ref=alignment_target_ref,
                resolver=resolver,
                allow_contact_penetration=allow_contact_penetration,
                max_translation_m=max_translation_step_m,
                max_rotation_rad=max_rotation_step_rad,
                clearance_lift_m=tool_axis_align_clearance_lift_m,
                primary_tool_point_tool=solve_result.primary_tool_point,
                table_top_z=table_top_z,
                bbox_points_tool=table_clearance_sample_points_tool,
                table_clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
                table_clearance_records=generation_table_clearance_records,
            )
            if axis_align_inserted:
                tool_axis_align_waypoints += 1
            if axis_align_clearance_inserted:
                tool_axis_align_clearance_waypoints += 1

            if canonical_stage.move_along_targets:
                stage_targets = []
                for move in canonical_stage.move_along_targets:
                    compiled_move_targets = compile_move_along_waypoints(
                        move=move,
                        orientation_xyzw=solve_result.pose["orientation_xyzw"],
                        current_pose=current_pose,
                        steps=move_along_steps,
                        table_clearance_context=table_clearance_context,
                    )
                    if (
                        move.source_ref not in locked_tool_points
                        and compiled_move_targets
                        and compiled_move_targets[0].point_tool is not None
                    ):
                        locked_tool_points[move.source_ref] = [
                            float(item) for item in compiled_move_targets[0].point_tool
                        ]
                    stage_targets.extend(compiled_move_targets)
            else:
                is_pre_contact_anchor = (
                    _pre_contact_point_target(
                        canonical_stage=canonical_stage,
                        primary_tool_ref=solve_result.primary_tool_ref,
                    )
                    is not None
                    or _is_pre_contact_target_ref(solve_result.primary_target_ref)
                )
                is_alignment_anchor = _is_axis_target_ref(solve_result.primary_target_ref)
                stage_targets = [
                    {
                        "pose": solve_result.pose,
                        "tool_ref": solve_result.primary_tool_ref,
                        "target_ref": solve_result.primary_target_ref,
                        "semantic_role": "pre_contact_anchor"
                        if is_pre_contact_anchor
                        else "alignment_anchor"
                        if is_alignment_anchor
                        else "semantic_anchor",
                        "semantic_gate": False
                        if (is_pre_contact_anchor or is_alignment_anchor)
                        else True,
                        "point_tool": solve_result.primary_tool_point,
                    }
                ]

            for target_index, stage_target in enumerate(stage_targets):
                if isinstance(stage_target, tuple):
                    next_pose = stage_target[0]
                    tool_ref = stage_target[1]
                    target_ref = stage_target[2]
                    semantic_role = stage_target[3] if len(stage_target) > 3 else None
                    semantic_gate = stage_target[4] if len(stage_target) > 4 else None
                    primary_tool_point_tool = getattr(stage_target, "point_tool", None)
                else:
                    next_pose = stage_target["pose"]
                    tool_ref = stage_target["tool_ref"]
                    target_ref = stage_target["target_ref"]
                    semantic_role = stage_target.get("semantic_role")
                    semantic_gate = stage_target.get("semantic_gate")
                    primary_tool_point_tool = stage_target.get("point_tool") or solve_result.primary_tool_point
                next_pose = apply_clearance_constraints(
                    pose=next_pose,
                    clearance_targets=canonical_stage.clearance_targets,
                )
                indexed_stage = indexed_base_stage
                if len(stage_targets) > 1:
                    indexed_stage = f"{indexed_base_stage}_waypoint_{target_index + 1}"
                next_pose, table_clearance_record = _pose_with_generation_time_table_clearance(
                    next_pose,
                    table_top_z=table_top_z,
                    bbox_points_tool=table_clearance_sample_points_tool,
                    clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
                    allow_contact_penetration=allow_contact_penetration,
                )
                if table_clearance_record is not None:
                    generation_table_clearance_records.append(
                        {
                            "stage": indexed_stage,
                            **table_clearance_record,
                        }
                    )
                _append_transition_if_needed(
                    goals=goals,
                    current_pose=current_pose,
                    next_pose=next_pose,
                    stage_name=indexed_stage,
                    source_constraints=canonical_stage.source_ids,
                    primary_tool_ref=tool_ref,
                    primary_target_ref=target_ref,
                    resolver=resolver,
                    allow_contact_penetration=allow_contact_penetration,
                    max_translation_m=max_translation_step_m,
                    max_rotation_rad=max_rotation_step_rad,
                    primary_tool_point_tool=primary_tool_point_tool,
                    table_top_z=table_top_z,
                    bbox_points_tool=table_clearance_sample_points_tool,
                    table_clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
                    table_clearance_records=generation_table_clearance_records,
                )
                goals.append(
                    _goal_record(
                        index=len(goals),
                        stage=indexed_stage,
                        pose=next_pose,
                        source_constraints=canonical_stage.source_ids,
                        primary_tool_ref=tool_ref,
                        primary_target_ref=target_ref,
                        resolver=resolver,
                        allow_contact_penetration=allow_contact_penetration,
                        semantic_role=semantic_role,
                        semantic_gate=semantic_gate,
                        primary_tool_point_tool=primary_tool_point_tool,
                    )
                )
                current_pose = next_pose

    ignored_objectives = max(0, len(subgoal_objectives) - len(compiled_objective_ids))

    if not goals:
        raise ProgramIRCompileError("ProgramIR did not compile to any tool-pose goals")
    hard_table_clearance_invariant = _validate_generated_goals_table_clearance(
        goals=goals,
        table_top_z=table_top_z,
        bbox_points_tool=table_clearance_sample_points_tool,
        clearance_m=DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M,
    )
    applied_records = [
        record
        for record in generation_table_clearance_records
        if record.get("applied") is True
    ]
    table_bbox_clearance_projection = {
        "enabled": bool(
            _support_table_top_z(observed_scene_graph) is not None
            and table_clearance_sample_points_tool
        ),
        "mode": "native_table_bbox_constraint_assertion",
        "sample_scope": "all_tool_part_obbs",
        "table_top_z": table_top_z,
        "clearance_m": float(DEFAULT_SUBGOAL_TABLE_PART_OBB_CLEARANCE_M),
        "num_bbox_corners": len(table_clearance_sample_points_tool),
        "applied_count": len(applied_records),
        "max_lift_m": max(
            [float(record.get("lift_m", 0.0)) for record in applied_records],
            default=0.0,
        ),
        "skipped_functional_contact_count": 0,
        "native_generation_constraint": True,
        "posthoc_pose_correction": False,
        "hard_invariant": hard_table_clearance_invariant,
        "records": generation_table_clearance_records,
    }
    goal_pose_deltas, pose_delta_summary = _pose_delta_records(
        start_pose=start_pose,
        goals=goals,
    )

    subgoals = {
        "schema_version": "v0.2",
        "subgoals_id": subgoals_id or f"{program_ir['program_id']}_tool_pose_subgoals",
        "tool_id": functional_tool_parts["tool_id"],
        "task_id": program_ir["program_id"],
        "world_frame": "world",
        "units": "meters",
        "quaternion_order": "xyzw",
        "start_pose": start_pose,
        "goals": goals,
        "source_trace": source_trace
        or {
            "observed_scene_graph": {
                "id": observed_scene_graph["scene_id"],
                "path": "<in-memory>",
            },
            "grounding_candidates": {
                "id": (grounding_candidates or {}).get("grounding_id", "<in-memory>"),
                "path": "<in-memory>",
            },
            "functional_tool_parts": {
                "id": functional_tool_parts["tool_id"],
                "path": "<in-memory>",
            },
            "planner_program": {
                "id": program_ir["program_id"],
                "path": "<in-memory>",
            },
            "program_ir": {
                "id": program_ir["program_id"],
                "path": "<in-memory>",
            },
        },
        "sanity_checks": {
            "compiled": True,
            "compiler": "parttoolgen.tool_pose_compiler.stagewise_se3_v0",
            "num_program_stages": len(program_ir["stages"]),
            "num_goals": len(goals),
            "compiled_objectives": len(compiled_objective_ids),
            "ignored_objectives": ignored_objectives,
            "objective_runtime": "restricted_pose_cost_v0" if objective_functions else "not_loaded",
            "objective_warnings": objective_warnings,
            "path_objectives_declared": len(path_objectives),
            "path_objectives_deferred": len(path_objectives),
            "path_objective_ids": path_objective_ids,
            "path_objective_runtime": "deferred_to_tool_path_solver" if path_objectives else "none",
            "move_along_steps": move_along_steps,
            "initial_grasp_lift_m": float(initial_grasp_lift_m),
            "initial_grasp_lift_steps": int(initial_grasp_lift_steps),
            "initial_grasp_lift_inserted": initial_grasp_lift_m > 1e-9,
            "execution_start_settle_inserted": True,
            "tool_axis_align_clearance_lift_m": float(
                tool_axis_align_clearance_lift_m
            ),
            "tool_axis_align_clearance_waypoints": (
                tool_axis_align_clearance_waypoints
            ),
            "tool_axis_align_waypoints": tool_axis_align_waypoints,
            "table_bbox_clearance_projection": table_bbox_clearance_projection,
            "pose_delta_summary": pose_delta_summary,
            "goal_pose_deltas": goal_pose_deltas,
            "compiler_stages": compiler_stages,
            "locked_tool_points": locked_tool_points,
        },
        "metadata": {
            "generated_by": "parttoolgen.tool_pose_compiler",
            "source": "ObservedSceneGraph+FunctionalToolParts+ProgramIR",
            "geometry_mode": observed_scene_graph["metadata"]["geometry_mode"],
        },
    }
    validate_tool_pose_subgoals(subgoals)
    return subgoals


def compile_files(
    *,
    scene_path: str | Path,
    tool_path: str | Path,
    program_ir_path: str | Path,
    start_pose_path: str | Path,
    output_path: str | Path,
    planner_program_path: str | Path | None = None,
    grounding_candidates_path: str | Path | None = None,
    target_part_graph_path: str | Path | None = None,
    tool_part_graph_path: str | Path | None = None,
) -> dict[str, Any]:
    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=load_structured_file(scene_path),
        functional_tool_parts=load_structured_file(tool_path),
        program_ir=load_structured_file(program_ir_path),
        start_pose=load_structured_file(start_pose_path),
        planner_program=load_structured_file(planner_program_path)
        if planner_program_path
        else None,
        grounding_candidates=load_structured_file(grounding_candidates_path)
        if grounding_candidates_path
        else None,
        target_part_graph=load_structured_file(target_part_graph_path)
        if target_part_graph_path
        else None,
        tool_part_graph=load_structured_file(tool_part_graph_path)
        if tool_part_graph_path
        else None,
        source_trace={
            "observed_scene_graph": {"id": "observed_scene_graph", "path": str(scene_path)},
            "grounding_candidates": {
                "id": "grounding_candidates",
                "path": str(grounding_candidates_path or "<none>"),
            },
            "functional_tool_parts": {"id": "functional_tool_parts", "path": str(tool_path)},
            "planner_program": {
                "id": "planner_program",
                "path": str(planner_program_path or "<not-provided-to-compiler>"),
            },
            "program_ir": {"id": "program_ir", "path": str(program_ir_path)},
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subgoals, indent=2) + "\n", encoding="utf-8")
    return subgoals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--program-ir", required=True)
    parser.add_argument("--start-pose", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--planner-program")
    parser.add_argument("--candidates")
    parser.add_argument("--target-parts")
    parser.add_argument("--tool-parts")
    args = parser.parse_args()
    compile_files(
        scene_path=args.scene,
        tool_path=args.tool,
        program_ir_path=args.program_ir,
        start_pose_path=args.start_pose,
        output_path=args.output,
        planner_program_path=args.planner_program,
        grounding_candidates_path=args.candidates,
        target_part_graph_path=args.target_parts,
        tool_part_graph_path=args.tool_parts,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
