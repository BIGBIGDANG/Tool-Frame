#!/usr/bin/env python3
"""Run the PartToolGen mainline artifact chain to SimToolReal fixed goals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ToGo.adapters.simtoolreal_adapter import convert_subgoals_to_simtoolreal
from ToGo.adapters.simtoolreal_policy_priors import (
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
    table_contact_start_pose_from_state_bundle,
    table_top_z_from_state_bundle,
    tool_mesh_z_bounds_for_pose,
)
from ToGo.compiler.tool_path_solver import solve_collision_safe_tool_path
from ToGo.compiler.tool_pose_compiler import compile_program_ir_to_tool_pose_subgoals
from ToGo.validation.validate_parttoolgen import load_structured_file


class MainlineArtifactChainError(ValueError):
    """Raised when the mainline artifact chain cannot safely continue."""


def run_mainline_fixed_goal_chain(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    program_ir: dict[str, Any],
    planner_program: dict[str, Any] | str,
    start_pose: dict[str, Any],
    current_start_pose: dict[str, Any] | None = None,
    state_bundle: dict[str, Any] | None = None,
    grounding_candidates: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    tool_mesh_path: str | Path | None = None,
    tool_id: str | None = None,
    project_start_pose_to_table: bool | None = None,
    start_min_tool_table_clearance_m: float = 0.003,
    move_along_steps: int = 1,
    max_translation_step_m: float = 0.04,
    max_rotation_step_rad: float = 0.35,
    initial_grasp_lift_m: float = 0.20,
    initial_grasp_lift_steps: int = 3,
    min_clearance_m: float = 0.003,
    swept_min_clearance_m: float = 0.006,
    allowed_contact_penetration_m: float = 0.006,
    allowed_table_contact_penetration_m: float | None = None,
    support_plane_point_world: list[float] | None = None,
    support_plane_normal_world: list[float] | None = None,
    path_objective_samples_per_segment: int = 9,
    path_objective_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compile, path-check, and adapt ProgramIR through the strict mainline.

    The VLM never writes fixed-goal JSON here. It can only contribute refs,
    constraints, and restricted objectives through ProgramIR. This runner always
    routes compiled whole-tool subgoals through ToolPathSolver before invoking
    the SimToolReal fixed-goal adapter.
    """

    start_pose_report: dict[str, Any] = {
        "mode": "provided",
        "input_start_pose": start_pose,
        "used_start_pose": start_pose,
    }
    if current_start_pose is not None:
        start_pose_report = {
            "mode": "observed_current_object_pose",
            "input_start_pose": start_pose,
            "used_start_pose": current_start_pose,
            "source": "current_object_pose_provider",
        }
        start_pose = current_start_pose
    should_project_start_pose = (
        state_bundle is not None
        if project_start_pose_to_table is None
        else project_start_pose_to_table
    )
    if current_start_pose is None and should_project_start_pose:
        if state_bundle is None:
            raise MainlineArtifactChainError(
                "project_start_pose_to_table requires a SimToolReal state bundle"
            )
        input_start_pose = start_pose
        start_pose = table_contact_start_pose_from_state_bundle(
            start_pose=start_pose,
            state_bundle=state_bundle,
            tool_id=tool_id,
            tool_mesh_path=tool_mesh_path,
            min_tool_table_clearance_m=start_min_tool_table_clearance_m,
        )
        table_top_z = table_top_z_from_state_bundle(state_bundle)
        resolved_mesh = resolve_simtoolreal_tool_mesh_path(
            tool_id,
            explicit_path=tool_mesh_path,
        )
        mesh_bounds = None
        if resolved_mesh is not None:
            mesh_bounds = tool_mesh_z_bounds_for_pose(
                [
                    *start_pose["position"],
                    *start_pose["orientation_xyzw"],
                ],
                load_obj_vertices(resolved_mesh),
            )
        start_pose_report = {
            "mode": "mesh_table_contact_projection",
            "input_start_pose": input_start_pose,
            "used_start_pose": start_pose,
            "table_top_z": table_top_z,
            "tool_mesh_path": str(resolved_mesh) if resolved_mesh else None,
            "projected_mesh_z_bounds": list(mesh_bounds) if mesh_bounds else None,
            "min_tool_table_clearance_m": float(start_min_tool_table_clearance_m),
            "source": "state_bundle_table_geometry_plus_tool_mesh_asset_root",
        }

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        program_ir=program_ir,
        planner_program=planner_program,
        start_pose=start_pose,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
        move_along_steps=move_along_steps,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
        initial_grasp_lift_m=initial_grasp_lift_m,
        initial_grasp_lift_steps=initial_grasp_lift_steps,
        support_plane_point_world=support_plane_point_world,
        support_plane_normal_world=support_plane_normal_world,
    )
    table_projection_report = {
        "enabled": False,
        "reason": "table_safety_is_enforced_by_compiler_bbox_projection",
    }
    path_solved_subgoals, path_report = solve_collision_safe_tool_path(
        subgoals,
        observed_scene_graph,
        program_ir=program_ir,
        planner_program=planner_program,
        functional_tool_parts=functional_tool_parts,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
        tool_id=tool_id,
        tool_mesh_path=tool_mesh_path,
        min_clearance_m=min_clearance_m,
        swept_min_clearance_m=swept_min_clearance_m,
        allowed_contact_penetration_m=allowed_contact_penetration_m,
        path_objective_samples_per_segment=path_objective_samples_per_segment,
        path_objective_tolerance=path_objective_tolerance,
    )
    if path_report.get("passed") is not True:
        path_report = {
            **path_report,
            "failure_reason": path_report.get("failure_reason") or "mainline_path_gate_failed",
            "mainline_path_policy": (
                "Current image-grounded mainline reports infeasible paths instead "
                "of inserting corridor/truncation/tracking waypoints."
            ),
        }
        partial_result = {
            "tool_pose_subgoals": subgoals,
            "path_solved_subgoals": path_solved_subgoals,
            "tool_path_solver_report": path_report,
            "start_pose_report": start_pose_report,
            "table_projection_report": table_projection_report,
        }
        exc = MainlineArtifactChainError(
            "ToolPathSolver did not pass; use tool_path_solver_report as repair evidence"
        )
        setattr(exc, "parttoolgen_partial_result", partial_result)
        raise exc
    fixed_goals = convert_subgoals_to_simtoolreal(path_solved_subgoals)
    return {
        "tool_pose_subgoals": subgoals,
        "path_solved_subgoals": path_solved_subgoals,
        "tool_path_solver_report": path_report,
        "simtoolreal_fixed_goals": fixed_goals,
        "start_pose_report": start_pose_report,
        "table_projection_report": table_projection_report,
    }


def _json_dump(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_mainline_fixed_goal_chain_files(
    *,
    scene_path: str | Path,
    tool_path: str | Path,
    program_ir_path: str | Path,
    planner_program_path: str | Path,
    start_pose_path: str | Path,
    output_dir: str | Path,
    current_start_pose_path: str | Path | None = None,
    state_bundle_path: str | Path | None = None,
    grounding_candidates_path: str | Path | None = None,
    target_part_graph_path: str | Path | None = None,
    tool_part_graph_path: str | Path | None = None,
    tool_mesh_path: str | Path | None = None,
    tool_id: str | None = None,
    project_start_pose_to_table: bool | None = None,
    start_min_tool_table_clearance_m: float = 0.003,
    swept_min_clearance_m: float = 0.006,
    allowed_contact_penetration_m: float = 0.006,
    allowed_table_contact_penetration_m: float | None = None,
    initial_grasp_lift_m: float = 0.20,
    initial_grasp_lift_steps: int = 3,
) -> dict[str, Any]:
    output = Path(output_dir)
    try:
        result = run_mainline_fixed_goal_chain(
            observed_scene_graph=load_structured_file(scene_path),
            functional_tool_parts=load_structured_file(tool_path),
            program_ir=load_structured_file(program_ir_path),
            planner_program=load_structured_file(planner_program_path),
            start_pose=load_structured_file(start_pose_path),
            current_start_pose=load_structured_file(current_start_pose_path)
            if current_start_pose_path
            else None,
            state_bundle=load_structured_file(state_bundle_path)
            if state_bundle_path
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
            tool_mesh_path=tool_mesh_path,
            tool_id=tool_id,
            project_start_pose_to_table=project_start_pose_to_table,
            start_min_tool_table_clearance_m=start_min_tool_table_clearance_m,
            swept_min_clearance_m=swept_min_clearance_m,
            allowed_contact_penetration_m=allowed_contact_penetration_m,
            allowed_table_contact_penetration_m=allowed_table_contact_penetration_m,
            initial_grasp_lift_m=initial_grasp_lift_m,
            initial_grasp_lift_steps=initial_grasp_lift_steps,
        )
    except MainlineArtifactChainError as exc:
        partial = getattr(exc, "parttoolgen_partial_result", None)
        if isinstance(partial, dict):
            if "tool_pose_subgoals" in partial:
                _json_dump(output / "tool_pose_subgoals.json", partial["tool_pose_subgoals"])
            if "path_solved_subgoals" in partial:
                _json_dump(
                    output / "path_solved_tool_pose_subgoals.json",
                    partial["path_solved_subgoals"],
                )
            if "tool_path_solver_report" in partial:
                _json_dump(output / "tool_path_solver_report.json", partial["tool_path_solver_report"])
            if "start_pose_report" in partial:
                _json_dump(output / "start_pose_report.json", partial["start_pose_report"])
            if "table_projection_report" in partial:
                _json_dump(output / "table_projection_report.json", partial["table_projection_report"])
        raise
    _json_dump(output / "tool_pose_subgoals.json", result["tool_pose_subgoals"])
    _json_dump(output / "path_solved_tool_pose_subgoals.json", result["path_solved_subgoals"])
    _json_dump(output / "tool_path_solver_report.json", result["tool_path_solver_report"])
    _json_dump(output / "simtoolreal_fixed_goals.json", result["simtoolreal_fixed_goals"])
    _json_dump(output / "start_pose_report.json", result["start_pose_report"])
    _json_dump(output / "table_projection_report.json", result["table_projection_report"])
    return result
