#!/usr/bin/env python3
"""Run the PartToolGen mainline artifact chain to SimToolReal fixed goals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from parttoolgen.adapters.simtoolreal_adapter import convert_subgoals_to_simtoolreal
from parttoolgen.adapters.simtoolreal_policy_priors import (
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
    table_contact_start_pose_from_state_bundle,
    table_top_z_from_state_bundle,
    tool_mesh_z_bounds_for_pose,
)
from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.compiler.residuals import pose_point
from parttoolgen.compiler.tool_path_solver import solve_collision_safe_tool_path
from parttoolgen.compiler.tool_pose_compiler import compile_program_ir_to_tool_pose_subgoals
from parttoolgen.validation.validate_parttoolgen import load_structured_file


class MainlineArtifactChainError(ValueError):
    """Raised when the mainline artifact chain cannot safely continue."""


def _refresh_contact_point_goal(
    *,
    goal: dict[str, Any],
    resolver: ReferenceResolver,
) -> None:
    functional = goal.get("contact_point_goal")
    pose = goal.get("pose_world_tool")
    if not isinstance(functional, dict) or not isinstance(pose, dict):
        return
    part_ref = functional.get("part_ref")
    if not isinstance(part_ref, str) or not part_ref:
        return
    try:
        tool_point = resolver.tool_point(part_ref)
    except Exception:  # noqa: BLE001 - keep safety projection best-effort.
        return
    functional["position_world"] = pose_point(pose, tool_point.point_tool or [])
    functional["orientation_xyzw"] = pose["orientation_xyzw"]


def _is_contact_path_stage(goal: dict[str, Any]) -> bool:
    """Return true for stages where the functional ref is the hard anchor."""

    stage = str(goal.get("stage") or "")
    if stage == "approach":
        return True
    return stage.startswith("engage")


def _is_startup_lift_stage(goal: dict[str, Any]) -> bool:
    stage = str(goal.get("stage") or "")
    return (
        stage == "execution_start_settle"
        or stage == "grasp_lift"
        or stage.startswith("grasp_lift_")
    )


def _lift_tool_pose_subgoals_to_table_clearance(
    subgoals: dict[str, Any],
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    state_bundle: dict[str, Any] | None,
    grounding_candidates: dict[str, Any] | None,
    target_part_graph: dict[str, Any] | None,
    tool_part_graph: dict[str, Any] | None,
    tool_id: str | None,
    tool_mesh_path: str | Path | None,
    min_tool_table_clearance_m: float,
    allowed_contact_table_penetration_m: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lift free-space subgoals enough to keep the mesh above the table.

    Contact-path goals may intentionally touch/compress compliant visual
    geometry such as bristles. For those stages, preserve the functional anchor
    while the required mesh lift stays within the configured table penetration
    tolerance; otherwise lift only the excess.
    """

    sanity = subgoals.get("sanity_checks", {})
    compiler_bbox_projection = (
        sanity.get("table_bbox_clearance_projection")
        if isinstance(sanity, dict)
        else None
    )
    if isinstance(compiler_bbox_projection, dict) and compiler_bbox_projection.get("enabled"):
        report = {
            "enabled": False,
            "reason": "skipped_after_compiler_bbox_table_projection",
            "compiler_projection": compiler_bbox_projection,
            "adjusted_goals": [],
            "skipped_contact_goals": [],
            "changed": False,
            "contact_preservation_active": False,
        }
        return subgoals, report

    table_top_z = table_top_z_from_state_bundle(state_bundle or {})
    resolved_mesh = resolve_simtoolreal_tool_mesh_path(
        tool_id,
        explicit_path=tool_mesh_path,
    )
    report: dict[str, Any] = {
        "enabled": bool(table_top_z is not None and resolved_mesh is not None),
        "table_top_z": table_top_z,
        "tool_mesh_path": str(resolved_mesh) if resolved_mesh else None,
        "min_tool_table_clearance_m": float(min_tool_table_clearance_m),
        "allowed_contact_table_penetration_m": float(
            max(0.0, allowed_contact_table_penetration_m)
        ),
        "adjusted_goals": [],
        "skipped_contact_goals": [],
    }
    if table_top_z is None or resolved_mesh is None:
        report["reason"] = "missing table geometry or tool mesh"
        return subgoals, report

    resolver = ReferenceResolver(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
    )
    vertices = load_obj_vertices(resolved_mesh)
    adjusted = copy.deepcopy(subgoals)
    target_min_z = float(table_top_z) + float(min_tool_table_clearance_m)
    allowed_contact_lift_m = max(0.0, float(allowed_contact_table_penetration_m))
    for goal in adjusted.get("goals", []):
        if not isinstance(goal, dict):
            continue
        pose = goal.get("pose_world_tool")
        if not isinstance(pose, dict):
            continue
        pose_xyzw = [*pose["position"], *pose["orientation_xyzw"]]
        mesh_min_z, _mesh_max_z = tool_mesh_z_bounds_for_pose(pose_xyzw, vertices)
        lift_m = target_min_z - float(mesh_min_z)
        if lift_m <= 1e-6:
            continue
        if _is_startup_lift_stage(goal):
            report["skipped_contact_goals"].append(
                {
                    "index": goal.get("index"),
                    "stage": goal.get("stage"),
                    "mesh_min_z_before": float(mesh_min_z),
                    "would_lift_m": float(lift_m),
                    "reason": "startup_lift_preserves_smooth_departure_from_table",
                }
            )
            continue
        if _is_contact_path_stage(goal) and isinstance(
            goal.get("contact_point_goal"), dict
        ):
            if lift_m <= allowed_contact_lift_m + 1e-6:
                report["skipped_contact_goals"].append(
                    {
                        "index": goal.get("index"),
                        "stage": goal.get("stage"),
                        "mesh_min_z_before": float(mesh_min_z),
                        "would_lift_m": float(lift_m),
                        "allowed_contact_table_penetration_m": float(
                            allowed_contact_lift_m
                        ),
                        "reason": "within_allowed_contact_table_penetration",
                    }
                )
                continue
            lift_m = lift_m - allowed_contact_lift_m
        pose["position"] = [
            float(pose["position"][0]),
            float(pose["position"][1]),
            float(pose["position"][2]) + lift_m,
        ]
        _refresh_contact_point_goal(goal=goal, resolver=resolver)
        report["adjusted_goals"].append(
            {
                "index": goal.get("index"),
                "stage": goal.get("stage"),
                "mesh_min_z_before": float(mesh_min_z),
                "lift_m": float(lift_m),
                "allowed_contact_table_penetration_m": float(
                    allowed_contact_lift_m
                )
                if _is_contact_path_stage(goal)
                else 0.0,
            }
        )
    report["changed"] = bool(report["adjusted_goals"])
    report["contact_preservation_active"] = bool(report["skipped_contact_goals"])
    sanity = adjusted.setdefault("sanity_checks", {})
    if isinstance(sanity, dict):
        sanity["table_clearance_projection"] = report
    metadata = adjusted.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["table_clearance_projected"] = bool(report["adjusted_goals"])
    return adjusted, report


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
    initial_grasp_lift_m: float = 0.08,
    initial_grasp_lift_steps: int = 3,
    min_clearance_m: float = 0.003,
    swept_min_clearance_m: float = 0.006,
    allowed_contact_penetration_m: float = 0.006,
    allowed_table_contact_penetration_m: float | None = None,
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
    initial_grasp_lift_m: float = 0.08,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--program-ir", required=True)
    parser.add_argument("--planner-program", required=True)
    parser.add_argument("--start-pose", required=True)
    parser.add_argument(
        "--current-start-pose",
        help=(
            "Current observed object/tool pose to compile from. When provided, "
            "this takes precedence over tabletop projection."
        ),
    )
    parser.add_argument("--state-bundle")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--target-parts")
    parser.add_argument("--tool-parts")
    parser.add_argument("--tool-mesh")
    parser.add_argument("--tool-id")
    parser.add_argument(
        "--project-start-pose-to-table",
        dest="project_start_pose_to_table",
        action="store_true",
        default=None,
        help="Project the compiler start pose to the tabletop; default is on when --state-bundle is provided.",
    )
    parser.add_argument(
        "--no-project-start-pose-to-table",
        dest="project_start_pose_to_table",
        action="store_false",
        help="Debug only: use the provided start pose without tabletop projection.",
    )
    parser.add_argument("--start-min-tool-table-clearance-m", type=float, default=0.003)
    parser.add_argument("--swept-min-clearance-m", type=float, default=0.006)
    parser.add_argument("--allowed-contact-penetration-m", type=float, default=0.006)
    parser.add_argument("--allowed-table-contact-penetration-m", type=float)
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.08)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    args = parser.parse_args()
    run_mainline_fixed_goal_chain_files(
        scene_path=args.scene,
        tool_path=args.tool,
        program_ir_path=args.program_ir,
        planner_program_path=args.planner_program,
        start_pose_path=args.start_pose,
        current_start_pose_path=args.current_start_pose,
        state_bundle_path=args.state_bundle,
        output_dir=args.output_dir,
        grounding_candidates_path=args.candidates,
        target_part_graph_path=args.target_parts,
        tool_part_graph_path=args.tool_parts,
        tool_mesh_path=args.tool_mesh,
        tool_id=args.tool_id,
        project_start_pose_to_table=args.project_start_pose_to_table,
        start_min_tool_table_clearance_m=args.start_min_tool_table_clearance_m,
        swept_min_clearance_m=args.swept_min_clearance_m,
        allowed_contact_penetration_m=args.allowed_contact_penetration_m,
        allowed_table_contact_penetration_m=args.allowed_table_contact_penetration_m,
        initial_grasp_lift_m=args.initial_grasp_lift_m,
        initial_grasp_lift_steps=args.initial_grasp_lift_steps,
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
