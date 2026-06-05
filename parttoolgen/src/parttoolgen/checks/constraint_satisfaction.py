#!/usr/bin/env python3
"""Evaluate compiled ToolPoseSubgoals against ProgramIR constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parttoolgen.checks.affordance_binding_sanity import (
    check_affordance_binding_sanity,
    check_program_ir_affordance_bindings,
)
from parttoolgen.compiler.constraint_canonicalizer import (
    CanonicalStageConstraints,
    canonicalize_stage_constraints,
)
from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.compiler.residuals import normalize_quat, stage_cost
from parttoolgen.compiler.se3_optimizer import stage_constraint_errors
from parttoolgen.validation.validate_parttoolgen import (
    load_structured_file,
    validate_compiler_report,
    validate_functional_tool_parts,
    validate_grounding_candidates,
    validate_observed_scene_graph,
    validate_program_ir,
    validate_target_part_graph,
    validate_tool_part_graph,
    validate_tool_pose_subgoals,
)


def _constraints_for_stage(program_ir: dict[str, Any], stage_name: str) -> list[dict[str, Any]]:
    return [
        constraint
        for constraint in program_ir.get("constraints", [])
        if constraint.get("stage") == stage_name
    ]


def _goal_matches_stage(goal_stage: str, stage_name: str) -> bool:
    return goal_stage == stage_name or goal_stage.startswith(f"{stage_name}_")


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


def _stage_report(
    *,
    stage_name: str,
    canonical_stage: CanonicalStageConstraints,
    pose: dict[str, Any],
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    errors = stage_constraint_errors(pose=pose, stage=canonical_stage)
    warnings = _warnings_from_errors(canonical_stage, errors)
    warnings.extend(extra_warnings or [])
    return {
        "name": stage_name,
        "success": not warnings,
        "final_cost": stage_cost(
            pose=pose,
            point_targets=canonical_stage.point_targets,
            axis_targets=canonical_stage.axis_targets,
            clearance_targets=canonical_stage.clearance_targets,
        ),
        "constraint_errors": errors,
        "warnings": warnings,
        "source_constraints": canonical_stage.source_ids,
    }


def build_compiler_report(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    program_ir: dict[str, Any],
    tool_pose_subgoals: dict[str, Any],
    grounding_candidates: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    affordance_bindings: dict[str, Any] | list[dict[str, Any]] | None = None,
    binding_symbol_table: dict[str, Any] | None = None,
    visual_evidence: dict[str, Any] | None = None,
    include_program_affordance_binding_sanity: bool = True,
) -> dict[str, Any]:
    validate_observed_scene_graph(observed_scene_graph)
    validate_functional_tool_parts(functional_tool_parts)
    validate_program_ir(program_ir)
    validate_tool_pose_subgoals(tool_pose_subgoals)
    if grounding_candidates:
        validate_grounding_candidates(grounding_candidates)
    if target_part_graph:
        validate_target_part_graph(target_part_graph)
    if tool_part_graph:
        validate_tool_part_graph(tool_part_graph)

    resolver = ReferenceResolver(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        grounding_candidates=grounding_candidates,
        target_part_graph=target_part_graph,
        tool_part_graph=tool_part_graph,
    )
    affordance_binding_sanity: dict[str, Any] | None = None
    if affordance_bindings is not None:
        affordance_binding_sanity = check_affordance_binding_sanity(
            affordance_bindings,
            resolver=resolver,
            symbol_table=binding_symbol_table,
            visual_evidence=visual_evidence,
        )
    elif include_program_affordance_binding_sanity:
        affordance_binding_sanity = check_program_ir_affordance_bindings(
            program_ir,
            resolver=resolver,
            symbol_table=binding_symbol_table,
            visual_evidence=visual_evidence,
        )
    current_pose = {
        "position": [float(x) for x in tool_pose_subgoals["start_pose"]["position"]],
        "orientation_xyzw": normalize_quat(tool_pose_subgoals["start_pose"]["orientation_xyzw"]),
    }
    stages: list[dict[str, Any]] = []
    goals = tool_pose_subgoals["goals"]
    for stage in program_ir["stages"]:
        stage_name = stage["name"]
        constraints = _constraints_for_stage(program_ir, stage_name)
        canonical_stage = canonicalize_stage_constraints(
            stage_name=stage_name,
            constraints=constraints,
            resolver=resolver,
            current_pose=current_pose,
        )
        matching_goals = [
            goal for goal in goals if _goal_matches_stage(goal["stage"], stage_name)
        ]
        warnings: list[str] = []
        if matching_goals:
            pose = matching_goals[-1]["pose_world_tool"]
        else:
            pose = current_pose
            warnings.append(f"no compiled goal found for stage {stage_name}")
        stages.append(
            _stage_report(
                stage_name=stage_name,
                canonical_stage=canonical_stage,
                pose=pose,
                extra_warnings=warnings,
            )
        )
        current_pose = pose

    metadata = {
        "generated_by": "parttoolgen.constraint_satisfaction",
        "source": "ProgramIR+ToolPoseSubgoals",
    }
    if affordance_binding_sanity is not None:
        metadata["affordance_binding_sanity"] = affordance_binding_sanity

    report = {
        "schema_version": "v0.1",
        "report_id": f"{tool_pose_subgoals['subgoals_id']}_compiler_report",
        "program_id": program_ir["program_id"],
        "subgoals_id": tool_pose_subgoals["subgoals_id"],
        "passed": all(stage["success"] for stage in stages)
        and (
            affordance_binding_sanity is None
            or bool(affordance_binding_sanity.get("passed", False))
        ),
        "stages": stages,
        "metadata": metadata,
    }
    validate_compiler_report(report)
    return report


def write_compiler_report(
    *,
    scene_path: str | Path,
    tool_path: str | Path,
    program_ir_path: str | Path,
    subgoals_path: str | Path,
    output_path: str | Path,
    grounding_candidates_path: str | Path | None = None,
    target_part_graph_path: str | Path | None = None,
    tool_part_graph_path: str | Path | None = None,
    affordance_bindings_path: str | Path | None = None,
    binding_symbol_table_path: str | Path | None = None,
    visual_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_compiler_report(
        observed_scene_graph=load_structured_file(scene_path),
        functional_tool_parts=load_structured_file(tool_path),
        program_ir=load_structured_file(program_ir_path),
        tool_pose_subgoals=load_structured_file(subgoals_path),
        grounding_candidates=load_structured_file(grounding_candidates_path)
        if grounding_candidates_path
        else None,
        target_part_graph=load_structured_file(target_part_graph_path)
        if target_part_graph_path
        else None,
        tool_part_graph=load_structured_file(tool_part_graph_path)
        if tool_part_graph_path
        else None,
        affordance_bindings=load_structured_file(affordance_bindings_path)
        if affordance_bindings_path
        else None,
        binding_symbol_table=load_structured_file(binding_symbol_table_path)
        if binding_symbol_table_path
        else None,
        visual_evidence=load_structured_file(visual_evidence_path)
        if visual_evidence_path
        else None,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--program-ir", required=True)
    parser.add_argument("--subgoals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--target-parts")
    parser.add_argument("--tool-parts")
    parser.add_argument("--affordance-bindings")
    parser.add_argument("--binding-symbol-table")
    parser.add_argument("--visual-evidence")
    args = parser.parse_args()
    write_compiler_report(
        scene_path=args.scene,
        tool_path=args.tool,
        program_ir_path=args.program_ir,
        subgoals_path=args.subgoals,
        output_path=args.output,
        grounding_candidates_path=args.candidates,
        target_part_graph_path=args.target_parts,
        tool_part_graph_path=args.tool_parts,
        affordance_bindings_path=args.affordance_bindings,
        binding_symbol_table_path=args.binding_symbol_table,
        visual_evidence_path=args.visual_evidence,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
