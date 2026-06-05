#!/usr/bin/env python3
"""Compile image-grounded ToolFrame refs through the staged mainline compiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.image_grounded import build_mainline_inputs_from_image_grounded_program  # noqa: E402
from parttoolgen.pipeline.mainline import MainlineArtifactChainError, run_mainline_fixed_goal_chain  # noqa: E402
from parttoolgen.validation.validate_parttoolgen import load_structured_file  # noqa: E402


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _pose_from_task(task_data: dict[str, Any]) -> dict[str, Any]:
    raw = task_data.get("execution", {}).get("simtoolreal_start_pose")
    if not isinstance(raw, list) or len(raw) != 7:
        raise ValueError("task.execution.simtoolreal_start_pose must be a 7D xyzw pose")
    return {
        "position": [float(item) for item in raw[:3]],
        "orientation_xyzw": [float(item) for item in raw[3:7]],
    }


def _tool_mesh_from_program_or_task(program: dict[str, Any], task_data: dict[str, Any] | None) -> str | None:
    tool_refs = program.get("tool_refs", {})
    if isinstance(tool_refs, dict) and isinstance(tool_refs.get("tool_mesh"), str):
        return str(tool_refs["tool_mesh"])
    if task_data:
        mesh = task_data.get("execution", {}).get("simtoolreal_tool_mesh")
        if isinstance(mesh, str):
            return mesh
    return None


def _tool_part_graph_from_program_or_task(
    program: dict[str, Any],
    task_data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    tool_refs = program.get("tool_refs", {})
    candidate_paths: list[Path] = []
    if isinstance(tool_refs, dict) and isinstance(tool_refs.get("tool_part_graph_path"), str):
        candidate_paths.append(Path(tool_refs["tool_part_graph_path"]))
    if task_data:
        prior_dir = task_data.get("tool", {}).get("tool_prior_dir")
        if isinstance(prior_dir, str):
            candidate_paths.append(ROOT / prior_dir / "tool_part_graph.json")
            candidate_paths.append(Path(prior_dir) / "tool_part_graph.json")
    for path in candidate_paths:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if resolved.exists():
            data = load_structured_file(resolved)
            if not isinstance(data, dict):
                raise ValueError(f"{resolved} must contain a JSON object")
            return data
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint-program", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-pose", default=None)
    parser.add_argument("--approach-offset-m", type=float, default=0.08)
    parser.add_argument("--project-start-pose-to-table", action="store_true")
    parser.add_argument(
        "--move-along-steps",
        type=int,
        default=1,
        help=(
            "Number of intervals for a move_along segment. The default 1 uses "
            "only the explicit start/end anchors; larger values are an explicit "
            "controller-smoothing choice, not task semantics."
        ),
    )
    parser.add_argument("--max-translation-step-m", type=float, default=0.04)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.35)
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.08)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    parser.add_argument("--min-clearance-m", type=float, default=0.003)
    parser.add_argument("--swept-min-clearance-m", type=float, default=0.006)
    parser.add_argument("--allowed-contact-penetration-m", type=float, default=0.006)
    parser.add_argument("--allowed-table-contact-penetration-m", type=float)
    args = parser.parse_args()

    program = load_structured_file(args.constraint_program)
    task_data = load_structured_file(args.task)
    start_pose = _load_json(args.start_pose) if args.start_pose else _pose_from_task(task_data)
    inputs = build_mainline_inputs_from_image_grounded_program(
        program,
        task_data=task_data,
        start_pose=start_pose,
        approach_offset_m=args.approach_offset_m,
    )
    inputs["tool_part_graph"] = _tool_part_graph_from_program_or_task(program, task_data)
    state_bundle = task_data.get("execution", {}).get("state_bundle")
    if not isinstance(state_bundle, dict):
        state_bundle = None
    tool_mesh_path = _tool_mesh_from_program_or_task(program, task_data)
    tool_id = (
        str(program.get("tool_refs", {}).get("tool_id"))
        if isinstance(program.get("tool_refs"), dict) and program["tool_refs"].get("tool_id")
        else str(task_data.get("tool", {}).get("tool_id") or "")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "observed_scene_graph": output_dir / "observed_scene_graph.json",
        "functional_tool_parts": output_dir / "functional_tool_parts.json",
        "grounding_candidates": output_dir / "grounding_candidates.json",
        "target_part_graph": output_dir / "target_part_graph.json",
        "tool_part_graph": output_dir / "tool_part_graph.json",
        "program_ir": output_dir / "program_ir.json",
        "planner_program": output_dir / "planner_program.json",
        "derived_scene_geometry": output_dir / "derived_scene_geometry.json",
        "selected_constraint_program": output_dir / "selected_constraint_program.json",
        "start_pose": output_dir / "start_pose.json",
    }
    for key, path in artifact_paths.items():
        if key == "start_pose":
            _write_json(path, start_pose)
        else:
            _write_json(path, inputs[key])

    try:
        result = run_mainline_fixed_goal_chain(
            observed_scene_graph=inputs["observed_scene_graph"],
            functional_tool_parts=inputs["functional_tool_parts"],
            program_ir=inputs["program_ir"],
            planner_program=inputs["planner_program"],
            start_pose=start_pose,
            current_start_pose=None,
            state_bundle=state_bundle,
            grounding_candidates=inputs["grounding_candidates"],
            target_part_graph=inputs["target_part_graph"],
            tool_part_graph=inputs["tool_part_graph"],
            tool_mesh_path=tool_mesh_path,
            tool_id=tool_id,
            project_start_pose_to_table=bool(args.project_start_pose_to_table),
            move_along_steps=args.move_along_steps,
            max_translation_step_m=args.max_translation_step_m,
            max_rotation_step_rad=args.max_rotation_step_rad,
            initial_grasp_lift_m=args.initial_grasp_lift_m,
            initial_grasp_lift_steps=args.initial_grasp_lift_steps,
            min_clearance_m=args.min_clearance_m,
            swept_min_clearance_m=args.swept_min_clearance_m,
            allowed_contact_penetration_m=args.allowed_contact_penetration_m,
            allowed_table_contact_penetration_m=args.allowed_table_contact_penetration_m,
        )
    except MainlineArtifactChainError as exc:
        partial = getattr(exc, "parttoolgen_partial_result", None)
        if isinstance(partial, dict):
            _write_json(output_dir / "mainline_partial_result.json", partial)
        _write_json(
            output_dir / "mainline_error.json",
            {
                "error": str(exc),
                "partial_result_path": str(output_dir / "mainline_partial_result.json")
                if isinstance(partial, dict)
                else None,
            },
        )
        raise

    output_map = {
        "tool_pose_subgoals": ("tool_pose_subgoals", output_dir / "tool_pose_subgoals.json"),
        "path_solved_subgoals": (
            "path_solved_tool_pose_subgoals",
            output_dir / "path_solved_tool_pose_subgoals.json",
        ),
        "tool_path_solver_report": ("tool_path_solver_report", output_dir / "tool_path_solver_report.json"),
        "simtoolreal_fixed_goals": ("simtoolreal_fixed_goals", output_dir / "simtoolreal_fixed_goals.json"),
        "start_pose_report": ("start_pose_report", output_dir / "start_pose_report.json"),
        "table_projection_report": ("table_projection_report", output_dir / "table_projection_report.json"),
    }
    named_output_paths: dict[str, Path] = {}
    for key, (artifact_name, path) in output_map.items():
        _write_json(path, result[key])
        named_output_paths[artifact_name] = path
    _write_json(
        output_dir / "compile_summary.json",
        {
            "constraint_program": str(Path(args.constraint_program)),
            "task": str(Path(args.task)),
            "method_terms": {
                "input_ir": "Tool-Scene Constraint IR",
                "compiler": "Stage-wise Geometric Constraint Compiler",
                "output": "Whole-tool SE(3) Subgoals",
                "backend_adapter": "SimToolReal fixed-goal adapter",
                "constraint_target_types": [
                    "PointTarget",
                    "AxisTarget",
                    "MoveAlongTarget",
                    "OrientationTarget",
                    "ClearanceTarget",
                ],
            },
            "axis_binding_source": "constraint_program.stages.axis_align",
            "mainline_params": {
                "approach_offset_m": float(args.approach_offset_m),
                "move_along_steps": int(args.move_along_steps),
                "max_translation_step_m": float(args.max_translation_step_m),
                "max_rotation_step_rad": float(args.max_rotation_step_rad),
                "initial_grasp_lift_m": float(args.initial_grasp_lift_m),
                "initial_grasp_lift_steps": int(args.initial_grasp_lift_steps),
            },
            "artifacts": {
                key: str(path)
                for key, path in {**artifact_paths, **named_output_paths}.items()
            },
            "num_goals": len(result["path_solved_subgoals"].get("goals", [])),
            "num_fixed_goals": len(result["simtoolreal_fixed_goals"].get("goals", [])),
            "path_solver_passed": result["tool_path_solver_report"].get("passed"),
            "compiler": result["path_solved_subgoals"].get("sanity_checks", {}).get("compiler"),
            "compiler_sanity_checks": result["path_solved_subgoals"].get("sanity_checks", {}),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "fixed_goals": str(named_output_paths["simtoolreal_fixed_goals"]),
                "path_solved_subgoals": str(named_output_paths["path_solved_tool_pose_subgoals"]),
                "summary": str(output_dir / "compile_summary.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
