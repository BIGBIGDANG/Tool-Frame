"""One orchestration path for tool references, scene grounding and execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[4]


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def asset_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def stage(module: str, arguments: list[str], *, python: str | None = None) -> None:
    script = ROOT / "ToGo/src" / (module.replace(".", "/") + ".py")
    subprocess.run([python or sys.executable, str(script), *arguments], cwd=ROOT, check=True)


def compact_tool_refs(refs: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    """Only the selected tool geometry crosses into the compiler."""
    if refs.get("tool_id", tool_id) != tool_id:
        raise ValueError(f"tool refs belong to {refs['tool_id']!r}, expected {tool_id!r}")
    import numpy as np
    keep = ("contact_point_ref", "contact_point_tool", "motion_axis_ref", "motion_axis_tool",
            "contact_point_face_marker", "motion_axis_face_marker", "contact_point_part_ref",
            "contact_point_source_ref", "contact_point_surface_ref", "motion_axis_surface_ref")
    result = {key: refs[key] for key in keep if key in refs}
    surface = dict(refs.get("surface_geometry_tool") or {})
    surface.pop("motion_axis_candidates_tool", None)
    result.update(tool_id=tool_id, surface_geometry_tool=surface)
    for key in ("contact_point_tool", "motion_axis_tool"):
        if key not in result:
            raise ValueError(f"tool refs missing {key}")
        vector = np.asarray(result[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{key} must contain three finite coordinates")
        if key == "motion_axis_tool" and np.linalg.norm(vector) <= 1e-12:
            raise ValueError("motion_axis_tool must be nonzero")
    return result


def select_tool_refs(task: Mapping[str, Any], output: Path, model: str | None) -> dict[str, Any]:
    tool_id = str(task["tool"]["tool_id"])
    prior = asset_path(task["tool"]["tool_prior_dir"])
    common = ["--tool-id", tool_id, "--task-instruction", str(task["instruction"]),
              "--output-dir", str(output)]
    if model:
        common.extend(["--model", model])
    for name in ("part", "contact"):
        evidence = output / f"{name}_evidence"
        render = ["--run-root", str(prior.parent.parent), "--tool-id", tool_id,
                  "--prior-subdir", prior.name, "--output-dir", str(evidence)]
        select = [*common, "--visual-evidence", str(evidence / "visual_evidence.json"), "--stage", name]
        if name == "contact":
            render.extend(["--part-role-selection", str(output / "part_selection.json")])
            select.extend(["--part-selection", str(output / "part_selection.json")])
        stage("ToGo.cli.tool_prior.render_tool_ref_evidence", render)
        stage("ToGo.cli.tool_prior.select_p3sam_tool_refs", select)
    return read_json(output / "tool_refs.json")


def compile_program(program: dict[str, Any], task: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    from ToGo.image_grounded import build_mainline_inputs_from_image_grounded_program
    from ToGo.pipeline.mainline import run_mainline_fixed_goal_chain
    execution = task["execution"]
    program = dict(program)
    program["tool_refs"] = {**program["tool_refs"],
                            "tool_mesh": str(asset_path(execution["simtoolreal_tool_mesh"])),
                            "tool_part_graph_path": str(asset_path(task["tool"]["tool_prior_dir"]) / "tool_part_graph.json")}
    pose = execution["simtoolreal_start_pose"]
    start = {"position": pose[:3], "orientation_xyzw": pose[3:]}
    # The compiler sees support geometry, not actor spawn poses as target references.
    state = {key: value for key, value in execution["state_bundle"].items()
             if key in ("table_pose_xyzw", "table_asset", "table_z", "min_tool_table_clearance_m")}
    method_task = {"execution": {"state_bundle": state}, "metadata": task.get("metadata", {})}
    inputs = build_mainline_inputs_from_image_grounded_program(program, task_data=method_task,
        start_pose=start, approach_offset_m=float(settings["approach_offset_m"]))
    prior = read_json(asset_path(task["tool"]["tool_prior_dir"]) / "tool_part_graph.json")
    result = run_mainline_fixed_goal_chain(
        observed_scene_graph=inputs["observed_scene_graph"],
        functional_tool_parts=inputs["functional_tool_parts"], program_ir=inputs["program_ir"],
        planner_program=inputs["planner_program"], start_pose=start, state_bundle=state,
        grounding_candidates=inputs["grounding_candidates"], target_part_graph=inputs["target_part_graph"],
        tool_part_graph=prior, tool_id=str(task["tool"]["tool_id"]),
        tool_mesh_path=str(asset_path(execution["simtoolreal_tool_mesh"])),
        project_start_pose_to_table=False, move_along_steps=int(settings["move_along_steps"]),
        max_translation_step_m=float(settings["max_translation_step_m"]),
        max_rotation_step_rad=float(settings["max_rotation_step_rad"]),
        initial_grasp_lift_m=float(settings["initial_grasp_lift_m"]),
        initial_grasp_lift_steps=int(settings["initial_grasp_lift_steps"]),
        min_clearance_m=float(settings["min_clearance_m"]),
        swept_min_clearance_m=float(settings["swept_min_clearance_m"]),
        allowed_contact_penetration_m=float(settings["allowed_contact_penetration_m"]),
        support_plane_point_world=task.get("metadata", {}).get("public_support_surface_point_world"),
        support_plane_normal_world=task.get("metadata", {}).get("public_support_surface_normal_world"))
    return {"fixed_goals": result["simtoolreal_fixed_goals"],
            "subgoals": result["path_solved_subgoals"]}


def plan(args: argparse.Namespace, task: dict[str, Any], settings: dict[str, Any]) -> Path:
    from ToGo.pipeline.lifting import build_program
    output = args.output.resolve()
    if args.scene_evidence:
        evidence = args.scene_evidence.resolve()
    else:
        stage("ToGo.cli.stages.render_simtoolreal_rgbd_scene_evidence",
              ["--task", str(args.task.resolve()), "--output-dir", str(output / "scene"),
               "--settings", str(args.settings.resolve()), "--device", args.device], python=args.runtime_python)
        evidence = output / "scene/scene_evidence.json"
    if args.scene_keypoints:
        keypoints = read_json(args.scene_keypoints.resolve())
    else:
        options = ["--scene-evidence", str(evidence), "--output-dir", str(output / "grounding"),
                   "--instruction", str(task["instruction"])]
        if args.model:
            options.extend(["--model", args.model])
        stage("ToGo.cli.stages.run_vlm_scene_keypoint_grounding", options)
        keypoints = read_json(output / "grounding/scene_keypoints.json")
    if args.tool_refs:
        refs = compact_tool_refs(read_json(args.tool_refs.resolve()), str(task["tool"]["tool_id"]))
    else:
        refs = select_tool_refs(task, output / "tool", args.model)
    program = build_program(task=task, evidence_path=evidence, keypoints=keypoints, tool_refs=refs,
                            approach_offset_m=float(settings["approach_offset_m"]))
    compiled = compile_program(program, task, settings)
    write_json(output / "constraint_program.json", program)
    write_json(output / "tool_pose_subgoals.json", compiled["subgoals"])
    goals = output / "fixed_goals.json"
    write_json(goals, compiled["fixed_goals"])
    return goals


def main() -> int:
    parser = argparse.ArgumentParser(description="ToGo: image-grounded tool manipulation")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check local runtime dependencies")
    prepare = commands.add_parser("prepare-tool", help="segment a mesh and prepare its geometric prior")
    prepare.add_argument("--tool-id", required=True)
    prepare.add_argument("--mesh", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--face-labels", type=Path)
    prepare.add_argument("--p3sam-root", type=Path)
    prepare.add_argument("--p3sam-python", default=sys.executable)
    for name in ("tool-ref", "plan", "execute", "run"):
        p = commands.add_parser(name)
        p.add_argument("--task", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--settings", type=Path, default=ROOT / "settings.json")
        if name in ("tool-ref", "plan", "run"):
            p.add_argument("--model", default=os.environ.get("VLM_MODEL"))
        if name in ("plan", "run"):
            p.add_argument("--scene-evidence", type=Path)
            p.add_argument("--scene-keypoints", type=Path)
            p.add_argument("--tool-refs", type=Path)
        if name == "execute":
            p.add_argument("--fixed-goals", type=Path, required=True)
        if name in ("plan", "execute", "run"):
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--runtime-python", default=os.environ.get("TOGO_RUNTIME_PYTHON", sys.executable))
    args = parser.parse_args()
    if args.command == "prepare-tool":
        from ToGo.pipeline.prepare_tool import prepare_tool
        prepare_tool(tool_id=args.tool_id, mesh=args.mesh, output=args.output,
                     face_labels=args.face_labels, p3sam_root=args.p3sam_root, p3sam_python=args.p3sam_python)
        return 0
    if args.command == "doctor":
        modules = ("numpy", "scipy", "PIL", "trimesh", "networkx", "yaml", "torch", "gym", "hydra", "omegaconf", "yourdfpy")
        print(json.dumps({"python": sys.version.split()[0],
                          "modules": {name: importlib.util.find_spec(name) is not None for name in modules},
                          "isaacgym_python": os.environ.get("TOGO_ISAACGYM"),
                          "policy_root": os.environ.get("TOGO_PRETRAINED_POLICY_ROOT")}, indent=2))
        return 0
    task = read_json(args.task.resolve())
    settings = read_json(args.settings.resolve())
    if args.command == "tool-ref":
        select_tool_refs(task, args.output.resolve(), args.model)
    else:
        if getattr(args, "scene_keypoints", None) and not args.scene_evidence:
            parser.error("--scene-keypoints requires its calibrated --scene-evidence")
        goals = plan(args, task, settings) if args.command in ("plan", "run") else args.fixed_goals.resolve()
        if args.command in ("execute", "run"):
            stage("ToGo.execution.simtoolreal_rollout", ["--task", str(args.task.resolve()),
                "--fixed-goals", str(goals), "--output", str(args.output.resolve()),
                "--settings", str(args.settings.resolve()), "--device", args.device], python=args.runtime_python)
        else:
            print(str(goals))
    return 0
