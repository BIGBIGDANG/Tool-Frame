#!/usr/bin/env python3
"""Run the clean image-grounded claw-hammer down-strike smoke."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from parttoolgen.evaluation.task_success import (
    HAMMER_DOWN_SUCCESS_POLICY,
    check_task_success,
    predicate_passed,
)
from parttoolgen.evaluation.visual_evidence_package import (
    package_trial_visual_evidence,
    package_trial_visual_evidence_from_artifacts,
)
from parttoolgen.evaluation.tool_selection_evidence import (
    write_tool_selection_visual_evidence,
)
from parttoolgen.adapters.simtoolreal_policy_priors import (
    table_contact_start_pose_from_state_bundle,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
TOOL_REFS_DIR = ROOT / "configs/tool_refs/claw_hammer"
TABLE_ASSET = "urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf"
TOOL_MESH = "simtoolreal/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.obj"
TOOL_START_POSE_SOURCE = "simtoolreal/dextoolbench/trajectories/hammer/claw_hammer/swing_down.json"
TARGET_ASSET = "urdf/parttoolgen_benchmark/hammer_strike_nail/strike_nail_035.urdf"
TARGET_POSE = [0.0, -0.09, 0.545, 0.0, 0.0, 0.0, 1.0]
TABLE_Z = 0.38
TOOL_TABLE_CLEARANCE_M = 0.003
START_POSE_SWING_DOWN = [
    0.09076502773615436,
    0.08070982635558854,
    0.5443193171844398,
    0.07781270573992773,
    -0.9964358547253277,
    -0.025800699064297956,
    0.019876975902544947,
]
DOWN_AXIS_WORLD = [0.0, 0.0, -1.0]


def _script(name: str) -> Path:
    scripts = ROOT / "parttoolgen/src/parttoolgen/cli"
    for subdir in ("stages", "aux", "tool_prior"):
        candidate = scripts / subdir / name
        if candidate.exists():
            return candidate
    return scripts / name


def _python() -> str:
    return os.environ.get("PARTTOOLGEN_CORE_PYTHON") or sys.executable


def _env() -> dict[str, str]:
    env = dict(os.environ)
    src = ROOT / "parttoolgen" / "src"
    entries = [
        str(src),
        str(ROOT / "parttoolgen"),
        str(ROOT),
        str(ROOT / "simtoolreal"),
        str(ROOT / "simtoolreal/isaacgymenvs"),
        str(ROOT / "simtoolreal/rl_games"),
    ]
    for key in ("PARTTOOLGEN_ISAACGYM", "PARTTOOLGEN_SIMTOOLREAL_SITE"):
        if env.get(key):
            entries.append(env[key])
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _table_contact_tool_pose(pose_xyzw: list[float], *, tool_mesh: str) -> list[float]:
    projected = table_contact_start_pose_from_state_bundle(
        start_pose={
            "position": [float(item) for item in pose_xyzw[:3]],
            "orientation_xyzw": [float(item) for item in pose_xyzw[3:7]],
        },
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, TABLE_Z, 0.0, 0.0, 0.0, 1.0],
            "table_asset": TABLE_ASSET,
        },
        tool_mesh_path=tool_mesh,
        min_tool_table_clearance_m=TOOL_TABLE_CLEARANCE_M,
    )
    return [
        *[float(item) for item in projected["position"][:3]],
        *[float(item) for item in projected["orientation_xyzw"][:4]],
    ]


def _method_facing_task(task: dict[str, Any]) -> dict[str, Any]:
    execution = task.get("execution", {})
    state_bundle = execution.get("state_bundle", {}) if isinstance(execution, dict) else {}
    public_state_keys = {
        "table_pose_xyzw",
        "table_asset",
        "table_z",
        "object_name",
        "object_category",
        "task_name",
        "min_tool_table_clearance_m",
    }
    public_execution_keys = {
        "simtoolreal_category",
        "simtoolreal_table_asset",
        "simtoolreal_tool_mesh",
        "simtoolreal_start_pose_source",
        "start_pose_policy",
        "simtoolreal_start_pose",
    }
    return {
        "schema_version": task.get("schema_version"),
        "task_id": task.get("task_id"),
        "family": task.get("family"),
        "split": task.get("split"),
        "instruction": task.get("instruction"),
        "tool": copy.deepcopy(task.get("tool", {})),
        "execution": {
            **{
                key: copy.deepcopy(execution[key])
                for key in public_execution_keys
                if isinstance(execution, dict) and key in execution
            },
            "state_bundle": {
                key: copy.deepcopy(state_bundle[key])
                for key in public_state_keys
                if isinstance(state_bundle, dict) and key in state_bundle
            },
        },
        "public_scene": copy.deepcopy(task.get("public_scene", {})),
        "metadata": {
            "method_contract": "method-facing task with private target actor geometry omitted",
            "private_scene_geometry_omitted": True,
            "public_support_surface_normal_world": copy.deepcopy(
                task.get("metadata", {}).get("public_support_surface_normal_world", [0.0, 0.0, 1.0])
            ),
        },
    }


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _tool_refs_dir(args: argparse.Namespace) -> Path:
    return _resolve_repo_path(args.tool_refs_dir)


def _load_benchmark_config_defaults(argv: list[str]) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--benchmark-config", type=Path, default=None)
    known, _ = pre_parser.parse_known_args(argv)
    if known.benchmark_config is None:
        return None, None, {}
    config_path = _resolve_repo_path(known.benchmark_config)
    config = _load_json(config_path)
    runner_args = config.get("runner_args", {})
    if not isinstance(runner_args, dict):
        raise ValueError(f"benchmark config runner_args must be an object: {config_path}")
    defaults: dict[str, Any] = {}
    path_keys = {"output_root", "tool_refs_dir"}
    scalar_keys = {
        "seed_start",
        "trials",
        "device",
        "rollout_max_steps",
        "rollout_max_tracking_error_m",
        "capture_video_every",
        "initial_grasp_lift_m",
        "initial_grasp_lift_steps",
        "initial_grasp_hold_steps",
        "target_kind",
        "start_pose_mode",
        "response_mode",
        "model",
        "tool_id",
        "tool_mesh",
        "tool_start_pose_source",
        "fixed_start_pose_xyzw",
        "skip_rollout",
        "resume",
    }
    for key, value in runner_args.items():
        if key in path_keys:
            defaults[key] = _resolve_repo_path(value)
        elif key in scalar_keys:
            defaults[key] = value
    return config_path, config, defaults


def _run(
    cmd: list[str],
    *,
    log_path: Path,
    allow_failure: bool = False,
    retries: int = 0,
    retry_delay_s: float = 5.0,
) -> subprocess.CompletedProcess[str]:
    attempts: list[dict[str, Any]] = []
    proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(retries + 1):
        started = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "returncode": proc.returncode,
                "elapsed_s": round(time.time() - started, 3),
                "stdout_tail": proc.stdout[-16000:],
                "stderr_tail": proc.stderr[-16000:],
            }
        )
        if proc.returncode == 0 or allow_failure:
            break
        if attempt < retries:
            time.sleep(retry_delay_s)
    assert proc is not None
    _write_json(
        log_path,
        {
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_s": sum(item["elapsed_s"] for item in attempts),
            "attempts": attempts,
            "stdout_tail": proc.stdout[-16000:],
            "stderr_tail": proc.stderr[-16000:],
        },
    )
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _task_spec(*, task_id: str, seed: int, start_pose: list[float], args: argparse.Namespace) -> dict[str, Any]:
    start_pose = _table_contact_tool_pose([float(item) for item in start_pose], tool_mesh=args.tool_mesh)
    tool_refs_rel = _repo_rel(_tool_refs_dir(args))
    return {
        "schema_version": "parttoolgen_tool_use_task_v0",
        "task_id": task_id,
        "family": "image_grounded_strike",
        "split": "dev",
        "instruction": "Use this hammer to hit the nail.",
        "tool": {
            "tool_id": args.tool_id,
            "tool_prior_dir": tool_refs_rel,
        },
        "execution": {
            "simtoolreal_category": "hammer",
            "simtoolreal_table_asset": TABLE_ASSET,
            "simtoolreal_tool_mesh": args.tool_mesh,
            "simtoolreal_start_pose_source": args.tool_start_pose_source,
            "start_pose_policy": "dextoolbench_swing_down_start",
            "simtoolreal_start_pose": [float(item) for item in start_pose],
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0],
                "table_asset": TABLE_ASSET,
                "object_name": args.tool_id,
                "object_category": "hammer",
                "task_name": "swing_down",
                "target_collision_role": "dynamic_visible_nail_actor",
                "min_tool_table_clearance_m": 0.0003,
                "parttoolgen_benchmark_extra_actors": [
                    {
                        "name": "strike_nail",
                        "asset_file": TARGET_ASSET,
                        "pose_xyzw": TARGET_POSE,
                        "color_rgb": [0.62, 0.64, 0.66],
                        "disable_gravity": False,
                        "fix_base_link": False,
                        "friction": 0.9
                    }
                ],
            },
        },
        "public_scene": {
            "provider": "simtoolreal_rgbd_image_grounding",
            "target_selection_mode": "vlm_select_visible_single_target",
            "scene_evidence": [],
            "method_inputs": [
                "task language",
                "SimToolReal RGB-D render",
                f"public {args.tool_id} tool refs"
            ],
            "forbidden_method_inputs": [
                "private_eval.target_geometry",
                "tool_family",
                "task_family",
                "primitive fallback"
            ],
        },
        "metadata": {
            "seed": int(seed),
            "benchmark_role": "image_grounded_hammer_down_strike_smoke",
            "target_kind": "dynamic_visible_nail_actor",
            "public_support_surface_normal_world": [0.0, 0.0, 1.0],
            "notes": (
                "The target nail actor is visible in the rendered SimToolReal scene. "
                "The runner does not copy hidden evaluator target refs into planning; "
                "the executable start point is a VLM-marked public ref lifted through RGB-D; "
                "the use-stage direction comes from VLM point refs or a VLM-selected public default axis."
            ),
        },
    }


def _axis_report(lifted_refs: dict[str, Any]) -> dict[str, Any]:
    vector = lifted_refs.get("direction")
    if not isinstance(vector, list) or len(vector) != 3:
        return {"present": False, "down_cosine": None, "passed": False}
    axis = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in axis))
    if norm <= 1e-9:
        return {
            "present": True,
            "down_cosine": None,
            "passed": False,
            "raw": vector,
            "failure_reason": "zero_norm_scene_direction",
        }
    unit_axis = [item / norm for item in axis]
    down_cosine = sum(unit_axis[i] * DOWN_AXIS_WORLD[i] for i in range(3))
    return {
        "present": True,
        "vector_world": unit_axis,
        "down_cosine": down_cosine,
        "passed": down_cosine >= 0.95,
        "source": lifted_refs.get("direction_source"),
    }


def _stage_passed(execution_report: dict[str, Any], stage_prefix: str) -> bool:
    reports = execution_report.get("stage_reports")
    if not isinstance(reports, list):
        return False
    matched = [item for item in reports if str(item.get("stage", "")).startswith(stage_prefix)]
    return bool(matched) and all(bool(item.get("success")) for item in matched)


def _grasp_lift_passed(
    execution_report: dict[str, Any],
    compile_summary: dict[str, Any],
) -> bool | None:
    compiler_sanity = compile_summary.get("compiler_sanity_checks")
    if not isinstance(compiler_sanity, dict):
        compiler_sanity = compile_summary.get("compiler")
    if (
        isinstance(compiler_sanity, dict)
        and compiler_sanity.get("initial_grasp_lift_inserted") is False
    ):
        return None
    return _stage_passed(execution_report, "grasp_lift")


def _load_optional_json(path_value: Any) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    data = _load_json(path)
    return data if isinstance(data, dict) else None


def _refresh_success_fields(result: dict[str, Any], *, skip_rollout: bool) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    rollout_summary = _load_optional_json(artifacts.get("rollout_summary")) if isinstance(artifacts, dict) else None
    execution_report = _load_optional_json(artifacts.get("execution_report")) if isinstance(artifacts, dict) else None
    compile_summary = _load_optional_json(artifacts.get("compile_summary")) if isinstance(artifacts, dict) else None
    if compile_summary is None:
        compile_summary = {"path_solver_passed": bool(result.get("compile_path_solver_passed"))}
    if rollout_summary is None and result.get("semantic_execution_passed") is not None:
        rollout_summary = {"semantic_execution_passed": bool(result.get("semantic_execution_passed"))}
    success_check = check_task_success(
        "hammer_down_strike",
        axis=result.get("scene_direction_axis", {}),
        compile_summary=compile_summary,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        skip_rollout=skip_rollout,
    )
    refreshed = dict(result)
    refreshed["task_success_check"] = success_check
    refreshed["hammer_contact_passed"] = predicate_passed(success_check, "semantic_contact")
    refreshed["success_policy"] = success_check["policy"]
    refreshed["passed"] = bool(success_check.get("passed"))
    return refreshed


def _run_trial(*, seed: int, index: int, args: argparse.Namespace) -> dict[str, Any]:
    task_id = f"{args.tool_id}_down_strike_nail_seed_{seed:03d}"
    trial_dir = args.output_root / task_id
    if args.resume and (trial_dir / "trial_result.json").exists():
        result = _refresh_success_fields(
            _load_json(trial_dir / "trial_result.json"),
            skip_rollout=bool(args.skip_rollout),
        )
        visual_evidence = package_trial_visual_evidence_from_artifacts(
            trial_dir=trial_dir,
            artifacts=result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {},
            tool_refs_dir=_tool_refs_dir(args),
        )
        result.setdefault("artifacts", {})["visual_evidence"] = visual_evidence["evidence_dir"]
        result["artifacts"]["visual_evidence_manifest"] = visual_evidence["manifest"]
        _write_json(trial_dir / "trial_result.json", result)
        return result
    if trial_dir.exists() and not args.resume:
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    task = _task_spec(task_id=task_id, seed=seed, start_pose=args.fixed_start_pose_xyzw, args=args)
    task_path = trial_dir / "task.json"
    method_task_path = trial_dir / "method_task.json"
    _write_json(task_path, task)
    _write_json(method_task_path, _method_facing_task(task))
    py = _python()

    render_dir = trial_dir / "sim_rgbd"
    _run(
        [
            py,
            str(_script("render_simtoolreal_rgbd_scene_evidence.py")),
            "--task",
            str(task_path),
            "--output-dir",
            str(render_dir),
            "--device",
            args.device,
            "--headless",
            "--topdown-width",
            "1024",
            "--topdown-height",
            "768",
            "--topdown-fov-deg",
            "52",
        ],
        log_path=trial_dir / "logs/render.json",
    )

    grounding_dir = trial_dir / "vlm_scene_keypoints"
    vlm_cmd = [
        py,
        str(_script("run_vlm_scene_keypoint_grounding.py")),
        "--scene-evidence",
        str(render_dir / "scene_evidence.json"),
        "--output-dir",
        str(grounding_dir),
        "--response-mode",
        args.response_mode,
    ]
    if args.model:
        vlm_cmd.extend(["--model", args.model])
    _run(
        vlm_cmd,
        log_path=trial_dir / "logs/vlm_scene_keypoints.json",
    )
    _run(
        [
            py,
            str(_script("visualize_scene_keypoint_grounding.py")),
            "--scene-evidence",
            str(render_dir / "scene_evidence.json"),
            "--scene-keypoints",
            str(grounding_dir / "scene_keypoints.json"),
            "--output",
            str(grounding_dir / "topdown_grounding_overlay.png"),
        ],
        log_path=trial_dir / "logs/visualize_grounding.json",
    )

    lifted_dir = trial_dir / "lifted_program"
    _run(
        [
            py,
            str(_script("lift_scene_keypoints_to_constraint_program.py")),
            "--scene-evidence",
            str(render_dir / "scene_evidence.json"),
            "--scene-keypoints",
            str(grounding_dir / "scene_keypoints.json"),
            "--task",
            str(method_task_path),
            "--output-dir",
            str(lifted_dir),
            "--depth-sample-policy",
            "nearest",
            "--action-direction-z-policy",
            "anchor_height",
            "--direction-source",
            "path_geometry",
            "--end-surface-policy",
            "anchor_contact_plane",
            "--tool-motion-axis-ref",
            "tool.motion_axis",
            "--tool-motion-axis-target-ref",
            "scene.direction",
            "--tool-motion-axis-relation",
            "parallel",
            "--target-approach-axis-source",
            "opposite_scene_direction",
            "--target-surface-normal-world",
            "0",
            "0",
            "1",
        ],
        log_path=trial_dir / "logs/lift.json",
    )

    enriched_program = lifted_dir / "constraint_program_with_tool_refs.json"
    _run(
        [
            py,
            str(_script("merge_tool_refs_into_constraint_program.py")),
            "--program",
            str(lifted_dir / "constraint_program.json"),
            "--axis-binding",
            str(_tool_refs_dir(args) / "axis_binding.json"),
            "--axis-mesh-report",
            str(_tool_refs_dir(args) / "axis_mesh_overlay_report.json"),
            "--contact-point-selection",
            str(_tool_refs_dir(args) / "contact_point_selection.json"),
            "--output",
            str(enriched_program),
        ],
        log_path=trial_dir / "logs/merge_tool_refs.json",
    )

    compile_dir = trial_dir / "compile"
    _run(
        [
            py,
            str(_script("compile_image_grounded_with_mainline.py")),
            "--constraint-program",
            str(enriched_program),
            "--task",
            str(method_task_path),
            "--output-dir",
            str(compile_dir),
            "--max-translation-step-m",
            "0.035",
            "--max-rotation-step-rad",
            "0.30",
            "--initial-grasp-lift-m",
            str(args.initial_grasp_lift_m),
            "--initial-grasp-lift-steps",
            str(args.initial_grasp_lift_steps),
            "--min-clearance-m",
            "0.003",
            "--swept-min-clearance-m",
            "0.006",
            "--allowed-contact-penetration-m",
            "0.006",
        ],
        log_path=trial_dir / "logs/compile.json",
    )
    _write_json(trial_dir / "state_bundle.json", task["execution"]["state_bundle"])

    rollout_summary: dict[str, Any] = {}
    execution_report: dict[str, Any] = {}
    if not args.skip_rollout:
        rollout_dir = trial_dir / "rollout"
        rollout_cmd = [
            py,
            str(ROOT / "parttoolgen/src/parttoolgen/execution/simtoolreal_rollout.py"),
            "--fixed-goals",
            str(compile_dir / "simtoolreal_fixed_goals.json"),
            "--scene",
            str(compile_dir / "observed_scene_graph.json"),
            "--subgoals",
            str(compile_dir / "path_solved_tool_pose_subgoals.json"),
            "--state-bundle",
            str(trial_dir / "state_bundle.json"),
            "--tool-mesh",
            args.tool_mesh,
            "--output-dir",
            str(rollout_dir),
            "--device",
            args.device,
            "--headless",
            "--max-steps",
            str(args.rollout_max_steps),
            "--success-steps",
            "3",
            "--initial-grasp-hold-steps",
            str(args.initial_grasp_hold_steps),
            "--post-success-steps",
            str(args.post_success_steps),
            "--max-tracking-error-m",
            str(args.rollout_max_tracking_error_m),
            "--force-no-reset",
        ]
        if args.capture_video_every > 0 and (index - 1) % args.capture_video_every == 0:
            rollout_cmd.extend(
                [
                    "--capture-video",
                    "--capture-topdown-video",
                    "--hide-goal-object-visual",
                    "--video-fps",
                    "24",
                    "--video-stride",
                    "12",
                ]
            )
        _run(rollout_cmd, log_path=trial_dir / "logs/rollout.json", allow_failure=True)
        rollout_summary = _load_optional_json(rollout_dir / "rollout_summary.json") or {}
        execution_report = _load_optional_json(rollout_dir / "execution_report.json") or {}

    scene_keypoints = _load_json(grounding_dir / "scene_keypoints.json")
    lifted_refs = _load_json(lifted_dir / "lifted_refs.json")
    compile_summary = _load_json(compile_dir / "compile_summary.json")
    axis = _axis_report(lifted_refs)
    tracking_passed = bool(rollout_summary.get("tracking_execution_passed")) if rollout_summary else None
    policy_goal_passed = bool(rollout_summary.get("policy_goal_passed")) if rollout_summary else None
    semantic_execution_passed = (
        bool(rollout_summary.get("semantic_execution_passed")) if rollout_summary else None
    )
    success_check = check_task_success(
        "hammer_down_strike",
        axis=axis,
        compile_summary=compile_summary,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        skip_rollout=bool(args.skip_rollout),
    )
    visual_evidence = package_trial_visual_evidence(
        trial_dir=trial_dir,
        render_dir=render_dir,
        grounding_dir=grounding_dir,
        tool_refs_dir=_tool_refs_dir(args),
        compile_dir=compile_dir,
        rollout_dir=trial_dir / "rollout",
    )
    semantic_contact_passed = predicate_passed(success_check, "semantic_contact")
    grasp_lift_passed = (
        _grasp_lift_passed(execution_report, compile_summary)
        if execution_report
        else None
    )
    passed = bool(success_check.get("passed"))
    result = {
        "index": index,
        "seed": seed,
        "task_id": task_id,
        "passed": passed,
        "skip_rollout": bool(args.skip_rollout),
        "scene_direction_axis": axis,
        "target_object": scene_keypoints.get("target_object"),
        "lifted_start_point_world": lifted_refs.get("start_point", {}).get("position_world"),
        "motion_axis_world": lifted_refs.get("motion_axis"),
        "direction_source": lifted_refs.get("direction_source"),
        "selected_contact_point_face": _load_json(enriched_program).get("tool_refs", {}).get("contact_point_face_marker"),
        "selected_contact_point_ref": _load_json(enriched_program).get("tool_refs", {}).get("contact_point_ref"),
        "compile_path_solver_passed": bool(compile_summary.get("path_solver_passed")),
        "policy_goal_passed": policy_goal_passed,
        "tracking_execution_passed": tracking_passed,
        "semantic_execution_passed": semantic_execution_passed,
        "hammer_semantic_contact_passed": semantic_contact_passed,
        "hammer_contact_passed": semantic_contact_passed,
        "task_success_check": success_check,
        "success_policy": success_check["policy"],
        "grasp_lift_passed": grasp_lift_passed,
        "final_successes": int(rollout_summary.get("final_successes", 0)) if rollout_summary else None,
        "num_goals": int(rollout_summary.get("num_goals", 0)) if rollout_summary else None,
        "num_steps": int(rollout_summary.get("num_steps", 0)) if rollout_summary else None,
        "artifacts": {
            "task": str(task_path),
            "method_task": str(method_task_path),
            "scene_evidence": str(render_dir / "scene_evidence.json"),
            "grounding_overlay": str(grounding_dir / "topdown_grounding_overlay.png"),
            "scene_keypoints": str(grounding_dir / "scene_keypoints.json"),
            "lifted_refs": str(lifted_dir / "lifted_refs.json"),
            "constraint_program": str(enriched_program),
            "compile_summary": str(compile_dir / "compile_summary.json"),
            "fixed_goals": str(compile_dir / "simtoolreal_fixed_goals.json"),
            "rollout_summary": str(trial_dir / "rollout/rollout_summary.json")
            if not args.skip_rollout
            else None,
            "execution_report": str(trial_dir / "rollout/execution_report.json")
            if not args.skip_rollout
            else None,
            "rollout_video": str(trial_dir / "rollout/rollout_render.mp4")
            if (trial_dir / "rollout/rollout_render.mp4").exists()
            else None,
            "rollout_topdown_video": str(trial_dir / "rollout/rollout_topdown_render.mp4")
            if (trial_dir / "rollout/rollout_topdown_render.mp4").exists()
            else None,
            "visual_evidence": visual_evidence["evidence_dir"],
            "visual_evidence_manifest": visual_evidence["manifest"],
        },
    }
    _write_json(trial_dir / "trial_result.json", result)
    return result


def _summary(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    passed = [row for row in rows if row.get("passed")]
    contact_rows = [
        row
        for row in rows
        if row.get("hammer_semantic_contact_passed", row.get("hammer_contact_passed")) is not None
    ]
    contact_passed = [
        row
        for row in contact_rows
        if row.get("hammer_semantic_contact_passed", row.get("hammer_contact_passed"))
    ]
    lift_rows = [row for row in rows if row.get("grasp_lift_passed") is not None]
    summary = {
        "schema_version": "toolframe_image_grounded_hammer_down_strike_eval_v1",
        "success_policy": HAMMER_DOWN_SUCCESS_POLICY["name"],
        "success_requires_target_displacement": False,
        "success_requires_tracking_execution_passed": False,
        "output_root": str(output_root),
        "trial_count": len(rows),
        "passed_count": len(passed),
        "pass_rate": (len(passed) / len(rows)) if rows else 0.0,
        "semantic_contact_pass_rate": (len(contact_passed) / len(contact_rows)) if contact_rows else None,
        "contact_pass_rate": (len(contact_passed) / len(contact_rows)) if contact_rows else None,
        "grasp_lift_pass_rate": (
            sum(1 for row in lift_rows if row.get("grasp_lift_passed")) / len(lift_rows)
        )
        if lift_rows
        else None,
        "trials": rows,
    }
    _write_json(output_root / "summary.json", summary)
    lines = [
        "# Image-Grounded Hammer Down-Strike Eval",
        "",
        f"- trials: {summary['trial_count']}",
        f"- success policy: {HAMMER_DOWN_SUCCESS_POLICY['name']}",
        f"- pass rate: {summary['pass_rate']:.3f}",
        f"- semantic contact pass rate: {summary['semantic_contact_pass_rate']:.3f}"
        if summary["semantic_contact_pass_rate"] is not None
        else "- semantic contact pass rate: n/a",
        f"- grasp-lift pass rate: {summary['grasp_lift_pass_rate']:.3f}"
        if summary["grasp_lift_pass_rate"] is not None
        else "- grasp-lift pass rate: n/a",
        "",
        "| seed | pass | semantic contact | tracking | axis cosine | face | goals | steps | overlay | video |",
        "|---:|---|---|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        artifacts = row.get("artifacts", {})
        axis = row.get("scene_direction_axis", {})
        lines.append(
            f"| {row.get('seed')} | {row.get('passed')} | "
            f"{row.get('hammer_semantic_contact_passed', row.get('hammer_contact_passed'))} | "
            f"{row.get('tracking_execution_passed')} | "
            f"{float(axis.get('down_cosine') or 0.0):.3f} | "
            f"{row.get('selected_contact_point_face')} | "
            f"{row.get('final_successes')}/{row.get('num_goals')} | {row.get('num_steps')} | "
            f"{artifacts.get('grounding_overlay')} | {artifacts.get('rollout_video')} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    benchmark_config_path, benchmark_config, benchmark_defaults = (
        _load_benchmark_config_defaults(sys.argv[1:])
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, default=benchmark_config_path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/image_grounded/hammer_strike/down_original_rgb_smoke",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--rollout-max-tracking-error-m", type=float, default=0.018)
    parser.add_argument("--capture-video-every", type=int, default=1)
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.08)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    parser.add_argument("--initial-grasp-hold-steps", type=int, default=0)
    parser.add_argument("--post-success-steps", type=int, default=60)
    parser.add_argument("--target-kind", choices=["dynamic_visible_nail"], default="dynamic_visible_nail")
    parser.add_argument(
        "--start-pose-mode",
        choices=["dextoolbench_swing_down_start"],
        default="dextoolbench_swing_down_start",
    )
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tool-id", default="claw_hammer")
    parser.add_argument("--tool-refs-dir", type=Path, default=TOOL_REFS_DIR)
    parser.add_argument("--tool-mesh", default=TOOL_MESH)
    parser.add_argument("--tool-start-pose-source", default=TOOL_START_POSE_SOURCE)
    parser.add_argument("--fixed-start-pose-xyzw", nargs=7, type=float, default=START_POSE_SWING_DOWN)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**benchmark_defaults)
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if benchmark_config is not None:
        _write_json(
            args.output_root / "benchmark_config.resolved.json",
            {
                "source": str(benchmark_config_path),
                "runner_args_public": benchmark_config.get("runner_args", {}),
                "private_manifest_omitted": True,
                "private_manifest_note": (
                    "Full benchmark config remains under artifacts/tool_use_benchmark; "
                    "method outputs record only runner args used by the public pipeline."
                ),
            },
        )
    tool_selection_evidence = write_tool_selection_visual_evidence(
        output_root=args.output_root,
        tool_refs_dir=_tool_refs_dir(args),
        task_family="hammer_strike",
        split="down",
    )
    _write_json(
        args.output_root / "run_config.json",
        {
            "benchmark_config": str(args.benchmark_config.resolve())
            if args.benchmark_config is not None
            else None,
            "tool_id": args.tool_id,
            "tool_refs_dir": str(_tool_refs_dir(args)),
            "tool_selection_visual_evidence": tool_selection_evidence,
            "table_asset": TABLE_ASSET,
            "tool_mesh": args.tool_mesh,
            "tool_start_pose_source": args.tool_start_pose_source,
            "seed_start": args.seed_start,
            "trials": args.trials,
            "device": args.device,
            "render_lighting_preset": "soft01",
            "rollout_max_steps": args.rollout_max_steps,
            "rollout_max_tracking_error_m": args.rollout_max_tracking_error_m,
            "capture_video_every": args.capture_video_every,
            "initial_grasp_lift_m": args.initial_grasp_lift_m,
            "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
            "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
            "target_kind": args.target_kind,
            "start_pose_mode": args.start_pose_mode,
            "response_mode": args.response_mode,
            "model": args.model,
            "skip_rollout": args.skip_rollout,
            "mainline_params": {
                "depth_sample_policy": "nearest",
                "direction_source": "vlm_point_refs",
                "end_surface_policy": "anchor_contact_plane",
                "end_point_policy": "start_equals_end_contact_only",
                "tool_motion_axis_ref": "tool.motion_axis",
                "tool_motion_axis_target_ref": "scene.direction",
                "tool_motion_axis_relation": "parallel",
                "target_surface_normal_world": [0.0, 0.0, 1.0],
                "max_translation_step_m": 0.035,
                "max_rotation_step_rad": 0.30,
                "initial_grasp_lift_m": args.initial_grasp_lift_m,
                "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
                "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
                "rollout_max_tracking_error_m": args.rollout_max_tracking_error_m,
                "rollout_max_rotation_error_rad": 0.28,
                "force_no_reset": True,
            },
        },
    )
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.trials), start=1):
        print(f"[{index}/{args.trials}] running hammer seed {seed:03d}", flush=True)
        try:
            row = _run_trial(seed=seed, index=index, args=args)
        except Exception as exc:
            task_id = f"{args.tool_id}_down_strike_nail_seed_{seed:03d}"
            row = {
                "index": index,
                "seed": seed,
                "task_id": task_id,
                "passed": False,
                "status": "exception",
                "error": str(exc),
            }
            trial_dir = args.output_root / task_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            _write_json(trial_dir / "trial_result.json", row)
        rows.append(row)
        summary = _summary(rows, args.output_root)
        print(
            f"[{index}/{args.trials}] seed {seed:03d}: passed={row.get('passed')} "
            f"pass_rate={summary['pass_rate']:.3f}",
            flush=True,
        )
    final_summary = _summary(rows, args.output_root)
    print(
        json.dumps(
            {
                "summary": str(args.output_root / "summary.json"),
                "pass_rate": final_summary["pass_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
