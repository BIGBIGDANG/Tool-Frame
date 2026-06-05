#!/usr/bin/env python3
"""Run the image-grounded side-board marker-trace benchmark.

The marker task uses the same public scene-ref interface as the tabletop tasks:
VLM marks a scene start point, optional waypoints, and an end point/direction;
RGB-D lifting turns those 2D refs into 3D constraints; public tool refs provide
tool.contact_point and tool.motion_axis; the compiler emits whole-tool SE(3)
subgoals; rollout produces marker-tip contact samples; a VLM judge scores the
rendered ink trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from parttoolgen.evaluation.tool_selection_evidence import (
    write_tool_selection_visual_evidence,
)


ROOT = Path(__file__).resolve().parents[5]
SRC_TASK = ROOT / "artifacts/image_grounded/marker_trace/path_whiteboard_preview_staples_seed_000/task.json"
DEFAULT_TOOL_REFS = ROOT / "configs/tool_refs/staples_marker"

PINK = (224, 112, 176)
BLUE = (107, 197, 222)
GREEN = (111, 207, 126)
BLACK = (35, 35, 35)


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
    entries = [
        str(ROOT / "parttoolgen/src"),
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


def _run_logged(log_dir: Path, name: str, cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== START {name} =====\n")
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env or _env(),
        )
        handle.write(f"===== END {name} code={result.returncode} =====\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _point(row: Any) -> tuple[float, float] | None:
    if isinstance(row, dict) and "u" in row and "v" in row:
        return float(row["u"]), float(row["v"])
    return None


def _draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[float, float] | None, color: tuple[int, int, int]) -> None:
    if xy is None:
        return
    x, y = xy
    radius = 16
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=BLACK, width=4)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> None:
    if a is None or b is None:
        return
    import math

    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux = dx / length
    uy = dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + 14.0, length)
        draw.line((ax + ux * pos, ay + uy * pos, ax + ux * end, ay + uy * end), fill=BLACK, width=4)
        pos += 24.0


def _draw_scene_overlay(scene_keypoints: Path, image_path: Path, output_path: Path) -> None:
    data = _load_json(scene_keypoints)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    action = data.get("action_direction", {}) if isinstance(data.get("action_direction"), dict) else {}
    start = _point(data.get("start_point")) or _point(action.get("start"))
    path_points = [
        point for point in (_point(row) for row in data.get("path_points", [])) if point is not None
    ]
    if path_points:
        path = path_points
        start = path[0]
        end = path[-1]
        waypoints = path[1:-1]
    else:
        waypoints = [_point(row) for row in data.get("waypoints", []) if _point(row) is not None]
        end = _point(data.get("end_point"))
        if end is None and waypoints:
            end = waypoints[-1]
            waypoints = waypoints[:-1]
        if end is None:
            end = _point(action.get("end"))
        path = [point for point in [start, *waypoints, end] if point is not None]
    for a, b in zip(path, path[1:]):
        _draw_dashed_line(draw, a, b)
    _draw_marker(draw, start, PINK)
    for waypoint in waypoints:
        _draw_marker(draw, waypoint, BLUE)
    _draw_marker(draw, end, GREEN)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_state_bundle(task: Path, output: Path) -> None:
    data = _load_json(task)
    state = data.get("execution", {}).get("state_bundle")
    if not isinstance(state, dict):
        raise ValueError(f"missing execution.state_bundle in {task}")
    _write_json(output, state)


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [float(item) for item in value[:3]]


def _unit(value: list[float], name: str) -> list[float]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-12:
        raise ValueError(f"{name} is near zero")
    return [item / norm for item in value]


def _signed_plane_gap(point: list[float], surface_point: list[float], normal: list[float]) -> float:
    return sum((point[index] - surface_point[index]) * normal[index] for index in range(3))


def _raw_ref_world(row: dict[str, Any]) -> list[float] | None:
    raw = row.get("raw_position_world")
    if isinstance(raw, list) and len(raw) >= 3:
        return [float(item) for item in raw[:3]]
    world = row.get("position_world")
    if isinstance(world, list) and len(world) >= 3:
        return [float(item) for item in world[:3]]
    return None


def _validate_lifted_refs_on_support_surface(
    *,
    task_path: Path,
    lifted_refs_path: Path,
    output_path: Path,
    max_surface_ref_gap_m: float,
) -> dict[str, Any]:
    task = _load_json(task_path)
    lifted = _load_json(lifted_refs_path)
    metadata = task.get("metadata") if isinstance(task, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    surface_point_raw = metadata.get("public_support_surface_point_world")
    surface_normal_raw = metadata.get("public_support_surface_normal_world")
    if surface_point_raw is None or surface_normal_raw is None:
        report = {
            "schema_version": "toolframe_surface_ref_validation_v1",
            "checked": False,
            "passed": True,
            "reason": "task metadata has no public support surface plane",
        }
        _write_json(output_path, report)
        return report
    surface_point = _vec3(surface_point_raw, "public_support_surface_point_world")
    surface_normal = _unit(_vec3(surface_normal_raw, "public_support_surface_normal_world"), "public_support_surface_normal_world")
    rows: list[tuple[str, dict[str, Any]]] = []
    for key in ("start_point", "end_point"):
        row = lifted.get(key)
        if isinstance(row, dict):
            rows.append((key, row))
    for index, row in enumerate(lifted.get("path_points") or []):
        if isinstance(row, dict):
            rows.append((f"path_points[{index}]", row))
    checks = []
    failed = []
    for name, row in rows:
        point = _raw_ref_world(row)
        if point is None:
            checks.append({"name": name, "checked": False, "passed": False, "reason": "missing raw/world position"})
            failed.append(name)
            continue
        signed_gap = _signed_plane_gap(point, surface_point, surface_normal)
        passed = abs(signed_gap) <= float(max_surface_ref_gap_m)
        item = {
            "name": name,
            "checked": True,
            "passed": passed,
            "signed_surface_gap_m": signed_gap,
            "abs_surface_gap_m": abs(signed_gap),
            "max_surface_ref_gap_m": float(max_surface_ref_gap_m),
            "raw_or_world_position": point,
        }
        checks.append(item)
        if not passed:
            failed.append(name)
    report = {
        "schema_version": "toolframe_surface_ref_validation_v1",
        "checked": True,
        "passed": not failed,
        "failed_refs": failed,
        "surface_point_world": surface_point,
        "surface_normal_world": surface_normal,
        "max_surface_ref_gap_m": float(max_surface_ref_gap_m),
        "checks": checks,
    }
    _write_json(output_path, report)
    if failed:
        raise ValueError(
            "scene refs are not on the selected support surface before projection: "
            + ", ".join(failed)
        )
    return report


def _prepare_task_copy(src: Path, dst: Path, *, instruction: str, seed_index: int, split: str) -> None:
    data = _load_json(src)
    data["task_id"] = f"marker_trace_{split}_seed_{seed_index:03d}"
    data["family"] = "marker_trace"
    data["split"] = split
    data["instruction"] = instruction
    data["seed"] = seed_index
    metadata = data.setdefault("metadata", {})
    if isinstance(metadata, dict):
        for key in (
            "public_strike_axis_world",
            "public_insertion_axis_world",
            "side_strike_target_center_world",
            "socket_opening_center_world",
        ):
            metadata.pop(key, None)
        metadata["evaluation_seed_index"] = seed_index
        metadata["environment_id"] = "side_left_vertical_whiteboard_soft02"
    _write_json(dst, data)


def _letter_from_config(config: dict[str, Any], fallback: str) -> str:
    runner_args = config.get("runner_args") if isinstance(config.get("runner_args"), dict) else {}
    letter = runner_args.get("requested_letter")
    target = config.get("scene_initialization", {}).get("target", {})
    if letter is None and isinstance(target, dict):
        letter = target.get("requested_letter")
    return str(letter or fallback).upper()


def _run_one(args: argparse.Namespace, *, config: dict[str, Any], trial_index: int) -> dict[str, Any]:
    seed = int(args.seed_start) + trial_index
    letter = str(args.requested_letter or _letter_from_config(config, args.split[-1:])).upper()
    split = str(args.split or f"write_{letter.lower()}")
    name = f"{split}_seed_{seed:03d}"
    task_id = f"marker_trace_{split}_seed_{seed:03d}"
    out = Path(args.output_root) / f"seed_{seed:03d}"
    task = out / "task.json"
    state = out / "state_bundle.json"
    logs = out / "logs"
    tool_refs = Path(args.tool_refs_dir).expanduser()
    if not tool_refs.is_absolute():
        tool_refs = ROOT / tool_refs

    instruction = str(args.task_language or f"Use this marker to write the letter {letter} on the whiteboard.")
    _prepare_task_copy(Path(args.source_task), task, instruction=instruction, seed_index=seed, split=split)
    _write_state_bundle(task, state)

    trial: dict[str, Any] = {
        "seed": seed,
        "letter": letter,
        "split": split,
        "task_id": task_id,
        "trial_dir": str(out),
        "pipeline_status": "started",
    }
    try:
        _run_logged(
            logs,
            "01_render",
            [
                _python(),
                str(_script("render_simtoolreal_rgbd_scene_evidence.py")),
                "--task",
                str(task),
                "--output-dir",
                str(out / "render_soft02_front"),
                "--device",
                str(args.device),
                "--grounding-camera",
                "whiteboard_front",
                "--light-preset",
                "soft",
                "--soft-light-indices",
                "0,2",
            ],
        )
        _run_logged(
            logs,
            "02_vlm_scene",
            [
                _python(),
                str(_script("run_vlm_scene_keypoint_grounding.py")),
                "--scene-evidence",
                str(out / "render_soft02_front/scene_evidence.json"),
                "--output-dir",
                str(out / "scene_grounding"),
            ],
        )
        _draw_scene_overlay(
            out / "scene_grounding/scene_keypoints.json",
            out / "render_soft02_front/sim_grounding_whiteboard_front_rgb.png",
            out / "scene_grounding/scene_keypoints_overlay.png",
        )
        _run_logged(
            logs,
            "03_lift",
            [
                _python(),
                str(_script("lift_scene_keypoints_to_constraint_program.py")),
                "--scene-evidence",
                str(out / "render_soft02_front/scene_evidence.json"),
                "--scene-keypoints",
                str(out / "scene_grounding/scene_keypoints.json"),
                "--output-dir",
                str(out / "lifted_program"),
                "--task",
                str(task),
                "--direction-source",
                "path_geometry",
                "--tool-motion-axis-target-ref",
                "scene.direction",
                "--tool-motion-axis-relation",
                "parallel",
                "--tool-motion-axis-ref",
                "tool.motion_axis",
                "--target-approach-axis-source",
                str(args.target_approach_axis_source),
                "--end-surface-policy",
                "anchor_contact_plane",
                "--depth-sample-policy",
                "nearest",
            ],
        )
        _validate_lifted_refs_on_support_surface(
            task_path=task,
            lifted_refs_path=out / "lifted_program/lifted_refs.json",
            output_path=out / "lifted_program/surface_ref_validation.json",
            max_surface_ref_gap_m=float(args.max_surface_ref_gap_m),
        )
        _run_logged(
            logs,
            "04_merge_tool_refs",
            [
                _python(),
                str(_script("merge_tool_refs_into_constraint_program.py")),
                "--program",
                str(out / "lifted_program/constraint_program.json"),
                "--axis-binding",
                str(tool_refs / "axis_binding.json"),
                "--axis-mesh-report",
                str(tool_refs / "axis_mesh_overlay_report.json"),
                "--contact-point-selection",
                str(tool_refs / "contact_point_selection.json"),
                "--output",
                str(out / "lifted_program/constraint_program_with_tool_refs.json"),
            ],
        )
        _run_logged(
            logs,
            "05_compile",
            [
                _python(),
                str(_script("compile_image_grounded_with_mainline.py")),
                "--constraint-program",
                str(out / "lifted_program/constraint_program_with_tool_refs.json"),
                "--task",
                str(task),
                "--output-dir",
                str(out / "compile"),
                "--min-clearance-m",
                str(args.compile_min_clearance_m),
                "--swept-min-clearance-m",
                str(args.compile_swept_min_clearance_m),
            ],
        )
        if not args.skip_rollout:
            rollout_env = _env()
            if args.gpu_side is not None:
                rollout_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_side)
            rollout_cmd = [
                _python(),
                str(ROOT / "parttoolgen/src/parttoolgen/execution/simtoolreal_rollout.py"),
                "--fixed-goals",
                str(out / "compile/simtoolreal_fixed_goals.json"),
                "--subgoals",
                str(out / "compile/path_solved_tool_pose_subgoals.json"),
                "--scene",
                str(out / "compile/observed_scene_graph.json"),
                "--state-bundle",
                str(state),
                "--output-dir",
                str(out / "rollout"),
                "--object-name",
                str(args.object_name),
                "--device",
                str(args.device),
                "--headless",
                "--max-steps",
                str(args.rollout_max_steps),
                "--eval-success-tolerance-m",
                str(args.eval_success_tolerance_m),
                "--initial-grasp-hold-steps",
                "8",
                "--hide-goal-object-visual",
                "--light-preset",
                "soft",
                "--soft-light-indices",
                "0,2",
            ]
            if args.capture_video_every > 0 and trial_index % args.capture_video_every == 0:
                rollout_cmd.extend(["--capture-video", "--capture-side-board-video", "--video-stride", str(args.video_stride)])
            _run_logged(logs, "06_rollout", rollout_cmd, env=rollout_env)
            (out / "ink_evidence").mkdir(parents=True, exist_ok=True)
            _run_logged(
                logs,
                "07_ink",
                [
                    _python(),
                    "-m",
                    "parttoolgen.evaluation.marker_trace_ink_evidence",
                    "--base-rgb",
                    str(out / "render_soft02_front/sim_grounding_whiteboard_front_rgb.png"),
                    "--camera",
                    str(out / "render_soft02_front/sim_grounding_whiteboard_front_camera.json"),
                    "--tracking-samples",
                    str(out / "rollout/tracking_samples.json"),
                    "--tool-ref-selection",
                    str(tool_refs / "contact_point_selection.json"),
                    "--allow-pending-tool-labels",
                    "--task",
                    str(task),
                    "--max-contact-gap-m",
                    str(args.max_contact_gap_m),
                    "--max-penetration-m",
                    str(args.max_penetration_m),
                    "--ink-width-px",
                    str(args.ink_width_px),
                    "--ink-rgb",
                    "0",
                    "0",
                    "0",
                    "--output-png",
                    str(out / "ink_evidence/ink_trace_overlay.png"),
                    "--output-metrics",
                    str(out / "ink_evidence/ink_trace_metrics.json"),
                ],
            )
            judge_cmd = [
                _python(),
                "-m",
                "parttoolgen.evaluation.marker_trace_vlm_judge",
                "--task",
                str(task),
                "--ink-overlay",
                str(out / "ink_evidence/ink_trace_overlay.png"),
                "--output",
                str(out / "ink_evidence/vlm_judge_result.json"),
                "--requested-letter",
                letter,
                "--confidence-threshold",
                str(args.vlm_judge_confidence_threshold),
            ]
            if args.vlm_judge_model:
                judge_cmd.extend(["--model", str(args.vlm_judge_model)])
            _run_logged(logs, "08_vlm_judge", judge_cmd)
        trial["pipeline_status"] = "ok"
        trial["exit_code"] = 0
    except subprocess.CalledProcessError as exc:
        trial["pipeline_status"] = "failed"
        trial["exit_code"] = int(exc.returncode)
        trial["failed_command"] = " ".join(str(item) for item in exc.cmd)
    except Exception as exc:
        trial["pipeline_status"] = "failed"
        trial["exit_code"] = 1
        trial["failure_reason"] = str(exc)
        if "scene refs are not on the selected support surface" in str(exc):
            trial["failure_category"] = "vlm_scene_marking_error"
            trial["failure_layer"] = "vlm_scene_grounding"

    rollout = _load_json(out / "rollout/rollout_summary.json") if (out / "rollout/rollout_summary.json").exists() else {}
    ink = _load_json(out / "ink_evidence/ink_trace_metrics.json") if (out / "ink_evidence/ink_trace_metrics.json").exists() else {}
    judge = _load_json(out / "ink_evidence/vlm_judge_result.json") if (out / "ink_evidence/vlm_judge_result.json").exists() else {}
    passed = bool(judge.get("success")) if isinstance(judge, dict) and judge else False
    trial.update(
        {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "vlm_judge_checked": bool(judge),
            "vlm_judge_success": passed,
            "vlm_judge_recognizable": judge.get("recognizable") if isinstance(judge, dict) else None,
            "vlm_judge_confidence": judge.get("confidence") if isinstance(judge, dict) else None,
            "vlm_judge_reason": judge.get("reason") if isinstance(judge, dict) else None,
            "vlm_judge_failure_modes": judge.get("failure_modes") if isinstance(judge, dict) else None,
            "contact_sample_count": ink.get("contact_sample_count") if isinstance(ink, dict) else None,
            "ink_segment_count": ink.get("ink_segment_count") if isinstance(ink, dict) else None,
            "ink_length_m": ink.get("ink_length_m") if isinstance(ink, dict) else None,
            "policy_goal_passed": rollout.get("policy_goal_passed") if isinstance(rollout, dict) else None,
            "tracking_execution_passed": rollout.get("tracking_execution_passed") if isinstance(rollout, dict) else None,
            "semantic_execution_passed": rollout.get("semantic_execution_passed") if isinstance(rollout, dict) else None,
            "reached_goal_count": rollout.get("reached_goal_count") if isinstance(rollout, dict) else None,
            "num_goals": rollout.get("num_goals") if isinstance(rollout, dict) else None,
            "scene_keypoints_overlay": str(out / "scene_grounding/scene_keypoints_overlay.png"),
            "ink_trace_overlay": str(out / "ink_evidence/ink_trace_overlay.png"),
            "vlm_judge_result": str(out / "ink_evidence/vlm_judge_result.json"),
            "side_board_front_render_video_path": rollout.get("side_board_front_render_video_path") if isinstance(rollout, dict) else None,
        }
    )
    _write_json(out / "trial_result.json", trial)
    return trial


def _summarize(root: Path, trials: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    normalized_trials = []
    for row in trials:
        trial = dict(row)
        if "passed" not in trial:
            trial["passed"] = trial.get("vlm_judge_success") is True
        if "status" not in trial:
            trial["status"] = "passed" if trial["passed"] else "failed"
        if "task_id" not in trial and "seed" in trial and trial.get("split"):
            trial["task_id"] = f"marker_trace_{trial['split']}_seed_{int(trial['seed']):03d}"
        normalized_trials.append(trial)
    successes = sum(1 for row in normalized_trials if row.get("passed") is True)
    pass_rate = successes / len(normalized_trials) if normalized_trials else 0.0
    summary = {
        "schema_version": "toolframe_image_grounded_marker_trace_eval_v1",
        "output_root": str(root),
        "root": str(root),
        "success_metric": "vlm_judge_success",
        "vlm_judge_confidence_threshold": float(threshold),
        "trial_count": len(normalized_trials),
        "passed_count": successes,
        "pass_rate": pass_rate,
        "successes": successes,
        "success_rate": successes / len(normalized_trials) if normalized_trials else None,
        "trials": normalized_trials,
    }
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "vlm_judge_manifest.json",
        {
            "root": str(root),
            "success_metric": "vlm_judge_success",
            "judge_labels": ["success", "failure"],
            "judge_items": normalized_trials,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-task", default=str(SRC_TASK))
    parser.add_argument("--tool-refs-dir", default=str(DEFAULT_TOOL_REFS))
    parser.add_argument("--requested-letter", default=None)
    parser.add_argument("--split", default="write_c")
    parser.add_argument("--task-language", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-side", default=None)
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--capture-video-every", type=int, default=1)
    parser.add_argument("--video-stride", type=int, default=8)
    parser.add_argument("--eval-success-tolerance-m", type=float, default=0.018)
    parser.add_argument("--vlm-judge-model", default="")
    parser.add_argument("--vlm-judge-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--object-name", default="staples_marker")
    parser.add_argument(
        "--target-approach-axis-source",
        default="opposite_scene_direction",
        choices=["opposite_scene_direction"],
    )
    parser.add_argument("--compile-min-clearance-m", type=float, default=0.0)
    parser.add_argument("--compile-swept-min-clearance-m", type=float, default=0.0)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.002)
    parser.add_argument("--max-penetration-m", type=float, default=0.002)
    parser.add_argument("--max-surface-ref-gap-m", type=float, default=0.03)
    parser.add_argument("--ink-width-px", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    config = _load_json(args.benchmark_config) if args.benchmark_config else {}
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.benchmark_config:
        _write_json(
            root / "benchmark_config.resolved.json",
            {
                "source": str(Path(args.benchmark_config).resolve()),
                "runner_args_public": config.get("runner_args", {}) if isinstance(config, dict) else {},
                "private_manifest_omitted": True,
            },
        )
    tool_refs_dir = Path(args.tool_refs_dir).expanduser()
    if not tool_refs_dir.is_absolute():
        tool_refs_dir = ROOT / tool_refs_dir
    tool_selection_evidence = write_tool_selection_visual_evidence(
        output_root=root,
        tool_refs_dir=tool_refs_dir,
        task_family="marker_trace",
        split=args.split,
    )
    _write_json(
        root / "run_config.json",
        {
            "benchmark_config": str(Path(args.benchmark_config).resolve())
            if args.benchmark_config
            else None,
            "task_source": str(Path(args.source_task).resolve()),
            "tool_refs_dir": str(tool_refs_dir),
            "tool_selection_visual_evidence": tool_selection_evidence,
            "requested_letter": args.requested_letter,
            "split": args.split,
            "seed_start": args.seed_start,
            "trials": args.trials,
            "device": args.device,
            "rollout_max_steps": args.rollout_max_steps,
            "capture_video_every": args.capture_video_every,
            "skip_rollout": args.skip_rollout,
        },
    )
    trials: list[dict[str, Any]] = []
    for trial_index in range(int(args.trials)):
        seed = int(args.seed_start) + trial_index
        trial_result = root / f"seed_{seed:03d}" / "trial_result.json"
        if args.resume and trial_result.exists():
            trials.append(_load_json(trial_result))
            continue
        trials.append(_run_one(args, config=config, trial_index=trial_index))
    summary = _summarize(root, trials, threshold=args.vlm_judge_confidence_threshold)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
