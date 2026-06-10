#!/usr/bin/env python3
"""Run the side-board HDU eraser benchmark.

This task reuses the canonical side-board whiteboard environment and places a
visual-only HDU decal on the whiteboard. The method chain is the same public
image-grounded pipeline used by the other ToolFrame tasks: VLM scene refs,
RGB-D lifting, public flat-eraser tool refs, whole-tool SE(3) subgoals,
SimToolReal rollout, and an explicit task-level erased-area success check.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from parttoolgen.adapters.simtoolreal_policy_priors import (
    table_contact_start_pose_from_state_bundle,
)
from parttoolgen.evaluation.tool_selection_evidence import (
    write_tool_selection_visual_evidence,
)
from parttoolgen.pipeline.trial_tool_refs import (
    add_tool_ref_mode_args,
    resolve_trial_tool_refs,
)


ROOT = Path(__file__).resolve().parents[5]
TOOL_REFS_DIR = ROOT / "configs/tool_refs/flat_eraser"
TABLE_ASSET = "urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf"
WHITEBOARD_ASSET = "urdf/parttoolgen_benchmark/marker_trace_whiteboard/left_side_vertical_whiteboard_002x0475x101.urdf"
HDU_ASSET = "urdf/parttoolgen_benchmark/eraser_wipe_hdu_sticker/hdu_sticker_0144x0065.urdf"
TOOL_MESH = "simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj"
FRONT_OF_ARM_FLAT_ERASER_START_POSE = [
    0.02,
    0.135,
    0.560028845279411,
    0.7198971082016096,
    -0.010326421680771691,
    0.00572532125736636,
    0.6939803594589194,
]
TABLE_Z = 0.38
TOOL_TABLE_CLEARANCE_M = 0.003

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


def _front_of_arm_tool_start_pose() -> list[float]:
    projected = table_contact_start_pose_from_state_bundle(
        start_pose={
            "position": [float(item) for item in FRONT_OF_ARM_FLAT_ERASER_START_POSE[:3]],
            "orientation_xyzw": [
                float(item) for item in FRONT_OF_ARM_FLAT_ERASER_START_POSE[3:7]
            ],
        },
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, TABLE_Z, 0.0, 0.0, 0.0, 1.0],
            "table_asset": TABLE_ASSET,
        },
        tool_mesh_path=TOOL_MESH,
        min_tool_table_clearance_m=TOOL_TABLE_CLEARANCE_M,
    )
    return [
        *[float(item) for item in projected["position"][:3]],
        *[float(item) for item in projected["orientation_xyzw"][:4]],
    ]


def _run_logged(log_dir: Path, name: str, cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== START {name} =====\n")
        handle.write(" ".join(str(item) for item in cmd) + "\n")
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


def _draw_dashed_line(draw: ImageDraw.ImageDraw, a: tuple[float, float] | None, b: tuple[float, float] | None) -> None:
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
    for waypoint in waypoints:
        _draw_marker(draw, waypoint, BLUE)
    _draw_marker(draw, end, GREEN)
    _draw_marker(draw, start, PINK)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _sample_hdu_center(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [
        0.359,
        round(rng.uniform(0.02, 0.08), 4),
        round(rng.uniform(0.60, 0.68), 4),
    ]


def _task_for_seed(seed: int) -> dict[str, Any]:
    center = _sample_hdu_center(seed)
    start_pose = _front_of_arm_tool_start_pose()
    return {
        "schema_version": "parttoolgen_tool_use_task_v0",
        "task_id": f"flat_eraser_side_whiteboard_hdu_erase_seed_{seed:03d}",
        "family": "eraser_wipe",
        "split": "side_whiteboard_hdu_erase",
        "instruction": "Use this eraser to erase the whole HDU pattern on the whiteboard.",
        "tool": {
            "tool_id": "flat_eraser",
            "tool_prior_dir": "configs/tool_refs/flat_eraser",
        },
        "execution": {
            "simtoolreal_category": "eraser",
            "simtoolreal_table_asset": TABLE_ASSET,
            "simtoolreal_tool_mesh": TOOL_MESH,
            "simtoolreal_start_pose_trajectory": "simtoolreal/dextoolbench/trajectories/eraser/flat_eraser/wipe_c.json",
            "start_pose_policy": "fixed_front_of_arm_flat_eraser_start_pose",
            "simtoolreal_start_pose": start_pose,
            "state_bundle": {
                "environment_family": "side_board",
                "environment_family_source": "parttoolgen_side_board_hdu_erase_benchmark",
                "table_pose_xyzw": [0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0],
                "table_asset": TABLE_ASSET,
                "table_z": 0.38,
                "object_name": "flat_eraser",
                "object_category": "eraser",
                "task_name": "side_whiteboard_hdu_erase",
                "target_collision_role": "visual_hdu_sticker_on_left_side_vertical_whiteboard_no_collision",
                "min_tool_table_clearance_m": 0.0003,
                "sampled_hdu_center_world": center,
                "sample_seed": seed,
                "parttoolgen_benchmark_extra_actors": [
                    {
                        "name": "marker_trace_whiteboard",
                        "asset_file": WHITEBOARD_ASSET,
                        "pose_xyzw": [0.38, 0.0, 0.505, 0.0, 0.0, 0.0, 1.0],
                        "color_rgb": [0.96, 0.98, 0.96],
                        "disable_gravity": True,
                        "fix_base_link": True,
                        "friction": 0.8,
                    },
                    {
                        "name": "side_whiteboard_hdu_sticker",
                        "asset_file": HDU_ASSET,
                        "pose_xyzw": [center[0], center[1], center[2], 0.0, 0.0, 0.0, 1.0],
                        "color_rgb": [0.02, 0.08, 0.95],
                        "disable_gravity": True,
                        "fix_base_link": True,
                        "friction": 0.0,
                        "semantic_role": "visual_sticker_to_erase",
                        "collision_policy": "visual_only_no_collision_expected",
                    },
                ],
            },
        },
        "public_scene": {
            "provider": "simtoolreal_rgbd_image_grounding",
            "target_selection_mode": "vlm_select_visible_hdu_pattern_start_waypoint_end_refs_on_whiteboard",
            "method_inputs": [
                "task language",
                "whiteboard-front SimToolReal RGB-D render",
                "VLM-marked public HDU pattern refs",
                "public flat_eraser tool refs",
            ],
            "forbidden_method_inputs": [
                "private_eval.target_geometry",
                "oracle target axes",
                "oracle trajectories",
                "direct robot trajectory from VLM",
                "tool_family",
                "task_family",
                "primitive fallback",
            ],
        },
        "metadata": {
            "seed": seed,
            "environment_family": "side_board",
            "environment_id": "side_left_vertical_whiteboard_soft02",
            "target_kind": "left_side_vertical_whiteboard_visible_hdu_sticker_decal",
            "whiteboard_center_world": [0.38, 0.0, 0.505],
            "public_whiteboard_surface_normal_world": [-1.0, 0.0, 0.0],
            "hdu_sticker_center_world": center,
            "hdu_sticker_size_yz": [0.144, 0.065],
            "hdu_sticker_stroke_width_m": 0.008,
            "hdu_sticker_asset_file": HDU_ASSET,
            "hdu_sticker_collision_policy": "visual_only_no_collision; eraser contact should be with whiteboard surface",
            "tool_start_pose_xyzw": start_pose,
            "tool_start_policy": "fixed_front_of_arm_start_pose_shared_by_vlm_render_and_rollout",
            "randomization_policy": "seeded HDU center within reviewed near-arm side-board reachable window",
        },
    }


def _write_state_bundle(task: dict[str, Any], output: Path) -> None:
    state = task.get("execution", {}).get("state_bundle")
    if not isinstance(state, dict):
        raise ValueError("missing execution.state_bundle")
    _write_json(output, state)


def _path_record(path: Path, *, role: str, required: bool = True) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "required": bool(required),
    }


def _safe_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _copy_tool_selection_visual_evidence(*, output_root: Path, tool_refs_dir: Path) -> dict[str, Any]:
    return write_tool_selection_visual_evidence(
        output_root=output_root,
        tool_refs_dir=tool_refs_dir,
        task_family="eraser_wipe",
        split="side_whiteboard_hdu_erase",
        repo_root=ROOT,
    )


def _write_evidence_manifest(
    *,
    trial_dir: Path,
    task_path: Path,
    state_path: Path,
    tool_refs_dir: Path,
) -> dict[str, Any]:
    scene_keypoints = _safe_json(trial_dir / "scene_grounding/scene_keypoints.json")
    tool_selection = _safe_json(tool_refs_dir / "contact_point_selection.json")
    lifted_refs = _safe_json(trial_dir / "lifted_program/lifted_refs.json")
    rollout_summary = _safe_json(trial_dir / "rollout/rollout_summary.json")
    success_metrics = _safe_json(trial_dir / "hdu_erase_success/erased_area_metrics.json")

    manifest = {
        "schema_version": "toolframe_trial_evidence_manifest_v1",
        "task_family": "eraser_wipe",
        "split": "side_whiteboard_hdu_erase",
        "trial_dir": str(trial_dir),
        "purpose": (
            "Per-seed audit package for public VLM choices, lifted refs, "
            "tool refs, rollout evidence, and task-level success."
        ),
        "paths": {
            "task": _path_record(task_path, role="task_json"),
            "state_bundle": _path_record(state_path, role="simtoolreal_state_bundle"),
            "run_level_tool_selection_visual_evidence": _path_record(
                trial_dir.parent / "tool_selection_visual_evidence/manifest.json",
                role="shared_tool_selection_visual_evidence_manifest",
                required=False,
            ),
            "public_scene_evidence": [
                _path_record(trial_dir / "render_soft02_front/scene_evidence.json", role="scene_evidence_json"),
                _path_record(trial_dir / "render_soft02_front/sim_grounding_whiteboard_front_rgb.png", role="vlm_input_rgb"),
                _path_record(trial_dir / "render_soft02_front/sim_grounding_whiteboard_front_depth_vis.png", role="vlm_input_depth_visualization"),
                _path_record(trial_dir / "render_soft02_front/sim_grounding_whiteboard_front_camera.json", role="vlm_input_camera"),
            ],
            "vlm_scene_selection": [
                _path_record(trial_dir / "scene_grounding/scene_keypoints.json", role="vlm_scene_keypoints_json"),
                _path_record(trial_dir / "scene_grounding/scene_keypoints_overlay.png", role="vlm_scene_keypoints_overlay"),
            ],
            "tool_selection": [
                _path_record(tool_refs_dir / "contact_point_selection.json", role="tool_contact_point_and_motion_axis_selection"),
                _path_record(tool_refs_dir / "axis_binding.json", role="tool_axis_binding"),
                _path_record(tool_refs_dir / "axis_mesh_overlay_report.json", role="tool_axis_mesh_report"),
            ],
            "lifting": [
                _path_record(trial_dir / "lifted_program/lifted_refs.json", role="lifted_3d_scene_refs"),
                _path_record(trial_dir / "lifted_program/constraint_program.json", role="scene_constraint_program"),
                _path_record(trial_dir / "lifted_program/constraint_program_with_tool_refs.json", role="constraint_program_with_tool_refs"),
            ],
            "compile": [
                _path_record(trial_dir / "compile/compile_summary.json", role="compile_summary"),
                _path_record(trial_dir / "compile/program_ir.json", role="program_ir"),
                _path_record(trial_dir / "compile/path_solved_tool_pose_subgoals.json", role="path_solved_tool_pose_subgoals"),
                _path_record(trial_dir / "compile/simtoolreal_fixed_goals.json", role="simtoolreal_fixed_goals"),
            ],
            "rollout": [
                _path_record(trial_dir / "rollout/rollout_summary.json", role="rollout_summary"),
                _path_record(trial_dir / "rollout/tracking_samples.json", role="tracking_samples"),
                _path_record(trial_dir / "rollout/rollout_side_board_front_render.mp4", role="side_board_front_rollout_video", required=False),
                _path_record(trial_dir / "rollout/contact_events.json", role="sim_contact_events", required=False),
            ],
            "task_success": [
                _path_record(trial_dir / "hdu_erase_success/erased_area_metrics.json", role="hdu_erased_area_metrics"),
                _path_record(trial_dir / "hdu_erase_success/erased_area_overlay.png", role="hdu_erased_area_overlay"),
                _path_record(trial_dir / "hdu_erase_success/eraser_contact_coverage_mask.png", role="estimated_eraser_contact_coverage_mask"),
                _path_record(trial_dir / "hdu_erase_success/hdu_pattern_mask.png", role="hdu_pattern_mask"),
            ],
        },
        "vlm_scene_selection_summary": {
            "target_object": scene_keypoints.get("target_object") if isinstance(scene_keypoints, dict) else None,
            "start_point": scene_keypoints.get("start_point") if isinstance(scene_keypoints, dict) else None,
            "path_points": scene_keypoints.get("path_points") if isinstance(scene_keypoints, dict) else None,
            "intermediate_path_points": (
                scene_keypoints.get("intermediate_path_points")
                if isinstance(scene_keypoints, dict)
                else None
            ),
            "end_point": scene_keypoints.get("end_point") if isinstance(scene_keypoints, dict) else None,
            "action_direction": scene_keypoints.get("action_direction") if isinstance(scene_keypoints, dict) else None,
        },
        "tool_selection_summary": {
            "tool_id": tool_selection.get("tool_id") if isinstance(tool_selection, dict) else None,
            "selected_contact_point_part_ref": tool_selection.get("selected_contact_point_part_ref") if isinstance(tool_selection, dict) else None,
            "selected_contact_point_part_marker": tool_selection.get("selected_contact_point_part_marker") if isinstance(tool_selection, dict) else None,
            "selected_contact_point_face_marker": tool_selection.get("selected_contact_point_face_marker") if isinstance(tool_selection, dict) else None,
            "selected_contact_point_source_ref": tool_selection.get("selected_contact_point_source_ref") if isinstance(tool_selection, dict) else None,
            "selected_contact_point_surface_ref": tool_selection.get("selected_contact_point_surface_ref") if isinstance(tool_selection, dict) else None,
            "selected_motion_axis_face_marker": tool_selection.get("selected_motion_axis_face_marker") if isinstance(tool_selection, dict) else None,
            "selected_motion_axis_surface_ref": tool_selection.get("selected_motion_axis_surface_ref") if isinstance(tool_selection, dict) else None,
        },
        "lifted_refs_summary": {
            "start": lifted_refs.get("start") if isinstance(lifted_refs, dict) else None,
            "end": lifted_refs.get("end") if isinstance(lifted_refs, dict) else None,
            "path_points": lifted_refs.get("path_points") if isinstance(lifted_refs, dict) else None,
            "motion_path": lifted_refs.get("motion_path") if isinstance(lifted_refs, dict) else None,
            "direction": lifted_refs.get("direction") if isinstance(lifted_refs, dict) else None,
            "direction_source": lifted_refs.get("direction_source") if isinstance(lifted_refs, dict) else None,
            "end_surface_policy": lifted_refs.get("end_surface_policy") if isinstance(lifted_refs, dict) else None,
        },
        "rollout_summary": {
            "policy_goal_passed": rollout_summary.get("policy_goal_passed") if isinstance(rollout_summary, dict) else None,
            "tracking_execution_passed": rollout_summary.get("tracking_execution_passed") if isinstance(rollout_summary, dict) else None,
            "semantic_execution_passed": rollout_summary.get("semantic_execution_passed") if isinstance(rollout_summary, dict) else None,
            "reached_goal_count": rollout_summary.get("reached_goal_count") if isinstance(rollout_summary, dict) else None,
            "num_goals": rollout_summary.get("num_goals") if isinstance(rollout_summary, dict) else None,
        },
        "task_success_summary": {
            "metric": "estimated_hdu_erased_area_fraction",
            "passed": success_metrics.get("passed") if isinstance(success_metrics, dict) else None,
            "erased_area_fraction": success_metrics.get("erased_area_fraction") if isinstance(success_metrics, dict) else None,
            "thresholds": success_metrics.get("thresholds") if isinstance(success_metrics, dict) else None,
            "note": (
                "This HDU wipe checker estimates erased area from selected "
                "flat_eraser tool.contact_point/contact-face footprint samples; "
                "it is not simulator contact-manifold ground truth."
            ),
        },
    }
    output = trial_dir / "evidence_manifest.json"
    _write_json(output, manifest)
    return manifest


def _run_one(args: argparse.Namespace, *, trial_index: int) -> dict[str, Any]:
    seed = int(args.seed_start) + trial_index
    out = Path(args.output_root) / f"seed_{seed:03d}"
    logs = out / "logs"
    task_path = out / "task.json"
    state_path = out / "state_bundle.json"
    tool_refs = Path(args.tool_refs_dir)
    if not tool_refs.is_absolute():
        tool_refs = ROOT / tool_refs
    task = _task_for_seed(seed)
    _write_json(task_path, task)
    _write_state_bundle(task, state_path)
    trial_tool_refs = resolve_trial_tool_refs(
        repo_root=ROOT,
        python=_python(),
        args=args,
        trial_dir=out,
        static_tool_refs_dir=tool_refs,
        tool_id="flat_eraser",
        task_id=str(task.get("task_id") or f"side_hdu_erase_seed_{seed:03d}"),
        task_instruction=str(task.get("instruction") or "Use this eraser to erase the whole HDU pattern on the whiteboard."),
        log_path=logs / "tool_refs.json",
        env=_env(),
    )

    trial: dict[str, Any] = {
        "seed": seed,
        "task_id": task["task_id"],
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
                str(task_path),
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
                str(task_path),
                "--require-end-point",
                "--direction-source",
                "path_geometry",
                "--tool-motion-axis-target-ref",
                "scene.direction",
                "--tool-motion-axis-relation",
                "parallel",
                "--tool-motion-axis-ref",
                "tool.motion_axis",
                "--target-surface-normal-world",
                "-1",
                "0",
                "0",
                "--target-approach-axis-source",
                "opposite_scene_direction",
                "--end-surface-policy",
                "anchor_contact_plane",
                "--depth-sample-policy",
                "nearest",
            ],
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
                str(trial_tool_refs / "axis_binding.json"),
                "--axis-mesh-report",
                str(trial_tool_refs / "axis_mesh_overlay_report.json"),
                "--contact-point-selection",
                str(trial_tool_refs / "contact_point_selection.json"),
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
                str(task_path),
                "--output-dir",
                str(out / "compile"),
                "--approach-offset-m",
                str(args.approach_offset_m),
                "--min-clearance-m",
                "0.0",
                "--swept-min-clearance-m",
                "0.0",
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
                str(state_path),
                "--output-dir",
                str(out / "rollout"),
                "--object-name",
                "flat_eraser",
                "--device",
                str(args.device),
                "--headless",
                "--max-steps",
                str(args.rollout_max_steps),
                "--eval-success-tolerance-m",
                str(args.rollout_max_tracking_error_m),
                "--initial-grasp-hold-steps",
                "8",
                "--hide-goal-object-visual",
                "--light-preset",
                "soft",
                "--soft-light-indices",
                "0,2",
            ]
            if args.capture_video_every > 0 and trial_index % args.capture_video_every == 0:
                rollout_cmd.extend([
                    "--capture-video",
                    "--capture-side-board-video",
                    "--capture-subgoal-video",
                    "--capture-subgoal-side-board-video",
                    "--video-stride",
                    str(args.video_stride),
                ])
            _run_logged(logs, "06_rollout", rollout_cmd, env=rollout_env)
            _run_logged(
                logs,
                "07_hdu_erase_success",
                [
                    _python(),
                    "-m",
                    "parttoolgen.evaluation.side_board_hdu_erase_success",
                    "--task",
                    str(task_path),
                    "--tracking-samples",
                    str(out / "rollout/tracking_samples.json"),
                    "--output-dir",
                    str(out / "hdu_erase_success"),
                    "--tool-ref-selection",
                    str(tool_refs / "contact_point_selection.json"),
                    "--min-erased-fraction",
                    str(args.min_erased_fraction),
                ],
            )
        trial["pipeline_status"] = "ok"
        trial["exit_code"] = 0
    except subprocess.CalledProcessError as exc:
        trial["pipeline_status"] = "failed"
        trial["exit_code"] = int(exc.returncode)
        trial["failed_command"] = " ".join(str(item) for item in exc.cmd)

    rollout = _load_json(out / "rollout/rollout_summary.json") if (out / "rollout/rollout_summary.json").exists() else {}
    metrics = _load_json(out / "hdu_erase_success/erased_area_metrics.json") if (out / "hdu_erase_success/erased_area_metrics.json").exists() else {}
    passed = bool(metrics.get("passed")) if isinstance(metrics, dict) and metrics else False
    trial.update(
        {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "task_success_passed": passed,
            "erased_area_fraction": metrics.get("erased_area_fraction") if isinstance(metrics, dict) else None,
            "accepted_contact_sample_count": metrics.get("accepted_contact_sample_count") if isinstance(metrics, dict) else None,
            "hdu_erase_metrics": str(out / "hdu_erase_success/erased_area_metrics.json"),
            "hdu_erase_overlay": str(out / "hdu_erase_success/erased_area_overlay.png"),
            "policy_goal_passed": rollout.get("policy_goal_passed") if isinstance(rollout, dict) else None,
            "tracking_execution_passed": rollout.get("tracking_execution_passed") if isinstance(rollout, dict) else None,
            "semantic_execution_passed": rollout.get("semantic_execution_passed") if isinstance(rollout, dict) else None,
            "reached_goal_count": rollout.get("reached_goal_count") if isinstance(rollout, dict) else None,
            "num_goals": rollout.get("num_goals") if isinstance(rollout, dict) else None,
            "scene_keypoints_overlay": str(out / "scene_grounding/scene_keypoints_overlay.png"),
            "side_board_front_render_video_path": rollout.get("side_board_front_render_video_path") if isinstance(rollout, dict) else None,
        }
    )
    evidence_manifest = _write_evidence_manifest(
        trial_dir=out,
        task_path=task_path,
        state_path=state_path,
        tool_refs_dir=tool_refs,
    )
    trial["evidence_manifest"] = str(out / "evidence_manifest.json")
    trial["evidence_manifest_schema"] = evidence_manifest["schema_version"]
    _write_json(out / "trial_result.json", trial)
    return trial


def _summarize(root: Path, trials: list[dict[str, Any]], *, min_erased_fraction: float) -> dict[str, Any]:
    normalized_trials = []
    for row in trials:
        trial = dict(row)
        if "passed" not in trial:
            trial["passed"] = trial.get("task_success_passed") is True
        if "status" not in trial:
            trial["status"] = "passed" if trial["passed"] else "failed"
        if "task_id" not in trial and "seed" in trial:
            trial["task_id"] = f"flat_eraser_side_whiteboard_hdu_erase_seed_{int(trial['seed']):03d}"
        normalized_trials.append(trial)
    successes = sum(1 for row in normalized_trials if row.get("passed") is True)
    pass_rate = successes / len(normalized_trials) if normalized_trials else 0.0
    summary = {
        "schema_version": "toolframe_image_grounded_side_board_hdu_erase_eval_v1",
        "output_root": str(root),
        "root": str(root),
        "success_metric": "hdu_erased_area_fraction",
        "min_erased_fraction": float(min_erased_fraction),
        "trial_count": len(normalized_trials),
        "passed_count": successes,
        "pass_rate": pass_rate,
        "successes": successes,
        "success_rate": successes / len(normalized_trials) if normalized_trials else None,
        "diagnostic_metrics": [
            "policy_goal_passed",
            "tracking_execution_passed",
            "semantic_execution_passed",
            "accepted_contact_sample_count",
        ],
        "trials": normalized_trials,
    }
    _write_json(root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tool-refs-dir", default=str(TOOL_REFS_DIR))
    add_tool_ref_mode_args(parser)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-side", default=None)
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--rollout-max-tracking-error-m", type=float, default=0.018)
    parser.add_argument("--capture-video-every", type=int, default=1)
    parser.add_argument("--video-stride", type=int, default=8)
    parser.add_argument("--approach-offset-m", type=float, default=0.03)
    parser.add_argument("--min-erased-fraction", type=float, default=0.70)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    config = _load_json(args.benchmark_config) if args.benchmark_config else {}
    if args.benchmark_config:
        _write_json(
            root / "benchmark_config.resolved.json",
            {
                "source": str(Path(args.benchmark_config).resolve()),
                "runner_args_public": config.get("runner_args", {}) if isinstance(config, dict) else {},
                "private_manifest_omitted": True,
            },
        )
    tool_refs = Path(args.tool_refs_dir)
    if not tool_refs.is_absolute():
        tool_refs = ROOT / tool_refs
    tool_selection_evidence = _copy_tool_selection_visual_evidence(output_root=root, tool_refs_dir=tool_refs)
    _write_json(
        root / "run_config.json",
        {
            "benchmark_config": str(Path(args.benchmark_config).resolve())
            if args.benchmark_config
            else None,
            "task_source": "runner_generated_side_board_hdu_erase_task",
            "tool_refs_dir": str(tool_refs),
            "tool_selection_visual_evidence": tool_selection_evidence,
            "seed_start": args.seed_start,
            "trials": args.trials,
            "device": args.device,
            "rollout_max_steps": args.rollout_max_steps,
            "rollout_max_tracking_error_m": args.rollout_max_tracking_error_m,
            "capture_video_every": args.capture_video_every,
            "approach_offset_m": args.approach_offset_m,
            "min_erased_fraction": args.min_erased_fraction,
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
        trials.append(_run_one(args, trial_index=trial_index))
    summary = _summarize(root, trials, min_erased_fraction=args.min_erased_fraction)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
