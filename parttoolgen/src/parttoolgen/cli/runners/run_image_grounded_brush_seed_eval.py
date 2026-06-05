#!/usr/bin/env python3
"""Run the clean image-grounded red-brush sweep benchmark over randomized seeds."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from parttoolgen.evaluation.task_success import (
    BRUSH_SWEEP_SUCCESS_POLICY,
    check_task_success,
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
TOOL_REFS_DIR = ROOT / "configs/tool_refs/red_brush"
TABLE_ASSET = "urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf"
TOOL_MESH = "simtoolreal/assets/urdf/dextoolbench/brush/red_brush/red_brush.obj"
TOOL_START_POSE_SOURCE = "simtoolreal/dextoolbench/trajectories/brush/red_brush/sweep_forward.json"
BLOCK_ASSET = "urdf/parttoolgen_benchmark/brush_sweep_block/sweep_block_045.urdf"
RED_TARGET_ZONE_ASSET = "urdf/parttoolgen_benchmark/brush_sweep_target_zone/red_target_zone_014x010.urdf"
TABLE_Z = 0.38
TOOL_TABLE_CLEARANCE_M = 0.003
BLOCK_Z = 0.5525
BASE_BLOCK_XY = [0.24, -0.13]
DEFAULT_TARGET_ZONE_CENTER_XY = [0.24, -0.205]
DEFAULT_TARGET_ZONE_SIZE_XY = [0.14, 0.10]
DEFAULT_TARGET_ZONE_Z = 0.5315
FIXED_GRASP_READY_START_POSE = [
    0.02,
    0.135,
    0.5506258883605348,
    0.7035186228215238,
    0.021312152321177134,
    0.016647372965644915,
    0.7101620973271443,
]
BRUSH_BACK_TO_ARM_YAW_OFFSET_DEG = 180.0
SWEEP_DIRECTIONS = {
    "forward": {
        "axis": [0.0, -1.0, 0.0],
        "instruction": "Use this brush to sweep the block forward.",
    },
    "backward": {
        "axis": [0.0, 1.0, 0.0],
        "instruction": "Use this brush to sweep the block backward.",
    },
    "right": {
        "axis": [1.0, 0.0, 0.0],
        "instruction": "Use this brush to sweep the block to the right.",
    },
    "left": {
        "axis": [-1.0, 0.0, 0.0],
        "instruction": "Use this brush to sweep the block to the left.",
    },
}


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = [float(item) for item in a]
    bx, by, bz, bw = [float(item) for item in b]
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_from_z_yaw(yaw_rad: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)]


def _quat_rotate(q: list[float], p: list[float]) -> list[float]:
    x, y, z, w = [float(item) for item in q]
    px, py, pz = [float(item) for item in p]
    ix = w * px + y * pz - z * py
    iy = w * py + z * px - x * pz
    iz = w * pz + x * py - y * px
    iw = -x * px - y * py - z * pz
    return [
        ix * w + iw * -x + iy * -z - iz * -y,
        iy * w + iw * -y + iz * -x - ix * -z,
        iz * w + iw * -z + ix * -y - iy * -x,
    ]


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


def _angle_wrap(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def _start_pose_for_sweep_direction(base_pose: list[float], sweep_direction: str, mode: str) -> tuple[list[float], float]:
    pose = [float(item) for item in base_pose]
    if mode != "perpendicular_motion":
        return pose, 0.0
    action_axis = SWEEP_DIRECTIONS[sweep_direction]["axis"]
    ax, ay = float(action_axis[0]), float(action_axis[1])
    if abs(ax) + abs(ay) <= 1e-9:
        return pose, 0.0
    q0 = pose[3:7]
    long_axis_world = _quat_rotate(q0, [1.0, 0.0, 0.0])
    current_yaw = math.atan2(long_axis_world[1], long_axis_world[0])
    candidates = [
        math.atan2(ax, -ay),  # clockwise perpendicular
        math.atan2(-ax, ay),  # counter-clockwise perpendicular
    ]
    # Red-brush handle lies on local -X. Prefer the perpendicular placement
    # whose handle side points toward the robot arm (+Y), then minimize yaw edit.
    desired_yaw = min(
        candidates,
        key=lambda yaw: (-(-math.sin(yaw)), abs(_angle_wrap(yaw - current_yaw))),
    )
    yaw_delta = _angle_wrap(desired_yaw - current_yaw)
    pose[3:7] = _quat_mul(_quat_from_z_yaw(yaw_delta), q0)
    return pose, math.degrees(yaw_delta)


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
        "grasp_observation_offset_m",
        "approach_offset_m",
        "target_mode",
        "target_zone_asset",
        "target_zone_center_xy",
        "target_zone_size_xy",
        "target_zone_z",
        "block_asset",
        "block_friction",
        "block_position_mode",
        "line_x_min",
        "line_x_max",
        "line_y",
        "sweep_direction",
        "start_orientation_mode",
        "tool_id",
        "tool_mesh",
        "tool_start_pose_source",
        "fixed_start_pose_xyzw",
        "resume",
        "skip_rollout",
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


def _sample_seed_params(seed: int, *, index: int, total: int, args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(seed)
    # SimToolReal's fixed policy is sensitive to the initial grasp pose. Keep
    # the brush at a stable grasp-ready pose near the robot-arm center, and
    # randomize only the target block. This makes the seeded benchmark test
    # image-grounded target localization and action planning rather than
    # low-level grasp recovery from arbitrary tool placements.
    if args.block_position_mode == "line_x":
        denom = max(1, int(total) - 1)
        alpha = float(index) / float(denom)
        block_x = float(args.line_x_min) * (1.0 - alpha) + float(args.line_x_max) * alpha
        block_y = float(args.line_y)
    else:
        block_x = BASE_BLOCK_XY[0] + rng.uniform(-0.045, 0.035)
        block_y = BASE_BLOCK_XY[1] + rng.uniform(-0.035, 0.045)
    start_pose, start_yaw_delta_deg = _start_pose_for_sweep_direction(
        [float(item) for item in args.fixed_start_pose_xyzw],
        args.sweep_direction,
        args.start_orientation_mode,
    )
    start_pose = _table_contact_tool_pose(start_pose, tool_mesh=args.tool_mesh)
    return {
        "block_xy": [block_x, block_y],
        "start_pose": start_pose,
        "start_yaw_delta_deg": start_yaw_delta_deg,
        "start_yaw_total_deg": BRUSH_BACK_TO_ARM_YAW_OFFSET_DEG + start_yaw_delta_deg,
        "brush_back_to_arm_yaw_offset_deg": BRUSH_BACK_TO_ARM_YAW_OFFSET_DEG,
        "start_to_block_distance_xy_m": math.hypot(
            float(start_pose[0]) - block_x,
            float(start_pose[1]) - block_y,
        ),
    }


def _build_task(
    *,
    task_id: str,
    seed: int,
    params: dict[str, Any],
    args: argparse.Namespace,
    sweep_direction: str = "forward",
    block_asset: str | None = None,
    block_friction: float | None = None,
) -> dict[str, Any]:
    bx, by = [float(item) for item in params["block_xy"]]
    direction = SWEEP_DIRECTIONS[sweep_direction]
    action_axis = [float(item) for item in direction["axis"]]
    actor: dict[str, Any] = {
        "name": "sweep_block",
        "asset_file": block_asset or BLOCK_ASSET,
        "pose_xyzw": [bx, by, BLOCK_Z, 0.0, 0.0, 0.0, 1.0],
        "color_rgb": [0.05, 0.2, 0.95],
        "disable_gravity": False,
        "fix_base_link": False,
        "friction": 0.8,
    }
    if block_friction is not None:
        actor["friction"] = float(block_friction)
    extra_actors = [actor]
    target_zone_metadata: dict[str, Any] = {}
    target_mode = str(args.target_mode)
    if target_mode == "red_zone":
        zone_center = [float(item) for item in args.target_zone_center_xy[:2]]
        zone_size = [float(item) for item in args.target_zone_size_xy[:2]]
        extra_actors.append(
            {
                "name": "red_target_zone",
                "asset_file": args.target_zone_asset,
                "pose_xyzw": [
                    zone_center[0],
                    zone_center[1],
                    float(args.target_zone_z),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "color_rgb": [1.0, 0.02, 0.02],
                "disable_gravity": True,
                "fix_base_link": True,
                "collapse_fixed_joints": False,
                "collision_filter": 1,
                "segmentation_id": 7,
            }
        )
        target_zone_metadata = {
            "target_mode": "red_zone",
            "target_zone_actor": "red_target_zone",
            "target_zone_center_xy_world": zone_center,
            "target_zone_size_xy_m": zone_size,
            "success_policy": "brush_sweep_target_zone_occupancy",
        }
    start_pose = [float(item) for item in params["start_pose"]]
    tool_refs_rel = _repo_rel(_tool_refs_dir(args))
    instruction = (
        "Use this brush to sweep the blue block into the red target zone."
        if target_mode == "red_zone"
        else str(direction["instruction"])
    )
    return {
        "schema_version": "parttoolgen_tool_use_task_v0",
        "task_id": task_id,
        "family": "image_grounded_sweep",
        "split": "dev",
        "instruction": instruction,
        "tool": {
            "tool_id": args.tool_id,
            "tool_prior_dir": tool_refs_rel,
        },
        "execution": {
            "simtoolreal_category": "brush",
            "simtoolreal_table_asset": TABLE_ASSET,
            "simtoolreal_tool_mesh": args.tool_mesh,
            "simtoolreal_start_pose_source": args.tool_start_pose_source,
            "start_pose_policy": "fixed_grasp_ready",
            "simtoolreal_start_pose": start_pose,
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, TABLE_Z, 0.0, 0.0, 0.0, 1.0],
                "table_asset": TABLE_ASSET,
                "table_z": TABLE_Z,
                "object_name": args.tool_id,
                "object_category": "brush",
                "task_name": "sweep_block_to_red_zone" if target_mode == "red_zone" else "sweep_forward",
                "target_collision_role": "dynamic_manipulable",
                "min_tool_table_clearance_m": 0.0003,
                "parttoolgen_benchmark_extra_actors": extra_actors,
            },
        },
        "public_scene": {
            "provider": "simtoolreal_rgbd_image_grounding",
            "target_selection_mode": (
                "vlm_select_source_block_and_target_zone"
                if target_mode == "red_zone"
                else "vlm_select_visible_single_target"
            ),
            "scene_evidence": [],
            "method_inputs": [
                "task language",
                "SimToolReal RGB-D render",
                f"public {args.tool_id} tool refs",
            ],
            "forbidden_method_inputs": [
                "private_eval.target_geometry",
                "tool_family",
                "task_family",
                "primitive fallback",
            ],
        },
        "metadata": {
            "seed": int(seed),
            "benchmark_role": "image_grounded_brush_seed_eval",
            "block_xy_world": [bx, by],
            "start_pose_xyzw": start_pose,
            "start_yaw_delta_deg": params["start_yaw_delta_deg"],
            "start_yaw_total_deg": params["start_yaw_total_deg"],
            "brush_back_to_arm_yaw_offset_deg": params["brush_back_to_arm_yaw_offset_deg"],
            "start_to_block_distance_xy_m": params["start_to_block_distance_xy_m"],
            "sweep_direction": sweep_direction,
            "action_axis_world": action_axis,
            "public_support_surface_normal_world": [0.0, 0.0, 1.0],
            "approach_offset_m": float(args.approach_offset_m),
            **target_zone_metadata,
            "seed_perturbation_policy": (
                "The brush starts from the same stable grasp-ready pose near the "
                "robot-arm center for every seed. Only the target block position "
                "is randomized within a table-safe workspace."
            ),
            "notes": (
                "Seeded ToolFrame image-grounded brush sweep task. The method-facing "
                f"inputs are task language, SimToolReal RGB-D, and public {args.tool_id} "
                "tool refs; hidden target geometry and primitive-family fields are "
                "not copied into the planning input."
            ),
        },
    }


def _target_zone_from_task(task: dict[str, Any]) -> dict[str, Any] | None:
    metadata = task.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("target_mode") != "red_zone":
        return None
    center = metadata.get("target_zone_center_xy_world")
    size = metadata.get("target_zone_size_xy_m")
    if not (isinstance(center, list) and len(center) >= 2 and isinstance(size, list) and len(size) >= 2):
        return None
    cx, cy = float(center[0]), float(center[1])
    sx, sy = abs(float(size[0])), abs(float(size[1]))
    return {
        "name": str(metadata.get("target_zone_actor") or "red_target_zone"),
        "center_xy_world": [cx, cy],
        "size_xy_m": [sx, sy],
        "bounds_xy_world": {
            "x": [cx - 0.5 * sx, cx + 0.5 * sx],
            "y": [cy - 0.5 * sy, cy + 0.5 * sy],
        },
    }


def _actor_by_name(actors: list[Any], name: str) -> dict[str, Any]:
    for actor in actors:
        if isinstance(actor, dict) and actor.get("name") == name:
            return actor
    return {}


def _effect_metrics(
    rollout_summary: dict[str, Any],
    lifted_refs: dict[str, Any],
    *,
    task: dict[str, Any],
) -> dict[str, Any]:
    actors = rollout_summary.get("benchmark_extra_actor_summary", {}).get("actors", [])
    actor = _actor_by_name(actors, "sweep_block") if isinstance(actors, list) else {}
    if not actor and actors and isinstance(actors[0], dict):
        actor = actors[0]
    init = actor.get("initial_pose_xyzw") or [0.0, 0.0, 0.0]
    final = actor.get("final_pose_xyzw") or [0.0, 0.0, 0.0]
    delta = [float(final[i]) - float(init[i]) for i in range(3)]
    xy = math.hypot(delta[0], delta[1])
    target_region = _target_zone_from_task(task)
    target_region_passed = None
    if target_region is not None:
        bounds = target_region["bounds_xy_world"]
        final_xy = [float(final[0]), float(final[1])]
        target_region_passed = (
            float(bounds["x"][0]) <= final_xy[0] <= float(bounds["x"][1])
            and float(bounds["y"][0]) <= final_xy[1] <= float(bounds["y"][1])
        )
    direction = lifted_refs.get("direction")
    if not isinstance(direction, list) or len(direction) < 2:
        metrics = {
            "target_xy_displacement_m": xy,
            "target_delta_xyz_m": delta,
            "target_final_xy_world": [float(final[0]), float(final[1])],
            "target_region": target_region,
            "target_region_passed": target_region_passed,
            "scene_direction_world": None,
            "projected_displacement_m": None,
            "direction_cosine": None,
            "target_effect_passed": bool(target_region_passed) if target_region is not None else False,
            "unresolved_reason": "missing_lifted_direction",
        }
        return metrics
    projected = delta[0] * float(direction[0]) + delta[1] * float(direction[1])
    direction_cosine = projected / max(xy, 1e-9)
    return {
        "target_xy_displacement_m": xy,
        "target_delta_xyz_m": delta,
        "target_final_xy_world": [float(final[0]), float(final[1])],
        "target_region": target_region,
        "target_region_passed": target_region_passed,
        "scene_direction_world": [float(item) for item in direction[:3]],
        "projected_displacement_m": projected,
        "direction_cosine": direction_cosine,
        "target_effect_passed": (
            bool(target_region_passed)
            if target_region is not None
            else xy >= 0.06 and direction_cosine >= 0.6
        ),
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


def _refresh_success_fields(result: dict[str, Any], *, skip_rollout: bool = False) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    rollout_summary = _load_optional_json(artifacts.get("rollout_summary")) if isinstance(artifacts, dict) else None
    execution_report = _load_optional_json(artifacts.get("execution_report")) if isinstance(artifacts, dict) else None
    compile_summary = _load_optional_json(artifacts.get("compile_summary")) if isinstance(artifacts, dict) else None
    effect = result.get("effect")
    if not isinstance(effect, dict):
        effect = {"target_effect_passed": bool(result.get("task_effect_passed"))}
    success_check = check_task_success(
        "brush_sweep",
        effect=effect,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        skip_rollout=skip_rollout,
    )
    refreshed = dict(result)
    refreshed["task_success_check"] = success_check
    refreshed["task_success_passed"] = bool(success_check.get("passed"))
    refreshed["success_policy"] = success_check["policy"]
    refreshed["passed"] = bool(success_check.get("passed"))
    if isinstance(compile_summary, dict):
        refreshed["compile_path_solver_passed"] = bool(
            compile_summary.get("path_solver_passed")
        )
        if not refreshed.get("num_goals"):
            refreshed["num_goals"] = int(
                compile_summary.get("num_fixed_goals")
                or compile_summary.get("num_subgoals")
                or 0
            )
    return refreshed


def _run_trial(*, seed: int, index: int, args: argparse.Namespace) -> dict[str, Any]:
    task_id = f"{args.tool_id}_sweep_block_seed_{seed:03d}"
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
    params = _sample_seed_params(seed, index=index - 1, total=args.trials, args=args)
    task = _build_task(
        task_id=task_id,
        seed=seed,
        params=params,
        sweep_direction=args.sweep_direction,
        block_asset=args.block_asset,
        block_friction=args.block_friction,
        args=args,
    )
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
    _run(
        [
            py,
            str(_script("run_vlm_scene_keypoint_grounding.py")),
            "--scene-evidence",
            str(render_dir / "scene_evidence.json"),
            "--output-dir",
            str(grounding_dir),
        ],
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
            "--end-surface-policy",
            "anchor_contact_plane",
            "--support-surface-normal-world",
            "0",
            "0",
            "1",
            "--tool-motion-axis-ref",
            "tool.motion_axis",
            "--tool-motion-axis-target-ref",
            "scene.direction",
            "--tool-motion-axis-relation",
            "parallel",
            "--require-end-point",
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
            "--approach-offset-m",
            str(args.approach_offset_m),
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
    rollout_dir = trial_dir / "rollout"
    rollout_summary_path = rollout_dir / "rollout_summary.json"
    execution_report_path = rollout_dir / "execution_report.json"
    rollout_summary: dict[str, Any] = {}
    execution_report: dict[str, Any] = {}
    rollout_returncode: int | None = None
    if not args.skip_rollout:
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
            "--max-tracking-error-m",
            str(args.rollout_max_tracking_error_m),
            "--force-no-reset",
        ]
        if args.grasp_observation_offset_m > 0:
            rollout_cmd.extend(
                ["--grasp-observation-offset-m", str(args.grasp_observation_offset_m)]
            )
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
        rollout_proc = _run(
            rollout_cmd, log_path=trial_dir / "logs/rollout.json", allow_failure=True
        )
        rollout_returncode = int(rollout_proc.returncode)
        rollout_summary = _load_optional_json(rollout_summary_path) or {}
        execution_report = _load_optional_json(execution_report_path) or {}
    if (
        not args.skip_rollout
        and (
            rollout_returncode != 0
            or not rollout_summary_path.exists()
            or not execution_report_path.exists()
        )
    ):
        compile_summary = _load_json(compile_dir / "compile_summary.json")
        visual_evidence = package_trial_visual_evidence(
            trial_dir=trial_dir,
            render_dir=render_dir,
            grounding_dir=grounding_dir,
            tool_refs_dir=_tool_refs_dir(args),
            compile_dir=compile_dir,
            rollout_dir=rollout_dir,
        )
        result = {
            "index": index,
            "seed": seed,
            "task_id": task_id,
            "passed": False,
            "task_effect_passed": False,
            "task_success_passed": False,
            "task_success_check": {
                "passed": False,
                "policy": "brush_sweep_directional_target_displacement",
                "failure_category": "rollout_failure",
                "reason": "rollout_failed_or_missing_summary",
                "rollout_returncode": rollout_returncode,
                "rollout_summary_exists": rollout_summary_path.exists(),
                "execution_report_exists": execution_report_path.exists(),
            },
            "success_policy": "brush_sweep_directional_target_displacement",
            "grasp_lift_passed": False,
            "diagnostic_policy_tracking_effect_passed": False,
            "strict_execution_passed": False,
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 0,
            "num_goals": int(compile_summary.get("num_subgoals", 0)),
            "num_steps": 0,
            "effect": {
                "target_effect_passed": False,
                "failure_category": "rollout_failure",
            },
            "seed_params": params,
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
                "rollout_summary": str(rollout_summary_path)
                if rollout_summary_path.exists()
                else None,
                "execution_report": str(execution_report_path)
                if execution_report_path.exists()
                else None,
                "rollout_video": str(rollout_dir / "rollout_render.mp4")
                if (rollout_dir / "rollout_render.mp4").exists()
                else None,
                "rollout_topdown_video": str(rollout_dir / "rollout_topdown_render.mp4")
                if (rollout_dir / "rollout_topdown_render.mp4").exists()
                else None,
                "visual_evidence": visual_evidence["evidence_dir"],
                "visual_evidence_manifest": visual_evidence["manifest"],
            },
            "failure_category": "rollout_failure",
            "failure_reason": "rollout_failed_or_missing_summary",
            "rollout_returncode": rollout_returncode,
        }
        _write_json(trial_dir / "trial_result.json", result)
        return result

    compile_summary = _load_json(compile_dir / "compile_summary.json")
    lifted_refs = _load_json(lifted_dir / "lifted_refs.json")
    effect = _effect_metrics(rollout_summary, lifted_refs, task=task)
    grasp_lift_passed = (
        _grasp_lift_passed(execution_report, compile_summary)
        if execution_report
        else None
    )
    diagnostic_policy_tracking_effect_passed = (
        bool(rollout_summary.get("policy_goal_passed"))
        and bool(rollout_summary.get("tracking_execution_passed"))
        and bool(effect["target_effect_passed"])
    )
    success_check = check_task_success(
        "brush_sweep",
        effect=effect,
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
        rollout_dir=rollout_dir,
    )
    passed = bool(success_check.get("passed"))
    result = {
        "index": index,
        "seed": seed,
        "task_id": task_id,
        "passed": passed,
        "skip_rollout": bool(args.skip_rollout),
        "task_effect_passed": bool(effect["target_effect_passed"]),
        "task_success_passed": passed,
        "task_success_check": success_check,
        "success_policy": success_check["policy"],
        "grasp_lift_passed": grasp_lift_passed,
        "diagnostic_policy_tracking_effect_passed": diagnostic_policy_tracking_effect_passed,
        "strict_execution_passed": diagnostic_policy_tracking_effect_passed,
        "policy_goal_passed": (
            bool(rollout_summary.get("policy_goal_passed")) if rollout_summary else None
        ),
        "tracking_execution_passed": (
            bool(rollout_summary.get("tracking_execution_passed")) if rollout_summary else None
        ),
        "semantic_execution_passed": (
            bool(rollout_summary.get("semantic_execution_passed")) if rollout_summary else None
        ),
        "final_successes": int(rollout_summary.get("final_successes", 0)) if rollout_summary else None,
        "num_goals": (
            int(rollout_summary.get("num_goals", 0))
            if rollout_summary
            else int(
                compile_summary.get("num_fixed_goals")
                or compile_summary.get("num_subgoals")
                or 0
            )
        ),
        "num_steps": int(rollout_summary.get("num_steps", 0)) if rollout_summary else None,
        "compile_path_solver_passed": bool(compile_summary.get("path_solver_passed")),
        "effect": effect,
        "seed_params": params,
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
            "rollout_summary": str(rollout_dir / "rollout_summary.json")
            if not args.skip_rollout
            else None,
            "execution_report": str(rollout_dir / "execution_report.json")
            if not args.skip_rollout
            else None,
            "rollout_video": str(rollout_dir / "rollout_render.mp4")
            if (rollout_dir / "rollout_render.mp4").exists()
            else None,
            "rollout_topdown_video": str(rollout_dir / "rollout_topdown_render.mp4")
            if (rollout_dir / "rollout_topdown_render.mp4").exists()
            else None,
            "visual_evidence": visual_evidence["evidence_dir"],
            "visual_evidence_manifest": visual_evidence["manifest"],
        },
        "failure_category": success_check.get("diagnostics", {}).get(
            "failure_category"
        ),
        "semantic_failed_stage": execution_report.get("semantic_failed_stage"),
        "semantic_failure_reason": (
            execution_report.get("stage_reports", [{}])[-1].get("semantic_failure_reason")
            if isinstance(execution_report.get("stage_reports"), list)
            and execution_report.get("stage_reports")
            else None
        ),
    }
    _write_json(trial_dir / "trial_result.json", result)
    return result


def _summary(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    passed = [row for row in rows if row.get("passed")]
    lift_rows = [row for row in rows if row.get("grasp_lift_passed") is not None]
    displacements = [
        float(row.get("effect", {}).get("target_xy_displacement_m"))
        for row in rows
        if isinstance(row.get("effect", {}).get("target_xy_displacement_m"), (int, float))
    ]
    projections = [
        float(row.get("effect", {}).get("projected_displacement_m"))
        for row in rows
        if isinstance(row.get("effect", {}).get("projected_displacement_m"), (int, float))
    ]
    summary = {
        "schema_version": "toolframe_image_grounded_brush_seed_eval_v1",
        "success_policy": (
            rows[0].get("task_success_check", {}).get("policy", {}).get("name")
            if rows
            and isinstance(rows[0].get("task_success_check"), dict)
            and isinstance(rows[0].get("task_success_check", {}).get("policy"), dict)
            else BRUSH_SWEEP_SUCCESS_POLICY["name"]
        ),
        "success_requires_tracking_execution_passed": False,
        "output_root": str(output_root),
        "trial_count": len(rows),
        "passed_count": len(passed),
        "pass_rate": (len(passed) / len(rows)) if rows else 0.0,
        "task_success_pass_rate": (
            sum(1 for row in rows if row.get("task_success_passed", row.get("passed"))) / len(rows)
        )
        if rows
        else 0.0,
        "policy_goal_pass_rate": (
            sum(1 for row in rows if row.get("policy_goal_passed")) / len(rows)
        )
        if rows
        else 0.0,
        "tracking_pass_rate": (
            sum(1 for row in rows if row.get("tracking_execution_passed")) / len(rows)
        )
        if rows
        else 0.0,
        "compile_path_pass_rate": (
            sum(1 for row in rows if row.get("compile_path_solver_passed")) / len(rows)
        )
        if rows
        else 0.0,
        "effect_pass_rate": (
            sum(1 for row in rows if row.get("effect", {}).get("target_effect_passed")) / len(rows)
        )
        if rows
        else 0.0,
        "grasp_lift_pass_rate": (
            sum(1 for row in lift_rows if row.get("grasp_lift_passed")) / len(lift_rows)
        )
        if lift_rows
        else None,
        "diagnostic_policy_tracking_effect_pass_rate": (
            sum(
                1
                for row in rows
                if row.get("diagnostic_policy_tracking_effect_passed", row.get("strict_execution_passed"))
            )
            / len(rows)
        )
        if rows
        else 0.0,
        "strict_execution_pass_rate": (
            sum(
                1
                for row in rows
                if row.get("diagnostic_policy_tracking_effect_passed", row.get("strict_execution_passed"))
            )
            / len(rows)
        )
        if rows
        else 0.0,
        "mean_target_xy_displacement_m": (sum(displacements) / len(displacements))
        if displacements
        else None,
        "mean_projected_displacement_m": (sum(projections) / len(projections))
        if projections
        else None,
        "trials": rows,
    }
    _write_json(output_root / "summary.json", summary)
    lines = [
        "# Image-Grounded Brush Seed Eval",
        "",
        f"- trials: {summary['trial_count']}",
        f"- success policy: {summary['success_policy']}",
        f"- pass rate: {summary['pass_rate']:.3f}",
        f"- task success pass rate: {summary['task_success_pass_rate']:.3f}",
        f"- policy goal pass rate: {summary['policy_goal_pass_rate']:.3f}",
        f"- tracking pass rate: {summary['tracking_pass_rate']:.3f}",
        f"- compile path pass rate: {summary['compile_path_pass_rate']:.3f}",
        f"- target-effect pass rate: {summary['effect_pass_rate']:.3f}",
        f"- grasp-lift pass rate: {summary['grasp_lift_pass_rate']:.3f}"
        if summary["grasp_lift_pass_rate"] is not None
        else "- grasp-lift pass rate: n/a",
        "- diagnostic policy+tracking+effect pass rate: "
        f"{summary['diagnostic_policy_tracking_effect_pass_rate']:.3f}",
        f"- mean xy displacement: {summary['mean_target_xy_displacement_m']:.4f} m"
        if summary["mean_target_xy_displacement_m"] is not None
        else "- mean xy displacement: n/a",
        "",
        "| seed | pass | goals | steps | yaw deg | block xy | xy disp | projected | cosine | overlay | video |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    def fmt(value: object, default: float = 0.0, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return f"{float(default):.{digits}f}"

    for row in rows:
        effect = row.get("effect", {})
        params = row.get("seed_params", {})
        block_xy = params.get("block_xy", [0.0, 0.0])
        artifacts = row.get("artifacts", {})
        lines.append(
            f"| {row.get('seed')} | {row.get('passed')} | "
            f"{row.get('final_successes')}/{row.get('num_goals')} | {row.get('num_steps')} | "
            f"{fmt(params.get('start_yaw_delta_deg'), digits=1)} | "
            f"[{fmt(block_xy[0])}, {fmt(block_xy[1])}] | "
            f"{fmt(effect.get('target_xy_displacement_m'))} | "
            f"{fmt(effect.get('projected_displacement_m'))} | "
            f"{fmt(effect.get('direction_cosine'))} | "
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
        default=ROOT / "artifacts/image_grounded/brush_seed20_mainline",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--rollout-max-tracking-error-m", type=float, default=0.012)
    parser.add_argument("--capture-video-every", type=int, default=0)
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.08)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    parser.add_argument("--initial-grasp-hold-steps", type=int, default=0)
    parser.add_argument("--grasp-observation-offset-m", type=float, default=0.0)
    parser.add_argument("--approach-offset-m", type=float, default=0.12)
    parser.add_argument(
        "--target-mode",
        choices=["directional_displacement", "red_zone"],
        default="directional_displacement",
    )
    parser.add_argument("--target-zone-asset", default=RED_TARGET_ZONE_ASSET)
    parser.add_argument(
        "--target-zone-center-xy",
        nargs=2,
        type=float,
        default=DEFAULT_TARGET_ZONE_CENTER_XY,
    )
    parser.add_argument(
        "--target-zone-size-xy",
        nargs=2,
        type=float,
        default=DEFAULT_TARGET_ZONE_SIZE_XY,
    )
    parser.add_argument("--target-zone-z", type=float, default=DEFAULT_TARGET_ZONE_Z)
    parser.add_argument("--block-asset", default=None)
    parser.add_argument("--block-friction", type=float, default=None)
    parser.add_argument("--block-position-mode", choices=["random", "line_x"], default="random")
    parser.add_argument("--line-x-min", type=float, default=BASE_BLOCK_XY[0] - 0.08)
    parser.add_argument("--line-x-max", type=float, default=BASE_BLOCK_XY[0] + 0.08)
    parser.add_argument("--line-y", type=float, default=BASE_BLOCK_XY[1])
    parser.add_argument("--sweep-direction", choices=sorted(SWEEP_DIRECTIONS), default="forward")
    parser.add_argument("--tool-id", default="red_brush")
    parser.add_argument("--tool-refs-dir", type=Path, default=TOOL_REFS_DIR)
    parser.add_argument("--tool-mesh", default=TOOL_MESH)
    parser.add_argument("--tool-start-pose-source", default=TOOL_START_POSE_SOURCE)
    parser.add_argument("--fixed-start-pose-xyzw", nargs=7, type=float, default=FIXED_GRASP_READY_START_POSE)
    parser.add_argument(
        "--start-orientation-mode",
        choices=["fixed", "perpendicular_motion"],
        default="fixed",
    )
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
        task_family="brush_sweep",
        split=args.sweep_direction,
    )
    _write_json(
        args.output_root / "run_config.json",
        {
            "benchmark_config": str(args.benchmark_config.resolve())
            if args.benchmark_config is not None
            else None,
            "task_source": "runner_generated_image_grounded_brush_task",
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
            "skip_rollout": args.skip_rollout,
            "initial_grasp_lift_m": args.initial_grasp_lift_m,
            "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
            "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
            "grasp_observation_offset_m": args.grasp_observation_offset_m,
            "approach_offset_m": args.approach_offset_m,
            "target_mode": args.target_mode,
            "target_zone_asset": args.target_zone_asset,
            "target_zone_center_xy": args.target_zone_center_xy,
            "target_zone_size_xy": args.target_zone_size_xy,
            "target_zone_z": args.target_zone_z,
            "block_asset": args.block_asset,
            "block_friction": args.block_friction,
            "block_position_mode": args.block_position_mode,
            "line_x_min": args.line_x_min,
            "line_x_max": args.line_x_max,
            "line_y": args.line_y,
            "sweep_direction": args.sweep_direction,
            "start_orientation_mode": args.start_orientation_mode,
            "mainline_params": {
                "depth_sample_policy": "nearest",
                "end_point_policy": "required_vlm_end_point_for_tool_contact_path",
                "tool_motion_axis_ref": "tool.motion_axis",
                "tool_motion_axis_target_ref": "scene.direction",
                "tool_motion_axis_relation": "parallel",
                "direction_ref": "scene.direction",
                "brush_back_to_arm_yaw_offset_deg": BRUSH_BACK_TO_ARM_YAW_OFFSET_DEG,
                "approach_offset_m": args.approach_offset_m,
                "max_translation_step_m": 0.035,
                "max_rotation_step_rad": 0.30,
                "initial_grasp_lift_m": args.initial_grasp_lift_m,
                "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
                "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
                "rollout_max_tracking_error_m": 0.012,
                "rollout_max_rotation_error_rad": 0.28,
                "force_no_reset": True,
            },
        },
    )
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(range(args.seed_start, args.seed_start + args.trials), start=1):
        print(f"[{index}/{args.trials}] running seed {seed:03d}", flush=True)
        try:
            row = _run_trial(seed=seed, index=index, args=args)
        except Exception as exc:  # keep benchmark resumable after a bad seed
            task_id = f"{args.tool_id}_sweep_block_seed_{seed:03d}"
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
    print(json.dumps({"summary": str(args.output_root / "summary.json"), "pass_rate": final_summary["pass_rate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
