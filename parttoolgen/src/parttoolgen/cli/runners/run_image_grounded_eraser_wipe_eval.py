#!/usr/bin/env python3
"""Run the image-grounded flat-eraser tabletop-wipe smoke."""

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
    ERASER_WIPE_SUCCESS_POLICY,
    check_task_success,
    predicate_passed,
)
from parttoolgen.evaluation.failure_attribution import (
    attribute_eraser_wipe_failure,
    failure_category_counts,
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
from parttoolgen.pipeline.trial_tool_refs import (
    add_tool_ref_mode_args,
    resolve_trial_tool_refs,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
DEFAULT_TOOL_REFS_DIR = ROOT / "configs/tool_refs/flat_eraser"
TABLE_ASSET = "urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf"
DEFAULT_TOOL_ID = "flat_eraser"
DEFAULT_TOOL_MESH = "simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj"
DEFAULT_TOOL_START_POSE_SOURCE = "simtoolreal/dextoolbench/trajectories/eraser/flat_eraser/wipe_c.json"
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
TABLETOP_SURFACE_NORMAL_WORLD = [0.0, 0.0, 1.0]
PATCH_ASSET = (
    "urdf/parttoolgen_benchmark/eraser_wipe_tabletop_hdu_board/"
    "tabletop_hdu_board_018x011x030.urdf"
)
TABLETOP_BOARD_BOTTOM_Z = TABLE_Z + 0.150
TABLETOP_BOARD_THICKNESS_M = 0.030
PATCH_CENTER_WORLD = [0.14, -0.04, TABLETOP_BOARD_BOTTOM_Z + 0.5 * TABLETOP_BOARD_THICKNESS_M]
PATCH_SIZE_XYZ = [0.18, 0.11, TABLETOP_BOARD_THICKNESS_M]
ERASER_WIPE_TOOL_GROUNDING_CONTRACT = {
    "schema_version": "toolframe_tool_grounding_contract_v1",
    "task_type": "eraser_wipe",
    "tool_id": "flat_eraser",
    "label_source": "manual_review_2026_05_27",
    "manual_review_status": "approved",
    "review_file": "artifacts/tool_use_benchmark/tool_label_review/eraser_wipe__flat_eraser.json",
    "allowed_contact_point_part_markers": ["4"],
    "allowed_contact_point_part_refs": ["tool.parts.part_292"],
    "allowed_contact_point_face_markers": ["F3", "F4"],
    "allowed_contact_point_refs": [
        "candidate.points.part_292_y_negative_contact",
        "candidate.points.part_292_y_positive_contact",
    ],
    "allowed_contact_point_surface_refs": [
        "candidate.surfaces.part_292_y_negative_contact",
        "candidate.surfaces.part_292_y_positive_contact",
    ],
    "allowed_motion_axis_face_markers": ["F1", "F2", "F5", "F6"],
    "allowed_motion_axis_surface_refs": [
        "candidate.surfaces.part_292_x_negative_endpoint",
        "candidate.surfaces.part_292_x_positive_endpoint",
        "candidate.surfaces.part_292_z_negative_contact",
        "candidate.surfaces.part_292_z_positive_contact",
    ],
    "candidate_filter_policy": {
        "exclude_internal_tool_surfaces_from_vlm_selection": True,
        "applies_to_all_tools": True,
    },
    "notes": (
        "Human-reviewed eraser contract: contact point part marker 4; wiping contact "
        "uses the non-internal faces among F3/F4; all other non-internal faces may "
        "serve as motion-axis refs."
    ),
}
SEED_VARIATION_POLICY = {
    "sampler": "deterministic_tabletop_wipe_stratified_pose_v5_nonoverlap",
    "target_grid_shape": [5, 4],
    "target_center_x_m": [-0.215, 0.225],
    "target_center_y_m": [-0.140, 0.120],
    "target_xy_jitter_m": [-0.010, 0.010],
    "target_yaw_deg": [-45.0, 45.0],
    "tool_safe_x_m": [-0.280, 0.280],
    "tool_safe_y_m": [-0.185, 0.185],
    "tool_target_distance_m": [0.190, 0.230],
    "tool_angle_degrees": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    "tool_angle_jitter_deg": [-12.0, 12.0],
    "tool_yaw_jitter_deg": [-35.0, 35.0],
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


def _normalize_quat(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in q))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(item) / norm for item in q]


def _lerp(lo: float, hi: float, alpha: float) -> float:
    return float(lo) * (1.0 - float(alpha)) + float(hi) * float(alpha)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _table_contact_tool_pose(pose_xyzw: list[float], *, tool_mesh: str = DEFAULT_TOOL_MESH) -> list[float]:
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


def _sample_seed_params(seed: int, *, args: argparse.Namespace | None = None) -> dict[str, Any]:
    rng = random.Random(int(seed))
    policy = copy.deepcopy(SEED_VARIATION_POLICY)
    if args is not None:
        policy["target_center_x_m"] = [float(item) for item in args.target_center_x_range]
        policy["target_center_y_m"] = [float(item) for item in args.target_center_y_range]
        policy["target_xy_jitter_m"] = [float(item) for item in args.target_xy_jitter_range]
    cols, rows = [int(item) for item in policy["target_grid_shape"]]
    col = int(seed) % cols
    row = (int(seed) // cols) % rows
    x_alpha = (float(col) + 0.5) / float(cols)
    y_alpha = (float(row) + 0.5) / float(rows)
    target_x = _lerp(*policy["target_center_x_m"], x_alpha)
    target_y = _lerp(*policy["target_center_y_m"], y_alpha)
    target_x += rng.uniform(*policy["target_xy_jitter_m"])
    target_y += rng.uniform(*policy["target_xy_jitter_m"])
    target_x = _clamp(target_x, *policy["target_center_x_m"])
    target_y = _clamp(target_y, *policy["target_center_y_m"])
    yaw_span = policy["target_yaw_deg"]
    target_yaw_deg = _lerp(
        float(yaw_span[0]),
        float(yaw_span[1]),
        ((int(seed) * 7) % 20 + 0.5) / 20.0,
    )
    target_yaw_deg += rng.uniform(-3.0, 3.0)
    target_yaw_deg = _clamp(target_yaw_deg, *yaw_span)

    tool_mesh = str(args.tool_mesh) if args is not None else DEFAULT_TOOL_MESH
    nominal_start_pose = (
        [float(item) for item in args.fixed_start_pose_xyzw]
        if args is not None and getattr(args, "fixed_start_pose_xyzw", None) is not None
        else [float(item) for item in FRONT_OF_ARM_FLAT_ERASER_START_POSE]
    )
    tool_pose = _table_contact_tool_pose(
        nominal_start_pose,
        tool_mesh=tool_mesh,
    )
    tool_x = float(tool_pose[0])
    tool_y = float(tool_pose[1])
    tool_dx = tool_x - float(nominal_start_pose[0])
    tool_dy = tool_y - float(nominal_start_pose[1])
    rel_x = tool_x - target_x
    rel_y = tool_y - target_y
    tool_distance = math.hypot(rel_x, rel_y)
    tool_angle_deg = math.degrees(math.atan2(rel_y, rel_x))
    tool_yaw_deg = 0.0

    target_quat = _quat_from_z_yaw(math.radians(target_yaw_deg))
    target_pose = [target_x, target_y, PATCH_CENTER_WORLD[2], *target_quat]
    return {
        "seed": int(seed),
        "target_grid_cell": [col, row],
        "tool_target_angle_deg": tool_angle_deg,
        "tool_target_distance_m": tool_distance,
        "tool_start_pose": tool_pose,
        "tool_nominal_start_pose_xyzw": nominal_start_pose,
        "tool_xy_offset_from_nominal_m": [tool_dx, tool_dy],
        "tool_xy_relative_to_target_m": [tool_x - target_x, tool_y - target_y],
        "tool_yaw_jitter_deg": tool_yaw_deg,
        "target_center_world": [target_x, target_y, PATCH_CENTER_WORLD[2]],
        "target_yaw_deg": target_yaw_deg,
        "target_pose_xyzw": target_pose,
        "target_size_xyz": list(PATCH_SIZE_XYZ),
        "sampler_policy": policy,
    }


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
        "target_collision_role",
        "min_tool_table_clearance_m",
        "parttoolgen_benchmark_extra_actors",
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
            "method_contract": (
                "method-facing task keeps public benchmark actors visible in "
                "the VLM scene evidence; hidden evaluator-only geometry remains omitted"
            ),
            "private_scene_geometry_omitted": False,
            "public_tabletop_surface_normal_world": copy.deepcopy(
                task.get("metadata", {}).get("public_tabletop_surface_normal_world", list(TABLETOP_SURFACE_NORMAL_WORLD))
            ),
            "public_support_surface_normal_world": copy.deepcopy(
                task.get("metadata", {}).get("public_support_surface_normal_world", list(TABLETOP_SURFACE_NORMAL_WORLD))
            ),
        },
    }


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _repo_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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
        "approach_offset_m",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "target_approach_axis_source",
        "initial_grasp_lift_m",
        "initial_grasp_lift_steps",
        "initial_grasp_hold_steps",
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


def _task_spec(
    *,
    task_id: str,
    seed: int,
    tool_refs_dir: Path,
    seed_params: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_pose = [float(item) for item in seed_params["tool_start_pose"]]
    target_pose = [float(item) for item in seed_params["target_pose_xyzw"]]
    target_center = [float(item) for item in seed_params["target_center_world"]]
    return {
        "schema_version": "parttoolgen_tool_use_task_v0",
        "task_id": task_id,
        "family": "image_grounded_wipe",
        "split": "dev",
        "instruction": "Use this eraser to erase the whole HDU pattern on the table.",
        "tool": {
            "tool_id": args.tool_id,
            "tool_prior_dir": _repo_display(tool_refs_dir),
        },
        "execution": {
            "simtoolreal_category": "eraser",
            "simtoolreal_table_asset": TABLE_ASSET,
            "simtoolreal_tool_mesh": args.tool_mesh,
            "simtoolreal_start_pose_source": args.tool_start_pose_source,
            "start_pose_policy": f"fixed_front_of_arm_{args.tool_id}_start_pose",
            "simtoolreal_start_pose": start_pose,
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, TABLE_Z, 0.0, 0.0, 0.0, 1.0],
                "table_asset": TABLE_ASSET,
                "table_z": TABLE_Z,
                "object_name": args.tool_id,
                "object_category": "eraser",
                "task_name": "tabletop_wipe",
                "target_collision_role": "fixed_physical_hdu_board_on_tabletop",
                "min_tool_table_clearance_m": 0.0003,
                "parttoolgen_benchmark_extra_actors": [
                    {
                        "name": "tabletop_hdu_board",
                        "asset_file": PATCH_ASSET,
                        "pose_xyzw": target_pose,
                        "disable_gravity": True,
                        "fix_base_link": True,
                        "friction": 0.8,
                        "semantic_role": "fixed_whiteboard_with_hdu_pattern_to_erase",
                        "collision_policy": "physical_board_collision; eraser contact should be with board top surface",
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
                f"public {args.tool_id} tool refs from P3SAM/VLM part-face grounding",
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
            "benchmark_role": "image_grounded_eraser_tabletop_wipe_smoke",
            "target_kind": "tabletop_visible_hdu_board",
            "tabletop_hdu_board_center_world": target_center,
            "tabletop_hdu_board_yaw_deg": float(seed_params["target_yaw_deg"]),
            "tabletop_hdu_board_size_xyz": list(PATCH_SIZE_XYZ),
            "tabletop_hdu_board_asset_file": PATCH_ASSET,
            "tabletop_hdu_board_collision_policy": (
                "fixed physical board with collision; eraser contact should be with board top surface"
            ),
            "start_pose_xyzw": start_pose,
            "target_grid_cell": list(seed_params["target_grid_cell"]),
            "tool_target_angle_deg": float(seed_params["tool_target_angle_deg"]),
            "tool_target_distance_m": float(seed_params["tool_target_distance_m"]),
            "tool_xy_offset_from_nominal_m": [
                float(item) for item in seed_params["tool_xy_offset_from_nominal_m"]
            ],
            "tool_xy_relative_to_target_m": [
                float(item) for item in seed_params["tool_xy_relative_to_target_m"]
            ],
            "tool_yaw_jitter_deg": float(seed_params["tool_yaw_jitter_deg"]),
            "seed_variation_policy": copy.deepcopy(SEED_VARIATION_POLICY),
            "tool_start_policy": "fixed_front_of_arm_start_pose_shared_by_vlm_render_and_rollout",
            "tool_id": args.tool_id,
            "tool_mesh": args.tool_mesh,
            "public_tabletop_surface_normal_world": list(TABLETOP_SURFACE_NORMAL_WORLD),
            "tool_grounding_contract": copy.deepcopy(ERASER_WIPE_TOOL_GROUNDING_CONTRACT),
            "notes": (
                f"Tabletop {args.tool_id} HDU wipe task on a raised white HDU board in the current ToolFrame wood-table scene. "
                "The tool motion axis is aligned to the public wipe direction, "
                "while the wipe path is grounded from VLM marks lifted through RGB-D."
            ),
        },
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


def _load_optional_json_value(path_value: Any) -> Any | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return _load_json(path)


def _tracking_samples_path_from_artifacts(artifacts: dict[str, Any]) -> Path | None:
    tracking_samples = artifacts.get("tracking_samples")
    if tracking_samples:
        return Path(str(tracking_samples))
    execution_report = artifacts.get("execution_report")
    if execution_report:
        return Path(str(execution_report)).parent / "tracking_samples.json"
    rollout_summary = artifacts.get("rollout_summary")
    if rollout_summary:
        return Path(str(rollout_summary)).parent / "tracking_samples.json"
    return None


def _tool_grounding_contract_from_task(task: dict[str, Any] | None) -> dict[str, Any]:
    metadata = task.get("metadata") if isinstance(task, dict) else None
    contract = metadata.get("tool_grounding_contract") if isinstance(metadata, dict) else None
    default_contract = copy.deepcopy(ERASER_WIPE_TOOL_GROUNDING_CONTRACT)
    if not isinstance(contract, dict):
        return default_contract
    if str(contract.get("manual_review_status", "")).lower() == "approved":
        return dict(contract)
    if str(default_contract.get("manual_review_status", "")).lower() == "approved":
        return default_contract
    return dict(contract)


def _tool_mesh_from_task(task: dict[str, Any] | None) -> str:
    if isinstance(task, dict):
        mesh = task.get("execution", {}).get("simtoolreal_tool_mesh")
        if isinstance(mesh, str) and mesh:
            return mesh
    return DEFAULT_TOOL_MESH


def _rollout_max_steps_from_result(result: dict[str, Any], artifacts: dict[str, Any]) -> int | None:
    value = result.get("rollout_max_steps")
    if isinstance(value, int):
        return value
    task_path = artifacts.get("task")
    if task_path:
        rollout_log = Path(str(task_path)).parent / "logs" / "rollout.json"
        if rollout_log.exists():
            log = _load_optional_json(rollout_log)
            cmd = log.get("cmd") if isinstance(log, dict) else None
            if isinstance(cmd, list):
                for index, item in enumerate(cmd[:-1]):
                    if item == "--max-steps":
                        try:
                            return int(cmd[index + 1])
                        except (TypeError, ValueError):
                            break
        run_config = Path(str(task_path)).parent.parent / "run_config.json"
        if run_config.exists():
            data = _load_optional_json(run_config)
            if isinstance(data, dict) and isinstance(data.get("rollout_max_steps"), int):
                return int(data["rollout_max_steps"])
    return None


def _selected_tool_refs_from_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    compile_summary = artifacts.get("compile_summary")
    if not compile_summary:
        return {}
    selected_program = _load_optional_json(Path(str(compile_summary)).parent / "selected_constraint_program.json")
    if not isinstance(selected_program, dict):
        return {}
    tool_refs = selected_program.get("tool_refs")
    return dict(tool_refs) if isinstance(tool_refs, dict) else {}


def _apply_selected_tool_ref_fields(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return result
    selected_tool_refs = _selected_tool_refs_from_artifacts(artifacts)
    if not selected_tool_refs:
        return result
    refreshed = dict(result)
    refreshed["selected_contact_point_face"] = selected_tool_refs.get("contact_point_face_marker")
    refreshed["selected_contact_point_ref"] = (
        selected_tool_refs.get("contact_point_ref")
        or selected_tool_refs.get("contact_point_source_ref")
    )
    refreshed["selected_motion_axis_face"] = selected_tool_refs.get("motion_axis_face_marker")
    active_selection = selected_tool_refs.get("active_contact_point_candidate_selection")
    if isinstance(active_selection, dict):
        refreshed["tool_ref_candidate_selection"] = active_selection
    return refreshed


def _refresh_success_fields(result: dict[str, Any], *, skip_rollout: bool) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    rollout_summary = _load_optional_json(artifacts.get("rollout_summary")) if isinstance(artifacts, dict) else None
    execution_report = _load_optional_json(artifacts.get("execution_report")) if isinstance(artifacts, dict) else None
    compile_summary = _load_optional_json(artifacts.get("compile_summary")) if isinstance(artifacts, dict) else None
    task = _load_optional_json(artifacts.get("task")) if isinstance(artifacts, dict) else None
    scene_keypoints = _load_optional_json(artifacts.get("scene_keypoints")) if isinstance(artifacts, dict) else None
    lifted_refs = _load_optional_json(artifacts.get("lifted_refs")) if isinstance(artifacts, dict) else None
    tool_pose_subgoals = (
        _load_optional_json(Path(str(artifacts.get("compile_summary"))).parent / "path_solved_tool_pose_subgoals.json")
        if isinstance(artifacts, dict) and artifacts.get("compile_summary")
        else None
    )
    tracking_samples = (
        _load_optional_json_value(_tracking_samples_path_from_artifacts(artifacts))
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(tracking_samples, list):
        tracking_samples = None
    if compile_summary is None:
        compile_summary = {"path_solver_passed": bool(result.get("compile_path_solver_passed"))}
    if rollout_summary is None and result.get("semantic_execution_passed") is not None:
        rollout_summary = {"semantic_execution_passed": bool(result.get("semantic_execution_passed"))}
    success_check = check_task_success(
        "eraser_wipe",
        compile_summary=compile_summary,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        lifted_refs=lifted_refs,
        task=task,
        tracking_samples=tracking_samples,
        tool_mesh_path=_tool_mesh_from_task(task),
        skip_rollout=skip_rollout,
    )
    refreshed = dict(result)
    refreshed["task_success_check"] = success_check
    refreshed["wipe_contact_coverage_passed"] = predicate_passed(success_check, "contact_coverage")
    refreshed["wipe_semantic_passed"] = predicate_passed(success_check, "semantic_contact_path")
    refreshed["failure_attribution"] = attribute_eraser_wipe_failure(
        task=task,
        scene_keypoints=scene_keypoints,
        lifted_refs=lifted_refs,
        compile_summary=compile_summary,
        tool_pose_subgoals=tool_pose_subgoals,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        success_check=success_check,
        tracking_samples=tracking_samples,
        selected_tool_refs=_selected_tool_refs_from_artifacts(artifacts)
        if isinstance(artifacts, dict)
        else None,
        tool_grounding_contract=_tool_grounding_contract_from_task(task),
        max_steps=_rollout_max_steps_from_result(result, artifacts)
        if isinstance(artifacts, dict)
        else None,
    )
    refreshed["failure_category"] = refreshed["failure_attribution"]["primary_category"]
    refreshed["failure_layer"] = refreshed["failure_attribution"]["primary_layer"]
    refreshed["success_policy"] = success_check["policy"]
    refreshed["passed"] = bool(success_check.get("passed"))
    return _apply_selected_tool_ref_fields(refreshed)


def _run_trial(*, seed: int, index: int, args: argparse.Namespace) -> dict[str, Any]:
    task_id = f"{args.tool_id}_tabletop_wipe_seed_{seed:03d}"
    trial_dir = args.output_root / task_id
    if args.resume and (trial_dir / "trial_result.json").exists():
        result = _refresh_success_fields(
            _load_json(trial_dir / "trial_result.json"),
            skip_rollout=bool(args.skip_rollout),
        )
        visual_evidence = package_trial_visual_evidence_from_artifacts(
            trial_dir=trial_dir,
            artifacts=result.get("artifacts", {}) if isinstance(result.get("artifacts"), dict) else {},
            tool_refs_dir=args.tool_refs_dir,
            copy_files=False,
        )
        result.setdefault("artifacts", {})["visual_evidence"] = visual_evidence["evidence_dir"]
        result["artifacts"]["visual_evidence_manifest"] = visual_evidence["manifest"]
        _write_json(trial_dir / "trial_result.json", result)
        return result
    if trial_dir.exists() and not args.resume:
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    seed_params = _sample_seed_params(seed, args=args)
    task = _task_spec(
        task_id=task_id,
        seed=seed,
        tool_refs_dir=args.tool_refs_dir,
        seed_params=seed_params,
        args=args,
    )
    task_path = trial_dir / "task.json"
    method_task_path = trial_dir / "method_task.json"
    _write_json(task_path, task)
    _write_json(method_task_path, _method_facing_task(task))
    py = _python()
    trial_tool_refs_dir = resolve_trial_tool_refs(
        repo_root=ROOT,
        python=py,
        args=args,
        trial_dir=trial_dir,
        static_tool_refs_dir=args.tool_refs_dir,
        tool_id=args.tool_id,
        task_id=task_id,
        task_instruction=str(task.get("instruction") or ""),
        log_path=trial_dir / "logs/tool_refs.json",
        env=_env(),
    )

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
            "--grounding-camera",
            "topdown",
            "--presentation-camera",
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
        retries=3,
        retry_delay_s=8.0,
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
            str(grounding_dir / "grounding_overlay.png"),
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
            "--end-surface-policy",
            "anchor_contact_plane",
            "--action-direction-z-policy",
            "anchor_height",
            "--tool-motion-axis-ref",
            "tool.motion_axis",
            "--tool-motion-axis-target-ref",
            "scene.direction",
            "--tool-motion-axis-relation",
            "parallel",
            "--target-approach-axis-source",
            args.target_approach_axis_source,
            "--require-end-point",
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
            str(trial_tool_refs_dir / "axis_binding.json"),
            "--axis-mesh-report",
            str(trial_tool_refs_dir / "axis_mesh_overlay_report.json"),
            "--contact-point-selection",
            str(trial_tool_refs_dir / "contact_point_selection.json"),
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
            str(args.max_translation_step_m),
            "--max-rotation-step-rad",
            str(args.max_rotation_step_rad),
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
    tracking_samples: list[dict[str, Any]] | None = None
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
            "--max-tracking-error-m",
            str(args.rollout_max_tracking_error_m),
            "--force-no-reset",
        ]
        if args.capture_video_every > 0 and (index - 1) % args.capture_video_every == 0:
            rollout_cmd.extend(
                [
                    "--capture-video",
                    "--capture-topdown-video",
                    "--capture-subgoal-video",
                    "--capture-subgoal-topdown-video",
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
        loaded_tracking_samples = _load_optional_json_value(rollout_dir / "tracking_samples.json")
        tracking_samples = loaded_tracking_samples if isinstance(loaded_tracking_samples, list) else None

    scene_keypoints = _load_json(grounding_dir / "scene_keypoints.json")
    lifted_refs = _load_json(lifted_dir / "lifted_refs.json")
    compile_summary = _load_json(compile_dir / "compile_summary.json")
    tool_pose_subgoals = _load_json(compile_dir / "path_solved_tool_pose_subgoals.json")
    enriched = _load_json(enriched_program)
    selected_program = _load_optional_json(compile_dir / "selected_constraint_program.json") or {}
    selected_tool_refs = selected_program.get("tool_refs") if isinstance(selected_program, dict) else None
    if not isinstance(selected_tool_refs, dict):
        selected_tool_refs = enriched.get("tool_refs", {}) if isinstance(enriched.get("tool_refs"), dict) else {}
    tracking_passed = bool(rollout_summary.get("tracking_execution_passed")) if rollout_summary else None
    policy_goal_passed = bool(rollout_summary.get("policy_goal_passed")) if rollout_summary else None
    semantic_execution_passed = (
        bool(rollout_summary.get("semantic_execution_passed")) if rollout_summary else None
    )
    success_check = check_task_success(
        "eraser_wipe",
        compile_summary=compile_summary,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        lifted_refs=lifted_refs,
        task=task,
        tracking_samples=tracking_samples,
        tool_mesh_path=args.tool_mesh,
        skip_rollout=bool(args.skip_rollout),
    )
    failure_attribution = attribute_eraser_wipe_failure(
        task=task,
        scene_keypoints=scene_keypoints,
        lifted_refs=lifted_refs,
        compile_summary=compile_summary,
        tool_pose_subgoals=tool_pose_subgoals,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        success_check=success_check,
        tracking_samples=tracking_samples,
        selected_tool_refs=selected_tool_refs,
        tool_grounding_contract=_tool_grounding_contract_from_task(task),
        max_steps=int(args.rollout_max_steps),
    )
    grasp_lift_passed = (
        _grasp_lift_passed(execution_report, compile_summary)
        if execution_report
        else None
    )
    visual_evidence = package_trial_visual_evidence(
        trial_dir=trial_dir,
        render_dir=render_dir,
        grounding_dir=grounding_dir,
        tool_refs_dir=trial_tool_refs_dir,
        compile_dir=compile_dir,
        rollout_dir=trial_dir / "rollout",
        copy_files=False,
    )
    semantic_path_passed = predicate_passed(success_check, "semantic_contact_path")
    diagnostic_surface_contact_coverage_passed = predicate_passed(success_check, "contact_coverage")
    passed = bool(success_check.get("passed"))
    result = {
        "index": index,
        "seed": seed,
        "task_id": task_id,
        "seed_params": seed_params,
        "passed": passed,
        "skip_rollout": bool(args.skip_rollout),
        "target_object": scene_keypoints.get("target_object"),
        "lifted_start_point_world": lifted_refs.get("start_point", {}).get("position_world"),
        "direction_world": lifted_refs.get("direction"),
        "direction_source": lifted_refs.get("direction_source"),
        "action_distance_m": lifted_refs.get("action_distance_m"),
        "target_anchor_policy": lifted_refs.get("target_anchor_policy"),
        "selected_contact_point_face": selected_tool_refs.get("contact_point_face_marker"),
        "selected_contact_point_ref": (
            selected_tool_refs.get("contact_point_ref")
            or selected_tool_refs.get("contact_point_source_ref")
        ),
        "selected_motion_axis_face": selected_tool_refs.get("motion_axis_face_marker"),
        "tool_ref_candidate_selection": selected_tool_refs.get("active_contact_point_candidate_selection"),
        "tool_grounding_contract": copy.deepcopy(ERASER_WIPE_TOOL_GROUNDING_CONTRACT),
        "compile_path_solver_passed": bool(compile_summary.get("path_solver_passed")),
        "policy_goal_passed": policy_goal_passed,
        "tracking_execution_passed": tracking_passed,
        "semantic_execution_passed": semantic_execution_passed,
        "diagnostic_surface_contact_coverage_passed": diagnostic_surface_contact_coverage_passed,
        "wipe_contact_coverage_passed": diagnostic_surface_contact_coverage_passed,
        "wipe_semantic_passed": semantic_path_passed,
        "task_success_check": success_check,
        "failure_attribution": failure_attribution,
        "failure_category": failure_attribution["primary_category"],
        "failure_layer": failure_attribution["primary_layer"],
        "success_policy": success_check["policy"],
        "grasp_lift_passed": grasp_lift_passed,
        "rollout_max_steps": int(args.rollout_max_steps),
        "final_successes": int(rollout_summary.get("final_successes", 0)) if rollout_summary else None,
        "num_goals": int(rollout_summary.get("num_goals", 0)) if rollout_summary else None,
        "num_steps": int(rollout_summary.get("num_steps", 0)) if rollout_summary else None,
        "artifacts": {
            "task": str(task_path),
            "method_task": str(method_task_path),
            "scene_evidence": str(render_dir / "scene_evidence.json"),
            "grounding_overlay": str(grounding_dir / "grounding_overlay.png"),
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
            "tracking_samples": str(trial_dir / "rollout/tracking_samples.json")
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
        "semantic_failed_stage": execution_report.get("semantic_failed_stage") if execution_report else None,
        "semantic_failure_reason": (
            execution_report.get("stage_reports", [{}])[-1].get("semantic_failure_reason")
            if isinstance(execution_report.get("stage_reports"), list)
            and execution_report.get("stage_reports")
            else None
        )
        if execution_report
        else None,
    }
    _write_json(trial_dir / "trial_result.json", result)
    return result


def _summary(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    passed = [row for row in rows if row.get("passed")]
    contact_rows = [
        row
        for row in rows
        if row.get("diagnostic_surface_contact_coverage_passed", row.get("wipe_contact_coverage_passed"))
        is not None
    ]
    contact_passed = [
        row
        for row in contact_rows
        if row.get("diagnostic_surface_contact_coverage_passed", row.get("wipe_contact_coverage_passed"))
    ]
    semantic_rows = [row for row in rows if row.get("wipe_semantic_passed") is not None]
    semantic_passed = [row for row in semantic_rows if row.get("wipe_semantic_passed")]
    lift_rows = [row for row in rows if row.get("grasp_lift_passed") is not None]
    summary = {
        "schema_version": "toolframe_image_grounded_eraser_wipe_eval_v1",
        "success_policy": ERASER_WIPE_SUCCESS_POLICY["name"],
        "success_requires_visual_patch_disappearance": False,
        "success_requires_tracking_execution_passed": False,
        "output_root": str(output_root),
        "trial_count": len(rows),
        "passed_count": len(passed),
        "pass_rate": (len(passed) / len(rows)) if rows else 0.0,
        "diagnostic_surface_contact_coverage_pass_rate": (
            len(contact_passed) / len(contact_rows)
        )
        if contact_rows
        else None,
        "wipe_contact_coverage_pass_rate": (
            len(contact_passed) / len(contact_rows)
        )
        if contact_rows
        else None,
        "semantic_execution_pass_rate": (
            len(semantic_passed) / len(semantic_rows)
        )
        if semantic_rows
        else None,
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
        "grasp_lift_pass_rate": (
            sum(1 for row in lift_rows if row.get("grasp_lift_passed")) / len(lift_rows)
        )
        if lift_rows
        else None,
        "failure_category_counts": failure_category_counts(rows),
        "trials": rows,
    }
    _write_json(output_root / "summary.json", summary)
    lines = [
        "# Image-Grounded Eraser Wipe Eval",
        "",
        f"- trials: {summary['trial_count']}",
        f"- success policy: {ERASER_WIPE_SUCCESS_POLICY['name']}",
        f"- pass rate: {summary['pass_rate']:.3f}",
        "- diagnostic surface-contact coverage pass rate: "
        f"{summary['diagnostic_surface_contact_coverage_pass_rate']:.3f}"
        if summary["diagnostic_surface_contact_coverage_pass_rate"] is not None
        else "- diagnostic surface-contact coverage pass rate: n/a",
        f"- semantic execution pass rate: {summary['semantic_execution_pass_rate']:.3f}"
        if summary["semantic_execution_pass_rate"] is not None
        else "- semantic execution pass rate: n/a",
        f"- policy goal pass rate: {summary['policy_goal_pass_rate']:.3f}",
        f"- tracking pass rate: {summary['tracking_pass_rate']:.3f}",
        f"- compile path pass rate: {summary['compile_path_pass_rate']:.3f}",
        f"- grasp-lift pass rate: {summary['grasp_lift_pass_rate']:.3f}"
        if summary["grasp_lift_pass_rate"] is not None
        else "- grasp-lift pass rate: n/a",
        f"- failure categories: {summary['failure_category_counts']}",
        "",
        "| seed | pass | failure category | diagnostic contact coverage | semantic point | target xy | target yaw | tool yaw | face | action dist | goals | steps | overlay | video |",
        "|---:|---|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        artifacts = row.get("artifacts", {})
        action_distance = row.get("action_distance_m")
        action_distance_text = (
            f"{float(action_distance):.3f}"
            if isinstance(action_distance, (int, float))
            else "n/a"
        )
        params = row.get("seed_params", {}) if isinstance(row.get("seed_params"), dict) else {}
        target_center = params.get("target_center_world")
        if isinstance(target_center, list) and len(target_center) >= 2:
            target_xy_text = f"{float(target_center[0]):.3f},{float(target_center[1]):.3f}"
        else:
            target_xy_text = "n/a"
        target_yaw = params.get("target_yaw_deg")
        tool_yaw = params.get("tool_yaw_jitter_deg")
        target_yaw_text = (
            f"{float(target_yaw):.1f}" if isinstance(target_yaw, (int, float)) else "n/a"
        )
        tool_yaw_text = (
            f"{float(tool_yaw):.1f}" if isinstance(tool_yaw, (int, float)) else "n/a"
        )
        lines.append(
            f"| {row.get('seed')} | {row.get('passed')} | "
            f"{row.get('failure_category')} | "
            f"{row.get('diagnostic_surface_contact_coverage_passed', row.get('wipe_contact_coverage_passed'))} | "
            f"{row.get('wipe_semantic_passed')} | "
            f"{target_xy_text} | "
            f"{target_yaw_text} | "
            f"{tool_yaw_text} | "
            f"{row.get('selected_contact_point_face')} | "
            f"{action_distance_text} | "
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
        default=ROOT / "artifacts/image_grounded/eraser_wipe/tabletop_wipe_original_rgb_smoke",
    )
    parser.add_argument("--tool-refs-dir", type=Path, default=DEFAULT_TOOL_REFS_DIR)
    add_tool_ref_mode_args(parser)
    parser.add_argument("--tool-id", default=DEFAULT_TOOL_ID)
    parser.add_argument("--tool-mesh", default=DEFAULT_TOOL_MESH)
    parser.add_argument("--tool-start-pose-source", default=DEFAULT_TOOL_START_POSE_SOURCE)
    parser.add_argument(
        "--fixed-start-pose-xyzw",
        nargs=7,
        type=float,
        default=FRONT_OF_ARM_FLAT_ERASER_START_POSE,
        help="Fixed table-projected tool start pose, in xyzw quaternion order.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--rollout-max-tracking-error-m", type=float, default=0.018)
    parser.add_argument("--capture-video-every", type=int, default=1)
    parser.add_argument("--approach-offset-m", type=float, default=0.03)
    parser.add_argument("--max-translation-step-m", type=float, default=0.035)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.30)
    parser.add_argument(
        "--target-approach-axis-source",
        choices=["opposite_scene_direction"],
        default="opposite_scene_direction",
    )
    parser.add_argument(
        "--target-center-x-range",
        nargs=2,
        type=float,
        default=SEED_VARIATION_POLICY["target_center_x_m"],
        help="Inclusive table-safe x range for seeded tabletop HDU board centers.",
    )
    parser.add_argument(
        "--target-center-y-range",
        nargs=2,
        type=float,
        default=SEED_VARIATION_POLICY["target_center_y_m"],
        help="Inclusive table-safe y range for seeded tabletop HDU board centers.",
    )
    parser.add_argument(
        "--target-xy-jitter-range",
        nargs=2,
        type=float,
        default=SEED_VARIATION_POLICY["target_xy_jitter_m"],
        help="Uniform xy jitter range applied independently to seeded board centers.",
    )
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.12)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    parser.add_argument("--initial-grasp-hold-steps", type=int, default=0)
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**benchmark_defaults)
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.tool_refs_dir = args.tool_refs_dir.resolve()
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
        tool_refs_dir=args.tool_refs_dir,
        task_family="eraser_wipe",
        split="tabletop_wipe",
    )
    _write_json(
        args.output_root / "run_config.json",
        {
            "benchmark_config": str(args.benchmark_config.resolve())
            if args.benchmark_config is not None
            else None,
            "task_source": "runner_generated_image_grounded_eraser_tabletop_wipe_task",
            "tool_refs_dir": str(args.tool_refs_dir),
            "tool_selection_visual_evidence": tool_selection_evidence,
            "table_asset": TABLE_ASSET,
            "tabletop_hdu_board_asset": PATCH_ASSET,
            "nominal_tabletop_hdu_board_center_world": PATCH_CENTER_WORLD,
            "seed_variation_policy": copy.deepcopy(SEED_VARIATION_POLICY),
            "active_target_center_x_range": [float(item) for item in args.target_center_x_range],
            "active_target_center_y_range": [float(item) for item in args.target_center_y_range],
            "active_target_xy_jitter_range": [float(item) for item in args.target_xy_jitter_range],
            "tool_id": args.tool_id,
            "tool_mesh": args.tool_mesh,
            "tool_start_pose_source": args.tool_start_pose_source,
            "fixed_start_pose_xyzw": [float(item) for item in args.fixed_start_pose_xyzw],
            "seed_start": args.seed_start,
            "trials": args.trials,
            "device": args.device,
            "render_lighting_preset": "soft01",
            "rollout_max_steps": args.rollout_max_steps,
            "rollout_max_tracking_error_m": args.rollout_max_tracking_error_m,
            "capture_video_every": args.capture_video_every,
            "approach_offset_m": args.approach_offset_m,
            "max_translation_step_m": args.max_translation_step_m,
            "max_rotation_step_rad": args.max_rotation_step_rad,
            "target_approach_axis_source": args.target_approach_axis_source,
            "initial_grasp_lift_m": args.initial_grasp_lift_m,
            "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
            "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
            "response_mode": args.response_mode,
            "model": args.model,
            "skip_rollout": args.skip_rollout,
            "mainline_params": {
                "depth_sample_policy": "nearest",
                "goal_z_policy": "contact_height",
                "action_direction_z_policy": "anchor_height",
                "end_point_policy": "required_vlm_end_point_for_tool_contact_path",
                "grounding_camera": "topdown",
                "target_anchor_policy": "raw_topdown_vlm_point",
                "tool_motion_axis_ref": "tool.motion_axis",
                "tool_motion_axis_target_ref": "scene.direction",
                "tool_motion_axis_relation": "parallel",
                "target_surface_normal_world": TABLETOP_SURFACE_NORMAL_WORLD,
                "approach_offset_m": args.approach_offset_m,
                "target_approach_axis_source": args.target_approach_axis_source,
                "max_translation_step_m": args.max_translation_step_m,
                "max_rotation_step_rad": args.max_rotation_step_rad,
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
        print(f"[{index}/{args.trials}] running eraser wipe seed {seed:03d}", flush=True)
        try:
            row = _run_trial(seed=seed, index=index, args=args)
        except Exception as exc:
            task_id = f"{args.tool_id}_tabletop_wipe_seed_{seed:03d}"
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
