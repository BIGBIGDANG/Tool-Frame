#!/usr/bin/env python3
"""Run the image-grounded long-screwdriver deep-socket insertion smoke."""

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
from typing import Any, Mapping, Sequence

from parttoolgen.evaluation.compiled_task_goal import (
    check_screwdriver_insert_compiled_task_goal,
)
from parttoolgen.evaluation.task_success import (
    SCREWDRIVER_INSERT_SUCCESS_POLICY,
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
DEFAULT_TOOL_REFS_DIR = ROOT / "configs/tool_refs/long_screwdriver"
TABLE_ASSET = "urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf"
TOOL_MESH = "simtoolreal/assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.obj"
TOOL_TRAJECTORY = "simtoolreal/dextoolbench/trajectories/screwdriver/long_screwdriver/spin_vertical.json"
SOCKET_ASSET = "urdf/parttoolgen_benchmark/screwdriver_insert_socket/deep_socket_075x075x055.urdf"
TABLE_Z = 0.38
TABLE_TOP_Z = 0.53
SOCKET_TOP_LOCAL_Z = 0.055
SOCKET_TERMINAL_INSIDE_DEPTH_M = 0.035
SUCCESS_MIN_INSIDE_DEPTH_M = 0.005
SOCKET_SIZE_XYZ = [0.036, 0.036, 0.055]
SOCKET_INNER_DIAMETER_M = 0.008
SOCKET_TIP_RADIAL_TOLERANCE_M = 0.004
MAX_TRANSLATION_STEP_M = 0.025
TOOL_TABLE_CLEARANCE_M = 0.003
WORLD_DOWN = [0.0, 0.0, -1.0]
START_POSE_INSERT = [
    -0.08121582760511159,
    0.04770862987737767,
    0.5719587969799858,
    0.9926745259136501,
    0.0055147057933565835,
    -0.019325801689775535,
    0.11913600216211073,
]
SEED_VARIATION_POLICY = {
    "sampler": "hammer_style_hollow_column_socket_fixed_yaw_stratified_pose_v1",
    "target_grid_shape": [3, 3],
    "target_center_x_m": [-0.145, 0.145],
    "target_center_y_m": [-0.115, 0.115],
    "target_xy_jitter_m": [-0.008, 0.008],
    "target_yaw_deg": [0.0, 0.0],
    "tool_safe_x_m": [-0.250, 0.250],
    "tool_safe_y_m": [-0.180, 0.180],
    "tool_target_distance_m": [0.105, 0.155],
    "tool_angle_degrees": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
    "tool_angle_jitter_deg": [-10.0, 10.0],
    "tool_yaw_jitter_deg": [-30.0, 30.0],
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


def _unit_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in values[:3]))
    if norm <= 1e-12:
        raise ValueError("cannot normalize near-zero vector")
    return [float(item) / norm for item in values[:3]]


def _lerp(lo: float, hi: float, alpha: float) -> float:
    return float(lo) * (1.0 - float(alpha)) + float(hi) * float(alpha)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _table_contact_tool_pose(pose_xyzw: list[float]) -> list[float]:
    projected = table_contact_start_pose_from_state_bundle(
        start_pose={
            "position": [float(item) for item in pose_xyzw[:3]],
            "orientation_xyzw": [float(item) for item in pose_xyzw[3:7]],
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


def _sample_seed_params(
    seed: int,
    *,
    socket_top_local_z_m: float = SOCKET_TOP_LOCAL_Z,
    socket_size_xyz: list[float] | None = None,
) -> dict[str, Any]:
    if socket_size_xyz is None:
        socket_size_xyz = list(SOCKET_SIZE_XYZ)
    rng = random.Random(int(seed))
    cols, rows = [int(item) for item in SEED_VARIATION_POLICY["target_grid_shape"]]
    col = int(seed) % cols
    row = (int(seed) // cols) % rows
    target_x = _lerp(
        *SEED_VARIATION_POLICY["target_center_x_m"],
        (float(col) + 0.5) / float(cols),
    )
    target_y = _lerp(
        *SEED_VARIATION_POLICY["target_center_y_m"],
        (float(row) + 0.5) / float(rows),
    )
    target_x = _clamp(
        target_x + rng.uniform(*SEED_VARIATION_POLICY["target_xy_jitter_m"]),
        *SEED_VARIATION_POLICY["target_center_x_m"],
    )
    target_y = _clamp(
        target_y + rng.uniform(*SEED_VARIATION_POLICY["target_xy_jitter_m"]),
        *SEED_VARIATION_POLICY["target_center_y_m"],
    )
    target_yaw_deg = 0.0

    tool_pose = _table_contact_tool_pose([float(item) for item in START_POSE_INSERT])
    tool_x = float(tool_pose[0])
    tool_y = float(tool_pose[1])
    rel_x = tool_x - target_x
    rel_y = tool_y - target_y
    tool_distance = math.hypot(rel_x, rel_y)
    tool_angle_deg = math.degrees(math.atan2(rel_y, rel_x))
    tool_yaw_deg = 0.0
    target_quat = _quat_from_z_yaw(math.radians(target_yaw_deg))
    socket_pose = [target_x, target_y, TABLE_TOP_Z, *target_quat]
    opening_world = [target_x, target_y, TABLE_TOP_Z + float(socket_top_local_z_m)]
    return {
        "seed": int(seed),
        "target_grid_cell": [col, row],
        "tool_start_pose": tool_pose,
        "tool_target_angle_deg": tool_angle_deg,
        "tool_target_distance_m": tool_distance,
        "tool_xy_relative_to_target_m": [tool_x - target_x, tool_y - target_y],
        "tool_xy_offset_from_nominal_m": [
            tool_x - float(START_POSE_INSERT[0]),
            tool_y - float(START_POSE_INSERT[1]),
        ],
        "tool_yaw_jitter_deg": tool_yaw_deg,
        "target_center_world": opening_world,
        "target_yaw_deg": target_yaw_deg,
        "target_pose_xyzw": socket_pose,
        "target_size_xyz": [float(item) for item in socket_size_xyz],
        "terminal_inside_depth_m": SOCKET_TERMINAL_INSIDE_DEPTH_M,
        "sampler_policy": copy.deepcopy(SEED_VARIATION_POLICY),
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


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _repo_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _variant_slug(value: Any) -> str:
    text = str(value or "deep_socket").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.split("_") if part) or "deep_socket"


def _load_benchmark_config_defaults(
    argv: list[str],
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any]]:
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
        "target_approach_axis_source",
        "terminal_inside_depth_m",
        "success_min_inside_depth_m",
        "initial_grasp_lift_m",
        "initial_grasp_lift_steps",
        "initial_grasp_hold_steps",
        "response_mode",
        "model",
        "skip_rollout",
        "resume",
        "socket_asset",
        "socket_size_xyz",
        "socket_top_local_z_m",
        "socket_inner_diameter_m",
        "socket_tip_radial_tolerance_m",
        "max_translation_step_m",
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
            "public_support_surface_normal_world": copy.deepcopy(
                task.get("metadata", {}).get("public_support_surface_normal_world", [0.0, 0.0, 1.0])
            ),
        },
    }


def _task_spec(
    *,
    task_id: str,
    seed: int,
    tool_refs_dir: Path,
    seed_params: dict[str, Any],
    variant: str = "deep_socket",
    socket_asset: str = SOCKET_ASSET,
    socket_size_xyz: list[float] | None = None,
    socket_top_local_z_m: float = SOCKET_TOP_LOCAL_Z,
) -> dict[str, Any]:
    if socket_size_xyz is None:
        socket_size_xyz = list(SOCKET_SIZE_XYZ)
    start_pose = [float(item) for item in seed_params["tool_start_pose"]]
    target_pose = [float(item) for item in seed_params["target_pose_xyzw"]]
    target_center = [float(item) for item in seed_params["target_center_world"]]
    return {
        "schema_version": "parttoolgen_tool_use_task_v0",
        "task_id": task_id,
        "split": "dev",
        "instruction": "Use this screwdriver to insert its tip into the socket.",
        "tool": {
            "tool_id": "long_screwdriver",
            "tool_prior_dir": _repo_display(tool_refs_dir),
        },
        "execution": {
            "simtoolreal_category": "screwdriver",
            "simtoolreal_table_asset": TABLE_ASSET,
            "simtoolreal_tool_mesh": TOOL_MESH,
            "simtoolreal_start_pose_source": TOOL_TRAJECTORY,
            "start_pose_policy": "seeded_dextoolbench_long_screwdriver_insert_start",
            "simtoolreal_start_pose": start_pose,
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, TABLE_Z, 0.0, 0.0, 0.0, 1.0],
                "table_asset": TABLE_ASSET,
                "table_z": TABLE_Z,
                "object_name": "long_screwdriver",
                "object_category": "screwdriver",
                "task_name": "screwdriver_insert",
                "target_collision_role": "movable_public_hammer_style_hollow_column_socket",
                "min_tool_table_clearance_m": 0.0003,
                "parttoolgen_benchmark_extra_actors": [
                    {
                        "name": "screwdriver_insert_socket",
                        "asset_file": socket_asset,
                        "pose_xyzw": target_pose,
                        "color_rgb": [0.62, 0.64, 0.66],
                        "disable_gravity": True,
                        "fix_base_link": True,
                        "friction": 1.0,
                    }
                ],
            },
        },
        "public_scene": {
            "provider": "simtoolreal_rgbd_image_grounding",
            "target_selection_mode": "vlm_select_visible_socket_opening",
            "scene_evidence": [],
            "method_inputs": [
                "task language",
                "SimToolReal RGB-D render",
                "public long_screwdriver tool refs from P3SAM/VLM part-face grounding",
                "public visible screw-hole actor geometry in the scene render",
            ],
            "forbidden_method_inputs": [
                "private_eval.target_geometry",
                "tool_family",
                "task_family",
                "primitive fallback",
                "legacy screwdriver spin task",
            ],
        },
        "metadata": {
            "seed": int(seed),
            "variant": variant,
            "benchmark_role": f"image_grounded_screwdriver_insert_{variant}_smoke",
            "target_kind": f"public_{variant}_opening",
            "public_support_surface_normal_world": [0.0, 0.0, 1.0],
            "socket_opening_center_world": target_center,
            "socket_root_pose_xyzw": target_pose,
            "socket_yaw_deg": float(seed_params["target_yaw_deg"]),
            "socket_size_xyz": [float(item) for item in socket_size_xyz],
            "socket_top_local_z_m": float(socket_top_local_z_m),
            "terminal_inside_depth_m": float(seed_params["terminal_inside_depth_m"]),
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
            "tool_start_policy": "fixed_near_hand_start_pose_shared_by_vlm_render_and_rollout",
            "notes": (
                "Long-screwdriver insertion uses a public, movable deep-socket "
                "actor. The VLM grounds the opening center from RGB-D evidence; "
                "the use-stage direction comes from VLM point refs, with the "
                "terminal inside-socket goal supplied by the task contract."
            ),
        },
    }


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


def _validate_tool_refs_dir(path: Path) -> None:
    required = [
        "axis_binding.json",
        "axis_mesh_overlay_report.json",
        "contact_point_selection.json",
    ]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "missing screwdriver tool refs in "
            f"{path}: {', '.join(missing)}. Generate task-conditioned "
            "long_screwdriver refs before running this benchmark."
        )


def _point_along(point: list[float], axis: list[float], distance_m: float) -> list[float]:
    unit = _unit_vector(axis)
    return [float(point[i]) + unit[i] * float(distance_m) for i in range(3)]


def _rewrite_scene_refs_for_socket_terminal_goal(
    scene_refs: dict[str, Any],
    *,
    terminal_inside_depth_m: float,
) -> dict[str, Any]:
    contact = scene_refs.get("start_point") or scene_refs.get("start")
    if not isinstance(contact, dict) or not isinstance(contact.get("position_world"), list):
        raise ValueError("lifted scene refs missing start_point.position_world")
    axis = None
    axis_source = "vlm_scene_direction"
    direction = scene_refs.get("direction")
    if isinstance(direction, list) and len(direction) >= 3:
        axis = [float(item) for item in direction[:3]]
    if axis is None:
        raise ValueError("screwdriver insertion requires lifted VLM scene.direction")
    axis = _unit_vector(axis)
    contact_position = [float(item) for item in contact["position_world"][:3]]
    end = dict(contact)
    end["keypoint_id"] = "target_socket_inside_terminal"
    end["role"] = "target_socket_inside_terminal"
    end["position_world"] = _point_along(contact_position, axis, float(terminal_inside_depth_m))
    end["source"] = f"task_contract_socket_inside_goal_along_{axis_source}"
    end["terminal_inside_depth_m"] = float(terminal_inside_depth_m)
    motion_path = [dict(contact), end]
    scene_refs["end"] = end
    scene_refs["use"] = end
    scene_refs["action_endpoint"] = end
    scene_refs["direction"] = axis
    scene_refs["motion_path"] = motion_path
    scene_refs["direction_source"] = f"{axis_source}_plus_task_socket_inside_goal"
    scene_refs["action_distance_m"] = float(terminal_inside_depth_m)
    scene_refs["action_distance_source"] = f"task_contract_socket_inside_goal_along_{axis_source}"
    scene_refs["terminal_socket_goal"] = {
        "type": "tip_inside_socket",
        "socket_center_ref": "scene.start",
        "terminal_ref": "scene.end",
        "axis_ref": "scene.direction",
        "terminal_inside_depth_m": float(terminal_inside_depth_m),
        "source": "public_task_contract",
    }
    scene_refs["target_approach_axis"] = {
        "vector_world": [-axis[0], -axis[1], -axis[2]],
        "source": f"opposite_{axis_source}",
    }
    semantic_refs = scene_refs.get("semantic_refs")
    if isinstance(semantic_refs, dict):
        use_stage = semantic_refs.get("use_stage")
        if isinstance(use_stage, dict):
            use_stage["end"] = end
            use_stage["path"] = motion_path
            use_stage["action_distance_m"] = float(terminal_inside_depth_m)
            use_stage["terminal_socket_goal"] = dict(scene_refs["terminal_socket_goal"])
            axis_ref = use_stage.get("axis")
            if isinstance(axis_ref, dict):
                axis_ref["vector_world"] = axis
                axis_ref["source"] = f"{axis_source}_plus_task_socket_inside_goal"
        semantic_refs["use_stage_refs"] = semantic_refs.get("use_stage")
    return scene_refs


def _apply_socket_terminal_goal(lifted_dir: Path, *, terminal_inside_depth_m: float) -> None:
    for filename in ("lifted_refs.json", "constraint_program.json"):
        path = lifted_dir / filename
        data = _load_json(path)
        if filename == "constraint_program.json":
            scene_refs = data.get("scene_refs")
            if not isinstance(scene_refs, dict):
                raise ValueError("constraint_program.json missing scene_refs")
            data["scene_refs"] = _rewrite_scene_refs_for_socket_terminal_goal(
                scene_refs,
                terminal_inside_depth_m=terminal_inside_depth_m,
            )
            metadata = data.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["screwdriver_terminal_socket_goal"] = {
                    "terminal_inside_depth_m": float(terminal_inside_depth_m),
                    "source": "public_task_contract",
                }
        else:
            _rewrite_scene_refs_for_socket_terminal_goal(
                data,
                terminal_inside_depth_m=terminal_inside_depth_m,
            )
            data["screwdriver_terminal_socket_goal"] = {
                "terminal_inside_depth_m": float(terminal_inside_depth_m),
                "source": "public_task_contract",
            }
        _write_json(path, data)


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


def _semantic_contact_error(execution_report: Mapping[str, Any]) -> float | None:
    reports = execution_report.get("stage_reports")
    if not isinstance(reports, list):
        return None
    values: list[float] = []
    for row in reports:
        if not isinstance(row, Mapping):
            continue
        for key in ("min_semantic_contact_error_m", "semantic_contact_error_m"):
            if row.get(key) is not None:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    pass
    return min(values) if values else None


def _axis_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3:
        return None
    ua = _unit_vector([float(item) for item in a[:3]])
    ub = _unit_vector([float(item) for item in b[:3]])
    return sum(ua[i] * ub[i] for i in range(3))


def _as_world_point(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _ref_position(ref: Any) -> list[float] | None:
    if isinstance(ref, Mapping):
        return _as_world_point(ref.get("position_world"))
    return None


def _scene_point_from_lifted_refs(
    lifted_refs: Mapping[str, Any],
    keys: Sequence[str],
) -> list[float] | None:
    for key in keys:
        point = _ref_position(lifted_refs.get(key))
        if point is not None:
            return point
    return None


def _vector_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _vector_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _vector_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(3)))


def _scene_direction_from_lifted_refs(lifted_refs: Mapping[str, Any]) -> list[float]:
    axis = lifted_refs.get("direction")
    if isinstance(axis, list) and len(axis) >= 3:
        return _unit_vector([float(item) for item in axis[:3]])
    raise ValueError("screwdriver insertion metrics require a lifted scene.direction")


def _tip_path_metrics_from_tracking_samples(
    *,
    lifted_refs: Mapping[str, Any],
    tracking_samples: Sequence[Mapping[str, Any]] | None,
    socket_radial_tolerance_m: float | None,
) -> dict[str, Any] | None:
    if not tracking_samples:
        return None
    contact_point = _scene_point_from_lifted_refs(
        lifted_refs,
        ("start", "start_point"),
    )
    if contact_point is None:
        return None
    end_point = _scene_point_from_lifted_refs(lifted_refs, ("end", "end_point", "action_endpoint"))
    axis = _scene_direction_from_lifted_refs(lifted_refs)

    semantic_candidates: list[dict[str, Any]] = []
    for row in tracking_samples:
        if not isinstance(row, Mapping):
            continue
        actual = _as_world_point(row.get("actual_contact_point_world"))
        if actual is None:
            continue
        delta = _vector_sub(actual, contact_point)
        signed_depth = _vector_dot(delta, axis)
        parallel = [axis[i] * signed_depth for i in range(3)]
        radial_vec = [delta[i] - parallel[i] for i in range(3)]
        radial_error = _vector_norm(radial_vec)
        depth = max(0.0, signed_depth)
        end_error = _vector_norm(_vector_sub(actual, end_point)) if end_point is not None else None
        candidate = {
            "step": row.get("step"),
            "goal_index": row.get("goal_index"),
            "stage": row.get("stage"),
            "actual_tip_world": actual,
            "tip_xy_radial_error_m": radial_error,
            "tip_insertion_depth_m": depth,
            "tip_signed_insertion_depth_m": signed_depth,
            "tip_end_3d_error_m": end_error,
            "semantic_gate": bool(row.get("semantic_gate")),
        }
        if row.get("semantic_gate") is True:
            semantic_candidates.append(candidate)

    candidates = semantic_candidates
    if not candidates:
        return None

    min_radial = min(candidates, key=lambda item: float(item["tip_xy_radial_error_m"]))
    deepest = max(candidates, key=lambda item: float(item["tip_insertion_depth_m"]))
    radial_gated_candidates = candidates
    if socket_radial_tolerance_m is not None:
        radial_gated_candidates = [
            item
            for item in candidates
            if float(item["tip_xy_radial_error_m"]) <= float(socket_radial_tolerance_m)
        ]
    deepest_inside = (
        max(radial_gated_candidates, key=lambda item: float(item["tip_insertion_depth_m"]))
        if radial_gated_candidates
        else None
    )
    depth_source = deepest_inside if deepest_inside is not None else deepest
    best_end = (
        min(
            (item for item in candidates if item.get("tip_end_3d_error_m") is not None),
            key=lambda item: float(item["tip_end_3d_error_m"]),
            default=None,
        )
        if end_point is not None
        else None
    )
    return {
        "source": "rollout_tracking_samples_axis_projection",
        "socket_center_world": contact_point,
        "target_insert_end_world": end_point,
        "insertion_axis_world": axis,
        "tip_xy_radial_error_m": float(min_radial["tip_xy_radial_error_m"]),
        "tip_radial_error_m": float(min_radial["tip_xy_radial_error_m"]),
        "tip_insertion_depth_m": float(depth_source["tip_insertion_depth_m"]),
        "max_tip_insertion_depth_m": float(deepest["tip_insertion_depth_m"]),
        "tip_end_3d_error_m": (
            float(best_end["tip_end_3d_error_m"]) if best_end is not None else None
        ),
        "tip_radial_error_at_max_depth_m": float(deepest["tip_xy_radial_error_m"]),
        "tip_insertion_depth_source": (
            "max_depth_within_radial_tolerance"
            if deepest_inside is not None
            else "max_depth_without_radial_tolerance_match"
        ),
        "socket_radial_tolerance_m": socket_radial_tolerance_m,
        "radial_gated_sample_count": len(radial_gated_candidates),
        "semantic_sample_count": len(semantic_candidates),
        "sample_count": len(candidates),
        "metric_sample_policy": "semantic_gate_samples_only",
        "best_radial_sample": min_radial,
        "best_depth_sample": depth_source,
    }


def _insertion_metrics(
    *,
    lifted_refs: Mapping[str, Any],
    rollout_summary: Mapping[str, Any] | None,
    execution_report: Mapping[str, Any] | None,
    tracking_samples: Sequence[Mapping[str, Any]] | None = None,
    socket_radial_tolerance_m: float | None = None,
) -> dict[str, Any]:
    semantic_passed = None
    for report in (execution_report, rollout_summary):
        if isinstance(report, Mapping) and report.get("semantic_execution_passed") is not None:
            semantic_passed = bool(report["semantic_execution_passed"])
            break
    terminal_goal = lifted_refs.get("terminal_socket_goal")
    terminal_depth = (
        terminal_goal.get("terminal_inside_depth_m")
        if isinstance(terminal_goal, Mapping)
        else None
    )
    if terminal_depth is None:
        terminal_depth = lifted_refs.get("action_distance_m")
    try:
        terminal_inside_depth_m = float(terminal_depth)
    except (TypeError, ValueError):
        terminal_inside_depth_m = None
    semantic_contact_error = (
        _semantic_contact_error(execution_report)
        if isinstance(execution_report, Mapping)
        else None
    )
    path_metrics = _tip_path_metrics_from_tracking_samples(
        lifted_refs=lifted_refs,
        tracking_samples=tracking_samples,
        socket_radial_tolerance_m=socket_radial_tolerance_m,
    )
    support_axis_vector = None
    target_surface_normal = lifted_refs.get("target_surface_normal")
    normal_vector = (
        target_surface_normal.get("vector_world")
        if isinstance(target_surface_normal, Mapping)
        else None
    )
    if isinstance(normal_vector, list) and len(normal_vector) >= 3:
        normal = _unit_vector([float(item) for item in normal_vector[:3]])
        support_axis_vector = [-normal[0], -normal[1], -normal[2]]
    axis_cosine = _axis_cosine(
        lifted_refs.get("direction") if isinstance(lifted_refs.get("direction"), list) else None,
        support_axis_vector,
    )
    metrics = {
        "schema_version": "toolframe_screwdriver_insert_metrics_v1",
        "source": (
            path_metrics.get("source")
            if isinstance(path_metrics, Mapping)
            else "semantic_gate_only_missing_tracking_samples"
        ),
        "tip_radial_error_m": None,
        "tip_xy_radial_error_m": None,
        "tip_insertion_depth_m": None,
        "tip_end_3d_error_m": semantic_contact_error,
        "semantic_contact_error_to_depth_end_m": semantic_contact_error,
        "terminal_inside_depth_m": terminal_inside_depth_m,
        "axis_cosine": axis_cosine,
        "semantic_execution_passed": semantic_passed,
    }
    if isinstance(path_metrics, Mapping):
        metrics.update(path_metrics)
        metrics["terminal_inside_depth_m"] = terminal_inside_depth_m
        metrics["axis_cosine"] = axis_cosine
        metrics["semantic_execution_passed"] = semantic_passed
        metrics["semantic_contact_error_to_depth_end_m"] = semantic_contact_error
    return metrics


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


def _refresh_success_fields(
    result: dict[str, Any],
    *,
    skip_rollout: bool,
    max_tip_radial_error_m: float,
    min_tip_inside_depth_m: float,
) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    rollout_summary = _load_optional_json(artifacts.get("rollout_summary")) if isinstance(artifacts, dict) else None
    execution_report = _load_optional_json(artifacts.get("execution_report")) if isinstance(artifacts, dict) else None
    compile_summary = _load_optional_json(artifacts.get("compile_summary")) if isinstance(artifacts, dict) else None
    lifted_refs = _load_optional_json(artifacts.get("lifted_refs")) if isinstance(artifacts, dict) else None
    tracking_samples = None
    if isinstance(artifacts, dict):
        tracking_samples = _load_optional_json_value(artifacts.get("tracking_samples"))
        if tracking_samples is None and artifacts.get("rollout_summary"):
            tracking_samples = _load_optional_json_value(
                Path(str(artifacts["rollout_summary"])).parent / "tracking_samples.json"
            )
    if compile_summary is None:
        compile_summary = {"path_solver_passed": bool(result.get("compile_path_solver_passed"))}
    insertion = None
    if lifted_refs is not None:
        insertion = _insertion_metrics(
            lifted_refs=lifted_refs,
            rollout_summary=rollout_summary,
            execution_report=execution_report,
            tracking_samples=tracking_samples if isinstance(tracking_samples, list) else None,
            socket_radial_tolerance_m=float(max_tip_radial_error_m),
        )
    if not isinstance(insertion, Mapping):
        insertion = result.get("screwdriver_insert_metrics")
    success_check = check_task_success(
        "screwdriver_insert",
        compile_summary=compile_summary,
        insertion=insertion if isinstance(insertion, Mapping) else None,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        skip_rollout=skip_rollout,
        max_tip_radial_error_m=float(max_tip_radial_error_m),
        min_tip_insertion_depth_m=float(min_tip_inside_depth_m),
    )
    compiled_goal_check = None
    subgoals = None
    if isinstance(artifacts, dict):
        subgoals = _load_optional_json(artifacts.get("path_solved_tool_pose_subgoals"))
        if subgoals is None:
            subgoals = _load_optional_json(artifacts.get("tool_pose_subgoals"))
    if isinstance(lifted_refs, Mapping) and isinstance(subgoals, Mapping):
        compiled_goal_check = check_screwdriver_insert_compiled_task_goal(
            subgoals=subgoals,
            lifted_refs=lifted_refs,
            max_tip_radial_error_m=float(max_tip_radial_error_m),
            min_tip_inside_depth_m=float(min_tip_inside_depth_m),
        )
    refreshed = dict(result)
    refreshed["task_success_check"] = success_check
    if compiled_goal_check is not None:
        refreshed["compiled_task_goal_check"] = compiled_goal_check
        refreshed["compiled_task_goal_passed"] = bool(compiled_goal_check.get("passed"))
    refreshed["insert_semantic_passed"] = predicate_passed(success_check, "semantic_insertion_path")
    refreshed["insert_radial_passed"] = predicate_passed(success_check, "tip_radial_alignment")
    refreshed["insert_depth_passed"] = predicate_passed(success_check, "tip_insertion_depth")
    refreshed["screwdriver_insert_metrics"] = insertion if isinstance(insertion, Mapping) else None
    refreshed["success_policy"] = success_check["policy"]
    if compiled_goal_check is not None:
        refreshed["passed"] = bool(success_check.get("passed")) and bool(
            compiled_goal_check.get("passed")
        )
        refreshed["failure_category"] = (
            compiled_goal_check.get("diagnostics", {}).get("failure_category")
            if not bool(compiled_goal_check.get("passed"))
            else success_check.get("diagnostics", {}).get("failure_category")
        )
    else:
        refreshed["passed"] = bool(success_check.get("passed"))
        refreshed["failure_category"] = success_check.get("diagnostics", {}).get("failure_category")
    return _apply_selected_tool_ref_fields(refreshed)


def _run_trial(*, seed: int, index: int, args: argparse.Namespace) -> dict[str, Any]:
    variant = _variant_slug(getattr(args, "task_variant", "deep_socket"))
    task_id = f"long_screwdriver_{variant}_insert_seed_{seed:03d}"
    trial_dir = args.output_root / task_id
    if args.resume and (trial_dir / "trial_result.json").exists():
        result = _refresh_success_fields(
            _load_json(trial_dir / "trial_result.json"),
            skip_rollout=bool(args.skip_rollout),
            max_tip_radial_error_m=float(args.socket_tip_radial_tolerance_m),
            min_tip_inside_depth_m=float(args.success_min_inside_depth_m),
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

    seed_params = _sample_seed_params(
        seed,
        socket_top_local_z_m=float(args.socket_top_local_z_m),
        socket_size_xyz=[float(item) for item in args.socket_size_xyz],
    )
    seed_params["terminal_inside_depth_m"] = float(args.terminal_inside_depth_m)
    task = _task_spec(
        task_id=task_id,
        seed=seed,
        tool_refs_dir=args.tool_refs_dir,
        seed_params=seed_params,
        variant=variant,
        socket_asset=str(args.socket_asset),
        socket_size_xyz=[float(item) for item in args.socket_size_xyz],
        socket_top_local_z_m=float(args.socket_top_local_z_m),
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
            "--action-direction-z-policy",
            "anchor_height",
            "--direction-source",
            "path_geometry",
            "--tool-motion-axis-ref",
            "tool.motion_axis",
            "--tool-motion-axis-target-ref",
            "scene.direction",
            "--tool-motion-axis-relation",
            "parallel",
            "--target-approach-axis-source",
            args.target_approach_axis_source,
            "--end-surface-policy",
            "anchor_contact_plane",
            "--target-surface-normal-world",
            "0",
            "0",
            "1",
        ],
        log_path=trial_dir / "logs/lift.json",
    )
    _apply_socket_terminal_goal(
        lifted_dir,
        terminal_inside_depth_m=float(args.terminal_inside_depth_m),
    )

    enriched_program = lifted_dir / "constraint_program_with_tool_refs.json"
    _run(
        [
            py,
            str(_script("merge_tool_refs_into_constraint_program.py")),
            "--program",
            str(lifted_dir / "constraint_program.json"),
            "--axis-binding",
            str(args.tool_refs_dir / "axis_binding.json"),
            "--axis-mesh-report",
            str(args.tool_refs_dir / "axis_mesh_overlay_report.json"),
            "--contact-point-selection",
            str(args.tool_refs_dir / "contact_point_selection.json"),
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
            "0.25",
            "--initial-grasp-lift-m",
            str(args.initial_grasp_lift_m),
            "--initial-grasp-lift-steps",
            str(args.initial_grasp_lift_steps),
            "--min-clearance-m",
            "0.003",
            "--swept-min-clearance-m",
            "0.006",
            "--allowed-contact-penetration-m",
            "0.004",
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
            TOOL_MESH,
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
    enriched = _load_json(enriched_program)
    selected_program = _load_optional_json(compile_dir / "selected_constraint_program.json") or {}
    selected_tool_refs = selected_program.get("tool_refs") if isinstance(selected_program, dict) else None
    if not isinstance(selected_tool_refs, dict):
        selected_tool_refs = enriched.get("tool_refs", {}) if isinstance(enriched.get("tool_refs"), dict) else {}
    insertion = _insertion_metrics(
        lifted_refs=lifted_refs,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        tracking_samples=tracking_samples,
        socket_radial_tolerance_m=float(args.socket_tip_radial_tolerance_m),
    )
    subgoals = _load_json(compile_dir / "path_solved_tool_pose_subgoals.json")
    compiled_goal_check = check_screwdriver_insert_compiled_task_goal(
        subgoals=subgoals,
        lifted_refs=lifted_refs,
        max_tip_radial_error_m=float(args.socket_tip_radial_tolerance_m),
        min_tip_inside_depth_m=float(args.success_min_inside_depth_m),
    )
    _write_json(compile_dir / "compiled_task_goal_check.json", compiled_goal_check)
    success_check = check_task_success(
        "screwdriver_insert",
        compile_summary=compile_summary,
        insertion=insertion,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
        skip_rollout=bool(args.skip_rollout),
        max_tip_radial_error_m=float(args.socket_tip_radial_tolerance_m),
        min_tip_insertion_depth_m=float(args.success_min_inside_depth_m),
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
        tool_refs_dir=args.tool_refs_dir,
        compile_dir=compile_dir,
        rollout_dir=trial_dir / "rollout",
        copy_files=False,
    )
    passed = bool(success_check.get("passed")) and bool(compiled_goal_check.get("passed"))
    failure_category = (
        compiled_goal_check.get("diagnostics", {}).get("failure_category")
        if not bool(compiled_goal_check.get("passed"))
        else success_check.get("diagnostics", {}).get("failure_category")
    )
    result = {
        "index": index,
        "seed": seed,
        "task_id": task_id,
        "seed_params": seed_params,
        "passed": passed,
        "skip_rollout": bool(args.skip_rollout),
        "target_object": scene_keypoints.get("target_object"),
        "lifted_start_point_world": lifted_refs.get("start_point", {}).get("position_world"),
        "insertion_axis_world": lifted_refs.get("direction"),
        "direction_source": lifted_refs.get("direction_source"),
        "action_distance_m": lifted_refs.get("action_distance_m"),
        "terminal_inside_depth_m": (
            lifted_refs.get("terminal_socket_goal", {}).get("terminal_inside_depth_m")
            if isinstance(lifted_refs.get("terminal_socket_goal"), Mapping)
            else None
        ),
        "selected_contact_point_face": selected_tool_refs.get("contact_point_face_marker"),
        "selected_contact_point_ref": (
            selected_tool_refs.get("contact_point_ref")
            or selected_tool_refs.get("contact_point_source_ref")
        ),
        "selected_motion_axis_face": selected_tool_refs.get("motion_axis_face_marker"),
        "tool_ref_candidate_selection": selected_tool_refs.get("active_contact_point_candidate_selection"),
        "compile_path_solver_passed": bool(compile_summary.get("path_solver_passed")),
        "policy_goal_passed": (
            bool(rollout_summary.get("policy_goal_passed")) if rollout_summary else None
        ),
        "tracking_execution_passed": (
            bool(rollout_summary.get("tracking_execution_passed")) if rollout_summary else None
        ),
        "semantic_execution_passed": (
            bool(rollout_summary.get("semantic_execution_passed")) if rollout_summary else None
        ),
        "insert_semantic_passed": predicate_passed(success_check, "semantic_insertion_path"),
        "insert_radial_passed": predicate_passed(success_check, "tip_radial_alignment"),
        "insert_depth_passed": predicate_passed(success_check, "tip_insertion_depth"),
        "compiled_task_goal_passed": bool(compiled_goal_check.get("passed")),
        "compiled_task_goal_check": compiled_goal_check,
        "screwdriver_insert_metrics": insertion,
        "task_success_check": success_check,
        "success_policy": success_check["policy"],
        "failure_category": failure_category,
        "grasp_lift_passed": grasp_lift_passed,
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
            "path_solved_tool_pose_subgoals": str(
                compile_dir / "path_solved_tool_pose_subgoals.json"
            ),
            "compiled_task_goal_check": str(compile_dir / "compiled_task_goal_check.json"),
            "fixed_goals": str(compile_dir / "simtoolreal_fixed_goals.json"),
            "rollout_summary": str(trial_dir / "rollout/rollout_summary.json")
            if not args.skip_rollout
            else None,
            "execution_report": str(trial_dir / "rollout/execution_report.json")
            if not args.skip_rollout
            else None,
            "tracking_samples": str(trial_dir / "rollout/tracking_samples.json")
            if not args.skip_rollout and (trial_dir / "rollout/tracking_samples.json").exists()
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
        "execution_failed_stage": execution_report.get("execution_failed_stage") if execution_report else None,
    }
    _write_json(trial_dir / "trial_result.json", result)
    return result


def _summary(rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    passed = [row for row in rows if row.get("passed")]
    semantic_rows = [row for row in rows if row.get("insert_semantic_passed") is not None]
    semantic_passed = [row for row in semantic_rows if row.get("insert_semantic_passed")]
    radial_rows = [row for row in rows if row.get("insert_radial_passed") is not None]
    radial_passed = [row for row in radial_rows if row.get("insert_radial_passed")]
    depth_rows = [row for row in rows if row.get("insert_depth_passed") is not None]
    depth_passed = [row for row in depth_rows if row.get("insert_depth_passed")]
    compiled_rows = [row for row in rows if row.get("compiled_task_goal_passed") is not None]
    compiled_passed = [row for row in compiled_rows if row.get("compiled_task_goal_passed")]
    lift_rows = [row for row in rows if row.get("grasp_lift_passed") is not None]
    summary = {
        "schema_version": "toolframe_image_grounded_screwdriver_insert_eval_v1",
        "success_policy": SCREWDRIVER_INSERT_SUCCESS_POLICY["name"],
        "success_includes_compiled_task_goal": True,
        "success_requires_tracking_execution_passed": False,
        "output_root": str(output_root),
        "trial_count": len(rows),
        "passed_count": len(passed),
        "pass_rate": (len(passed) / len(rows)) if rows else 0.0,
        "semantic_execution_pass_rate": (
            len(semantic_passed) / len(semantic_rows)
        )
        if semantic_rows
        else None,
        "tip_xy_radial_pass_rate": (
            len(radial_passed) / len(radial_rows)
        )
        if radial_rows
        else None,
        "tip_insert_depth_pass_rate": (
            len(depth_passed) / len(depth_rows)
        )
        if depth_rows
        else None,
        "compiled_task_goal_pass_rate": (
            len(compiled_passed) / len(compiled_rows)
        )
        if compiled_rows
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
        "trials": rows,
    }
    _write_json(output_root / "summary.json", summary)
    lines = [
        "# Image-Grounded Screwdriver Insert Eval",
        "",
        f"- trials: {summary['trial_count']}",
        f"- success policy: {SCREWDRIVER_INSERT_SUCCESS_POLICY['name']}",
        f"- pass rate: {summary['pass_rate']:.3f}",
        f"- semantic execution pass rate: {summary['semantic_execution_pass_rate']:.3f}"
        if summary["semantic_execution_pass_rate"] is not None
        else "- semantic execution pass rate: n/a",
        f"- tip XY/radial pass rate: {summary['tip_xy_radial_pass_rate']:.3f}"
        if summary["tip_xy_radial_pass_rate"] is not None
        else "- tip XY/radial pass rate: n/a",
        f"- tip insertion-depth pass rate: {summary['tip_insert_depth_pass_rate']:.3f}"
        if summary["tip_insert_depth_pass_rate"] is not None
        else "- tip insertion-depth pass rate: n/a",
        f"- compiled task-goal pass rate: {summary['compiled_task_goal_pass_rate']:.3f}"
        if summary["compiled_task_goal_pass_rate"] is not None
        else "- compiled task-goal pass rate: n/a",
        f"- policy goal pass rate: {summary['policy_goal_pass_rate']:.3f}",
        f"- tracking pass rate: {summary['tracking_pass_rate']:.3f}",
        f"- compile path pass rate: {summary['compile_path_pass_rate']:.3f}",
        f"- grasp-lift pass rate: {summary['grasp_lift_pass_rate']:.3f}"
        if summary["grasp_lift_pass_rate"] is not None
        else "- grasp-lift pass rate: n/a",
        "",
        "| seed | pass | compiled | semantic | radial | inside-depth | measured depth | xy radial err | terminal endpoint err diag | target xy | face | goals | steps | failure | video |",
        "|---:|---|---|---|---|---|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        artifacts = row.get("artifacts", {})
        metrics = row.get("screwdriver_insert_metrics", {})
        params = row.get("seed_params", {}) if isinstance(row.get("seed_params"), dict) else {}
        target_center = params.get("target_center_world")
        target_xy_text = (
            f"{float(target_center[0]):.3f},{float(target_center[1]):.3f}"
            if isinstance(target_center, list) and len(target_center) >= 2
            else "n/a"
        )
        depth = metrics.get("tip_insertion_depth_m") if isinstance(metrics, dict) else None
        radial = metrics.get("tip_radial_error_m") if isinstance(metrics, dict) else None
        end_error = metrics.get("tip_end_3d_error_m") if isinstance(metrics, dict) else None
        depth_text = f"{float(depth):.3f}" if isinstance(depth, (int, float)) else "n/a"
        radial_text = f"{float(radial):.3f}" if isinstance(radial, (int, float)) else "n/a"
        end_error_text = (
            f"{float(end_error):.3f}" if isinstance(end_error, (int, float)) else "n/a"
        )
        lines.append(
            f"| {row.get('seed')} | {row.get('passed')} | "
            f"{row.get('compiled_task_goal_passed')} | "
            f"{row.get('insert_semantic_passed')} | "
            f"{row.get('insert_radial_passed')} | "
            f"{row.get('insert_depth_passed')} | "
            f"{depth_text} | {radial_text} | {end_error_text} | {target_xy_text} | "
            f"{row.get('selected_contact_point_face')} | "
            f"{row.get('final_successes')}/{row.get('num_goals')} | {row.get('num_steps')} | "
            f"{row.get('failure_category')} | {artifacts.get('rollout_video')} |"
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
        default=ROOT / "artifacts/image_grounded/screwdriver_insert/deep_socket_original_rgb_smoke",
    )
    parser.add_argument("--tool-refs-dir", type=Path, default=DEFAULT_TOOL_REFS_DIR)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-max-steps", type=int, default=1500)
    parser.add_argument("--rollout-max-tracking-error-m", type=float, default=0.018)
    parser.add_argument("--capture-video-every", type=int, default=1)
    parser.add_argument("--approach-offset-m", type=float, default=0.045)
    parser.add_argument("--max-translation-step-m", type=float, default=MAX_TRANSLATION_STEP_M)
    parser.add_argument(
        "--target-approach-axis-source",
        choices=["opposite_scene_direction"],
        default="opposite_scene_direction",
    )
    parser.add_argument(
        "--terminal-inside-depth-m",
        type=float,
        default=SOCKET_TERMINAL_INSIDE_DEPTH_M,
        help=(
            "Compiler terminal goal depth inside the socket aperture along the "
            "VLM/default insertion axis. This defines the 3D task endpoint."
        ),
    )
    parser.add_argument(
        "--success-min-inside-depth-m",
        type=float,
        default=SUCCESS_MIN_INSIDE_DEPTH_M,
        help=(
            "Minimum measured tip depth inside the socket aperture required to "
            "count as inserted."
        ),
    )
    parser.add_argument("--socket-asset", default=SOCKET_ASSET)
    parser.add_argument("--socket-size-xyz", nargs=3, type=float, default=list(SOCKET_SIZE_XYZ))
    parser.add_argument("--socket-top-local-z-m", type=float, default=SOCKET_TOP_LOCAL_Z)
    parser.add_argument("--socket-inner-diameter-m", type=float, default=SOCKET_INNER_DIAMETER_M)
    parser.add_argument(
        "--socket-tip-radial-tolerance-m",
        type=float,
        default=SOCKET_TIP_RADIAL_TOLERANCE_M,
    )
    parser.add_argument("--initial-grasp-lift-m", type=float, default=0.08)
    parser.add_argument("--initial-grasp-lift-steps", type=int, default=3)
    parser.add_argument("--initial-grasp-hold-steps", type=int, default=0)
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(**benchmark_defaults)
    args = parser.parse_args()

    args.task_variant = _variant_slug(
        (benchmark_config or {}).get("variant") if isinstance(benchmark_config, dict) else None
    )
    args.output_root = args.output_root.resolve()
    args.tool_refs_dir = args.tool_refs_dir.resolve()
    _validate_tool_refs_dir(args.tool_refs_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if benchmark_config is not None:
        _write_json(
            args.output_root / "benchmark_config.resolved.json",
            {
                "source": str(benchmark_config_path),
                "runner_args_public": benchmark_config.get("runner_args", {}),
                "private_manifest_omitted": True,
            },
        )
    tool_selection_evidence = write_tool_selection_visual_evidence(
        output_root=args.output_root,
        tool_refs_dir=args.tool_refs_dir,
        task_family="screwdriver_insert",
        split=args.task_variant,
    )
    _write_json(
        args.output_root / "run_config.json",
        {
            "benchmark_config": str(args.benchmark_config.resolve())
            if args.benchmark_config is not None
            else None,
            "task_source": "runner_generated_image_grounded_screwdriver_insert_task",
            "task_variant": args.task_variant,
            "tool_refs_dir": str(args.tool_refs_dir),
            "tool_selection_visual_evidence": tool_selection_evidence,
            "table_asset": TABLE_ASSET,
            "socket_asset": str(args.socket_asset),
            "socket_size_xyz": [float(item) for item in args.socket_size_xyz],
            "socket_top_local_z_m": float(args.socket_top_local_z_m),
            "socket_inner_diameter_m": float(args.socket_inner_diameter_m),
            "socket_tip_radial_tolerance_m": float(args.socket_tip_radial_tolerance_m),
            "terminal_inside_depth_m": float(args.terminal_inside_depth_m),
            "success_min_inside_depth_m": float(args.success_min_inside_depth_m),
            "seed_variation_policy": copy.deepcopy(SEED_VARIATION_POLICY),
            "tool_mesh": TOOL_MESH,
            "seed_start": args.seed_start,
            "trials": args.trials,
            "device": args.device,
            "render_lighting_preset": "soft01",
            "rollout_max_steps": args.rollout_max_steps,
            "rollout_max_tracking_error_m": args.rollout_max_tracking_error_m,
            "capture_video_every": args.capture_video_every,
            "approach_offset_m": args.approach_offset_m,
            "max_translation_step_m": args.max_translation_step_m,
            "target_approach_axis_source": args.target_approach_axis_source,
            "initial_grasp_lift_m": args.initial_grasp_lift_m,
            "initial_grasp_lift_steps": args.initial_grasp_lift_steps,
            "initial_grasp_hold_steps": args.initial_grasp_hold_steps,
            "response_mode": args.response_mode,
            "model": args.model,
            "skip_rollout": args.skip_rollout,
            "mainline_params": {
                "depth_sample_policy": "nearest",
                "direction_source": "vlm_scene_direction_plus_task_socket_inside_goal",
                "end_depth_policy": "task_contract_socket_inside_goal_along_vlm_scene_direction",
                "tool_motion_axis_ref": "tool.motion_axis",
                "tool_motion_axis_target_ref": "scene.direction",
                "tool_motion_axis_relation": "parallel",
                "approach_offset_m": args.approach_offset_m,
                "target_approach_axis_source": args.target_approach_axis_source,
                "max_translation_step_m": args.max_translation_step_m,
                "max_rotation_step_rad": 0.25,
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
        print(f"[{index}/{args.trials}] running screwdriver insert seed {seed:03d}", flush=True)
        try:
            row = _run_trial(seed=seed, index=index, args=args)
        except Exception as exc:
            task_id = f"long_screwdriver_{args.task_variant}_insert_seed_{seed:03d}"
            row = {
                "index": index,
                "seed": seed,
                "task_id": task_id,
                "passed": False,
                "status": "exception",
                "error": str(exc),
                "failure_category": "runner exception or missing prerequisite",
            }
            _write_json(args.output_root / task_id / "trial_result.json", row)
        rows.append(row)
        _summary(rows, args.output_root)
    _summary(rows, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
