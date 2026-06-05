from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def _load_screwdriver_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "cli" / "runners" / "run_image_grounded_screwdriver_insert_eval.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_screwdriver_insert_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screwdriver_eval = _load_screwdriver_eval_module()


def test_seed_params_vary_visible_hole_while_tool_stays_near_hand() -> None:
    params = [screwdriver_eval._sample_seed_params(seed) for seed in range(9)]
    policy = screwdriver_eval.SEED_VARIATION_POLICY

    tool_poses = {
        tuple(round(float(item), 6) for item in row["tool_start_pose"])
        for row in params
    }
    target_poses = {
        tuple(round(float(item), 6) for item in row["target_pose_xyzw"])
        for row in params
    }

    assert len(tool_poses) == 1
    assert len(target_poses) == 9
    assert all(abs(float(row["tool_yaw_jitter_deg"])) <= 1e-6 for row in params)
    target_xs = [float(row["target_center_world"][0]) for row in params]
    target_ys = [float(row["target_center_world"][1]) for row in params]
    assert max(target_xs) - min(target_xs) > 0.12
    assert max(target_ys) - min(target_ys) > 0.10
    for row in params:
        q = row["tool_start_pose"][3:7]
        assert math.isclose(math.sqrt(sum(float(item) * float(item) for item in q)), 1.0)
        assert row["target_yaw_deg"] == 0.0
        assert policy["target_center_x_m"][0] <= row["target_pose_xyzw"][0] <= policy["target_center_x_m"][1]
        assert policy["target_center_y_m"][0] <= row["target_pose_xyzw"][1] <= policy["target_center_y_m"][1]
        assert policy["tool_safe_x_m"][0] <= row["tool_start_pose"][0] <= policy["tool_safe_x_m"][1]
        assert policy["tool_safe_y_m"][0] <= row["tool_start_pose"][1] <= policy["tool_safe_y_m"][1]
        assert row["tool_target_distance_m"] >= 0.0
        rel = row["tool_xy_relative_to_target_m"]
        assert math.isclose(
            math.hypot(float(rel[0]), float(rel[1])),
            float(row["tool_target_distance_m"]),
        )


def test_method_task_keeps_public_visible_hole_actor_for_vlm_rollout_parity() -> None:
    seed_params = screwdriver_eval._sample_seed_params(0)
    task = screwdriver_eval._task_spec(
        task_id="unit_long_screwdriver_deep_socket_insert_seed_000",
        seed=0,
        tool_refs_dir=Path("configs/tool_refs/long_screwdriver"),
        seed_params=seed_params,
    )
    method_task = screwdriver_eval._method_facing_task(task)

    state_bundle = task["execution"]["state_bundle"]
    assert state_bundle["table_asset"].endswith("tabletop_polyhaven_wood_table.urdf")
    actor = state_bundle["parttoolgen_benchmark_extra_actors"][0]
    assert actor["pose_xyzw"] == seed_params["target_pose_xyzw"]
    method_actor = method_task["execution"]["state_bundle"]["parttoolgen_benchmark_extra_actors"][0]
    assert method_actor["pose_xyzw"] == seed_params["target_pose_xyzw"]
    assert method_task["metadata"]["private_scene_geometry_omitted"] is False
    assert "insert its tip into the socket" in task["instruction"].lower()


def test_benchmark_config_runs_three_seed_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "artifacts/tool_use_benchmark/screwdriver_insert/deep_socket/benchmark_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["runner"] == "parttoolgen/src/parttoolgen/cli/runners/run_image_grounded_screwdriver_insert_eval.py"
    assert config["runner_args"]["trials"] == 3
    assert config["runner_args"]["terminal_inside_depth_m"] == 0.035
    assert config["runner_args"]["success_min_inside_depth_m"] == 0.005
    assert (
        config["scene_initialization"]["seed_variation_policy"]["sampler"]
        == screwdriver_eval.SEED_VARIATION_POLICY["sampler"]
    )


def test_insertion_metrics_split_xy_radial_from_depth_end_error() -> None:
    lifted_refs = {
        "action_distance_m": 0.035,
        "direction": [0.0, 0.0, -1.0],
        "start_point": {"position_world": [0.0, 0.0, 0.0]},
        "end_point": {"position_world": [0.0, 0.0, -0.035]},
    }
    tracking_samples = [
        {
            "step": 1,
            "semantic_gate": True,
            "actual_contact_point_world": [0.004, 0.003, -0.017],
        }
    ]
    execution_report = {
        "semantic_execution_passed": True,
        "stage_reports": [
            {
                "semantic_contact_error_m": math.dist(
                    [0.004, 0.003, -0.017],
                    [0.0, 0.0, -0.035],
                )
            }
        ],
    }

    metrics = screwdriver_eval._insertion_metrics(
        lifted_refs=lifted_refs,
        rollout_summary={"semantic_execution_passed": True},
        execution_report=execution_report,
        tracking_samples=tracking_samples,
        socket_radial_tolerance_m=0.012,
    )

    assert math.isclose(metrics["tip_xy_radial_error_m"], 0.005)
    assert math.isclose(metrics["tip_radial_error_m"], 0.005)
    assert math.isclose(metrics["tip_insertion_depth_m"], 0.017)
    assert metrics["tip_end_3d_error_m"] > metrics["tip_xy_radial_error_m"]
    assert math.isclose(metrics["semantic_contact_error_to_depth_end_m"], metrics["tip_end_3d_error_m"])
    assert math.isclose(metrics["terminal_inside_depth_m"], 0.035)


def test_compiled_task_goal_accepts_terminal_tip_inside_socket() -> None:
    lifted_refs = {
        "direction": [0.0, 0.0, -1.0],
        "start_point": {"position_world": [0.0, 0.0, 0.0]},
    }
    subgoals = {
        "goals": [
            {
                "stage": "action_waypoint_2",
                "semantic_gate": True,
                "contact_point_goal": {"position_world": [0.001, 0.002, -0.035]},
            }
        ]
    }

    check = screwdriver_eval.check_screwdriver_insert_compiled_task_goal(
        subgoals=subgoals,
        lifted_refs=lifted_refs,
        max_tip_radial_error_m=0.004,
        min_tip_inside_depth_m=0.005,
    )

    assert check["passed"] is True
    diagnostics = check["diagnostics"]
    assert math.isclose(diagnostics["compiled_tip_inside_depth_m"], 0.035)
    assert diagnostics["compiled_tip_radial_error_m"] < 0.004
