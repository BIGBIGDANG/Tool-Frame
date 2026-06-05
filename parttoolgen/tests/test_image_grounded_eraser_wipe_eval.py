from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


def _load_eraser_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "cli" / "runners" / "run_image_grounded_eraser_wipe_eval.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_eraser_wipe_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eraser_eval = _load_eraser_eval_module()


def _write_box_obj(path: Path, *, size: tuple[float, float, float] = (0.06, 0.02, 0.02)) -> None:
    sx, sy, sz = [float(item) for item in size]
    hx = sx * 0.5
    hy = sy * 0.5
    vertices = [
        (-hx, -hy, 0.0),
        (hx, -hy, 0.0),
        (hx, hy, 0.0),
        (-hx, hy, 0.0),
        (-hx, -hy, sz),
        (hx, -hy, sz),
        (hx, hy, sz),
        (-hx, hy, sz),
    ]
    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    lines = [f"v {x} {y} {z}" for x, y, z in vertices]
    lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_seed_params_vary_target_pose_with_fixed_grasp_ready_tool_pose() -> None:
    params = [eraser_eval._sample_seed_params(seed) for seed in range(20)]
    policy = eraser_eval.SEED_VARIATION_POLICY

    target_poses = {
        tuple(round(float(item), 6) for item in row["target_pose_xyzw"])
        for row in params
    }
    tool_poses = {
        tuple(round(float(item), 6) for item in row["tool_start_pose"])
        for row in params
    }

    assert len(target_poses) == 20
    assert len(tool_poses) == 1
    assert any(abs(float(row["target_yaw_deg"])) > 1e-6 for row in params)
    target_xs = [float(row["target_center_world"][0]) for row in params]
    target_ys = [float(row["target_center_world"][1]) for row in params]
    tool_xs = [float(row["tool_start_pose"][0]) for row in params]
    tool_ys = [float(row["tool_start_pose"][1]) for row in params]
    relative_xs = [float(row["tool_xy_relative_to_target_m"][0]) for row in params]
    relative_ys = [float(row["tool_xy_relative_to_target_m"][1]) for row in params]
    distances = [float(row["tool_target_distance_m"]) for row in params]
    assert max(target_xs) - min(target_xs) > 0.12
    assert max(target_ys) - min(target_ys) > 0.15
    assert min(target_xs) < -0.10 and max(target_xs) > 0.10
    assert math.isclose(max(tool_xs), min(tool_xs))
    assert math.isclose(max(tool_ys), min(tool_ys))
    assert any(value < -0.04 for value in relative_xs)
    assert any(value > 0.04 for value in relative_xs)
    assert any(value < -0.04 for value in relative_ys)
    assert any(value > 0.04 for value in relative_ys)
    assert max(distances) - min(distances) > 0.03
    for row in params:
        q = row["tool_start_pose"][3:7]
        assert math.isclose(math.sqrt(sum(float(item) * float(item) for item in q)), 1.0)
        assert policy["target_center_x_m"][0] <= row["target_center_world"][0] <= policy["target_center_x_m"][1]
        assert policy["target_center_y_m"][0] <= row["target_center_world"][1] <= policy["target_center_y_m"][1]
        assert policy["tool_safe_x_m"][0] <= row["tool_start_pose"][0] <= policy["tool_safe_x_m"][1]
        assert policy["tool_safe_y_m"][0] <= row["tool_start_pose"][1] <= policy["tool_safe_y_m"][1]
        assert row["tool_target_distance_m"] >= 0.0
        assert row["tool_yaw_jitter_deg"] == 0
        rel = row["tool_xy_relative_to_target_m"]
        assert math.isclose(
            math.hypot(float(rel[0]), float(rel[1])),
            float(row["tool_target_distance_m"]),
        )
        assert policy["target_yaw_deg"][0] <= row["target_yaw_deg"] <= policy["target_yaw_deg"][1]
        assert policy["tool_yaw_jitter_deg"][0] <= row["tool_yaw_jitter_deg"] <= policy["tool_yaw_jitter_deg"][1]


def test_method_task_keeps_public_target_actor_geometry_for_vlm_rollout_parity() -> None:
    seed_params = eraser_eval._sample_seed_params(3)
    task = eraser_eval._task_spec(
        task_id="unit_flat_eraser_tabletop_wipe_seed_003",
        seed=3,
        tool_refs_dir=Path("configs/tool_refs/flat_eraser"),
        seed_params=seed_params,
    )
    method_task = eraser_eval._method_facing_task(task)

    assert (
        method_task["execution"]["simtoolreal_start_pose"]
        == task["execution"]["simtoolreal_start_pose"]
    )
    actor = task["execution"]["state_bundle"]["parttoolgen_benchmark_extra_actors"][0]
    assert actor["pose_xyzw"] == seed_params["target_pose_xyzw"]
    method_actor = method_task["execution"]["state_bundle"]["parttoolgen_benchmark_extra_actors"][0]
    assert method_actor["pose_xyzw"] == seed_params["target_pose_xyzw"]
    assert method_task["metadata"]["private_scene_geometry_omitted"] is False
    assert "tool_grounding_contract" in task["metadata"]
    assert "tool_grounding_contract" not in method_task["metadata"]
    instruction = task["instruction"].lower()
    assert "eraser" in instruction
    assert "hdu pattern" in instruction
    assert "table" in instruction


def test_benchmark_config_runs_twenty_seeded_trials() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "artifacts/tool_use_benchmark/eraser_wipe/tabletop_wipe/benchmark_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["runner_args"]["trials"] == 20
    assert config["runner_args"]["rollout_max_steps"] == 1500
    contract = config["acceptance"]["tool_grounding_contract"]
    assert contract["manual_review_status"] == "approved"
    assert contract["allowed_contact_point_part_markers"] == ["4"]
    assert contract["allowed_contact_point_face_markers"] == ["F3", "F4"]
    assert contract["allowed_motion_axis_face_markers"] == ["F1", "F2", "F5", "F6"]
    assert contract["candidate_filter_policy"]["exclude_internal_tool_surfaces_from_vlm_selection"] is True
    assert (
        config["scene_initialization"]["seed_variation_policy"]["sampler"]
        == eraser_eval.SEED_VARIATION_POLICY["sampler"]
    )


def test_pending_task_contract_uses_current_approved_eraser_contract() -> None:
    contract = eraser_eval._tool_grounding_contract_from_task(
        {
            "metadata": {
                "tool_grounding_contract": {
                    "manual_review_status": "pending",
                    "notes": "historical task artifact before human review",
                }
            }
        }
    )

    assert contract["manual_review_status"] == "approved"
    assert contract["allowed_contact_point_part_markers"] == ["4"]


def test_resume_refresh_reports_explicit_selected_contact_point_face(tmp_path: Path, monkeypatch) -> None:
    compile_dir = tmp_path / "compile"
    compile_dir.mkdir()
    (compile_dir / "compile_summary.json").write_text(
        json.dumps({"path_solver_passed": True}),
        encoding="utf-8",
    )
    (compile_dir / "selected_constraint_program.json").write_text(
        json.dumps(
            {
                "tool_refs": {
                    "contact_point_face_marker": "explicit_reviewed_contact_face",
                    "contact_point_source_ref": "candidate.points.active",
                    "motion_axis_face_marker": "explicit_motion_axis_face",
                    "active_contact_point_candidate_selection": {
                        "selection_mode": "explicit_public_contact_point_selection"
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rollout_summary = tmp_path / "rollout_summary.json"
    rollout_summary.write_text(
        json.dumps({"policy_goal_passed": True, "semantic_execution_passed": True}),
        encoding="utf-8",
    )
    tool_mesh = tmp_path / "eraser_box.obj"
    _write_box_obj(tool_mesh)
    monkeypatch.setattr(eraser_eval, "TOOL_MESH", str(tool_mesh))
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "metadata": {
                    "tabletop_mark_center_world": [0.0, 0.0, 0.0],
                    "tabletop_mark_size_xyz": [0.10, 0.03, 0.006],
                    "tabletop_mark_yaw_deg": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    lifted_refs = tmp_path / "lifted_refs.json"
    lifted_refs.write_text(
        json.dumps(
            {
                "start_point": {"position_world": [-0.04, 0.0, 0.003]},
                "end_point": {"position_world": [0.04, 0.0, 0.003]},
            }
        ),
        encoding="utf-8",
    )
    tracking_samples = tmp_path / "tracking_samples.json"
    tracking_samples.write_text(
        json.dumps(
            [
                {
                    "step": step,
                    "goal_index": step,
                    "stage": f"action_waypoint_{step}",
                    "actual_pose_xyzw": [x, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0],
                }
                for step, x in enumerate([-0.035, 0.0, 0.035], start=1)
            ]
        ),
        encoding="utf-8",
    )
    result = {
        "passed": False,
        "selected_contact_point_face": "stale_face_from_prior_result",
        "artifacts": {
            "task": str(task),
            "compile_summary": str(compile_dir / "compile_summary.json"),
            "rollout_summary": str(rollout_summary),
            "lifted_refs": str(lifted_refs),
            "tracking_samples": str(tracking_samples),
        },
    }

    refreshed = eraser_eval._refresh_success_fields(result, skip_rollout=False)

    assert refreshed["passed"] is True
    assert refreshed["wipe_contact_coverage_passed"] is True
    assert refreshed["selected_contact_point_face"] == "explicit_reviewed_contact_face"
    assert refreshed["selected_contact_point_ref"] == "candidate.points.active"
    assert refreshed["selected_motion_axis_face"] == "explicit_motion_axis_face"
