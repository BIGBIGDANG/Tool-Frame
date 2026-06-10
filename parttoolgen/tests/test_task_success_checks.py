from __future__ import annotations

from parttoolgen.evaluation.task_success import (
    check_task_success,
    evaluate_eraser_wipe_surface_contact_path,
    predicate_passed,
)


def _write_box_obj(path, *, size=(0.06, 0.02, 0.02)) -> None:
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
    path.write_text(
        "\n".join(
            [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a} {b} {c}" for a, b, c in faces)]
        )
        + "\n",
        encoding="utf-8",
    )


def _wipe_task() -> dict:
    return {
        "metadata": {
            "tabletop_mark_center_world": [0.0, 0.0, 0.0],
            "tabletop_mark_size_xyz": [0.10, 0.03, 0.006],
            "tabletop_mark_yaw_deg": 0.0,
        }
    }


def _wipe_sample(*, step: int, x: float, z: float) -> dict:
    return {
        "step": step,
        "goal_index": step,
        "stage": f"action_waypoint_{step}",
        "actual_pose_xyzw": [x, 0.0, z, 0.0, 0.0, 0.0, 1.0],
    }


def _wipe_lifted_refs(*, z: float = 0.003) -> dict:
    return {
        "start_point": {"position_world": [-0.04, 0.0, z]},
        "end_point": {"position_world": [0.04, 0.0, z]},
    }


def test_brush_sweep_success_uses_directional_displacement_not_tracking() -> None:
    success_check = check_task_success(
        "brush_sweep",
        effect={
            "target_effect_passed": True,
            "target_xy_displacement_m": 0.11,
            "projected_displacement_m": 0.10,
            "direction_cosine": 0.91,
        },
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": False,
            "semantic_execution_passed": True,
        },
        execution_report=None,
    )

    assert success_check["passed"] is True
    assert predicate_passed(success_check, "target_directional_displacement") is True
    assert success_check["policy"]["requires_tracking_execution_passed"] is False


def test_brush_sweep_red_zone_success_uses_target_region_not_contact_gate() -> None:
    success_check = check_task_success(
        "brush_sweep",
        effect={
            "target_effect_passed": True,
            "target_region_passed": True,
            "target_final_xy_world": [0.257, -0.206],
            "target_region": {
                "center_xy_world": [0.24, -0.205],
                "size_xy_m": [0.14, 0.10],
                "bounds_xy_world": {
                    "x": [0.17, 0.31],
                    "y": [-0.255, -0.155],
                },
            },
        },
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
        },
        execution_report={"semantic_execution_passed": False},
    )

    assert success_check["passed"] is True
    assert success_check["policy"]["name"] == "brush_sweep_target_zone_occupancy"
    assert predicate_passed(success_check, "target_region_occupancy") is True
    assert success_check["policy"]["requires_semantic_execution_passed"] is False


def test_brush_sweep_off_table_success_uses_table_exit_not_displacement_threshold() -> None:
    success_check = check_task_success(
        "brush_sweep",
        effect={
            "target_effect_passed": True,
            "off_table_passed": True,
            "target_final_xy_world": [0.10, -0.285],
            "off_table_region": {
                "bounds_xy_world": {
                    "x": [-0.35, 0.35],
                    "y": [-0.25, 0.25],
                },
                "table_top_z_world": 0.53,
            },
        },
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
        },
        execution_report={"semantic_execution_passed": False},
    )

    assert success_check["passed"] is True
    assert success_check["policy"]["name"] == "brush_sweep_off_table"
    assert predicate_passed(success_check, "target_off_table") is True
    assert success_check["policy"]["requires_semantic_execution_passed"] is False


def test_brush_sweep_success_fails_without_target_effect() -> None:
    success_check = check_task_success(
        "brush_sweep",
        effect={
            "target_effect_passed": False,
            "target_xy_displacement_m": 0.01,
            "projected_displacement_m": 0.0,
            "direction_cosine": 0.0,
        },
        rollout_summary={"tracking_execution_passed": True},
        execution_report=None,
    )

    assert success_check["passed"] is False


def test_eraser_wipe_surface_contact_coverage_is_task_success() -> None:
    success_check = check_task_success(
        "eraser_wipe",
        compile_summary={"path_solver_passed": True},
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
        },
        wipe_contact={
            "passed": True,
            "coverage_fraction": 0.93,
            "contact_sample_count": 5,
        },
        lifted_refs=_wipe_lifted_refs(),
        task=_wipe_task(),
        execution_report=None,
    )

    assert success_check["passed"] is True
    assert predicate_passed(success_check, "contact_coverage") is True
    assert predicate_passed(success_check, "lifted_target_surface_path") is True
    assert predicate_passed(success_check, "semantic_contact_path") is False
    assert success_check["policy"]["surface_contact_coverage_is_diagnostic"] is False
    assert success_check["policy"]["contact_coverage_is_task_success"] is True
    assert success_check["policy"]["requires_visual_patch_disappearance"] is False
    assert success_check["policy"]["requires_tracking_execution_passed"] is False
    assert success_check["policy"]["requires_semantic_contact_path"] is False


def test_eraser_wipe_success_requires_contact_coverage_not_semantic_policy_path() -> None:
    success_check = check_task_success(
        "flat_eraser_tabletop_wipe",
        compile_summary={"path_solver_passed": True},
        rollout_summary={
            "policy_goal_passed": True,
            "semantic_execution_passed": True,
        },
        wipe_contact={
            "passed": False,
            "failure_reason": "insufficient_target_patch_coverage",
            "coverage_fraction": 0.22,
            "contact_sample_count": 5,
        },
        lifted_refs=_wipe_lifted_refs(),
        task=_wipe_task(),
        execution_report=None,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "contact_coverage") is False
    assert predicate_passed(success_check, "policy_goal_path") is True
    assert predicate_passed(success_check, "semantic_contact_path") is True


def test_eraser_wipe_success_ignores_semantic_without_contact_coverage() -> None:
    success_check = check_task_success(
        "flat_eraser_tabletop_wipe",
        compile_summary={"path_solver_passed": True},
        rollout_summary={"semantic_execution_passed": True},
        wipe_contact={
            "passed": False,
            "failure_reason": "no_surface_contact_on_target_patch",
            "coverage_fraction": 0.0,
        },
        lifted_refs=_wipe_lifted_refs(),
        task=_wipe_task(),
        execution_report=None,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "semantic_contact_path") is True
    assert predicate_passed(success_check, "contact_coverage") is False


def test_eraser_wipe_success_fails_when_lifted_path_height_is_wrong() -> None:
    success_check = check_task_success(
        "flat_eraser_tabletop_wipe",
        compile_summary={"path_solver_passed": True},
        rollout_summary={
            "policy_goal_passed": True,
            "semantic_execution_passed": True,
        },
        wipe_contact={
            "passed": False,
            "failure_reason": "no_surface_contact_on_target_patch",
            "coverage_fraction": 0.0,
        },
        lifted_refs=_wipe_lifted_refs(z=0.040),
        task=_wipe_task(),
        execution_report=None,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "contact_coverage") is False
    assert predicate_passed(success_check, "lifted_target_surface_path") is False
    assert (
        success_check["diagnostics"]["lifted_target_surface_path"]["failure_reason"]
        == "lifted_points_wrong_surface_height"
    )


def test_eraser_wipe_skip_rollout_is_missing_required_evidence() -> None:
    success_check = check_task_success(
        "eraser_wipe",
        compile_summary={"path_solver_passed": True},
        rollout_summary=None,
        execution_report=None,
        skip_rollout=True,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "rollout_evidence_present") is False
    assert success_check["policy"]["requires_contact_coverage"] is True


def test_eraser_wipe_surface_contact_metric_requires_tool_on_patch(tmp_path) -> None:
    mesh = tmp_path / "eraser_box.obj"
    _write_box_obj(mesh)
    contact_z = 0.003
    metrics = evaluate_eraser_wipe_surface_contact_path(
        task=_wipe_task(),
        tracking_samples=[
            _wipe_sample(step=1, x=-0.035, z=contact_z),
            _wipe_sample(step=2, x=0.0, z=contact_z),
            _wipe_sample(step=3, x=0.035, z=contact_z),
        ],
        tool_mesh_path=mesh,
    )

    assert metrics["passed"] is True
    assert metrics["coverage_fraction"] >= 0.30

    hovering = evaluate_eraser_wipe_surface_contact_path(
        task=_wipe_task(),
        tracking_samples=[
            _wipe_sample(step=1, x=-0.035, z=0.03),
            _wipe_sample(step=2, x=0.0, z=0.03),
            _wipe_sample(step=3, x=0.035, z=0.03),
        ],
        tool_mesh_path=mesh,
    )

    assert hovering["passed"] is False
    assert hovering["failure_reason"] == "no_surface_contact_on_target_patch"


def test_screwdriver_insert_success_uses_tip_path_not_tracking() -> None:
    success_check = check_task_success(
        "screwdriver_insert",
        compile_summary={"path_solver_passed": True},
        insertion={
            "tip_xy_radial_error_m": 0.004,
            "tip_insertion_depth_m": 0.016,
            "tip_end_3d_error_m": 0.019,
            "axis_cosine": 0.97,
        },
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": False,
            "semantic_execution_passed": True,
        },
        execution_report=None,
    )

    assert success_check["passed"] is True
    assert predicate_passed(success_check, "semantic_insertion_path") is True
    assert predicate_passed(success_check, "tip_radial_alignment") is True
    assert predicate_passed(success_check, "tip_insertion_depth") is True
    assert success_check["policy"]["requires_tracking_execution_passed"] is False


def test_screwdriver_insert_success_does_not_treat_end_3d_error_as_radial() -> None:
    success_check = check_task_success(
        "screwdriver_insert",
        compile_summary={"path_solver_passed": True},
        insertion={
            "tip_xy_radial_error_m": 0.004,
            "tip_radial_error_m": 0.004,
            "tip_insertion_depth_m": 0.017,
            "tip_end_3d_error_m": 0.018,
            "axis_cosine": 0.99,
        },
        rollout_summary={"semantic_execution_passed": True},
        execution_report=None,
        max_tip_radial_error_m=0.012,
        min_tip_insertion_depth_m=0.012,
    )

    assert success_check["passed"] is True
    assert predicate_passed(success_check, "tip_radial_alignment") is True
    assert success_check["diagnostics"]["tip_end_3d_error_m"] == 0.018


def test_screwdriver_insert_success_requires_depth_when_rollout_runs() -> None:
    success_check = check_task_success(
        "long_screwdriver_deep_socket_insert",
        compile_summary={"path_solver_passed": True},
        insertion={
            "tip_radial_error_m": 0.003,
            "tip_insertion_depth_m": 0.008,
            "axis_cosine": 0.99,
        },
        rollout_summary={"semantic_execution_passed": True},
        execution_report=None,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "tip_insertion_depth") is False


def test_screwdriver_insert_skip_rollout_is_missing_required_evidence() -> None:
    success_check = check_task_success(
        "screwdriver_insert",
        compile_summary={"path_solver_passed": True},
        insertion=None,
        rollout_summary=None,
        execution_report=None,
        skip_rollout=True,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "rollout_evidence_present") is False
    assert success_check["policy"]["requires_tip_insertion_depth"] is True
