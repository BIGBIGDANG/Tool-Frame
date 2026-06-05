from __future__ import annotations

from parttoolgen.evaluation.failure_attribution import (
    MODULE_FAILURE_CATEGORIES,
    attribute_eraser_wipe_failure,
    failure_category_counts,
)


def _task() -> dict:
    return {
        "metadata": {
            "tabletop_hdu_sticker_center_world": [0.0, 0.0, 0.0],
            "tabletop_hdu_sticker_size_xyz": [0.10, 0.03, 0.006],
            "tabletop_hdu_sticker_yaw_deg": 0.0,
        }
    }


def _scene(label: str = "HDU sticker", description: str = "the visible white HDU sticker on the tabletop") -> dict:
    return {
        "target_object": {
            "label": label,
            "description": description,
        },
        "start_point": {"u": 100.0, "v": 100.0},
        "end_point": {"u": 200.0, "v": 100.0},
        "action_direction": {
            "start": {"u": 100.0, "v": 100.0},
            "end": {"u": 200.0, "v": 100.0},
        },
    }


def _lifted(start=None, goal=None) -> dict:
    return {
        "start": {
            "position_world": start if start is not None else [-0.049, 0.0, 0.003],
        },
        "end": {
            "position_world": goal if goal is not None else [0.049, 0.0, 0.003],
        },
    }


def _subgoals() -> dict:
    return {
        "goals": [
            {
                "stage": "action_waypoint_1",
                "semantic_role": "contact_point_approach",
                "contact_point_goal": {"position_world": [-0.049, 0.0, 0.003]},
            },
            {
                "stage": "action_waypoint_2",
                "semantic_role": "contact_follow_terminal",
                "contact_point_goal": {"position_world": [0.049, 0.0, 0.003]},
            },
        ]
    }


def _tool_refs() -> dict:
    return {
        "contact_point_part_ref": "tool.parts.synthetic_contact_pad",
        "contact_point_face_marker": "contact_face_b",
        "contact_point_source_ref": "candidate.points.synthetic_contact_pad_y_positive",
        "contact_point_surface_ref": "candidate.surfaces.synthetic_contact_pad_y_positive",
        "motion_axis_face_marker": "axis_face_a",
        "motion_axis_surface_ref": "candidate.surfaces.synthetic_contact_pad_x_negative",
    }


def _tool_contract() -> dict:
    return {
        "manual_review_status": "approved",
        "allowed_contact_point_part_refs": ["tool.parts.synthetic_contact_pad"],
        "allowed_contact_point_face_markers": ["contact_face_a", "contact_face_b"],
        "allowed_contact_point_source_refs": [
            "candidate.points.synthetic_contact_pad_y_negative",
            "candidate.points.synthetic_contact_pad_y_positive",
        ],
        "allowed_contact_point_surface_refs": [
            "candidate.surfaces.synthetic_contact_pad_y_negative",
            "candidate.surfaces.synthetic_contact_pad_y_positive",
        ],
        "allowed_motion_axis_face_markers": ["axis_face_a", "axis_face_b"],
        "allowed_motion_axis_surface_refs": [
            "candidate.surfaces.synthetic_contact_pad_x_negative",
            "candidate.surfaces.synthetic_contact_pad_x_positive",
        ],
    }


def _success_check(*, passed: bool, reason: str | None = None) -> dict:
    return {
        "passed": passed,
        "diagnostics": {
            "wipe_contact": {
                "passed": passed,
                "failure_reason": reason,
                "coverage_fraction": 1.0 if passed else 0.0,
                "contact_sample_count": 5 if passed else 0,
            }
        },
    }


def test_eraser_failure_attribution_separates_rollout_timeout_from_good_vlm() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 22,
            "num_goals": 25,
            "num_steps": 2500,
        },
        success_check=_success_check(passed=False, reason="insufficient_target_patch_coverage"),
        tracking_samples=[{"stage": "action_waypoint_4"}],
        selected_tool_refs=_tool_refs(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "policy_tracking_failure"
    assert "policy_tracking_failure" in MODULE_FAILURE_CATEGORIES
    assert attribution["layers"]["vlm_scene_grounding"]["passed"] is True
    assert attribution["layers"]["rgbd_lifting"]["passed"] is True


def test_eraser_failure_attribution_detects_vlm_scene_selection_error() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(
            label="black eraser",
            description="the black eraser tool on the tabletop",
        ),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={"policy_goal_passed": False, "num_steps": 1},
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=_tool_refs(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "vlm_scene_marking_error"


def test_eraser_failure_attribution_detects_vlm_scene_marking_error() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(start=[-0.049, 0.10, 0.003], goal=[0.049, 0.10, 0.003]),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={"policy_goal_passed": False, "num_steps": 1},
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=_tool_refs(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "vlm_scene_marking_error"
    assert (
        attribution["layers"]["rgbd_lifting"]["failure_reason"]
        == "lifted_points_outside_target_patch_xy"
    )


def test_eraser_failure_attribution_detects_policy_tracking_failure() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": False,
            "semantic_execution_passed": True,
            "final_successes": 10,
            "num_goals": 10,
            "num_steps": 500,
        },
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=_tool_refs(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "policy_tracking_failure"


def test_eraser_failure_attribution_detects_vlm_tool_selection_error() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 0,
            "num_goals": 10,
            "num_steps": 10,
        },
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs={"contact_point_face_marker": None},
        tool_grounding_contract=_tool_contract(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "vlm_tool_selection_error"


def test_eraser_failure_attribution_records_wrong_tool_label_as_audit_only() -> None:
    wrong_refs = _tool_refs()
    wrong_refs["contact_point_face_marker"] = "axis_face_a"
    wrong_refs["contact_point_source_ref"] = "candidate.points.synthetic_handle"
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 0,
            "num_goals": 10,
            "num_steps": 10,
        },
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=wrong_refs,
        tool_grounding_contract=_tool_contract(),
        max_steps=2500,
    )

    tool_layer = attribution["layers"]["tool_grounding"]
    assert attribution["primary_category"] == "policy_tracking_failure"
    assert tool_layer["passed"] is True
    assert tool_layer["failure_reason"] is None
    assert tool_layer["audit_policy"] == "diagnostic_only_not_used_for_selection_or_task_success"
    assert tool_layer["audit_label_passed"] is False
    assert any(not row["passed"] for row in tool_layer["label_checks"])


def test_eraser_failure_attribution_does_not_label_check_pending_contract() -> None:
    pending_contract = _tool_contract()
    pending_contract["manual_review_status"] = "pending"
    wrong_refs = _tool_refs()
    wrong_refs["contact_point_face_marker"] = "unreviewed_face"

    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 0,
            "num_goals": 10,
            "num_steps": 10,
        },
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=wrong_refs,
        tool_grounding_contract=pending_contract,
        max_steps=2500,
    )

    tool_layer = attribution["layers"]["tool_grounding"]
    assert tool_layer["contract_checked"] is False
    assert tool_layer["label_checks"] == []
    assert tool_layer["failure_reason"] is None


def test_eraser_failure_attribution_allows_empty_approved_audit_contract() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals=_subgoals(),
        rollout_summary={
            "policy_goal_passed": False,
            "tracking_execution_passed": False,
            "semantic_execution_passed": False,
            "final_successes": 0,
            "num_goals": 10,
            "num_steps": 10,
        },
        success_check=_success_check(passed=False, reason="no_surface_contact_on_target_patch"),
        selected_tool_refs=_tool_refs(),
        tool_grounding_contract={"manual_review_status": "approved"},
        max_steps=2500,
    )

    tool_layer = attribution["layers"]["tool_grounding"]
    assert attribution["primary_category"] == "policy_tracking_failure"
    assert tool_layer["passed"] is True
    assert tool_layer["contract_checked"] is False
    assert tool_layer["audit_label_passed"] is None
    assert tool_layer["failure_reason"] is None


def test_eraser_failure_attribution_detects_compiler_subgoal_error() -> None:
    attribution = attribute_eraser_wipe_failure(
        task=_task(),
        scene_keypoints=_scene(),
        lifted_refs=_lifted(),
        compile_summary={"path_solver_passed": True},
        tool_pose_subgoals={
            "goals": [
                {
                    "stage": "action_waypoint_1",
                    "semantic_role": "contact_point_approach",
                    "contact_point_goal": {"position_world": [0.0, 0.0, 0.003]},
                }
            ]
        },
        rollout_summary={
            "policy_goal_passed": True,
            "tracking_execution_passed": True,
            "semantic_execution_passed": True,
            "final_successes": 1,
            "num_goals": 1,
            "num_steps": 10,
        },
        success_check=_success_check(passed=False, reason="insufficient_target_patch_coverage"),
        selected_tool_refs=_tool_refs(),
        max_steps=2500,
    )

    assert attribution["primary_category"] == "compiler_subgoal_error"
    assert (
        attribution["layers"]["subgoal_generation"]["failure_reason"]
        == "subgoal_action_contact_path_span_too_short"
    )


def test_failure_category_counts_uses_attribution() -> None:
    counts = failure_category_counts(
        [
            {"failure_attribution": {"primary_category": "success"}},
            {"failure_attribution": {"primary_category": "vlm_scene_marking_error"}},
            {"failure_attribution": {"primary_category": "vlm_scene_marking_error"}},
            {"passed": False},
        ]
    )

    assert counts == {
        "needs_review": 1,
        "success": 1,
        "vlm_scene_marking_error": 2,
    }
