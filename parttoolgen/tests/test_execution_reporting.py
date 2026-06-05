from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


def _load_reporting_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "execution" / "reporting.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_execution_reporting", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporting = _load_reporting_module()


def _fixed_goals() -> dict:
    return {
        "start_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "goals": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "metadata": {
            "schema_version": "v0.1",
            "source_subgoals_id": "unit_subgoals",
            "target_api": "fixed_goal_states_json",
            "coordinate_frame": "world",
            "units": "meters",
            "quaternion_order": "xyzw",
            "generated_by": "parttoolgen_simtoolreal_adapter",
        },
    }


def _subgoals() -> dict:
    return {
        "schema_version": "v0.2",
        "subgoals_id": "unit_subgoals",
        "tool_id": "tool",
        "task_id": "task",
        "world_frame": "world",
        "units": "meters",
        "quaternion_order": "xyzw",
        "start_pose": {
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "goals": [
            {
                "index": 0,
                "stage": "contact",
                "pose_world_tool": {
                    "position": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "duration_sec": 0.2,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["contact_constraint"],
                "expected_contact": {
                    "tool_part": "candidate.points.part_004_contact",
                    "scene_target": "scene.keypoints.target_contact_point",
                },
                "contact_point_goal": {
                    "part_ref": "candidate.points.part_004_contact",
                    "position_world": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
        "source_trace": {
            "observed_scene_graph": {"id": "scene", "path": "<in-memory>"},
            "grounding_candidates": {"id": "candidates", "path": "<in-memory>"},
            "functional_tool_parts": {"id": "tool", "path": "<in-memory>"},
            "planner_program": {"id": "planner", "path": "<in-memory>"},
            "program_ir": {"id": "program", "path": "<in-memory>"},
        },
        "sanity_checks": {"compiled": True},
        "metadata": {
            "generated_by": "unit_test",
            "source": "unit_test",
            "geometry_mode": "observed",
        },
    }


def _sample(*, functional_error: float | None) -> dict:
    sample = {
        "step": 5,
        "goal_index": 0,
        "stage": "contact",
        "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    if functional_error is not None:
        sample["semantic_contact_check"] = {
            "checked": True,
            "tool_part_ref": "candidate.points.part_004_contact",
            "scene_target_ref": "scene.keypoints.target_contact_point",
            "semantic_contact_error_m": functional_error,
            "threshold_m": 0.03,
        }
    return sample


def test_policy_success_does_not_hide_functional_contact_failure() -> None:
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[_sample(functional_error=0.08)],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is False
    assert report["tracking_execution_passed"] is True
    assert report["semantic_execution_passed"] is False
    assert report["failed_stage_reason"] == "semantic_contact_failure"
    assert report["semantic_failed_stage"] == "contact"
    stage = report["stage_reports"][0]
    assert stage["success"] is True
    assert stage["policy_goal_success"] is True
    assert stage["semantic_contact_success"] is False
    assert stage["semantic_gate_passed"] is False
    assert stage["semantic_failure_is_repair_evidence"] is True
    assert "candidate.points.part_004_contact" in report["failed_refs"]


def test_functional_contact_passes_when_policy_and_semantic_goal_agree() -> None:
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[_sample(functional_error=0.01)],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is True
    assert report["stage_reports"][0]["semantic_contact_success"] is True


def test_functional_contact_uses_best_sample_within_goal() -> None:
    early_contact = _sample(functional_error=0.01)
    early_contact["step"] = 5
    later_lifted = _sample(functional_error=0.08)
    later_lifted["step"] = 6

    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[early_contact, later_lifted],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    stage = report["stage_reports"][0]
    assert report["passed"] is True
    assert stage["semantic_contact_success"] is True
    assert stage["semantic_success_policy"] == "best_sample_within_goal"
    assert stage["semantic_contact_error_m"] == 0.01
    assert stage["last_semantic_contact_error_m"] == 0.08
    assert stage["best_semantic_contact_sample_step"] == 5


def test_functional_contact_can_be_observed_after_fixed_goal_switch() -> None:
    fixed_goals = _fixed_goals()
    fixed_goals["goals"] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    subgoals = _subgoals()
    second_goal = deepcopy(subgoals["goals"][0])
    second_goal["index"] = 1
    second_goal["stage"] = "contact_followthrough"
    second_goal["pose_world_tool"]["position"] = [1.1, 0.0, 0.0]
    subgoals["goals"].append(second_goal)
    early_miss = _sample(functional_error=0.08)
    early_miss["step"] = 5
    early_miss["goal_index"] = 0
    later_contact = _sample(functional_error=0.01)
    later_contact["step"] = 6
    later_contact["goal_index"] = 1
    later_contact["stage"] = "contact_followthrough"
    later_contact["actual_pose_xyzw"] = [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    report = reporting.build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=subgoals,
        tracking_samples=[early_miss, later_contact],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            },
            {
                "goal_index": 1,
                "stage": "contact_followthrough",
                "type": "simtoolreal_goal_success",
            },
        ],
        max_contact_point_target_error_m=0.03,
    )

    first_stage = report["stage_reports"][0]
    assert report["passed"] is True
    assert first_stage["semantic_contact_success"] is True
    assert first_stage["semantic_success_policy"] == "best_sample_for_expected_contact"
    assert first_stage["semantic_contact_error_m"] == 0.01
    assert first_stage["last_semantic_contact_error_m"] == 0.08


def test_policy_success_does_not_hide_rotation_tracking_failure() -> None:
    bad_pose = _sample(functional_error=0.01)
    bad_pose["actual_pose_xyzw"] = [1.0, 0.0, 0.0, 0.0, 0.70710678, 0.0, 0.70710678]

    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[bad_pose],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_rotation_error_rad=0.30,
        max_contact_point_target_error_m=0.03,
    )

    stage = report["stage_reports"][0]
    assert report["passed"] is False
    assert report["tracking_execution_passed"] is False
    assert report["semantic_execution_passed"] is True
    assert stage["policy_goal_success"] is True
    assert stage["tracking_success"] is False
    assert stage["tracking_gate_required"] is True
    assert stage["tracking_gate_success"] is False
    assert stage["success"] is False
    assert stage["rotation_error_rad"] > 0.30


def test_missing_semantic_evidence_for_expected_contact_is_failure() -> None:
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[_sample(functional_error=None)],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
    )

    assert report["passed"] is False
    assert report["tracking_execution_passed"] is True
    assert report["semantic_execution_passed"] is False
    assert report["stage_reports"][0]["success"] is True
    assert (
        report["stage_reports"][0]["semantic_failure_reason"]
        == "semantic_contact_verification_missing"
    )


def test_generated_transition_waypoint_does_not_require_semantic_contact() -> None:
    subgoals = _subgoals()
    subgoals["goals"][0]["stage"] = "approach_transition_1"
    subgoals["goals"][0]["semantic_role"] = "generated_transition_waypoint"
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=subgoals,
        tracking_samples=[
            {
                "step": 5,
                "goal_index": 0,
                "stage": "approach_transition_1",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.08,
                "semantic_gate": False,
            }
        ],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "approach_transition_1",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is True
    assert report["stage_reports"][0]["semantic_contact_required"] is False
    assert report["stage_reports"][0]["semantic_gate"] is False


def test_generated_transition_does_not_require_semantic_contact() -> None:
    subgoals = _subgoals()
    subgoals["goals"][0]["stage"] = "approach_transition_3"
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=subgoals,
        tracking_samples=[
            {
                "step": 5,
                "goal_index": 0,
                "stage": "approach_transition_3",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.08,
            }
        ],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "approach_transition_3",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is True
    assert report["stage_reports"][0]["semantic_contact_required"] is False
    assert report["stage_reports"][0]["semantic_gate"] is False


def test_named_semantic_waypoint_still_requires_contact_check() -> None:
    subgoals = _subgoals()
    subgoals["goals"][0]["stage"] = "action_waypoint_4"
    report = reporting.build_execution_report(
        fixed_goals=_fixed_goals(),
        tool_pose_subgoals=subgoals,
        tracking_samples=[
            {
                "step": 5,
                "goal_index": 0,
                "stage": "action_waypoint_4",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.08,
            }
        ],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "action_waypoint_4",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is False
    assert report["tracking_execution_passed"] is True
    assert report["semantic_execution_passed"] is False
    assert report["failed_stage_reason"] == "semantic_contact_failure"
    assert report["stage_reports"][0]["success"] is True
    assert report["stage_reports"][0]["semantic_contact_required"] is True
    assert report["stage_reports"][0]["semantic_gate"] is True


def test_semantic_contact_failure_does_not_override_tracking_failed_stage() -> None:
    fixed_goals = _fixed_goals()
    fixed_goals["goals"] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    subgoals = _subgoals()
    second_goal = deepcopy(subgoals["goals"][0])
    second_goal["index"] = 1
    second_goal["stage"] = "action_followthrough"
    second_goal["pose_world_tool"]["position"] = [1.1, 0.0, 0.0]
    subgoals["goals"].append(second_goal)

    first_sample = _sample(functional_error=0.08)
    report = reporting.build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=subgoals,
        tracking_samples=[first_sample],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            }
        ],
        max_contact_point_target_error_m=0.03,
    )

    assert report["passed"] is False
    assert report["tracking_execution_passed"] is False
    assert report["semantic_execution_passed"] is False
    assert report["failed_stage"] == "action_followthrough"
    assert report["execution_failed_stage"] == "action_followthrough"
    assert report["semantic_failed_stage"] == "contact"
    assert report["failed_stage_reason"] == "tracking_failure"
    assert report["stage_reports"][0]["success"] is True
    assert report["stage_reports"][0]["semantic_contact_success"] is False
    assert report["stage_reports"][1]["success"] is False


def test_start_hold_goal_index_map_does_not_require_semantic_contact() -> None:
    fixed_goals = _fixed_goals()
    fixed_goals["goals"] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    report = reporting.build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[
            {
                "step": 1,
                "goal_index": 1,
                "stage": "contact",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.01,
            }
        ],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "start_hold",
                "type": "simtoolreal_goal_success",
            },
            {
                "goal_index": 1,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            },
        ],
        goal_index_map=[
            {
                "fixed_goal_index": 0,
                "source_goal_index": None,
                "stage": "start_hold",
            },
            {
                "fixed_goal_index": 1,
                "source_goal_index": 0,
                "stage": "contact",
            },
        ],
    )

    assert report["passed"] is True
    assert report["stage_reports"][0]["stage"] == "start_hold"
    assert report["stage_reports"][0]["semantic_contact_required"] is False


def test_start_hold_goal_index_map_is_inferred_from_fixed_goals() -> None:
    fixed_goals = _fixed_goals()
    fixed_goals["goals"] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    report = reporting.build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[
            {
                "step": 1,
                "goal_index": 1,
                "stage": "contact",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.01,
            }
        ],
        contact_events=[
            {
                "goal_index": 0,
                "stage": "start_hold",
                "type": "simtoolreal_goal_success",
            },
            {
                "goal_index": 1,
                "stage": "contact",
                "type": "simtoolreal_goal_success",
            },
        ],
    )

    assert report["passed"] is True
    assert report["stage_reports"][0]["stage"] == "start_hold"
    assert report["stage_reports"][0]["semantic_contact_required"] is False
    assert report["metadata"]["goal_index_map_applied"] is True


def test_metadata_goal_index_map_is_used_before_shape_inference() -> None:
    fixed_goals = _fixed_goals()
    fixed_goals["goals"] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    fixed_goals["metadata"]["goal_index_map"] = [
        {
            "fixed_goal_index": 0,
            "source_goal_index": None,
            "stage": "start_hold",
            "source": "simtoolreal_fixed_goals.goals[0]",
        },
        {
            "fixed_goal_index": 1,
            "source_goal_index": None,
            "stage": "contact_transition_1",
            "source": "explicit_test_map",
        },
        {
            "fixed_goal_index": 2,
            "source_goal_index": 0,
            "stage": "contact",
            "source": "tool_pose_subgoals.goals",
        },
    ]
    report = reporting.build_execution_report(
        fixed_goals=fixed_goals,
        tool_pose_subgoals=_subgoals(),
        tracking_samples=[
            {
                "step": 2,
                "goal_index": 1,
                "stage": "contact_transition_1",
                "actual_pose_xyzw": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.08,
                "semantic_gate": False,
            },
            {
                "step": 5,
                "goal_index": 2,
                "stage": "contact",
                "actual_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "functional_target_error_m": 0.01,
            },
        ],
        contact_events=[
            {"goal_index": 0, "stage": "start_hold", "type": "simtoolreal_goal_success"},
            {
                "goal_index": 1,
                "stage": "contact_transition_1",
                "type": "simtoolreal_goal_success",
            },
            {"goal_index": 2, "stage": "contact", "type": "simtoolreal_goal_success"},
        ],
    )

    assert report["passed"] is True
    assert report["stage_reports"][1]["stage"] == "contact_transition_1"
    assert report["stage_reports"][1]["semantic_contact_required"] is False
    assert report["stage_reports"][2]["stage"] == "contact"
