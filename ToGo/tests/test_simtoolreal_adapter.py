from __future__ import annotations

import pytest

from ToGo.adapters.simtoolreal_adapter import (
    SimToolRealAdapterError,
    convert_subgoals_to_simtoolreal,
)


def _subgoals(*, deferred_path_objectives: int = 0, tool_path_solver: dict | None = None) -> dict:
    sanity_checks = {
        "compiled": True,
        "path_objectives_deferred": deferred_path_objectives,
    }
    if tool_path_solver is not None:
        sanity_checks["tool_path_solver"] = tool_path_solver
    return {
        "schema_version": "v0.2",
        "subgoals_id": "subgoals",
        "tool_id": "tool",
        "task_id": "task",
        "world_frame": "world",
        "units": "meters",
        "quaternion_order": "xyzw",
        "start_pose": {"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        "goals": [
            {
                "index": 0,
                "stage": "move",
                "pose_world_tool": {
                    "position": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["move"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            }
        ],
        "source_trace": {
            "observed_scene_graph": {"id": "scene", "path": "<in-memory>"},
            "grounding_candidates": {"id": "<none>", "path": "<none>"},
            "functional_tool_parts": {"id": "tool", "path": "<in-memory>"},
            "planner_program": {"id": "planner", "path": "<in-memory>"},
            "program_ir": {"id": "program", "path": "<in-memory>"},
        },
        "sanity_checks": sanity_checks,
        "metadata": {
            "generated_by": "unit_test",
            "source": "unit_test",
            "geometry_mode": "observed",
        },
    }


def test_adapter_rejects_deferred_path_objectives_without_path_gate() -> None:
    with pytest.raises(SimToolRealAdapterError, match="tool_path_solver"):
        convert_subgoals_to_simtoolreal(_subgoals(deferred_path_objectives=1))


def test_adapter_accepts_deferred_path_objectives_after_path_gate() -> None:
    fixed_goals = convert_subgoals_to_simtoolreal(
        _subgoals(
            deferred_path_objectives=1,
            tool_path_solver={
                "passed": True,
                "semantic_goal_preserved": True,
                "path_objectives_checked": True,
                "path_objectives_passed": True,
            },
        )
    )

    assert fixed_goals["goals"] == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    assert fixed_goals["metadata"]["goal_index_map"] == [
        {
            "fixed_goal_index": 0,
            "source_goal_index": 0,
            "stage": "move",
            "source": "tool_pose_subgoals.goals",
        },
    ]


def test_adapter_accepts_subgoals_without_deferred_path_objectives() -> None:
    fixed_goals = convert_subgoals_to_simtoolreal(_subgoals())

    assert fixed_goals["start_pose"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert fixed_goals["goals"][0] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert fixed_goals["metadata"]["start_goal_in_fixed_goals"] is False


def test_adapter_preserves_post_terminal_goal_when_not_truncating() -> None:
    subgoals = _subgoals()
    subgoals["goals"].extend(
        [
            {
                "index": 1,
                "stage": "impact",
                "semantic_role": "swing_terminal",
                "semantic_gate": True,
                "pose_world_tool": {
                    "position": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["impact"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
            {
                "index": 2,
                "stage": "post_terminal_clearance",
                "semantic_role": "post_terminal_clearance",
                "semantic_gate": False,
                "pose_world_tool": {
                    "position": [3.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["post_terminal_clearance"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [3.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
        ]
    )

    fixed_goals = convert_subgoals_to_simtoolreal(subgoals)

    assert fixed_goals["goals"] == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    assert fixed_goals["metadata"]["task_terminal_source_goal_index"] == 1
    assert fixed_goals["metadata"]["task_terminal_stage"] == "impact"
    assert fixed_goals["metadata"]["truncated_post_terminal_goals"] == []


def test_adapter_ignores_generated_transition_waypoint_as_task_terminal() -> None:
    subgoals = _subgoals()
    subgoals["goals"].extend(
        [
            {
                "index": 1,
                "stage": "impact_transition_1",
                "semantic_role": "generated_transition_waypoint",
                "semantic_gate": True,
                "pose_world_tool": {
                    "position": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["impact"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
            {
                "index": 2,
                "stage": "impact",
                "semantic_role": "swing_terminal",
                "semantic_gate": True,
                "pose_world_tool": {
                    "position": [3.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["impact"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [3.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            },
        ]
    )

    fixed_goals = convert_subgoals_to_simtoolreal(subgoals)

    assert fixed_goals["goals"] == [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    assert fixed_goals["metadata"]["task_terminal_source_goal_index"] == 2
    assert fixed_goals["metadata"]["task_terminal_stage"] == "impact"
    assert fixed_goals["metadata"]["truncated_post_terminal_goals"] == []


def test_adapter_rejects_semantic_drift_even_without_deferred_path_objectives() -> None:
    with pytest.raises(SimToolRealAdapterError, match="semantic_goal_preserved=true"):
        convert_subgoals_to_simtoolreal(
            _subgoals(
                tool_path_solver={
                    "passed": False,
                    "semantic_goal_preserved": False,
                    "path_objectives_checked": False,
                    "path_objectives_passed": True,
                }
            )
        )


def test_adapter_rejects_incomplete_tool_path_gate() -> None:
    with pytest.raises(SimToolRealAdapterError, match="semantic_goal_preserved=true"):
        convert_subgoals_to_simtoolreal(
            _subgoals(
                deferred_path_objectives=1,
                tool_path_solver={
                    "passed": True,
                    "path_objectives_checked": True,
                    "path_objectives_passed": True,
                },
            )
        )
