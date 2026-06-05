from parttoolgen.program.repair import (
    build_failure_report,
    render_repair_prompt,
    summarize_tool_path_report,
)


def _tool_path_report() -> dict:
    return {
        "schema_version": "v0.1",
        "passed": False,
        "changed": True,
        "semantic_goal_preserved": False,
        "path_objectives_checked": True,
        "path_objectives_passed": True,
        "path_objective_ids": ["move_objective_1"],
        "warnings": ["goal 0 was path-objective-clipped"],
        "repair_hints": [{"goal_index": 0, "hint": "Change ProgramIR refs"}],
        "path_objective_violations": [
            {
                "objective_id": "move_objective_1",
                "stage": "move",
                "goal_index": 0,
                "sample_fraction": 0.75,
                "residual": 0.2,
                "tolerance": 0.0001,
                "pose_world_tool": {"position": [1.2, 0.0, 0.0]},
            }
        ],
        "path_objective_failures": [
            {
                "goal_index": 0,
                "stage": "move",
                "safe_fraction": 0.5,
                "safe_pose_world_tool": {"position": [1.0, 0.0, 0.0]},
                "path_objective_violations": [
                    {
                        "objective_id": "move_objective_1",
                        "stage": "move",
                        "goal_index": 0,
                        "sample_fraction": 0.75,
                        "residual": 0.2,
                        "tolerance": 0.0001,
                        "pose_world_tool": {"position": [1.2, 0.0, 0.0]},
                    }
                ],
            }
        ],
        "path_failures": [
            {
                "goal_index": 0,
                "stage": "move",
                "safe_fraction": 0.5,
                "blocked_by": [{"object_ref": "scene.objects.block"}],
            }
        ],
    }


def test_tool_path_summary_keeps_path_gate_evidence() -> None:
    summary = summarize_tool_path_report(_tool_path_report())

    assert summary["semantic_goal_preserved"] is False
    assert summary["path_objective_violations"][0]["objective_id"] == "move_objective_1"
    assert "pose_world_tool" not in summary["path_objective_violations"][0]
    assert (
        "pose_world_tool"
        not in summary["path_objective_failures"][0]["path_objective_violations"][0]
    )
    assert summary["path_failures"][0]["blocked_by"][0]["object_ref"] == "scene.objects.block"


def test_failure_report_and_prompt_include_tool_path_summary() -> None:
    report = build_failure_report(tool_path_report=_tool_path_report())
    prompt = render_repair_prompt(failure_report=report)

    assert report["failed_stage"] == "tool_path_solver"
    assert report["tool_path_summary"]["semantic_goal_preserved"] is False
    assert "tool_path_summary" in prompt
    assert "semantic_goal_preserved" in prompt
    assert "do not patch raw poses" in prompt
