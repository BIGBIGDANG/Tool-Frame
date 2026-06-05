from __future__ import annotations

import json
from pathlib import Path

from parttoolgen.evaluation.report import build_task_evaluation_report
from parttoolgen.evaluation.task_spec import ToolUseTask


def _task(path: Path) -> ToolUseTask:
    return ToolUseTask(
        path=path,
        data={
            "task_id": "unit_report",
            "family": "unit",
            "split": "unit",
            "instruction": "Check semantic success.",
            "tool": {"tool_id": "unit_tool"},
            "public_scene": {
                "provider": "unit",
                "scene_evidence": [],
                "target_candidates": [],
                "leakage_policy": {"forbidden_public_fields": []},
            },
            "private_eval": {
                "success_predicates": [{"kind": "semantic_execution_passed"}],
            },
        },
    )


def test_task_report_total_pass_is_not_implicitly_gated_by_tracking(tmp_path: Path) -> None:
    rollout_summary_path = tmp_path / "rollout_summary.json"
    rollout_summary_path.write_text(
        json.dumps(
            {
                "semantic_execution_passed": True,
                "policy_goal_passed": False,
                "tracking_execution_passed": False,
            }
        ),
        encoding="utf-8",
    )

    report = build_task_evaluation_report(
        task=_task(tmp_path / "task.json"),
        output_dir=tmp_path / "out",
        rollout_summary_path=rollout_summary_path,
    )

    assert report["passed"] is True
    assert report["task_success"] is True
    assert report["tracking_passed"] is False
    assert report["failure_category"] is None
