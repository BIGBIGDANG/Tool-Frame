from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from parttoolgen.evaluation.task_success import check_task_success, predicate_passed


def _load_hammer_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "cli" / "runners" / "run_image_grounded_hammer_strike_eval.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_hammer_strike_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hammer_eval = _load_hammer_eval_module()


def test_hammer_down_success_uses_semantic_contact_not_tracking_or_displacement() -> None:
    axis = {"passed": True}
    compile_summary = {"path_solver_passed": True}
    rollout_summary = {
        "policy_goal_passed": True,
        "tracking_execution_passed": False,
        "semantic_execution_passed": True,
        "benchmark_extra_actor_summary": {
            "actors": [{"name": "strike_nail", "displacement_m": 0.0, "z_delta_m": 0.0}],
            "max_displacement_m": 0.0,
        },
    }
    execution_report = {"semantic_execution_passed": True}

    success_check = check_task_success(
        "hammer_down_strike",
        axis=axis,
        compile_summary=compile_summary,
        rollout_summary=rollout_summary,
        execution_report=execution_report,
    )

    assert predicate_passed(success_check, "semantic_contact") is True
    assert success_check["passed"] is True
    assert success_check["policy"]["requires_target_displacement"] is False


def test_hammer_down_success_requires_semantic_contact_when_rollout_runs() -> None:
    success_check = check_task_success(
        "hammer_down_strike",
        axis={"passed": True},
        compile_summary={"path_solver_passed": True},
        rollout_summary={"semantic_execution_passed": False},
        execution_report=None,
    )

    assert success_check["passed"] is False


def test_hammer_down_skip_rollout_is_missing_required_evidence() -> None:
    success_check = check_task_success(
        "hammer_down_strike",
        axis={"passed": True},
        compile_summary={"path_solver_passed": True},
        rollout_summary=None,
        execution_report=None,
        skip_rollout=True,
    )

    assert success_check["passed"] is False
    assert predicate_passed(success_check, "semantic_contact") is False
    assert predicate_passed(success_check, "rollout_evidence_present") is False
    assert success_check["policy"]["requires_semantic_contact"] is True


def test_resume_refresh_recomputes_old_tracking_based_trial_result(tmp_path: Path) -> None:
    rollout_summary_path = tmp_path / "rollout_summary.json"
    execution_report_path = tmp_path / "execution_report.json"
    compile_summary_path = tmp_path / "compile_summary.json"
    rollout_summary_path.write_text(
        json.dumps(
            {
                "policy_goal_passed": True,
                "tracking_execution_passed": False,
                "semantic_execution_passed": True,
            }
        ),
        encoding="utf-8",
    )
    execution_report_path.write_text(
        json.dumps({"semantic_execution_passed": True}),
        encoding="utf-8",
    )
    compile_summary_path.write_text(
        json.dumps({"path_solver_passed": True}),
        encoding="utf-8",
    )
    old_result = {
        "passed": False,
        "scene_direction_axis": {"passed": True},
        "compile_path_solver_passed": True,
        "semantic_execution_passed": True,
        "artifacts": {
            "rollout_summary": str(rollout_summary_path),
            "execution_report": str(execution_report_path),
            "compile_summary": str(compile_summary_path),
        },
    }

    refreshed = hammer_eval._refresh_success_fields(old_result, skip_rollout=False)

    assert refreshed["passed"] is True
    assert refreshed["hammer_contact_passed"] is True
    assert refreshed["success_policy"]["requires_target_displacement"] is False
    assert refreshed["success_policy"]["requires_tracking_execution_passed"] is False
