"""Semantic execution feedback helpers.

These helpers keep ReKep-style closed-loop decisions and RoboTwin-style repair
handoff above the low-level SimToolReal policy. They consume execution reports;
they do not read simulator state, patch fixed goals, or command the robot.
"""

from __future__ import annotations

from typing import Any


REPLAN_FROM_LATEST_POSE = "replan_from_latest_execution_pose"
REQUEST_VISUAL_REPAIR = "request_visual_diagnosis_repair"
NO_REPAIR_NEEDED = "no_repair_needed"
STRUCTURED_REPAIR = "structured_symbolic_repair"


def semantic_contact_failures(execution_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return failed stages with semantic functional-contact evidence."""

    if not isinstance(execution_report, dict):
        return []
    failures: list[dict[str, Any]] = []
    for stage in execution_report.get("stage_reports", []):
        if not isinstance(stage, dict):
            continue
        if not stage.get("semantic_contact_required"):
            continue
        if stage.get("semantic_contact_success") is not False:
            continue
        failures.append(
            {
                key: stage[key]
                for key in (
                    "goal_index",
                    "stage",
                    "semantic_contact_error_m",
                    "last_semantic_contact_error_m",
                    "min_semantic_contact_error_m",
                    "semantic_failure_reason",
                    "expected_contact",
                    "contact_point_goal",
                    "policy_goal_success",
                    "tracking_success",
                    "tracking_error_m",
                )
                if key in stage
            }
        )
    return failures


def build_semantic_feedback_decision(
    execution_report: dict[str, Any] | None,
    *,
    replan_attempts: int = 0,
    max_replan_attempts_before_visual_repair: int = 1,
) -> dict[str, Any]:
    """Decide whether to re-solve subgoals or invoke visual repair.

    The intended mainline order is:

    1. If structured semantic contact failed, first re-read execution feedback
       and recompile/re-solve the next whole-tool subgoal from the latest tool
       pose and current target geometry.
    2. If the same semantic failure survives the allowed replan attempts, hand
       the execution report plus visual evidence to the repair diagnosis pass.
    """

    failures = semantic_contact_failures(execution_report)
    if not failures and isinstance(execution_report, dict) and execution_report.get("passed") is True:
        return {
            "schema_version": "v0.1",
            "next_step": NO_REPAIR_NEEDED,
            "reason": "execution_report_passed",
            "semantic_contact_failures": [],
            "repair_hints": [],
        }
    if failures:
        if int(replan_attempts) < int(max_replan_attempts_before_visual_repair):
            return {
                "schema_version": "v0.1",
                "next_step": REPLAN_FROM_LATEST_POSE,
                "reason": "semantic_contact_failed_before_replan_budget_exhausted",
                "rekep_style_feedback": True,
                "robotwin_style_visual_repair": False,
                "semantic_contact_failures": failures,
                "repair_hints": [
                    "Read latest execution pose and current target geometry.",
                    "Re-run ProgramIR -> compiler -> ToolPathSolver -> SimToolReal adapter for the next whole-tool subgoal.",
                    "Do not patch fixed-goal JSON or SimToolReal controller internals.",
                ],
            }
        return {
            "schema_version": "v0.1",
            "next_step": REQUEST_VISUAL_REPAIR,
            "reason": "semantic_contact_failed_after_replan_budget_exhausted",
            "rekep_style_feedback": True,
            "robotwin_style_visual_repair": True,
            "semantic_contact_failures": failures,
            "repair_hints": [
                "Use rollout visual evidence to diagnose wrong region, point, target, orientation, path, or tracking category.",
                "Apply only symbolic repairs to region binding, AffordanceBinding, ProgramIR constraints, tolerances, offsets, or objectives.",
                "Recompile and pass through all gates before another rollout.",
            ],
        }
    return {
        "schema_version": "v0.1",
        "next_step": STRUCTURED_REPAIR,
        "reason": "execution_report_failed_without_semantic_contact_failure",
        "semantic_contact_failures": [],
        "repair_hints": [
            "Use compiler, path, execution, and optional visual diagnosis reports as symbolic repair evidence.",
        ],
    }
