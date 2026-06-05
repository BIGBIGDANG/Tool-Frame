from parttoolgen.providers.feedback.semantic_contact import (
    NO_REPAIR_NEEDED,
    REPLAN_FROM_LATEST_POSE,
    REQUEST_VISUAL_REPAIR,
    build_semantic_feedback_decision,
    semantic_contact_failures,
)


def _semantic_failed_report() -> dict:
    return {
        "passed": False,
        "stage_reports": [
            {
                "goal_index": 2,
                "stage": "contact",
                "success": False,
                "semantic_contact_required": True,
                "semantic_contact_success": False,
                "semantic_contact_error_m": 0.07,
                "semantic_failure_reason": "functional_target_error_above_threshold",
                "policy_goal_success": True,
                "tracking_success": False,
                "expected_contact": {
                    "tool_part": "candidate.points.part_004_contact",
                    "scene_target": "scene.keypoints.target_contact_point",
                },
            }
        ],
    }


def test_semantic_contact_failures_extract_repair_evidence() -> None:
    failures = semantic_contact_failures(_semantic_failed_report())

    assert failures[0]["stage"] == "contact"
    assert failures[0]["semantic_contact_error_m"] == 0.07
    assert failures[0]["expected_contact"]["tool_part"] == "candidate.points.part_004_contact"


def test_semantic_feedback_replans_before_visual_repair() -> None:
    decision = build_semantic_feedback_decision(
        _semantic_failed_report(),
        replan_attempts=0,
        max_replan_attempts_before_visual_repair=1,
    )

    assert decision["next_step"] == REPLAN_FROM_LATEST_POSE
    assert decision["rekep_style_feedback"] is True
    assert decision["robotwin_style_visual_repair"] is False


def test_semantic_feedback_requests_visual_repair_after_replan_budget() -> None:
    decision = build_semantic_feedback_decision(
        _semantic_failed_report(),
        replan_attempts=1,
        max_replan_attempts_before_visual_repair=1,
    )

    assert decision["next_step"] == REQUEST_VISUAL_REPAIR
    assert decision["robotwin_style_visual_repair"] is True


def test_semantic_feedback_noops_on_passed_report() -> None:
    decision = build_semantic_feedback_decision({"passed": True, "stage_reports": []})

    assert decision["next_step"] == NO_REPAIR_NEEDED
