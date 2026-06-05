import json

from parttoolgen.program.repair import build_failure_report, render_repair_prompt
from parttoolgen.program.visual_diagnosis import (
    build_visual_diagnosis_report,
    parse_visual_diagnosis_response,
    render_visual_diagnosis_prompt,
    summarize_visual_diagnosis_report,
)


def _visual_evidence() -> dict:
    return {
        "schema_version": "v0.1",
        "evidence_id": "unit_visual_evidence",
        "items": [
            {
                "id": "mallet_anchor_overlay",
                "kind": "contact_sheet",
                "media_type": "image",
                "path": "artifacts/overlay.png",
                "description": "Numbered geometry functional-frame candidates for mallet.",
                "provider_id": "sampart3d_geometry",
                "source_stage": "tool_prior",
                "allowed_use": "planning_context",
            },
            {
                "id": "rollout_final",
                "kind": "rollout_frame",
                "media_type": "image",
                "path": "artifacts/final.png",
                "description": "Final rollout frame.",
                "provider_id": "simtoolreal",
                "source_stage": "execution",
                "allowed_use": "repair_diagnosis",
            },
        ],
        "metadata": {
            "anchor_ref_table": [
                {
                    "tool_id": "mallet_hammer",
                    "number": 4,
                    "point_ref": "candidate.points.part_002_surface_center",
                    "part_ref": "tool.parts.part_002",
                }
            ],
            "contact_marker_table": [
                {
                    "tool_id": "mallet_hammer",
                    "part_marker": "2",
                    "part_ref": "tool.parts.part_002",
                    "contact_marker": "C3",
                    "point_ref": "candidate.points.part_002_y_negative_contact",
                    "surface_ref": "candidate.surfaces.part_002_y_negative_contact",
                    "polarity": "negative",
                }
            ],
        },
    }


def test_visual_diagnosis_prompt_and_report_are_symbolic() -> None:
    visual_evidence = _visual_evidence()
    prompt = render_visual_diagnosis_prompt(
        report_id="diag_001",
        visual_evidence=visual_evidence,
        execution_report={"passed": False, "failed_refs": ["candidate.points.part_002_surface_center"]},
        task_instruction="push the target",
    )

    assert "return exactly one JSON object" in prompt
    assert "Do not output Python, robot actions, coordinates, poses" in prompt
    assert "candidate.points.part_002_surface_center" in prompt
    assert "contact_marker_refs" in prompt
    assert "C3" in prompt

    report = build_visual_diagnosis_report(
        report_id="diag_001",
        visual_evidence=visual_evidence,
        execution_report={"passed": False, "failed_refs": ["candidate.points.part_002_surface_center"]},
        task_instruction="push the target",
    )
    summary = summarize_visual_diagnosis_report(report)
    assert summary["diagnosis_category"] == "tracking_or_execution_failure"
    assert "candidate.points.part_002_surface_center" in summary["related_refs"]
    assert "rollout_final" in summary["evidence_items"]
    assert summary["repair_decision"]["pcap_repair"] is True
    assert summary["metrics"]["semantic_contact_failed_count"] == 0


def test_visual_diagnosis_response_parses_json_fence() -> None:
    raw = """```json
{
  "schema_version": "v0.1",
  "report_id": "diag_002",
  "diagnosis_category": "wrong_function_point",
  "observations": ["The marked contact appears offset from the active tip."],
  "evidence_items": ["rollout_final"],
  "related_refs": ["candidate.points.part_002_surface_center"],
  "repair_hints": ["Select a different candidate point in the same selected region."],
  "confidence": 0.7,
  "metadata": {}
}
```"""
    report = parse_visual_diagnosis_response(raw, report_id="diag_002")

    assert report["diagnosis_category"] == "wrong_function_point"
    assert report["metadata"]["created_by"] == "vlm_visual_diagnosis"


def test_visual_diagnosis_response_drops_empty_hints_and_refs() -> None:
    raw = """
{
  "schema_version": "v0.1",
  "report_id": "diag_empty_hint",
  "diagnosis_category": "no_visual_failure_observed",
  "observations": ["No issue.", "  "],
  "evidence_items": ["rollout_final"],
  "related_refs": ["candidate.points.part_002_surface_center", "", "not_a_ref"],
  "repair_hints": ["", "  "],
  "confidence": 0.7,
  "metadata": {}
}
"""
    report = parse_visual_diagnosis_response(raw, report_id="diag_empty_hint")

    assert report["observations"] == ["No issue."]
    assert report["repair_hints"] == []
    assert report["related_refs"] == ["candidate.points.part_002_surface_center"]
    assert report["metadata"]["repair_decision"]["pcap_repair"] is False
    assert "metrics" in report["metadata"]


def test_visual_diagnosis_scaffold_calls_out_semantic_contact_failure() -> None:
    report = build_visual_diagnosis_report(
        report_id="diag_semantic",
        visual_evidence=_visual_evidence(),
        execution_report={
            "passed": False,
            "stage_reports": [
                {
                    "goal_index": 0,
                    "stage": "contact",
                    "success": False,
                    "tracking_error_m": 0.012,
                    "rotation_error_rad": 0.19,
                    "policy_goal_success": True,
                    "semantic_contact_required": True,
                    "semantic_contact_success": False,
                    "semantic_contact_error_m": 0.052,
                    "semantic_failure_reason": "functional_target_error_above_threshold",
                    "expected_contact": {
                        "tool_part": "candidate.points.part_004_contact",
                        "scene_target": "scene.keypoints.target_contact_point",
                    },
                }
            ],
        },
        task_instruction="hit the block with the hammer",
    )

    assert report["diagnosis_category"] == "tracking_or_execution_failure"
    assert any("functional point" in item for item in report["observations"])
    assert any("re-solve" in item for item in report["repair_hints"])
    metrics = report["metadata"]["metrics"]
    assert metrics["max_tracking_error_m"] == 0.012
    assert metrics["max_rotation_error_rad"] == 0.19
    assert metrics["max_semantic_contact_error_m"] == 0.052
    assert metrics["stage_metrics"][0]["semantic_contact_error_m"] == 0.052


def test_repair_prompt_includes_visual_diagnosis_report() -> None:
    diagnosis = json.loads(
        json.dumps(
            {
                "schema_version": "v0.1",
                "report_id": "diag_003",
                "diagnosis_category": "orientation_relation_missing",
                "observations": ["The contact point reached the target without stable tool orientation."],
                "evidence_items": ["rollout_final"],
                "related_refs": [
                    "candidate.points.part_002_surface_center",
                    "tool.parts.part_002",
                ],
                "repair_hints": ["Add point-pair orientation alignment using the bound contact ref."],
                "confidence": 0.8,
                "metadata": {"created_by": "unit_test"},
            }
        )
    )
    report = build_failure_report(
        execution_report={"passed": False},
        visual_diagnosis_report=diagnosis,
    )
    prompt = render_repair_prompt(
        failure_report=report,
        visual_diagnosis_report=diagnosis,
    )

    assert "visual_diagnosis" in prompt
    assert "orientation_relation_missing" in prompt
    assert "symbolic changes to binding refs" in prompt
