#!/usr/bin/env python3
"""Prompt-safe visual diagnosis artifacts for repair.

The diagnosis pass may look at rollout frames/videos and planning overlays, but
its output remains symbolic: failure category, evidence item ids, related refs,
and repair hints. It never becomes a source of poses, actions, or controller
edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parttoolgen.constraints.vlm_client import build_vlm_client
from parttoolgen.program.visual_evidence import load_visual_evidence, summarize_visual_evidence
from parttoolgen.program.visual_evidence import prompt_image_paths as collect_prompt_image_paths
from parttoolgen.validation.validate_parttoolgen import (
    validate_visual_diagnosis_report as _validate_visual_diagnosis_report,
)

DIAGNOSIS_CATEGORIES = {
    "no_visual_failure_observed",
    "wrong_function_region",
    "wrong_function_point",
    "wrong_target_contact",
    "orientation_relation_missing",
    "orientation_relation_wrong",
    "path_collision_or_occlusion",
    "tracking_or_execution_failure",
    "insufficient_visual_evidence",
    "unknown",
}
_SYMBOL_REF_PREFIXES = ("scene.", "tool.", "target.", "candidate.")


def load_visual_diagnosis_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_visual_diagnosis_report(report)
    return report


def validate_visual_diagnosis_report(report: Any) -> None:
    _validate_visual_diagnosis_report(report)


def _looks_like_symbol_ref(value: str) -> bool:
    return value.startswith(_SYMBOL_REF_PREFIXES) and "*" not in value


def _collect_symbol_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        if _looks_like_symbol_ref(value):
            refs.add(value)
    elif isinstance(value, dict):
        for child in value.values():
            refs.update(_collect_symbol_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_collect_symbol_refs(child))
    return refs


def _compact_report(report: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {key: report[key] for key in keys if key in report}


def _execution_metrics(execution_report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(execution_report, dict):
        return {}
    stage_reports = execution_report.get("stage_reports")
    max_tracking_error_m: float | None = None
    max_rotation_error_rad: float | None = None
    max_semantic_contact_error_m: float | None = None
    semantic_required = 0
    semantic_failed = 0
    failed_stage_index: int | None = None
    stage_metrics: list[dict[str, Any]] = []
    if isinstance(stage_reports, list):
        for index, stage in enumerate(stage_reports):
            if not isinstance(stage, dict):
                continue
            tracking = stage.get("tracking_error_m")
            if isinstance(tracking, (int, float)):
                max_tracking_error_m = max(float(tracking), max_tracking_error_m or 0.0)
            rotation = stage.get("rotation_error_rad")
            if isinstance(rotation, (int, float)):
                max_rotation_error_rad = max(float(rotation), max_rotation_error_rad or 0.0)
            semantic_error = stage.get("semantic_contact_error_m")
            if isinstance(semantic_error, (int, float)):
                max_semantic_contact_error_m = max(
                    float(semantic_error),
                    max_semantic_contact_error_m or 0.0,
                )
            if stage.get("semantic_contact_required") is True:
                semantic_required += 1
                if stage.get("semantic_contact_success") is False:
                    semantic_failed += 1
            if failed_stage_index is None and stage.get("success") is False:
                failed_stage_index = index
            stage_metrics.append(
                {
                    "stage_index": index,
                    "goal_index": stage.get("goal_index"),
                    "tracking_error_m": float(tracking)
                    if isinstance(tracking, (int, float))
                    else None,
                    "rotation_error_rad": float(rotation)
                    if isinstance(rotation, (int, float))
                    else None,
                    "semantic_contact_error_m": float(semantic_error)
                    if isinstance(semantic_error, (int, float))
                    else None,
                    "tracking_success": stage.get("tracking_success")
                    if isinstance(stage.get("tracking_success"), bool)
                    else None,
                    "semantic_contact_success": stage.get("semantic_contact_success")
                    if isinstance(stage.get("semantic_contact_success"), bool)
                    else None,
                    "semantic_gate_passed": stage.get("semantic_gate_passed")
                    if isinstance(stage.get("semantic_gate_passed"), bool)
                    else None,
                }
            )
    return {
        "max_tracking_error_m": max_tracking_error_m,
        "max_rotation_error_rad": max_rotation_error_rad,
        "max_semantic_contact_error_m": max_semantic_contact_error_m,
        "semantic_contact_required_count": semantic_required,
        "semantic_contact_failed_count": semantic_failed,
        "first_failed_stage_index": failed_stage_index,
        "stage_metrics": stage_metrics[:32],
    }


def _default_repair_decision(category: str) -> dict[str, bool]:
    symbolic_failure = category not in {
        "no_visual_failure_observed",
        "insufficient_visual_evidence",
    }
    contact_axis_rebind = category in {
        "wrong_function_point",
        "orientation_relation_missing",
        "orientation_relation_wrong",
    }
    return {
        "pcap_repair": bool(symbolic_failure),
        "contact_axis_rebind": bool(contact_axis_rebind),
        "region_rebind": category == "wrong_function_region",
        "lock_function_refs": category in {
            "tracking_or_execution_failure",
            "path_collision_or_occlusion",
        },
    }


def summarize_visual_diagnosis_report(
    report: dict[str, Any] | None,
    *,
    max_observations: int = 8,
    max_hints: int = 8,
    max_refs: int = 16,
) -> dict[str, Any] | None:
    """Return the repair-prompt subset of a diagnosis report."""

    if report is None:
        return None
    validate_visual_diagnosis_report(report)
    return {
        "schema_version": report["schema_version"],
        "report_id": report["report_id"],
        "diagnosis_category": report["diagnosis_category"],
        "observations": report.get("observations", [])[:max_observations],
        "evidence_items": report.get("evidence_items", [])[:max_observations],
        "related_refs": report.get("related_refs", [])[:max_refs],
        "repair_hints": report.get("repair_hints", [])[:max_hints],
        "confidence": report.get("confidence"),
        "repair_decision": (report.get("metadata") or {}).get("repair_decision"),
        "metrics": (report.get("metadata") or {}).get("metrics"),
        "usage_contract": (
            "Use this diagnosis only to repair symbolic refs, binding choices, "
            "stages, constraints, tolerances, or objective registrations. Do not "
            "turn observations into raw poses, coordinates, actions, or controller edits."
        ),
    }


def build_visual_diagnosis_report(
    *,
    report_id: str,
    visual_evidence: dict[str, Any] | None,
    execution_report: dict[str, Any] | None = None,
    tool_path_report: dict[str, Any] | None = None,
    compiler_report: dict[str, Any] | None = None,
    affordance_binding: dict[str, Any] | None = None,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Build a conservative non-VLM diagnosis scaffold from available reports."""

    visual_summary = summarize_visual_evidence(
        visual_evidence,
        allowed_uses=("planning_context", "repair_diagnosis"),
    )
    evidence_items = [
        str(item["id"])
        for item in (visual_summary or {}).get("items", [])
        if isinstance(item, dict) and item.get("id")
    ]
    related_refs = sorted(
        _collect_symbol_refs(
            {
                "execution_report": execution_report,
                "tool_path_report": tool_path_report,
                "compiler_report": compiler_report,
                "affordance_binding": affordance_binding,
            }
        )
    )
    observations: list[str] = []
    repair_hints: list[str] = []
    category = "insufficient_visual_evidence"
    confidence = 0.25

    if tool_path_report and tool_path_report.get("semantic_goal_preserved") is False:
        category = "path_collision_or_occlusion"
        observations.append("ToolPathSolver reported that the repaired path did not preserve the semantic goal.")
        repair_hints.append("Repair ProgramIR refs, stages, path constraints, or objectives and recompile.")
        confidence = 0.55
    elif execution_report and (
        execution_report.get("passed") is False or execution_report.get("success") is False
    ):
        semantic_failures = [
            stage
            for stage in execution_report.get("stage_reports", [])
            if isinstance(stage, dict)
            and stage.get("semantic_contact_required")
            and stage.get("semantic_contact_success") is False
        ]
        if semantic_failures:
            category = "tracking_or_execution_failure"
            first = semantic_failures[0]
            observations.append(
                "Execution report shows that the actual tool functional point did not satisfy the expected contact target."
            )
            if first.get("policy_goal_success") is True:
                observations.append(
                    "The SimToolReal policy success signal was not sufficient evidence for semantic functional contact."
                )
            repair_hints.append(
                "First re-solve the next whole-tool subgoal from the latest execution pose and target geometry; if the failure repeats, use visual evidence to repair selected refs or constraints."
            )
            confidence = 0.55
        else:
            category = "tracking_or_execution_failure"
            observations.append("Execution report marks the rollout as failed.")
            repair_hints.append("Use execution evidence to adjust symbolic constraints or selected refs, then recompile.")
            confidence = 0.45
    elif compiler_report and compiler_report.get("passed") is False:
        category = "unknown"
        observations.append("Compiler report marks the program as failed before execution.")
        repair_hints.append("Repair ProgramIR constraints or refs before attempting execution diagnosis.")
        confidence = 0.35
    elif execution_report or tool_path_report or compiler_report:
        category = "no_visual_failure_observed"
        observations.append("Structured reports do not indicate a repair-stage failure.")
        repair_hints.append("Keep current symbolic choices unless visual diagnosis finds a specific mismatch.")
        confidence = 0.4

    if not evidence_items:
        observations.append("No prompt-safe visual evidence items were available for diagnosis.")
    else:
        observations.append("Visual diagnosis evidence is available as item ids, not raw coordinates or poses.")

    report = {
        "schema_version": "v0.1",
        "report_id": report_id,
        "diagnosis_category": category,
        "observations": observations,
        "evidence_items": evidence_items,
        "related_refs": related_refs[:32],
        "repair_hints": repair_hints,
        "confidence": confidence,
        "metadata": {
            "created_by": "parttoolgen.visual_diagnosis",
            "mode": "deterministic_scaffold",
            "task_instruction": task_instruction or "",
            "repair_decision": _default_repair_decision(category),
            "metrics": _execution_metrics(execution_report),
        },
    }
    validate_visual_diagnosis_report(report)
    return report


def render_visual_diagnosis_prompt(
    *,
    report_id: str,
    visual_evidence: dict[str, Any] | None,
    execution_report: dict[str, Any] | None = None,
    tool_path_report: dict[str, Any] | None = None,
    compiler_report: dict[str, Any] | None = None,
    affordance_binding: dict[str, Any] | None = None,
    task_instruction: str | None = None,
) -> str:
    """Render the JSON-only VLM prompt for visual repair diagnosis."""

    payload = {
        "report_id": report_id,
        "task_instruction": task_instruction,
        "visual_evidence": summarize_visual_evidence(
            visual_evidence,
            allowed_uses=("planning_context", "repair_diagnosis"),
        ),
        "execution_report": _compact_report(
            execution_report,
            (
                "passed",
                "success",
                "tracking_execution_passed",
                "semantic_execution_passed",
                "failed_stage",
                "failed_stage_reason",
                "execution_failed_stage",
                "semantic_failed_stage",
                "failed_refs",
                "stage_reports",
                "tracking_error",
                "tracking_error_m",
                "contact_evidence",
                "collision_evidence",
                "drop_evidence",
                "dropped",
            ),
        ),
        "quantitative_execution_metrics": _execution_metrics(execution_report),
        "tool_path_report": _compact_report(
            tool_path_report,
            (
                "passed",
                "semantic_goal_preserved",
                "functional_goal_drift_m",
                "warnings",
                "repair_hints",
                "repairs",
                "path_objective_violations",
            ),
        ),
        "compiler_report": _compact_report(
            compiler_report,
            ("passed", "failed_refs", "failed_stages", "stages", "warnings"),
        ),
        "affordance_binding_refs": sorted(_collect_symbol_refs(affordance_binding)),
    }
    return (
        "You are a PartToolGen visual diagnosis model for repair only.\n"
        "Look at the attached/manifested visual evidence and the structured reports, then return exactly one JSON object.\n"
        "Do not output Python, robot actions, coordinates, poses, transforms, controller edits, or SimToolReal JSON.\n"
        "Marker numbers are only indexes into listed refs; any related_refs must be exact refs from the payload.\n"
        "Use rollout video/frames only to classify symbolic failure modes such as wrong function region, "
        "wrong function point, wrong target contact, missing/wrong orientation relation, path collision, "
        "or tracking failure. Do not infer raw metric coordinates or fixed goals from pixels.\n"
        "A rollout can pass structured tracking/contact metrics while still using a visually wrong "
        "functional surface, edge, endpoint, or side of the tool. If the visible tool-target interaction "
        "uses a different functional surface/point than the task implies, classify it as "
        "wrong_function_point or orientation_relation_wrong and set metadata.repair_decision."
        "contact_axis_rebind=true. This check is generic and must compare the rollout frames against "
        "the provided candidate contact/axis cards and exact marker-ref tables.\n"
        "If the visual evidence includes C*/A* marker tables or contact/axis choice cards, compare the "
        "current functional refs against those candidate refs before claiming wrong_function_region or "
        "wrong_function_point. If the selected candidate appears visually plausible but execution did not "
        "reach or maintain the semantic target, prefer tracking_or_execution_failure, "
        "orientation_relation_wrong, or path_collision_or_occlusion as appropriate.\n"
        "The contact/axis card is scoped to the currently selected part. If the selected part's card has "
        "no visually plausible function contact for the task, classify the failure as wrong_function_region "
        "and set metadata.repair_decision.region_rebind=true so the repair loop returns to part selection. "
        "If the selected part is plausible but the chosen C marker/direction is wrong, classify it as "
        "wrong_function_point or orientation_relation_wrong and set contact_axis_rebind=true.\n"
        "The report must include structured quantitative repair metadata under metadata.repair_decision "
        "and metadata.metrics. Downstream code will use metadata.repair_decision booleans, not "
        "diagnosis prose or repair_hints wording, for machine decisions. Use this shape: "
        "metadata.repair_decision={pcap_repair: bool, contact_axis_rebind: bool, region_rebind: bool, "
        "lock_function_refs: bool}; metadata.metrics={function_ref_visual_confidence: 0..1 or null, "
        "target_ref_visual_confidence: 0..1 or null, max_tracking_error_m: number or null, "
        "max_rotation_error_rad: number or null, semantic_contact_error_m: number or null, "
        "stage_metrics: [{stage_index, goal_index, tracking_error_m, rotation_error_rad, "
        "semantic_contact_error_m, tracking_success, semantic_contact_success}]}.\n"
        "Choose one diagnosis_category from: "
        + ", ".join(sorted(DIAGNOSIS_CATEGORIES))
        + ".\n"
        "Required JSON fields: schema_version='v0.1', report_id, diagnosis_category, observations, "
        "evidence_items, related_refs, repair_hints, confidence, metadata.\n"
        "repair_hints must describe symbolic repairs only: AffordanceBinding slot choice, ref choice, stage, "
        "constraint, bounded tolerance/offset, or objective registration.\n\n"
        "Diagnosis payload:\n"
        + json.dumps(payload, indent=2)
        + "\n"
    )


def parse_visual_diagnosis_response(
    raw_text: str,
    *,
    report_id: str,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Parse and validate a JSON-only diagnosis response."""

    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("visual diagnosis response did not contain a JSON object")
    data = json.loads(text[start : end + 1])
    data.setdefault("schema_version", "v0.1")
    data.setdefault("report_id", report_id)
    data.setdefault("metadata", {})
    if task_instruction:
        data["metadata"].setdefault("task_instruction", task_instruction)
    data["metadata"].setdefault("created_by", "vlm_visual_diagnosis")
    if isinstance(data["metadata"], dict):
        data["metadata"].setdefault(
            "repair_decision",
            _default_repair_decision(str(data.get("diagnosis_category") or "unknown")),
        )
        data["metadata"].setdefault("metrics", {})
    for key in ("observations", "evidence_items", "repair_hints"):
        value = data.get(key)
        if isinstance(value, list):
            data[key] = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    related_refs = data.get("related_refs")
    if isinstance(related_refs, list):
        valid_refs = [
            ref.strip()
            for ref in related_refs
            if isinstance(ref, str) and ref.strip() and _looks_like_symbol_ref(ref.strip())
        ]
        dropped_refs = [
            ref
            for ref in related_refs
            if not (isinstance(ref, str) and ref.strip() and _looks_like_symbol_ref(ref.strip()))
        ]
        data["related_refs"] = valid_refs
        if dropped_refs:
            data["metadata"]["dropped_non_symbol_related_refs"] = [
                str(ref) for ref in dropped_refs[:32]
            ]
    validate_visual_diagnosis_report(data)
    return data


def run_vlm_visual_diagnosis(
    *,
    report_id: str,
    visual_evidence: dict[str, Any] | None,
    response_mode: str,
    execution_report: dict[str, Any] | None = None,
    tool_path_report: dict[str, Any] | None = None,
    compiler_report: dict[str, Any] | None = None,
    affordance_binding: dict[str, Any] | None = None,
    task_instruction: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_new_tokens: int = 768,
    max_images: int = 4,
) -> tuple[dict[str, Any], str, str]:
    """Run the visual observer VLM and return a validated diagnosis report.

    The observer is intentionally report-only: attached images can justify
    symbolic repair hints, but cannot become metric poses, actions, or controller
    edits.
    """

    prompt = render_visual_diagnosis_prompt(
        report_id=report_id,
        visual_evidence=visual_evidence,
        execution_report=execution_report,
        tool_path_report=tool_path_report,
        compiler_report=compiler_report,
        affordance_binding=affordance_binding,
        task_instruction=task_instruction,
    )
    image_paths = collect_prompt_image_paths(
        visual_evidence,
        allowed_uses=("planning_context", "repair_diagnosis"),
        max_images=max_images,
    )
    client = build_vlm_client(
        response_mode=response_mode,
        model=model,
        base_url=base_url,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
    )
    raw_response = client.generate(
        prompt=prompt,
        task_instruction=task_instruction or "",
        context_id=report_id,
        system_prompt=(
            "Return exactly one JSON object and no prose. "
            "Do not output Python, robot actions, coordinates, poses, transforms, "
            "controller edits, or SimToolReal JSON."
        ),
    )
    report = parse_visual_diagnosis_response(
        raw_response.raw_text,
        report_id=report_id,
        task_instruction=task_instruction,
    )
    report.setdefault("metadata", {})
    report["metadata"].setdefault("created_by", "vlm_visual_diagnosis")
    report["metadata"]["vlm_response"] = raw_response.metadata
    report["metadata"]["prompt_images"] = image_paths
    validate_visual_diagnosis_report(report)
    return report, prompt, raw_response.raw_text


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-evidence")
    parser.add_argument("--execution-report")
    parser.add_argument("--tool-path-report")
    parser.add_argument("--compiler-report")
    parser.add_argument("--affordance-binding")
    parser.add_argument("--task", default="")
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--prompt-output")
    parser.add_argument("--raw-response")
    parser.add_argument("--raw-response-output")
    parser.add_argument(
        "--response-mode",
        default="none",
        choices=["none", "openai-compatible"],
        help="Use none for deterministic scaffold, or OpenAI-compatible VLM observer diagnosis.",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    visual_evidence = load_visual_evidence(args.visual_evidence)
    execution_report = _read_json(args.execution_report)
    tool_path_report = _read_json(args.tool_path_report)
    compiler_report = _read_json(args.compiler_report)
    affordance_binding = _read_json(args.affordance_binding)
    if args.raw_response:
        prompt = render_visual_diagnosis_prompt(
            report_id=args.report_id,
            visual_evidence=visual_evidence,
            execution_report=execution_report,
            tool_path_report=tool_path_report,
            compiler_report=compiler_report,
            affordance_binding=affordance_binding,
            task_instruction=args.task or None,
        )
        report = parse_visual_diagnosis_response(
            Path(args.raw_response).read_text(encoding="utf-8"),
            report_id=args.report_id,
            task_instruction=args.task or None,
        )
    elif args.response_mode != "none":
        report, prompt, raw_response = run_vlm_visual_diagnosis(
            report_id=args.report_id,
            visual_evidence=visual_evidence,
            response_mode=args.response_mode,
            execution_report=execution_report,
            tool_path_report=tool_path_report,
            compiler_report=compiler_report,
            affordance_binding=affordance_binding,
            task_instruction=args.task or None,
            model=args.model,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            max_images=args.max_images,
        )
        if args.raw_response_output:
            Path(args.raw_response_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.raw_response_output).write_text(raw_response, encoding="utf-8")
    else:
        prompt = render_visual_diagnosis_prompt(
            report_id=args.report_id,
            visual_evidence=visual_evidence,
            execution_report=execution_report,
            tool_path_report=tool_path_report,
            compiler_report=compiler_report,
            affordance_binding=affordance_binding,
            task_instruction=args.task or None,
        )
        report = build_visual_diagnosis_report(
            report_id=args.report_id,
            visual_evidence=visual_evidence,
            execution_report=execution_report,
            tool_path_report=tool_path_report,
            compiler_report=compiler_report,
            affordance_binding=affordance_binding,
            task_instruction=args.task or None,
        )
    if args.prompt_output:
        Path(args.prompt_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prompt_output).write_text(prompt, encoding="utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
