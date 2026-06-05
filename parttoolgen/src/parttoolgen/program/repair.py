#!/usr/bin/env python3
"""Repair prompts for restricted Python planning programs.

Repair is intentionally constrained to the restricted Python module. Compiler
and execution evidence can explain what failed, but the requested output stays
``def build_program(ctx): ...`` rather than raw poses or controller edits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parttoolgen.program.visual_diagnosis import (
    load_visual_diagnosis_report,
    summarize_visual_diagnosis_report,
)
from parttoolgen.program.visual_evidence import load_visual_evidence, summarize_visual_evidence


FAILURE_CATEGORIES = {
    "syntax_error",
    "ast_policy_violation",
    "api_contract_violation",
    "unknown_symbol_ref",
    "runtime_error",
    "optimizer_infeasible",
    "tool_part_missing",
    "vlm_unavailable",
    "unknown_failure",
}
_SYMBOL_REF_PREFIXES = ("scene.", "tool.", "target.", "candidate.")


def classify_failure(*, failed_stage: str, error_type: str, error: str) -> str:
    message = error.lower()
    kind = error_type.lower()
    if "api_key" in message or "vlm" in message and "unavailable" in message:
        return "vlm_unavailable"
    if "syntax" in kind or "not valid python" in message:
        return "syntax_error"
    if "forbidden" in message or "dunder" in message or "import" in message:
        return "ast_policy_violation"
    if "unknown symbol reference" in message or "unknown ref" in message:
        return "unknown_symbol_ref"
    if "api" in message or "contract" in message:
        return "api_contract_violation"
    if failed_stage == "tool_part_grounding":
        return "tool_part_missing"
    if failed_stage in {"compile", "optimize"}:
        return "optimizer_infeasible"
    if "runtime" in kind:
        return "runtime_error"
    return "unknown_failure"


def enrich_failure_report(report: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(report)
    enriched["failure_category"] = classify_failure(
        failed_stage=str(report.get("failed_stage", "")),
        error_type=str(report.get("error_type", "")),
        error=str(report.get("error", "")),
    )
    if "compiler_report" in enriched and "compiler_summary" not in enriched:
        enriched["compiler_summary"] = summarize_compiler_report(enriched["compiler_report"])
    if "tool_path_report" in enriched and "tool_path_summary" not in enriched:
        enriched["tool_path_summary"] = summarize_tool_path_report(enriched["tool_path_report"])
    if "execution_report" in enriched and "execution_summary" not in enriched:
        enriched["execution_summary"] = summarize_execution_report(enriched["execution_report"])
    if "affordance_binding" in enriched and "affordance_binding_summary" not in enriched:
        enriched["affordance_binding_summary"] = summarize_affordance_binding(
            enriched["affordance_binding"]
        )
    if "visual_diagnosis_report" in enriched and "visual_diagnosis_summary" not in enriched:
        enriched["visual_diagnosis_summary"] = summarize_visual_diagnosis_report(
            enriched["visual_diagnosis_report"]
        )
    return enriched


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


def summarize_affordance_binding(binding: dict[str, Any] | None) -> dict[str, Any]:
    """Return repair-facing binding slots without geometry or raw model chatter."""

    if not isinstance(binding, dict):
        return {}
    raw_slots = binding.get("slot_refs") or binding.get("slot_bindings") or {}
    slots: dict[str, list[str]] = {}
    if isinstance(raw_slots, dict):
        for slot_name, slot_value in sorted(raw_slots.items()):
            refs = sorted(_collect_symbol_refs(slot_value))
            if refs:
                slots[str(slot_name)] = refs
    selected_refs = sorted(
        _collect_symbol_refs(
            {
                "slot_refs": raw_slots,
                "selected_refs": binding.get("selected_refs"),
                "bound_allowed_refs": binding.get("bound_allowed_refs"),
            }
        )
    )
    summary: dict[str, Any] = {
        "schema_version": binding.get("schema_version"),
        "binding_id": binding.get("binding_id"),
        "slot_refs": slots,
        "slot_bindings": slots,
        "selected_refs": selected_refs,
    }
    gate = binding.get("binding_gate")
    if isinstance(gate, dict):
        summary["binding_gate"] = {
            key: gate[key]
            for key in ("passed", "failed_refs", "warning_refs", "repair_hints")
            if key in gate
        }
    metadata = binding.get("metadata") or binding.get("generation_metadata")
    if isinstance(metadata, dict):
        for key in ("created_by", "generated_by", "model", "prompt_id", "selected_ref_count"):
            if key in metadata:
                summary.setdefault("metadata", {})[key] = metadata[key]
    return summary


def summarize_compiler_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract repair-relevant compiler evidence without exposing pose edits."""

    if not isinstance(report, dict):
        return {}
    failed_stages = []
    failed_refs: set[str] = set()
    constraint_errors: dict[str, Any] = {}
    warnings: dict[str, Any] = {}
    for stage in report.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("name", ""))
        stage_failed = not bool(stage.get("success", False))
        if stage_failed:
            failed_stages.append(stage_name)
        for source_constraint in stage.get("source_constraints", []):
            if isinstance(source_constraint, str) and stage_failed:
                failed_refs.add(source_constraint)
        errors = stage.get("constraint_errors", {})
        if isinstance(errors, dict) and errors:
            constraint_errors[stage_name] = errors
        stage_warnings = stage.get("warnings", [])
        if isinstance(stage_warnings, list) and stage_warnings:
            warnings[stage_name] = stage_warnings
    binding_sanity = _extract_affordance_binding_sanity(report)
    binding_summary = summarize_affordance_binding_sanity(binding_sanity)
    failed_refs.update(str(ref) for ref in binding_summary.get("failed_refs", []))
    summary = {
        "passed": report.get("passed"),
        "failed_stages": failed_stages,
        "failed_refs": sorted(failed_refs),
        "constraint_errors": constraint_errors,
        "warnings": warnings,
    }
    if binding_summary:
        summary["affordance_binding_sanity"] = binding_summary
    return summary


def _extract_affordance_binding_sanity(report: dict[str, Any]) -> dict[str, Any] | None:
    direct = report.get("affordance_binding_sanity")
    if isinstance(direct, dict):
        return direct
    metadata = report.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("affordance_binding_sanity"), dict):
        return metadata["affordance_binding_sanity"]
    return None


def summarize_affordance_binding_sanity(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract binding-gate evidence for repair prompts."""

    if not isinstance(report, dict):
        return {}
    failed_checks: list[dict[str, Any]] = []
    warning_checks: list[dict[str, Any]] = []
    for check in report.get("checks", []):
        if not isinstance(check, dict) or check.get("passed") is not False:
            if check.get("status") not in {"fail", "warn"}:
                continue
        compact = {
            key: check[key]
            for key in (
                "name",
                "check",
                "binding_id",
                "slot",
                "ref",
                "refs",
                "message",
                "reason",
                "repair_hint",
                "severity",
                "status",
            )
            if key in check
        }
        if check.get("severity") == "error" or check.get("status") == "fail":
            failed_checks.append(compact)
        elif check.get("severity") == "warning" or check.get("status") == "warn":
            warning_checks.append(compact)
    return {
        "passed": report.get("passed"),
        "failed_refs": report.get("failed_refs", []),
        "warning_refs": report.get("warning_refs", []),
        "repair_hints": report.get("repair_hints", []),
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
    }


def summarize_execution_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract execution evidence for program repair prompts.

    The execution report shape is intentionally permissive while SimToolReal
    instrumentation is still evolving. Known keys are copied into a compact
    summary that repair prompts can consume.
    """

    if not isinstance(report, dict):
        return {}
    known_keys = {
        "passed",
        "success",
        "tracking_execution_passed",
        "semantic_execution_passed",
        "failed_stage",
        "failed_stage_reason",
        "execution_failed_stage",
        "semantic_failed_stage",
        "failed_refs",
        "tracking_error",
        "tracking_error_m",
        "contact_evidence",
        "collision_evidence",
        "drop_evidence",
        "dropped",
        "task_report",
        "stage_reports",
        "thresholds",
        "semantic_contact_verification_present",
    }
    summary = {key: report[key] for key in known_keys if key in report}
    if "stage_reports" in summary and isinstance(summary["stage_reports"], list):
        failed = [
            stage.get("stage")
            for stage in summary["stage_reports"]
            if isinstance(stage, dict) and stage.get("success") is False
        ]
        summary.setdefault("failed_stages", [item for item in failed if item])
        semantic_failures = [
            {
                key: stage[key]
                for key in (
                    "goal_index",
                    "stage",
                    "semantic_contact_success",
                    "semantic_contact_error_m",
                    "last_semantic_contact_error_m",
                    "min_semantic_contact_error_m",
                    "semantic_failure_reason",
                    "expected_contact",
                    "contact_point_goal",
                )
                if key in stage
            }
            for stage in summary["stage_reports"]
            if isinstance(stage, dict)
            and stage.get("semantic_contact_required")
            and stage.get("semantic_contact_success") is False
        ]
        if semantic_failures:
            summary["semantic_contact_failures"] = semantic_failures
    return summary


def summarize_tool_path_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract ToolPathSolver evidence for planner repair prompts."""

    if not isinstance(report, dict):
        return {}

    def compact_violation(violation: dict[str, Any]) -> dict[str, Any]:
        return {
            key: violation[key]
            for key in (
                "objective_id",
                "stage",
                "goal_index",
                "goal_stage",
                "sample_fraction",
                "residual",
                "tolerance",
            )
            if key in violation
        }

    def compact_violations(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [
            compact_violation(item)
            for item in value
            if isinstance(item, dict)
        ]

    summary = {
        key: report[key]
        for key in (
            "passed",
            "changed",
            "semantic_goal_preserved",
            "path_objectives_checked",
            "path_objectives_passed",
            "path_objective_ids",
            "path_objective_samples_per_segment",
            "path_objective_tolerance",
            "warnings",
            "repair_hints",
        )
        if key in report
    }
    compact_path_failures = []
    for failure in report.get("path_failures", []):
        if not isinstance(failure, dict):
            continue
        compact = {
            key: failure[key]
            for key in (
                "goal_index",
                "stage",
                "safe_fraction",
                "segment_first_bad_fraction",
                "blocked_by",
            )
            if key in failure
        }
        compact_path_failures.append(compact)
    if compact_path_failures:
        summary["path_failures"] = compact_path_failures
    compacted_violations = compact_violations(report.get("path_objective_violations"))
    if compacted_violations:
        summary["path_objective_violations"] = compacted_violations
    compact_objective_failures = []
    for failure in report.get("path_objective_failures", []):
        if not isinstance(failure, dict):
            continue
        compact = {
            key: failure[key]
            for key in (
                "goal_index",
                "stage",
                "safe_fraction",
            )
            if key in failure
        }
        nested = compact_violations(failure.get("path_objective_violations"))
        if nested:
            compact["path_objective_violations"] = nested
        compact_objective_failures.append(compact)
    if compact_objective_failures:
        summary["path_objective_failures"] = compact_objective_failures
    return summary


def build_failure_report(
    *,
    failed_stage: str = "",
    error_type: str = "",
    error: str = "",
    repair_hint: str | None = None,
    run_id: str | None = None,
    target: str | None = None,
    compiler_report: dict[str, Any] | None = None,
    tool_path_report: dict[str, Any] | None = None,
    execution_report: dict[str, Any] | None = None,
    visual_diagnosis_report: dict[str, Any] | None = None,
    affordance_binding: dict[str, Any] | None = None,
    prompt_path: str | None = None,
    raw_response_path: str | None = None,
) -> dict[str, Any]:
    """Build a normalized repair report from compiler/execution evidence."""

    report: dict[str, Any] = {
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error": error,
    }
    if repair_hint:
        report["repair_hint"] = repair_hint
    if run_id:
        report["run_id"] = run_id
    if target:
        report["target"] = target
    if compiler_report is not None:
        report["compiler_report"] = compiler_report
        report["compiler_summary"] = summarize_compiler_report(compiler_report)
        if not failed_stage:
            failed = report["compiler_summary"].get("failed_stages", [])
            if failed:
                report["failed_stage"] = failed[0]
    if tool_path_report is not None:
        report["tool_path_report"] = tool_path_report
        report["tool_path_summary"] = summarize_tool_path_report(tool_path_report)
        if not report.get("failed_stage") and tool_path_report.get("passed") is not True:
            report["failed_stage"] = "tool_path_solver"
    if execution_report is not None:
        report["execution_report"] = execution_report
        report["execution_summary"] = summarize_execution_report(execution_report)
        if not report.get("failed_stage"):
            report["failed_stage"] = str(report["execution_summary"].get("failed_stage", ""))
    if visual_diagnosis_report is not None:
        report["visual_diagnosis_report"] = visual_diagnosis_report
        report["visual_diagnosis_summary"] = summarize_visual_diagnosis_report(
            visual_diagnosis_report
        )
    if affordance_binding is not None:
        report["affordance_binding"] = affordance_binding
        report["affordance_binding_summary"] = summarize_affordance_binding(affordance_binding)
    if prompt_path:
        report["prompt_path"] = prompt_path
    if raw_response_path:
        report["raw_response_path"] = raw_response_path
    if not report.get("error") and compiler_report is not None:
        report["error"] = "compiler constraints were not satisfied"
        report["error_type"] = report.get("error_type") or "compiler_report"
    if not report.get("error") and tool_path_report is not None:
        report["error"] = "tool path solver reported a path or semantic-goal failure"
        report["error_type"] = report.get("error_type") or "tool_path_solver_report"
    if not report.get("error") and execution_report is not None:
        report["error"] = "execution verifier reported a failure"
        report["error_type"] = report.get("error_type") or "execution_report"
    return enrich_failure_report(report)


def render_repair_prompt(
    *,
    failure_report: dict[str, Any],
    original_prompt: str | None = None,
    raw_response: str | None = None,
    visual_evidence: dict[str, Any] | None = None,
    visual_diagnosis_report: dict[str, Any] | None = None,
    affordance_binding: dict[str, Any] | None = None,
) -> str:
    category = failure_report.get("failure_category") or classify_failure(
        failed_stage=str(failure_report.get("failed_stage", "")),
        error_type=str(failure_report.get("error_type", "")),
        error=str(failure_report.get("error", "")),
    )
    binding_summary = failure_report.get("affordance_binding_summary")
    if not binding_summary and affordance_binding is not None:
        binding_summary = summarize_affordance_binding(affordance_binding)
    payload = {
        "failure_category": category,
        "failed_stage": failure_report.get("failed_stage"),
        "error_type": failure_report.get("error_type"),
        "error": failure_report.get("error"),
        "repair_hint": failure_report.get("repair_hint"),
        "run_id": failure_report.get("run_id"),
        "target": failure_report.get("target"),
        "failed_refs": failure_report.get("failed_refs"),
        "compiler_summary": failure_report.get("compiler_summary"),
        "tool_path_summary": failure_report.get("tool_path_summary"),
        "execution_summary": failure_report.get("execution_summary"),
        "visual_diagnosis": (
            failure_report.get("visual_diagnosis_summary")
            or summarize_visual_diagnosis_report(visual_diagnosis_report)
        ),
        "affordance_binding": binding_summary,
        "visual_evidence": summarize_visual_evidence(
            visual_evidence,
            allowed_uses=("planning_context", "repair_diagnosis"),
        ),
    }
    sections = [
        "You are repairing a PartToolGen restricted Python planning program.",
        "Return exactly one Python module and no prose.",
        "The module must define exactly: def build_program(ctx):",
        "Use only ctx.ref(...) values from the original prompt symbol_table and only the ctx API.",
        "If AffordanceBinding is present, use its contact_ref and slot_bindings as the "
        "active semantic choices; do not choose a different tool function.",
        "Null AffordanceBinding slots are hard vetoes. If function_axis_ref is null, remove "
        "align_axis instead of inventing a fallback tool axis.",
        "If function_axis_ref and a world motion/target axis are available, prefer "
        "align_axis(function_axis_ref, target_axis) for the tool surface normal.",
        "If functional_frame_binding.axis_source_ref and a world motion/target axis are available, "
        "use align_vector_between_points(axis_source_ref, contact_ref, target_axis) only as a "
        "fallback when no reliable function_axis_ref is bound. axis_source_ref is system-derived; "
        "do not invent or replace it.",
        "For align_axis, source_axis must be a tool.functional_parts.*.normal ref or "
        "a tool-frame candidate.surfaces.* ref listed as function_axis_ref; never use "
        "scene.* or target.* as source_axis.",
        "For unknown refs, replace the ref with one of the binding slot refs or bound symbol_table refs; "
        "do not request or assume a wider whitelist.",
        "Repair build_program(ctx) by changing stages, refs, constraints, bounded tolerances, "
        "offsets, or top-level scalar objective functions registered with explicit refs.",
        "If tool_path_summary reports semantic_goal_preserved=false or path objective violations, "
        "repair ProgramIR refs/constraints/objectives and recompile; do not patch raw poses or "
        "fixed-goal JSON.",
        "If using stage.objective(...), the objective function must have signature "
        "objective_name(pose, refs), return a finite scalar whole-tool pose cost, and list every "
        "usable ref in stage.objective(..., refs=[...]). Use kind=\"subgoal\" for terminal pose "
        "costs and kind=\"path\" only for per-sampled-pose ToolPathSolver checks. Do not write "
        "lambda, nested functions, or arbitrary action scripts.",
        "Do not output robot actions, joint commands, controller edits, raw SimToolReal JSON, "
        "direct simulator reads, imports, filesystem/network calls, raw whole-tool pose lists, "
        "or patched numeric poses.",
        "If visual evidence is provided, use it only to diagnose semantic/ref/constraint choices; "
        "numbered markers only resolve to listed refs, and you must not infer raw coordinates, poses, "
        "or simulator state from pixels.",
        "If visual_diagnosis is provided, treat it as repair-stage evidence only. It may suggest "
        "wrong region/point/target/orientation/path categories, but repairs must still be symbolic "
        "changes to binding refs, stages, constraints, tolerances, offsets, or objectives.",
        "",
        "Failure report:",
        json.dumps(payload, indent=2),
    ]
    if original_prompt:
        sections.extend(["", "Original prompt:", original_prompt])
    if raw_response:
        sections.extend(["", "Invalid raw response:", raw_response])
    return "\n".join(sections).rstrip() + "\n"


def write_repair_prompt(
    *,
    failure_report_path: str | Path,
    output_path: str | Path,
    compiler_report_path: str | Path | None = None,
    tool_path_report_path: str | Path | None = None,
    execution_report_path: str | Path | None = None,
    original_prompt_path: str | Path | None = None,
    raw_response_path: str | Path | None = None,
    visual_evidence_path: str | Path | None = None,
    visual_diagnosis_report_path: str | Path | None = None,
    affordance_binding_path: str | Path | None = None,
) -> dict[str, Any]:
    failure_report_path = Path(failure_report_path)
    output_path = Path(output_path)
    report = json.loads(failure_report_path.read_text(encoding="utf-8"))
    if compiler_report_path:
        report["compiler_report"] = json.loads(Path(compiler_report_path).read_text(encoding="utf-8"))
    if tool_path_report_path:
        report["tool_path_report"] = json.loads(
            Path(tool_path_report_path).read_text(encoding="utf-8")
        )
    if execution_report_path:
        report["execution_report"] = json.loads(Path(execution_report_path).read_text(encoding="utf-8"))
    if visual_diagnosis_report_path:
        report["visual_diagnosis_report"] = load_visual_diagnosis_report(
            visual_diagnosis_report_path
        )
    if original_prompt_path:
        report["prompt_path"] = str(original_prompt_path)
    if raw_response_path:
        report["raw_response_path"] = str(raw_response_path)
    if affordance_binding_path:
        report["affordance_binding"] = json.loads(
            Path(affordance_binding_path).read_text(encoding="utf-8")
        )
    enriched = enrich_failure_report(report)
    prompt = render_repair_prompt(
        failure_report=enriched,
        original_prompt=_read_optional_artifact(enriched.get("prompt_path")),
        raw_response=_read_optional_artifact(enriched.get("raw_response_path")),
        visual_evidence=load_visual_evidence(visual_evidence_path),
        visual_diagnosis_report=enriched.get("visual_diagnosis_report"),
        affordance_binding=enriched.get("affordance_binding"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    enriched["repair_prompt_path"] = _display_path(output_path)
    failure_report_path.write_text(json.dumps(enriched, indent=2) + "\n", encoding="utf-8")
    return enriched


def _read_optional_artifact(rel_path: Any) -> str | None:
    if not isinstance(rel_path, str) or not rel_path:
        return None
    path = Path(rel_path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        package_path = Path(__file__).resolve().parents[3] / path
        path = cwd_path if cwd_path.exists() else package_path
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _display_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compiler-report")
    parser.add_argument("--tool-path-report")
    parser.add_argument("--execution-report")
    parser.add_argument("--visual-diagnosis-report")
    parser.add_argument("--original-prompt")
    parser.add_argument("--raw-response")
    parser.add_argument("--visual-evidence")
    parser.add_argument("--affordance-binding")
    args = parser.parse_args()
    write_repair_prompt(
        failure_report_path=args.failure_report,
        output_path=args.output,
        compiler_report_path=args.compiler_report,
        tool_path_report_path=args.tool_path_report,
        execution_report_path=args.execution_report,
        visual_diagnosis_report_path=args.visual_diagnosis_report,
        original_prompt_path=args.original_prompt,
        raw_response_path=args.raw_response,
        visual_evidence_path=args.visual_evidence,
        affordance_binding_path=args.affordance_binding,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
