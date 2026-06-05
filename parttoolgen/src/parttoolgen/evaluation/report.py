"""Build task-effect evaluation reports from existing PartToolGen artifacts.

This report builder supports legacy ``private_eval`` task specs. Treat its
private-target metrics as evaluator-side diagnostics only, not as planning or
compilation inputs for the current image-grounded tabletop mainline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .task_spec import ToolUseTask, validate_public_payload


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result_from_predicate(kind: str, passed: bool, value: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
    }


def _find_key(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metric_entry(task_effect_report: dict[str, Any] | None, metric: str) -> Any:
    if not isinstance(task_effect_report, dict) or not metric:
        return None
    metrics = task_effect_report.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        return metrics[metric]
    return task_effect_report.get(metric)


def _metric_value(entry: Any) -> float | None:
    direct = _as_float(entry)
    if direct is not None:
        return direct
    if isinstance(entry, dict):
        for key in ("value", "rate", "score", "mean", "distance_m", "error_rad", "cosine"):
            value = _as_float(entry.get(key))
            if value is not None:
                return value
    return None


def _metric_component(entry: Any, keys: Iterable[str]) -> float | None:
    if isinstance(entry, dict):
        for key in keys:
            value = _as_float(entry.get(key))
            if value is not None:
                return value
    return None


def _tracking_passed(rollout_summary: dict[str, Any] | None, execution_report: dict[str, Any] | None) -> bool:
    if isinstance(execution_report, dict) and execution_report.get("tracking_execution_passed") is not None:
        return bool(execution_report.get("tracking_execution_passed"))
    if isinstance(rollout_summary, dict) and rollout_summary.get("policy_goal_passed") is not None:
        return bool(rollout_summary.get("policy_goal_passed"))
    if isinstance(execution_report, dict) and execution_report.get("passed") is not None:
        return bool(execution_report.get("passed"))
    if isinstance(rollout_summary, dict) and rollout_summary.get("passed") is not None:
        return bool(rollout_summary.get("passed"))
    return False


def _trace_passed(repair_trace: dict[str, Any] | None) -> bool:
    return isinstance(repair_trace, dict) and repair_trace.get("status") == "passed"


def _semantic_execution_passed(rollout_summary: dict[str, Any] | None, execution_report: dict[str, Any] | None) -> bool:
    for report in (execution_report, rollout_summary):
        if isinstance(report, dict) and report.get("semantic_execution_passed") is not None:
            return bool(report.get("semantic_execution_passed"))
    return False


def _rollout_video_present(rollout_summary: dict[str, Any] | None, rollout_summary_path: Path | None) -> tuple[bool, dict[str, Any]]:
    if not isinstance(rollout_summary, dict):
        return False, {"reason": "missing_rollout_summary"}
    frame_count = _as_float(rollout_summary.get("render_video_frames"))
    candidates = [
        rollout_summary.get("render_video_path"),
        rollout_summary.get("video_path"),
    ]
    artifact_refs = rollout_summary.get("artifact_refs")
    if isinstance(artifact_refs, dict):
        candidates.append(artifact_refs.get("rollout_video"))
    checked: list[dict[str, Any]] = []
    for path_value in candidates:
        if not path_value:
            continue
        video_path = Path(str(path_value))
        exists = video_path.exists()
        if not exists and rollout_summary_path is not None and not video_path.is_absolute():
            exists = (rollout_summary_path.parent / video_path.name).exists()
        checked.append({"path": str(path_value), "exists": exists})
        has_frames = frame_count is None or frame_count > 0
        if exists and has_frames:
            return True, {"path": str(path_value), "exists": True, "frames": frame_count, "checked": checked}
    if not checked:
        return False, {"reason": "missing_video_path", "frames": frame_count}
    return False, {"reason": "video_path_missing_or_empty", "frames": frame_count, "checked": checked}


def _target_selection_result(task: ToolUseTask, target_selection: dict[str, Any] | None) -> dict[str, Any]:
    expected_sequence = task.private_eval.get("target_marker_sequence")
    if isinstance(expected_sequence, list) and expected_sequence:
        expected = [str(marker) for marker in expected_sequence]
        selected = []
        if isinstance(target_selection, dict):
            raw_selected = target_selection.get("selected_target_markers") or target_selection.get("selected_markers")
            if isinstance(raw_selected, list):
                selected = [str(marker) for marker in raw_selected]
        return _result_from_predicate(
            "target_sequence_correct",
            selected == expected,
            value=selected,
            threshold=expected,
        )
    expected = task.private_eval.get("target_marker")
    if not expected:
        return _result_from_predicate("target_selection_correct", True, value=None, threshold=None)
    selected = None
    if isinstance(target_selection, dict):
        selected = target_selection.get("selected_marker") or target_selection.get("selected_target_marker")
    return _result_from_predicate(
        "target_selection_correct",
        selected == expected,
        value=selected,
        threshold=expected,
    )


def _contact_point_allowed_result(task: ToolUseTask, predicate: dict[str, Any], task_effect_report: dict[str, Any] | None) -> dict[str, Any]:
    allowed_key = str(predicate.get("allowed_refs_key") or "allowed_contact_point_source_refs")
    allowed = task.private_eval.get(allowed_key, [])
    if not isinstance(allowed, list):
        allowed = []
    selected = None
    if isinstance(task_effect_report, dict):
        selected = (
            task_effect_report.get("selected_contact_point_ref")
            or _find_key(task_effect_report.get("grounding"), "selected_contact_point_ref")
            or _find_key(task_effect_report, "contact_point_source_ref")
            or _find_key(task_effect_report, "function_keypoint_ref")
        )
    return _result_from_predicate(
        "contact_point_allowed",
        isinstance(selected, str) and selected in set(map(str, allowed)),
        value=selected,
        threshold=list(map(str, allowed)),
    )


def _min_metric_result(kind: str, predicate: dict[str, Any], task_effect_report: dict[str, Any] | None, threshold_key: str) -> dict[str, Any]:
    metric = str(predicate.get("metric") or kind)
    entry = _metric_entry(task_effect_report, metric)
    value = _metric_value(entry)
    threshold = _as_float(predicate.get(threshold_key))
    return _result_from_predicate(
        kind,
        value is not None and threshold is not None and value >= threshold,
        value=value if value is not None else "missing_metric",
        threshold={threshold_key: threshold, "metric": metric},
    )


def _max_metric_result(kind: str, predicate: dict[str, Any], task_effect_report: dict[str, Any] | None, threshold_key: str) -> dict[str, Any]:
    metric = str(predicate.get("metric") or kind)
    entry = _metric_entry(task_effect_report, metric)
    value = _metric_value(entry)
    threshold = _as_float(predicate.get(threshold_key))
    return _result_from_predicate(
        kind,
        value is not None and threshold is not None and value <= threshold,
        value=value if value is not None else "missing_metric",
        threshold={threshold_key: threshold, "metric": metric},
    )


def _effect_displacement_result(
    kind: str,
    predicate: dict[str, Any],
    task_effect_report: dict[str, Any] | None,
) -> dict[str, Any]:
    metric = str(predicate.get("metric") or "target_displacement_m")
    entry = _metric_entry(task_effect_report, metric)
    displacement = _metric_value(entry)
    direction_cosine = _metric_component(entry, ("direction_cosine", "axis_cosine", "cosine"))
    min_m = _as_float(predicate.get("min_m"))
    min_cosine = _as_float(predicate.get("min_direction_cosine"))
    passed = (
        displacement is not None
        and min_m is not None
        and displacement >= min_m
        and (min_cosine is None or (direction_cosine is not None and direction_cosine >= min_cosine))
    )
    return _result_from_predicate(
        kind,
        passed,
        value={
            "displacement_m": displacement if displacement is not None else "missing_metric",
            "direction_cosine": direction_cosine if direction_cosine is not None else "missing_metric",
            "metric": metric,
        },
        threshold={"min_m": min_m, "min_direction_cosine": min_cosine},
    )


def _proxy_tool_use_result(
    predicate: dict[str, Any],
    task_effect_report: dict[str, Any] | None,
    rollout_summary: dict[str, Any] | None,
    execution_report: dict[str, Any] | None,
) -> dict[str, Any]:
    csr = _metric_entry(task_effect_report, "csr")
    fdc = _metric_entry(task_effect_report, "fdc")
    tmr = _metric_entry(task_effect_report, "tmr")
    max_semantic_error = _metric_entry(task_effect_report, "max_semantic_contact_error_m")

    min_csr = _as_float(predicate.get("min_contact_rate"))
    min_fdc = _as_float(predicate.get("min_direction_cosine"))
    max_tmr = _as_float(predicate.get("max_tool_misuse_rate"))
    max_contact_error = _as_float(predicate.get("max_contact_error_m"))
    require_tracking = bool(predicate.get("require_tracking", True))
    require_semantic = bool(predicate.get("require_semantic_execution", True))

    csr_value = _metric_component(csr, ("rate", "value"))
    fdc_value = _metric_component(fdc, ("cosine", "value"))
    tmr_value = _metric_component(tmr, ("rate", "value"))
    semantic_error_value = _metric_value(max_semantic_error)
    tracking_passed = _tracking_passed(rollout_summary, execution_report)
    semantic_passed = _semantic_execution_passed(rollout_summary, execution_report)

    checks = {
        "contact_rate": min_csr is None or (csr_value is not None and csr_value >= min_csr),
        "direction_consistency": min_fdc is None or (fdc_value is not None and fdc_value >= min_fdc),
        "tool_misuse": max_tmr is None or (tmr_value is not None and tmr_value <= max_tmr),
        "contact_error": max_contact_error is None
        or (semantic_error_value is not None and semantic_error_value <= max_contact_error),
        "tracking": (not require_tracking) or tracking_passed,
        "semantic_execution": (not require_semantic) or semantic_passed,
    }
    return _result_from_predicate(
        "proxy_tool_use_success",
        all(checks.values()),
        value={
            "checks": checks,
            "contact_rate": csr_value if csr_value is not None else "missing_metric",
            "direction_cosine": fdc_value if fdc_value is not None else "missing_metric",
            "tool_misuse_rate": tmr_value if tmr_value is not None else "missing_metric",
            "max_semantic_contact_error_m": (
                semantic_error_value if semantic_error_value is not None else "missing_metric"
            ),
            "tracking_passed": tracking_passed,
            "semantic_execution_passed": semantic_passed,
        },
        threshold={
            "min_contact_rate": min_csr,
            "min_direction_cosine": min_fdc,
            "max_tool_misuse_rate": max_tmr,
            "max_contact_error_m": max_contact_error,
            "require_tracking": require_tracking,
            "require_semantic_execution": require_semantic,
        },
    )


def build_task_evaluation_report(
    *,
    task: ToolUseTask,
    output_dir: Path,
    rollout_summary_path: Path | None = None,
    execution_report_path: Path | None = None,
    repair_trace_path: Path | None = None,
    target_selection_path: Path | None = None,
    task_effect_report_path: Path | None = None,
) -> dict[str, Any]:
    rollout_summary = _load_json_if_exists(rollout_summary_path)
    execution_report = _load_json_if_exists(execution_report_path)
    repair_trace = _load_json_if_exists(repair_trace_path)
    target_selection = _load_json_if_exists(target_selection_path)
    task_effect_report = _load_json_if_exists(task_effect_report_path)
    predicate_results: list[dict[str, Any]] = []
    predicate_results.append(_target_selection_result(task, target_selection))
    for predicate in task.private_eval.get("success_predicates", []):
        kind = predicate.get("kind")
        if kind == "tracking_passed":
            predicate_results.append(
                _result_from_predicate(
                    kind,
                    _tracking_passed(rollout_summary, execution_report),
                    value=_tracking_passed(rollout_summary, execution_report),
                )
            )
        elif kind == "repair_trace_passed":
            predicate_results.append(
                _result_from_predicate(kind, _trace_passed(repair_trace), value=repair_trace.get("status") if repair_trace else None)
            )
        elif kind == "manual_pending":
            predicate_results.append(_result_from_predicate(kind, False, value="pending"))
        elif kind in {"target_selection_correct", "target_sequence_correct"}:
            continue
        elif kind == "semantic_execution_passed":
            passed = _semantic_execution_passed(rollout_summary, execution_report)
            predicate_results.append(_result_from_predicate(kind, passed, value=passed))
        elif kind == "rollout_video_present":
            passed, value = _rollout_video_present(rollout_summary, rollout_summary_path)
            predicate_results.append(_result_from_predicate(kind, passed, value=value))
        elif kind == "contact_point_allowed":
            predicate_results.append(_contact_point_allowed_result(task, predicate, task_effect_report))
        elif kind == "contact_success_rate":
            predicate_results.append(_min_metric_result(kind, predicate, task_effect_report, "min_rate"))
        elif kind == "pre_contact_alignment_error":
            predicate_results.append(_max_metric_result(kind, predicate, task_effect_report, "max_rad"))
        elif kind == "tool_misuse_rate":
            predicate_results.append(_max_metric_result(kind, predicate, task_effect_report, "max_rate"))
        elif kind == "functional_direction_consistency":
            predicate_results.append(_min_metric_result(kind, predicate, task_effect_report, "min_cosine"))
        elif kind == "language_axis_binding_accuracy":
            predicate_results.append(_min_metric_result(kind, predicate, task_effect_report, "min_score"))
        elif kind in {"strike_effect_displacement", "target_effect_displacement"}:
            predicate_results.append(_effect_displacement_result(kind, predicate, task_effect_report))
        elif kind == "proxy_tool_use_success":
            predicate_results.append(
                _proxy_tool_use_result(predicate, task_effect_report, rollout_summary, execution_report)
            )
        else:
            predicate_results.append(_result_from_predicate(kind, False, value="unsupported_predicate"))

    tracking_passed = _tracking_passed(rollout_summary, execution_report) or _trace_passed(repair_trace)
    task_success = bool(predicate_results) and all(bool(row.get("passed")) for row in predicate_results)
    leakage_warnings = validate_public_payload(task)
    debug_count = None
    if isinstance(repair_trace, dict):
        debug_count = _as_float(repair_trace.get("total_debug_count"))
        if debug_count is None:
            debug_count = _as_float(repair_trace.get("debug_count"))
    failure_category = None
    if leakage_warnings:
        failure_category = "public_private_leakage_warning"
    elif not task_success:
        failed = next((row for row in predicate_results if not row.get("passed")), None)
        failure_category = str(failed.get("kind")) if failed else "task_metric_failed"

    artifact_refs = {
        "task_spec": str(task.path),
        "rollout_summary": str(rollout_summary_path) if rollout_summary_path else None,
        "execution_report": str(execution_report_path) if execution_report_path else None,
        "repair_trace": str(repair_trace_path) if repair_trace_path else None,
        "target_selection": str(target_selection_path) if target_selection_path else None,
        "task_effect_report": str(task_effect_report_path) if task_effect_report_path else None,
    }
    report = {
        "schema_version": "parttoolgen_task_evaluation_report_v0",
        "report_id": f"{task.task_id}_task_eval",
        "task_id": task.task_id,
        "family": task.family,
        "passed": bool(task_success and not leakage_warnings),
        "tracking_passed": bool(tracking_passed),
        "task_success": bool(task_success),
        "failure_category": failure_category,
        "predicate_results": predicate_results,
        "metrics": {
            "predicate_pass_rate": (
                sum(1 for row in predicate_results if row.get("passed")) / float(len(predicate_results))
                if predicate_results
                else 0.0
            ),
            "public_private_leakage_warning_count": len(leakage_warnings),
            "debug_count": debug_count,
        },
        "artifact_refs": artifact_refs,
        "metadata": {
            "split": task.split,
            "tool_id": task.tool_id,
            "instruction": task.instruction,
            "public_private_leakage_warnings": leakage_warnings,
            "created_by": "parttoolgen.evaluation.report",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_evaluation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
