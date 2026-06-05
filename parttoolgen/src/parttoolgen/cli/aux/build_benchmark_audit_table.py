#!/usr/bin/env python3
"""Build a layered audit table from benchmark summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


UNKNOWN = "unknown"
NOT_RECORDED = "not_recorded"
PASS = "pass"
FAIL = "fail"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_status(value: Any) -> str:
    if value is True:
        return PASS
    if value is False:
        return FAIL
    return NOT_RECORDED


def _nested(data: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _task_family_from_path(path: Path) -> str:
    parts = path.parts
    if "marker_trace" in parts:
        return "marker_trace"
    if "full_chain" in parts:
        index = parts.index("full_chain")
        if index + 1 < len(parts):
            return parts[index + 1]
    for marker in ("target_sweep", "rerun_promptfix", "rerun_assetfix"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return "/".join(parts[-4:-1])


def _task_variant_from_path(path: Path) -> str:
    parts = path.parts
    if "marker_trace" in parts:
        index = parts.index("marker_trace")
        if index + 1 < len(parts):
            return parts[index + 1]
    if "full_chain" in parts:
        index = parts.index("full_chain")
        if index + 2 < len(parts):
            return parts[index + 2]
    for marker in ("target_sweep", "rerun_promptfix", "rerun_assetfix"):
        if marker in parts:
            index = parts.index(marker)
            if index + 2 < len(parts):
                return parts[index + 2]
    return path.parent.name


def _run_group_from_path(path: Path) -> str:
    try:
        index = path.parts.index("test002")
    except ValueError:
        return str(path.parent)
    return "/".join(path.parts[index + 1 : -1])


def _perception_status(trial: Mapping[str, Any]) -> str:
    layer_passed = _nested(trial, "failure_attribution", "layers", "vlm_scene_grounding", "passed")
    if layer_passed is not None:
        return _bool_status(layer_passed)
    if trial.get("failure_category") == "vlm_scene_marking_error":
        return FAIL
    if trial.get("target_object") is not None or _nested(trial, "artifacts", "grounding_overlay"):
        return UNKNOWN
    if trial.get("scene_keypoints_overlay") or trial.get("evidence_manifest"):
        return UNKNOWN
    return NOT_RECORDED


def _tool_grounding_status(trial: Mapping[str, Any]) -> str:
    layer_passed = _nested(trial, "failure_attribution", "layers", "tool_grounding", "passed")
    if layer_passed is not None:
        return _bool_status(layer_passed)
    contact_face = _first_present(
        trial.get("selected_contact_point_face"),
        trial.get("selected_contact_point_face_marker"),
    )
    contact_ref = _first_present(
        trial.get("selected_contact_point_ref"),
        trial.get("selected_contact_point_source_ref"),
    )
    motion_face = _first_present(
        trial.get("selected_motion_axis_face"),
        trial.get("selected_motion_axis_face_marker"),
    )
    if contact_face or contact_ref or motion_face:
        return PASS if contact_face or contact_ref else UNKNOWN
    if trial.get("tool_ref_candidate_selection") is not None:
        return UNKNOWN
    return NOT_RECORDED


def _compile_status(trial: Mapping[str, Any]) -> str:
    value = _first_present(
        trial.get("compile_path_solver_passed"),
        _nested(trial, "failure_attribution", "layers", "constraint_compile", "passed"),
    )
    if value is not None:
        return _bool_status(value)
    pipeline_status = trial.get("pipeline_status")
    if isinstance(pipeline_status, str) and pipeline_status:
        return PASS if "compile" not in pipeline_status.lower() else UNKNOWN
    return NOT_RECORDED


def _execution_status(trial: Mapping[str, Any]) -> str:
    value = _first_present(
        trial.get("tracking_execution_passed"),
        trial.get("strict_execution_passed"),
        _nested(trial, "failure_attribution", "layers", "rollout_policy", "tracking_execution_passed"),
    )
    if value is not None:
        return _bool_status(value)
    policy_goal = trial.get("policy_goal_passed")
    if policy_goal is not None:
        return _bool_status(policy_goal)
    if trial.get("num_steps") is not None or trial.get("reached_goal_count") is not None:
        return UNKNOWN
    return NOT_RECORDED


def _task_success_status(trial: Mapping[str, Any]) -> str:
    wipe = _nested(trial, "task_success_check", "diagnostics", "wipe_contact")
    if isinstance(wipe, Mapping) and wipe.get("coverage_fraction") is not None:
        try:
            return PASS if float(wipe["coverage_fraction"]) >= 0.30 else FAIL
        except (TypeError, ValueError):
            return UNKNOWN
    value = _first_present(
        trial.get("passed"),
        trial.get("task_success_passed"),
        trial.get("task_effect_passed"),
        trial.get("insert_semantic_passed"),
        trial.get("wipe_semantic_passed"),
        trial.get("hammer_semantic_contact_passed"),
        trial.get("vlm_judge_success"),
        _nested(trial, "task_success_check", "passed"),
    )
    if value is not None:
        return _bool_status(value)
    if trial.get("erased_area_fraction") is not None:
        return UNKNOWN
    return NOT_RECORDED


def _primary_failure(trial: Mapping[str, Any], layers: Mapping[str, str]) -> str:
    if layers.get("task_success") == PASS:
        return "success"
    category = _nested(trial, "failure_attribution", "primary_category")
    if isinstance(category, str) and category:
        return category
    category = trial.get("failure_category")
    if isinstance(category, str) and category:
        return category
    for key in ("perception", "tool_grounding", "compile", "execution", "task_success"):
        if layers.get(key) == FAIL:
            return f"{key}_failure"
    return "needs_review"


def _metric_summary(trial: Mapping[str, Any]) -> str:
    bits: list[str] = []
    metrics = trial.get("screwdriver_insert_metrics")
    if isinstance(metrics, Mapping):
        radial = metrics.get("tip_radial_error_m")
        depth = metrics.get("tip_insertion_depth_m")
        if radial is not None:
            bits.append(f"tip_radial_m={float(radial):.5f}")
        if depth is not None:
            bits.append(f"inside_depth_m={float(depth):.5f}")
    effect = trial.get("effect")
    if isinstance(effect, Mapping):
        disp = effect.get("target_xy_displacement_m")
        proj = effect.get("projected_displacement_m")
        if disp is not None:
            bits.append(f"target_xy_disp_m={float(disp):.5f}")
        if proj is not None:
            bits.append(f"projected_disp_m={float(proj):.5f}")
    wipe = _nested(trial, "task_success_check", "diagnostics", "wipe_contact")
    if isinstance(wipe, Mapping):
        coverage = wipe.get("coverage_fraction")
        contacts = wipe.get("contact_sample_count")
        if coverage is not None:
            bits.append(f"wipe_coverage={float(coverage):.3f}")
        if contacts is not None:
            bits.append(f"contact_samples={contacts}")
    erased = trial.get("erased_area_fraction")
    if erased is not None:
        bits.append(f"erased_area_fraction={float(erased):.3f}")
    contact_samples = trial.get("contact_sample_count")
    if contact_samples is not None:
        bits.append(f"marker_contact_samples={contact_samples}")
    ink_segments = trial.get("ink_segment_count")
    if ink_segments is not None:
        bits.append(f"ink_segments={ink_segments}")
    ink_length = trial.get("ink_length_m")
    if ink_length is not None:
        bits.append(f"ink_length_m={float(ink_length):.3f}")
    judge_conf = trial.get("vlm_judge_confidence")
    if judge_conf is not None:
        bits.append(f"judge_conf={float(judge_conf):.2f}")
    return "; ".join(bits)


def _trial_rows(summary_path: Path, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    trials = summary.get("trials")
    if not isinstance(trials, list):
        return rows
    for index, trial in enumerate(trials):
        if not isinstance(trial, Mapping):
            continue
        layers = {
            "perception": _perception_status(trial),
            "tool_grounding": _tool_grounding_status(trial),
            "compile": _compile_status(trial),
            "execution": _execution_status(trial),
            "task_success": _task_success_status(trial),
        }
        seed = trial.get("seed")
        row = {
            "summary_path": str(summary_path),
            "run_group": _run_group_from_path(summary_path),
            "task_family": _task_family_from_path(summary_path),
            "task_variant": _task_variant_from_path(summary_path),
            "task_id": trial.get("task_id"),
            "trial_index": trial.get("index", index),
            "seed": seed,
            **layers,
            "primary_failure": _primary_failure(trial, layers),
            "failure_category": trial.get("failure_category"),
            "policy_goal_passed": trial.get("policy_goal_passed"),
            "tracking_execution_passed": trial.get("tracking_execution_passed"),
            "semantic_execution_passed": trial.get("semantic_execution_passed"),
            "final_successes": _first_present(trial.get("final_successes"), trial.get("reached_goal_count")),
            "num_goals": trial.get("num_goals"),
            "num_steps": trial.get("num_steps"),
            "selected_contact_point_face": trial.get("selected_contact_point_face"),
            "selected_motion_axis_face": trial.get("selected_motion_axis_face"),
            "semantic_failed_stage": trial.get("semantic_failed_stage"),
            "semantic_failure_reason": trial.get("semantic_failure_reason"),
            "metrics": _metric_summary(trial),
            "rollout_video": _first_present(
                _nested(trial, "artifacts", "rollout_video"),
                trial.get("side_board_front_render_video_path"),
            ),
            "rollout_topdown_video": _nested(trial, "artifacts", "rollout_topdown_video"),
        }
        rows.append(row)
    return rows


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_group",
        "task_family",
        "task_variant",
        "task_id",
        "trial_index",
        "seed",
        "perception",
        "tool_grounding",
        "compile",
        "execution",
        "task_success",
        "primary_failure",
        "failure_category",
        "policy_goal_passed",
        "tracking_execution_passed",
        "semantic_execution_passed",
        "final_successes",
        "num_goals",
        "num_steps",
        "selected_contact_point_face",
        "selected_motion_axis_face",
        "semantic_failed_stage",
        "semantic_failure_reason",
        "metrics",
        "rollout_video",
        "rollout_topdown_video",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_group",
        "task_family",
        "task_variant",
        "seed",
        "perception",
        "tool_grounding",
        "compile",
        "execution",
        "task_success",
        "primary_failure",
        "metrics",
    ]
    lines = [
        "# Benchmark Audit Table",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("test002"))
    parser.add_argument("--output-prefix", type=Path, default=Path("test002/benchmark_audit_table"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(args.root.rglob("summary.json")):
        data = _load_json(summary_path)
        if isinstance(data, Mapping):
            rows.extend(_trial_rows(summary_path, data))

    rows.sort(key=lambda row: (str(row.get("run_group")), str(row.get("seed")), str(row.get("trial_index"))))
    _write_json(args.output_prefix.with_suffix(".json"), rows)
    _write_csv(args.output_prefix.with_suffix(".csv"), rows)
    _write_markdown(args.output_prefix.with_suffix(".md"), rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "json": str(args.output_prefix.with_suffix(".json")),
                "csv": str(args.output_prefix.with_suffix(".csv")),
                "markdown": str(args.output_prefix.with_suffix(".md")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
