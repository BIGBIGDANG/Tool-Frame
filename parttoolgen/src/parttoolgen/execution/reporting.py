#!/usr/bin/env python3
"""Build execution reports from fixed-goal rollout evidence.

This verifier is PartToolGen-side bookkeeping. It does not change SimToolReal's
policy, observation space, or action space; it only summarizes tracking,
contact, collision, and drop evidence so repair prompts can modify
``build_program(ctx)`` and recompile.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.validation.validate_parttoolgen import (
    load_structured_file,
    validate_simtoolreal_fixed_goals,
    validate_tool_pose_subgoals,
)


DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M = 0.03


def _normalize_quat(q: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(item) * float(item) for item in q))
    if magnitude <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(item) / magnitude for item in q]


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _quat_angle_rad(a: list[float], b: list[float]) -> float:
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = abs(sum(float(x) * float(y) for x, y in zip(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _pose_from_sample(sample: dict[str, Any]) -> list[float] | None:
    for key in ("pose_xyzw", "tool_pose_xyzw", "actual_pose_xyzw"):
        value = sample.get(key)
        if isinstance(value, list) and len(value) == 7:
            return [float(item) for item in value]
    return None


def _events_for_goal(events: list[dict[str, Any]], goal_index: int, stage: str) -> list[dict[str, Any]]:
    matched = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("goal_index") == goal_index or event.get("stage") == stage:
            matched.append(event)
    return matched


def _pose7_close(a: list[float], b: list[float], *, tol: float = 1e-9) -> bool:
    return len(a) == 7 and len(b) == 7 and all(
        abs(float(x) - float(y)) <= tol for x, y in zip(a, b)
    )


def _derive_goal_index_map(
    fixed_goals: dict[str, Any],
    tool_pose_subgoals: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Infer the adapter's explicit start-hold goal mapping."""

    explicit_map = _explicit_goal_index_map(fixed_goals)
    if explicit_map:
        return explicit_map
    if not tool_pose_subgoals:
        return []
    fixed_goal_poses = fixed_goals.get("goals", [])
    source_goals = tool_pose_subgoals.get("goals", [])
    if not isinstance(fixed_goal_poses, list) or not isinstance(source_goals, list):
        return []
    if len(fixed_goal_poses) != len(source_goals) + 1:
        return []
    if not _pose7_close(fixed_goal_poses[0], fixed_goals.get("start_pose", [])):
        return []
    goal_index_map = [
        {
            "fixed_goal_index": 0,
            "source_goal_index": None,
            "stage": "start_hold",
            "source": "simtoolreal_fixed_goals.goals[0]",
        }
    ]
    goal_index_map.extend(
        {
            "fixed_goal_index": index + 1,
            "source_goal_index": index,
            "stage": str(goal.get("stage", f"source_goal_{index}"))
            if isinstance(goal, dict)
            else f"source_goal_{index}",
            "source": "tool_pose_subgoals.goals",
        }
        for index, goal in enumerate(source_goals)
    )
    return goal_index_map


def _explicit_goal_index_map(fixed_goals: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = fixed_goals.get("metadata")
    candidates = []
    if isinstance(metadata, dict):
        candidates.append(metadata.get("goal_index_map"))
    candidates.append(fixed_goals.get("goal_index_map"))
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def _last_sample_for_goal(samples: list[dict[str, Any]], goal_index: int, stage: str) -> dict[str, Any] | None:
    matched = _events_for_goal(samples, goal_index, stage)
    if not matched:
        return None
    return matched[-1]


def _samples_for_goal(samples: list[dict[str, Any]], goal_index: int, stage: str) -> list[dict[str, Any]]:
    return _events_for_goal(samples, goal_index, stage)


def _expected_contact_key_from_metadata(metadata: dict[str, Any]) -> tuple[str, str] | None:
    expected_contact = metadata.get("expected_contact")
    contact_point_goal = metadata.get("contact_point_goal")
    if not isinstance(expected_contact, dict):
        return None
    tool_ref = expected_contact.get("tool_part")
    if not isinstance(tool_ref, str) and isinstance(contact_point_goal, dict):
        tool_ref = contact_point_goal.get("part_ref")
    target_ref = expected_contact.get("scene_target")
    if isinstance(tool_ref, str) and isinstance(target_ref, str):
        return tool_ref, target_ref
    return None


def _expected_contact_key_from_sample(sample: dict[str, Any]) -> tuple[str, str] | None:
    nested = sample.get("semantic_contact_check")
    if isinstance(nested, dict):
        tool_ref = nested.get("tool_part_ref")
        target_ref = nested.get("scene_target_ref")
    else:
        tool_ref = sample.get("contact_point_tool_ref")
        target_ref = sample.get("contact_target_ref")
    if isinstance(tool_ref, str) and isinstance(target_ref, str):
        return tool_ref, target_ref
    return None


def _samples_for_expected_contact(
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    key = _expected_contact_key_from_metadata(metadata)
    if key is None:
        return []
    return [
        sample
        for sample in samples
        if _expected_contact_key_from_sample(sample) == key
    ]


def _dedupe_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for sample in samples:
        key = (sample.get("step"), sample.get("goal_index"), sample.get("stage"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def _subgoal_metadata(
    tool_pose_subgoals: dict[str, Any] | None,
    goal_index: int,
    goal_index_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if goal_index_map:
        mapped = next(
            (
                item
                for item in goal_index_map
                if int(item.get("fixed_goal_index", -1)) == goal_index
            ),
            None,
        )
        if mapped:
            source_index = mapped.get("source_goal_index")
            if source_index is None:
                stage = str(mapped.get("stage", f"goal_{goal_index}"))
                return {
                    "stage": stage,
                    "source_constraints": [str(mapped.get("source", "goal_index_map"))],
                    "expected_contact": {},
                    "contact_point_goal": {},
                    "source_goal_index": None,
                    "semantic_gate": _goal_semantic_gate(None, stage),
                }
            goal_index = int(source_index)
    if not tool_pose_subgoals:
        stage = f"goal_{goal_index}"
        return {
            "stage": stage,
            "source_constraints": [],
            "expected_contact": {},
            "contact_point_goal": {},
            "source_goal_index": goal_index,
            "semantic_gate": _goal_semantic_gate(None, stage),
        }
    goals = tool_pose_subgoals.get("goals", [])
    if goal_index >= len(goals):
        stage = f"goal_{goal_index}"
        return {
            "stage": stage,
            "source_constraints": [],
            "expected_contact": {},
            "contact_point_goal": {},
            "source_goal_index": goal_index,
            "semantic_gate": _goal_semantic_gate(None, stage),
        }
    goal = goals[goal_index]
    stage = str(goal.get("stage", f"goal_{goal_index}"))
    return {
        "stage": stage,
        "source_constraints": goal.get("source_constraints", []),
        "expected_contact": goal.get("expected_contact", {}),
        "contact_point_goal": goal.get("contact_point_goal", {}),
        "source_goal_index": goal_index,
        "semantic_gate": _goal_semantic_gate(goal, stage),
    }


def _goal_semantic_gate(goal: dict[str, Any] | None, stage: str) -> bool:
    if isinstance(goal, dict):
        explicit_gate = goal.get("semantic_gate")
        if isinstance(explicit_gate, bool):
            return explicit_gate
        semantic_role = goal.get("semantic_role")
        if isinstance(semantic_role, str) and semantic_role == "generated_transition_waypoint":
            return False
    return not _stage_is_generated_path_waypoint(stage)


def _stage_is_generated_path_waypoint(stage: str) -> bool:
    return "_transition_" in stage


def _tracking_gate_required(metadata: dict[str, Any]) -> bool:
    """Return whether PartToolGen should verify the realized whole-tool pose.

    Source-less adapter goals such as the inserted start-hold pose may be
    advanced by SimToolReal's own fixed-goal counter before any useful tracking
    sample exists. All compiler-originated goals must pass explicit translation
    and rotation tracking checks; the policy success signal is recorded but does
    not override these geometry errors.
    """

    return metadata.get("source_goal_index") is not None


def _semantic_contact_check_from_sample(
    sample: dict[str, Any] | None,
    *,
    max_functional_target_error_m: float,
) -> dict[str, Any] | None:
    if not isinstance(sample, dict):
        return None
    if sample.get("semantic_gate") is False:
        return None
    nested = sample.get("semantic_contact_check")
    if isinstance(nested, dict):
        if nested.get("checked") is False:
            return {
                "checked": False,
                "passed": False,
                "unchecked_reason": nested.get("unchecked_reason", "semantic_contact_unchecked"),
                "evidence": nested,
            }
        error = nested.get("semantic_contact_error_m")
        threshold = nested.get("threshold_m", max_functional_target_error_m)
        if isinstance(error, (int, float)):
            passed = float(error) <= float(threshold)
            evidence = dict(nested)
            evidence["passed"] = passed
            evidence["threshold_m"] = float(threshold)
            evidence.setdefault("step", sample.get("step"))
            evidence.setdefault("goal_index", sample.get("goal_index"))
            evidence.setdefault("stage", sample.get("stage"))
            return {
                "checked": True,
                "passed": passed,
                "error_m": float(error),
                "threshold_m": float(threshold),
                "evidence": evidence,
            }
    error = sample.get("contact_target_error_m")
    source = "rollout_sample_contact_target_error"
    if not isinstance(error, (int, float)):
        error = sample.get("functional_target_error_m")
        source = "rollout_sample_functional_target_error"
    if isinstance(error, (int, float)):
        passed = float(error) <= float(max_functional_target_error_m)
        return {
            "checked": True,
            "passed": passed,
            "error_m": float(error),
            "threshold_m": float(max_functional_target_error_m),
            "evidence": {
                "checked": True,
                "step": sample.get("step"),
                "goal_index": sample.get("goal_index"),
                "stage": sample.get("stage"),
                "tool_part_ref": sample.get("contact_point_tool_ref"),
                "scene_target_ref": sample.get("contact_target_ref"),
                "contact_point_tool": sample.get("contact_point_tool"),
                "planned_contact_point_world": sample.get("planned_contact_point_world"),
                "actual_contact_point_world": sample.get("actual_contact_point_world"),
                "target_position_world": sample.get("expected_contact_target_world"),
                "semantic_contact_error_m": float(error),
                "threshold_m": float(max_functional_target_error_m),
                "passed": passed,
                "source": source,
            },
        }
    return None


def _semantic_contact_summary(
    samples: list[dict[str, Any]],
    *,
    relation_samples: list[dict[str, Any]] | None = None,
    last_sample: dict[str, Any] | None,
    metadata: dict[str, Any],
    max_functional_target_error_m: float,
) -> dict[str, Any]:
    if metadata.get("semantic_gate") is False:
        return {
            "semantic_contact_required": False,
            "semantic_contact_success": None,
            "semantic_gate": False,
            "semantic_gate_reason": "generated_path_waypoint",
        }
    relation_samples = relation_samples or []
    semantic_samples = _dedupe_samples(samples + relation_samples)
    checks = [
        check
        for sample in semantic_samples
        if (
            check := _semantic_contact_check_from_sample(
                sample,
                max_functional_target_error_m=max_functional_target_error_m,
            )
        )
        is not None
    ]
    last_check = _semantic_contact_check_from_sample(
        last_sample,
        max_functional_target_error_m=max_functional_target_error_m,
    )
    requires_check = bool(metadata.get("expected_contact")) or bool(metadata.get("contact_point_goal"))
    if not checks:
        if requires_check:
            return {
                "semantic_contact_required": True,
                "semantic_contact_success": False,
                "semantic_gate": True,
                "semantic_failure_reason": "semantic_contact_verification_missing",
                "semantic_failure_is_repair_evidence": True,
            }
        return {
            "semantic_contact_required": False,
            "semantic_contact_success": None,
            "semantic_gate": True,
        }
    errors = [check["error_m"] for check in checks if isinstance(check.get("error_m"), float)]
    best_check = min(
        (check for check in checks if isinstance(check.get("error_m"), float)),
        key=lambda item: item["error_m"],
        default=None,
    )
    best_success = any(bool(check.get("passed")) for check in checks)
    last_error = last_check.get("error_m") if last_check else None
    best_error = best_check.get("error_m") if isinstance(best_check, dict) else None
    relation_scope_used = len(semantic_samples) > len(_dedupe_samples(samples))
    semantic_failure_reason = None
    if not best_success:
        semantic_failure_reason = (
            last_check.get("unchecked_reason")
            if last_check and last_check.get("checked") is False
            else "functional_target_error_above_threshold"
        )
    return {
        "semantic_contact_required": True,
        "semantic_contact_success": best_success,
        "semantic_gate": True,
        "semantic_contact_error_m": best_error,
        "last_semantic_contact_error_m": last_error,
        "min_semantic_contact_error_m": min(errors) if errors else None,
        "best_semantic_contact_sample_step": (
            best_check.get("evidence", {}).get("step")
            if isinstance(best_check, dict)
            else None
        ),
        "semantic_contact_evidence": best_check.get("evidence") if isinstance(best_check, dict) else None,
        "last_semantic_contact_evidence": last_check.get("evidence") if last_check else None,
        "semantic_success_policy": (
            "best_sample_for_expected_contact"
            if relation_scope_used
            else "best_sample_within_goal"
        ),
        "semantic_failure_reason": semantic_failure_reason,
        "semantic_failure_is_repair_evidence": not best_success,
    }


def _tool_path_solver_gate(tool_pose_subgoals: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool_pose_subgoals, dict):
        return {}
    sanity = tool_pose_subgoals.get("sanity_checks", {})
    if not isinstance(sanity, dict):
        return {}
    solver = sanity.get("tool_path_solver")
    if not isinstance(solver, dict):
        return {
            "path_objectives_deferred": sanity.get("path_objectives_deferred", 0),
            "tool_path_solver_present": False,
        }
    return {
        "path_objectives_deferred": sanity.get("path_objectives_deferred", 0),
        "tool_path_solver_present": True,
        "checked": solver.get("checked"),
        "passed": solver.get("passed"),
        "semantic_goal_preserved": solver.get("semantic_goal_preserved"),
        "collision_report_passed": solver.get("collision_report_passed"),
        "path_objectives_checked": solver.get("path_objectives_checked"),
        "path_objectives_passed": solver.get("path_objectives_passed"),
        "path_failures": solver.get("path_failures"),
        "path_objective_failures": solver.get("path_objective_failures"),
    }


def build_execution_report(
    *,
    fixed_goals: dict[str, Any],
    tool_pose_subgoals: dict[str, Any] | None = None,
    tracking_samples: list[dict[str, Any]] | None = None,
    contact_events: list[dict[str, Any]] | None = None,
    collision_events: list[dict[str, Any]] | None = None,
    drop_events: list[dict[str, Any]] | None = None,
    report_id: str | None = None,
    max_tracking_error_m: float = 0.035,
    max_rotation_error_rad: float = 0.30,
    max_functional_target_error_m: float = DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M,
    max_contact_point_target_error_m: float | None = None,
    require_contact: bool = False,
    goal_index_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize rollout evidence for repair.

    Evidence lists are expected to be JSON objects keyed by ``goal_index`` or
    ``stage``. Missing tracking samples are treated as verifier failures because
    the execution loop cannot confirm the fixed policy reached the goal.
    """

    validate_simtoolreal_fixed_goals(fixed_goals)
    if tool_pose_subgoals is not None:
        validate_tool_pose_subgoals(tool_pose_subgoals)
    missing_tool_pose_subgoals = tool_pose_subgoals is None
    if not goal_index_map:
        goal_index_map = _derive_goal_index_map(fixed_goals, tool_pose_subgoals)
    if max_contact_point_target_error_m is not None:
        max_functional_target_error_m = float(max_contact_point_target_error_m)
    tracking_samples = tracking_samples or []
    contact_events = contact_events or []
    collision_events = collision_events or []
    drop_events = drop_events or []

    stage_reports: list[dict[str, Any]] = []
    for goal_index, goal_pose in enumerate(fixed_goals["goals"]):
        metadata = _subgoal_metadata(tool_pose_subgoals, goal_index, goal_index_map)
        stage = metadata["stage"]
        samples = _samples_for_goal(tracking_samples, goal_index, stage)
        sample = samples[-1] if samples else None
        actual_pose = _pose_from_sample(sample) if sample else None
        if actual_pose is None:
            tracking_error_m = None
            rotation_error_rad = None
            tracking_success = False
        else:
            tracking_error_m = _dist(goal_pose[:3], actual_pose[:3])
            rotation_error_rad = _quat_angle_rad(goal_pose[3:7], actual_pose[3:7])
            tracking_success = (
                tracking_error_m <= max_tracking_error_m
                and rotation_error_rad <= max_rotation_error_rad
            )
        contacts = _events_for_goal(contact_events, goal_index, stage)
        collisions = _events_for_goal(collision_events, goal_index, stage)
        drops = _events_for_goal(drop_events, goal_index, stage)
        policy_goal_success = any(
            event.get("type") == "simtoolreal_goal_success" for event in contacts
        )
        contact_success = bool(contacts) or not require_contact
        semantic = _semantic_contact_summary(
            samples,
            relation_samples=_samples_for_expected_contact(tracking_samples, metadata),
            last_sample=sample,
            metadata=metadata,
            max_functional_target_error_m=max_functional_target_error_m,
        )
        semantic_contact_success = semantic.get("semantic_contact_success")
        if semantic_contact_success is None:
            semantic_gate_passed = None
        else:
            semantic_gate_passed = bool(semantic_contact_success)
        tracking_required = _tracking_gate_required(metadata)
        if tracking_required:
            tracking_gate_success = bool(tracking_success)
        else:
            tracking_gate_success = bool(tracking_success or policy_goal_success)
        execution_success = (
            tracking_gate_success
            and contact_success
            and not collisions
            and not drops
        )
        stage_report = {
            "goal_index": goal_index,
            "stage": stage,
            "success": execution_success,
            "tracking_error_m": tracking_error_m,
            "rotation_error_rad": rotation_error_rad,
            "tracking_success": tracking_success,
            "tracking_gate_required": tracking_required,
            "tracking_gate_success": tracking_gate_success,
            "policy_goal_success": policy_goal_success,
            "contact_success": contact_success,
            "contact_evidence": contacts,
            "collision_evidence": collisions,
            "drop_evidence": drops,
            "source_constraints": metadata["source_constraints"],
            "expected_contact": metadata["expected_contact"],
            "contact_point_goal": metadata.get("contact_point_goal", {}),
            "source_goal_index": metadata.get("source_goal_index"),
            "semantic_gate_passed": semantic_gate_passed,
        }
        stage_report.update(semantic)
        stage_reports.append(stage_report)

    execution_failed = [stage for stage in stage_reports if not stage["success"]]
    semantic_failed = [
        stage
        for stage in stage_reports
        if stage.get("semantic_contact_success") is False
    ]
    first_execution_failed = execution_failed[0] if execution_failed else None
    first_semantic_failed = semantic_failed[0] if semantic_failed else None
    first_failed = first_execution_failed or first_semantic_failed
    tracking_execution_passed = not execution_failed
    semantic_execution_passed = (not semantic_failed) and not missing_tool_pose_subgoals
    passed = tracking_execution_passed and semantic_execution_passed
    failed_refs: list[str] = []
    if missing_tool_pose_subgoals:
        failed_refs.append("missing_tool_pose_subgoals")
    if first_failed:
        failed_refs.extend(str(item) for item in first_failed.get("source_constraints", []))
        expected_contact = first_failed.get("expected_contact", {})
        if isinstance(expected_contact, dict):
            failed_refs.extend(str(item) for item in expected_contact.values() if item)
    return {
        "schema_version": "v0.1",
        "report_id": report_id or f"{fixed_goals['metadata']['source_subgoals_id']}_execution_report",
        "source_subgoals_id": fixed_goals["metadata"]["source_subgoals_id"],
        "passed": passed,
        "success": passed,
        "tracking_execution_passed": tracking_execution_passed,
        "semantic_execution_passed": semantic_execution_passed,
        "failed_stage": first_failed.get("stage") if first_failed else None,
        "failed_stage_reason": (
            "tracking_failure"
            if first_execution_failed
            else ("semantic_contact_failure" if first_semantic_failed else None)
        ),
        "execution_failed_stage": (
            first_execution_failed.get("stage") if first_execution_failed else None
        ),
        "semantic_failed_stage": (
            first_semantic_failed.get("stage") if first_semantic_failed else None
        ),
        "failed_refs": failed_refs,
        "stage_reports": stage_reports,
        "thresholds": {
            "max_tracking_error_m": max_tracking_error_m,
            "max_rotation_error_rad": max_rotation_error_rad,
            "max_functional_target_error_m": max_functional_target_error_m,
            "require_contact": require_contact,
        },
        "metadata": {
            "generated_by": "parttoolgen.execution.reporting",
            "source": "SimToolRealFixedGoals+rollout_evidence",
            "policy_modified": False,
            "observed_scene_graph_used_for_verification_only": True,
            "semantic_contact_verification_present": any(
                stage.get("semantic_contact_required") for stage in stage_reports
            ),
            "missing_tool_pose_subgoals": missing_tool_pose_subgoals,
            "goal_index_map_applied": bool(goal_index_map),
            "tool_path_solver_gate": _tool_path_solver_gate(tool_pose_subgoals),
        },
    }


def _load_event_list(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = load_structured_file(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return [item for item in data["events"] if isinstance(item, dict)]
    raise ValueError(f"expected an event list or object with events: {path}")


def _load_goal_index_map(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = load_structured_file(path)
    goal_index_map = data.get("goal_index_map") if isinstance(data, dict) else None
    if not isinstance(goal_index_map, list):
        return []
    return [item for item in goal_index_map if isinstance(item, dict)]


def write_execution_report(
    *,
    fixed_goals_path: str | Path,
    output_path: str | Path,
    subgoals_path: str | Path | None = None,
    tracking_samples_path: str | Path | None = None,
    contact_events_path: str | Path | None = None,
    collision_events_path: str | Path | None = None,
    drop_events_path: str | Path | None = None,
    repair_report_path: str | Path | None = None,
    max_tracking_error_m: float = 0.035,
    max_rotation_error_rad: float = 0.30,
    max_functional_target_error_m: float = DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M,
    require_contact: bool = False,
) -> dict[str, Any]:
    report = build_execution_report(
        fixed_goals=load_structured_file(fixed_goals_path),
        tool_pose_subgoals=load_structured_file(subgoals_path) if subgoals_path else None,
        tracking_samples=_load_event_list(tracking_samples_path),
        contact_events=_load_event_list(contact_events_path),
        collision_events=_load_event_list(collision_events_path),
        drop_events=_load_event_list(drop_events_path),
        max_tracking_error_m=max_tracking_error_m,
        max_rotation_error_rad=max_rotation_error_rad,
        max_functional_target_error_m=max_functional_target_error_m,
        require_contact=require_contact,
        goal_index_map=_load_goal_index_map(repair_report_path),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-goals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subgoals")
    parser.add_argument("--tracking-samples")
    parser.add_argument("--contact-events")
    parser.add_argument("--collision-events")
    parser.add_argument("--drop-events")
    parser.add_argument("--repair-report")
    parser.add_argument("--max-tracking-error-m", type=float, default=0.035)
    parser.add_argument("--max-rotation-error-rad", type=float, default=0.30)
    parser.add_argument(
        "--max-functional-target-error-m",
        type=float,
        default=DEFAULT_MAX_FUNCTIONAL_TARGET_ERROR_M,
    )
    parser.add_argument("--require-contact", action="store_true")
    args = parser.parse_args()
    write_execution_report(
        fixed_goals_path=args.fixed_goals,
        output_path=args.output,
        subgoals_path=args.subgoals,
        tracking_samples_path=args.tracking_samples,
        contact_events_path=args.contact_events,
        collision_events_path=args.collision_events,
        drop_events_path=args.drop_events,
        repair_report_path=args.repair_report,
        max_tracking_error_m=args.max_tracking_error_m,
        max_rotation_error_rad=args.max_rotation_error_rad,
        max_functional_target_error_m=args.max_functional_target_error_m,
        require_contact=args.require_contact,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
