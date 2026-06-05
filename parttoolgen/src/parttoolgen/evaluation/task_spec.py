"""Load and lightly validate legacy private-eval tool-use benchmark specs.

The active 6.4 image-grounded runners build method-facing tasks directly from
public RGB-D/VLM evidence and execution configs. This module is retained for
older evaluation-suite reports that still use ``private_eval`` and
``target_candidates``. New benchmark code must not use private fixture refs,
families, or primitive lists as method inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TaskSpecError(ValueError):
    """Raised when a benchmark V0 task/suite spec is malformed."""


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskSpecError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskSpecError(f"{name} must be a non-empty string")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TaskSpecError(f"{name} must be a list")
    return value


@dataclass(frozen=True)
class ToolUseTask:
    path: Path
    data: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def family(self) -> str:
        return str(self.data["family"])

    @property
    def split(self) -> str:
        return str(self.data["split"])

    @property
    def instruction(self) -> str:
        return str(self.data["instruction"])

    @property
    def tool_id(self) -> str:
        return str(self.data["tool"]["tool_id"])

    @property
    def public_scene(self) -> dict[str, Any]:
        return self.data["public_scene"]

    @property
    def private_eval(self) -> dict[str, Any]:
        return self.data["private_eval"]


@dataclass(frozen=True)
class ToolUseEvalSuite:
    path: Path
    data: dict[str, Any]
    tasks: list[ToolUseTask]

    @property
    def suite_id(self) -> str:
        return str(self.data["suite_id"])


def validate_public_payload(task: ToolUseTask) -> list[str]:
    """Return public/private leakage warnings for a task spec."""

    warnings: list[str] = []
    public_scene = task.public_scene
    private_eval = task.private_eval
    policy = public_scene.get("leakage_policy") if isinstance(public_scene, dict) else None
    forbidden = []
    if isinstance(policy, dict):
        forbidden = [str(item).lower() for item in policy.get("forbidden_public_fields", [])]
    else:
        forbidden = ["color", "semantic", "oracle", "target_hint", "sim_actor", "asset_name"]
    scrubbed_public_scene = {
        key: value
        for key, value in public_scene.items()
        if key not in {"leakage_policy"}
    }
    public_blob = json.dumps(scrubbed_public_scene, sort_keys=True).lower()
    for token in forbidden:
        if token and re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", public_blob):
            warnings.append(f"public_scene contains forbidden token: {token}")
    target_marker = private_eval.get("target_marker")
    if isinstance(target_marker, str) and target_marker:
        count = sum(1 for row in public_scene.get("target_candidates", []) if row.get("marker") == target_marker)
        if count != 1:
            warnings.append(f"private target_marker {target_marker!r} is not a unique public candidate")
    return warnings


def public_task_payload(task: ToolUseTask) -> dict[str, Any]:
    """Project a private benchmark task into the method-facing task payload."""

    tool = task.data.get("tool", {})
    public_tool = {
        key: tool[key]
        for key in ("tool_id", "tool_prior_dir")
        if isinstance(tool, dict) and key in tool
    }
    public_scene = task.public_scene
    public_candidates = []
    for row in public_scene.get("target_candidates", []):
        if not isinstance(row, dict):
            continue
        public_candidates.append(
            {
                key: json.loads(json.dumps(row[key]))
                for key in (
                    "marker",
                    "target_ref",
                    "contact_refs",
                    "axis_refs",
                    "pre_contact_refs",
                    "post_contact_refs",
                )
                if key in row
            }
        )
    public_scene_payload = {
        "provider": public_scene.get("provider"),
        "scene_evidence": json.loads(json.dumps(public_scene.get("scene_evidence", []))),
        "target_candidates": public_candidates,
        "allowed_primitives": json.loads(json.dumps(public_scene.get("allowed_primitives", []))),
        "public_metrics": json.loads(json.dumps(public_scene.get("public_metrics", []))),
    }
    selection_mode = public_scene.get("target_selection_mode")
    if selection_mode is not None:
        public_scene_payload["target_selection_mode"] = selection_mode

    return {
        "schema_version": "parttoolgen_public_tool_use_task_v0",
        "task_id": task.task_id,
        "family": task.family,
        "split": task.split,
        "instruction": task.instruction,
        "tool": public_tool,
        "public_scene": public_scene_payload,
        "metadata": {
            "private_fields_included": False,
            "created_by": "parttoolgen.evaluation.task_spec.public_task_payload",
        },
    }


def load_task(path: str | Path) -> ToolUseTask:
    task_path = Path(path)
    data = _require_mapping(_load_json(task_path), str(task_path))
    if data.get("schema_version") != "parttoolgen_tool_use_task_v0":
        raise TaskSpecError(f"{task_path} has unsupported schema_version {data.get('schema_version')!r}")
    for key in ("task_id", "family", "split", "instruction"):
        _require_string(data.get(key), key)
    tool = _require_mapping(data.get("tool"), "tool")
    _require_string(tool.get("tool_id"), "tool.tool_id")
    public_scene = _require_mapping(data.get("public_scene"), "public_scene")
    _require_string(public_scene.get("provider"), "public_scene.provider")
    candidates = _require_list(public_scene.get("target_candidates"), "public_scene.target_candidates")
    if not candidates:
        raise TaskSpecError("public_scene.target_candidates must not be empty")
    markers: set[str] = set()
    for index, candidate in enumerate(candidates):
        row = _require_mapping(candidate, f"public_scene.target_candidates[{index}]")
        marker = _require_string(row.get("marker"), f"public_scene.target_candidates[{index}].marker")
        _require_string(row.get("target_ref"), f"public_scene.target_candidates[{index}].target_ref")
        if marker in markers:
            raise TaskSpecError(f"duplicate target marker {marker!r}")
        markers.add(marker)
    private_eval = _require_mapping(data.get("private_eval"), "private_eval")
    _require_list(private_eval.get("success_predicates"), "private_eval.success_predicates")
    task = ToolUseTask(path=task_path, data=data)
    return task


def load_suite(path: str | Path) -> ToolUseEvalSuite:
    suite_path = Path(path)
    data = _require_mapping(_load_json(suite_path), str(suite_path))
    if data.get("schema_version") != "parttoolgen_tool_use_eval_suite_v0":
        raise TaskSpecError(f"{suite_path} has unsupported schema_version {data.get('schema_version')!r}")
    _require_string(data.get("suite_id"), "suite_id")
    task_rows = _require_list(data.get("tasks"), "tasks")
    tasks: list[ToolUseTask] = []
    for index, row_value in enumerate(task_rows):
        row = _require_mapping(row_value, f"tasks[{index}]")
        if row.get("enabled", True) is False:
            continue
        task_path = Path(_require_string(row.get("task_path"), f"tasks[{index}].task_path"))
        if not task_path.is_absolute():
            task_path = suite_path.parent / task_path
        tasks.append(load_task(task_path))
    if not tasks:
        raise TaskSpecError("suite has no enabled tasks")
    return ToolUseEvalSuite(path=suite_path, data=data, tasks=tasks)
