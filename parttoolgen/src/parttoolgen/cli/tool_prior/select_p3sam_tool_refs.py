#!/usr/bin/env python3
"""Select ToolFrame tool refs from P3SAM visual evidence.

This is the image-grounded tool-side entry point: P3SAM proposes tool parts and
candidate contact refs, a VLM or reference selection chooses the task-facing
part, contact point, and tool motion axis, and the script emits the same public
tool-ref files consumed by the clean image-grounded compiler.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.constraints.vlm_client import build_vlm_client  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _strip_json_response(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _canonical_marker(value: Any, *, prefix: str) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    upper = text.upper()
    if upper.startswith(prefix.upper()):
        return upper
    if text.isdigit():
        return f"{prefix.upper()}{text}"
    return text


def _metadata_table(visual_evidence: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    metadata = visual_evidence.get("metadata")
    rows = metadata.get(key) if isinstance(metadata, Mapping) else None
    return [dict(row) for row in rows or [] if isinstance(row, Mapping)]


def _is_internal_tool_surface(row: Mapping[str, Any]) -> bool:
    if row.get("is_internal_tool_surface") is True:
        return True
    visibility = str(row.get("tool_surface_visibility") or "").strip().lower()
    if visibility in {"internal", "tool_internal", "inside_tool", "occluded_internal"}:
        return True
    exposure = row.get("tool_surface_exposure")
    return isinstance(exposure, Mapping) and exposure.get("is_internal") is True


def _bbox_axis_from_patch_id(patch_id: Any) -> list[float] | None:
    text = str(patch_id or "")
    if not text:
        return None
    axis = text[0].lower()
    value = -1.0 if "negative" in text else 1.0
    if axis == "x":
        return [value, 0.0, 0.0]
    if axis == "y":
        return [0.0, value, 0.0]
    if axis == "z":
        return [0.0, 0.0, value]
    return None


def _axis_for_face_row(row: Mapping[str, Any]) -> tuple[list[float], str]:
    bbox_axis = row.get("bbox_axis_tool")
    if isinstance(bbox_axis, list) and len(bbox_axis) == 3:
        return [float(item) for item in bbox_axis], "bbox_axis_tool"
    patch_axis = _bbox_axis_from_patch_id(row.get("patch_id"))
    if (
        patch_axis is not None
        and str(row.get("selection_semantics") or "").endswith("bbox_face")
        and str(row.get("patch_id") or "").endswith("_endpoint")
    ):
        return patch_axis, "bbox_endpoint_axis"
    normal = row.get("normal_tool")
    if isinstance(normal, list) and len(normal) == 3:
        return [float(item) for item in normal], "mesh_patch_normal"
    marker = row.get("contact_marker") or row.get("face_marker") or row.get("marker")
    raise ValueError(
        "selected P3SAM contact row has no explicit tool-axis evidence "
        f"(marker={marker!r}, surface_ref={row.get('surface_ref')!r}, patch_id={row.get('patch_id')!r})"
    )


def _part_rows(visual_evidence: Mapping[str, Any], tool_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _metadata_table(visual_evidence, "region_marker_table"):
        if row.get("tool_id") not in {None, tool_id}:
            continue
        if row.get("marker") is None or not row.get("part_ref"):
            continue
        rows.append(row)
    return rows


def _contact_rows(
    visual_evidence: Mapping[str, Any],
    *,
    tool_id: str,
    part_marker: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _metadata_table(visual_evidence, "contact_marker_table"):
        if row.get("tool_id") not in {None, tool_id}:
            continue
        if str(row.get("part_marker")) != str(part_marker):
            continue
        if _is_internal_tool_surface(row):
            continue
        if not row.get("point_ref") or not row.get("surface_ref"):
            continue
        rows.append(row)
    return rows


def _row_by_marker(rows: list[dict[str, Any]], key: str, prefix: str) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = str(row.get(key) or "").strip()
        canonical = _canonical_marker(raw, prefix=prefix)
        unprefixed = canonical
        if unprefixed.upper().startswith(prefix.upper()):
            unprefixed = unprefixed[len(prefix) :]
        for marker in {raw, canonical, unprefixed}:
            if marker:
                table[marker] = row
    return table


def _image_items(
    visual_evidence: Mapping[str, Any],
    *,
    tool_id: str,
    part_marker: str | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for item in visual_evidence.get("items", []):
        if not isinstance(item, Mapping):
            continue
        kind = item.get("kind")
        if kind in {"whole_mesh_part_choice_overview", "isolated_part_choice_overview"}:
            if str(item.get("id", "")).startswith(f"{tool_id}_"):
                paths.append(Path(str(item.get("path"))))
        elif part_marker is not None and kind == "isolated_part_card":
            if str(item.get("part_marker") or "") == str(part_marker) or str(item.get("id", "")).startswith(
                f"{tool_id}_part_candidate_{part_marker}"
            ):
                paths.append(Path(str(item.get("path"))))
        elif part_marker is not None and kind in {"selected_part_bbox_overview", "selected_part_face_highlight"}:
            if str(item.get("part_marker") or "") == str(part_marker) or str(item.get("id", "")).startswith(
                f"{tool_id}_part_candidate_{part_marker}"
            ):
                paths.append(Path(str(item.get("path"))))
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _part_prompt(tool_id: str, task_instruction: str, rows: list[dict[str, Any]]) -> str:
    compact_rows = [
        {
            "marker": str(row.get("marker")),
            "part_ref": row.get("part_ref"),
            "part_id": row.get("part_id"),
            "role_candidates": row.get("role_candidates"),
        }
        for row in rows
    ]
    return (
        "Select the task-facing tool part from P3SAM part evidence.\n"
        "Use only the listed marker; do not invent coordinates or tool categories.\n"
        "Return exactly one JSON object:\n"
        "{\n"
        '  "selected_marker": "1",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short visual/task reason"\n'
        "}\n\n"
        f"tool_id: {tool_id}\n"
        f"task_instruction: {task_instruction}\n"
        f"valid_parts:\n{json.dumps(compact_rows, indent=2)}\n"
    )


def _contact_prompt(
    tool_id: str,
    task_instruction: str,
    part_marker: str,
    rows: list[dict[str, Any]],
) -> str:
    compact_rows = [
        {
            "contact_point_marker": row.get("contact_marker") or row.get("face_marker"),
            "point_ref": row.get("point_ref"),
            "surface_ref": row.get("surface_ref"),
            "polarity": row.get("polarity"),
            "patch_id": row.get("patch_id"),
        }
        for row in rows
    ]
    return (
        "Select two task-conditioned tool refs on the chosen P3SAM tool part.\n"
        "The listed candidates have already been filtered to exclude faces marked as internal to the tool; "
        "do not request or infer unlisted internal faces.\n"
        "1) selected_contact_point_marker: the tool point/face center that should be point-matched to the scene "
        "target contact point because it is the active tool reference that physically acts on the target during use. "
        "This is a tool ref, not a scene start-edge or path-direction choice.\n"
        "Also list candidate_contact_point_markers: every listed contact point candidate that is semantically valid "
        "for this task and could serve as the same active tool reference after downstream pose optimization. "
        "Include the selected marker in this list. For tasks involving wiping, erasing, rubbing, scrubbing, "
        "polishing, spreading, pressing, stamping, or area contact, broad exposed working surfaces on the "
        "functional part are valid contact candidates; keep symmetric or opposite working faces when either "
        "face could physically act on the target under a feasible tool pose. Contact candidates are not "
        "motion-axis candidates: do not exclude a valid working face merely because its normal is not the "
        "use-stage travel direction. A listed face may appear in both candidate_contact_point_markers and "
        "candidate_motion_axis_markers when it is semantically valid for both roles. For thin, flat, or "
        "geometrically near-degenerate functional parts where several highlighted bbox faces correspond to "
        "the same exposed working sheet, keep all visually equivalent working-surface markers as contact "
        "candidates instead of forcing one label. Do not list every geometric face; include only candidates "
        "justified by the task and visual evidence.\n"
        "2) selected_motion_axis_marker: the tool face whose normal or declared axis should be aligned parallel to the "
        "VLM scene motion axis during use. This may differ from the contact point marker.\n"
        "Also list candidate_motion_axis_markers: every listed face/axis candidate that is semantically valid "
        "for motion-axis alignment during this task. Include the selected marker in this list. "
        "For symmetric, opposite, or interchangeable axis faces, keep all valid alternatives instead of forcing "
        "a single geometry.\n"
        "Make both choices from the task instruction and the visual evidence only; do not assume a fixed tool-specific rule. "
        "Do not choose the motion-axis marker as the contact point unless the same candidate is also the active tool "
        "reference that should physically act on the target.\n"
        "Use only listed contact_point_marker values. Return exactly one JSON object:\n"
        "{\n"
        '  "selected_contact_point_marker": "F1",\n'
        '  "candidate_contact_point_markers": ["F1"],\n'
        '  "selected_motion_axis_marker": "F2",\n'
        '  "candidate_motion_axis_markers": ["F2"],\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short visual/task reason"\n'
        "}\n\n"
        f"tool_id: {tool_id}\n"
        f"task_instruction: {task_instruction}\n"
        f"selected_part_marker: {part_marker}\n"
        f"valid_contact_point_candidates:\n{json.dumps(compact_rows, indent=2)}\n"
    )


def _vlm_json(
    *,
    prompt: str,
    images: list[Path],
    task_instruction: str,
    context_id: str,
    model: str,
    base_url: str | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    client = build_vlm_client(
        response_mode="openai-compatible",
        image_paths=images,
        model=model,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
    )
    response = client.generate(
        prompt=prompt,
        task_instruction=task_instruction,
        context_id=context_id,
        system_prompt="Return only the requested JSON object. Do not include prose.",
    )
    data = json.loads(_strip_json_response(response.raw_text))
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    data["_raw_response"] = {
        "raw_text": response.raw_text,
        "metadata": response.metadata,
    }
    return data


def _select_part(
    *,
    visual_evidence: Mapping[str, Any],
    tool_id: str,
    task_instruction: str,
    selection_source: str,
    marker: str | None,
    model: str,
    base_url: str | None,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _part_rows(visual_evidence, tool_id)
    if not rows:
        raise ValueError(f"visual evidence has no P3SAM part rows for {tool_id}")
    by_marker = _row_by_marker(rows, "marker", "P")
    raw: dict[str, Any]
    if selection_source == "vlm":
        raw = _vlm_json(
            prompt=_part_prompt(tool_id, task_instruction, rows),
            images=_image_items(visual_evidence, tool_id=tool_id),
            task_instruction=task_instruction,
            context_id=f"{tool_id}:p3sam_part_selection",
            model=model,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
        )
        marker = str(raw.get("selected_marker") or "").strip()
    else:
        if marker is None:
            raise ValueError("--part-marker is required for reference selection")
        raw = {"selected_marker": marker, "confidence": 1.0, "reason": "reference P3SAM part selection"}
    if marker not in by_marker:
        raise ValueError(f"selected part marker must be one of {sorted(by_marker)}; got {marker!r}")
    return by_marker[marker], raw


def _select_contact(
    *,
    visual_evidence: Mapping[str, Any],
    tool_id: str,
    task_instruction: str,
    part_marker: str,
    selection_source: str,
    marker: str | None,
    candidate_markers: list[str] | None,
    axis_marker: str | None = None,
    candidate_axis_markers: list[str] | None = None,
    model: str,
    base_url: str | None,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _contact_rows(visual_evidence, tool_id=tool_id, part_marker=part_marker)
    if not rows:
        raise ValueError(f"visual evidence has no P3SAM contact rows for {tool_id} part marker {part_marker}")
    by_marker = _row_by_marker(rows, "contact_marker", "F")
    raw: dict[str, Any]
    if selection_source == "vlm":
        raw = _vlm_json(
            prompt=_contact_prompt(tool_id, task_instruction, part_marker, rows),
            images=_image_items(visual_evidence, tool_id=tool_id, part_marker=part_marker),
            task_instruction=task_instruction,
            context_id=f"{tool_id}:p3sam_contact_selection",
            model=model,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
        )
        marker = _canonical_marker(
            raw.get("selected_contact_point_marker") or "",
            prefix="F",
        )
        axis_marker = _canonical_marker(
            raw.get("selected_motion_axis_marker") or "",
            prefix="F",
        )
        raw_candidates = raw.get("candidate_contact_point_markers")
        raw_axis_candidates = raw.get("candidate_motion_axis_markers")
        if not marker:
            raise ValueError("VLM response must include selected_contact_point_marker")
        if not axis_marker:
            raise ValueError("VLM response must include selected_motion_axis_marker")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("VLM response must include non-empty candidate_contact_point_markers")
        if not isinstance(raw_axis_candidates, list) or not raw_axis_candidates:
            raise ValueError("VLM response must include non-empty candidate_motion_axis_markers")
        candidate_markers = [
            _canonical_marker(item, prefix="F") for item in raw_candidates if str(item).strip()
        ]
        candidate_axis_markers = [
            _canonical_marker(item, prefix="F") for item in raw_axis_candidates if str(item).strip()
        ]
    else:
        if marker is None:
            raise ValueError("--contact-point-marker is required for reference selection")
        marker = _canonical_marker(marker, prefix="F")
        axis_marker = _canonical_marker(axis_marker or marker, prefix="F")
        candidate_markers = [
            _canonical_marker(item, prefix="F") for item in (candidate_markers or [marker])
        ]
        candidate_axis_markers = [
            _canonical_marker(item, prefix="F") for item in (candidate_axis_markers or [axis_marker])
        ]
        raw = {
            "selected_contact_point_marker": marker,
            "candidate_contact_point_markers": candidate_markers,
            "selected_motion_axis_marker": axis_marker,
            "candidate_motion_axis_markers": candidate_axis_markers,
            "confidence": 1.0,
            "reason": "reference P3SAM contact-point and motion-axis selection",
        }
    if marker not in by_marker:
        raise ValueError(f"selected contact point marker must be one of {sorted(by_marker)}; got {marker!r}")
    if axis_marker not in by_marker:
        raise ValueError(f"selected motion-axis marker must be one of {sorted(by_marker)}; got {axis_marker!r}")
    candidate_markers = list(candidate_markers or [marker])
    if marker not in candidate_markers:
        candidate_markers.insert(0, marker)
    candidate_axis_markers = list(candidate_axis_markers or [axis_marker])
    if axis_marker not in candidate_axis_markers:
        candidate_axis_markers.insert(0, axis_marker)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_marker in candidate_markers or [marker]:
        if candidate_marker not in by_marker:
            raise ValueError(
                f"candidate contact point marker must be one of {sorted(by_marker)}; got {candidate_marker!r}"
            )
        if candidate_marker in seen:
            continue
        seen.add(candidate_marker)
        candidates.append(by_marker[candidate_marker])
    axis_candidates: list[dict[str, Any]] = []
    seen_axis: set[str] = set()
    for candidate_axis_marker in candidate_axis_markers or [axis_marker]:
        if candidate_axis_marker not in by_marker:
            raise ValueError(
                f"candidate motion-axis marker must be one of {sorted(by_marker)}; got {candidate_axis_marker!r}"
            )
        if candidate_axis_marker in seen_axis:
            continue
        seen_axis.add(candidate_axis_marker)
        axis_candidates.append(by_marker[candidate_axis_marker])
    return by_marker[marker], by_marker[axis_marker], candidates, axis_candidates, raw


def _axis_candidate_json(row: Mapping[str, Any]) -> dict[str, Any]:
    vector_tool, axis_role = _axis_for_face_row(row)
    marker = row.get("contact_marker") or row.get("face_marker")
    ref = row.get("surface_ref") or row.get("point_ref")
    return {
        "candidate_id": str(ref or marker),
        "axis_marker": marker,
        "marker": marker,
        "axis_role": axis_role,
        "role": axis_role,
        "axis_ref": ref,
        "ref": ref,
        "surface_ref": row.get("surface_ref"),
        "patch_id": row.get("patch_id"),
        "polarity": row.get("polarity") or "unsigned",
        "vector_tool": vector_tool,
        "mesh_normal_tool": row.get("normal_tool"),
    }


def _contact_selection_json(
    *,
    tool_id: str,
    task_id: str,
    task_instruction: str,
    part_row: Mapping[str, Any],
    contact_row: Mapping[str, Any],
    motion_axis_row: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    axis_candidate_rows: list[dict[str, Any]],
    all_contact_rows: list[dict[str, Any]],
    confidence: float,
    reason: str,
    raw_response: Any,
    visual_evidence_path: Path,
) -> dict[str, Any]:
    marker = str(part_row["marker"])
    selected_contact_point_marker = contact_row.get("contact_marker") or contact_row.get("face_marker")
    selected_motion_axis_marker = motion_axis_row.get("contact_marker") or motion_axis_row.get("face_marker")
    candidate_contact_points = [
        {
            "marker": row.get("contact_marker") or row.get("face_marker"),
            "point_ref": row.get("point_ref"),
            "surface_ref": row.get("surface_ref"),
            "patch_id": row.get("patch_id"),
            "row": dict(row),
        }
        for row in candidate_rows
    ]
    return {
        "schema_version": "parttoolgen_contact_point_motion_axis_selection_v1",
        "selection_semantics": "p3sam_vlm_contact_point_motion_axis",
        "artifact_id": f"{tool_id}_{task_id}_p3sam_contact_point_selection",
        "tool_id": tool_id,
        "task_id": task_id,
        "task_instruction": task_instruction,
        "selected_part": {
            "marker": marker,
            "part_ref": part_row["part_ref"],
        },
        "contact_point": {
            "marker": selected_contact_point_marker,
            "point_ref": contact_row.get("point_ref"),
            "surface_ref": contact_row.get("surface_ref"),
            "patch_id": contact_row.get("patch_id"),
            "row": dict(contact_row),
        },
        "contact_point_candidates": candidate_contact_points,
        "motion_axis": {
            "marker": selected_motion_axis_marker,
            "surface_ref": motion_axis_row.get("surface_ref"),
            "patch_id": motion_axis_row.get("patch_id"),
            "row": dict(motion_axis_row),
        },
        "motion_axis_candidates": [
            {
                "marker": row.get("contact_marker") or row.get("face_marker"),
                "surface_ref": row.get("surface_ref"),
                "patch_id": row.get("patch_id"),
                "row": dict(row),
            }
            for row in axis_candidate_rows
        ],
        "contact_point_selection_policy": "compiler_optimizes_over_public_vlm_contact_candidates",
        "confidence": float(confidence),
        "reason": reason,
        "contact_point_marker_table": all_contact_rows,
        "visual_evidence": {
            "visual_evidence": str(visual_evidence_path),
            "prompt_images": [],
        },
        "raw_response": raw_response,
        "metadata": {
            "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
            "tool_side_grounding_source": "p3sam",
        },
    }


def _axis_binding_json(
    *,
    tool_id: str,
    task_id: str,
    task_instruction: str,
    part_row: Mapping[str, Any],
    contact_row: Mapping[str, Any],
    motion_axis_row: Mapping[str, Any],
    candidate_rows: list[dict[str, Any]],
    axis_candidate_rows: list[dict[str, Any]],
    confidence: float,
    reason: str,
    raw_response: Any,
    axis_mesh_report_path: Path,
    visual_evidence_path: Path,
) -> dict[str, Any]:
    motion_axis_ref = "tool.motion_axis"
    motion_axis_tool, motion_axis_source = _axis_for_face_row(motion_axis_row)
    selected_contact_point_marker = contact_row.get("contact_marker") or contact_row.get("face_marker")
    selected_motion_axis_marker = motion_axis_row.get("contact_marker") or motion_axis_row.get("face_marker")
    axis_candidates = [_axis_candidate_json(row) for row in axis_candidate_rows]
    candidate_contact_points = [
        {
            "marker": row.get("contact_marker") or row.get("face_marker"),
            "point_ref": row.get("point_ref"),
            "surface_ref": row.get("surface_ref"),
            "patch_id": row.get("patch_id"),
            "row": dict(row),
        }
        for row in candidate_rows
    ]
    return {
        "schema_version": "parttoolgen_contact_point_motion_axis_binding_v1",
        "artifact_id": f"{tool_id}_{task_id}_p3sam_axis_binding",
        "tool_id": tool_id,
        "task_id": task_id,
        "task_instruction": task_instruction,
        "selected_part": {
            "marker": str(part_row["marker"]),
            "part_ref": part_row["part_ref"],
        },
        "contact_point": {
            "marker": selected_contact_point_marker,
            "point_ref": contact_row.get("point_ref"),
            "surface_ref": contact_row.get("surface_ref"),
            "patch_id": contact_row.get("patch_id"),
            "candidates": candidate_contact_points,
        },
        "motion_axis": {
            "source": "selected_motion_axis_surface",
            "marker": selected_motion_axis_marker,
            "role": motion_axis_source,
            "ref": motion_axis_ref,
            "surface_ref": motion_axis_row.get("surface_ref"),
            "polarity": motion_axis_row.get("polarity") or "unsigned",
            "vector_tool": motion_axis_tool,
            "mesh_normal_tool": motion_axis_row.get("normal_tool"),
            "candidates": axis_candidates,
        },
        "axis_candidate_table": axis_candidates,
        "confidence": float(confidence),
        "reason": reason,
        "visual_evidence": {
            "visual_evidence": str(visual_evidence_path),
            "axis_mesh_overlay_report": str(axis_mesh_report_path),
            "prompt_images": [],
        },
        "raw_response": raw_response,
        "metadata": {
            "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
            "tool_side_grounding_source": "p3sam",
        },
    }


def _axis_mesh_report_json(
    *,
    tool_id: str,
    task_id: str,
    part_row: Mapping[str, Any],
    contact_row: Mapping[str, Any],
    motion_axis_row: Mapping[str, Any],
    axis_candidate_rows: list[dict[str, Any]],
    prior_dir: Path,
    visual_evidence_path: Path,
) -> dict[str, Any]:
    motion_axis_ref = "tool.motion_axis"
    motion_axis_tool, motion_axis_source = _axis_for_face_row(motion_axis_row)
    selected_contact_point_marker = contact_row.get("contact_marker") or contact_row.get("face_marker")
    selected_motion_axis_marker = motion_axis_row.get("contact_marker") or motion_axis_row.get("face_marker")
    axis_candidates = [_axis_candidate_json(row) for row in axis_candidate_rows]
    return {
        "schema_version": "parttoolgen_contact_point_motion_axis_mesh_evidence_v1",
        "tool_id": tool_id,
        "task_id": task_id,
        "selected_part": {
            "marker": str(part_row["marker"]),
            "part_ref": part_row["part_ref"],
        },
        "view_indices": [],
        "image": None,
        "render_sources": [],
        "contact_point": {
            "marker": selected_contact_point_marker,
            "point_ref": contact_row.get("point_ref"),
            "surface_ref": contact_row.get("surface_ref"),
            "patch_id": contact_row.get("patch_id"),
            "xyz_tool": contact_row.get("xyz_tool"),
        },
        "contact_point_surface": {
            "frame_source": contact_row.get("selection_semantics") or "p3sam_selected_bbox_face",
            "patch_id": contact_row.get("patch_id"),
            "surface_ref": contact_row.get("surface_ref"),
            "raw_mesh_patch_center_xyz_tool": contact_row.get("xyz_tool"),
            "contact_point_xyz_tool": contact_row.get("xyz_tool"),
            "surface_center_xyz_tool": contact_row.get("xyz_tool"),
            "rectangle_corners_xyz_tool": None,
            "long_edge_m": None,
            "short_edge_m": None,
            "source_surface_ref": contact_row.get("surface_ref"),
        },
        "motion_axis_surface": {
            "frame_source": motion_axis_row.get("selection_semantics") or "p3sam_selected_bbox_face",
            "patch_id": motion_axis_row.get("patch_id"),
            "surface_ref": motion_axis_row.get("surface_ref"),
            "surface_center_xyz_tool": motion_axis_row.get("xyz_tool"),
            "normal_role": motion_axis_source,
            "motion_axis_ref": motion_axis_ref,
            "motion_axis_tool": motion_axis_tool,
            "mesh_normal_tool": motion_axis_row.get("normal_tool"),
        },
        "motion_axis": {
            "source": "selected_motion_axis_surface",
            "marker": selected_motion_axis_marker,
            "role": motion_axis_source,
            "ref": motion_axis_ref,
            "vector_tool": motion_axis_tool,
            "surface_ref": motion_axis_row.get("surface_ref"),
            "mesh_normal_tool": motion_axis_row.get("normal_tool"),
        },
        "axis_candidates": axis_candidates,
        "metadata": {
            "source_prior": str(prior_dir),
            "visual_evidence": str(visual_evidence_path),
            "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
        },
    }


def _part_selection_json(
    *,
    tool_id: str,
    task_id: str,
    task_instruction: str,
    part_row: Mapping[str, Any],
    selection_source: str,
    visual_evidence_path: Path,
    raw_response: Any,
) -> dict[str, Any]:
    marker = str(part_row["marker"])
    return {
        "schema_version": "parttoolgen_p3sam_part_selection_v0",
        "tool_id": tool_id,
        "task_id": task_id,
        "task_instruction": task_instruction,
        "selection_source": selection_source,
        "selected_marker": marker,
        "selected_function_marker": marker,
        "selected_part": {
            "marker": marker,
            "part_ref": part_row["part_ref"],
        },
        "selected_function_part_ref": part_row["part_ref"],
        "selected_part_row": dict(part_row),
        "visual_evidence": str(visual_evidence_path),
        "raw_response": raw_response,
        "metadata": {
            "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
            "created_unix_s": time.time(),
        },
    }


def _part_row_from_selection(
    *,
    path: Path,
    visual_evidence: Mapping[str, Any],
    tool_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _load_json(path)
    marker = (
        data.get("selected_marker")
        or data.get("selected_function_marker")
        or (data.get("selected_part") or {}).get("marker")
    )
    if marker is None:
        raise ValueError(f"part selection has no selected marker: {path}")
    row = data.get("selected_part_row")
    if isinstance(row, Mapping):
        row = dict(row)
        row.setdefault("marker", str(marker))
        return row, data
    rows = _part_rows(visual_evidence, tool_id)
    by_marker = _row_by_marker(rows, "marker", "P")
    marker_text = str(marker)
    if marker_text not in by_marker:
        raise ValueError(f"selected part marker {marker_text!r} is not present in visual evidence")
    return by_marker[marker_text], data


def run(args: argparse.Namespace) -> dict[str, Any]:
    prior_dir = args.p3sam_prior_dir
    visual_evidence_path = args.visual_evidence
    if not prior_dir.exists():
        raise FileNotFoundError(f"P3SAM prior dir does not exist: {prior_dir}")
    if not visual_evidence_path.exists():
        raise FileNotFoundError(f"P3SAM visual evidence does not exist: {visual_evidence_path}")
    tool_part_graph_path = prior_dir / "tool_part_graph.json"
    if not tool_part_graph_path.exists():
        raise FileNotFoundError(f"P3SAM prior has no tool_part_graph.json: {tool_part_graph_path}")

    visual_evidence = _load_json(visual_evidence_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "contact":
        if args.part_selection is None:
            raise ValueError("--part-selection is required for --stage contact")
        part_row, part_selection = _part_row_from_selection(
            path=args.part_selection,
            visual_evidence=visual_evidence,
            tool_id=args.tool_id,
        )
        part_raw = part_selection.get("raw_response") or part_selection
    else:
        part_row, part_raw = _select_part(
            visual_evidence=visual_evidence,
            tool_id=args.tool_id,
            task_instruction=args.task_instruction,
            selection_source=args.selection_source,
            marker=args.part_marker,
            model=args.model,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
        )
    part_marker = str(part_row["marker"])
    part_selection = _part_selection_json(
        tool_id=args.tool_id,
        task_id=args.task_id,
        task_instruction=args.task_instruction,
        part_row=part_row,
        selection_source=args.selection_source,
        visual_evidence_path=visual_evidence_path,
        raw_response=part_raw,
    )
    part_selection_path = args.output_dir / "part_selection.json"
    if args.stage != "contact":
        _write_json(part_selection_path, part_selection)
    if args.stage == "part":
        summary = {
            "schema_version": "parttoolgen_p3sam_tool_ref_selection_v0",
            "stage": "part",
            "tool_id": args.tool_id,
            "task_id": args.task_id,
            "selection_source": args.selection_source,
            "p3sam_prior_dir": str(prior_dir),
            "visual_evidence": str(visual_evidence_path),
            "selected_part": {
                "marker": part_marker,
                "part_ref": part_row["part_ref"],
            },
            "outputs": {"part_selection": str(part_selection_path)},
            "metadata": {
                "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
                "created_unix_s": time.time(),
            },
        }
        _write_json(args.output_dir / "p3sam_tool_ref_selection_summary.json", summary)
        return summary

    contact_row, motion_axis_row, candidate_rows, axis_candidate_rows, contact_raw = _select_contact(
        visual_evidence=visual_evidence,
        tool_id=args.tool_id,
        task_instruction=args.task_instruction,
        part_marker=part_marker,
        selection_source=args.selection_source,
        marker=args.contact_point_marker,
        candidate_markers=args.candidate_contact_point_markers,
        axis_marker=args.motion_axis_marker,
        candidate_axis_markers=args.candidate_motion_axis_markers,
        model=args.model,
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
    )

    shutil.copy2(tool_part_graph_path, args.output_dir / "tool_part_graph.json")
    confidence = float(contact_raw.get("confidence", part_raw.get("confidence", 0.0)) or 0.0)
    reason = str(contact_raw.get("reason") or part_raw.get("reason") or "").strip()

    axis_mesh_report = _axis_mesh_report_json(
        tool_id=args.tool_id,
        task_id=args.task_id,
        part_row=part_row,
        contact_row=contact_row,
        motion_axis_row=motion_axis_row,
        axis_candidate_rows=axis_candidate_rows,
        prior_dir=prior_dir,
        visual_evidence_path=visual_evidence_path,
    )
    axis_mesh_report_path = args.output_dir / "axis_mesh_overlay_report.json"
    _write_json(axis_mesh_report_path, axis_mesh_report)

    all_contact_rows = _contact_rows(visual_evidence, tool_id=args.tool_id, part_marker=part_marker)
    contact_selection = _contact_selection_json(
        tool_id=args.tool_id,
        task_id=args.task_id,
        task_instruction=args.task_instruction,
        part_row=part_row,
        contact_row=contact_row,
        motion_axis_row=motion_axis_row,
        candidate_rows=candidate_rows,
        axis_candidate_rows=axis_candidate_rows,
        all_contact_rows=all_contact_rows,
        confidence=confidence,
        reason=reason,
        raw_response={"part": part_raw, "contact_point_motion_axis": contact_raw},
        visual_evidence_path=visual_evidence_path,
    )
    contact_selection_path = args.output_dir / "contact_point_selection.json"
    _write_json(contact_selection_path, contact_selection)

    axis_binding = _axis_binding_json(
        tool_id=args.tool_id,
        task_id=args.task_id,
        task_instruction=args.task_instruction,
        part_row=part_row,
        contact_row=contact_row,
        motion_axis_row=motion_axis_row,
        candidate_rows=candidate_rows,
        axis_candidate_rows=axis_candidate_rows,
        confidence=confidence,
        reason=reason,
        raw_response={"part": part_raw, "contact_point_motion_axis": contact_raw},
        axis_mesh_report_path=axis_mesh_report_path,
        visual_evidence_path=visual_evidence_path,
    )
    axis_binding_path = args.output_dir / "axis_binding.json"
    _write_json(axis_binding_path, axis_binding)

    summary = {
        "schema_version": "parttoolgen_p3sam_tool_ref_selection_v0",
        "stage": args.stage,
        "tool_id": args.tool_id,
        "task_id": args.task_id,
        "selection_source": args.selection_source,
        "p3sam_prior_dir": str(prior_dir),
        "visual_evidence": str(visual_evidence_path),
        "selected_part": {
            "marker": part_marker,
            "part_ref": part_row["part_ref"],
        },
        "contact_point": {
            "marker": contact_row.get("contact_marker") or contact_row.get("face_marker"),
            "point_ref": contact_row.get("point_ref"),
            "surface_ref": contact_row.get("surface_ref"),
        },
        "contact_point_candidates": [
            {
                "marker": row.get("contact_marker") or row.get("face_marker"),
                "point_ref": row.get("point_ref"),
                "surface_ref": row.get("surface_ref"),
            }
            for row in candidate_rows
        ],
        "motion_axis": {
            "marker": motion_axis_row.get("contact_marker") or motion_axis_row.get("face_marker"),
            "surface_ref": motion_axis_row.get("surface_ref"),
        },
        "motion_axis_candidates": [
            {
                "marker": row.get("contact_marker") or row.get("face_marker"),
                "surface_ref": row.get("surface_ref"),
            }
            for row in axis_candidate_rows
        ],
        "outputs": {
            "part_selection": str(part_selection_path if args.stage != "contact" else args.part_selection),
            "axis_binding": str(axis_binding_path),
            "axis_mesh_overlay_report": str(axis_mesh_report_path),
            "contact_point_selection": str(contact_selection_path),
            "tool_part_graph": str(args.output_dir / "tool_part_graph.json"),
        },
        "metadata": {
            "created_by": "parttoolgen.cli.tool_prior.select_p3sam_tool_refs",
            "created_unix_s": time.time(),
        },
    }
    _write_json(args.output_dir / "p3sam_tool_ref_selection_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--p3sam-prior-dir", type=Path, required=True)
    parser.add_argument("--visual-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["part", "contact", "full"], default="full")
    parser.add_argument("--part-selection", type=Path)
    parser.add_argument("--selection-source", choices=["vlm", "reference"], default="vlm")
    parser.add_argument("--part-marker")
    parser.add_argument("--contact-point-marker")
    parser.add_argument("--candidate-contact-point-markers", nargs="*")
    parser.add_argument("--candidate-motion-axis-markers", nargs="*")
    parser.add_argument(
        "--motion-axis-marker",
        help=(
            "Reference-mode marker for the tool face whose normal/declared axis "
            "should align with the scene motion axis. Defaults to --contact-point-marker."
        ),
    )
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL") or "gpt-5.4")
    parser.add_argument("--base-url", default=os.environ.get("VLM_API_BASE"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
