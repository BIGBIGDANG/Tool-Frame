#!/usr/bin/env python3
"""Select one working part, contact point and local motion axis from P3SAM views."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "ToGo/src"))
from ToGo.constraints.vlm_client import build_vlm_client
from ToGo.compiler.residuals import unit
from ToGo.pipeline.handoff import compact_tool_refs, read_json, write_json

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
    if patch_axis is not None and str(row.get("selection_semantics") or "").endswith("bbox_face"):
        return patch_axis, "bbox_face_axis"
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


def _simple_overview_contact_axis_prompt(
    tool_id: str,
    task_instruction: str,
    part_marker: str,
    contact_rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
) -> str:
    compact_contact_rows = []
    for row in contact_rows:
        compact_contact_rows.append({"contact_point_marker": row.get("contact_marker") or row.get("face_marker")})
    compact_axis_rows = []
    for row in axis_rows:
        compact_axis_rows.append({"motion_axis_marker": row.get("contact_marker") or row.get("face_marker")})
    return (
        "Choose practical tool-use references for the selected tool part and task.\n"
        "The attached tool image is a visual evidence overview for candidate faces on the selected tool part.\n"
        "Overview images show candidate F faces on the selected part.\n"
        "Use ordinary tool-use common sense for the selected part and task.\n"
        "The image is labeled F1, F2, ... . Each F marker corresponds to one candidate local face on the selected "
        "part. A colored or yellow highlighted region shows which candidate face the marker refers to.\n"
        "For contact selection:\n"
        "Select contact candidates by asking which highlighted candidate face a person would normally use to touch "
        "and act on the target with this tool part in this task.\n"
        "For motion-axis selection:\n"
        "The motion axis is a tool-local axis on the working part for carrying out the task action. It is fixed in "
        "the tool frame and rotates with the tool. It is not a world, camera, screen, table, or image direction.\n"
        "Do not choose the motion-axis marker by the requested scene-side direction in the picture, such as left, "
        "right, up, down, or toward the table. The compiler will later rotate the whole tool so the selected local "
        "axis aligns with the required scene-side action direction.\n"
        "Select motion-axis candidates by asking: which local axis of the same working part physically represents how "
        "the chosen contact face, tip, edge, or pad produces the task effect in ordinary use?\n"
        "Before choosing the motion axis, reason about the physical interaction needed for the task.\n"
        "If the working part acts by driving the target through a local face normal, the contact face and motion-axis "
        "face may be the same. If the working part stays in contact while sliding tangentially, the contact marker "
        "should be the touching surface/tip/edge, while the motion-axis marker should represent the local sliding "
        "direction rather than the contact-pressure normal.\n"
        "Do not decide from tool name alone. Decide from the task effect, the highlighted candidate faces, and ordinary "
        "physical use of the selected part.\n"
        "A marker may be useful as a contact point, a motion axis, both, or neither; decide the two roles separately.\n"
        "Choose one practical marker for each role, using the depicted geometry. "
        "Avoid unusual, awkward, hidden, or accidental contacts when a normal working "
        "contact is available. Use only the listed F markers; do not invent coordinates or new parts.\n"
        "Return exactly one JSON object:\n"
        "{\n"
        '  "ordinary_use_modes": [\n'
        '    {\n'
        '      "description": "short use mode",\n'
        '      "contact_markers": ["F1"],\n'
        '      "motion_axis_markers": ["F3"]\n'
        '    }\n'
        "  ],\n"
        '  "selected_contact_point_marker": "F1",\n'
        '  "selected_motion_axis_marker": "F3",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short task-and-visual reason"\n'
        "}\n\n"
        f"tool_id: {tool_id}\n"
        f"task_instruction: {task_instruction}\n"
        f"selected_part_marker: {part_marker}\n"
        f"valid_contact_point_candidates:\n{json.dumps(compact_contact_rows, indent=2)}\n"
        f"valid_motion_axis_candidates:\n{json.dumps(compact_axis_rows, indent=2)}\n"
    )


def _images(evidence: Mapping[str, Any], path: Path, tool_id: str, part_marker=None) -> list[Path]:
    kinds = {"whole_mesh_part_choice_overview", "isolated_part_choice_overview"} if part_marker is None else {"candidate_part_contact_overview"}
    images = []
    for item in evidence.get("items", []):
        if item.get("kind") not in kinds or not str(item.get("id", "")).startswith(tool_id + "_"):
            continue
        if part_marker is not None and str(item.get("part_marker", "")) != str(part_marker):
            continue
        image = Path(item["path"])
        image = image if image.is_absolute() else path.parent / image
        if not image.is_file():
            raise FileNotFoundError(f"missing tool evidence image: {image}")
        if image not in images:
            images.append(image)
    if not images:
        raise ValueError("tool evidence has no matching part/face overview image")
    return images if part_marker is None else images[:1]


def _vlm_json(prompt, images, instruction, model):
    client = build_vlm_client(response_mode="openai-compatible", image_paths=images,
                              model=model, max_new_tokens=768)
    response = client.generate(prompt=prompt, task_instruction=instruction,
                               context_id="tool_reference_selection",
                               system_prompt="Return only the requested JSON object.")
    result = json.loads(_strip_json_response(response.raw_text))
    if not isinstance(result, dict):
        raise ValueError("VLM response must be a JSON object")
    return result


def select(*, tool_id: str, instruction: str, evidence_path: Path, output: Path,
           stage: str, part_selection: Path | None, model: str | None) -> dict[str, Any]:
    evidence_path = evidence_path.resolve()
    evidence = read_json(evidence_path)
    rows = _part_rows(evidence, tool_id)
    parts = _row_by_marker(rows, "marker", "P")
    if not parts:
        raise ValueError(f"tool evidence has no parts for {tool_id}")
    if stage == "part":
        response = _vlm_json(_part_prompt(tool_id, instruction, rows),
                             _images(evidence, evidence_path, tool_id), instruction, model)
        marker = _canonical_marker(response.get("selected_marker"), prefix="P")
        if marker not in parts:
            raise ValueError(f"selected part {marker!r} is absent from tool evidence")
        result = {"tool_id": tool_id, "selected_marker": str(parts[marker]["marker"]),
                  "selected_part": {"marker": str(parts[marker]["marker"]), "part_ref": parts[marker]["part_ref"]}}
        write_json(output / "part_selection.json", result)
        return result
    if part_selection is None:
        raise ValueError("contact selection requires its prior part selection")
    selected = read_json(part_selection)
    if selected.get("tool_id") != tool_id:
        raise ValueError("part selection belongs to another tool")
    marker = _canonical_marker(selected.get("selected_marker"), prefix="P")
    if marker not in parts:
        raise ValueError("selected part is absent from tool evidence")
    part = parts[marker]
    marker = str(part["marker"])
    contacts = _contact_rows(evidence, tool_id=tool_id, part_marker=marker)
    axes = contacts
    if not contacts or not axes:
        raise ValueError("selected part has no contact or local motion-axis evidence")
    response = _vlm_json(_simple_overview_contact_axis_prompt(tool_id, instruction, marker, contacts, axes),
                         _images(evidence, evidence_path, tool_id, marker), instruction, model)
    contact_marker = _canonical_marker(response.get("selected_contact_point_marker"), prefix="F")
    axis_marker = _canonical_marker(response.get("selected_motion_axis_marker"), prefix="F")
    contact_map, axis_map = _row_by_marker(contacts, "contact_marker", "F"), _row_by_marker(axes, "contact_marker", "F")
    if contact_marker not in contact_map or axis_marker not in axis_map:
        raise ValueError("selected contact/motion-axis marker is absent from valid tool evidence")
    contact, axis = contact_map[contact_marker], axis_map[axis_marker]
    vector, source = _axis_for_face_row(axis)
    vector = unit(vector)
    result = compact_tool_refs({"tool_id": tool_id,
        "contact_point_ref": "tool.contact_point", "motion_axis_ref": "tool.motion_axis",
        "contact_point_tool": contact.get("xyz_tool"), "motion_axis_tool": vector,
        "contact_point_face_marker": contact_marker, "motion_axis_face_marker": axis_marker,
        "contact_point_part_ref": part["part_ref"], "contact_point_source_ref": contact.get("point_ref"),
        "contact_point_surface_ref": contact.get("surface_ref"), "motion_axis_surface_ref": axis.get("surface_ref"),
        "surface_geometry_tool": {"frame_source": contact.get("selection_semantics") or "p3sam_selected_bbox_face",
            "patch_id": contact.get("patch_id"), "surface_center": contact.get("xyz_tool"),
            "rectangle_corners": None, "long_edge_m": None, "short_edge_m": None,
            "motion_axis_tool": vector, "motion_axis_source": source}}, tool_id)
    write_json(output / "tool_refs.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--visual-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["part", "contact"], required=True)
    parser.add_argument("--part-selection", type=Path)
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL"))
    args = parser.parse_args()
    select(tool_id=args.tool_id, instruction=args.task_instruction, evidence_path=args.visual_evidence,
           output=args.output_dir.resolve(), stage=args.stage, part_selection=args.part_selection, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
