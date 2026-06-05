#!/usr/bin/env python3
"""Semantic aliases for SAMPart3D tool regions.

This module keeps part naming in the evidence lane. A VLM may caption
``tool.parts.*`` regions from SAMPart3D region visual evidence, but the aliases are
never executable refs and never replace geometry or affordance-binding gates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from parttoolgen.program.visual_evidence import load_visual_evidence, summarize_visual_evidence
from parttoolgen.program.visual_evidence import validate_visual_evidence
from parttoolgen.validation.validate_parttoolgen import (
    validate_grounding_candidates,
    validate_tool_part_graph,
    validate_tool_part_semantic_labels as _validate_tool_part_semantic_labels,
)


def load_tool_part_semantic_labels(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_tool_part_semantic_labels(artifact)
    return artifact


def validate_tool_part_semantic_labels(artifact: Any) -> None:
    _validate_tool_part_semantic_labels(artifact)


def _strip_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("semantic-label response did not contain a JSON object")
    return text[start : end + 1]


def _part_cards(tool_part_graph: dict[str, Any]) -> list[dict[str, Any]]:
    validate_tool_part_graph(tool_part_graph)
    cards: list[dict[str, Any]] = []
    for part_id, part in sorted(tool_part_graph.get("parts", {}).items()):
        cards.append(
            {
                "part_ref": f"tool.parts.{part_id}",
                "current_label": part.get("label", part_id),
                "role_candidates": part.get("role_candidates", []),
                "mask_ref": part.get("mask_ref"),
                "point_count": part.get("point_count"),
                "obb_extent_xyz": part.get("obb", {}).get("extent_xyz"),
                "surface_patch_ids": sorted((part.get("surface_patches") or {}).keys()),
            }
        )
    return cards


def render_part_semantic_label_prompt(
    *,
    tool_part_graph: dict[str, Any],
    visual_evidence: dict[str, Any] | None = None,
    task_instruction: str | None = None,
    label_id: str = "tool_part_semantic_labels",
) -> str:
    """Render a JSON-only prompt for naming SAMPart3D regions."""

    validate_tool_part_graph(tool_part_graph)
    payload = {
        "label_id": label_id,
        "tool_id": tool_part_graph["tool_id"],
        "task_instruction_optional_context": task_instruction or "",
        "parts_to_label": _part_cards(tool_part_graph),
        "visual_evidence": summarize_visual_evidence(
            visual_evidence,
            allowed_uses=("planning_context", "repair_diagnosis"),
            preferred_tool_id=tool_part_graph["tool_id"],
        ),
    }
    return (
        "You are labeling SAMPart3D tool regions for PartToolGen.\n"
        "Return exactly one JSON object and no prose. Do not output Python.\n"
        "The labels are prompt-only semantic aliases for region binding. They are not executable refs, "
        "task rules, coordinates, poses, or robot actions.\n"
        "Name every listed tool.parts.* region using the visual evidence and part geometry. If uncertain, "
        "use a cautious label such as 'unknown tool region' and lower confidence.\n"
        "Do not choose the functional part for the task; only describe visible/tool-part semantics.\n"
        "Required JSON shape: schema_version='v0.1', label_id, tool_id, parts, metadata.\n"
        "Each parts entry key must be the part id, and must include: part_ref, semantic_label, aliases, "
        "visual_description, confidence, evidence_items, source, provenance.\n"
        "aliases should be natural-language names such as handle, head, tip, blade, bristles, flat face, "
        "side surface, or unknown region. Final planners will still use exact refs only.\n\n"
        "Labeling payload:\n"
        + json.dumps(payload, indent=2)
        + "\n"
    )


def build_unlabeled_scaffold(
    *,
    tool_part_graph: dict[str, Any],
    label_id: str,
    source_artifact: str = "",
) -> dict[str, Any]:
    """Build a valid fallback artifact that does not claim semantic knowledge."""

    validate_tool_part_graph(tool_part_graph)
    parts: dict[str, Any] = {}
    for part_id, part in sorted(tool_part_graph.get("parts", {}).items()):
        part_ref = f"tool.parts.{part_id}"
        label = str(part.get("label") or part_id)
        parts[part_id] = {
            "part_ref": part_ref,
            "semantic_label": "unknown tool region",
            "aliases": [label, part_id, "unknown tool region"],
            "visual_description": (
                f"Unlabeled SAMPart3D region {part_id}; use visual evidence and exact refs."
            ),
            "confidence": 0.2,
            "evidence_items": [],
            "source": "deterministic_unlabeled_scaffold",
            "provenance": {
                "method": "deterministic_unlabeled_scaffold",
                "observation_id": str(tool_part_graph["tool_id"]),
                "source_artifact": source_artifact,
            },
        }
    artifact = {
        "schema_version": "v0.1",
        "label_id": label_id,
        "tool_id": tool_part_graph["tool_id"],
        "parts": parts,
        "metadata": {
            "created_by": "parttoolgen.program.part_semantic_labels",
            "mode": "deterministic_unlabeled_scaffold",
        },
    }
    validate_tool_part_semantic_labels(artifact)
    return artifact


def _part_label_index(part_id: str, part: dict[str, Any]) -> int | None:
    provenance = part.get("provenance", {}) if isinstance(part, dict) else {}
    raw = provenance.get("part_label") if isinstance(provenance, dict) else None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        return int(raw)
    suffix = part_id.split("_")[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def _find_sampart_label_set(
    semantic_map: Any,
    *,
    semantic_key: str | None,
    tool_id: str,
) -> list[str] | dict[str, str]:
    """Find a SAMPart3D-style label set in common semantic-map layouts."""

    if isinstance(semantic_map, list):
        return [str(item) for item in semantic_map]
    if not isinstance(semantic_map, dict):
        raise ValueError("SAMPart3D semantic map must be a list or object")
    for key in (semantic_key, tool_id):
        if key and isinstance(semantic_map.get(key), list):
            return [str(item) for item in semantic_map[key]]
        if key and isinstance(semantic_map.get(key), dict):
            nested = semantic_map[key]
            if all(isinstance(value, str) for value in nested.values()):
                return {str(k): str(v) for k, v in nested.items()}
    for category_value in semantic_map.values():
        if not isinstance(category_value, dict):
            continue
        for key in (semantic_key, tool_id):
            if key and isinstance(category_value.get(key), list):
                return [str(item) for item in category_value[key]]
            if key and isinstance(category_value.get(key), dict):
                nested = category_value[key]
                if all(isinstance(value, str) for value in nested.values()):
                    return {str(k): str(v) for k, v in nested.items()}
    if all(isinstance(value, str) for value in semantic_map.values()):
        return {str(k): str(v) for k, v in semantic_map.items()}
    raise ValueError(
        "could not find a SAMPart3D semantic label set; pass --sampart-semantic-key "
        "when the map is keyed by mesh uid"
    )


def build_from_sampart_semantic_map(
    *,
    tool_part_graph: dict[str, Any],
    semantic_map: Any,
    label_id: str,
    semantic_key: str | None = None,
    source_artifact: str = "",
) -> dict[str, Any]:
    """Build labels from SAMPart3D-provided semantic names when available."""

    validate_tool_part_graph(tool_part_graph)
    label_set = _find_sampart_label_set(
        semantic_map,
        semantic_key=semantic_key,
        tool_id=tool_part_graph["tool_id"],
    )
    parts: dict[str, Any] = {}
    for part_id, part in sorted(tool_part_graph.get("parts", {}).items()):
        part_label = _part_label_index(part_id, part)
        if isinstance(label_set, list):
            if part_label is None or part_label < 0 or part_label >= len(label_set):
                raise ValueError(f"part {part_id} label index {part_label!r} is not in semantic map")
            semantic_label = str(label_set[part_label])
        else:
            keys = [part_id]
            if part_label is not None:
                keys.extend([str(part_label), f"part_{part_label:03d}"])
            semantic_label = ""
            for key in keys:
                if key in label_set:
                    semantic_label = str(label_set[key])
                    break
            if not semantic_label:
                raise ValueError(f"part {part_id} is missing from semantic map")
        aliases = [semantic_label]
        source_label = str(part.get("label") or part_id)
        if source_label not in aliases:
            aliases.append(source_label)
        parts[part_id] = {
            "part_ref": f"tool.parts.{part_id}",
            "semantic_label": semantic_label,
            "aliases": aliases,
            "visual_description": (
                f"SAMPart3D semantic label for {part_id}: {semantic_label}."
            ),
            "confidence": float(part.get("confidence", 0.7)),
            "evidence_items": [],
            "source": "sampart3d_semantic_labels",
            "provenance": {
                "method": "sampart3d_semantic_label_map",
                "observation_id": str(tool_part_graph["tool_id"]),
                "source_artifact": source_artifact,
                "part_label": part_label,
            },
        }
    artifact = {
        "schema_version": "v0.1",
        "label_id": label_id,
        "tool_id": tool_part_graph["tool_id"],
        "parts": parts,
        "metadata": {
            "created_by": "parttoolgen.program.part_semantic_labels",
            "mode": "provider_supplied_part_semantic_aliases",
            "semantic_key": semantic_key or tool_part_graph["tool_id"],
        },
    }
    validate_tool_part_semantic_labels(artifact)
    return artifact


def parse_part_semantic_label_response(
    raw_text: str,
    *,
    tool_part_graph: dict[str, Any],
    label_id: str,
    prompt_path: str | None = None,
    raw_response_path: str | None = None,
) -> dict[str, Any]:
    """Parse VLM JSON into a validated semantic-label artifact."""

    validate_tool_part_graph(tool_part_graph)
    data = json.loads(_strip_json_fence(raw_text))
    data.setdefault("schema_version", "v0.1")
    data.setdefault("label_id", label_id)
    data.setdefault("tool_id", tool_part_graph["tool_id"])
    data.setdefault("metadata", {})
    data["metadata"].setdefault("created_by", "vlm_part_semantic_labeler")
    data["metadata"].setdefault("mode", "vlm_part_semantic_aliases")
    if prompt_path:
        data["metadata"].setdefault("prompt_path", prompt_path)
    if raw_response_path:
        data["metadata"].setdefault("raw_response_path", raw_response_path)
    expected_parts = set(tool_part_graph.get("parts", {}))
    actual_parts = set(data.get("parts", {})) if isinstance(data.get("parts"), dict) else set()
    missing = sorted(expected_parts - actual_parts)
    extra = sorted(actual_parts - expected_parts)
    if missing or extra:
        raise ValueError(f"semantic-label parts mismatch; missing={missing}, extra={extra}")
    validate_tool_part_semantic_labels(data)
    return data


def _semantic_payload(label: dict[str, Any], *, source_label: str | None) -> dict[str, Any]:
    aliases = [str(item) for item in label.get("aliases", []) if str(item)]
    semantic_label = str(label.get("semantic_label") or "")
    if semantic_label and semantic_label not in aliases:
        aliases.insert(0, semantic_label)
    return {
        "source_label": source_label,
        "semantic_label": semantic_label,
        "semantic_aliases": aliases,
        "semantic_description": label.get("visual_description"),
        "semantic_confidence": label.get("confidence"),
        "semantic_evidence_items": label.get("evidence_items", []),
        "semantic_source": label.get("source"),
    }


def merge_part_semantic_labels(
    *,
    tool_part_graph: dict[str, Any],
    semantic_labels: dict[str, Any],
    grounding_candidates: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Merge semantic aliases into prompt-facing provider artifacts."""

    validate_tool_part_graph(tool_part_graph)
    validate_tool_part_semantic_labels(semantic_labels)
    if semantic_labels["tool_id"] != tool_part_graph["tool_id"]:
        raise ValueError(
            f"semantic-label tool_id {semantic_labels['tool_id']!r} does not match "
            f"{tool_part_graph['tool_id']!r}"
        )
    merged_tool = copy.deepcopy(tool_part_graph)
    for part_id, label in semantic_labels["parts"].items():
        if part_id not in merged_tool.get("parts", {}):
            raise ValueError(f"semantic labels reference unknown part: {part_id}")
        part = merged_tool["parts"][part_id]
        payload = _semantic_payload(label, source_label=part.get("label"))
        part.update(payload)
        if label.get("semantic_label") and label.get("semantic_label") != "unknown tool region":
            part["label"] = str(label["semantic_label"])
        roles = list(part.get("role_candidates", []))
        for alias in payload["semantic_aliases"]:
            role_key = f"semantic_alias:{alias}"
            if role_key not in roles:
                roles.append(role_key)
        part["role_candidates"] = roles
    merged_tool.setdefault("metadata", {})["part_semantic_labels"] = {
        "label_id": semantic_labels["label_id"],
        "mode": semantic_labels["metadata"].get("mode"),
        "source": "prompt_only_semantic_aliases",
    }
    validate_tool_part_graph(merged_tool)

    merged_candidates = None
    if grounding_candidates is not None:
        validate_grounding_candidates(grounding_candidates)
        merged_candidates = copy.deepcopy(grounding_candidates)
        for part_id, label in semantic_labels["parts"].items():
            candidate = merged_candidates.get("parts", {}).get(part_id)
            if not isinstance(candidate, dict):
                continue
            payload = _semantic_payload(label, source_label=candidate.get("label"))
            candidate.update(payload)
            if label.get("semantic_label") and label.get("semantic_label") != "unknown tool region":
                candidate["label"] = str(label["semantic_label"])
            roles = list(candidate.get("role_candidates", []))
            for alias in payload["semantic_aliases"]:
                role_key = f"semantic_alias:{alias}"
                if role_key not in roles:
                    roles.append(role_key)
            candidate["role_candidates"] = roles
            part_ref = f"tool.parts.{part_id}"
            for collection in ("points", "regions", "axes", "surfaces"):
                for entity in merged_candidates.get(collection, {}).values():
                    if isinstance(entity, dict) and entity.get("part_ref") == part_ref:
                        entity.setdefault("part_semantic_label", label.get("semantic_label"))
                        entity.setdefault("part_semantic_aliases", payload["semantic_aliases"])
        merged_candidates.setdefault("metadata", {})["part_semantic_labels"] = {
            "label_id": semantic_labels["label_id"],
            "mode": semantic_labels["metadata"].get("mode"),
            "source": "prompt_only_semantic_aliases",
        }
        validate_grounding_candidates(merged_candidates)
    return merged_tool, merged_candidates


def augment_visual_evidence_with_part_semantics(
    *,
    visual_evidence: dict[str, Any],
    semantic_labels: dict[str, Any],
) -> dict[str, Any]:
    """Attach part aliases to numbered anchor rows in visual evidence metadata."""

    validate_visual_evidence(visual_evidence)
    validate_tool_part_semantic_labels(semantic_labels)
    augmented = copy.deepcopy(visual_evidence)
    rows = augmented.get("metadata", {}).get("anchor_ref_table")
    if not isinstance(rows, list):
        return augmented
    for row in rows:
        if not isinstance(row, dict):
            continue
        part_ref = row.get("part_ref")
        part_id = None
        if isinstance(part_ref, str) and part_ref.startswith("tool.parts."):
            part_id = part_ref.split(".")[-1]
        elif isinstance(row.get("part_id"), str):
            part_id = row["part_id"]
        if not part_id:
            continue
        label = semantic_labels.get("parts", {}).get(part_id)
        if not isinstance(label, dict):
            continue
        row["part_semantic_label"] = label.get("semantic_label")
        row["part_semantic_aliases"] = label.get("aliases", [])
    validate_visual_evidence(augmented)
    return augmented


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-part-graph", required=True)
    parser.add_argument("--grounding-candidates")
    parser.add_argument("--visual-evidence")
    parser.add_argument("--sampart-semantic-map")
    parser.add_argument("--sampart-semantic-key")
    parser.add_argument("--task", default="")
    parser.add_argument("--label-id", required=True)
    parser.add_argument("--prompt-output")
    parser.add_argument("--raw-response")
    parser.add_argument("--output", required=True)
    parser.add_argument("--merged-tool-part-graph")
    parser.add_argument("--merged-grounding-candidates")
    parser.add_argument("--visual-evidence-output")
    args = parser.parse_args()

    tool_part_graph = _read_json(args.tool_part_graph)
    grounding_candidates = _read_json(args.grounding_candidates) if args.grounding_candidates else None
    visual_evidence = load_visual_evidence(args.visual_evidence)
    prompt = render_part_semantic_label_prompt(
        tool_part_graph=tool_part_graph,
        visual_evidence=visual_evidence,
        task_instruction=args.task or None,
        label_id=args.label_id,
    )
    if args.prompt_output:
        Path(args.prompt_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prompt_output).write_text(prompt, encoding="utf-8")
    if args.sampart_semantic_map:
        labels = build_from_sampart_semantic_map(
            tool_part_graph=tool_part_graph,
            semantic_map=_read_json(args.sampart_semantic_map),
            label_id=args.label_id,
            semantic_key=args.sampart_semantic_key,
            source_artifact=args.sampart_semantic_map,
        )
    elif args.raw_response:
        labels = parse_part_semantic_label_response(
            Path(args.raw_response).read_text(encoding="utf-8"),
            tool_part_graph=tool_part_graph,
            label_id=args.label_id,
            prompt_path=args.prompt_output,
            raw_response_path=args.raw_response,
        )
    else:
        labels = build_unlabeled_scaffold(
            tool_part_graph=tool_part_graph,
            label_id=args.label_id,
            source_artifact=args.tool_part_graph,
        )
    _write_json(args.output, labels)
    if args.merged_tool_part_graph or args.merged_grounding_candidates:
        merged_tool, merged_candidates = merge_part_semantic_labels(
            tool_part_graph=tool_part_graph,
            semantic_labels=labels,
            grounding_candidates=grounding_candidates,
        )
        if args.merged_tool_part_graph:
            _write_json(args.merged_tool_part_graph, merged_tool)
        if args.merged_grounding_candidates and merged_candidates is not None:
            _write_json(args.merged_grounding_candidates, merged_candidates)
    if args.visual_evidence_output:
        if visual_evidence is None:
            raise RuntimeError("--visual-evidence-output requires --visual-evidence")
        _write_json(
            args.visual_evidence_output,
            augment_visual_evidence_with_part_semantics(
                visual_evidence=visual_evidence,
                semantic_labels=labels,
            ),
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
