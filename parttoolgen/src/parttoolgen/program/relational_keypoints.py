#!/usr/bin/env python3
"""Prompt-safe relational keypoint cards for VLM ref binding.

The cards summarize which symbolic refs belong together without exposing raw
metric geometry. They are advisory prompt context only; ProgramContext still
uses the compact symbol table as the executable whitelist.
"""

from __future__ import annotations

from typing import Any


_TOOL_FUNCTIONAL_ATTRIBUTES = ("center", "normal", "axis")
_SCENE_TARGET_COLLECTIONS = ("keypoints", "regions", "surfaces")


def _flatten_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str) and "." in value:
        refs.add(value)
    elif isinstance(value, dict):
        for child in value.values():
            refs.update(_flatten_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_flatten_refs(child))
    return refs


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _roles(entity: dict[str, Any]) -> list[str]:
    roles = _string_list(entity.get("role_candidates"))
    role = entity.get("role")
    if isinstance(role, str) and role:
        roles.append(role)
    result: list[str] = []
    seen: set[str] = set()
    for role_item in roles:
        if role_item in seen:
            continue
        result.append(role_item)
        seen.add(role_item)
    return result


def _confidence(entity: dict[str, Any]) -> float | None:
    value = entity.get("confidence")
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return None


def _source(entity: dict[str, Any]) -> str | None:
    value = entity.get("source")
    return str(value) if isinstance(value, str) and value else None


def _part_id_from_label(value: Any) -> str | None:
    if isinstance(value, int):
        return f"part_{value:03d}"
    return None


def _infer_functional_part_id(
    functional_part_id: str,
    functional_part: dict[str, Any],
    part_ids: set[str],
) -> str | None:
    provenance = functional_part.get("provenance", {})
    if isinstance(provenance, dict):
        from_label = _part_id_from_label(provenance.get("part_label"))
        if from_label in part_ids:
            return from_label
    for part_id in sorted(part_ids, key=len, reverse=True):
        if functional_part_id == part_id or functional_part_id.startswith(f"{part_id}_"):
            return part_id
    return None


def _tool_functional_refs(
    functional_part_id: str,
    functional_part: dict[str, Any],
    allowed_refs: set[str],
) -> dict[str, Any] | None:
    refs: dict[str, str] = {}
    for attribute in _TOOL_FUNCTIONAL_ATTRIBUTES:
        ref = f"tool.functional_parts.{functional_part_id}.{attribute}"
        if attribute in functional_part and ref in allowed_refs:
            refs[f"{attribute}_ref"] = ref
    if "center_ref" not in refs:
        return None
    card: dict[str, Any] = {
        "functional_part_id": functional_part_id,
        "role": str(functional_part.get("role", "")),
        "executable_tool_refs": refs,
    }
    confidence = _confidence(functional_part)
    source = _source(functional_part)
    if confidence is not None:
        card["confidence"] = confidence
    if source:
        card["source"] = source
    return card


def _anchor_rows_by_part(
    visual_evidence_summary: dict[str, Any] | None,
    *,
    allowed_refs: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(visual_evidence_summary, dict):
        return {}
    numbered = visual_evidence_summary.get("numbered_anchor_refs", {})
    rows = numbered.get("rows", []) if isinstance(numbered, dict) else []
    if not isinstance(rows, list):
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        part_id = row.get("part_id")
        if not isinstance(part_id, str) or not part_id:
            continue
        compact: dict[str, Any] = {}
        for key in ("number", "anchor_id", "view_index", "feature_cluster_id", "score"):
            if key in row:
                compact[key] = row[key]
        for key, prompt_key in (
            ("part_ref", "part_ref"),
            ("point_ref", "candidate_point_ref"),
            ("axis_ref", "candidate_axis_ref"),
            ("surface_ref", "candidate_surface_ref"),
        ):
            ref = row.get(key)
            if isinstance(ref, str) and ref in allowed_refs:
                compact[prompt_key] = ref
        if "score" in compact and isinstance(compact["score"], (int, float)):
            compact["score"] = round(float(compact["score"]), 3)
        if compact:
            grouped.setdefault(part_id, []).append(compact)
    return grouped


def _candidate_refs_by_part(
    grounding_candidates: dict[str, Any] | None,
    *,
    allowed_refs: set[str],
) -> dict[str, dict[str, list[str]]]:
    if not isinstance(grounding_candidates, dict):
        return {}
    grouped: dict[str, dict[str, list[str]]] = {}
    for collection in ("points", "axes", "surfaces", "parts"):
        entities = grounding_candidates.get(collection, {})
        if not isinstance(entities, dict):
            continue
        for entity_id, entity in sorted(entities.items()):
            if not isinstance(entity, dict):
                continue
            part_ref = entity.get("part_ref")
            if not isinstance(part_ref, str) or not part_ref.startswith("tool.parts."):
                continue
            part_id = part_ref.split(".")[-1]
            ref = f"candidate.{collection}.{entity_id}"
            if ref not in allowed_refs:
                continue
            grouped.setdefault(part_id, {}).setdefault(collection, []).append(ref)
    return grouped


def _scene_refs_for_object(
    observed_scene_graph: dict[str, Any] | None,
    *,
    object_ref: str | None,
    allowed_refs: set[str],
) -> dict[str, list[str]]:
    if not isinstance(observed_scene_graph, dict) or not object_ref:
        return {}
    object_id = object_ref.split(".")[-1]
    matches = {object_ref, object_id}
    grouped: dict[str, list[str]] = {}
    for collection in _SCENE_TARGET_COLLECTIONS:
        entities = observed_scene_graph.get(collection, {})
        if not isinstance(entities, dict):
            continue
        refs: list[str] = []
        for entity_id, entity in sorted(entities.items()):
            if not isinstance(entity, dict):
                continue
            if entity.get("object_ref") not in matches:
                continue
            ref = f"scene.{collection}.{entity_id}"
            if ref in allowed_refs:
                refs.append(ref)
        if refs:
            grouped[collection] = refs
    return grouped


def _target_axis_cards(
    target_part_graph: dict[str, Any] | None,
    observed_scene_graph: dict[str, Any] | None,
    *,
    allowed_refs: set[str],
    max_cards: int,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if isinstance(target_part_graph, dict):
        axes = target_part_graph.get("axes", {})
        if isinstance(axes, dict):
            for axis_id, axis in sorted(axes.items()):
                if not isinstance(axis, dict):
                    continue
                ref = f"target.axes.{axis_id}"
                if ref not in allowed_refs:
                    continue
                card: dict[str, Any] = {
                    "axis_id": axis_id,
                    "axis_ref": ref,
                }
                object_ref = axis.get("object_ref")
                if isinstance(object_ref, str) and object_ref in allowed_refs:
                    card["object_ref"] = object_ref
                source = _source(axis)
                if source:
                    card["source"] = source
                cards.append(card)
                if len(cards) >= max_cards:
                    return cards
    if isinstance(observed_scene_graph, dict):
        axes = observed_scene_graph.get("axes", {})
        if isinstance(axes, dict):
            for axis_id, axis in sorted(axes.items()):
                if not isinstance(axis, dict):
                    continue
                ref = f"scene.axes.{axis_id}"
                if ref not in allowed_refs:
                    continue
                card = {"axis_id": axis_id, "axis_ref": ref}
                object_ref = axis.get("object_ref")
                if isinstance(object_ref, str) and object_ref:
                    if object_ref in allowed_refs:
                        card["object_ref"] = object_ref
                    elif "." not in object_ref:
                        card["object_id"] = object_ref
                cards.append(card)
                if len(cards) >= max_cards:
                    break
    return cards


def build_relational_keypoint_cards(
    *,
    symbol_table: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    observed_scene_graph: dict[str, Any] | None = None,
    grounding_candidates: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    visual_evidence_summary: dict[str, Any] | None = None,
    max_tool_cards: int = 24,
    max_functional_refs_per_card: int = 8,
    max_visual_anchors_per_card: int = 8,
    max_candidate_refs_per_card: int = 8,
    max_target_cards: int = 16,
    max_target_axes: int = 16,
) -> dict[str, Any]:
    """Build prompt-only relational cards linking evidence to executable refs."""

    allowed_refs = _flatten_refs(symbol_table)
    tool_parts = tool_part_graph.get("parts", {}) if isinstance(tool_part_graph, dict) else {}
    if not isinstance(tool_parts, dict):
        tool_parts = {}
    functional_parts = functional_tool_parts.get("functional_parts", {})
    if not isinstance(functional_parts, dict):
        functional_parts = {}

    part_ids = set(tool_parts)
    if not part_ids:
        for functional_part_id in functional_parts:
            prefix = functional_part_id.split("_", 2)
            if len(prefix) >= 2 and prefix[0] == "part" and prefix[1].isdigit():
                part_ids.add(f"part_{int(prefix[1]):03d}")

    functional_by_part: dict[str, list[dict[str, Any]]] = {}
    for functional_part_id, functional_part in sorted(functional_parts.items()):
        if not isinstance(functional_part, dict):
            continue
        part_id = _infer_functional_part_id(functional_part_id, functional_part, part_ids)
        if not part_id:
            continue
        ref_card = _tool_functional_refs(functional_part_id, functional_part, allowed_refs)
        if ref_card is None:
            continue
        functional_by_part.setdefault(part_id, []).append(ref_card)

    visual_by_part = _anchor_rows_by_part(
        visual_evidence_summary,
        allowed_refs=allowed_refs,
    )
    candidates_by_part = _candidate_refs_by_part(
        grounding_candidates,
        allowed_refs=allowed_refs,
    )

    tool_cards: list[dict[str, Any]] = []
    for part_id in sorted(part_ids):
        functional_refs = functional_by_part.get(part_id, [])
        visual_refs = visual_by_part.get(part_id, [])
        candidate_refs = candidates_by_part.get(part_id, {})
        if not functional_refs and not visual_refs and not candidate_refs:
            continue
        part = tool_parts.get(part_id, {}) if isinstance(tool_parts.get(part_id), dict) else {}
        part_ref = f"tool.parts.{part_id}"
        card: dict[str, Any] = {
            "part_id": part_id,
            "part_label": str(part.get("label", part_id)),
            "role_candidates": _roles(part),
        }
        if part_ref in allowed_refs:
            card["part_ref"] = part_ref
        confidence = _confidence(part)
        source = _source(part)
        if confidence is not None:
            card["confidence"] = confidence
        if source:
            card["source"] = source
        if functional_refs:
            card["functional_refs"] = functional_refs[:max_functional_refs_per_card]
        if visual_refs:
            card["visual_anchor_evidence"] = visual_refs[:max_visual_anchors_per_card]
        if candidate_refs:
            card["candidate_evidence_refs"] = {
                key: refs[:max_candidate_refs_per_card]
                for key, refs in sorted(candidate_refs.items())
                if refs
            }
        tool_cards.append(card)
        if len(tool_cards) >= max_tool_cards:
            break

    target_cards: list[dict[str, Any]] = []
    if isinstance(target_part_graph, dict):
        target_parts = target_part_graph.get("parts", {})
        if isinstance(target_parts, dict):
            for target_part_id, target_part in sorted(target_parts.items()):
                if not isinstance(target_part, dict):
                    continue
                target_ref = f"target.parts.{target_part_id}"
                if target_ref not in allowed_refs:
                    continue
                object_ref = target_part.get("object_ref")
                object_ref_str = object_ref if isinstance(object_ref, str) else None
                card = {
                    "target_part_id": target_part_id,
                    "target_ref": target_ref,
                    "label": str(target_part.get("label", target_part_id)),
                    "role_candidates": _roles(target_part),
                    "usable_target_refs": {
                        "part_ref": target_ref,
                        **_scene_refs_for_object(
                            observed_scene_graph,
                            object_ref=object_ref_str,
                            allowed_refs=allowed_refs,
                        ),
                    },
                }
                if object_ref_str in allowed_refs:
                    card["object_ref"] = object_ref_str
                patch_roles = []
                patches = target_part.get("surface_patches", {})
                if isinstance(patches, dict):
                    for patch_id, patch in sorted(patches.items()):
                        if not isinstance(patch, dict):
                            continue
                        role_hint = patch.get("role_hint")
                        if isinstance(role_hint, str) and role_hint:
                            patch_roles.append({"patch_id": patch_id, "role_hint": role_hint})
                if patch_roles:
                    card["surface_patch_roles"] = patch_roles
                confidence = _confidence(target_part)
                source = _source(target_part)
                if confidence is not None:
                    card["confidence"] = confidence
                if source:
                    card["source"] = source
                target_cards.append(card)
                if len(target_cards) >= max_target_cards:
                    break

    return {
        "schema_version": "v0.1",
        "usage_contract": (
            "Relational keypoint cards are prompt-only binding aids. Use them to "
            "choose a small slot set, then write build_program(ctx) with exact "
            "refs from symbol_table. visual_anchor_evidence and "
            "candidate_evidence_refs are executable only when the exact "
            "tool-frame candidate point/surface ref is listed in the active "
            "binding slot table or symbol_table."
        ),
        "slot_schema": {
            "tool_contact_point": (
                "tool.functional_parts.*.center from functional_refs, or listed "
                "tool-frame candidate.points/surfaces refs"
            ),
            "tool_contact_axis": (
                "same functional part's tool.functional_parts.*.normal, "
                "or listed tool-frame candidate.surfaces refs"
            ),
            "target_contact": (
                "target part, scene keypoint/region/surface, or candidate target "
                "point listed in symbol_table"
            ),
            "target_approach_axis": "target.axes.* or scene.axes.* listed in symbol_table",
            "motion_start_or_goal": "scene/candidate/target refs listed in symbol_table",
        },
        "tool_cards": tool_cards,
        "target_cards": target_cards,
        "target_axis_cards": _target_axis_cards(
            target_part_graph,
            observed_scene_graph,
            allowed_refs=allowed_refs,
            max_cards=max_target_axes,
        ),
    }
