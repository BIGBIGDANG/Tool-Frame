#!/usr/bin/env python3
"""Symbol-table helpers for part-centric affordance lowering prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parttoolgen.validation.validate_parttoolgen import (
    validate_grounding_candidates,
    validate_functional_tool_parts,
    validate_observed_scene_graph,
)


PROMPT_TEMPLATE_ID = "parttoolgen.part_centric_affordance_lowering.v1"
_PROMPT_TOOL_ATTRIBUTE_SUFFIXES = (".center", ".normal", ".axis", ".tangent_long", ".tangent_short")
_PROMPT_CANDIDATE_REF_FIELDS = ("part_ref", "point_ref", "axis_ref", "surface_ref")


def _filter_refs_by_suffix(refs: list[Any], suffixes: tuple[str, ...]) -> list[str]:
    return [
        ref
        for ref in refs
        if isinstance(ref, str) and any(ref.endswith(suffix) for suffix in suffixes)
    ]


def _visual_candidate_refs(visual_evidence_summary: dict[str, Any] | None) -> set[str]:
    refs: set[str] = set()
    if not isinstance(visual_evidence_summary, dict):
        return refs
    numbered = visual_evidence_summary.get("numbered_anchor_refs", {})
    rows = numbered.get("rows", []) if isinstance(numbered, dict) else []
    if not isinstance(rows, list):
        return refs
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in _PROMPT_CANDIDATE_REF_FIELDS:
            ref = row.get(key)
            if isinstance(ref, str):
                refs.add(ref)
    return refs


def _filter_candidate_symbols(
    candidate_symbols: dict[str, Any],
    *,
    visual_refs: set[str],
    max_refs_per_kind: int,
) -> dict[str, list[str]]:
    compact: dict[str, list[str]] = {}
    for kind, refs in candidate_symbols.items():
        if not isinstance(refs, list):
            continue
        string_refs = [ref for ref in refs if isinstance(ref, str)]
        if visual_refs:
            selected = [ref for ref in string_refs if ref in visual_refs]
        else:
            selected = string_refs[:max_refs_per_kind]
        compact[kind] = selected
    return compact


def _compact_functional_role_index(role_index: dict[str, Any]) -> dict[str, Any]:
    compact = json.loads(json.dumps(role_index))
    functional = compact.get("tool_functional_part_attributes", {})
    if not isinstance(functional, dict):
        return compact
    for role, by_attribute in list(functional.items()):
        if not isinstance(by_attribute, dict):
            del functional[role]
            continue
        for attribute in list(by_attribute):
            if attribute not in {"center", "normal", "axis", "tangent_long", "tangent_short"}:
                del by_attribute[attribute]
    return compact


def compact_symbol_table_for_prompt(
    symbol_table: dict[str, Any],
    *,
    visual_evidence_summary: dict[str, Any] | None = None,
    max_candidate_refs_per_kind: int = 32,
) -> dict[str, Any]:
    """Return a prompt-facing symbol table without compiler-only refs."""

    visual_refs = _visual_candidate_refs(visual_evidence_summary)
    tool_symbols = dict(symbol_table.get("tool_symbols", {}))
    functional_refs = tool_symbols.get("functional_parts", [])
    if isinstance(functional_refs, list):
        tool_symbols["functional_parts"] = _filter_refs_by_suffix(
            functional_refs,
            _PROMPT_TOOL_ATTRIBUTE_SUFFIXES,
        )
    return {
        "scene_id": symbol_table.get("scene_id"),
        "tool_id": symbol_table.get("tool_id"),
        "scene_symbols": symbol_table.get("scene_symbols", {}),
        "tool_symbols": tool_symbols,
        "target_symbols": symbol_table.get("target_symbols", {}),
        "candidate_symbols": _filter_candidate_symbols(
            symbol_table.get("candidate_symbols", {}),
            visual_refs=visual_refs,
            max_refs_per_kind=max_candidate_refs_per_kind,
        ),
        "role_index": _compact_functional_role_index(symbol_table.get("role_index", {})),
    }


def _symbol_refs(prefix: str, collection: dict[str, Any]) -> list[str]:
    return [f"{prefix}.{entity_id}" for entity_id in sorted(collection)]


def _candidate_refs(grounding_candidates: dict[str, Any] | None) -> dict[str, list[str]]:
    if not grounding_candidates:
        return {}
    validate_grounding_candidates(grounding_candidates)
    refs: dict[str, list[str]] = {}
    for key in ("marks", "points", "regions", "axes", "surfaces", "parts"):
        values = grounding_candidates.get(key, {})
        if isinstance(values, dict):
            refs[key] = [f"candidate.{key}.{candidate_id}" for candidate_id in sorted(values)]
    return refs


def _role_index(prefix: str, collection: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for entity_id, entity in sorted(collection.items()):
        if not isinstance(entity, dict):
            continue
        ref = f"{prefix}.{entity_id}"
        roles = entity.get("role_candidates") or []
        if isinstance(roles, str):
            roles = [roles]
        role = entity.get("role")
        if isinstance(role, str) and role:
            roles = [*roles, role]
        for role in roles:
            role_key = str(role)
            if role_key:
                index.setdefault(role_key, []).append(ref)
        label = entity.get("label")
        if isinstance(label, str) and label:
            index.setdefault(f"label:{label}", []).append(ref)
        semantic_label = entity.get("semantic_label") or entity.get("part_semantic_label")
        if isinstance(semantic_label, str) and semantic_label:
            index.setdefault(f"semantic_label:{semantic_label}", []).append(ref)
        aliases = entity.get("semantic_aliases") or entity.get("part_semantic_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias:
                    index.setdefault(f"semantic_alias:{alias}", []).append(ref)
    return index


def _functional_part_role_index(prefix: str, collection: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    index: dict[str, dict[str, list[str]]] = {}
    for entity_id, entity in sorted(collection.items()):
        if not isinstance(entity, dict):
            continue
        refs = {
            attribute: f"{prefix}.{entity_id}.{attribute}"
            for attribute in ("center", "normal", "axis", "frame", "T_tool_functional")
            if attribute in entity
        }
        if not refs:
            continue
        roles = entity.get("role_candidates") or []
        if isinstance(roles, str):
            roles = [roles]
        role = entity.get("role")
        if isinstance(role, str) and role:
            roles = [*roles, role]
        label = entity.get("label")
        if isinstance(label, str) and label:
            roles = [*roles, f"label:{label}"]
        semantic_label = entity.get("semantic_label") or entity.get("part_semantic_label")
        if isinstance(semantic_label, str) and semantic_label:
            roles = [*roles, f"semantic_label:{semantic_label}"]
        aliases = entity.get("semantic_aliases") or entity.get("part_semantic_aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(aliases, list):
            roles = [
                *roles,
                *[
                    f"semantic_alias:{alias}"
                    for alias in aliases
                    if isinstance(alias, str) and alias
                ],
            ]
        for role in roles:
            role_key = str(role)
            if not role_key:
                continue
            role_bucket = index.setdefault(role_key, {})
            for attribute, ref in refs.items():
                role_bucket.setdefault(attribute, []).append(ref)
    return index


def build_program_symbol_table(
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    *,
    grounding_candidates: dict[str, Any] | None = None,
    tool_part_graph: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return symbol refs that restricted planner programs may pass to `ctx.ref(...)`."""

    validate_observed_scene_graph(observed_scene_graph)
    validate_functional_tool_parts(functional_tool_parts)
    scene_symbols = {
        collection: _symbol_refs(f"scene.{collection}", observed_scene_graph.get(collection, {}))
        for collection in ("objects", "keypoints", "axes", "regions", "surfaces")
    }
    tool_symbols: dict[str, list[str]] = {
        "functional_parts": [],
        "grasp_references": [],
        "parts": [],
    }
    for part_id, part in sorted(functional_tool_parts["functional_parts"].items()):
        for attribute in (
            "center",
            "normal",
            "axis",
            "tangent_long",
            "tangent_short",
            "frame",
            "T_tool_functional",
        ):
            if attribute in part:
                tool_symbols["functional_parts"].append(
                    f"tool.functional_parts.{part_id}.{attribute}"
                )
        frame = part.get("frame") if isinstance(part, dict) else None
        if isinstance(frame, dict):
            if "tangent_long" not in part and "axis" in part:
                tool_symbols["functional_parts"].append(f"tool.functional_parts.{part_id}.tangent_long")
            if "tangent_short" not in part and "y_axis" in frame:
                tool_symbols["functional_parts"].append(f"tool.functional_parts.{part_id}.tangent_short")
    for grasp_id, grasp in sorted(functional_tool_parts["grasp_references"].items()):
        provenance = grasp.get("provenance", {}) if isinstance(grasp, dict) else {}
        method = str(provenance.get("method", "")) if isinstance(provenance, dict) else ""
        if grasp_id == "schema_required_not_policy_input" or "not_policy_input" in method:
            continue
        tool_symbols["grasp_references"].append(f"tool.grasp_references.{grasp_id}")
    if tool_part_graph:
        for part_id in sorted(tool_part_graph.get("parts", {})):
            tool_symbols["parts"].append(f"tool.parts.{part_id}")

    target_symbols: dict[str, list[str]] = {"parts": [], "axes": [], "regions": []}
    if target_part_graph:
        for part_id in sorted(target_part_graph.get("parts", {})):
            target_symbols["parts"].append(f"target.parts.{part_id}")
        for axis_id in sorted(target_part_graph.get("axes", {})):
            target_symbols["axes"].append(f"target.axes.{axis_id}")
        for region_id in sorted(target_part_graph.get("regions", {})):
            target_symbols["regions"].append(f"target.regions.{region_id}")

    role_index: dict[str, Any] = {}
    if functional_tool_parts:
        role_index["tool_functional_part_attributes"] = _functional_part_role_index(
            "tool.functional_parts",
            functional_tool_parts.get("functional_parts", {}),
        )
    if tool_part_graph:
        role_index["tool_parts"] = _role_index("tool.parts", tool_part_graph.get("parts", {}))
    if target_part_graph:
        role_index["target_parts"] = _role_index("target.parts", target_part_graph.get("parts", {}))
    if grounding_candidates:
        role_index["candidate_parts"] = _role_index(
            "candidate.parts",
            grounding_candidates.get("parts", {}),
        )

    return {
        "scene_id": observed_scene_graph["scene_id"],
        "tool_id": functional_tool_parts["tool_id"],
        "scene_symbols": scene_symbols,
        "tool_symbols": tool_symbols,
        "target_symbols": target_symbols,
        "candidate_symbols": _candidate_refs(grounding_candidates),
        "role_index": role_index,
    }


def flatten_allowed_refs(symbol_table: dict[str, Any]) -> set[str]:
    """Flatten nested symbol-table lists into the exact ref whitelist."""

    refs: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and "." in value:
            refs.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(symbol_table)
    return refs


def _first_ref(symbol_table: dict[str, Any], path: tuple[str, ...], suffix: str | None = None) -> str | None:
    value: Any = symbol_table
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if not isinstance(value, list):
        return None
    for ref in value:
        if isinstance(ref, str) and (suffix is None or ref.endswith(suffix)):
            return ref
    return None


def _preferred_ref_by_tokens(
    symbol_table: dict[str, Any],
    path: tuple[str, ...],
    token_groups: tuple[tuple[str, ...], ...],
) -> str | None:
    value: Any = symbol_table
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if not isinstance(value, list):
        return None
    refs = [ref for ref in value if isinstance(ref, str)]
    for token_group in token_groups:
        for ref in refs:
            lowered = ref.lower()
            if all(token in lowered for token in token_group):
                return ref
    return refs[0] if refs else None


def _first_role_ref(symbol_table: dict[str, Any], group: str, roles: tuple[str, ...]) -> str | None:
    role_index = symbol_table.get("role_index", {})
    if not isinstance(role_index, dict):
        return None
    values = role_index.get(group, {})
    if not isinstance(values, dict):
        return None
    for role in roles:
        refs = values.get(role)
        if isinstance(refs, list) and refs:
            return refs[0]
    return None


def _first_functional_role_attribute(
    symbol_table: dict[str, Any],
    roles: tuple[str, ...],
    attribute: str,
) -> str | None:
    role_index = symbol_table.get("role_index", {})
    if not isinstance(role_index, dict):
        return None
    values = role_index.get("tool_functional_part_attributes", {})
    if not isinstance(values, dict):
        return None
    for role in roles:
        by_attribute = values.get(role)
        if not isinstance(by_attribute, dict):
            continue
        refs = by_attribute.get(attribute)
        if isinstance(refs, list) and refs:
            return refs[0]
    return None


def _rounded_vec(values: Any) -> list[float] | None:
    if not isinstance(values, list) or len(values) != 3:
        return None
    rounded: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            return None
        rounded.append(round(float(value), 4))
    return rounded


def _scene_object_ref(object_id: str) -> str:
    return object_id if object_id.startswith("scene.") else f"scene.objects.{object_id}"


def _scene_refs_by_object(
    observed_scene_graph: dict[str, Any],
    *,
    object_ref: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "keypoints": [],
        "surfaces": [],
        "regions": [],
        "axes": [],
    }
    raw_object_id = (
        object_ref[len("scene.objects.") :]
        if object_ref.startswith("scene.objects.")
        else object_ref
    )
    for collection, ref_prefix in (
        ("keypoints", "scene.keypoints"),
        ("surfaces", "scene.surfaces"),
        ("regions", "scene.regions"),
        ("axes", "scene.axes"),
    ):
        entities = observed_scene_graph.get(collection, {})
        if not isinstance(entities, dict):
            continue
        for entity_id, entity in sorted(entities.items()):
            if not isinstance(entity, dict):
                continue
            entity_object = str(entity.get("object_ref", ""))
            inferred_by_id = (
                not entity_object
                and (
                    (raw_object_id.startswith("target_") and entity_id.startswith("target_"))
                    or (
                        raw_object_id.startswith("support_")
                        and entity_id.startswith("support_")
                    )
                )
            )
            if entity_object not in {object_ref, raw_object_id} and not inferred_by_id:
                continue
            grouped[collection].append(
                {
                    "ref": f"{ref_prefix}.{entity_id}",
                    "role": entity.get("provenance", {}).get("semantic_role")
                    or entity.get("role")
                    or entity_id,
                    "point_xyz_world": _rounded_vec(
                        entity.get("xyz_world")
                        or entity.get("point_xyz_world")
                        or entity.get("center_xyz_world")
                    ),
                    "normal_world": _rounded_vec(
                        entity.get("normal_world")
                        or entity.get("vector_world")
                    ),
                }
            )
    return grouped


def build_scene_planning_cards(
    *,
    observed_scene_graph: dict[str, Any],
    target_part_graph: dict[str, Any] | None = None,
    max_targets: int = 4,
) -> dict[str, Any]:
    """Build prompt-only scene cards for VLM route planning.

    These cards expose target/support relationships and usable symbolic refs.
    Legacy planner-program prompts may use the refs in `build_program(ctx)`,
    but must not emit raw coordinates from the card values.
    """

    objects = observed_scene_graph.get("objects", {})
    target_cards: list[dict[str, Any]] = []
    support_cards: list[dict[str, Any]] = []
    if isinstance(objects, dict):
        for object_id, obj in sorted(objects.items()):
            if not isinstance(obj, dict):
                continue
            category = str(obj.get("category", ""))
            attrs = obj.get("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
            object_ref = _scene_object_ref(object_id)
            card = {
                "object_ref": object_ref,
                "category": category,
                "center_xyz_world": _rounded_vec(attrs.get("center_xyz_world")),
                "extent_xyz": _rounded_vec(attrs.get("extent_xyz")),
            }
            if category == "support_surface" or attrs.get("role") == "support_surface":
                card["usable_refs"] = _scene_refs_by_object(
                    observed_scene_graph,
                    object_ref=object_ref,
                )
                support_cards.append(card)
                continue
            if attrs.get("role") == "target_object" or object_id.startswith("target_"):
                card["usable_refs"] = _scene_refs_by_object(
                    observed_scene_graph,
                    object_ref=object_ref,
                )
                target_cards.append(card)
                if len(target_cards) >= max_targets:
                    break

    target_graph_summary: list[dict[str, Any]] = []
    if isinstance(target_part_graph, dict):
        for part_id, part in sorted(target_part_graph.get("parts", {}).items()):
            if not isinstance(part, dict):
                continue
            target_graph_summary.append(
                {
                    "part_ref": f"target.parts.{part_id}",
                    "label": part.get("label", part_id),
                    "roles": part.get("roles", []),
                    "object_ref": part.get("object_ref"),
                }
            )

    return {
        "purpose": (
            "Prompt-only scene route evidence. Use listed refs to choose "
            "free-space pre-contact, contact, and motion-path constraints; do "
            "not output raw coordinates."
        ),
        "target_objects": target_cards,
        "support_surfaces": support_cards,
        "target_part_refs": target_graph_summary,
    }
