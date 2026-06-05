#!/usr/bin/env python3
"""Legacy V0 task-agnostic affordance binding helpers.

This module belongs to the older symbolic affordance-binding planner stack. The
current image-grounded benchmark mainline uses scene refs plus
``tool.contact_point`` and ``tool.motion_axis`` refs
instead. Keep this module for legacy/debug workflows, but do not treat its
``function_*`` field names as current benchmark interfaces.

This module has no VLM client dependency. It only prepares prompt payloads,
parses JSON responses, and validates that selected slots are exact symbolic refs
from the allowed binding table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


AFFORDANCE_BINDING_TEMPLATE_ID = "parttoolgen.affordance_binding.v1"
AFFORDANCE_BINDING_TEMPLATE_VERSION = "v1"
AFFORDANCE_BINDING_SCHEMA_V1 = "v0.1"
AFFORDANCE_BINDING_SCHEMA_V2 = "v0.2"
AFFORDANCE_BINDING_SCHEMA_VERSION = AFFORDANCE_BINDING_SCHEMA_V2
AFFORDANCE_BINDING_TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "prompt_templates"
    / "affordance_binding_v1.md"
)
__all__ = [
    "AFFORDANCE_BINDING_TEMPLATE_ID",
    "AFFORDANCE_BINDING_TEMPLATE_VERSION",
    "AFFORDANCE_BINDING_SCHEMA_VERSION",
    "AffordanceBindingError",
    "AffordanceBindingParseError",
    "collect_binding_allowed_refs",
    "filter_symbol_table_to_binding",
    "load_template",
    "parse_affordance_binding_response",
    "render_affordance_binding_prompt",
    "validate_affordance_binding_refs",
]

_DYNAMIC_PAYLOAD_PLACEHOLDER = "{{DYNAMIC_PAYLOAD_JSON}}"
_REF_PREFIXES = ("tool.", "scene.", "target.", "candidate.")
_SLOT_NAMES = (
    "grasp_ref",
    "function_keypoint_ref",
    "function_axis_ref",
    "target_pre_contact_ref",
    "target_contact_ref",
    "target_post_contact_ref",
    "motion_axis_ref",
)
_FUNCTIONAL_FRAME_SLOT_NAMES = (
    "contact_ref",
    "primary_ref",
    "axis_source_ref",
    "secondary_ref",
    "tertiary_ref",
    "parent_part_ref",
)
_TARGET_BINDING_SLOT_NAMES = (
    "target_pre_contact_ref",
    "target_contact_ref",
    "target_post_contact_ref",
    "motion_axis_ref",
)
_ALL_SLOT_NAMES = (*_SLOT_NAMES, *_FUNCTIONAL_FRAME_SLOT_NAMES)
_REQUIRED_NON_NULL_SLOTS = frozenset({"function_keypoint_ref", "target_contact_ref"})
_REQUIRED_NON_NULL_FRAME_SLOTS = frozenset({"contact_ref", "primary_ref"})
_FUNCTIONAL_FRAME_CONSTRUCTION = "function_point_center_axis_v1"
_LEGACY_FUNCTIONAL_FRAME_CONSTRUCTION = "multi_keypoint_frame_v1"
_FUNCTIONAL_FRAME_CONSTRUCTIONS = frozenset(
    {_FUNCTIONAL_FRAME_CONSTRUCTION, _LEGACY_FUNCTIONAL_FRAME_CONSTRUCTION}
)
_FUNCTION_KEYPOINT_SUFFIXES = (".center",)
_FUNCTION_AXIS_SUFFIXES = (".normal", ".axis", ".tangent_long", ".tangent_short")
_TARGET_CONTACT_SYMBOL_COLLECTIONS = frozenset(
    {"keypoints", "points", "regions", "surfaces", "parts"}
)
_AXIS_SYMBOL_COLLECTIONS = frozenset({"axes"})
_VISUAL_ROW_REF_KEYS = (
    "part_ref",
    "point_ref",
    "axis_ref",
    "surface_ref",
    "candidate_point_ref",
    "candidate_axis_ref",
    "candidate_surface_ref",
)
_VISUAL_POINT_REF_KEYS = ("point_ref", "candidate_point_ref")
_VISUAL_AXIS_REF_KEYS = ("axis_ref", "candidate_axis_ref")
_VISUAL_SURFACE_REF_KEYS = ("surface_ref", "candidate_surface_ref")


class AffordanceBindingError(ValueError):
    """Raised when an affordance binding artifact violates the contract."""


class AffordanceBindingParseError(AffordanceBindingError):
    """Raised when a VLM response cannot be parsed as an affordance binding."""


def load_template(template_path: str | Path | None = None) -> str:
    """Load the fixed JSON-only affordance binding prompt template."""

    path = (
        Path(template_path)
        if template_path is not None
        else AFFORDANCE_BINDING_TEMPLATE_PATH
    )
    template = path.read_text(encoding="utf-8")
    if _DYNAMIC_PAYLOAD_PLACEHOLDER not in template:
        raise ValueError(
            f"affordance binding template missing {_DYNAMIC_PAYLOAD_PLACEHOLDER}"
        )
    return template


def render_affordance_binding_prompt(
    *,
    task_instruction: str,
    symbol_table: dict[str, Any],
    visual_evidence: dict[str, Any] | None = None,
    relational_keypoint_cards: dict[str, Any] | None = None,
    scene_planning_cards: dict[str, Any] | None = None,
    extra_payload: dict[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> str:
    """Render a JSON-only prompt for task-conditioned symbolic slot binding."""

    if not isinstance(task_instruction, str) or not task_instruction:
        raise ValueError("task_instruction must be a non-empty string")
    if not isinstance(symbol_table, dict):
        raise ValueError("symbol_table must be a dict")

    allowed_refs = collect_binding_allowed_refs(
        symbol_table,
        visual_evidence=visual_evidence,
        relational_keypoint_cards=relational_keypoint_cards,
    )
    filtered_symbol_table = filter_symbol_table_to_binding(
        symbol_table,
        allowed_refs=allowed_refs,
    )
    allowed_ref_set, _ = _coerce_allowed_refs(allowed_refs)
    dynamic_payload: dict[str, Any] = {
        "prompt_template": {
            "id": AFFORDANCE_BINDING_TEMPLATE_ID,
            "version": AFFORDANCE_BINDING_TEMPLATE_VERSION,
        },
        "payload_compaction": {
            "symbol_table": "binding_relevant_refs_only_v1",
            "visual_evidence": "numbered_anchor_refs_binding_refs_v1",
            "relational_keypoint_cards": "prompt_only_binding_cards_v0.1",
        },
        "task_instruction": task_instruction,
        "allowed_refs": allowed_refs,
        "symbol_table": filtered_symbol_table,
        "relational_keypoint_cards": _filter_relational_keypoint_cards_to_binding(
            relational_keypoint_cards,
            allowed_ref_set=allowed_ref_set,
        ),
        "scene_planning_cards": scene_planning_cards,
        "visual_evidence": _filter_visual_evidence_to_binding(
            visual_evidence,
            allowed_ref_set=allowed_ref_set,
        ),
    }
    if extra_payload:
        dynamic_payload.update(extra_payload)

    payload_text = json.dumps(dynamic_payload, indent=2)
    return (
        load_template(template_path)
        .replace(_DYNAMIC_PAYLOAD_PLACEHOLDER, payload_text)
        .rstrip()
        + "\n"
    )


def parse_affordance_binding_response(
    raw_text: str,
    *,
    allowed_refs: dict[str, Any] | Iterable[str] | None = None,
) -> dict[str, Any]:
    """Parse and optionally ref-validate a JSON-only affordance binding reply."""

    if not isinstance(raw_text, str):
        raise AffordanceBindingParseError("response must be text")
    text = _strip_optional_json_fence(raw_text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AffordanceBindingParseError(f"invalid JSON response: {exc}") from exc

    binding = _normalize_affordance_binding(parsed)
    if allowed_refs is not None:
        validate_affordance_binding_refs(binding, allowed_refs)
    return binding


def collect_binding_allowed_refs(
    symbol_table: dict[str, Any],
    *,
    visual_evidence: dict[str, Any] | None = None,
    visual_evidence_summary: dict[str, Any] | None = None,
    relational_keypoint_cards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect slot-specific ref whitelists from prompt-safe binding inputs.

    The collection is task-agnostic: it uses the existing prompt symbol table,
    numbered visual anchor rows, and relational cards without assuming concrete
    tool names, object categories, or task verbs.
    """

    if not isinstance(symbol_table, dict):
        raise ValueError("symbol_table must be a dict")
    visual = visual_evidence if visual_evidence is not None else visual_evidence_summary

    symbol_refs = _dedupe(_iter_symbol_refs(symbol_table))
    symbol_ref_set = set(symbol_refs)

    visual_refs = _dedupe(
        ref
        for ref in _refs_from_numbered_anchor_rows(visual)
        if ref in symbol_ref_set
    )
    visual_function_point_refs = _only_known_refs(
        _tool_visual_refs_from_numbered_anchor_rows(
            visual,
            keys=_VISUAL_POINT_REF_KEYS,
        ),
        symbol_ref_set,
    )
    visual_function_keypoint_refs = _only_known_refs(
        [
            *visual_function_point_refs,
            *_tool_visual_refs_from_numbered_anchor_rows(
                visual,
                keys=_VISUAL_SURFACE_REF_KEYS,
            ),
        ],
        symbol_ref_set,
    )
    visual_function_axis_refs = _only_known_refs(
        _tool_visual_refs_from_numbered_anchor_rows(
            visual,
            keys=(*_VISUAL_AXIS_REF_KEYS, *_VISUAL_SURFACE_REF_KEYS),
        ),
        symbol_ref_set,
    )
    visual_parent_part_refs = _only_known_refs(
        _tool_part_refs_from_numbered_anchor_rows(visual),
        symbol_ref_set,
    )
    tool_visual_function_refs = set(
        _dedupe([*visual_function_keypoint_refs, *visual_function_axis_refs])
    )
    executable_tool_refs = _collect_relational_executable_tool_refs(
        relational_keypoint_cards,
        allowed_ref_set=symbol_ref_set,
    )
    relational_target_refs, relational_axis_refs = _collect_relational_target_refs(
        relational_keypoint_cards,
        allowed_ref_set=symbol_ref_set,
    )

    grasp_refs = _dedupe(
        _refs_at_path(symbol_table, ("tool_symbols", "grasp_references"))
    )
    function_keypoint_refs = _dedupe(
        [
            *visual_function_keypoint_refs,
            *executable_tool_refs.get("center_ref", []),
            *_filter_refs_by_suffix(
                _tool_functional_refs(symbol_table),
                _FUNCTION_KEYPOINT_SUFFIXES,
            ),
        ]
    )
    function_axis_refs = _dedupe(
        [
            *visual_function_axis_refs,
            *executable_tool_refs.get("normal_ref", []),
            *executable_tool_refs.get("axis_ref", []),
            *_filter_refs_by_suffix(
                _tool_functional_refs(symbol_table),
                _FUNCTION_AXIS_SUFFIXES,
            ),
        ]
    )
    functional_frame_point_refs = _dedupe(
        [
            *visual_function_point_refs,
            *executable_tool_refs.get("center_ref", []),
            *_filter_refs_by_suffix(
                _tool_functional_refs(symbol_table),
                _FUNCTION_KEYPOINT_SUFFIXES,
            ),
        ]
    )
    axis_source_refs = _dedupe(
        [
            *visual_parent_part_refs,
            *_refs_at_path(symbol_table, ("tool_symbols", "parts")),
        ]
    )
    target_contact_refs = _dedupe(
        ref
        for ref in [
            *relational_target_refs,
            *_target_contact_refs_from_symbol_table(symbol_table),
        ]
        if ref not in tool_visual_function_refs
    )
    motion_axis_refs = _dedupe(
        ref
        for ref in [
            *relational_axis_refs,
            *_axis_refs_from_symbol_table(symbol_table),
        ]
        if ref not in tool_visual_function_refs
    )

    slot_refs = {
        "grasp_ref": _only_known_refs(grasp_refs, symbol_ref_set),
        "function_keypoint_ref": _only_known_refs(
            function_keypoint_refs,
            symbol_ref_set,
        ),
        "function_axis_ref": _only_known_refs(function_axis_refs, symbol_ref_set),
        "target_pre_contact_ref": _only_known_refs(
            target_contact_refs,
            symbol_ref_set,
        ),
        "target_contact_ref": _only_known_refs(target_contact_refs, symbol_ref_set),
        "target_post_contact_ref": _only_known_refs(
            target_contact_refs,
            symbol_ref_set,
        ),
        "motion_axis_ref": _only_known_refs(motion_axis_refs, symbol_ref_set),
        "contact_ref": _only_known_refs(functional_frame_point_refs, symbol_ref_set),
        "primary_ref": _only_known_refs(functional_frame_point_refs, symbol_ref_set),
        "axis_source_ref": _only_known_refs(axis_source_refs, symbol_ref_set),
        "secondary_ref": [],
        "tertiary_ref": _only_known_refs(functional_frame_point_refs, symbol_ref_set),
        "parent_part_ref": _only_known_refs(axis_source_refs, symbol_ref_set),
    }

    all_refs = _dedupe(
        [
            *(ref for refs in slot_refs.values() for ref in refs),
            *visual_refs,
        ]
    )
    return {
        "schema_version": AFFORDANCE_BINDING_SCHEMA_VERSION,
        "usage_contract": (
            "All non-null slot_bindings and functional_frame_binding values "
            "must be copied exactly from slot_refs for that slot. Visual marker "
            "numbers are not refs. For the v0.2 functional frame, VLM selects "
            "the contact/function point and may bind a listed same-contact "
            "function_axis_ref when a reliable tool-side surface normal or axis "
            "is available. axis_source_ref is a tool/part-center ref derived by "
            "PartToolGen from parent_part_ref or the selected contact ref and is "
            "used only as a fallback orientation relation. secondary_ref is "
            "legacy-only and should remain null in new bindings. Geometry "
            "candidates are executable when listed; visual scores, when present, "
            "are only evidence."
        ),
        "slot_refs": slot_refs,
        "visual_anchor_refs": visual_refs,
        "relational_executable_tool_refs": executable_tool_refs,
        "all_refs": all_refs,
    }


def validate_affordance_binding_refs(
    binding: dict[str, Any],
    allowed_refs: dict[str, Any] | Iterable[str],
    *,
    require_slot_specific: bool = True,
) -> None:
    """Validate that every non-null binding slot uses an allowed symbolic ref."""

    normalized = _normalize_affordance_binding(binding)
    allowed_ref_set, slot_ref_sets = _coerce_allowed_refs(allowed_refs)
    for slot_name, ref in _iter_binding_slot_refs(normalized):
        if ref is None:
            continue
        if ref not in allowed_ref_set:
            raise AffordanceBindingError(
                f"{slot_name} ref is not in allowed_refs: {ref}"
            )
        slot_ref_set = slot_ref_sets.get(slot_name, set())
        if require_slot_specific and slot_name in slot_ref_sets and ref not in slot_ref_set:
            raise AffordanceBindingError(
                f"{slot_name} ref is not allowed for that slot: {ref}"
            )


def filter_symbol_table_to_binding(
    symbol_table: dict[str, Any],
    *,
    allowed_refs: dict[str, Any] | Iterable[str] | None = None,
    visual_evidence: dict[str, Any] | None = None,
    relational_keypoint_cards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a symbol table pruned to refs useful for affordance binding."""

    if not isinstance(symbol_table, dict):
        raise ValueError("symbol_table must be a dict")
    if allowed_refs is None:
        allowed_refs = collect_binding_allowed_refs(
            symbol_table,
            visual_evidence=visual_evidence,
            relational_keypoint_cards=relational_keypoint_cards,
        )
    allowed_ref_set, _ = _coerce_allowed_refs(allowed_refs)
    filtered = _filter_ref_tree(symbol_table, allowed_ref_set)
    return filtered if isinstance(filtered, dict) else {}


def _normalize_affordance_binding(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise AffordanceBindingError("affordance binding must be a JSON object")
    schema_version = data.get("schema_version")
    if schema_version == AFFORDANCE_BINDING_SCHEMA_V1:
        return _normalize_affordance_binding_v1(data)
    if schema_version == AFFORDANCE_BINDING_SCHEMA_V2:
        return _normalize_affordance_binding_v2(data)
    raise AffordanceBindingError(
        f"schema_version must be {AFFORDANCE_BINDING_SCHEMA_V1} or {AFFORDANCE_BINDING_SCHEMA_V2}"
    )


def _normalize_affordance_binding_v1(data: dict[str, Any]) -> dict[str, Any]:
    allowed_top_level = {"schema_version", "slot_bindings"}
    extra_top_level = sorted(set(data) - allowed_top_level)
    if extra_top_level:
        raise AffordanceBindingError(
            f"unexpected top-level fields: {', '.join(extra_top_level)}"
        )
    normalized_slots = _normalize_slot_bindings(
        data.get("slot_bindings"),
        require_all=True,
    )
    _require_non_null_slots(
        normalized_slots,
        required_slots=_REQUIRED_NON_NULL_SLOTS,
    )
    return {
        "schema_version": AFFORDANCE_BINDING_SCHEMA_V1,
        "slot_bindings": normalized_slots,
    }


def _normalize_affordance_binding_v2(data: dict[str, Any]) -> dict[str, Any]:
    allowed_top_level = {
        "schema_version",
        "slot_bindings",
        "functional_frame_binding",
        "target_binding",
    }
    extra_top_level = sorted(set(data) - allowed_top_level)
    if extra_top_level:
        raise AffordanceBindingError(
            f"unexpected top-level fields: {', '.join(extra_top_level)}"
        )
    missing_top_level = sorted(
        {"slot_bindings", "functional_frame_binding", "target_binding"} - set(data)
    )
    if missing_top_level:
        raise AffordanceBindingError(
            f"v0.2 affordance binding missing fields: {', '.join(missing_top_level)}"
        )
    normalized_slots = _normalize_slot_bindings(
        data.get("slot_bindings"),
        require_all=True,
    )
    frame_binding = _normalize_functional_frame_binding(
        data.get("functional_frame_binding"),
        slot_bindings=normalized_slots,
    )
    target_binding = _normalize_target_binding(
        data.get("target_binding"),
        slot_bindings=normalized_slots,
    )

    if normalized_slots["function_keypoint_ref"] is None:
        normalized_slots["function_keypoint_ref"] = frame_binding["contact_ref"]
    for slot_name in _TARGET_BINDING_SLOT_NAMES:
        if normalized_slots[slot_name] is None:
            normalized_slots[slot_name] = target_binding[slot_name]

    _require_non_null_slots(
        normalized_slots,
        required_slots=_REQUIRED_NON_NULL_SLOTS,
    )
    _require_non_null_slots(
        frame_binding,
        required_slots=_REQUIRED_NON_NULL_FRAME_SLOTS,
    )
    return {
        "schema_version": AFFORDANCE_BINDING_SCHEMA_V2,
        "slot_bindings": normalized_slots,
        "functional_frame_binding": frame_binding,
        "target_binding": target_binding,
    }


def _normalize_slot_bindings(value: Any, *, require_all: bool) -> dict[str, str | None]:
    if value is None and not require_all:
        slots: dict[str, Any] = {}
    elif isinstance(value, dict):
        slots = value
    else:
        raise AffordanceBindingError("slot_bindings must be an object")
    extra_slots = sorted(set(slots) - set(_SLOT_NAMES))
    if extra_slots:
        raise AffordanceBindingError(
            f"unexpected slot fields: {', '.join(extra_slots)}"
        )

    normalized_slots: dict[str, str | None] = {}
    for slot_name in _SLOT_NAMES:
        if slot_name not in slots:
            if require_all:
                raise AffordanceBindingError(f"slot_bindings missing {slot_name}")
            normalized_slots[slot_name] = None
            continue
        value = slots[slot_name]
        normalized_slots[slot_name] = _normalize_nullable_ref(value, slot_name)
    return normalized_slots


def _normalize_functional_frame_binding(
    value: Any,
    *,
    slot_bindings: dict[str, str | None],
) -> dict[str, Any]:
    if value is None:
        frame: dict[str, Any] = {}
    elif isinstance(value, dict):
        frame = value
    else:
        raise AffordanceBindingError("functional_frame_binding must be an object")
    allowed_fields = {*_FUNCTIONAL_FRAME_SLOT_NAMES, "construction"}
    extra_fields = sorted(set(frame) - allowed_fields)
    if extra_fields:
        raise AffordanceBindingError(
            f"unexpected functional_frame_binding fields: {', '.join(extra_fields)}"
        )

    fallback_contact = slot_bindings.get("function_keypoint_ref")
    normalized: dict[str, Any] = {}
    for slot_name in _FUNCTIONAL_FRAME_SLOT_NAMES:
        if slot_name == "axis_source_ref":
            continue
        fallback = fallback_contact if slot_name in {"contact_ref", "primary_ref"} else None
        normalized[slot_name] = _normalize_nullable_ref(
            frame.get(slot_name, fallback),
            slot_name,
        )
    if normalized["parent_part_ref"] is None:
        normalized["parent_part_ref"] = _derive_parent_part_ref(normalized["contact_ref"])
    normalized["axis_source_ref"] = _normalize_nullable_ref(
        frame.get("axis_source_ref") or normalized["parent_part_ref"],
        "axis_source_ref",
    )
    construction = frame.get("construction", _FUNCTIONAL_FRAME_CONSTRUCTION)
    if construction not in _FUNCTIONAL_FRAME_CONSTRUCTIONS:
        raise AffordanceBindingError(
            "functional_frame_binding.construction must be one of "
            f"{sorted(_FUNCTIONAL_FRAME_CONSTRUCTIONS)}"
        )
    normalized["construction"] = construction
    return normalized


def _normalize_target_binding(
    value: Any,
    *,
    slot_bindings: dict[str, str | None],
) -> dict[str, str | None]:
    if value is None:
        target: dict[str, Any] = {}
    elif isinstance(value, dict):
        target = value
    else:
        raise AffordanceBindingError("target_binding must be an object")
    extra_fields = sorted(set(target) - set(_TARGET_BINDING_SLOT_NAMES))
    if extra_fields:
        raise AffordanceBindingError(
            f"unexpected target_binding fields: {', '.join(extra_fields)}"
        )
    normalized: dict[str, str | None] = {}
    for slot_name in _TARGET_BINDING_SLOT_NAMES:
        normalized[slot_name] = _normalize_nullable_ref(
            target.get(slot_name, slot_bindings.get(slot_name)),
            slot_name,
        )
    return normalized


def _normalize_nullable_ref(value: Any, slot_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AffordanceBindingError(
            f"{slot_name} must be a non-empty string or null"
        )
    return value


def _require_non_null_slots(
    slots: dict[str, Any],
    *,
    required_slots: frozenset[str],
) -> None:
    for slot_name in sorted(required_slots):
        if slots.get(slot_name) is None:
            raise AffordanceBindingError(f"{slot_name} must be non-null")


def _iter_binding_slot_refs(binding: dict[str, Any]) -> Iterable[tuple[str, str | None]]:
    for slot_name, ref in binding.get("slot_bindings", {}).items():
        yield str(slot_name), ref
    frame_binding = binding.get("functional_frame_binding")
    if isinstance(frame_binding, dict):
        for slot_name in _FUNCTIONAL_FRAME_SLOT_NAMES:
            yield slot_name, frame_binding.get(slot_name)
    target_binding = binding.get("target_binding")
    if isinstance(target_binding, dict):
        for slot_name in _TARGET_BINDING_SLOT_NAMES:
            yield slot_name, target_binding.get(slot_name)


def _coerce_allowed_refs(
    allowed_refs: dict[str, Any] | Iterable[str],
) -> tuple[set[str], dict[str, set[str]]]:
    if isinstance(allowed_refs, dict):
        slot_source = allowed_refs.get("slot_refs")
        if isinstance(slot_source, dict):
            slot_refs = {
                slot_name: set(_string_list(slot_source.get(slot_name)))
                for slot_name in _ALL_SLOT_NAMES
                if slot_name in slot_source
            }
            all_refs = set(_string_list(allowed_refs.get("all_refs")))
            if not all_refs:
                all_refs = set().union(*slot_refs.values()) if slot_refs else set()
            return all_refs, slot_refs
        slot_refs = {
            slot_name: set(_string_list(allowed_refs.get(slot_name)))
            for slot_name in _ALL_SLOT_NAMES
            if slot_name in allowed_refs
        }
        if slot_refs:
            return set().union(*slot_refs.values()), slot_refs
        return set(_string_list(allowed_refs.get("all_refs"))), {}
    return set(_string_list(list(allowed_refs))), {}


def _strip_optional_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        first = lines[0].strip().lower()
        if first in {"```", "```json"}:
            return "\n".join(lines[1:-1]).strip()
    raise AffordanceBindingParseError("response must be a single JSON object")


def _derive_parent_part_ref(ref: str | None) -> str | None:
    if not isinstance(ref, str) or not ref:
        return None
    if ref.startswith("candidate.points."):
        entity_id = ref.split(".")[-1]
        match = re.match(r"^(part_\d{3})(?:_|$)", entity_id)
        if match:
            return f"tool.parts.{match.group(1)}"
        if "_anchor_" in entity_id:
            part_id = entity_id.split("_anchor_", 1)[0]
            return f"tool.parts.{part_id}"
    if ref.startswith("tool.functional_parts."):
        functional_id = ref.split(".")[2]
        pieces = functional_id.split("_")
        if len(pieces) >= 2 and pieces[0] == "part" and pieces[1].isdigit():
            return f"tool.parts.{pieces[0]}_{pieces[1]}"
    return None


def _is_symbol_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_REF_PREFIXES)


def _iter_symbol_refs(value: Any) -> Iterable[str]:
    if _is_symbol_ref(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_symbol_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_symbol_refs(child)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _only_known_refs(refs: Iterable[str], symbol_ref_set: set[str]) -> list[str]:
    return _dedupe(ref for ref in refs if ref in symbol_ref_set)


def _refs_at_path(root: dict[str, Any], path: tuple[str, ...]) -> list[str]:
    value: Any = root
    for key in path:
        if not isinstance(value, dict):
            return []
        value = value.get(key)
    return [ref for ref in _string_list(value) if _is_symbol_ref(ref)]


def _refs_from_section(
    symbol_table: dict[str, Any],
    section_name: str,
    *,
    include_collections: frozenset[str] | None = None,
    exclude_collections: frozenset[str] | None = None,
) -> list[str]:
    section = symbol_table.get(section_name, {})
    if not isinstance(section, dict):
        return []
    refs: list[str] = []
    for collection, values in section.items():
        collection_name = str(collection)
        if include_collections is not None and collection_name not in include_collections:
            continue
        if exclude_collections is not None and collection_name in exclude_collections:
            continue
        refs.extend(ref for ref in _string_list(values) if _is_symbol_ref(ref))
    return refs


def _filter_refs_by_suffix(refs: Iterable[str], suffixes: tuple[str, ...]) -> list[str]:
    return [
        ref
        for ref in refs
        if isinstance(ref, str) and any(ref.endswith(suffix) for suffix in suffixes)
    ]


def _tool_functional_refs(symbol_table: dict[str, Any]) -> list[str]:
    refs = _refs_at_path(symbol_table, ("tool_symbols", "functional_parts"))
    role_index = symbol_table.get("role_index", {})
    if isinstance(role_index, dict):
        functional_roles = role_index.get("tool_functional_part_attributes", {})
        refs.extend(_iter_symbol_refs(functional_roles))
    return _dedupe(refs)


def _target_contact_refs_from_symbol_table(symbol_table: dict[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_section(
                symbol_table,
                "target_symbols",
                exclude_collections=_AXIS_SYMBOL_COLLECTIONS,
            ),
            *_refs_from_section(
                symbol_table,
                "scene_symbols",
                include_collections=_TARGET_CONTACT_SYMBOL_COLLECTIONS,
            ),
            *_refs_from_section(
                symbol_table,
                "candidate_symbols",
                include_collections=_TARGET_CONTACT_SYMBOL_COLLECTIONS,
            ),
        ]
    )


def _axis_refs_from_symbol_table(symbol_table: dict[str, Any]) -> list[str]:
    return _dedupe(
        [
            *_refs_from_section(
                symbol_table,
                "target_symbols",
                include_collections=_AXIS_SYMBOL_COLLECTIONS,
            ),
            *_refs_from_section(
                symbol_table,
                "scene_symbols",
                include_collections=_AXIS_SYMBOL_COLLECTIONS,
            ),
            *_refs_from_section(
                symbol_table,
                "candidate_symbols",
                include_collections=_AXIS_SYMBOL_COLLECTIONS,
            ),
        ]
    )


def _numbered_anchor_rows(visual_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(visual_evidence, dict):
        return []
    numbered = visual_evidence.get("numbered_anchor_refs", {})
    rows = numbered.get("rows", []) if isinstance(numbered, dict) else []
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    metadata = visual_evidence.get("metadata", {})
    raw_rows = metadata.get("anchor_ref_table", []) if isinstance(metadata, dict) else []
    if isinstance(raw_rows, list):
        return [row for row in raw_rows if isinstance(row, dict)]
    return []


def _refs_from_numbered_anchor_rows(
    visual_evidence: dict[str, Any] | None,
    *,
    keys: tuple[str, ...] = _VISUAL_ROW_REF_KEYS,
) -> list[str]:
    refs: list[str] = []
    for row in _numbered_anchor_rows(visual_evidence):
        for key in keys:
            value = row.get(key)
            if _is_symbol_ref(value):
                refs.append(value)
    return _dedupe(refs)


def _tool_visual_refs_from_numbered_anchor_rows(
    visual_evidence: dict[str, Any] | None,
    *,
    keys: tuple[str, ...],
) -> list[str]:
    refs: list[str] = []
    for row in _numbered_anchor_rows(visual_evidence):
        part_ref = row.get("part_ref")
        if not (isinstance(part_ref, str) and part_ref.startswith("tool.parts.")):
            continue
        for key in keys:
            value = row.get(key)
            if _is_symbol_ref(value):
                refs.append(value)
    return _dedupe(refs)


def _tool_part_refs_from_numbered_anchor_rows(
    visual_evidence: dict[str, Any] | None,
) -> list[str]:
    refs: list[str] = []
    for row in _numbered_anchor_rows(visual_evidence):
        part_ref = row.get("part_ref")
        if isinstance(part_ref, str) and part_ref.startswith("tool.parts."):
            refs.append(part_ref)
    return _dedupe(refs)


def _collect_relational_executable_tool_refs(
    relational_keypoint_cards: dict[str, Any] | None,
    *,
    allowed_ref_set: set[str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "center_ref": [],
        "normal_ref": [],
        "axis_ref": [],
    }
    if not isinstance(relational_keypoint_cards, dict):
        return grouped
    tool_cards = relational_keypoint_cards.get("tool_cards", [])
    if not isinstance(tool_cards, list):
        return grouped
    for card in tool_cards:
        if not isinstance(card, dict):
            continue
        functional_refs = card.get("functional_refs", [])
        if isinstance(functional_refs, list):
            for functional_ref in functional_refs:
                if not isinstance(functional_ref, dict):
                    continue
                executable_refs = functional_ref.get("executable_tool_refs", {})
                if not isinstance(executable_refs, dict):
                    continue
                for key in grouped:
                    ref = executable_refs.get(key)
                    if _is_symbol_ref(ref) and ref in allowed_ref_set:
                        grouped[key].append(ref)
        candidate_refs = card.get("candidate_evidence_refs", {})
        if isinstance(candidate_refs, dict):
            for ref in _string_list(candidate_refs.get("points")):
                if _is_symbol_ref(ref) and ref in allowed_ref_set:
                    grouped["center_ref"].append(ref)
            for ref in _string_list(candidate_refs.get("surfaces")):
                if _is_symbol_ref(ref) and ref in allowed_ref_set:
                    grouped["center_ref"].append(ref)
                    grouped["normal_ref"].append(ref)
            for ref in _string_list(candidate_refs.get("axes")):
                if _is_symbol_ref(ref) and ref in allowed_ref_set:
                    grouped["axis_ref"].append(ref)
    return {key: _dedupe(refs) for key, refs in grouped.items()}


def _collect_relational_target_refs(
    relational_keypoint_cards: dict[str, Any] | None,
    *,
    allowed_ref_set: set[str],
) -> tuple[list[str], list[str]]:
    target_refs: list[str] = []
    axis_refs: list[str] = []
    if not isinstance(relational_keypoint_cards, dict):
        return target_refs, axis_refs

    target_cards = relational_keypoint_cards.get("target_cards", [])
    if isinstance(target_cards, list):
        for card in target_cards:
            if not isinstance(card, dict):
                continue
            for ref in _iter_symbol_refs(card.get("usable_target_refs", {})):
                if ref in allowed_ref_set:
                    target_refs.append(ref)
            target_ref = card.get("target_ref")
            if _is_symbol_ref(target_ref) and target_ref in allowed_ref_set:
                target_refs.append(target_ref)

    axis_cards = relational_keypoint_cards.get("target_axis_cards", [])
    if isinstance(axis_cards, list):
        for card in axis_cards:
            if not isinstance(card, dict):
                continue
            axis_ref = card.get("axis_ref")
            if _is_symbol_ref(axis_ref) and axis_ref in allowed_ref_set:
                axis_refs.append(axis_ref)

    return _dedupe(target_refs), _dedupe(axis_refs)


def _filter_relational_keypoint_cards_to_binding(
    relational_keypoint_cards: dict[str, Any] | None,
    *,
    allowed_ref_set: set[str],
) -> dict[str, Any] | None:
    if relational_keypoint_cards is None:
        return None
    filtered = _filter_ref_tree(relational_keypoint_cards, allowed_ref_set)
    return filtered if isinstance(filtered, dict) else None


def _filter_visual_evidence_to_binding(
    visual_evidence: dict[str, Any] | None,
    *,
    allowed_ref_set: set[str],
) -> dict[str, Any] | None:
    if not isinstance(visual_evidence, dict):
        return None

    compact: dict[str, Any] = {
        key: visual_evidence[key]
        for key in ("schema_version", "evidence_id", "usage_contract", "items")
        if key in visual_evidence
    }
    rows: list[dict[str, Any]] = []
    for row in _numbered_anchor_rows(visual_evidence):
        compact_row = {
            key: row[key]
            for key in (
                "tool_id",
                "number",
                "anchor_id",
                "part_id",
                "part_semantic_label",
                "part_semantic_aliases",
                "view_index",
                "feature_cluster_id",
                "score",
            )
            if key in row
        }
        row_ref_count = 0
        for key in _VISUAL_ROW_REF_KEYS:
            ref = row.get(key)
            if _is_symbol_ref(ref) and ref in allowed_ref_set:
                compact_row[key] = ref
                row_ref_count += 1
        if row_ref_count:
            rows.append(compact_row)
    if rows:
        compact["numbered_anchor_refs"] = {
            "usage": (
                "Marker numbers and visual scores identify evidence rows; "
                "they are not refs or affordance correctness labels."
            ),
            "rows": rows,
        }
        if isinstance(visual_evidence.get("numbered_anchor_refs"), dict):
            numbering_scope = visual_evidence["numbered_anchor_refs"].get(
                "numbering_scope"
            )
            if numbering_scope:
                compact["numbered_anchor_refs"]["numbering_scope"] = numbering_scope
    return compact


_MISSING = object()


def _filter_ref_tree(value: Any, allowed_ref_set: set[str]) -> Any:
    if _is_symbol_ref(value):
        return value if value in allowed_ref_set else _MISSING
    if isinstance(value, list):
        filtered_list = []
        for child in value:
            filtered = _filter_ref_tree(child, allowed_ref_set)
            if filtered is not _MISSING:
                filtered_list.append(filtered)
        return filtered_list
    if isinstance(value, dict):
        filtered_dict: dict[str, Any] = {}
        for key, child in value.items():
            filtered = _filter_ref_tree(child, allowed_ref_set)
            if filtered is not _MISSING:
                filtered_dict[key] = filtered
        return filtered_dict
    return value
