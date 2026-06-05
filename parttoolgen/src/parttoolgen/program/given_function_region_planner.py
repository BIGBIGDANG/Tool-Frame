#!/usr/bin/env python3
"""Legacy V0 planner for a VLM-authored Part-Centric Affordance Program.

The current image-grounded benchmark mainline does not use this function-region
planner as its public interface. Current benchmark tasks use public RGB-D scene
refs, RGB-D lifting, reviewed tool refs named ``tool.contact_point``,
``tool.motion_axis``, and ``tool.motion_axis``, and whole-tool SE(3)
compilation. This module is retained for legacy/debug workflows.

The VLM selects grounded tool/target functional refs and composes bounded
interaction and motion primitives into stages. PartToolGen validates those refs,
lowers the program to ProgramIR, and leaves metric geometry to the compiler.
There is no 5.15 fallback from a region-only choice to a fixed trajectory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.checks.affordance_binding_sanity import (  # noqa: E402
    check_program_ir_affordance_contract,
)
from parttoolgen.program.api import ProgramContext  # noqa: E402
from parttoolgen.program.prompting import (  # noqa: E402
    PROMPT_TEMPLATE_ID,
    build_program_symbol_table,
    flatten_allowed_refs,
)
from parttoolgen.program.sandbox import (  # noqa: E402
    execute_planner_program,
    make_planner_program_artifact,
)
from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    load_structured_file,
    validate_planner_program,
    validate_program_ir,
)


class GivenFunctionRegionPlannerError(ValueError):
    """Raised when a selected region cannot be lowered to executable geometry."""


def _json_dump(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _part_id_from_ref(ref: str | None) -> str | None:
    if not isinstance(ref, str):
        return None
    match = re.fullmatch(r"tool\.parts\.(part_\d{3})", ref)
    if match:
        return match.group(1)
    return None


def _load_selected_region(region_selection: Mapping[str, Any]) -> tuple[str, str]:
    selected_ref = region_selection.get("selected_part_ref")
    part_id = region_selection.get("selected_part_id") or _part_id_from_ref(selected_ref)
    if not isinstance(selected_ref, str) or not isinstance(part_id, str):
        raise GivenFunctionRegionPlannerError("region selection must contain selected_part_ref")
    parsed = _part_id_from_ref(selected_ref)
    if parsed is None:
        raise GivenFunctionRegionPlannerError(f"selected_part_ref is not a tool part ref: {selected_ref!r}")
    if part_id != parsed:
        raise GivenFunctionRegionPlannerError(
            f"selected_part_id {part_id!r} does not match selected_part_ref {selected_ref!r}"
        )
    return selected_ref, part_id


def _candidate_support_faces(candidate: Mapping[str, Any]) -> int:
    provenance = candidate.get("provenance")
    if not isinstance(provenance, Mapping):
        return 0
    value = provenance.get("support_faces")
    return int(value) if isinstance(value, (int, float)) else 0


def _candidate_role(candidate: Mapping[str, Any]) -> str:
    provenance = candidate.get("provenance")
    role = provenance.get("role") if isinstance(provenance, Mapping) else None
    return str(role or candidate.get("role") or "")


def _candidate_patch_id(candidate: Mapping[str, Any]) -> str:
    provenance = candidate.get("provenance")
    patch_id = provenance.get("patch_id") if isinstance(provenance, Mapping) else None
    return str(patch_id or candidate.get("label") or "")


_CONTACT_PATCH_POLICIES = {"vlm_selected_candidate"}
_COMPOSITIONAL_OP_TYPES = {
    "near",
    "contact",
    "maintain_contact",
    "align_axis",
    "pre_contact_offset",
    "move_along",
    "slide_along",
    "rotate_about",
    "clearance",
}
_INTERACTION_PRIMITIVES = {
    "point_contact",
    "surface_contact",
    "edge_contact",
    "impact_contact",
    "support_contact",
    "slot_contact",
}
_CONTACT_MODES = {
    "impact",
    "sustained_contact",
}
_CONTACT_MODE_INTERACTION_ALIASES = {
    "impact": "impact_contact",
    "impact_contact": "impact_contact",
    "sustained_contact": "surface_contact",
    "surface_contact": "surface_contact",
}
_MOTION_PRIMITIVES = {
    "translate",
    "rotate",
    "translate_along",
    "press_along",
    "slide_along",
    "rotate_about",
    "swing_to",
    "lift_along",
}
_FORBIDDEN_TEMPLATE_KEYS = {
    "tool_category",
    "tool_name",
    "task_category",
    "interaction_type",
    "stage_template",
    "template",
}
def _candidate_id_from_point_ref(ref: str) -> str | None:
    prefix = "candidate.points."
    return ref[len(prefix) :] if isinstance(ref, str) and ref.startswith(prefix) else None


def _candidate_id_from_surface_ref(ref: str) -> str | None:
    prefix = "candidate.surfaces."
    return ref[len(prefix) :] if isinstance(ref, str) and ref.startswith(prefix) else None


def _candidate_id_from_functional_axis_ref(ref: str) -> str | None:
    prefix = "tool.functional_parts."
    if not isinstance(ref, str) or not ref.startswith(prefix):
        return None
    body = ref[len(prefix) :]
    for suffix in (".axis", ".normal", ".tangent_long", ".tangent_short"):
        if body.endswith(suffix):
            return body[: -len(suffix)]
    return None


def _axis_candidate_id(ref: str) -> str | None:
    return _candidate_id_from_surface_ref(ref) or _candidate_id_from_functional_axis_ref(ref)


def _candidate_surface_ref(candidate_id: str, grounding_candidates: Mapping[str, Any]) -> str | None:
    surfaces = grounding_candidates.get("surfaces")
    if isinstance(surfaces, Mapping) and candidate_id in surfaces:
        return f"candidate.surfaces.{candidate_id}"
    return None


def _selected_region_candidate_rows(
    *,
    selected_ref: str,
    grounding_candidates: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    points = grounding_candidates.get("points")
    if not isinstance(points, Mapping):
        raise GivenFunctionRegionPlannerError("grounding candidates have no points")
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for candidate_id, candidate in sorted(points.items()):
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("part_ref") != selected_ref:
            continue
        if candidate.get("candidate_type") != "geometry_surface_center":
            continue
        if "xyz_tool" not in candidate:
            continue
        candidates.append((str(candidate_id), candidate))
    return candidates


def derive_region_contact_ref(
    *,
    region_selection: Mapping[str, Any],
    grounding_candidates: Mapping[str, Any],
    selected_contact_ref: str | None = None,
    contact_patch_policy: str = "vlm_selected_candidate",
) -> dict[str, Any]:
    """Validate the VLM-selected executable contact ref for the selected part."""

    if contact_patch_policy not in _CONTACT_PATCH_POLICIES:
        raise GivenFunctionRegionPlannerError(f"unsupported contact_patch_policy: {contact_patch_policy}")
    selected_ref, part_id = _load_selected_region(region_selection)
    candidates = _selected_region_candidate_rows(
        selected_ref=selected_ref,
        grounding_candidates=grounding_candidates,
    )
    if not candidates:
        raise GivenFunctionRegionPlannerError(
            f"selected region {selected_ref} has no geometry_surface_center candidate points"
        )
    if not selected_contact_ref:
        raise GivenFunctionRegionPlannerError(
            "5.15 mainline requires selected_function_contact_ref; "
            "region-only geometric contact fallback is disabled"
        )
    selected_candidate_id = _candidate_id_from_point_ref(selected_contact_ref)
    candidate_by_id = {candidate_id: candidate for candidate_id, candidate in candidates}
    if selected_candidate_id not in candidate_by_id:
        raise GivenFunctionRegionPlannerError(
            f"selected_function_contact_ref must be a candidate point inside {selected_ref}; "
            f"got {selected_contact_ref!r}"
        )
    candidate_id = str(selected_candidate_id)
    candidate = candidate_by_id[candidate_id]
    selection_method = "vlm_selected_candidate_ref"
    point_ref = f"candidate.points.{candidate_id}"
    surface_ref = _candidate_surface_ref(candidate_id, grounding_candidates)
    return {
        "selected_function_region_ref": selected_ref,
        "selected_function_region_id": part_id,
        "derived_contact_ref": point_ref,
        "derived_surface_ref": surface_ref,
        "derivation": {
            "method": selection_method,
            "candidate_id": candidate_id,
            "candidate_type": candidate.get("candidate_type"),
            "contact_patch_policy": contact_patch_policy,
            "role": _candidate_role(candidate),
            "patch_id": _candidate_patch_id(candidate),
            "support_faces": _candidate_support_faces(candidate),
            "confidence": candidate.get("confidence"),
            "contract": (
                "The executable contact point is an exact provider candidate ref "
                "inside the selected SAMPart3D region and must be selected by the "
                "VLM-authored part-centric affordance program."
            ),
        },
    }


def _required_ref(preferred: str | None, available: set[str], name: str) -> str:
    if preferred and preferred in available:
        return preferred
    raise GivenFunctionRegionPlannerError(
        f"{name} is required and must be an exact provider ref; got {preferred!r}"
    )


def _optional_ref(preferred: str | None, available: set[str], name: str) -> str | None:
    if not preferred:
        return None
    if preferred in available:
        return preferred
    raise GivenFunctionRegionPlannerError(
        f"{name} must be null or an exact provider ref; got {preferred!r}"
    )


def _available_refs(
    *,
    observed_scene_graph: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> set[str]:
    refs: set[str] = set()
    for collection in ("objects", "keypoints", "axes", "regions", "surfaces"):
        values = observed_scene_graph.get(collection, {})
        if isinstance(values, Mapping):
            refs.update(f"scene.{collection}.{key}" for key in values)
    if isinstance(target_part_graph, Mapping):
        for collection in ("parts", "axes", "regions"):
            values = target_part_graph.get(collection, {})
            if isinstance(values, Mapping):
                refs.update(f"target.{collection}.{key}" for key in values)
    return refs


def _selected_region_tool_refs(
    *,
    selected_ref: str,
    grounding_candidates: Mapping[str, Any],
) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {
        "all": {selected_ref},
        "points": set(),
        "axes": set(),
        "surfaces": set(),
        "parts": {selected_ref},
    }
    points = grounding_candidates.get("points")
    if isinstance(points, Mapping):
        for candidate_id, candidate in sorted(points.items()):
            if not isinstance(candidate, Mapping) or candidate.get("part_ref") != selected_ref:
                continue
            if "xyz_tool" in candidate:
                ref = f"candidate.points.{candidate_id}"
                refs["points"].add(ref)
                refs["all"].add(ref)
                for suffix in ("axis", "normal", "tangent_long", "tangent_short"):
                    axis_ref = f"tool.functional_parts.{candidate_id}.{suffix}"
                    refs["axes"].add(axis_ref)
                    refs["all"].add(axis_ref)
    surfaces = grounding_candidates.get("surfaces")
    if isinstance(surfaces, Mapping):
        for surface_id, surface in sorted(surfaces.items()):
            if not isinstance(surface, Mapping) or surface.get("part_ref") != selected_ref:
                continue
            if "normal_tool" in surface:
                ref = f"candidate.surfaces.{surface_id}"
                refs["surfaces"].add(ref)
                refs["axes"].add(ref)
                refs["all"].add(ref)
    return refs


def _selected_region_function_axis_refs(
    *,
    selected_ref: str,
    grounding_candidates: Mapping[str, Any],
) -> set[str]:
    """Refs that may serve as the selected functional contact axis.

    The surface ref exposes the local surface normal.  The functional-part axis
    exposes the provider's local tangent/main motion hint for the same contact
    anchor.  Both remain exact declared refs tied to a selected candidate point.
    """

    refs: set[str] = set()
    for candidate_id, _candidate in _selected_region_candidate_rows(
        selected_ref=selected_ref,
        grounding_candidates=grounding_candidates,
    ):
        refs.add(f"tool.functional_parts.{candidate_id}.axis")
        refs.add(f"tool.functional_parts.{candidate_id}.normal")
        refs.add(f"tool.functional_parts.{candidate_id}.tangent_long")
        refs.add(f"tool.functional_parts.{candidate_id}.tangent_short")
    surfaces = grounding_candidates.get("surfaces")
    if isinstance(surfaces, Mapping):
        for surface_id, surface in sorted(surfaces.items()):
            if not isinstance(surface, Mapping) or surface.get("part_ref") != selected_ref:
                continue
            if "normal_tool" in surface:
                refs.add(f"candidate.surfaces.{surface_id}")
    return refs


def _scene_object_ref_for_keypoint(
    *,
    observed_scene_graph: Mapping[str, Any],
    keypoint_ref: str,
) -> str | None:
    prefix = "scene.keypoints."
    if not keypoint_ref.startswith(prefix):
        return None
    keypoint_id = keypoint_ref[len(prefix) :]
    keypoint = observed_scene_graph.get("keypoints", {}).get(keypoint_id)
    if not isinstance(keypoint, Mapping):
        return None
    object_ref = keypoint.get("object_ref")
    if not isinstance(object_ref, str) or not object_ref:
        return None
    return object_ref if object_ref.startswith("scene.objects.") else f"scene.objects.{object_ref}"


def _choice(value: Any, *, choices: set[str], default: str, name: str) -> str:
    if value is None or value == "":
        return default
    text = str(value)
    if text not in choices:
        raise GivenFunctionRegionPlannerError(f"{name} must be one of {sorted(choices)}; got {text!r}")
    return text


def _bounded_float(
    value: Any,
    *,
    default: float,
    minimum: float,
    maximum: float,
    name: str,
) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GivenFunctionRegionPlannerError(f"{name} must be numeric")
    number = float(value)
    if number < minimum or number > maximum:
        raise GivenFunctionRegionPlannerError(f"{name} must be in [{minimum}, {maximum}]")
    return number


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int, name: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise GivenFunctionRegionPlannerError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise GivenFunctionRegionPlannerError(f"{name} must be in [{minimum}, {maximum}]")
    return int(value)


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _resolve_alias_ref(value: Any, aliases: Mapping[str, str], *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GivenFunctionRegionPlannerError(f"{name} must be a non-empty ref string")
    return aliases.get(value, value)


def _checked_ref(
    value: Any,
    *,
    aliases: Mapping[str, str],
    allowed: set[str],
    name: str,
) -> str:
    ref = _resolve_alias_ref(value, aliases, name=name)
    if ref not in allowed:
        raise GivenFunctionRegionPlannerError(f"{name} must be one of the declared refs; got {ref!r}")
    return ref


def _normalize_op_number(
    op: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    return _bounded_float(op.get(key), default=default, minimum=minimum, maximum=maximum, name=key)


def _normalize_op_int(
    op: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    return _bounded_int(op.get(key), default=default, minimum=minimum, maximum=maximum, name=key)


def _as_meta_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, Mapping) for item in value):
            return list(value)
    raise GivenFunctionRegionPlannerError("interaction/motion entries must be objects or lists of objects")


def _as_contact_mode_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": value}]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        entries: list[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                entries.append({"type": item})
            elif isinstance(item, Mapping):
                entries.append(item)
            else:
                raise GivenFunctionRegionPlannerError(
                    "contact_mode entries must be strings, objects, or lists of those"
                )
        return entries
    raise GivenFunctionRegionPlannerError("contact_mode entries must be strings, objects, or lists of those")


def _normalize_contact_mode_name(value: Any) -> str:
    mode = str(value or "").strip()
    if mode in _CONTACT_MODES:
        return mode
    if mode == "impact_contact":
        return "impact"
    if mode == "surface_contact":
        return "sustained_contact"
    raise GivenFunctionRegionPlannerError(
        f"contact_mode must be one of {sorted(_CONTACT_MODES)}"
    )


def _contact_mode_to_interaction(contact_mode: Mapping[str, Any]) -> dict[str, Any]:
    raw_mode = _first_value(contact_mode, "type", "mode", "contact_mode", "primitive")
    mode = _normalize_contact_mode_name(raw_mode)
    interaction = dict(contact_mode)
    interaction["type"] = _CONTACT_MODE_INTERACTION_ALIASES[mode]
    interaction["contact_mode"] = mode
    return interaction


def _contact_mode_from_interaction(interaction: Mapping[str, Any]) -> str | None:
    interaction_type = str(interaction.get("type") or interaction.get("primitive") or "").strip()
    if interaction_type in _CONTACT_MODE_INTERACTION_ALIASES:
        return _normalize_contact_mode_name(interaction_type)
    return None


def _interaction_to_ops(interaction: Mapping[str, Any]) -> list[dict[str, Any]]:
    interaction_type = str(interaction.get("type") or interaction.get("primitive") or "").strip()
    if interaction_type not in _INTERACTION_PRIMITIVES:
        raise GivenFunctionRegionPlannerError(
            f"interaction primitive must be one of {sorted(_INTERACTION_PRIMITIVES)}"
        )
    default_tol = 0.012 if interaction_type == "impact_contact" else 0.008
    tool_ref = _first_value(interaction, "tool_ref", "source_ref", "source") or "selected_function_contact_ref"
    target_ref = _first_value(interaction, "target_ref", "target") or "selected_target_contact_ref"
    ops: list[dict[str, Any]] = []
    ops.append(
        {
            "type": "align_axis",
            "tool_axis_ref": _first_value(interaction, "tool_axis_ref", "source_axis_ref", "tool_axis")
            or "selected_function_axis_ref",
            "target_axis_ref": _first_value(interaction, "target_axis_ref", "target_axis", "axis_ref", "axis")
            or "selected_target_axis_ref",
            "polarity": interaction.get("polarity", "same"),
            "min_cosine": interaction.get("min_cosine", 0.9),
            "interaction_primitive": interaction_type,
        }
    )
    ops.append({
        "type": "contact",
        "tool_ref": tool_ref,
        "target_ref": target_ref,
        "tolerance_m": interaction.get("tolerance_m", default_tol),
        "interaction_primitive": interaction_type,
    })
    return ops


def _motion_to_ops(motion: Mapping[str, Any]) -> list[dict[str, Any]]:
    motion_type = str(motion.get("type") or motion.get("primitive") or "").strip()
    if motion_type not in _MOTION_PRIMITIVES:
        raise GivenFunctionRegionPlannerError(
            f"motion primitive must be one of {sorted(_MOTION_PRIMITIVES)}"
        )
    tool_ref = _first_value(motion, "tool_ref", "source_ref", "source") or "selected_function_contact_ref"
    start_ref = (
        _first_value(motion, "start_ref", "start")
        or "target_pre_contact"
    )
    goal_ref = (
        _first_value(motion, "goal_ref", "goal", "target_ref", "target")
        or "selected_target_contact_ref"
    )
    if motion_type in {"rotate", "rotate_about", "swing_to"}:
        return [{
            "type": "rotate_about",
            "tool_ref": tool_ref,
            "start_ref": start_ref,
            "goal_ref": goal_ref,
            "axis_ref": _first_value(motion, "axis_ref", "rotation_axis_ref", "swing_axis_ref", "axis")
            or "selected_target_axis_ref",
            "path_family": "swing",
            "angle_rad": motion.get("angle_rad", motion.get("swing_angle_rad", 0.55)),
            "swing_lift_m": motion.get("swing_lift_m", 0.03),
            "motion_primitive": motion_type,
        }]
    if motion_type == "press_along":
        return [
            {
                "type": "pre_contact_offset",
                "tool_ref": tool_ref,
                "target_ref": goal_ref,
                "axis_ref": _first_value(motion, "axis_ref", "motion_axis_ref", "axis")
                or "selected_target_axis_ref",
                "distance_m": motion.get("distance_m", 0.08),
                "direction": motion.get("direction", "opposite_axis"),
                "motion_primitive": motion_type,
            },
            {
                "type": "contact",
                "tool_ref": tool_ref,
                "target_ref": goal_ref,
                "tolerance_m": motion.get("tolerance_m", 0.008),
                "motion_primitive": motion_type,
            },
        ]
    return [{
        "type": "slide_along" if motion_type == "slide_along" else "move_along",
        "tool_ref": tool_ref,
        "start_ref": start_ref,
        "goal_ref": goal_ref,
        "profile": "linear",
        "path_family": (
            "contact_follow"
            if motion_type == "translate"
            else motion.get(
                "path_family",
                "slide" if motion_type == "slide_along" else "contact_follow",
            )
        ),
        "motion_primitive": motion_type,
    }]


def _primitive_has_operational_fields(primitive: Mapping[str, Any]) -> bool:
    summary_keys = {"type", "primitive", "name", "description"}
    return any(key not in summary_keys for key in primitive)


def _motion_has_executable_op(motion_type: str, ops: list[Any]) -> bool:
    """Return whether a declared motion primitive has a real geometric op."""

    for op in ops:
        if not isinstance(op, Mapping):
            continue
        op_type = str(op.get("type") or op.get("op") or "").strip()
        if motion_type in {"rotate", "rotate_about", "swing_to"}:
            if op_type == "rotate_about":
                return True
            if op_type == "move_along" and op.get("profile") == "swing_arc":
                return True
        elif motion_type == "press_along":
            if op_type in {"pre_contact_offset", "move_along"}:
                return True
        elif motion_type == "slide_along":
            if op_type in {"slide_along", "move_along"}:
                return True
        elif motion_type in {"translate", "translate_along", "lift_along"}:
            if op_type == "move_along":
                return True
    return False


def _canonical_motion_primitive(motion_type: str) -> str:
    if motion_type in {"rotate", "rotate_about", "swing_to"}:
        return "rotate"
    if motion_type in {"translate", "translate_along", "press_along", "slide_along", "lift_along"}:
        return "translate"
    return motion_type


def _normalized_motion_summaries_with_executable_ops(
    *,
    stage_name: str,
    motions: list[Mapping[str, Any]],
    ops: list[Any],
) -> list[dict[str, str]]:
    """Keep only motion summaries that correspond to executable geometry.

    VLM repair calls sometimes preserve a previous motion label on a stage whose
    explicit ops are only approach/clearance constraints. A summary-only motion
    label is not an executable primitive by itself, so it should not make the
    otherwise valid stage fail. Parameterized motion entries are still expected
    to lower into executable ops before this check.
    """

    normalized: list[dict[str, str]] = []
    for motion in motions:
        motion_type = str(motion.get("type") or motion.get("primitive") or "").strip()
        if motion_type not in _MOTION_PRIMITIVES:
            raise GivenFunctionRegionPlannerError(
                f"motion primitive must be one of {sorted(_MOTION_PRIMITIVES)}"
            )
        if _motion_has_executable_op(motion_type, ops):
            normalized.append({"type": _canonical_motion_primitive(motion_type)})
            continue
        if _primitive_has_operational_fields(motion):
            raise GivenFunctionRegionPlannerError(
                f"stage {stage_name!r} declares parameterized motion primitive {motion_type!r} "
                "but it did not lower to an executable geometric motion op"
            )
    return normalized


def normalize_compositional_affordance_program(
    *,
    part_centric_affordance_program: Mapping[str, Any],
    region_selection: Mapping[str, Any],
    grounding_candidates: Mapping[str, Any],
    observed_scene_graph: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate VLM-composed geometric relations over declared refs.

    This is the mainline generalization of the older metadata-style task
    affordance program.  The VLM chooses region-local tool refs, target refs,
    and a sequence of bounded primitive ops.  The VLM may compose relations,
    but it may not invent refs, coordinates, simulator actions, or free Python.
    """

    stages = part_centric_affordance_program.get("stages")
    if not isinstance(stages, list) or not stages:
        raise GivenFunctionRegionPlannerError("part-centric affordance program requires non-empty stages")
    if len(stages) > 8:
        raise GivenFunctionRegionPlannerError("part-centric affordance program supports at most 8 stages")
    input_already_normalized = isinstance(
        part_centric_affordance_program.get("raw_part_centric_affordance_program"),
        Mapping,
    )
    forbidden_keys = sorted(
        key for key in _FORBIDDEN_TEMPLATE_KEYS if key in part_centric_affordance_program
    )
    if forbidden_keys:
        raise GivenFunctionRegionPlannerError(
            "part-centric affordance program must use grounded refs and primitives, "
            f"not category/template metadata: {forbidden_keys}"
        )

    selected_ref, _ = _load_selected_region(region_selection)
    vlm_region_ref = part_centric_affordance_program.get("selected_tool_region_ref") or part_centric_affordance_program.get(
        "selected_part_ref"
    )
    if isinstance(vlm_region_ref, str) and vlm_region_ref and vlm_region_ref != selected_ref:
        raise GivenFunctionRegionPlannerError(
            "part-centric affordance program selected_tool_region_ref does not match region_selection: "
            f"{vlm_region_ref!r} != {selected_ref!r}"
        )

    tool_refs = _selected_region_tool_refs(
        selected_ref=selected_ref,
        grounding_candidates=grounding_candidates,
    )
    function_axis_refs = _selected_region_function_axis_refs(
        selected_ref=selected_ref,
        grounding_candidates=grounding_candidates,
    )
    scene_refs = _available_refs(
        observed_scene_graph=observed_scene_graph,
        target_part_graph=target_part_graph,
    )
    target_point_refs = {
        ref
        for ref in scene_refs
        if ref.startswith("scene.keypoints.")
        or ref.startswith("scene.regions.")
        or ref.startswith("scene.surfaces.")
        or ref.startswith("target.parts.")
        or ref.startswith("target.regions.")
    }
    target_axis_refs = {
        ref
        for ref in scene_refs
        if ref.startswith("scene.axes.")
        or ref.startswith("scene.surfaces.")
        or ref.startswith("scene.regions.")
        or ref.startswith("target.axes.")
        or ref.startswith("target.parts.")
        or ref.startswith("target.regions.")
    }
    motion_axis_refs = set(target_axis_refs) | set(function_axis_refs)
    any_refs = set(tool_refs["all"]) | scene_refs

    selected_contact_ref = _checked_ref(
        part_centric_affordance_program.get("selected_function_contact_ref")
        or part_centric_affordance_program.get("selected_tool_contact_ref")
        or part_centric_affordance_program.get("function_contact_ref"),
        aliases={},
        allowed=tool_refs["points"],
        name="selected_function_contact_ref",
    )
    selected_axis_ref = _checked_ref(
        part_centric_affordance_program.get("selected_function_axis_ref")
        or part_centric_affordance_program.get("selected_tool_axis_ref")
        or part_centric_affordance_program.get("function_axis_ref"),
        aliases={},
        allowed=function_axis_refs,
        name="selected_function_axis_ref",
    )
    contact_candidate_id = _candidate_id_from_point_ref(selected_contact_ref)
    axis_candidate_id = _axis_candidate_id(selected_axis_ref)
    if contact_candidate_id is None or axis_candidate_id != contact_candidate_id:
        raise GivenFunctionRegionPlannerError(
            "selected_function_axis_ref must be a same-contact candidate.surfaces.* normal "
            "or tool.functional_parts.* normal/tangent axis matching selected_function_contact_ref"
        )
    selected_target_contact_ref = _checked_ref(
        part_centric_affordance_program.get("selected_target_contact_ref")
        or part_centric_affordance_program.get("target_contact_ref"),
        aliases={},
        allowed=target_point_refs,
        name="selected_target_contact_ref",
    )
    selected_target_axis_ref = _checked_ref(
        part_centric_affordance_program.get("selected_target_axis_ref")
        or part_centric_affordance_program.get("target_axis_ref")
        or part_centric_affordance_program.get("motion_axis_ref"),
        aliases={},
        allowed=target_axis_refs,
        name="selected_target_axis_ref",
    )
    target_pre_contact_ref = _optional_ref(
        part_centric_affordance_program.get("target_pre_contact_ref"),
        target_point_refs,
        "target_pre_contact_ref",
    )
    target_post_contact_ref = _optional_ref(
        part_centric_affordance_program.get("target_post_contact_ref"),
        target_point_refs,
        "target_post_contact_ref",
    )
    aliases = {
        "tool_contact": selected_contact_ref,
        "function_contact": selected_contact_ref,
        "selected_tool_contact_ref": selected_contact_ref,
        "selected_function_contact_ref": selected_contact_ref,
        "tool_axis": selected_axis_ref,
        "function_axis": selected_axis_ref,
        "selected_tool_axis_ref": selected_axis_ref,
        "selected_function_axis_ref": selected_axis_ref,
        "target_contact": selected_target_contact_ref,
        "selected_target_contact_ref": selected_target_contact_ref,
        "target_axis": selected_target_axis_ref,
        "motion_axis": selected_target_axis_ref,
        "selected_target_axis_ref": selected_target_axis_ref,
    }
    if target_pre_contact_ref:
        aliases["target_pre_contact"] = target_pre_contact_ref
        aliases["target_pre_contact_ref"] = target_pre_contact_ref
        aliases["pre_contact"] = target_pre_contact_ref
    if target_post_contact_ref:
        aliases["target_post_contact"] = target_post_contact_ref
        aliases["target_post_contact_ref"] = target_post_contact_ref
        aliases["post_contact"] = target_post_contact_ref
    aliases["target_contact_ref"] = selected_target_contact_ref

    normalized_stages: list[dict[str, Any]] = []
    uses_meta_actions = False
    for stage_index, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            raise GivenFunctionRegionPlannerError("each compositional stage must be an object")
        forbidden_stage_keys = sorted(key for key in _FORBIDDEN_TEMPLATE_KEYS if key in stage)
        if forbidden_stage_keys:
            raise GivenFunctionRegionPlannerError(
                f"stage {stage_index} must use interaction/motion primitives or ops, "
                f"not template metadata: {forbidden_stage_keys}"
            )
        stage_name = str(stage.get("name") or f"stage_{stage_index}").strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]{0,39}", stage_name):
            raise GivenFunctionRegionPlannerError(f"invalid compositional stage name: {stage_name!r}")
        contact_modes = _as_contact_mode_list(stage.get("contact_mode"))
        contact_mode_interactions = [
            _contact_mode_to_interaction(contact_mode)
            for contact_mode in contact_modes
        ]
        explicit_interactions = _as_meta_list(stage.get("interaction"))
        interactions = explicit_interactions + contact_mode_interactions
        normalized_interactions = [
            {"type": str(item.get("type") or item.get("primitive"))}
            for item in explicit_interactions
        ]
        normalized_contact_modes = [
            _normalize_contact_mode_name(item.get("contact_mode") or item.get("type"))
            for item in contact_mode_interactions
        ]
        for interaction in explicit_interactions:
            mode = _contact_mode_from_interaction(interaction)
            if mode is not None and mode not in normalized_contact_modes:
                normalized_contact_modes.append(mode)
        meta_ops: list[dict[str, Any]] = []
        motions = _as_meta_list(stage.get("motion"))
        if interactions or motions:
            uses_meta_actions = True
        raw_ops = stage.get("ops") or stage.get("constraints") or []
        if not isinstance(raw_ops, list):
            raise GivenFunctionRegionPlannerError(f"stage {stage_name!r} ops/constraints must be a list")
        if not input_already_normalized:
            for interaction in interactions:
                if not raw_ops or _primitive_has_operational_fields(interaction):
                    meta_ops.extend(_interaction_to_ops(interaction))
        meta_ops.extend(raw_ops)
        if not input_already_normalized:
            for motion in motions:
                if not raw_ops or _primitive_has_operational_fields(motion):
                    meta_ops.extend(_motion_to_ops(motion))
        ops = meta_ops
        normalized_motions = (
            _normalized_motion_summaries_with_executable_ops(
                stage_name=stage_name,
                motions=motions,
                ops=ops,
            )
            if motions
            else []
        )
        if not isinstance(ops, list) or not ops:
            raise GivenFunctionRegionPlannerError(f"stage {stage_name!r} requires non-empty ops")
        if len(ops) > 16:
            raise GivenFunctionRegionPlannerError(f"stage {stage_name!r} has too many ops")
        normalized_ops: list[dict[str, Any]] = []
        for op_index, op in enumerate(ops, start=1):
            if not isinstance(op, Mapping):
                raise GivenFunctionRegionPlannerError(f"stage {stage_name!r} op {op_index} must be an object")
            op_type = str(op.get("type") or op.get("op") or "").strip()
            if op_type not in _COMPOSITIONAL_OP_TYPES:
                raise GivenFunctionRegionPlannerError(
                    f"stage {stage_name!r} op {op_index} type must be one of {sorted(_COMPOSITIONAL_OP_TYPES)}"
                )
            normalized: dict[str, Any] = {"type": op_type}
            if op_type in {"near", "contact", "maintain_contact"}:
                normalized["tool_ref"] = _checked_ref(
                    _first_value(op, "tool_ref", "source_ref", "source"),
                    aliases=aliases,
                    allowed=tool_refs["points"],
                    name=f"{op_type}.tool_ref",
                )
                normalized["target_ref"] = _checked_ref(
                    _first_value(op, "target_ref", "target"),
                    aliases=aliases,
                    allowed=target_point_refs,
                    name=f"{op_type}.target_ref",
                )
                default_tol = 0.03 if op_type == "near" else 0.008
                normalized["tolerance_m"] = _normalize_op_number(
                    op,
                    "tolerance_m",
                    default=default_tol,
                    minimum=0.0,
                    maximum=0.25,
                )
            elif op_type == "align_axis":
                normalized["tool_axis_ref"] = _checked_ref(
                    _first_value(op, "tool_axis_ref", "source_axis_ref", "source_axis", "tool_axis"),
                    aliases=aliases,
                    allowed=tool_refs["axes"],
                    name="align_axis.tool_axis_ref",
                )
                normalized["target_axis_ref"] = _checked_ref(
                    _first_value(op, "target_axis_ref", "target_axis", "axis_ref", "axis"),
                    aliases=aliases,
                    allowed=target_axis_refs,
                    name="align_axis.target_axis_ref",
                )
                normalized["polarity"] = _choice(
                    op.get("polarity"),
                    choices={"same", "opposite", "auto"},
                    default="same",
                    name="align_axis.polarity",
                )
                normalized["min_cosine"] = _normalize_op_number(
                    op,
                    "min_cosine",
                    default=0.9,
                    minimum=-1.0,
                    maximum=1.0,
                )
            elif op_type == "pre_contact_offset":
                normalized["tool_ref"] = _checked_ref(
                    _first_value(op, "tool_ref", "source_ref", "source"),
                    aliases=aliases,
                    allowed=tool_refs["points"],
                    name="pre_contact_offset.tool_ref",
                )
                normalized["target_ref"] = _checked_ref(
                    _first_value(op, "target_ref", "target"),
                    aliases=aliases,
                    allowed=target_point_refs,
                    name="pre_contact_offset.target_ref",
                )
                normalized["axis_ref"] = _checked_ref(
                    _first_value(op, "axis_ref", "axis"),
                    aliases=aliases,
                    allowed=target_axis_refs,
                    name="pre_contact_offset.axis_ref",
                )
                normalized["distance_m"] = _normalize_op_number(
                    op,
                    "distance_m",
                    default=0.08,
                    minimum=0.0,
                    maximum=0.5,
                )
                normalized["direction"] = _choice(
                    op.get("direction"),
                    choices={"along_axis", "opposite_axis"},
                    default="opposite_axis",
                    name="pre_contact_offset.direction",
                )
            elif op_type in {"move_along", "slide_along", "rotate_about"}:
                normalized["tool_ref"] = _checked_ref(
                    _first_value(op, "tool_ref", "source_ref", "source"),
                    aliases=aliases,
                    allowed=tool_refs["points"],
                    name=f"{op_type}.tool_ref",
                )
                normalized["start_ref"] = _checked_ref(
                    _first_value(op, "start_ref", "start"),
                    aliases=aliases,
                    allowed=target_point_refs,
                    name=f"{op_type}.start_ref",
                )
                normalized["goal_ref"] = _checked_ref(
                    _first_value(op, "goal_ref", "goal", "target_ref", "target"),
                    aliases=aliases,
                    allowed=target_point_refs,
                    name=f"{op_type}.goal_ref",
                )
                if op_type == "rotate_about":
                    profile = "swing_arc"
                else:
                    profile = _choice(
                        op.get("profile"),
                        choices={"linear", "swing_arc"},
                        default="linear",
                        name=f"{op_type}.profile",
                    )
                normalized["profile"] = profile
                normalized["path_family"] = _choice(
                    op.get("path_family"),
                    choices={"contact_follow", "slide", "swing"},
                    default=(
                        "swing"
                        if profile == "swing_arc" or op_type == "rotate_about"
                        else "slide"
                        if op_type == "slide_along"
                        else "contact_follow"
                    ),
                    name=f"{op_type}.path_family",
                )
                swing_axis_value = _first_value(op, "swing_axis_ref", "rotation_axis_ref", "axis_ref", "axis")
                if profile == "swing_arc":
                    normalized["swing_axis_ref"] = _checked_ref(
                        swing_axis_value,
                        aliases=aliases,
                        allowed=motion_axis_refs,
                        name=f"{op_type}.swing_axis_ref",
                    )
                    normalized["swing_angle_rad"] = _normalize_op_number(
                        op,
                        "swing_angle_rad" if "swing_angle_rad" in op else "angle_rad",
                        default=0.55,
                        minimum=0.01,
                        maximum=1.57,
                    )
                    normalized["swing_lift_m"] = _normalize_op_number(
                        op,
                        "swing_lift_m",
                        default=0.03,
                        minimum=0.0,
                        maximum=0.25,
                    )
            elif op_type == "clearance":
                normalized["moving_ref"] = _checked_ref(
                    _first_value(op, "moving_ref", "moving", "tool_ref", "source_ref", "source")
                    or "selected_function_contact_ref",
                    aliases=aliases,
                    allowed=any_refs,
                    name="clearance.moving_ref",
                )
                normalized["obstacle_ref"] = _checked_ref(
                    _first_value(
                        op,
                        "obstacle_ref",
                        "obstacle",
                        "object_ref",
                        "target_ref",
                        "target",
                    ),
                    aliases=aliases,
                    allowed=any_refs,
                    name="clearance.obstacle_ref",
                )
                min_distance_value = _first_value(op, "min_distance_m", "min_clearance_m")
                normalized["min_distance_m"] = _normalize_op_number(
                    {**op, "min_distance_m": min_distance_value},
                    "min_distance_m",
                    default=0.006,
                    minimum=0.0,
                    maximum=0.25,
                )
            normalized_ops.append(normalized)
        normalized_stage = {
            "name": stage_name,
            "description": stage_name,
            "repeat": _normalize_op_int(stage, "repeat", default=1, minimum=1, maximum=8),
            "interaction": normalized_interactions,
            "motion": normalized_motions,
            "ops": normalized_ops,
        }
        if normalized_contact_modes:
            normalized_stage["contact_mode"] = normalized_contact_modes
        normalized_stages.append(normalized_stage)

    return {
        "schema_version": "v0.3-meta-action" if uses_meta_actions else "v0.2-compositional",
        "selected_tool_region_ref": selected_ref,
        "selected_function_contact_ref": selected_contact_ref,
        "selected_function_axis_ref": selected_axis_ref,
        "target_pre_contact_ref": target_pre_contact_ref,
        "target_contact_ref": selected_target_contact_ref,
        "target_post_contact_ref": target_post_contact_ref,
        "motion_axis_ref": selected_target_axis_ref,
        "stages": normalized_stages,
        "reason": str(part_centric_affordance_program.get("reason", "")).strip(),
        "raw_part_centric_affordance_program": dict(part_centric_affordance_program),
    }


def build_region_affordance_binding(
    *,
    region_selection: Mapping[str, Any],
    grounding_candidates: Mapping[str, Any],
    observed_scene_graph: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None = None,
    target_pre_contact_ref: str | None = None,
    target_contact_ref: str | None = None,
    target_post_contact_ref: str | None = None,
    motion_axis_ref: str | None = None,
    selected_contact_ref: str | None = None,
    selected_axis_ref: str | None = None,
    contact_patch_policy: str = "vlm_selected_candidate",
) -> dict[str, Any]:
    derived = derive_region_contact_ref(
        region_selection=region_selection,
        grounding_candidates=grounding_candidates,
        selected_contact_ref=selected_contact_ref,
        contact_patch_policy=contact_patch_policy,
    )
    refs = _available_refs(
        observed_scene_graph=observed_scene_graph,
        target_part_graph=target_part_graph,
    )
    target_contact_ref = _required_ref(
        target_contact_ref,
        refs,
        "target_contact_ref",
    )
    target_pre_contact_ref = _optional_ref(
        target_pre_contact_ref,
        refs,
        "target_pre_contact_ref",
    )
    target_post_contact_ref = _optional_ref(
        target_post_contact_ref,
        refs,
        "target_post_contact_ref",
    )
    motion_axis_ref = _optional_ref(motion_axis_ref, refs, "motion_axis_ref")
    target_obstacle_ref = _scene_object_ref_for_keypoint(
        observed_scene_graph=observed_scene_graph,
        keypoint_ref=target_contact_ref,
    )
    if target_obstacle_ref not in refs:
        target_obstacle_ref = None
    contact_ref = str(derived["derived_contact_ref"])
    selected_region_ref = str(derived["selected_function_region_ref"])
    selected_axis_ref = _required_ref(
        selected_axis_ref,
        _selected_region_function_axis_refs(
            selected_ref=selected_region_ref,
            grounding_candidates=grounding_candidates,
        ),
        "selected_function_axis_ref",
    )
    if not motion_axis_ref:
        raise GivenFunctionRegionPlannerError(
            "selected region contact-frame mainline requires motion_axis_ref"
        )
    return {
        "schema_version": "v0.3",
        "binding_mode": "selected_region_to_target",
        "slot_bindings": {
            "grasp_ref": None,
            "function_keypoint_ref": contact_ref,
            "function_axis_ref": selected_axis_ref,
            "target_pre_contact_ref": target_pre_contact_ref,
            "target_contact_ref": target_contact_ref,
            "target_post_contact_ref": target_post_contact_ref,
            "motion_axis_ref": motion_axis_ref,
            "target_obstacle_ref": target_obstacle_ref,
        },
        "functional_frame_binding": {
            "contact_ref": contact_ref,
            "primary_ref": contact_ref,
            "axis_source_ref": selected_region_ref,
            "secondary_ref": None,
            "tertiary_ref": None,
            "parent_part_ref": selected_region_ref,
            "construction": "selected_region_contact_patch_v1",
        },
        "target_binding": {
            "target_pre_contact_ref": target_pre_contact_ref,
            "target_contact_ref": target_contact_ref,
            "target_post_contact_ref": target_post_contact_ref,
            "motion_axis_ref": motion_axis_ref,
        },
        "region_binding": {
            **derived,
            "source_region_selection": region_selection.get("artifact_id")
            or region_selection.get("metadata", {}).get("context_id"),
            "region_selection_reason": region_selection.get("reason"),
            "region_selection_confidence": region_selection.get("confidence"),
        },
    }


def _collect_program_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str) and value.startswith(("candidate.", "scene.", "target.", "tool.")):
        refs.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            refs.update(_collect_program_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_collect_program_refs(child))
    return refs


def render_compositional_affordance_program(
    compositional_program: Mapping[str, Any],
) -> str:
    """Render VLM-composed geometric ops into restricted ``build_program`` code."""

    executable_stages: list[dict[str, Any]] = []
    stages_for_refs = compositional_program.get("stages")
    if isinstance(stages_for_refs, list):
        for stage in stages_for_refs:
            if isinstance(stage, Mapping):
                executable_stages.append({"ops": stage.get("ops")})
    executable_ref_source = {
        key: compositional_program.get(key)
        for key in (
            "selected_tool_region_ref",
            "selected_function_contact_ref",
            "selected_function_axis_ref",
            "target_pre_contact_ref",
            "target_contact_ref",
            "target_post_contact_ref",
            "motion_axis_ref",
        )
    }
    executable_ref_source["stages"] = executable_stages
    refs = sorted(_collect_program_refs(executable_ref_source))
    ref_vars = {ref: f"r{index}" for index, ref in enumerate(refs)}

    def q(value: str) -> str:
        return json.dumps(value)

    def var(ref: str) -> str:
        return ref_vars[ref]

    lines = ["def build_program(ctx):"]
    for ref in refs:
        lines.append(f"    {var(ref)} = ctx.ref({q(ref)})")
    stages = compositional_program.get("stages")
    if not isinstance(stages, list) or not stages:
        raise GivenFunctionRegionPlannerError("part-centric affordance program has no stages")
    for stage_index, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            raise GivenFunctionRegionPlannerError("compositional stage must be an object")
        stage_var = f"stage_{stage_index}"
        stage_name = str(stage.get("name") or f"stage_{stage_index}")
        description = str(stage.get("description") or stage_name)
        repeat = int(stage.get("repeat") or 1)
        lines.extend(["", f"    {stage_var} = ctx.stage({q(stage_name)}, {q(description)})"])
        if repeat > 1:
            lines[-1] = f"{lines[-1]}.repeat({repeat})"
        ops = stage.get("ops")
        if not isinstance(ops, list):
            raise GivenFunctionRegionPlannerError(f"stage {stage_name!r} has no ops")
        for op in ops:
            if not isinstance(op, Mapping):
                raise GivenFunctionRegionPlannerError("compositional op must be an object")
            op_type = op.get("type")
            if op_type in {"near", "contact", "maintain_contact"}:
                lines.append(
                    f"    {stage_var}.constraint(ctx.constraints.contact("
                    f"{var(str(op['tool_ref']))}, {var(str(op['target_ref']))}, "
                    f"tolerance_m={float(op['tolerance_m']):.6g}))"
                )
            elif op_type == "align_axis":
                lines.extend(
                    [
                        f"    {stage_var}.constraint(ctx.constraints.align_axis(",
                        f"        {var(str(op['tool_axis_ref']))}, {var(str(op['target_axis_ref']))},",
                        f"        polarity={q(str(op['polarity']))}, min_cosine={float(op['min_cosine']):.6g}",
                        "    ))",
                    ]
                )
            elif op_type == "pre_contact_offset":
                lines.extend(
                    [
                        f"    {stage_var}.constraint(ctx.constraints.pre_contact_offset(",
                        f"        {var(str(op['tool_ref']))}, {var(str(op['target_ref']))}, {var(str(op['axis_ref']))},",
                        f"        distance_m={float(op['distance_m']):.6g}, direction={q(str(op['direction']))}",
                        "    ))",
                    ]
                )
            elif op_type in {"move_along", "slide_along", "rotate_about"}:
                if op.get("profile") == "swing_arc":
                    lines.extend(
                        [
                            f"    {stage_var}.constraint(ctx.constraints.move_along(",
                            f"        {var(str(op['tool_ref']))}, {var(str(op['start_ref']))}, {var(str(op['goal_ref']))},",
                            "        profile=\"swing_arc\",",
                            f"        path_family={q(str(op.get('path_family', 'swing')))},",
                            f"        swing_axis={var(str(op['swing_axis_ref']))},",
                            f"        swing_angle_rad={float(op['swing_angle_rad']):.6g},",
                            f"        swing_lift_m={float(op['swing_lift_m']):.6g},",
                            "    ))",
                        ]
                    )
                else:
                    lines.append(
                        f"    {stage_var}.constraint(ctx.constraints.move_along("
                        f"{var(str(op['tool_ref']))}, {var(str(op['start_ref']))}, {var(str(op['goal_ref']))}, "
                        f"path_family={q(str(op.get('path_family', 'contact_follow')))}))"
                    )
            elif op_type == "clearance":
                lines.extend(
                    [
                        f"    {stage_var}.constraint(ctx.constraints.clearance(",
                        f"        {var(str(op['moving_ref']))}, {var(str(op['obstacle_ref']))},",
                        f"        min_distance_m={float(op['min_distance_m']):.6g}",
                        "    ))",
                    ]
                )
            else:
                raise GivenFunctionRegionPlannerError(f"unsupported compositional op type: {op_type!r}")
    lines.extend(["", "    return ctx.program()", ""])
    return "\n".join(lines)


def build_compositional_affordance_program_ir(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    region_selection: dict[str, Any],
    task_instruction: str,
    program_id: str,
    grounding_candidates: dict[str, Any],
    part_centric_affordance_program: dict[str, Any],
    tool_part_graph: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_compositional_affordance_program(
        part_centric_affordance_program=part_centric_affordance_program,
        region_selection=region_selection,
        grounding_candidates=grounding_candidates,
        observed_scene_graph=observed_scene_graph,
        target_part_graph=target_part_graph,
    )
    binding = build_region_affordance_binding(
        region_selection=region_selection,
        grounding_candidates=grounding_candidates,
        observed_scene_graph=observed_scene_graph,
        target_part_graph=target_part_graph,
        target_pre_contact_ref=normalized.get("target_pre_contact_ref"),
        target_contact_ref=normalized.get("target_contact_ref"),
        target_post_contact_ref=normalized.get("target_post_contact_ref"),
        motion_axis_ref=normalized.get("motion_axis_ref"),
        selected_contact_ref=normalized.get("selected_function_contact_ref"),
        selected_axis_ref=normalized.get("selected_function_axis_ref"),
        contact_patch_policy="vlm_selected_candidate",
    )
    symbol_table = build_program_symbol_table(
        observed_scene_graph,
        functional_tool_parts,
        grounding_candidates=grounding_candidates,
        tool_part_graph=tool_part_graph,
        target_part_graph=target_part_graph,
    )
    program_text = render_compositional_affordance_program(normalized)
    planner_program = make_planner_program_artifact(
        raw_text=program_text,
        program_id=program_id,
        task_instruction=task_instruction,
        generated_by="parttoolgen.program.part_centric_meta_action_program",
        prompt_id=PROMPT_TEMPLATE_ID,
        declared_symbol_table=symbol_table,
    )
    planner_program["generation_metadata"]["part_centric_meta_action_program"] = {
        "schema_version": normalized["schema_version"],
        "selected_tool_region_ref": normalized["selected_tool_region_ref"],
        "selected_function_contact_ref": normalized["selected_function_contact_ref"],
        "selected_function_axis_ref": normalized["selected_function_axis_ref"],
        "target_contact_ref": normalized["target_contact_ref"],
        "motion_axis_ref": normalized["motion_axis_ref"],
        "stage_count": len(normalized["stages"]),
    }
    validate_planner_program(planner_program)
    context = ProgramContext(
        program_id=program_id,
        task_instruction=task_instruction,
        allowed_refs=flatten_allowed_refs(symbol_table),
    )
    program_ir = execute_planner_program(
        program_text=planner_program["program_text"],
        context=context,
    )
    validate_program_ir(program_ir)
    contract_report = check_program_ir_affordance_contract(program_ir, binding)
    if not contract_report.get("passed", False):
        hints = contract_report.get("repair_hints", [])
        message = "compositional ProgramIR failed affordance contract gate"
        if hints:
            message = f"{message}: {hints[0]}"
        raise GivenFunctionRegionPlannerError(message.replace("compositional", "part-centric affordance program"))
    return {
        "planner_program": planner_program,
        "program_ir": program_ir,
        "affordance_binding": binding,
        "part_centric_affordance_program": normalized,
        "grounding_candidates": grounding_candidates,
        "program_ir_affordance_contract_report": contract_report,
    }


def build_given_function_region_program_ir(
    *,
    observed_scene_graph: dict[str, Any],
    functional_tool_parts: dict[str, Any],
    region_selection: dict[str, Any],
    task_instruction: str,
    program_id: str,
    grounding_candidates: dict[str, Any],
    tool_part_graph: dict[str, Any] | None = None,
    target_part_graph: dict[str, Any] | None = None,
    part_centric_affordance_program: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if part_centric_affordance_program is None:
        raise GivenFunctionRegionPlannerError(
            "5.15 mainline requires a VLM-authored Part-Centric Affordance Program; "
            "region-only fixed-template planning is not available"
        )
    if not isinstance(part_centric_affordance_program.get("stages"), list):
        raise GivenFunctionRegionPlannerError(
            "part_centric_affordance_program must contain stages with ops/interactions/motions"
        )
    return build_compositional_affordance_program_ir(
        observed_scene_graph=observed_scene_graph,
        functional_tool_parts=functional_tool_parts,
        region_selection=region_selection,
        task_instruction=task_instruction,
        program_id=program_id,
        grounding_candidates=grounding_candidates,
        part_centric_affordance_program=part_centric_affordance_program,
        tool_part_graph=tool_part_graph,
        target_part_graph=target_part_graph,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--region-selection", required=True)
    parser.add_argument("--part-centric-affordance-program", required=True)
    parser.add_argument("--planner-output", required=True)
    parser.add_argument("--program-ir-output", required=True)
    parser.add_argument("--affordance-binding-output", required=True)
    parser.add_argument("--program-ir-affordance-contract-report-output")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--tool-parts")
    parser.add_argument("--target-parts")
    args = parser.parse_args()

    kwargs = {
        "observed_scene_graph": load_structured_file(args.scene),
        "functional_tool_parts": load_structured_file(args.tool),
        "region_selection": load_structured_file(args.region_selection),
        "task_instruction": args.task,
        "program_id": args.program_id,
        "grounding_candidates": load_structured_file(args.candidates),
        "tool_part_graph": load_structured_file(args.tool_parts) if args.tool_parts else None,
        "target_part_graph": load_structured_file(args.target_parts) if args.target_parts else None,
        "part_centric_affordance_program": load_structured_file(args.part_centric_affordance_program),
    }
    result = build_given_function_region_program_ir(**kwargs)
    _json_dump(args.affordance_binding_output, result["affordance_binding"])
    _json_dump(args.planner_output, result["planner_program"])
    _json_dump(args.program_ir_output, result["program_ir"])
    if args.program_ir_affordance_contract_report_output:
        _json_dump(
            args.program_ir_affordance_contract_report_output,
            result["program_ir_affordance_contract_report"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
