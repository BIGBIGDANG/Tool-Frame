#!/usr/bin/env python3
"""Resolve ProgramIR symbolic refs into typed grounded geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class ReferenceResolutionError(ValueError):
    """Raised when a ProgramIR ref cannot be resolved to usable geometry."""


@dataclass(frozen=True)
class ResolvedRef:
    ref: str
    kind: str
    frame: str
    entity: dict[str, Any]
    point_world: list[float] | None = None
    vector_world: list[float] | None = None
    point_tool: list[float] | None = None
    point_tool_candidates: list[dict[str, Any]] | None = None
    vector_tool: list[float] | None = None
    vector_tool_candidates: list[dict[str, Any]] | None = None
    confidence: float | None = None
    source: str | None = None
    provenance: dict[str, Any] | None = None


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ReferenceResolutionError(f"{name} must be a 3-vector")
    return [float(item) for item in value]


def _unit(value: list[float], name: str) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in value))
    if norm <= 1e-12:
        raise ReferenceResolutionError(f"{name} is a zero vector")
    return [float(item) / norm for item in value]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def _mul(a: list[float], scale: float) -> list[float]:
    return [float(x) * float(scale) for x in a]


def _grounding(entity: dict[str, Any]) -> tuple[float | None, str | None, dict[str, Any] | None]:
    confidence = entity.get("confidence")
    source = entity.get("source")
    provenance = entity.get("provenance")
    return (
        float(confidence) if isinstance(confidence, (int, float)) else None,
        str(source) if isinstance(source, str) else None,
        provenance if isinstance(provenance, dict) else None,
    )


def _axis_candidates_tool(entity: dict[str, Any], ref: str) -> list[dict[str, Any]]:
    raw_candidates = entity.get("axis_candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if isinstance(raw, dict):
            vector = raw.get("vector_tool")
            if vector is None:
                continue
            candidate = dict(raw)
            candidate["vector_tool"] = _unit(
                _vec3(vector, f"{ref}.axis_candidates[{index}].vector_tool"),
                f"{ref}.axis_candidates[{index}].vector_tool",
            )
            candidates.append(candidate)
        elif isinstance(raw, list):
            candidates.append(
                {
                    "candidate_id": f"axis_candidate_{index}",
                    "vector_tool": _unit(
                        _vec3(raw, f"{ref}.axis_candidates[{index}]"),
                        f"{ref}.axis_candidates[{index}]",
                    ),
                }
            )
    return candidates


def _point_candidates_tool(entity: dict[str, Any], ref: str) -> list[dict[str, Any]]:
    raw_candidates = entity.get("point_candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if isinstance(raw, dict):
            point = (
                raw.get("point_tool")
                or raw.get("contact_point_tool")
                or raw.get("xyz_tool")
                or raw.get("center_xyz_tool")
            )
            if point is None:
                continue
            candidate = dict(raw)
            candidate["point_tool"] = _vec3(point, f"{ref}.point_candidates[{index}].point_tool")
            candidates.append(candidate)
        elif isinstance(raw, list):
            candidates.append(
                {
                    "candidate_id": f"point_candidate_{index}",
                    "point_tool": _vec3(raw, f"{ref}.point_candidates[{index}]"),
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        point = _vec3(candidate.get("point_tool"), f"{ref}.point_candidate.point_tool")
        key = tuple(round(value, 6) for value in point)
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["point_tool"] = list(key)
        deduped.append(row)
    return deduped


def _first_patch(part: dict[str, Any], ref: str) -> dict[str, Any]:
    patches = part.get("surface_patches", {})
    if not isinstance(patches, dict) or not patches:
        raise ReferenceResolutionError(f"{ref} has no surface patches")
    first_key = sorted(patches)[0]
    patch = patches[first_key]
    if not isinstance(patch, dict):
        raise ReferenceResolutionError(f"{ref}.{first_key} patch is invalid")
    return patch


class ReferenceResolver:
    """Resolve refs from ProgramIR against grounded scene/tool artifacts."""

    def __init__(
        self,
        *,
        observed_scene_graph: dict[str, Any],
        functional_tool_parts: dict[str, Any],
        grounding_candidates: dict[str, Any] | None = None,
        target_part_graph: dict[str, Any] | None = None,
        tool_part_graph: dict[str, Any] | None = None,
    ) -> None:
        self.scene = observed_scene_graph
        self.tool = functional_tool_parts
        self.candidates = grounding_candidates or {}
        self.target = target_part_graph or {}
        self.tool_parts = tool_part_graph or {}

    def resolve(self, ref: str) -> ResolvedRef:
        parts = ref.split(".")
        if not parts:
            raise ReferenceResolutionError("empty ref")
        if parts[0] == "scene":
            return self._resolve_scene(parts, ref)
        if parts[0] == "tool":
            return self._resolve_tool(parts, ref)
        if parts[0] == "target":
            return self._resolve_target(parts, ref)
        if parts[0] == "candidate":
            return self._resolve_candidate(parts, ref)
        raise ReferenceResolutionError(f"unsupported ref namespace: {ref}")

    def point(self, ref: str) -> ResolvedRef:
        resolved = self.resolve(ref)
        if resolved.point_world is None and resolved.point_tool is None:
            raise ReferenceResolutionError(f"{ref} does not resolve to a point")
        return resolved

    def vector(self, ref: str) -> ResolvedRef:
        resolved = self.resolve(ref)
        if resolved.vector_world is None and resolved.vector_tool is None:
            raise ReferenceResolutionError(f"{ref} does not resolve to an axis/vector")
        return resolved

    def tool_point(self, ref: str) -> ResolvedRef:
        resolved = self.point(ref)
        if resolved.point_tool is None:
            raise ReferenceResolutionError(f"{ref} is not a tool-frame point")
        return resolved

    def world_point(self, ref: str) -> ResolvedRef:
        resolved = self.point(ref)
        if resolved.point_world is None:
            raise ReferenceResolutionError(f"{ref} is not a world-frame point")
        return resolved

    def tool_vector(self, ref: str) -> ResolvedRef:
        resolved = self.vector(ref)
        if resolved.vector_tool is None:
            raise ReferenceResolutionError(f"{ref} is not a tool-frame vector")
        return resolved

    def world_vector(self, ref: str) -> ResolvedRef:
        resolved = self.vector(ref)
        if resolved.vector_world is None:
            raise ReferenceResolutionError(f"{ref} is not a world-frame vector")
        return resolved

    def default_tool_point_ref(self) -> str:
        return "tool.contact_point"

    def orientation_sample_points_tool(self) -> list[list[float]]:
        metadata = self.tool.get("metadata", {})
        if not isinstance(metadata, dict):
            return []
        metric = metadata.get("orientation_change_metric", {})
        if not isinstance(metric, dict):
            return []
        points = metric.get("sample_points_tool")
        if not isinstance(points, list):
            return []
        parsed: list[list[float]] = []
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 3:
                raise ReferenceResolutionError(
                    f"orientation_change_metric.sample_points_tool[{index}] must be a 3-vector"
                )
            parsed.append([float(item) for item in point])
        return parsed

    def table_clearance_sample_points_tool(self) -> list[list[float]]:
        points = self.tool_part_obb_corners_tool()
        parsed: list[list[float]] = []
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 3:
                raise ReferenceResolutionError(
                    f"table_clearance_metric.sample_points_tool[{index}] must be a 3-vector"
                )
            parsed.append([float(item) for item in point])
        return parsed

    def tool_part_obb_corners_tool(self) -> list[list[float]]:
        parts = self.tool_parts.get("parts", {})
        if not isinstance(parts, dict):
            return []
        corners: list[list[float]] = []
        for part_id in sorted(parts):
            part = parts.get(part_id)
            if not isinstance(part, dict):
                continue
            obb = part.get("obb")
            if not isinstance(obb, dict):
                continue
            center = _vec3(obb.get("center_xyz_tool"), f"tool.parts.{part_id}.obb.center_xyz_tool")
            extent = _vec3(obb.get("extent_xyz"), f"tool.parts.{part_id}.obb.extent_xyz")
            axes_raw = obb.get("axes_tool")
            if not isinstance(axes_raw, dict):
                raise ReferenceResolutionError(f"tool.parts.{part_id}.obb.axes_tool must be an object")
            axes = [
                _unit(_vec3(axes_raw.get(axis), f"tool.parts.{part_id}.obb.axes_tool.{axis}"), f"tool.parts.{part_id}.obb.axes_tool.{axis}")
                for axis in ("x", "y", "z")
            ]
            half = [0.5 * value for value in extent]
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        point = center
                        for axis_vec, sign, half_extent in zip(axes, (sx, sy, sz), half):
                            point = _add(point, _mul(axis_vec, sign * half_extent))
                        corners.append(point)
        return corners

    def _resolved(
        self,
        *,
        ref: str,
        kind: str,
        frame: str,
        entity: dict[str, Any],
        point_world: list[float] | None = None,
        vector_world: list[float] | None = None,
        point_tool: list[float] | None = None,
        point_tool_candidates: list[dict[str, Any]] | None = None,
        vector_tool: list[float] | None = None,
        vector_tool_candidates: list[dict[str, Any]] | None = None,
    ) -> ResolvedRef:
        confidence, source, provenance = _grounding(entity)
        return ResolvedRef(
            ref=ref,
            kind=kind,
            frame=frame,
            entity=entity,
            point_world=point_world,
            vector_world=_unit(vector_world, ref) if vector_world is not None else None,
            point_tool=point_tool,
            point_tool_candidates=point_tool_candidates,
            vector_tool=_unit(vector_tool, ref) if vector_tool is not None else None,
            vector_tool_candidates=vector_tool_candidates,
            confidence=confidence,
            source=source,
            provenance=provenance,
        )

    def _resolve_scene(self, parts: list[str], ref: str) -> ResolvedRef:
        if len(parts) != 3:
            raise ReferenceResolutionError(f"invalid scene ref: {ref}")
        collection, entity_id = parts[1], parts[2]
        entity = self.scene.get(collection, {}).get(entity_id)
        if not isinstance(entity, dict):
            raise ReferenceResolutionError(f"unknown scene ref: {ref}")
        if collection == "keypoints":
            return self._resolved(
                ref=ref,
                kind="point",
                frame="world",
                entity=entity,
                point_world=_vec3(entity.get("xyz_world"), f"{ref}.xyz_world"),
            )
        if collection == "axes":
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="world",
                entity=entity,
                vector_world=_vec3(entity.get("vector_world"), f"{ref}.vector_world"),
            )
        if collection == "regions":
            return self._resolved(
                ref=ref,
                kind="region",
                frame="world",
                entity=entity,
                point_world=_vec3(entity.get("center_xyz_world"), f"{ref}.center_xyz_world"),
                vector_world=_vec3(entity["normal_world"], f"{ref}.normal_world")
                if "normal_world" in entity
                else None,
            )
        if collection == "surfaces":
            return self._resolved(
                ref=ref,
                kind="surface",
                frame="world",
                entity=entity,
                point_world=_vec3(entity.get("point_xyz_world"), f"{ref}.point_xyz_world"),
                vector_world=_vec3(entity.get("normal_world"), f"{ref}.normal_world"),
            )
        if collection == "objects":
            attributes = entity.get("attributes", {})
            point = None
            if isinstance(attributes, dict):
                for key in ("center_xyz_world", "bbox_center_xyz_world", "position_world"):
                    if key in attributes:
                        point = _vec3(attributes[key], f"{ref}.attributes.{key}")
                        break
            return self._resolved(ref=ref, kind="object", frame="world", entity=entity, point_world=point)
        raise ReferenceResolutionError(f"unsupported scene collection in {ref}")

    def _resolve_tool(self, parts: list[str], ref: str) -> ResolvedRef:
        if len(parts) == 2 and parts[1] in {"contact_point", "motion_axis"}:
            functional_parts = self.tool.get("functional_parts", {})
            if not isinstance(functional_parts, dict) or not functional_parts:
                raise ReferenceResolutionError("FunctionalToolParts has no functional parts")
            entity = functional_parts.get("function") or functional_parts.get(sorted(functional_parts)[0])
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown tool ref: {ref}")
            if parts[1] == "contact_point":
                return self._resolved(
                    ref=ref,
                    kind="point",
                    frame="tool",
                    entity=entity,
                    point_tool=_vec3(entity.get("center", {}).get("xyz_tool"), f"{ref}.xyz_tool"),
                    point_tool_candidates=_point_candidates_tool(entity, ref),
                )
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="tool",
                entity=entity,
                vector_tool=_vec3(entity.get("axis", {}).get("vector_tool"), f"{ref}.vector_tool"),
                vector_tool_candidates=_axis_candidates_tool(entity, ref),
            )
        if len(parts) >= 3 and parts[1] == "grasp_references":
            grasp_id = parts[2]
            entity = self.tool.get("grasp_references", {}).get(grasp_id)
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown tool grasp ref: {ref}")
            return self._resolved(
                ref=ref,
                kind="grasp",
                frame="tool",
                entity=entity,
                point_tool=_vec3(entity.get("position_tool"), f"{ref}.position_tool"),
            )
        if len(parts) >= 3 and parts[1] == "parts":
            part_id = parts[2]
            entity = self.tool_parts.get("parts", {}).get(part_id)
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown tool part ref: {ref}")
            return self._resolved(
                ref=ref,
                kind="part",
                frame="tool",
                entity=entity,
                point_tool=_vec3(entity.get("centroid_xyz_tool"), f"{ref}.centroid_xyz_tool"),
            )
        if len(parts) != 4 or parts[1] != "functional_parts":
            raise ReferenceResolutionError(f"invalid tool ref: {ref}")
        part_id, attribute = parts[2], parts[3]
        entity = self.tool.get("functional_parts", {}).get(part_id)
        if not isinstance(entity, dict):
            raise ReferenceResolutionError(f"unknown functional part ref: {ref}")
        if attribute == "center":
            return self._resolved(
                ref=ref,
                kind="point",
                frame="tool",
                entity=entity,
                point_tool=_vec3(entity.get("center", {}).get("xyz_tool"), f"{ref}.xyz_tool"),
                point_tool_candidates=_point_candidates_tool(entity, ref),
            )
        if attribute == "normal":
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="tool",
                entity=entity,
                vector_tool=_vec3(entity.get("normal", {}).get("vector_tool"), f"{ref}.vector_tool"),
            )
        if attribute == "axis":
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="tool",
                entity=entity,
                vector_tool=_vec3(entity.get("axis", {}).get("vector_tool"), f"{ref}.vector_tool"),
                vector_tool_candidates=_axis_candidates_tool(entity, ref),
            )
        if attribute == "tangent_long":
            vector = entity.get("tangent_long", {}).get("vector_tool")
            if vector is None:
                vector = entity.get("axis", {}).get("vector_tool") or entity.get("frame", {}).get("x_axis")
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="tool",
                entity=entity,
                vector_tool=_vec3(vector, f"{ref}.vector_tool"),
            )
        if attribute == "tangent_short":
            vector = entity.get("tangent_short", {}).get("vector_tool")
            if vector is None:
                vector = entity.get("frame", {}).get("y_axis")
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="tool",
                entity=entity,
                vector_tool=_vec3(vector, f"{ref}.vector_tool"),
            )
        if attribute in {"frame", "T_tool_functional"}:
            frame = entity.get("frame", {})
            return self._resolved(
                ref=ref,
                kind="frame",
                frame="tool",
                entity=entity,
                point_tool=_vec3(frame.get("origin"), f"{ref}.origin"),
                vector_tool=_vec3(frame.get("z_axis"), f"{ref}.z_axis"),
            )
        raise ReferenceResolutionError(f"unsupported functional part attribute in {ref}")

    def _resolve_target(self, parts: list[str], ref: str) -> ResolvedRef:
        if len(parts) != 3:
            raise ReferenceResolutionError(f"invalid target ref: {ref}")
        collection, entity_id = parts[1], parts[2]
        if collection == "axes":
            entity = self.target.get("axes", {}).get(entity_id)
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown target axis ref: {ref}")
            return self._resolved(
                ref=ref,
                kind="axis",
                frame="world",
                entity=entity,
                vector_world=_vec3(entity.get("vector_world"), f"{ref}.vector_world"),
            )
        if collection == "parts":
            entity = self.target.get("parts", {}).get(entity_id)
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown target part ref: {ref}")
            patch = _first_patch(entity, ref)
            return self._resolved(
                ref=ref,
                kind="part",
                frame="world",
                entity=entity,
                point_world=_vec3(patch.get("center_xyz_world"), f"{ref}.center_xyz_world"),
                vector_world=_vec3(patch.get("normal_world"), f"{ref}.normal_world"),
            )
        if collection == "regions":
            entity = self.target.get("regions", {}).get(entity_id)
            if not isinstance(entity, dict):
                raise ReferenceResolutionError(f"unknown target region ref: {ref}")
            return self._resolved(
                ref=ref,
                kind="region",
                frame="world",
                entity=entity,
                point_world=_vec3(entity.get("center_xyz_world"), f"{ref}.center_xyz_world"),
                vector_world=_vec3(entity["normal_world"], f"{ref}.normal_world")
                if "normal_world" in entity
                else None,
            )
        raise ReferenceResolutionError(f"unsupported target collection in {ref}")

    def _resolve_candidate(self, parts: list[str], ref: str) -> ResolvedRef:
        if len(parts) != 3:
            raise ReferenceResolutionError(f"invalid candidate ref: {ref}")
        collection, entity_id = parts[1], parts[2]
        entity = self.candidates.get(collection, {}).get(entity_id)
        if not isinstance(entity, dict):
            raise ReferenceResolutionError(f"unknown candidate ref: {ref}")
        if collection in {"marks", "points"}:
            if "xyz_world" in entity:
                return self._resolved(
                    ref=ref,
                    kind="point",
                    frame="world",
                    entity=entity,
                    point_world=_vec3(entity.get("xyz_world"), f"{ref}.xyz_world"),
                )
            if "xyz_tool" in entity:
                return self._resolved(
                    ref=ref,
                    kind="point",
                    frame="tool",
                    entity=entity,
                    point_tool=_vec3(entity.get("xyz_tool"), f"{ref}.xyz_tool"),
                )
            raise ReferenceResolutionError(f"{ref} has no xyz_world or xyz_tool")
        if collection == "regions":
            if "center_xyz_world" in entity:
                return self._resolved(
                    ref=ref,
                    kind="region",
                    frame="world",
                    entity=entity,
                    point_world=_vec3(entity.get("center_xyz_world"), f"{ref}.center_xyz_world"),
                )
            if "center_xyz_tool" in entity:
                return self._resolved(
                    ref=ref,
                    kind="region",
                    frame="tool",
                    entity=entity,
                    point_tool=_vec3(entity.get("center_xyz_tool"), f"{ref}.center_xyz_tool"),
                )
            raise ReferenceResolutionError(f"{ref} has no center_xyz_world or center_xyz_tool")
        if collection == "axes":
            if "vector_world" in entity:
                return self._resolved(
                    ref=ref,
                    kind="axis",
                    frame="world",
                    entity=entity,
                    vector_world=_vec3(entity.get("vector_world"), f"{ref}.vector_world"),
                )
            if "vector_tool" in entity:
                return self._resolved(
                    ref=ref,
                    kind="axis",
                    frame="tool",
                    entity=entity,
                    vector_tool=_vec3(entity.get("vector_tool"), f"{ref}.vector_tool"),
                )
            raise ReferenceResolutionError(f"{ref} has no vector_world or vector_tool")
        if collection == "surfaces":
            if "point_xyz_world" in entity and "normal_world" in entity:
                return self._resolved(
                    ref=ref,
                    kind="surface",
                    frame="world",
                    entity=entity,
                    point_world=_vec3(entity.get("point_xyz_world"), f"{ref}.point_xyz_world"),
                    vector_world=_vec3(entity.get("normal_world"), f"{ref}.normal_world"),
                )
            if "point_xyz_tool" in entity and "normal_tool" in entity:
                return self._resolved(
                    ref=ref,
                    kind="surface",
                    frame="tool",
                    entity=entity,
                    point_tool=_vec3(entity.get("point_xyz_tool"), f"{ref}.point_xyz_tool"),
                    vector_tool=_vec3(entity.get("normal_tool"), f"{ref}.normal_tool"),
                )
            raise ReferenceResolutionError(
                f"{ref} has no complete point/normal pair in world or tool frame"
            )
        if collection == "parts":
            part_ref = entity.get("part_ref")
            if isinstance(part_ref, str) and part_ref.startswith("target.parts."):
                return self._resolve_target(part_ref.split("."), part_ref)
            if isinstance(part_ref, str) and part_ref.startswith("tool.parts."):
                return self._resolve_tool(part_ref.split("."), part_ref)
            return self._resolved(
                ref=ref,
                kind="part",
                frame="tool",
                entity=entity,
                point_tool=_vec3(entity.get("center_xyz_tool"), f"{ref}.center_xyz_tool")
                if "center_xyz_tool" in entity
                else None,
            )
        raise ReferenceResolutionError(f"{ref} cannot be used as metric geometry")
