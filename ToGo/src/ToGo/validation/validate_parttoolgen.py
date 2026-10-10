#!/usr/bin/env python3
"""Strict validators for the PartToolGen runtime artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is present in the target env.
    yaml = None


class SchemaValidationError(ValueError):
    """Raised when an artifact violates a PartToolGen schema."""


SCENE_REF_RE = re.compile(
    r"^(scene\.(objects|keypoints|axes|regions|surfaces)\.[A-Za-z_][A-Za-z0-9_]*|"
    r"scene\.(start|end|direction|motion_path)(\[[0-9]+\])?)$"
)
TOOL_REF_RE = re.compile(
    r"^(tool\.(contact_point|motion_axis)|"
    r"tool\.functional_parts\.[A-Za-z_][A-Za-z0-9_]*\."
    r"(center|normal|axis|tangent_long|tangent_short|frame|T_tool_functional)|"
    r"tool\.grasp_references\.[A-Za-z_][A-Za-z0-9_]*|"
    r"tool\.parts\.[A-Za-z_][A-Za-z0-9_]*)$"
)
TARGET_REF_RE = re.compile(
    r"^target\.(parts|axes|regions)\.[A-Za-z_][A-Za-z0-9_]*$"
)
CANDIDATE_REF_RE = re.compile(
    r"^candidate\.(marks|points|regions|axes|surfaces|parts)\.[A-Za-z_][A-Za-z0-9_]*$"
)
GROUNDED_REF_RES = (SCENE_REF_RE, TOOL_REF_RE, TARGET_REF_RE, CANDIDATE_REF_RE)
SEMANTIC_QUERY_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SEMANTIC_QUERY_SCOPES = {"scene", "tool", "target", "candidate"}
SEMANTIC_QUERY_GEOMETRY = {"object", "part", "point", "axis", "region", "surface"}
SEMANTIC_QUERY_CATEGORIES = {"object", "tool_part", "functional_tool_part", "target_part"}
PROGRAM_API_VERSIONS = {
    "parttoolgen_program_api_v0.2",
    "parttoolgen_program_api_v0.3",
}
SEMANTIC_QUERY_FORBIDDEN_KEYS = {
    "bbox",
    "box",
    "center_xyz_world",
    "extent_xyz",
    "goal",
    "mask_path",
    "mask_ref",
    "normal_world",
    "orientation",
    "orientation_xyzw",
    "pixel",
    "point_xyz_world",
    "pose",
    "pose_world_tool",
    "position",
    "position_world",
    "segmentation_id",
    "T_tool_functional",
    "T_world_tool",
    "tool_pose_world",
    "vector_world",
    "waypoint",
    "xyz_world",
}


def load_structured_file(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise SchemaValidationError("PyYAML is required to load YAML files")
            return yaml.safe_load(f)
        return json.load(f)


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaValidationError(f"{name} must be an object")
    return data


def _check_keys(data: dict[str, Any], required: set[str], allowed: set[str], name: str) -> None:
    missing = sorted(required - set(data))
    extra = sorted(set(data) - allowed)
    if missing:
        raise SchemaValidationError(f"{name} missing required fields: {missing}")
    if extra:
        raise SchemaValidationError(f"{name} has unexpected fields: {extra}")


def _expect_const(data: dict[str, Any], key: str, expected: Any) -> None:
    if data.get(key) != expected:
        raise SchemaValidationError(f"{key} must be {expected!r}, got {data.get(key)!r}")


def _non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{name} must be a non-empty string")


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SchemaValidationError(f"{name} must be a finite number")
    return float(value)


def _number_list(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaValidationError(f"{name} must be a list of length {length}")
    return [_finite_number(item, f"{name}[{i}]") for i, item in enumerate(value)]


def _quat(value: Any, name: str) -> list[float]:
    quat = _number_list(value, 4, name)
    norm = math.sqrt(sum(x * x for x in quat))
    if abs(norm - 1.0) > 1e-3:
        raise SchemaValidationError(f"{name} quaternion norm {norm:.6f} is not near 1")
    return quat


def _nonzero_vec3(value: Any, name: str) -> list[float]:
    vec = _number_list(value, 3, name)
    if math.sqrt(sum(x * x for x in vec)) <= 1e-12:
        raise SchemaValidationError(f"{name} must be non-zero")
    return vec


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in vector))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _require_unit_vec3(value: Any, name: str, *, tolerance: float = 1e-3) -> list[float]:
    vec = _nonzero_vec3(value, name)
    norm = _norm(vec)
    if abs(norm - 1.0) > tolerance:
        raise SchemaValidationError(f"{name} norm {norm:.6f} is not near 1")
    return vec


def _confidence(value: Any, name: str) -> None:
    number = _finite_number(value, name)
    if number < 0.0 or number > 1.0:
        raise SchemaValidationError(f"{name} must be in [0, 1]")


def _provenance(value: Any, name: str) -> None:
    provenance = _require_mapping(value, name)
    if "observation_id" not in provenance or "method" not in provenance:
        raise SchemaValidationError(f"{name} must include observation_id and method")
    _non_empty_string(provenance["observation_id"], f"{name}.observation_id")
    _non_empty_string(provenance["method"], f"{name}.method")


def _grounding_metadata(value: dict[str, Any], name: str) -> None:
    for key in ("confidence", "source", "provenance"):
        if key not in value:
            raise SchemaValidationError(f"{name} missing required grounding field: {key}")
    _confidence(value["confidence"], f"{name}.confidence")
    _non_empty_string(value["source"], f"{name}.source")
    _provenance(value["provenance"], f"{name}.provenance")


def _pose_object(value: Any, name: str) -> None:
    pose = _require_mapping(value, name)
    _check_keys(
        pose,
        required={"position", "orientation_xyzw"},
        allowed={"position", "orientation_xyzw"},
        name=name,
    )
    _number_list(pose["position"], 3, f"{name}.position")
    _quat(pose["orientation_xyzw"], f"{name}.orientation_xyzw")


def _artifact_ref(value: Any, name: str) -> None:
    ref = _require_mapping(value, name)
    _check_keys(ref, required={"id", "path"}, allowed={"id", "path"}, name=name)
    _non_empty_string(ref["id"], f"{name}.id")
    _non_empty_string(ref["path"], f"{name}.path")


def _contains_forbidden_debug_marker(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return "debug" in lowered or "manual_authored" in lowered or "stub" in lowered
    if isinstance(value, dict):
        return any(_contains_forbidden_debug_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_debug_marker(item) for item in value)
    return False


def validate_observed_scene_graph(data: Any, *, observed_only: bool = False) -> None:
    data = _require_mapping(data, "ObservedSceneGraph")
    required = {
        "schema_version",
        "scene_id",
        "frames",
        "objects",
        "keypoints",
        "axes",
        "regions",
        "surfaces",
        "relations",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="ObservedSceneGraph")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["scene_id"], "scene_id")

    frames = _require_mapping(data["frames"], "frames")
    if "world" not in frames:
        raise SchemaValidationError("frames must include world")
    for frame_id, frame in frames.items():
        frame = _require_mapping(frame, f"frames.{frame_id}")
        if "type" not in frame:
            raise SchemaValidationError(f"frames.{frame_id} missing type")
        _non_empty_string(frame["type"], f"frames.{frame_id}.type")

    objects = _require_mapping(data["objects"], "objects")
    keypoints = _require_mapping(data["keypoints"], "keypoints")
    axes = _require_mapping(data["axes"], "axes")
    regions = _require_mapping(data["regions"], "regions")
    surfaces = _require_mapping(data["surfaces"], "surfaces")

    for object_id, obj in objects.items():
        obj = _require_mapping(obj, f"objects.{object_id}")
        _grounding_metadata(obj, f"objects.{object_id}")
        _non_empty_string(obj.get("category"), f"objects.{object_id}.category")

    for keypoint_id, keypoint in keypoints.items():
        keypoint = _require_mapping(keypoint, f"keypoints.{keypoint_id}")
        _grounding_metadata(keypoint, f"keypoints.{keypoint_id}")
        _expect_const(keypoint, "frame", "world")
        if "pixel" not in keypoint and "xyz_world" not in keypoint:
            raise SchemaValidationError(f"keypoints.{keypoint_id} needs pixel or xyz_world")
        if "pixel" in keypoint:
            _number_list(keypoint["pixel"], 2, f"keypoints.{keypoint_id}.pixel")
        if "xyz_world" in keypoint:
            _number_list(keypoint["xyz_world"], 3, f"keypoints.{keypoint_id}.xyz_world")
        if "object_ref" in keypoint and keypoint["object_ref"] not in objects:
            raise SchemaValidationError(f"keypoints.{keypoint_id}.object_ref does not resolve")

    for axis_id, axis in axes.items():
        axis = _require_mapping(axis, f"axes.{axis_id}")
        _grounding_metadata(axis, f"axes.{axis_id}")
        _expect_const(axis, "frame", "world")
        _require_unit_vec3(axis.get("vector_world"), f"axes.{axis_id}.vector_world")
        if "object_ref" in axis and axis["object_ref"] not in objects:
            raise SchemaValidationError(f"axes.{axis_id}.object_ref does not resolve")

    for region_id, region in regions.items():
        region = _require_mapping(region, f"regions.{region_id}")
        _grounding_metadata(region, f"regions.{region_id}")
        _expect_const(region, "frame", "world")
        _number_list(region.get("center_xyz_world"), 3, f"regions.{region_id}.center_xyz_world")
        if "extent_xyz" in region:
            _number_list(region["extent_xyz"], 3, f"regions.{region_id}.extent_xyz")
        if "normal_world" in region:
            _require_unit_vec3(region["normal_world"], f"regions.{region_id}.normal_world")
        if "object_ref" in region and region["object_ref"] not in objects:
            raise SchemaValidationError(f"regions.{region_id}.object_ref does not resolve")

    for surface_id, surface in surfaces.items():
        surface = _require_mapping(surface, f"surfaces.{surface_id}")
        _grounding_metadata(surface, f"surfaces.{surface_id}")
        _expect_const(surface, "frame", "world")
        _number_list(surface.get("point_xyz_world"), 3, f"surfaces.{surface_id}.point_xyz_world")
        _require_unit_vec3(surface.get("normal_world"), f"surfaces.{surface_id}.normal_world")

    relations = data["relations"]
    if not isinstance(relations, list):
        raise SchemaValidationError("relations must be a list")
    for i, relation in enumerate(relations):
        relation = _require_mapping(relation, f"relations[{i}]")
        for key in ("type", "subject", "object"):
            _non_empty_string(relation.get(key), f"relations[{i}].{key}")
        for key in ("subject", "object"):
            ref = relation[key]
            if ref.startswith("scene.objects.") and ref.split(".")[-1] not in objects:
                raise SchemaValidationError(f"relations[{i}].{key} does not resolve")
        if "confidence" in relation:
            _confidence(relation["confidence"], f"relations[{i}].confidence")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("geometry_mode") not in {"observed", "mixed"}:
        raise SchemaValidationError("metadata.geometry_mode is unsupported")
    if observed_only:
        if metadata.get("geometry_mode") != "observed":
            raise SchemaValidationError("observed_only scene must use observed geometry")
        if _contains_forbidden_debug_marker(data):
            raise SchemaValidationError("observed_only scene contains debug/stub provenance")


def validate_target_part_graph(data: Any) -> None:
    data = _require_mapping(data, "TargetPartGraph")
    required = {"schema_version", "graph_id", "target_id", "frame", "parts", "axes", "metadata"}
    allowed = required | {"relations"}
    _check_keys(data, required=required, allowed=allowed, name="TargetPartGraph")
    _expect_const(data, "schema_version", "v0.1")
    _expect_const(data, "frame", "world")
    _non_empty_string(data["graph_id"], "graph_id")
    _non_empty_string(data["target_id"], "target_id")

    parts = _require_mapping(data["parts"], "parts")
    if not parts:
        raise SchemaValidationError("parts must be non-empty")
    for part_id, part in parts.items():
        part = _require_mapping(part, f"parts.{part_id}")
        _grounding_metadata(part, f"parts.{part_id}")
        _non_empty_string(part.get("part_ref"), f"parts.{part_id}.part_ref")
        _non_empty_string(part.get("label"), f"parts.{part_id}.label")
        patches = _require_mapping(part.get("surface_patches"), f"parts.{part_id}.surface_patches")
        if not patches:
            raise SchemaValidationError(f"parts.{part_id}.surface_patches must be non-empty")
        for patch_id, patch in patches.items():
            patch = _require_mapping(patch, f"parts.{part_id}.surface_patches.{patch_id}")
            _grounding_metadata(patch, f"parts.{part_id}.surface_patches.{patch_id}")
            _non_empty_string(patch.get("patch_ref"), f"parts.{part_id}.surface_patches.{patch_id}.patch_ref")
            _number_list(
                patch.get("center_xyz_world"),
                3,
                f"parts.{part_id}.surface_patches.{patch_id}.center_xyz_world",
            )
            _require_unit_vec3(
                patch.get("normal_world"),
                f"parts.{part_id}.surface_patches.{patch_id}.normal_world",
            )

    axes = _require_mapping(data["axes"], "axes")
    for axis_id, axis in axes.items():
        axis = _require_mapping(axis, f"axes.{axis_id}")
        _grounding_metadata(axis, f"axes.{axis_id}")
        _non_empty_string(axis.get("axis_ref"), f"axes.{axis_id}.axis_ref")
        _require_unit_vec3(axis.get("vector_world"), f"axes.{axis_id}.vector_world")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("geometry_mode") not in {"observed", "mixed"}:
        raise SchemaValidationError("metadata.geometry_mode is unsupported")


def validate_functional_tool_parts(data: Any) -> None:
    data = _require_mapping(data, "FunctionalToolParts")
    required = {
        "schema_version",
        "tool_id",
        "tool_frame",
        "functional_parts",
        "grasp_references",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="FunctionalToolParts")
    _expect_const(data, "schema_version", "v0.1")
    _expect_const(data, "tool_frame", "tool")
    _non_empty_string(data["tool_id"], "tool_id")

    parts = _require_mapping(data["functional_parts"], "functional_parts")
    if not parts:
        raise SchemaValidationError("functional_parts must be non-empty")
    for part_id, part in parts.items():
        part = _require_mapping(part, f"functional_parts.{part_id}")
        _grounding_metadata(part, f"functional_parts.{part_id}")
        _non_empty_string(part.get("role"), f"functional_parts.{part_id}.role")
        center = _require_mapping(part.get("center"), f"functional_parts.{part_id}.center")
        center_xyz = _number_list(center.get("xyz_tool"), 3, f"functional_parts.{part_id}.center.xyz_tool")
        normal = _require_mapping(part.get("normal"), f"functional_parts.{part_id}.normal")
        _require_unit_vec3(normal.get("vector_tool"), f"functional_parts.{part_id}.normal.vector_tool")
        if "axis" in part:
            axis = _require_mapping(part["axis"], f"functional_parts.{part_id}.axis")
            _require_unit_vec3(axis.get("vector_tool"), f"functional_parts.{part_id}.axis.vector_tool")
        if "tangent_long" in part:
            tangent_long = _require_mapping(part["tangent_long"], f"functional_parts.{part_id}.tangent_long")
            _require_unit_vec3(
                tangent_long.get("vector_tool"),
                f"functional_parts.{part_id}.tangent_long.vector_tool",
            )
        if "tangent_short" in part:
            tangent_short = _require_mapping(part["tangent_short"], f"functional_parts.{part_id}.tangent_short")
            _require_unit_vec3(
                tangent_short.get("vector_tool"),
                f"functional_parts.{part_id}.tangent_short.vector_tool",
            )
        frame = _require_mapping(part.get("frame"), f"functional_parts.{part_id}.frame")
        frame_origin = _number_list(frame.get("origin"), 3, f"functional_parts.{part_id}.frame.origin")
        frame_axes = {
            key: _require_unit_vec3(frame.get(key), f"functional_parts.{part_id}.frame.{key}")
            for key in ("x_axis", "y_axis", "z_axis")
        }
        for a_key, b_key in (("x_axis", "y_axis"), ("x_axis", "z_axis"), ("y_axis", "z_axis")):
            if abs(_dot(frame_axes[a_key], frame_axes[b_key])) > 1e-3:
                raise SchemaValidationError(f"functional_parts.{part_id}.frame axes must be orthogonal")
        if _norm([frame_origin[i] - center_xyz[i] for i in range(3)]) > 1e-6:
            raise SchemaValidationError(
                f"functional_parts.{part_id}.frame.origin must match center.xyz_tool"
            )
        transform = part.get("T_tool_functional")
        if not isinstance(transform, list) or len(transform) != 4:
            raise SchemaValidationError(f"functional_parts.{part_id}.T_tool_functional must be 4x4")
        for row_i, row in enumerate(transform):
            _number_list(row, 4, f"functional_parts.{part_id}.T_tool_functional[{row_i}]")
        if transform[3] != [0.0, 0.0, 0.0, 1.0]:
            raise SchemaValidationError(f"functional_parts.{part_id}.T_tool_functional last row is invalid")
        translation = [float(transform[i][3]) for i in range(3)]
        if _norm([translation[i] - center_xyz[i] for i in range(3)]) > 1e-6:
            raise SchemaValidationError(
                f"functional_parts.{part_id}.T_tool_functional translation must match center.xyz_tool"
            )

    grasps = _require_mapping(data["grasp_references"], "grasp_references")
    if not grasps:
        raise SchemaValidationError("grasp_references must be non-empty")
    for grasp_id, grasp in grasps.items():
        grasp = _require_mapping(grasp, f"grasp_references.{grasp_id}")
        _grounding_metadata(grasp, f"grasp_references.{grasp_id}")
        _number_list(grasp.get("position_tool"), 3, f"grasp_references.{grasp_id}.position_tool")
        _quat(grasp.get("orientation_xyzw"), f"grasp_references.{grasp_id}.orientation_xyzw")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("geometry_mode") not in {"observed", "prior", "mixed"}:
        raise SchemaValidationError("metadata.geometry_mode is unsupported")


def validate_tool_part_graph(data: Any) -> None:
    data = _require_mapping(data, "ToolPartGraph")
    required = {"schema_version", "tool_id", "tool_frame", "parts", "relations", "metadata"}
    _check_keys(data, required=required, allowed=required, name="ToolPartGraph")
    _expect_const(data, "schema_version", "v0.1")
    _expect_const(data, "tool_frame", "tool")
    _non_empty_string(data["tool_id"], "tool_id")

    parts = _require_mapping(data["parts"], "parts")
    if not parts:
        raise SchemaValidationError("parts must be non-empty")
    for part_id, part in parts.items():
        part = _require_mapping(part, f"parts.{part_id}")
        _grounding_metadata(part, f"parts.{part_id}")
        _non_empty_string(part.get("label"), f"parts.{part_id}.label")
        roles = part.get("role_candidates")
        if not isinstance(roles, list) or not roles:
            raise SchemaValidationError(f"parts.{part_id}.role_candidates must be non-empty")
        for role_i, role in enumerate(roles):
            _non_empty_string(role, f"parts.{part_id}.role_candidates[{role_i}]")
        _non_empty_string(part.get("mask_ref"), f"parts.{part_id}.mask_ref")
        if not isinstance(part.get("point_count"), int) or part["point_count"] < 1:
            raise SchemaValidationError(f"parts.{part_id}.point_count must be a positive int")
        _number_list(part.get("centroid_xyz_tool"), 3, f"parts.{part_id}.centroid_xyz_tool")
        obb = _require_mapping(part.get("obb"), f"parts.{part_id}.obb")
        _number_list(obb.get("center_xyz_tool"), 3, f"parts.{part_id}.obb.center_xyz_tool")
        extent = _number_list(obb.get("extent_xyz"), 3, f"parts.{part_id}.obb.extent_xyz")
        if any(item <= 0.0 for item in extent):
            raise SchemaValidationError(f"parts.{part_id}.obb.extent_xyz must be positive")
        axes = _require_mapping(obb.get("axes_tool"), f"parts.{part_id}.obb.axes_tool")
        obb_axes = {
            key: _require_unit_vec3(axes.get(key), f"parts.{part_id}.obb.axes_tool.{key}")
            for key in ("x", "y", "z")
        }
        for a_key, b_key in (("x", "y"), ("x", "z"), ("y", "z")):
            if abs(_dot(obb_axes[a_key], obb_axes[b_key])) > 1e-3:
                raise SchemaValidationError(f"parts.{part_id}.obb axes must be orthogonal")
        patches = part.get("surface_patches", {})
        if patches is not None:
            patches = _require_mapping(patches, f"parts.{part_id}.surface_patches")
            for patch_id, patch in patches.items():
                patch = _require_mapping(patch, f"parts.{part_id}.surface_patches.{patch_id}")
                _grounding_metadata(patch, f"parts.{part_id}.surface_patches.{patch_id}")
                _non_empty_string(
                    patch.get("role_hint"),
                    f"parts.{part_id}.surface_patches.{patch_id}.role_hint",
                )
                _number_list(
                    patch.get("center_xyz_tool"),
                    3,
                    f"parts.{part_id}.surface_patches.{patch_id}.center_xyz_tool",
                )
                _require_unit_vec3(
                    patch.get("normal_tool"),
                    f"parts.{part_id}.surface_patches.{patch_id}.normal_tool",
                )

    relations = data["relations"]
    if not isinstance(relations, list):
        raise SchemaValidationError("relations must be a list")
    for index, relation in enumerate(relations):
        relation = _require_mapping(relation, f"relations[{index}]")
        for key in ("type", "subject", "object"):
            _non_empty_string(relation.get(key), f"relations[{index}].{key}")
        for key in ("subject", "object"):
            ref = relation[key]
            if ref.startswith("tool.parts.") and ref.split(".")[-1] not in parts:
                raise SchemaValidationError(f"relations[{index}].{key} does not resolve")
        if "confidence" in relation:
            _confidence(relation["confidence"], f"relations[{index}].confidence")
        if "source" in relation:
            _non_empty_string(relation["source"], f"relations[{index}].source")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("geometry_mode") not in {"observed", "prior", "mixed"}:
        raise SchemaValidationError("metadata.geometry_mode is unsupported")
    if metadata.get("segmentation_mode") not in {
        "sampart3d",
        "p3sam",
        "grounded_sam_part",
        "observed_tool_crop",
        "mesh_prior_adapter",
        "mixed",
    }:
        raise SchemaValidationError("metadata.segmentation_mode is unsupported")


def validate_grounding_candidates(data: Any) -> None:
    data = _require_mapping(data, "GroundingCandidates")
    required = {
        "schema_version",
        "grounding_id",
        "observation_id",
        "frames",
        "marks",
        "points",
        "regions",
        "axes",
        "surfaces",
        "parts",
        "metadata",
    }
    allowed = set(required) | {"edges"}
    _check_keys(data, required=required, allowed=allowed, name="GroundingCandidates")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["grounding_id"], "grounding_id")
    _non_empty_string(data["observation_id"], "observation_id")
    _require_mapping(data["frames"], "frames")

    def check_entity(entity: Any, name: str) -> dict[str, Any]:
        entity = _require_mapping(entity, name)
        _non_empty_string(entity.get("label"), f"{name}.label")
        _grounding_metadata(entity, name)
        return entity

    for mark_id, mark in _require_mapping(data["marks"], "marks").items():
        mark = check_entity(mark, f"marks.{mark_id}")
        _number_list(mark.get("pixel"), 2, f"marks.{mark_id}.pixel")
    for point_id, point in _require_mapping(data["points"], "points").items():
        point = check_entity(point, f"points.{point_id}")
        if "pixel" in point:
            _number_list(point["pixel"], 2, f"points.{point_id}.pixel")
        if "xyz_world" in point:
            _number_list(point["xyz_world"], 3, f"points.{point_id}.xyz_world")
        if "xyz_tool" in point:
            _number_list(point["xyz_tool"], 3, f"points.{point_id}.xyz_tool")
    for region_id, region in _require_mapping(data["regions"], "regions").items():
        region = check_entity(region, f"regions.{region_id}")
        if "center_xyz_world" in region:
            _number_list(region["center_xyz_world"], 3, f"regions.{region_id}.center_xyz_world")
        if "center_xyz_tool" in region:
            _number_list(region["center_xyz_tool"], 3, f"regions.{region_id}.center_xyz_tool")
        if "extent_xyz" in region:
            extent = _number_list(region["extent_xyz"], 3, f"regions.{region_id}.extent_xyz")
            if any(item <= 0.0 for item in extent):
                raise SchemaValidationError(f"regions.{region_id}.extent_xyz must be positive")
        if "extent_xyz_tool" in region:
            extent = _number_list(region["extent_xyz_tool"], 3, f"regions.{region_id}.extent_xyz_tool")
            if any(item <= 0.0 for item in extent):
                raise SchemaValidationError(f"regions.{region_id}.extent_xyz_tool must be positive")
    for axis_id, axis in _require_mapping(data["axes"], "axes").items():
        axis = check_entity(axis, f"axes.{axis_id}")
        if "vector_world" not in axis and "vector_tool" not in axis:
            raise SchemaValidationError(f"axes.{axis_id} must include vector_world or vector_tool")
        if "vector_world" in axis:
            _require_unit_vec3(axis.get("vector_world"), f"axes.{axis_id}.vector_world")
        if "vector_tool" in axis:
            _require_unit_vec3(axis.get("vector_tool"), f"axes.{axis_id}.vector_tool")
        if "origin_xyz_world" in axis:
            _number_list(axis["origin_xyz_world"], 3, f"axes.{axis_id}.origin_xyz_world")
        if "origin_xyz_tool" in axis:
            _number_list(axis["origin_xyz_tool"], 3, f"axes.{axis_id}.origin_xyz_tool")
    for surface_id, surface in _require_mapping(data["surfaces"], "surfaces").items():
        surface = check_entity(surface, f"surfaces.{surface_id}")
        if "point_xyz_world" in surface:
            _number_list(surface["point_xyz_world"], 3, f"surfaces.{surface_id}.point_xyz_world")
        if "point_xyz_tool" in surface:
            _number_list(surface["point_xyz_tool"], 3, f"surfaces.{surface_id}.point_xyz_tool")
        if "normal_world" in surface:
            _require_unit_vec3(surface["normal_world"], f"surfaces.{surface_id}.normal_world")
        if "normal_tool" in surface:
            _require_unit_vec3(surface["normal_tool"], f"surfaces.{surface_id}.normal_tool")
    for part_id, part in _require_mapping(data["parts"], "parts").items():
        part = check_entity(part, f"parts.{part_id}")
        roles = part.get("role_candidates", [])
        if roles is not None:
            if not isinstance(roles, list):
                raise SchemaValidationError(f"parts.{part_id}.role_candidates must be a list")
            for role_i, role in enumerate(roles):
                _non_empty_string(role, f"parts.{part_id}.role_candidates[{role_i}]")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("grounding_mode") not in {"model_grounded", "model_grounded_with_prior", "mixed"}:
        raise SchemaValidationError("metadata.grounding_mode is unsupported")


PROVIDER_TYPES = {
    "perception",
    "tool_prior",
    "environment",
    "execution_feedback",
    "simulation",
}
PROVIDER_SOURCE_KINDS = {
    "OBSERVED_RGBD",
    "MODEL_MASK",
    "MODEL_SEMANTIC",
    "REGISTRATION_PRIOR",
    "MESH_PRIOR",
    "TRACKED_POSE",
    "MARKER_OR_MOCAP",
    "ENVIRONMENT_CONFIG",
    "SIMULATOR_ORACLE_DEBUG",
}
PROVIDER_CAPABILITIES = {
    "rgbd",
    "segmentation",
    "mask_tracking",
    "pose_tracking",
    "tool_functional_parts",
    "target_geometry",
    "sdf",
    "collision_geometry",
    "execution_feedback",
    "environment_setup",
}
PROVIDER_FRAME_CONTRACTS = {
    "temporary_observed_tool_frame",
    "registered_tool_frame",
    "simtoolreal_asset_root",
    "debug_oracle_asset_root",
    "world_scene_frame",
    "target_local_frame",
}


VISUAL_EVIDENCE_KINDS = {
    "rgb",
    "depth",
    "mask_overlay",
    "contact_sheet",
    "goal_preview",
    "whole_mesh_part_choice_overview",
    "isolated_part_choice_overview",
    "isolated_part_card",
    "selected_part_bbox_overview",
    "selected_part_face_highlight",
    "candidate_part_face_highlight",
    "candidate_part_contact_point_highlight",
    "candidate_part_motion_axis_highlight",
    "rollout_frame",
    "rollout_video",
    "report",
    "other",
}
VISUAL_EVIDENCE_MEDIA_TYPES = {"image", "video", "json", "text", "other"}
VISUAL_EVIDENCE_ALLOWED_USES = {"planning_context", "repair_diagnosis", "debug_only"}
VISUAL_EVIDENCE_ITEM_KEYS = {
    "id",
    "kind",
    "media_type",
    "path",
    "description",
    "provider_id",
    "source_stage",
    "allowed_use",
    "part_marker",
    "face_marker",
    "point_ref",
    "surface_ref",
    "patch_id",
}
TOOL_PART_SEMANTIC_LABEL_MODES = {
    "vlm_part_semantic_aliases",
    "provider_supplied_part_semantic_aliases",
    "deterministic_unlabeled_scaffold",
}


def _validate_ref(value: Any, name: str) -> None:
    if not isinstance(value, str) or not any(regex.match(value) for regex in GROUNDED_REF_RES):
        raise SchemaValidationError(f"{name} must be a symbol reference, got {value!r}")


def validate_program_ir(data: Any) -> None:
    data = _require_mapping(data, "ProgramIR")
    required = {
        "schema_version",
        "program_id",
        "task_instruction",
        "api_version",
        "stages",
        "constraints",
        "objectives",
        "diagnostics",
    }
    _check_keys(data, required=required, allowed=required, name="ProgramIR")
    _expect_const(data, "schema_version", "v0.1")
    if data.get("api_version") not in PROGRAM_API_VERSIONS:
        raise SchemaValidationError(f"api_version must be one of {sorted(PROGRAM_API_VERSIONS)}")
    for key in ("program_id", "task_instruction"):
        _non_empty_string(data[key], key)
    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise SchemaValidationError("stages must be a non-empty list")
    stage_names: set[str] = set()
    for index, stage in enumerate(stages):
        stage = _require_mapping(stage, f"stages[{index}]")
        _non_empty_string(stage.get("name"), f"stages[{index}].name")
        if stage["name"] in stage_names:
            raise SchemaValidationError(f"duplicate stage: {stage['name']}")
        stage_names.add(stage["name"])
        if "repeat" in stage and (not isinstance(stage["repeat"], int) or stage["repeat"] < 1):
            raise SchemaValidationError(f"stages[{index}].repeat must be a positive int")
    constraints = data["constraints"]
    if not isinstance(constraints, list):
        raise SchemaValidationError("constraints must be a list")
    objectives = data["objectives"]
    if not isinstance(objectives, list):
        raise SchemaValidationError("objectives must be a list")
    if not constraints and not objectives:
        raise SchemaValidationError("ProgramIR must include at least one constraint or objective")
    for index, constraint in enumerate(constraints):
        constraint = _require_mapping(constraint, f"constraints[{index}]")
        for key in ("id", "type", "stage", "refs", "params", "hard"):
            if key not in constraint:
                raise SchemaValidationError(f"constraints[{index}] missing {key}")
        if constraint["stage"] not in stage_names:
            raise SchemaValidationError(f"constraints[{index}].stage is not declared")
        _non_empty_string(constraint["id"], f"constraints[{index}].id")
        _non_empty_string(constraint["type"], f"constraints[{index}].type")
        if constraint["type"] not in {
            "contact",
            "distance",
            "align_axis",
            "align_vector_between_points",
            "pre_contact_offset",
            "clearance",
            "move_along",
            "tool_occupancy_clearance",
        }:
            raise SchemaValidationError(f"constraints[{index}].type is unsupported")
        refs = _require_mapping(constraint["refs"], f"constraints[{index}].refs")
        for key, ref in refs.items():
            _non_empty_string(key, f"constraints[{index}].refs key")
            _validate_ref(ref, f"constraints[{index}].refs.{key}")
        params = _require_mapping(constraint["params"], f"constraints[{index}].params")
        ctype = constraint["type"]
        expected_ref_keys = {
            "contact": {"source", "target"},
            "distance": {"a", "b"},
            "align_axis": {"source_axis", "target_axis"},
            "align_vector_between_points": {"source_start", "source_end", "target_axis"},
            "pre_contact_offset": {"source", "target", "axis"},
            "clearance": {"moving", "obstacle"},
            "move_along": {"source", "start", "end"},
            "tool_occupancy_clearance": {"tool_part", "obstacle"},
        }[ctype]
        allowed_ref_keys = set(expected_ref_keys)
        if ctype == "move_along":
            allowed_ref_keys.add("swing_axis")
        _check_keys(
            refs,
            required=expected_ref_keys,
            allowed=allowed_ref_keys,
            name=f"constraints[{index}].refs",
        )
        if ctype == "align_vector_between_points" and refs["source_start"] == refs["source_end"]:
            raise SchemaValidationError(
                f"constraints[{index}].refs source_start and source_end must be distinct"
            )
        if ctype == "contact":
            _check_keys(params, required={"tolerance_m"}, allowed={"tolerance_m"}, name=f"constraints[{index}].params")
            tolerance = _finite_number(params["tolerance_m"], f"constraints[{index}].params.tolerance_m")
            if tolerance < 0.0 or tolerance > 0.5:
                raise SchemaValidationError(f"constraints[{index}].params.tolerance_m out of bounds")
        elif ctype == "distance":
            _check_keys(
                params,
                required={"target_m", "tolerance_m"},
                allowed={"target_m", "tolerance_m"},
                name=f"constraints[{index}].params",
            )
            target_m = _finite_number(params["target_m"], f"constraints[{index}].params.target_m")
            tolerance = _finite_number(params["tolerance_m"], f"constraints[{index}].params.tolerance_m")
            if target_m < 0.0 or target_m > 5.0 or tolerance < 0.0 or tolerance > 0.5:
                raise SchemaValidationError(f"constraints[{index}].params distance values out of bounds")
        elif ctype in {"align_axis", "align_vector_between_points"}:
            _check_keys(
                params,
                required={"polarity", "min_cosine"},
                allowed={"polarity", "min_cosine"},
                name=f"constraints[{index}].params",
            )
            if params["polarity"] not in {"same", "opposite", "auto"}:
                raise SchemaValidationError(f"constraints[{index}].params.polarity is unsupported")
            min_cosine = _finite_number(params["min_cosine"], f"constraints[{index}].params.min_cosine")
            if min_cosine < -1.0 or min_cosine > 1.0:
                raise SchemaValidationError(f"constraints[{index}].params.min_cosine out of bounds")
        elif ctype == "pre_contact_offset":
            _check_keys(
                params,
                required={"distance_m", "direction"},
                allowed={"distance_m", "direction"},
                name=f"constraints[{index}].params",
            )
            distance_m = _finite_number(params["distance_m"], f"constraints[{index}].params.distance_m")
            if distance_m < 0.0 or distance_m > 5.0:
                raise SchemaValidationError(f"constraints[{index}].params.distance_m out of bounds")
            if params["direction"] not in {"opposite_axis", "along_axis"}:
                raise SchemaValidationError(f"constraints[{index}].params.direction is unsupported")
        elif ctype == "clearance":
            _check_keys(
                params,
                required={"min_distance_m"},
                allowed={"min_distance_m"},
                name=f"constraints[{index}].params",
            )
            clearance = _finite_number(params["min_distance_m"], f"constraints[{index}].params.min_distance_m")
            if clearance < 0.0 or clearance > 5.0:
                raise SchemaValidationError(f"constraints[{index}].params.min_distance_m out of bounds")
        elif ctype == "move_along":
            _check_keys(
                params,
                required=set(),
                allowed={"profile", "path_family", "swing_angle_rad", "swing_lift_m"},
                name=f"constraints[{index}].params",
            )
            profile = params.get("profile", "linear")
            if profile not in {"linear", "swing_arc"}:
                raise SchemaValidationError(f"constraints[{index}].params.profile is unsupported")
            path_family = params.get("path_family", "swing" if profile == "swing_arc" else "contact_follow")
            if path_family not in {"contact_follow", "slide", "swing"}:
                raise SchemaValidationError(f"constraints[{index}].params.path_family is unsupported")
            if profile == "swing_arc":
                if "swing_axis" not in refs:
                    raise SchemaValidationError(f"constraints[{index}].refs.swing_axis is required for swing_arc")
                swing_angle = _finite_number(
                    params.get("swing_angle_rad", 0.55),
                    f"constraints[{index}].params.swing_angle_rad",
                )
                if swing_angle < 0.01 or swing_angle > 1.57:
                    raise SchemaValidationError(f"constraints[{index}].params.swing_angle_rad out of bounds")
                swing_lift = _finite_number(
                    params.get("swing_lift_m", 0.03),
                    f"constraints[{index}].params.swing_lift_m",
                )
                if swing_lift < 0.0 or swing_lift > 0.25:
                    raise SchemaValidationError(f"constraints[{index}].params.swing_lift_m out of bounds")
            elif "swing_angle_rad" in params:
                raise SchemaValidationError(
                    f"constraints[{index}].params.swing_angle_rad requires swing_arc profile"
                )
            elif "swing_lift_m" in params:
                raise SchemaValidationError(
                    f"constraints[{index}].params.swing_lift_m requires swing_arc profile"
                )
        elif ctype == "tool_occupancy_clearance":
            _check_keys(
                params,
                required={"min_distance_m", "samples"},
                allowed={"min_distance_m", "samples"},
                name=f"constraints[{index}].params",
            )
            clearance = _finite_number(params["min_distance_m"], f"constraints[{index}].params.min_distance_m")
            if clearance < 0.0 or clearance > 1.0:
                raise SchemaValidationError(f"constraints[{index}].params.min_distance_m out of bounds")
            if params["samples"] not in {"centroid", "obb_corners", "surface_patch_centers"}:
                raise SchemaValidationError(f"constraints[{index}].params.samples is unsupported")
        if not isinstance(constraint["hard"], bool):
            raise SchemaValidationError(f"constraints[{index}].hard must be bool")
        if "weight" in constraint:
            weight = _finite_number(constraint["weight"], f"constraints[{index}].weight")
            if weight < 0.0 or weight > 1000.0:
                raise SchemaValidationError(f"constraints[{index}].weight out of bounds")
    for index, objective in enumerate(objectives):
        objective = _require_mapping(objective, f"objectives[{index}]")
        for key in ("id", "stage", "kind", "function_name", "weight"):
            if key not in objective:
                raise SchemaValidationError(f"objectives[{index}] missing {key}")
        if objective["stage"] not in stage_names:
            raise SchemaValidationError(f"objectives[{index}].stage is not declared")
        if objective["kind"] not in {"subgoal", "path"}:
            raise SchemaValidationError(f"objectives[{index}].kind is unsupported")
        _non_empty_string(objective["function_name"], f"objectives[{index}].function_name")
        if _finite_number(objective["weight"], f"objectives[{index}].weight") < 0.0:
            raise SchemaValidationError(f"objectives[{index}].weight must be non-negative")
        if "refs" not in objective:
            raise SchemaValidationError(f"objectives[{index}] missing refs")
        refs = objective["refs"]
        if not isinstance(refs, list):
            raise SchemaValidationError(f"objectives[{index}].refs must be a list")
        if not refs:
            raise SchemaValidationError(f"objectives[{index}].refs must be non-empty")
        seen_refs: set[str] = set()
        for ref_index, ref in enumerate(refs):
            _validate_ref(ref, f"objectives[{index}].refs[{ref_index}]")
            if ref in seen_refs:
                raise SchemaValidationError(f"objectives[{index}].refs contains duplicates")
            seen_refs.add(ref)
    diagnostics = _require_mapping(data["diagnostics"], "diagnostics")
    if diagnostics.get("ast_checked") is not True:
        raise SchemaValidationError("diagnostics.ast_checked must be true")
    if diagnostics.get("executed") is not True:
        raise SchemaValidationError("diagnostics.executed must be true")


def validate_tool_pose_subgoals(data: Any) -> None:
    data = _require_mapping(data, "ToolPoseSubgoals")
    required = {
        "schema_version",
        "subgoals_id",
        "tool_id",
        "task_id",
        "world_frame",
        "units",
        "quaternion_order",
        "start_pose",
        "goals",
        "source_trace",
        "sanity_checks",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="ToolPoseSubgoals")
    _expect_const(data, "schema_version", "v0.2")
    _expect_const(data, "world_frame", "world")
    _expect_const(data, "units", "meters")
    _expect_const(data, "quaternion_order", "xyzw")
    for key in ("subgoals_id", "tool_id", "task_id"):
        _non_empty_string(data[key], key)
    _pose_object(data["start_pose"], "start_pose")
    goals = data["goals"]
    if not isinstance(goals, list) or not goals:
        raise SchemaValidationError("goals must be a non-empty list")
    for expected_index, goal in enumerate(goals):
        goal = _require_mapping(goal, f"goals[{expected_index}]")
        required_goal = {
            "index",
            "stage",
            "pose_world_tool",
            "duration_sec",
            "tolerance",
            "source_constraints",
            "expected_contact",
            "contact_point_goal",
        }
        _check_keys(
            goal,
            required_goal,
            required_goal | {"semantic_gate", "semantic_role"},
            f"goals[{expected_index}]",
        )
        if goal["index"] != expected_index:
            raise SchemaValidationError(f"goals[{expected_index}].index must be {expected_index}")
        _non_empty_string(goal["stage"], f"goals[{expected_index}].stage")
        if "semantic_gate" in goal and not isinstance(goal["semantic_gate"], bool):
            raise SchemaValidationError(f"goals[{expected_index}].semantic_gate must be bool")
        if "semantic_role" in goal:
            _non_empty_string(goal["semantic_role"], f"goals[{expected_index}].semantic_role")
        _pose_object(goal["pose_world_tool"], f"goals[{expected_index}].pose_world_tool")
        if _finite_number(goal["duration_sec"], f"goals[{expected_index}].duration_sec") <= 0.0:
            raise SchemaValidationError(f"goals[{expected_index}].duration_sec must be positive")
        tolerance = _require_mapping(goal["tolerance"], f"goals[{expected_index}].tolerance")
        position_tol = _finite_number(
            tolerance.get("position_m"),
            f"goals[{expected_index}].tolerance.position_m",
        )
        rotation_tol = _finite_number(
            tolerance.get("rotation_rad"),
            f"goals[{expected_index}].tolerance.rotation_rad",
        )
        if position_tol < 0.0 or rotation_tol < 0.0:
            raise SchemaValidationError(f"goals[{expected_index}].tolerance values must be non-negative")
        source_constraints = goal["source_constraints"]
        if not isinstance(source_constraints, list) or not source_constraints:
            raise SchemaValidationError(f"goals[{expected_index}].source_constraints must be non-empty")
        for item in source_constraints:
            _non_empty_string(item, f"goals[{expected_index}].source_constraints[]")
        expected_contact = _require_mapping(
            goal["expected_contact"],
            f"goals[{expected_index}].expected_contact",
        )
        _non_empty_string(expected_contact.get("tool_part"), f"goals[{expected_index}].expected_contact.tool_part")
        _non_empty_string(
            expected_contact.get("scene_target"),
            f"goals[{expected_index}].expected_contact.scene_target",
        )
        contact_point_goal = _require_mapping(
            goal["contact_point_goal"],
            f"goals[{expected_index}].contact_point_goal",
        )
        _validate_ref(contact_point_goal.get("part_ref"), f"goals[{expected_index}].contact_point_goal.part_ref")
        _number_list(
            contact_point_goal.get("position_world"),
            3,
            f"goals[{expected_index}].contact_point_goal.position_world",
        )
        _quat(
            contact_point_goal.get("orientation_xyzw"),
            f"goals[{expected_index}].contact_point_goal.orientation_xyzw",
        )

    trace = _require_mapping(data["source_trace"], "source_trace")
    for key in (
        "observed_scene_graph",
        "grounding_candidates",
        "functional_tool_parts",
        "planner_program",
        "program_ir",
    ):
        _artifact_ref(trace.get(key), f"source_trace.{key}")
    _require_mapping(data["sanity_checks"], "sanity_checks")
    metadata = _require_mapping(data["metadata"], "metadata")
    for key in ("generated_by", "source", "geometry_mode"):
        _non_empty_string(metadata.get(key), f"metadata.{key}")
    if metadata["geometry_mode"] not in {"observed", "mixed"}:
        raise SchemaValidationError("metadata.geometry_mode is unsupported")


def validate_simtoolreal_fixed_goals(data: Any) -> None:
    data = _require_mapping(data, "SimToolRealFixedGoals")
    _check_keys(
        data,
        required={"start_pose", "goals", "metadata"},
        allowed={"start_pose", "goals", "metadata"},
        name="SimToolRealFixedGoals",
    )
    _number_list(data["start_pose"], 7, "start_pose")
    _quat(data["start_pose"][3:7], "start_pose[3:7]")
    goals = data["goals"]
    if not isinstance(goals, list) or not goals:
        raise SchemaValidationError("goals must be a non-empty list")
    for i, goal in enumerate(goals):
        pose = _number_list(goal, 7, f"goals[{i}]")
        _quat(pose[3:7], f"goals[{i}][3:7]")
    metadata = _require_mapping(data["metadata"], "metadata")
    required = {
        "schema_version",
        "source_subgoals_id",
        "target_api",
        "coordinate_frame",
        "units",
        "quaternion_order",
        "generated_by",
    }
    _check_keys(
        metadata,
        required=required,
        allowed=required
        | {
            "goal_index_map",
            "start_goal_in_fixed_goals",
            "skipped_initial_goals",
            "task_terminal_source_goal_index",
            "task_terminal_stage",
            "truncated_post_terminal_goals",
        },
        name="metadata",
    )
    _expect_const(metadata, "schema_version", "v0.1")
    _expect_const(metadata, "target_api", "fixed_goal_states_json")
    _expect_const(metadata, "coordinate_frame", "world")
    _expect_const(metadata, "units", "meters")
    _expect_const(metadata, "quaternion_order", "xyzw")
    _expect_const(metadata, "generated_by", "parttoolgen_simtoolreal_adapter")
    goal_index_map = metadata.get("goal_index_map")
    if goal_index_map is not None:
        if not isinstance(goal_index_map, list):
            raise SchemaValidationError("metadata.goal_index_map must be a list")
        for i, item in enumerate(goal_index_map):
            entry = _require_mapping(item, f"metadata.goal_index_map[{i}]")
            _check_keys(
                entry,
                required={"fixed_goal_index", "source_goal_index", "stage", "source"},
                allowed={"fixed_goal_index", "source_goal_index", "stage", "source"},
                name=f"metadata.goal_index_map[{i}]",
            )
            if not isinstance(entry["fixed_goal_index"], int):
                raise SchemaValidationError(
                    f"metadata.goal_index_map[{i}].fixed_goal_index must be an integer"
                )
            if entry["source_goal_index"] is not None and not isinstance(
                entry["source_goal_index"],
                int,
            ):
                raise SchemaValidationError(
                    f"metadata.goal_index_map[{i}].source_goal_index must be an integer or null"
                )
            _non_empty_string(entry["stage"], f"metadata.goal_index_map[{i}].stage")
            _non_empty_string(entry["source"], f"metadata.goal_index_map[{i}].source")
    terminal_index = metadata.get("task_terminal_source_goal_index")
    if terminal_index is not None and not isinstance(terminal_index, int):
        raise SchemaValidationError(
            "metadata.task_terminal_source_goal_index must be an integer or null"
        )
    terminal_stage = metadata.get("task_terminal_stage")
    if terminal_stage is not None:
        _non_empty_string(terminal_stage, "metadata.task_terminal_stage")
    truncated = metadata.get("truncated_post_terminal_goals")
    if truncated is not None:
        if not isinstance(truncated, list):
            raise SchemaValidationError("metadata.truncated_post_terminal_goals must be a list")
        for i, item in enumerate(truncated):
            entry = _require_mapping(item, f"metadata.truncated_post_terminal_goals[{i}]")
            _check_keys(
                entry,
                required={"source_goal_index", "stage", "reason"},
                allowed={"source_goal_index", "stage", "reason"},
                name=f"metadata.truncated_post_terminal_goals[{i}]",
            )
            if not isinstance(entry["source_goal_index"], int):
                raise SchemaValidationError(
                    f"metadata.truncated_post_terminal_goals[{i}].source_goal_index must be an integer"
                )
            _non_empty_string(entry["stage"], f"metadata.truncated_post_terminal_goals[{i}].stage")
            _non_empty_string(entry["reason"], f"metadata.truncated_post_terminal_goals[{i}].reason")
