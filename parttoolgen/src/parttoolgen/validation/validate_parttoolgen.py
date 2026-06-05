#!/usr/bin/env python3
"""Strict validators for the PartToolGen runtime artifacts."""

from __future__ import annotations

import argparse
import ast
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


def _check_required(data: dict[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise SchemaValidationError(f"{name} missing required fields: {missing}")


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


def _string_list(value: Any, name: str, *, min_items: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < min_items:
        raise SchemaValidationError(f"{name} must be a list with at least {min_items} items")
    result: list[str] = []
    for index, item in enumerate(value):
        _non_empty_string(item, f"{name}[{index}]")
        result.append(str(item))
    return result


def _contains_forbidden_debug_marker(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return "debug" in lowered or "manual_authored" in lowered or "stub" in lowered
    if isinstance(value, dict):
        return any(_contains_forbidden_debug_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_debug_marker(item) for item in value)
    return False


def _contains_forbidden_semantic_query_geometry(value: Any, path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if str(key) in SEMANTIC_QUERY_FORBIDDEN_KEYS:
                return key_path
            if str(key).startswith("T_"):
                return key_path
            found = _contains_forbidden_semantic_query_geometry(item, key_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _contains_forbidden_semantic_query_geometry(item, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_semantic_query_artifact(data: Any) -> None:
    data = _require_mapping(data, "SemanticQueryArtifact")
    forbidden = _contains_forbidden_semantic_query_geometry(data)
    if forbidden:
        raise SchemaValidationError(
            f"SemanticQueryArtifact must not contain geometry/pose field: {forbidden}"
        )
    required = {
        "schema_version",
        "query_artifact_id",
        "observation_id",
        "task_instruction",
        "queries",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="SemanticQueryArtifact")
    _expect_const(data, "schema_version", "v0.1")
    for key in ("query_artifact_id", "observation_id", "task_instruction"):
        _non_empty_string(data[key], key)

    queries = data["queries"]
    if not isinstance(queries, list) or not queries:
        raise SchemaValidationError("queries must be a non-empty list")
    seen_ids: set[str] = set()
    allowed_query_keys = {
        "id",
        "query_string",
        "scope",
        "roles",
        "expected_geometry",
        "priority",
        "category_hint",
        "functional",
        "semantic_confidence",
        "constraints",
    }
    required_query_keys = {
        "id",
        "query_string",
        "scope",
        "roles",
        "expected_geometry",
        "priority",
    }
    for index, query in enumerate(queries):
        query = _require_mapping(query, f"queries[{index}]")
        _check_keys(query, required=required_query_keys, allowed=allowed_query_keys, name=f"queries[{index}]")
        query_id = query["id"]
        if not isinstance(query_id, str) or not SEMANTIC_QUERY_ID_RE.match(query_id):
            raise SchemaValidationError(f"queries[{index}].id is not a symbolic id")
        if query_id in seen_ids:
            raise SchemaValidationError(f"duplicate semantic query id: {query_id}")
        seen_ids.add(query_id)
        _non_empty_string(query["query_string"], f"queries[{index}].query_string")
        if len(query["query_string"]) > 160:
            raise SchemaValidationError(f"queries[{index}].query_string is too long")
        lowered_query = query["query_string"].lower()
        if "ctx.ref" in lowered_query or any(
            namespace in lowered_query
            for namespace in ("scene.", "candidate.", "tool.", "target.")
        ):
            raise SchemaValidationError(
                f"queries[{index}].query_string must not contain symbolic refs"
            )
        if query["scope"] not in SEMANTIC_QUERY_SCOPES:
            raise SchemaValidationError(f"queries[{index}].scope is unsupported")
        roles = query["roles"]
        if not isinstance(roles, list) or not roles:
            raise SchemaValidationError(f"queries[{index}].roles must be a non-empty list")
        for role_index, role in enumerate(roles):
            if not isinstance(role, str) or not SEMANTIC_QUERY_ID_RE.match(role):
                raise SchemaValidationError(f"queries[{index}].roles[{role_index}] is invalid")
            if len(role) > 64:
                raise SchemaValidationError(f"queries[{index}].roles[{role_index}] is too long")
        expected = query["expected_geometry"]
        if not isinstance(expected, list) or not expected:
            raise SchemaValidationError(f"queries[{index}].expected_geometry must be a non-empty list")
        for geometry_index, geometry in enumerate(expected):
            if geometry not in SEMANTIC_QUERY_GEOMETRY:
                raise SchemaValidationError(
                    f"queries[{index}].expected_geometry[{geometry_index}] is unsupported"
                )
        if not isinstance(query["priority"], int) or query["priority"] < 1 or query["priority"] > 100:
            raise SchemaValidationError(f"queries[{index}].priority must be an int in [1, 100]")
        if "category_hint" in query and query["category_hint"] not in SEMANTIC_QUERY_CATEGORIES:
            raise SchemaValidationError(f"queries[{index}].category_hint is unsupported")
        if "functional" in query and not isinstance(query["functional"], bool):
            raise SchemaValidationError(f"queries[{index}].functional must be bool")
        if "semantic_confidence" in query:
            _confidence(query["semantic_confidence"], f"queries[{index}].semantic_confidence")
        if "constraints" in query:
            constraints = query["constraints"]
            if not isinstance(constraints, list):
                raise SchemaValidationError(f"queries[{index}].constraints must be a list")
            for constraint_index, constraint in enumerate(constraints):
                if not isinstance(constraint, str) or not SEMANTIC_QUERY_ID_RE.match(constraint):
                    raise SchemaValidationError(
                        f"queries[{index}].constraints[{constraint_index}] is invalid"
                    )
                if len(constraint) > 64:
                    raise SchemaValidationError(
                        f"queries[{index}].constraints[{constraint_index}] is too long"
                    )

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("mode") != "vlm_authored_semantic_queries":
        raise SchemaValidationError("metadata.mode is unsupported")
    for key in ("model", "prompt_hash", "raw_response_path"):
        if key in metadata:
            _non_empty_string(metadata[key], f"metadata.{key}")


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


def validate_observation_bundle(data: Any) -> None:
    data = _require_mapping(data, "ObservationBundle")
    required = {
        "schema_version",
        "observation_id",
        "source_system",
        "capture_mode",
        "task_instruction",
        "frames",
        "cameras",
        "artifacts",
        "state",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="ObservationBundle")
    _expect_const(data, "schema_version", "v0.1")
    _expect_const(data, "source_system", "simtoolreal")
    _non_empty_string(data["observation_id"], "observation_id")
    if data["capture_mode"] not in {"rgbd"}:
        raise SchemaValidationError("capture_mode is unsupported")

    frames = _require_mapping(data["frames"], "frames")
    if "sim_world" not in frames:
        raise SchemaValidationError("frames must include sim_world")

    cameras = _require_mapping(data["cameras"], "cameras")
    for camera_id, camera in cameras.items():
        camera = _require_mapping(camera, f"cameras.{camera_id}")
        _non_empty_string(camera.get("frame"), f"cameras.{camera_id}.frame")
        width = camera.get("width")
        height = camera.get("height")
        if not isinstance(width, int) or width <= 0:
            raise SchemaValidationError(f"cameras.{camera_id}.width must be positive int")
        if not isinstance(height, int) or height <= 0:
            raise SchemaValidationError(f"cameras.{camera_id}.height must be positive int")

    artifacts = _require_mapping(data["artifacts"], "artifacts")
    for key in ("rgb", "depth", "segmentation", "masks", "camera", "state"):
        _non_empty_string(artifacts.get(key), f"artifacts.{key}")

    state = _require_mapping(data["state"], "state")
    _number_list(state.get("object_pose_xyzw"), 7, "state.object_pose_xyzw")
    _number_list(state.get("table_pose_xyzw"), 7, "state.table_pose_xyzw")
    if "goal_pose_xyzw" in state:
        _number_list(state["goal_pose_xyzw"], 7, "state.goal_pose_xyzw")
    _non_empty_string(state.get("object_name"), "state.object_name")
    _non_empty_string(state.get("table_asset"), "state.table_asset")

    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("simtoolreal_source_modified") is not False:
        raise SchemaValidationError("metadata.simtoolreal_source_modified must be false")


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


def validate_provider_contract(data: Any) -> None:
    data = _require_mapping(data, "PartToolGenProviderContract")
    required = {
        "schema_version",
        "provider_id",
        "provider_type",
        "source_kind",
        "deployable",
        "oracle",
        "capabilities",
        "frame_contracts",
        "provenance",
    }
    _check_required(data, required=required, name="PartToolGenProviderContract")
    _expect_const(data, "schema_version", "v0.1")
    for key in ("provider_id", "provider_type", "source_kind"):
        _non_empty_string(data[key], key)
    if data["provider_type"] not in PROVIDER_TYPES:
        raise SchemaValidationError(f"provider_type is unsupported: {data['provider_type']}")
    if data["source_kind"] not in PROVIDER_SOURCE_KINDS:
        raise SchemaValidationError(f"source_kind is unsupported: {data['source_kind']}")
    if not isinstance(data["deployable"], bool):
        raise SchemaValidationError("deployable must be bool")
    if not isinstance(data["oracle"], bool):
        raise SchemaValidationError("oracle must be bool")

    capabilities = _string_list(data["capabilities"], "capabilities", min_items=1)
    unsupported_capabilities = sorted(set(capabilities) - PROVIDER_CAPABILITIES)
    if unsupported_capabilities:
        raise SchemaValidationError(f"unsupported capabilities: {unsupported_capabilities}")
    frame_contracts = _string_list(data["frame_contracts"], "frame_contracts", min_items=1)
    unsupported_frames = sorted(set(frame_contracts) - PROVIDER_FRAME_CONTRACTS)
    if unsupported_frames:
        raise SchemaValidationError(f"unsupported frame contracts: {unsupported_frames}")

    provenance = _require_mapping(data["provenance"], "provenance")
    _non_empty_string(provenance.get("method"), "provenance.method")
    if "source_artifacts" in provenance:
        _string_list(provenance["source_artifacts"], "provenance.source_artifacts")
    if "calibration_chain" in provenance:
        _string_list(provenance["calibration_chain"], "provenance.calibration_chain")
    if "timestamp_or_capture_id" in provenance:
        _non_empty_string(provenance["timestamp_or_capture_id"], "provenance.timestamp_or_capture_id")
    if "notes" in provenance:
        _non_empty_string(provenance["notes"], "provenance.notes")
    if "metadata" in data:
        _require_mapping(data["metadata"], "metadata")

    is_debug_oracle = data["source_kind"] == "SIMULATOR_ORACLE_DEBUG" or (
        "debug_oracle_asset_root" in frame_contracts
    )
    if is_debug_oracle:
        if data["deployable"] is True:
            raise SchemaValidationError("debug/oracle providers must not be deployable")
        if data["oracle"] is not True:
            raise SchemaValidationError("debug/oracle providers must declare oracle=true")
    elif data["oracle"] is True:
        raise SchemaValidationError("oracle=true is only allowed for explicit debug/oracle providers")


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
VISUAL_DIAGNOSIS_CATEGORIES = {
    "no_visual_failure_observed",
    "wrong_function_region",
    "wrong_function_point",
    "wrong_contact_point",
    "wrong_target_contact",
    "orientation_relation_missing",
    "orientation_relation_wrong",
    "path_collision_or_occlusion",
    "tracking_or_execution_failure",
    "insufficient_visual_evidence",
    "unknown",
}
TOOL_PART_SEMANTIC_LABEL_MODES = {
    "vlm_part_semantic_aliases",
    "provider_supplied_part_semantic_aliases",
    "deterministic_unlabeled_scaffold",
}


def validate_visual_evidence(data: Any) -> None:
    data = _require_mapping(data, "VisualEvidence")
    required = {"schema_version", "evidence_id", "items"}
    allowed = required | {"metadata"}
    _check_keys(data, required=required, allowed=allowed, name="VisualEvidence")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["evidence_id"], "evidence_id")
    items = data["items"]
    if not isinstance(items, list) or not items:
        raise SchemaValidationError("items must be a non-empty list")
    for index, item in enumerate(items):
        item = _require_mapping(item, f"items[{index}]")
        _check_keys(
            item,
            required={"id", "kind", "media_type", "path"},
            allowed=VISUAL_EVIDENCE_ITEM_KEYS,
            name=f"items[{index}]",
        )
        _non_empty_string(item["id"], f"items[{index}].id")
        if item["kind"] not in VISUAL_EVIDENCE_KINDS:
            raise SchemaValidationError(f"items[{index}].kind is unsupported")
        if item["media_type"] not in VISUAL_EVIDENCE_MEDIA_TYPES:
            raise SchemaValidationError(f"items[{index}].media_type is unsupported")
        _non_empty_string(item["path"], f"items[{index}].path")
        for key in ("description", "provider_id", "source_stage"):
            if key in item:
                _non_empty_string(item[key], f"items[{index}].{key}")
        if item.get("allowed_use", "planning_context") not in VISUAL_EVIDENCE_ALLOWED_USES:
            raise SchemaValidationError(f"items[{index}].allowed_use is unsupported")
    if "metadata" in data:
        _require_mapping(data["metadata"], "metadata")


def validate_visual_diagnosis_report(data: Any) -> None:
    data = _require_mapping(data, "VisualDiagnosisReport")
    required = {
        "schema_version",
        "report_id",
        "diagnosis_category",
        "observations",
        "evidence_items",
        "related_refs",
        "repair_hints",
        "confidence",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="VisualDiagnosisReport")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["report_id"], "report_id")
    if data["diagnosis_category"] not in VISUAL_DIAGNOSIS_CATEGORIES:
        raise SchemaValidationError("diagnosis_category is unsupported")
    for key in ("observations", "evidence_items", "related_refs", "repair_hints"):
        if not isinstance(data[key], list):
            raise SchemaValidationError(f"{key} must be a list")
        for index, item in enumerate(data[key]):
            _non_empty_string(item, f"{key}[{index}]")
    for index, ref in enumerate(data["related_refs"]):
        if not any(regex.match(ref) for regex in GROUNDED_REF_RES):
            raise SchemaValidationError(f"related_refs[{index}] must be a symbol ref")
    confidence = _finite_number(data["confidence"], "confidence")
    if confidence < 0.0 or confidence > 1.0:
        raise SchemaValidationError("confidence must be in [0, 1]")
    _require_mapping(data["metadata"], "metadata")


def validate_tool_part_semantic_labels(data: Any) -> None:
    data = _require_mapping(data, "ToolPartSemanticLabels")
    required = {"schema_version", "label_id", "tool_id", "parts", "metadata"}
    _check_keys(data, required=required, allowed=required, name="ToolPartSemanticLabels")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["label_id"], "label_id")
    _non_empty_string(data["tool_id"], "tool_id")
    parts = _require_mapping(data["parts"], "parts")
    if not parts:
        raise SchemaValidationError("parts must be non-empty")
    for part_id, label in parts.items():
        label = _require_mapping(label, f"parts.{part_id}")
        required_label = {
            "part_ref",
            "semantic_label",
            "aliases",
            "visual_description",
            "confidence",
            "evidence_items",
            "source",
            "provenance",
        }
        _check_keys(label, required=required_label, allowed=required_label, name=f"parts.{part_id}")
        expected_ref = f"tool.parts.{part_id}"
        if label["part_ref"] != expected_ref:
            raise SchemaValidationError(f"parts.{part_id}.part_ref must be {expected_ref}")
        _non_empty_string(label["semantic_label"], f"parts.{part_id}.semantic_label")
        aliases = label["aliases"]
        if not isinstance(aliases, list):
            raise SchemaValidationError(f"parts.{part_id}.aliases must be a list")
        for index, alias in enumerate(aliases):
            _non_empty_string(alias, f"parts.{part_id}.aliases[{index}]")
        _non_empty_string(label["visual_description"], f"parts.{part_id}.visual_description")
        confidence = _finite_number(label["confidence"], f"parts.{part_id}.confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise SchemaValidationError(f"parts.{part_id}.confidence must be in [0, 1]")
        evidence_items = label["evidence_items"]
        if not isinstance(evidence_items, list):
            raise SchemaValidationError(f"parts.{part_id}.evidence_items must be a list")
        for index, item in enumerate(evidence_items):
            _non_empty_string(item, f"parts.{part_id}.evidence_items[{index}]")
        _non_empty_string(label["source"], f"parts.{part_id}.source")
        provenance = _require_mapping(label["provenance"], f"parts.{part_id}.provenance")
        _non_empty_string(provenance.get("method"), f"parts.{part_id}.provenance.method")
        _non_empty_string(
            provenance.get("observation_id"),
            f"parts.{part_id}.provenance.observation_id",
        )
        if "source_artifact" in provenance:
            _non_empty_string(
                provenance["source_artifact"],
                f"parts.{part_id}.provenance.source_artifact",
            )
    metadata = _require_mapping(data["metadata"], "metadata")
    _non_empty_string(metadata.get("created_by"), "metadata.created_by")
    if metadata.get("mode") not in TOOL_PART_SEMANTIC_LABEL_MODES:
        raise SchemaValidationError("metadata.mode is unsupported")


def _validate_ref(value: Any, name: str) -> None:
    if not isinstance(value, str) or not any(regex.match(value) for regex in GROUNDED_REF_RES):
        raise SchemaValidationError(f"{name} must be a symbol reference, got {value!r}")


def _validate_program_ast(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SchemaValidationError(f"program_text is not valid Python: {exc}") from exc
    entrypoints = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_program"
    ]
    if len(entrypoints) != 1:
        raise SchemaValidationError("program_text must define exactly one build_program(ctx)")
    args = entrypoints[0].args
    if len(args.args) != 1 or args.args[0].arg != "ctx":
        raise SchemaValidationError("build_program must take exactly one argument named ctx")
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Try,
        ast.While,
        ast.With,
    )
    forbidden_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "open",
        "vars",
    }
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise SchemaValidationError(f"program_text uses forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise SchemaValidationError("program_text uses dunder name")
            if isinstance(node.ctx, ast.Load) and node.id in forbidden_calls:
                raise SchemaValidationError(f"program_text uses forbidden call/name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SchemaValidationError("program_text uses dunder attribute")


def validate_planner_program(data: Any) -> None:
    data = _require_mapping(data, "PlannerProgram")
    required = {
        "schema_version",
        "program_id",
        "task_instruction",
        "language",
        "entrypoint",
        "allowed_api_version",
        "program_text",
        "safety_contract",
        "generation_metadata",
    }
    allowed = required | {"declared_symbol_table"}
    _check_keys(data, required=required, allowed=allowed, name="PlannerProgram")
    _expect_const(data, "schema_version", "v0.1")
    _expect_const(data, "language", "python")
    _expect_const(data, "entrypoint", "build_program")
    if data.get("allowed_api_version") not in PROGRAM_API_VERSIONS:
        raise SchemaValidationError(
            f"allowed_api_version must be one of {sorted(PROGRAM_API_VERSIONS)}"
        )
    _non_empty_string(data["program_id"], "program_id")
    _non_empty_string(data["task_instruction"], "task_instruction")
    _non_empty_string(data["program_text"], "program_text")
    _validate_program_ast(data["program_text"])

    safety = _require_mapping(data["safety_contract"], "safety_contract")
    _check_keys(
        safety,
        required={"ast_checked", "restricted_builtins", "imports_allowed", "io_allowed", "network_allowed"},
        allowed={"ast_checked", "restricted_builtins", "imports_allowed", "io_allowed", "network_allowed"},
        name="safety_contract",
    )
    if safety.get("ast_checked") is not True:
        raise SchemaValidationError("safety_contract.ast_checked must be true")
    if safety.get("restricted_builtins") is not True:
        raise SchemaValidationError("safety_contract.restricted_builtins must be true")
    for key in ("imports_allowed", "io_allowed", "network_allowed"):
        if safety.get(key) is not False:
            raise SchemaValidationError(f"safety_contract.{key} must be false")

    metadata = _require_mapping(data["generation_metadata"], "generation_metadata")
    _non_empty_string(metadata.get("generated_by"), "generation_metadata.generated_by")
    if metadata.get("mode") != "vlm_authored_python":
        raise SchemaValidationError("generation_metadata.mode is unsupported")


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


def validate_sanity_report(data: Any) -> None:
    data = _require_mapping(data, "SanityReport")
    required = {"passed", "failures", "warnings", "num_goals"}
    _check_required(data, required=required, name="SanityReport")
    if "schema_version" in data:
        _expect_const(data, "schema_version", "v0.1")
    if not isinstance(data["passed"], bool):
        raise SchemaValidationError("passed must be bool")
    for key in ("failures", "warnings"):
        if not isinstance(data[key], list):
            raise SchemaValidationError(f"{key} must be a list")
        for index, item in enumerate(data[key]):
            if not isinstance(item, str):
                raise SchemaValidationError(f"{key}[{index}] must be string")
    if not isinstance(data["num_goals"], int) or data["num_goals"] < 0:
        raise SchemaValidationError("num_goals must be a non-negative int")
    if "table_top_z" in data and data["table_top_z"] is not None:
        _finite_number(data["table_top_z"], "table_top_z")
    if "min_root_table_clearance_m" in data:
        _finite_number(data["min_root_table_clearance_m"], "min_root_table_clearance_m")
    if "metadata" in data:
        _require_mapping(data["metadata"], "metadata")


def validate_compiler_report(data: Any) -> None:
    data = _require_mapping(data, "CompilerReport")
    required = {
        "schema_version",
        "report_id",
        "program_id",
        "subgoals_id",
        "passed",
        "stages",
        "metadata",
    }
    _check_keys(data, required=required, allowed=required, name="CompilerReport")
    _expect_const(data, "schema_version", "v0.1")
    for key in ("report_id", "program_id", "subgoals_id"):
        _non_empty_string(data[key], key)
    if not isinstance(data["passed"], bool):
        raise SchemaValidationError("passed must be bool")
    stages = data["stages"]
    if not isinstance(stages, list):
        raise SchemaValidationError("stages must be a list")
    for index, stage in enumerate(stages):
        stage = _require_mapping(stage, f"stages[{index}]")
        required_stage = {
            "name",
            "success",
            "final_cost",
            "constraint_errors",
            "warnings",
            "source_constraints",
        }
        _check_required(stage, required=required_stage, name=f"stages[{index}]")
        _non_empty_string(stage["name"], f"stages[{index}].name")
        if not isinstance(stage["success"], bool):
            raise SchemaValidationError(f"stages[{index}].success must be bool")
        if _finite_number(stage["final_cost"], f"stages[{index}].final_cost") < 0.0:
            raise SchemaValidationError(f"stages[{index}].final_cost must be non-negative")
        errors = _require_mapping(stage["constraint_errors"], f"stages[{index}].constraint_errors")
        for key, value in errors.items():
            _non_empty_string(key, f"stages[{index}].constraint_errors key")
            _finite_number(value, f"stages[{index}].constraint_errors.{key}")
        warnings = stage["warnings"]
        if not isinstance(warnings, list):
            raise SchemaValidationError(f"stages[{index}].warnings must be a list")
        for warning_i, warning in enumerate(warnings):
            _non_empty_string(warning, f"stages[{index}].warnings[{warning_i}]")
        constraints = stage["source_constraints"]
        if not isinstance(constraints, list):
            raise SchemaValidationError(f"stages[{index}].source_constraints must be a list")
        for constraint_i, constraint in enumerate(constraints):
            _non_empty_string(constraint, f"stages[{index}].source_constraints[{constraint_i}]")
    metadata = _require_mapping(data["metadata"], "metadata")
    _check_required(metadata, required={"generated_by", "source"}, name="metadata")
    _non_empty_string(metadata["generated_by"], "metadata.generated_by")
    _non_empty_string(metadata["source"], "metadata.source")


def _parttoolgen_root_for(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if parent.name == "parttoolgen":
            return parent
    return path.parent


def validate_manifest(data: Any, *, artifact_root: str | Path | None = None) -> None:
    data = _require_mapping(data, "PartToolGenRunManifest")
    required = {
        "schema_version",
        "run_id",
        "task_instruction",
        "geometry_mode",
        "artifacts",
        "checks",
        "pipeline",
    }
    _check_keys(data, required=required, allowed=required, name="PartToolGenRunManifest")
    _expect_const(data, "schema_version", "v0.1")
    _non_empty_string(data["run_id"], "run_id")
    _non_empty_string(data["task_instruction"], "task_instruction")
    if data["geometry_mode"] not in {"observed", "mixed"}:
        raise SchemaValidationError("geometry_mode is unsupported")
    artifacts = _require_mapping(data["artifacts"], "artifacts")
    required_artifacts = {
        "observed_scene_graph",
        "grounding_candidates",
        "functional_tool_parts",
        "planner_program",
        "program_ir",
        "tool_pose_subgoals",
        "simtoolreal_fixed_goals",
        "sanity_report",
    }
    optional_artifacts = {
        "observation_bundle",
        "target_part_graph",
        "tool_part_graph",
        "provider_contract",
        "compiler_report",
        "execution_report",
        "preview",
        "visual_evidence",
        "visual_diagnosis_report",
        "tool_part_semantic_labels",
    }
    _check_keys(
        artifacts,
        required=required_artifacts,
        allowed=required_artifacts | optional_artifacts,
        name="artifacts",
    )
    for key, rel_path in artifacts.items():
        _non_empty_string(rel_path, f"artifacts.{key}")

    checks = _require_mapping(data["checks"], "checks")
    if checks.get("schema_validation") is not True:
        raise SchemaValidationError("checks.schema_validation must be true")
    pipeline = _require_mapping(data["pipeline"], "pipeline")
    required_pipeline = {
        "observation_source",
        "grounding_source",
        "scene_source",
        "tool_part_source",
        "constraint_source",
        "compiler",
        "executor_target",
        "target_geometry_source",
    }
    _check_keys(
        pipeline,
        required=required_pipeline,
        allowed=required_pipeline,
        name="pipeline",
    )
    for key in required_pipeline:
        _non_empty_string(pipeline[key], f"pipeline.{key}")

    if artifact_root is None:
        return
    root = Path(artifact_root)
    artifact_validators = {
        "observation_bundle": validate_observation_bundle,
        "observed_scene_graph": lambda artifact: validate_observed_scene_graph(
            artifact,
            observed_only=True,
        ),
        "grounding_candidates": validate_grounding_candidates,
        "target_part_graph": validate_target_part_graph,
        "functional_tool_parts": validate_functional_tool_parts,
        "planner_program": validate_planner_program,
        "program_ir": validate_program_ir,
        "tool_pose_subgoals": validate_tool_pose_subgoals,
        "simtoolreal_fixed_goals": validate_simtoolreal_fixed_goals,
        "provider_contract": validate_provider_contract,
        "compiler_report": validate_compiler_report,
        "sanity_report": validate_sanity_report,
        "visual_evidence": validate_visual_evidence,
        "visual_diagnosis_report": validate_visual_diagnosis_report,
        "tool_part_semantic_labels": validate_tool_part_semantic_labels,
    }
    for key, validator in artifact_validators.items():
        if key not in artifacts:
            continue
        path = root / artifacts[key]
        if not path.exists() or not path.is_file():
            raise SchemaValidationError(f"manifest artifact missing: {artifacts[key]}")
        validator(load_structured_file(path))
    if "preview" in artifacts:
        preview_path = root / artifacts["preview"]
        if not preview_path.exists() or preview_path.stat().st_size <= 0:
            raise SchemaValidationError(f"manifest preview missing or empty: {artifacts['preview']}")
    for key, rel_path in artifacts.items():
        if key in artifact_validators or key == "preview":
            continue
        path = root / rel_path
        if not path.exists() or not path.is_file():
            raise SchemaValidationError(f"manifest artifact missing: {rel_path}")


VALIDATORS = {
    "observation_bundle": validate_observation_bundle,
    "semantic_query_artifact": validate_semantic_query_artifact,
    "observed_scene_graph": validate_observed_scene_graph,
    "grounding_candidates": validate_grounding_candidates,
    "provider_contract": validate_provider_contract,
    "visual_evidence": validate_visual_evidence,
    "visual_diagnosis_report": validate_visual_diagnosis_report,
    "tool_part_semantic_labels": validate_tool_part_semantic_labels,
    "target_part_graph": validate_target_part_graph,
    "functional_tool_parts": validate_functional_tool_parts,
    "planner_program": validate_planner_program,
    "program_ir": validate_program_ir,
    "tool_pose_subgoals": validate_tool_pose_subgoals,
    "simtoolreal_fixed_goals": validate_simtoolreal_fixed_goals,
    "compiler_report": validate_compiler_report,
    "sanity_report": validate_sanity_report,
    "tool_part_graph": validate_tool_part_graph,
    "manifest": validate_manifest,
}


def validate_file(schema_name: str, path: str | Path) -> None:
    if schema_name not in VALIDATORS:
        raise SchemaValidationError(f"Unknown PartToolGen schema: {schema_name}")
    path = Path(path)
    if schema_name == "manifest":
        validate_manifest(load_structured_file(path), artifact_root=_parttoolgen_root_for(path))
    else:
        VALIDATORS[schema_name](load_structured_file(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, choices=sorted(VALIDATORS))
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    validate_file(args.schema, args.file)
    print(f"validated {args.file} as {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
