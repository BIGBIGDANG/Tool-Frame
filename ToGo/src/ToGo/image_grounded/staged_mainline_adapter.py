"""Bridge image-grounded ToolFrame programs to the staged mainline compiler.

The image-grounded front end gives us tool refs and scene refs.  The existing
mainline compiler already knows how to turn symbolic point/axis constraints
into staged whole-tool SE(3) goals, tracking waypoints, and SimToolReal fixed
goals.  This module adapts between those two representations.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ToGo.adapters.simtoolreal_policy_priors import (
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
)
from ToGo.compiler.geometric_solvers import quat_between_vectors, quat_mul, quat_rotate
from ToGo.compiler.residuals import unit


class ImageGroundedMainlineAdapterError(ValueError):
    """Raised when an image-grounded program cannot be adapted."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImageGroundedMainlineAdapterError(f"{name} must be an object")
    return value


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ImageGroundedMainlineAdapterError(f"{name} must be a 3-vector")
    return [float(item) for item in value]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _mul(a: Sequence[float], scale: float) -> list[float]:
    return [float(x) * float(scale) for x in a]


def _project_to_plane(vector: Sequence[float], normal: Sequence[float]) -> list[float]:
    return _sub(vector, _mul(normal, _dot(vector, normal)))


def _grounding(method: str, *, source: str = "image_grounded_vlm_rgbd") -> dict[str, Any]:
    return {
        "confidence": 1.0,
        "source": source,
        "provenance": {
            "observation_id": "image_grounded_scene",
            "method": method,
            "source_artifact": "ToolFrame image-grounded constraint program",
        },
    }


def _quat_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _quat_angle_rad(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(_quat_dot(a, b))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def _normalize_quat(quat: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in quat))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(item) / norm for item in quat]


def _quat_from_axis_angle(axis: Sequence[float], angle_rad: float) -> list[float]:
    axis_unit = unit([float(item) for item in axis])
    half = 0.5 * float(angle_rad)
    scale = math.sin(half)
    return [
        axis_unit[0] * scale,
        axis_unit[1] * scale,
        axis_unit[2] * scale,
        math.cos(half),
    ]


def _tool_geometry_displacement_cost(
    *,
    candidate_orientation: Sequence[float],
    reference_orientation: Sequence[float],
    sample_points_tool: Sequence[Sequence[float]],
) -> float:
    if not sample_points_tool:
        return _quat_angle_rad(candidate_orientation, reference_orientation) ** 2
    total = 0.0
    for point in sample_points_tool:
        before = quat_rotate(reference_orientation, point)
        after = quat_rotate(candidate_orientation, point)
        delta = _sub(after, before)
        total += _dot(delta, delta)
    return total / float(len(sample_points_tool))


def _axis_aligned_orientation_with_geometry_cost(
    *,
    start_orientation: Sequence[float],
    source_axis_tool: Sequence[float],
    target_axis_world: Sequence[float],
    sample_points_tool: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Align an axis, then resolve free roll by minimizing whole-tool motion."""

    seed = _normalize_quat(start_orientation)
    source_axis = unit(quat_rotate(seed, source_axis_tool))
    target_axis = unit([float(item) for item in target_axis_world])
    base_delta = quat_between_vectors(source_axis, target_axis)
    base_orientation = _normalize_quat(quat_mul(base_delta, seed))
    best: dict[str, Any] | None = None
    roll_step_rad = math.radians(2.0)
    for roll_index in range(-90, 91):
        roll_rad = roll_step_rad * roll_index
        roll_delta = _quat_from_axis_angle(target_axis, roll_rad)
        candidate = _normalize_quat(quat_mul(roll_delta, base_orientation))
        solved_axis = unit(quat_rotate(candidate, source_axis_tool))
        alignment = _dot(solved_axis, target_axis)
        alignment_error = 1.0 - alignment
        geometry_cost = _tool_geometry_displacement_cost(
            candidate_orientation=candidate,
            reference_orientation=seed,
            sample_points_tool=sample_points_tool,
        )
        rotation_delta = _quat_angle_rad(candidate, seed)
        score = 1000.0 * alignment_error * alignment_error + geometry_cost + 1e-8 * rotation_delta * rotation_delta
        row = {
            "orientation_xyzw": candidate,
            "solved_axis_world": solved_axis,
            "alignment_cosine": alignment,
            "rotation_delta_from_start_rad": rotation_delta,
            "rotation_delta_from_start_deg": math.degrees(rotation_delta),
            "axis_roll_rad": roll_rad,
            "orientation_geometry_displacement_m2": geometry_cost,
            "orientation_cost_sample_mode": "whole_tool_bbox_corners"
            if sample_points_tool
            else "fallback_so3_angle",
            "orientation_cost_num_samples": len(sample_points_tool),
            "score": score,
        }
        if best is None or (float(row["score"]), float(row["rotation_delta_from_start_rad"])) < (
            float(best["score"]),
            float(best["rotation_delta_from_start_rad"]),
        ):
            best = row
    if best is None:
        raise ImageGroundedMainlineAdapterError("axis alignment had no feasible orientation candidate")
    return best


def _start_orientation_from_task(
    task_data: Mapping[str, Any] | None,
    start_pose: Mapping[str, Any] | None,
) -> list[float] | None:
    if isinstance(start_pose, Mapping):
        orientation = start_pose.get("orientation_xyzw")
        if isinstance(orientation, Sequence) and not isinstance(orientation, (str, bytes)) and len(orientation) == 4:
            return [float(item) for item in orientation]
    if isinstance(task_data, Mapping):
        raw = task_data.get("execution", {}).get("simtoolreal_start_pose")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 7:
            return [float(item) for item in raw[3:7]]
    return None


def _tool_bbox_corners_from_mesh(tool_refs: Mapping[str, Any]) -> dict[str, Any] | None:
    mesh_path = tool_refs.get("tool_mesh")
    if not isinstance(mesh_path, str) or not mesh_path:
        return None
    resolved = resolve_simtoolreal_tool_mesh_path(explicit_path=mesh_path)
    if resolved is None:
        return None
    vertices = load_obj_vertices(resolved)
    if not vertices:
        return None
    mins = [min(float(vertex[i]) for vertex in vertices) for i in range(3)]
    maxs = [max(float(vertex[i]) for vertex in vertices) for i in range(3)]
    corners = [
        [x, y, z]
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]
    return {
        "mesh_path": str(resolved),
        "vertex_count": len(vertices),
        "bbox_min_tool": mins,
        "bbox_max_tool": maxs,
        "bbox_corners_tool": corners,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_existing_path(path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    raw = Path(path_value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, _repo_root() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _tool_part_obb_corners_from_graph(tool_refs: Mapping[str, Any]) -> dict[str, Any] | None:
    graph_path = _resolve_existing_path(tool_refs.get("tool_part_graph_path"))
    if graph_path is None:
        return None
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    parts_raw = data.get("parts")
    if isinstance(parts_raw, Mapping):
        part_items = [(str(part_id), part) for part_id, part in sorted(parts_raw.items())]
    elif isinstance(parts_raw, Sequence) and not isinstance(parts_raw, (str, bytes)):
        part_items = [(str(index), part) for index, part in enumerate(parts_raw)]
    else:
        return None

    corners: list[list[float]] = []
    part_ids: list[str] = []
    for part_id, part in part_items:
        if not isinstance(part, Mapping):
            continue
        obb = part.get("obb")
        if not isinstance(obb, Mapping):
            continue
        center = _vec3(obb.get("center_xyz_tool"), f"tool_part_graph.parts.{part_id}.obb.center_xyz_tool")
        extent = _vec3(obb.get("extent_xyz"), f"tool_part_graph.parts.{part_id}.obb.extent_xyz")
        axes_raw = _mapping(obb.get("axes_tool"), f"tool_part_graph.parts.{part_id}.obb.axes_tool")
        axes = [
            unit(_vec3(axes_raw.get(axis), f"tool_part_graph.parts.{part_id}.obb.axes_tool.{axis}"))
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
        part_ids.append(part_id)
    if not corners:
        return None
    return {
        "tool_part_graph_path": str(graph_path),
        "part_count": len(part_ids),
        "part_ids": part_ids,
        "obb_corners_tool": corners,
    }


def _orientation_change_sample_points(tool_refs: Mapping[str, Any]) -> dict[str, Any]:
    part_obbs = _tool_part_obb_corners_from_graph(tool_refs)
    if part_obbs is not None:
        return {
            "type": "tool_geometry_displacement",
            "sample_mode": "all_tool_part_obbs",
            "sample_points_tool": part_obbs["obb_corners_tool"],
            "tool_part_graph_path": part_obbs["tool_part_graph_path"],
            "part_count": part_obbs["part_count"],
            "part_ids": part_obbs["part_ids"],
            "num_sample_points": len(part_obbs["obb_corners_tool"]),
        }
    tool_bbox = _tool_bbox_corners_from_mesh(tool_refs)
    return {
        "type": "tool_geometry_displacement",
        "sample_mode": "whole_tool_bbox_corners",
        "sample_points_tool": tool_bbox["bbox_corners_tool"] if tool_bbox else [],
        "mesh_path": tool_bbox["mesh_path"] if tool_bbox else None,
        "vertex_count": tool_bbox["vertex_count"] if tool_bbox else 0,
        "bbox_min_tool": tool_bbox["bbox_min_tool"] if tool_bbox else None,
        "bbox_max_tool": tool_bbox["bbox_max_tool"] if tool_bbox else None,
        "num_sample_points": len(tool_bbox["bbox_corners_tool"]) if tool_bbox else 0,
    }


def _motion_distance(
    *,
    object_center: Sequence[float],
    end_center: Sequence[float],
) -> float:
    delta = [float(end_center[i]) - float(object_center[i]) for i in range(3)]
    distance = math.sqrt(sum(item * item for item in delta))
    if distance > 1e-4:
        return distance
    return 0.0


def _optional_surface_normal_from_program(
    *,
    scene_refs: Mapping[str, Any],
) -> list[float] | None:
    for key in ("surface_normal", "target_surface_normal"):
        value = scene_refs.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
            return unit(_vec3(value, f"scene_refs.{key}"))
        if isinstance(value, Mapping):
            vector = value.get("vector_world") or value.get("normal_world")
            if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)) and len(vector) == 3:
                return unit(_vec3(vector, f"scene_refs.{key}.vector_world"))
    return None


def _optional_axis_from_program(
    *,
    scene_refs: Mapping[str, Any],
    key: str,
) -> list[float] | None:
    value = scene_refs.get(key)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return unit(_vec3(value, f"scene_refs.{key}"))
    if isinstance(value, Mapping):
        vector = value.get("vector_world") or value.get("normal_world")
        if isinstance(vector, Sequence) and not isinstance(vector, (str, bytes)) and len(vector) == 3:
            return unit(_vec3(vector, f"scene_refs.{key}.vector_world"))
    return None


def _has_unsupported_axis_role(program: Mapping[str, Any]) -> bool:
    stages = program.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return False
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        for section_name in ("subgoal_constraints", "path_constraints", "constraints"):
            constraints = stage.get(section_name)
            if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
                continue
            for constraint in constraints:
                if not isinstance(constraint, Mapping):
                    continue
                params = constraint.get("params")
                if isinstance(params, Mapping) and str(params.get("axis_role") or "") not in {
                    "",
                    "tool_motion_axis",
                    "motion_axis",
                }:
                    return True
    return False


def _orthonormal_frame_from_normal(
    normal_tool: Sequence[float],
    *,
    rectangle_corners: Sequence[Sequence[float]] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    z_axis = unit([float(item) for item in normal_tool])
    tangent_hint: list[float] | None = None
    short_hint: list[float] | None = None
    if rectangle_corners and len(rectangle_corners) >= 4:
        c0 = _vec3(rectangle_corners[0], "rectangle_corners[0]")
        c1 = _vec3(rectangle_corners[1], "rectangle_corners[1]")
        c3 = _vec3(rectangle_corners[3], "rectangle_corners[3]")
        edge_a = _sub(c1, c0)
        edge_b = _sub(c3, c0)
        if _norm(edge_b) > _norm(edge_a):
            edge_a, edge_b = edge_b, edge_a
        tangent_hint = edge_a
        short_hint = edge_b
    if tangent_hint is None or _norm(_project_to_plane(tangent_hint, z_axis)) <= 1e-9:
        base = [1.0, 0.0, 0.0] if abs(z_axis[0]) < 0.9 else [0.0, 1.0, 0.0]
        tangent_hint = _cross(base, z_axis)
    x_axis = unit(_project_to_plane(tangent_hint, z_axis))
    y_axis = unit(_cross(z_axis, x_axis))
    if short_hint is not None and _dot(y_axis, short_hint) < 0.0:
        y_axis = _mul(y_axis, -1.0)
    return x_axis, y_axis, z_axis


def _motion_axis_candidates_from_tool_refs(
    tool_refs: Mapping[str, Any],
    *,
    primary_axis_tool: Sequence[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "primary_motion_axis",
            "vector_tool": unit([float(item) for item in primary_axis_tool]),
            "source": "tool_refs.motion_axis_tool",
        }
    ]
    raw_candidates = tool_refs.get("motion_axis_candidates_tool")
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes)):
        for index, raw in enumerate(raw_candidates):
            if isinstance(raw, Mapping):
                vector = raw.get("vector_tool")
                if vector is None:
                    continue
                candidate = {
                    "candidate_id": str(raw.get("candidate_id") or f"motion_axis_candidate_{index}"),
                    "vector_tool": unit(_vec3(vector, f"tool_refs.motion_axis_candidates_tool[{index}].vector_tool")),
                    "axis_marker": raw.get("axis_marker"),
                    "axis_role": raw.get("axis_role"),
                    "axis_ref": raw.get("axis_ref"),
                    "surface_ref": raw.get("surface_ref"),
                    "patch_id": raw.get("patch_id"),
                    "source": raw.get("source") or "tool_refs.motion_axis_candidates_tool",
                }
            else:
                candidate = {
                    "candidate_id": f"motion_axis_candidate_{index}",
                    "vector_tool": unit(_vec3(raw, f"tool_refs.motion_axis_candidates_tool[{index}]")),
                    "source": "tool_refs.motion_axis_candidates_tool",
                }
            candidates.append(candidate)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        vector = unit(_vec3(candidate.get("vector_tool"), "motion_axis_candidate.vector_tool"))
        key = tuple(round(value, 6) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["vector_tool"] = list(key)
        deduped.append(row)
    return deduped


def _contact_point_candidates_from_tool_refs(
    tool_refs: Mapping[str, Any],
    *,
    primary_contact_point_tool: Sequence[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "primary_contact_point",
            "point_tool": _vec3(primary_contact_point_tool, "tool_refs.contact_point_tool"),
            "source": "tool_refs.contact_point_tool",
        }
    ]
    raw_candidates = tool_refs.get("contact_point_candidates_tool")
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes)):
        for index, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                continue
            point = raw.get("contact_point_tool") or raw.get("point_tool")
            if point is None:
                continue
            candidates.append(
                {
                    "candidate_id": str(raw.get("candidate_id") or f"contact_point_candidate_{index}"),
                    "point_tool": _vec3(point, f"tool_refs.contact_point_candidates_tool[{index}].contact_point_tool"),
                    "source": raw.get("source") or "tool_refs.contact_point_candidates_tool",
                    "contact_point_face_marker": raw.get("contact_point_face_marker"),
                    "contact_point_face_id": raw.get("contact_point_face_id"),
                    "contact_point_source_ref": raw.get("contact_point_source_ref"),
                    "contact_point_surface_ref": raw.get("contact_point_surface_ref"),
                    "contact_point_part_ref": raw.get("contact_point_part_ref"),
                    "source_candidate_index": index,
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        point = _vec3(candidate.get("point_tool"), "contact_point_candidate.point_tool")
        key = tuple(round(value, 6) for value in point)
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["point_tool"] = list(key)
        row["candidate_index"] = len(deduped)
        deduped.append(row)
    return deduped


def build_functional_tool_parts_from_tool_refs(program: Mapping[str, Any]) -> dict[str, Any]:
    """Build the old FunctionalToolParts artifact from ToolFrame tool refs."""

    tool_refs = _mapping(program.get("tool_refs", {}), "program.tool_refs")
    orientation_metric = _orientation_change_sample_points(tool_refs)
    center = _vec3(
        tool_refs.get("contact_point_tool") or tool_refs.get("contact_point_xyz_tool"),
        "tool_refs.contact_point_tool",
    )
    contact_point_candidates = _contact_point_candidates_from_tool_refs(
        tool_refs,
        primary_contact_point_tool=center,
    )
    motion_axis_override = tool_refs.get("motion_axis_tool")
    if motion_axis_override is None:
        raise ImageGroundedMainlineAdapterError(
            "tool_refs missing motion_axis_tool; public tool refs must be merged before lowering"
        )
    axis_vector = unit(_vec3(motion_axis_override, "tool_refs.motion_axis_tool"))
    motion_axis_candidates = _motion_axis_candidates_from_tool_refs(
        tool_refs,
        primary_axis_tool=axis_vector,
    )
    base = [0.0, 0.0, 1.0] if abs(axis_vector[2]) < 0.9 else [0.0, 1.0, 0.0]
    normal = unit(_cross(axis_vector, base))
    surface_geometry = tool_refs.get("surface_geometry_tool", {})
    corners = None
    if isinstance(surface_geometry, Mapping):
        raw_corners = surface_geometry.get("rectangle_corners")
        if isinstance(raw_corners, Sequence) and not isinstance(raw_corners, (str, bytes)):
            corners = raw_corners
    x_axis, y_axis, z_axis = _orthonormal_frame_from_normal(normal, rectangle_corners=corners)
    part_id = "function"
    grounding = _grounding("tool_contact_motion_refs_from_mesh_axis_evidence", source="tool_mesh_prior_vlm")
    return {
        "schema_version": "v0.1",
        "tool_id": str(tool_refs.get("tool_id") or program.get("task_id") or "image_grounded_tool"),
        "tool_frame": "tool",
        "functional_parts": {
            part_id: {
                **grounding,
                "role": "contact_patch",
                "center": {"xyz_tool": center},
                "point_candidates": contact_point_candidates,
                "normal": {"vector_tool": z_axis},
                "axis": {"vector_tool": axis_vector},
                "axis_candidates": motion_axis_candidates,
                "tangent_long": {"vector_tool": axis_vector},
                "tangent_short": {"vector_tool": y_axis},
                "frame": {
                    "origin": center,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "z_axis": z_axis,
                },
                "T_tool_functional": [
                    [x_axis[0], y_axis[0], z_axis[0], center[0]],
                    [x_axis[1], y_axis[1], z_axis[1], center[1]],
                    [x_axis[2], y_axis[2], z_axis[2], center[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
        "grasp_references": {
            "default_grasp": {
                **_grounding("default_asset_root_grasp_reference", source="simtoolreal_tool_asset"),
                "position_tool": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "metadata": {
            "created_by": "toolframe.image_grounded.staged_mainline_adapter",
            "geometry_mode": "mixed",
            "segmentation_mode": "tool_mesh_contact_motion_refs",
            "source_tool_refs": copy.deepcopy(dict(tool_refs)),
            "orientation_change_metric": orientation_metric,
            "motion_axis_ref": str(
                tool_refs.get("motion_axis_ref") or "tool.motion_axis"
            ),
            "contact_point_ref": str(
                tool_refs.get("contact_point_ref") or "tool.contact_point"
            ),
            "motion_axis_candidate_count": len(motion_axis_candidates),
            "contact_point_candidate_count": len(contact_point_candidates),
        },
    }


def derive_scene_geometry_from_program(
    program: Mapping[str, Any],
    *,
    task_data: Mapping[str, Any] | None = None,
    approach_offset_m: float = 0.03,
) -> dict[str, Any]:
    """Derive backend target refs from VLM object/start/end refs.

    Top-down image grounding may mark an object reference on the visible top
    surface, while a second view can supply only the contact height.  When the
    program provides ``scene_refs.start`` we use it as the executable contact
    anchor and keep ``scene_refs.object_point`` as the target-object reference.
    """

    scene_refs = _mapping(program.get("scene_refs"), "program.scene_refs")
    start_ref = scene_refs.get("start") or scene_refs.get("start_point")
    action_end_ref = scene_refs.get("end") or scene_refs.get("end_point")
    object_ref = _mapping(scene_refs.get("object_point") or start_ref, "scene_refs.object_point")
    contact_ref = _mapping(start_ref or object_ref, "scene_refs.start")
    end_ref = _mapping(action_end_ref, "scene_refs.end")
    object_anchor = _vec3(object_ref.get("position_world"), "scene_refs.object_point.position_world")
    contact_anchor = _vec3(contact_ref.get("position_world"), "scene_refs.start.position_world")
    end_anchor = _vec3(end_ref.get("position_world"), "scene_refs.end.position_world")
    motion_axis = unit(
        _vec3(
            scene_refs.get("direction"),
            "scene_refs.direction",
        )
    )
    object_center = [float(item) for item in object_anchor]
    end_center = [float(item) for item in end_anchor]
    contact_point = [float(item) for item in contact_anchor]
    motion_distance = _motion_distance(
        object_center=contact_anchor,
        end_center=end_center,
    )
    surface_normal = _optional_surface_normal_from_program(scene_refs=scene_refs)
    explicit_approach_axis = _optional_axis_from_program(
        scene_refs=scene_refs,
        key="target_approach_axis",
    )
    approach_axis = _mul(motion_axis, -1.0)
    pre_contact_point = _add(contact_point, _mul(approach_axis, approach_offset_m))
    end_point = list(end_anchor)
    raw_motion_path = scene_refs.get("motion_path")
    path: list[list[float]] = []
    path_sources: list[str] = []
    if isinstance(raw_motion_path, Sequence) and not isinstance(raw_motion_path, (str, bytes)):
        for index, item in enumerate(raw_motion_path):
            if not isinstance(item, Mapping):
                raise ImageGroundedMainlineAdapterError(
                    f"scene_refs.motion_path[{index}] must be an object with position_world"
                )
            path.append(
                _vec3(item.get("position_world"), f"scene_refs.motion_path[{index}].position_world")
            )
            path_sources.append(
                str(item.get("keypoint_id") or item.get("id") or f"motion_path_{index}")
            )
    if len(path) >= 2:
        path[0] = list(contact_point)
        path[-1] = list(end_point)
    else:
        path = [list(contact_point), list(end_point)]
        path_sources = ["start_point", "end_point"]
    path_segment_axes = []
    for index, (start, goal) in enumerate(zip(path[:-1], path[1:])):
        delta = [float(goal[i]) - float(start[i]) for i in range(3)]
        distance = sum(value * value for value in delta) ** 0.5
        axis = unit(delta) if distance > 1e-9 else list(motion_axis)
        path_segment_axes.append(
            {
                "id": f"path_segment_{index}",
                "vector_world": axis,
                "start_index": index,
                "end_index": index + 1,
                "source": "scene_refs.motion_path_segment"
                if len(path) > 2
                else "scene_refs.direction",
            }
        )
    target_anchor_source = (
        "scene_refs.start"
        if scene_refs.get("start") is not None
        else "vlm_marked_pixel_rgbd_lift"
    )
    return {
        "object_center": object_center,
        "end_center": end_center,
        "target_size": [0.0, 0.0, 0.0],
        "start_point": contact_point,
        "pre_start_point": pre_contact_point,
        "end_point": end_point,
        "direction": motion_axis,
        "target_surface_normal": surface_normal,
        "target_approach_axis": approach_axis,
        "path": path,
        "path_sources": path_sources,
        "path_segment_axes": path_segment_axes,
        "motion_distance_m": motion_distance,
        "half_extent_along_motion_m": 0.0,
        "approach_offset_m": float(approach_offset_m),
        "target_anchor_source": target_anchor_source,
        "target_surface_normal_source": (
            "scene_refs.target_surface_normal"
            if surface_normal is not None
            else "not_required_for_direction_contact"
        ),
        "target_approach_axis_source": (
            "opposite_first_vlm_motion_direction_for_pre_contact"
            if explicit_approach_axis is None
            else "scene_refs.target_approach_axis_ignored_by_mainline_motion_direction_policy"
        ),
        "uses_private_target_extent_for_execution": False,
    }


def _scene_object(
    object_id: str,
    *,
    category: str,
    center: Sequence[float],
    extent: Sequence[float] | None = None,
    method: str,
    attributes_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {"center_xyz_world": [float(item) for item in center]}
    if extent is not None:
        attributes["extent_xyz"] = [float(item) for item in extent]
    if attributes_extra:
        attributes.update(dict(attributes_extra))
    return {
        **_grounding(method),
        "category": category,
        "attributes": attributes,
    }


def _keypoint(name: str, xyz_world: Sequence[float], *, object_ref: str, method: str) -> dict[str, Any]:
    return {
        **_grounding(method),
        "frame": "world",
        "object_ref": object_ref,
        "xyz_world": [float(item) for item in xyz_world],
    }


def _axis(name: str, vector_world: Sequence[float], *, object_ref: str, origin: Sequence[float]) -> dict[str, Any]:
    return {
        **_grounding("axis_from_image_grounded_motion"),
        "frame": "world",
        "object_ref": object_ref,
        "vector_world": unit([float(item) for item in vector_world]),
        "origin_xyz_world": [float(item) for item in origin],
    }


def _stage_name_aliases(stage_name: str) -> tuple[str, ...]:
    if stage_name in {"contact", "start"}:
        return ("contact", "start")
    if stage_name in {"action", "move", "use"}:
        return ("action", "move", "use")
    return (stage_name,)


def _stage_rows(program: Mapping[str, Any], stage_name: str) -> list[Mapping[str, Any]]:
    stages = program.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise ImageGroundedMainlineAdapterError("program.stages must be present; no task-family fallback is used")
    rows: list[Mapping[str, Any]] = []
    accepted_names = set(_stage_name_aliases(stage_name))
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("name") not in accepted_names:
            continue
        for group in ("subgoal_constraints", "path_constraints", "constraints"):
            group_rows = stage.get(group)
            if isinstance(group_rows, Sequence) and not isinstance(group_rows, (str, bytes)):
                rows.extend(row for row in group_rows if isinstance(row, Mapping))
    if not rows:
        raise ImageGroundedMainlineAdapterError(
            f"program.stages has no constraints for stage {stage_name!r}; explicit staged constraints are required"
        )
    return rows


def _relation_to_polarity(relation: str) -> str:
    normalized = relation.strip().lower()
    if normalized == "parallel":
        return "same"
    if normalized == "same":
        return "same"
    if normalized in {"anti_parallel", "antiparallel", "opposite"}:
        return "opposite"
    raise ImageGroundedMainlineAdapterError(f"unsupported axis relation: {relation}")


def _program_axis_ref_to_target_axis(axis_ref: str) -> str:
    aliases = {
        "scene.direction": "target.axes.direction",
        "scene.axes.direction": "target.axes.direction",
        "target.axes.direction": "target.axes.direction",
        "target.axes.target_approach_axis": "target.axes.target_approach_axis",
    }
    if axis_ref not in aliases:
        if axis_ref.startswith("target.axes.path_segment_"):
            return axis_ref
        if axis_ref.startswith("scene.axes.path_segment_"):
            return "target.axes." + axis_ref.rsplit(".", 1)[-1]
        raise ImageGroundedMainlineAdapterError(
            f"unsupported explicit target axis ref {axis_ref!r}; add it to scene_refs/target_part_graph before compiling"
        )
    return aliases[axis_ref]


def _program_tool_point_ref_to_program_ir_ref(point_ref: str) -> str:
    if point_ref == "tool.contact_point":
        return "tool.contact_point"
    raise ImageGroundedMainlineAdapterError(
        f"unsupported explicit tool point ref {point_ref!r}; use tool.contact_point"
    )


def _program_tool_axis_ref_to_program_ir_ref(axis_ref: str, *, default_axis_ref: str) -> str:
    aliases = {
        "tool.motion_axis": "tool.motion_axis",
    }
    if axis_ref in aliases:
        return aliases[axis_ref]
    raise ImageGroundedMainlineAdapterError(
        f"unsupported explicit tool axis ref {axis_ref!r}; use tool.motion_axis"
    )


def _program_scene_point_ref_to_program_ir_ref(point_ref: str) -> str:
    aliases = {
        "scene.start": "scene.keypoints.start",
        "scene.start_point": "scene.keypoints.start",
        "scene.end": "scene.keypoints.end",
        "scene.end_point": "scene.keypoints.end",
    }
    if point_ref in aliases:
        return aliases[point_ref]
    raise ImageGroundedMainlineAdapterError(
        f"unsupported explicit scene point ref {point_ref!r}; add it to scene_refs before compiling"
    )


def _approach_offset_binding_from_stage(program: Mapping[str, Any]) -> dict[str, Any]:
    for constraint in _stage_rows(program, "approach"):
        if constraint.get("type") != "point_offset":
            continue
        params = _mapping(constraint.get("params", {}), "approach.point_offset.params")
        if str(params.get("axis_role") or "") not in {"", "contact_approach"}:
            continue
        tool_ref = str(params.get("tool_ref") or "")
        target_ref = str(params.get("target_ref") or "")
        direction_ref = str(params.get("direction_ref") or "")
        if not tool_ref or not target_ref or not direction_ref:
            raise ImageGroundedMainlineAdapterError(
                "approach point_offset must declare tool_ref, target_ref, and direction_ref"
            )
        scene_refs = _mapping(program.get("scene_refs", {}), "program.scene_refs")
        raw_motion_path = scene_refs.get("motion_path") or []
        first_motion_axis_ref = (
            "target.axes.path_segment_0"
            if isinstance(raw_motion_path, Sequence)
            and not isinstance(raw_motion_path, (str, bytes))
            and len(raw_motion_path) > 2
            else "target.axes.direction"
        )
        distance_m = float(params.get("distance_m", 0.0))
        # Mainline image-grounded tasks use the VLM motion direction as the
        # semantic approach basis for every task family.  The pre-contact point
        # is before the public start point, so it is always offset opposite to
        # the first start->end/path-segment motion direction.
        return {
            "source": _program_tool_point_ref_to_program_ir_ref(tool_ref),
            "target": _program_scene_point_ref_to_program_ir_ref(target_ref),
            "axis": first_motion_axis_ref,
            "distance_m": abs(distance_m),
            "direction": "opposite_axis",
            "program_direction_ref": direction_ref,
            "program_distance_m": distance_m,
            "mainline_direction_policy": "opposite_first_vlm_motion_direction_for_pre_contact",
        }
    raise ImageGroundedMainlineAdapterError(
        "approach stage has no explicit point_offset contact approach constraint"
    )


def _with_approach_offset_override(
    approach_offset: Mapping[str, Any],
    *,
    approach_offset_m: float,
) -> dict[str, Any]:
    normalized = dict(approach_offset)
    normalized["distance_m"] = abs(float(approach_offset_m))
    normalized["configured_distance_m"] = abs(float(approach_offset_m))
    return normalized


def _tool_axis_ref_for_axis_role(program: Mapping[str, Any], axis_role: str) -> str:
    if not axis_role:
        return ""
    if axis_role not in {"tool_motion_axis", "motion_axis"}:
        raise ImageGroundedMainlineAdapterError(
            f"axis_role={axis_role!r} is unsupported; use axis_role='tool_motion_axis' with tool.motion_axis"
        )
    if axis_role == "motion_axis":
        raise ImageGroundedMainlineAdapterError(
            "axis_role='motion_axis' is deprecated; use axis_role='tool_motion_axis'"
        )
    if axis_role not in {"tool_motion_axis"}:
        return ""
    tool_refs = program.get("tool_refs")
    if not isinstance(tool_refs, Mapping):
        return ""
    return str(tool_refs.get("motion_axis_ref") or "tool.motion_axis")


def _axis_binding_from_stage(program: Mapping[str, Any], stage_name: str) -> dict[str, str]:
    for constraint in _stage_rows(program, stage_name):
        if constraint.get("type") != "axis_align":
            continue
        params = _mapping(constraint.get("params", {}), f"{stage_name}.axis_align.params")
        target_axis_ref = str(params.get("target_axis_ref") or params.get("target_axis") or "")
        if not target_axis_ref:
            raise ImageGroundedMainlineAdapterError(
                f"{stage_name} axis_align is missing params.target_axis_ref"
            )
        if not (params.get("relation") or params.get("polarity")):
            raise ImageGroundedMainlineAdapterError(
                f"{stage_name} axis_align is missing params.relation"
            )
        relation = str(params.get("relation") or params.get("polarity"))
        axis_role = str(params.get("axis_role") or "")
        source_axis = str(params.get("tool_axis_ref") or params.get("source_axis") or "")
        role_source_axis = _tool_axis_ref_for_axis_role(program, axis_role)
        if role_source_axis:
            if source_axis and source_axis != role_source_axis:
                raise ImageGroundedMainlineAdapterError(
                    f"{stage_name} axis_align has axis_role={axis_role!r} but "
                    f"params.tool_axis_ref={source_axis!r} differs from merged "
                    f"tool_refs role axis {role_source_axis!r}; rerun tool-ref merge "
                    "so staged constraints are synchronized"
                )
            source_axis = role_source_axis
        if not source_axis:
            raise ImageGroundedMainlineAdapterError(
                f"{stage_name} axis_align is missing params.tool_axis_ref/source_axis; "
                "explicit tool refs must be merged before compiling"
            )
        return {
            "target_axis": _program_axis_ref_to_target_axis(target_axis_ref),
            "polarity": _relation_to_polarity(relation),
            "source_axis": source_axis,
            "program_target_axis_ref": target_axis_ref,
            "program_relation": relation,
            "axis_role": axis_role,
        }
    raise ImageGroundedMainlineAdapterError(
        f"{stage_name} stage has no explicit axis_align constraint; no task-family fallback is used"
    )


def _optional_axis_binding_from_stage(
    program: Mapping[str, Any],
    stage_name: str,
) -> dict[str, str] | None:
    try:
        return _axis_binding_from_stage(program, stage_name)
    except ImageGroundedMainlineAdapterError as exc:
        if "stage has no explicit axis_align constraint" in str(exc):
            return None
        raise


def _axis_vector_from_program_ref(axis_ref: str, derived: Mapping[str, Any]) -> list[float]:
    if axis_ref in {
        "scene.direction",
        "scene.axes.direction",
        "target.axes.direction",
    }:
        return unit(_vec3(derived["direction"], "derived.direction"))
    if axis_ref == "target.axes.target_approach_axis":
        return unit(_vec3(derived["target_approach_axis"], "derived.target_approach_axis"))
    raise ImageGroundedMainlineAdapterError(
        f"unsupported explicit target axis ref {axis_ref!r}; add it to scene_refs/target_part_graph before compiling"
    )


def _candidate_target_axes(
    axis_world: Sequence[float],
    *,
    polarity: str,
) -> list[tuple[str, list[float]]]:
    target = unit([float(item) for item in axis_world])
    if polarity == "same":
        return [("same", target)]
    if polarity == "opposite":
        return [("opposite", _mul(target, -1.0))]
    if polarity == "auto":
        return [("same", target), ("opposite", _mul(target, -1.0))]
    raise ImageGroundedMainlineAdapterError(f"unsupported axis polarity {polarity!r}")


def _preserve_vlm_selected_tool_refs(program: Mapping[str, Any]) -> dict[str, Any]:
    tool_refs = dict(_mapping(program.get("tool_refs", {}), "program.tool_refs"))
    candidates = tool_refs.get("contact_point_candidates_tool")
    if not isinstance(candidates, list) or not candidates:
        return dict(program)
    # Keep the public VLM/P3SAM evidence set intact; the compiler compiles the
    # selected refs and does not run a second face-reselection pass.
    return dict(program)


def build_observed_scene_graph(
    program: Mapping[str, Any],
    *,
    task_data: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    derived_scene = dict(derived or derive_scene_geometry_from_program(program, task_data=task_data))
    scene_refs = _mapping(program.get("scene_refs", {}), "program.scene_refs")
    target_object_ref = scene_refs.get("target_object")
    target_object_label = None
    target_object_description = None
    if isinstance(target_object_ref, Mapping):
        target_object_label = target_object_ref.get("label")
        target_object_description = target_object_ref.get("description")
    state_bundle = task_data.get("execution", {}).get("state_bundle", {}) if isinstance(task_data, Mapping) else {}
    table_pose = state_bundle.get("table_pose_xyzw", [0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0])
    table_center = [float(table_pose[0]), float(table_pose[1]), float(table_pose[2])]
    table_extent = [0.475, 0.4, 0.3]
    target_center = _vec3(derived_scene["object_center"], "derived.object_center")
    target_extent = _vec3(derived_scene["target_size"], "derived.target_size")
    target_contact = _vec3(derived_scene["start_point"], "derived.start_point")
    raw_target_normal = derived_scene.get("target_surface_normal")
    target_normal = (
        _vec3(raw_target_normal, "derived.target_surface_normal")
        if raw_target_normal is not None
        else None
    )
    motion_axis = _vec3(derived_scene["direction"], "derived.direction")
    approach_axis = _vec3(derived_scene["target_approach_axis"], "derived.target_approach_axis")
    surface_normal_for_backend = target_normal if target_normal is not None else approach_axis
    axes = {
        "direction": _axis(
            "direction",
            motion_axis,
            object_ref="target_object",
            origin=target_contact,
        ),
        "target_approach_axis": _axis(
            "target_approach_axis",
            approach_axis,
            object_ref="target_object",
            origin=target_contact,
        ),
    }
    for index, axis in enumerate(derived_scene.get("path_segment_axes") or []):
        path_points = derived_scene.get("path") or [target_contact]
        origin = (
            _vec3(path_points[index], f"derived.path[{index}]")
            if index < len(path_points)
            else target_contact
        )
        axes[f"path_segment_{index}"] = _axis(
            f"path_segment_{index}",
            _vec3(axis["vector_world"], f"derived.path_segment_axes[{index}].vector_world"),
            object_ref="target_object",
            origin=origin,
        )
    if target_normal is not None:
        axes["target_surface_normal"] = _axis(
            "target_surface_normal",
            target_normal,
            object_ref="target_object",
            origin=target_contact,
        )
    surface = {
        **_grounding("view_aware_fused_contact_anchor"),
        "frame": "world",
        "object_ref": "target_object",
        "point_xyz_world": target_contact,
        "normal_world": surface_normal_for_backend,
        "normal_source": (
            "scene_refs.target_surface_normal"
            if target_normal is not None
            else "operational_target_approach_axis"
        ),
    }
    return {
        "schema_version": "v0.1",
        "scene_id": f"{program.get('task_id', 'image_grounded_task')}_image_grounded_scene",
        "frames": {"world": {"type": "world", "description": "SimToolReal world frame."}},
        "objects": {
            "support_table": _scene_object(
                "support_table",
                category="support_surface",
                center=table_center,
                extent=table_extent,
                method="simtoolreal_state_bundle_table",
            ),
            "target_object": _scene_object(
                "target_object",
                category="target",
                center=target_center,
                extent=target_extent,
                method="vlm_rgbd_target_object_point",
                attributes_extra={
                    "role": "target_object",
                    "collision_role": "dynamic_manipulable",
                    "vlm_target_label": str(target_object_label)
                    if target_object_label is not None
                    else None,
                    "vlm_target_description": str(target_object_description)
                    if target_object_description is not None
                    else None,
                },
            ),
        },
        "keypoints": {
            "target_center": _keypoint(
                "target_center",
                target_center,
                object_ref="target_object",
                method="vlm_rgbd_start_point_lifted",
            ),
            "start": _keypoint(
                "start",
                target_contact,
                object_ref="target_object",
                method="view_aware_fused_contact_anchor",
            ),
            "pre_start": _keypoint(
                "pre_start",
                _vec3(derived_scene["pre_start_point"], "derived.pre_start_point"),
                object_ref="target_object",
                method="derived_vlm_direction_pre_contact_offset",
            ),
            "end": _keypoint(
                "end",
                _vec3(derived_scene["end_point"], "derived.end_point"),
                object_ref="target_object",
                method="vlm_end_point_lifted_to_contact_end",
            ),
            **{
                f"path_{index}": _keypoint(
                    f"path_{index}",
                    _vec3(point, f"derived.path[{index}]"),
                    object_ref="target_object",
                    method="vlm_motion_path_keypoint_lifted",
                )
                for index, point in enumerate(derived_scene.get("path") or [])
            },
            "support_table_center": _keypoint(
                "support_table_center",
                table_center,
                object_ref="support_table",
                method="simtoolreal_state_bundle_table",
            ),
        },
        "axes": axes,
        "regions": {
            "target_bbox": {
                **_grounding("vlm_rgbd_target_object_point"),
                "frame": "world",
                "object_ref": "target_object",
                "center_xyz_world": target_center,
                "extent_xyz": target_extent,
                "note": "zero extent placeholder; simulator target box is not used for executable planning",
            },
            "support_table_bbox": {
                **_grounding("simtoolreal_state_bundle_table"),
                "frame": "world",
                "object_ref": "support_table",
                "center_xyz_world": table_center,
                "extent_xyz": table_extent,
            },
        },
        "surfaces": {
            "target_contact_surface": surface
        },
        "relations": [
            {
                "type": "surface_of",
                "subject": "scene.surfaces.target_contact_surface",
                "object": "scene.objects.target_object",
                "confidence": 1.0,
            }
        ],
        "metadata": {
            "created_by": "toolframe.image_grounded.staged_mainline_adapter",
            "geometry_mode": "observed",
            "source": "VLM target object/action grounding + optional contact-height view fusion + task support geometry",
            "uses_private_target_extent_for_execution": False,
        },
    }


def build_target_part_graph(
    program: Mapping[str, Any],
    *,
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    target_contact = _vec3(derived["start_point"], "derived.start_point")
    raw_target_normal = derived.get("target_surface_normal")
    target_normal = (
        _vec3(raw_target_normal, "derived.target_surface_normal")
        if raw_target_normal is not None
        else None
    )
    axes = {
        name: _axis(name, derived[value_key], object_ref="target_object", origin=target_contact)
        for name, value_key in {
            "direction": "direction",
            "target_approach_axis": "target_approach_axis",
        }.items()
    }
    if target_normal is not None:
        axes["target_surface_normal"] = _axis(
            "target_surface_normal",
            target_normal,
            object_ref="target_object",
            origin=target_contact,
        )
    for index, axis in enumerate(derived.get("path_segment_axes") or []):
        path_points = derived.get("path") or [target_contact]
        origin = (
            _vec3(path_points[index], f"derived.path[{index}]")
            if index < len(path_points)
            else target_contact
        )
        axes[f"path_segment_{index}"] = _axis(
            f"path_segment_{index}",
            _vec3(axis["vector_world"], f"derived.path_segment_axes[{index}].vector_world"),
            object_ref="target_object",
            origin=origin,
        )
    for axis_id, axis in axes.items():
        axis["axis_ref"] = f"target.axes.{axis_id}"
        axis.pop("frame", None)
    contact_patch_normal = target_normal if target_normal is not None else _vec3(
        derived["target_approach_axis"],
        "derived.target_approach_axis",
    )
    contact_patch = {
        **_grounding("view_aware_fused_contact_anchor"),
        "patch_ref": "target.parts.target_contact_region.surface_patches.contact_patch",
        "role_hint": "contact_region",
        "center_xyz_world": target_contact,
        "normal_world": contact_patch_normal,
        "normal_source": (
            "scene_refs.target_surface_normal"
            if target_normal is not None
            else "operational_target_approach_axis"
        ),
    }
    return {
        "schema_version": "v0.1",
        "graph_id": f"{program.get('task_id', 'image_grounded_task')}_target_part_graph",
        "target_id": "target_object",
        "frame": "world",
        "parts": {
            "target_contact_region": {
                **_grounding("view_aware_fused_contact_anchor"),
                "part_ref": "target.parts.target_contact_region",
                "label": "view-aware fused target contact anchor",
                "object_ref": "scene.objects.target_object",
                "role_candidates": ["contact_region", "target_part"],
                "surface_patches": {
                    "contact_patch": contact_patch
                },
            }
        },
        "axes": axes,
        "relations": [],
        "metadata": {
            "created_by": "toolframe.image_grounded.staged_mainline_adapter",
            "geometry_mode": "observed",
            "uses_private_target_extent_for_execution": False,
        },
    }


def build_grounding_candidates(
    program: Mapping[str, Any],
    *,
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    point = _vec3(
        _mapping(program["tool_refs"], "program.tool_refs").get("contact_point_tool"),
        "tool_refs.contact_point_tool",
    )
    motion_axis = _vec3(
        _mapping(program["tool_refs"], "program.tool_refs").get("motion_axis_tool"),
        "tool_refs.motion_axis_tool",
    )
    contact = _vec3(derived["start_point"], "derived.start_point")
    pre_contact = _vec3(derived["pre_start_point"], "derived.pre_start_point")
    end = _vec3(derived["end_point"], "derived.end_point")
    return {
        "schema_version": "v0.1",
        "grounding_id": f"{program.get('task_id', 'image_grounded_task')}_image_grounded_candidates",
        "observation_id": f"{program.get('task_id', 'image_grounded_task')}_image_grounded_observation",
        "frames": {"world": {"type": "world"}, "tool": {"type": "tool"}},
        "marks": {},
        "points": {
            "contact_point": {
                **_grounding("tool_contact_motion_refs_from_mesh_axis_evidence", source="tool_mesh_prior_vlm"),
                "label": "selected tool contact point",
                "candidate_type": "point",
                "xyz_tool": point,
            },
            "start": {
                **_grounding("view_aware_fused_contact_anchor"),
                "label": "view-aware fused target contact point",
                "candidate_type": "point",
                "xyz_world": contact,
            },
            "pre_start": {
                **_grounding("derived_vlm_direction_pre_contact_offset"),
                "label": "target pre-contact point",
                "candidate_type": "point",
                "xyz_world": pre_contact,
            },
            "end": {
                **_grounding("vlm_end_point_lifted_to_contact_end"),
                "label": "target contact end point",
                "candidate_type": "point",
                "xyz_world": end,
            },
        },
        "regions": {},
        "axes": {
            "motion_axis": {
                **_grounding("tool_contact_motion_refs_from_mesh_axis_evidence", source="tool_mesh_prior_vlm"),
                "label": "selected tool motion axis",
                "candidate_type": "axis",
                "vector_tool": unit(motion_axis),
                "origin_xyz_tool": point,
            }
        },
        "surfaces": {},
        "parts": {},
        "metadata": {
            "created_by": "toolframe.image_grounded.staged_mainline_adapter",
            "grounding_mode": "mixed",
            "uses_private_target_extent_for_execution": False,
        },
    }


def build_program_ir_from_image_grounded_program(
    program: Mapping[str, Any],
    *,
    contact_ref: str = "tool.contact_point",
    axis_ref: str = "tool.motion_axis",
    approach_offset_m: float = 0.03,
) -> dict[str, Any]:
    """Lower the Tool-Scene Constraint IR to the legacy ProgramIR backend."""

    task_id = str(program.get("task_id") or "image_grounded_task")
    instruction = str(program.get("instruction") or "")
    program_ir_contact_ref = _program_tool_point_ref_to_program_ir_ref(contact_ref)
    action_axis = _axis_binding_from_stage(program, "action")
    action_source_axis = _program_tool_axis_ref_to_program_ir_ref(
        action_axis["source_axis"],
        default_axis_ref=axis_ref,
    )
    approach_axis = _axis_binding_from_stage(program, "align")
    approach_source_axis = _program_tool_axis_ref_to_program_ir_ref(
        approach_axis["source_axis"],
        default_axis_ref=axis_ref,
    )
    approach_offset = _with_approach_offset_override(
        _approach_offset_binding_from_stage(program),
        approach_offset_m=approach_offset_m,
    )
    align_constraints: list[dict[str, Any]] = []
    scene_refs = _mapping(program.get("scene_refs", {}), "program.scene_refs")
    raw_motion_path = scene_refs.get("motion_path") or []
    path_refs = (
        [
            f"scene.keypoints.path_{index}"
            for index, _ in enumerate(raw_motion_path)
        ]
        if isinstance(raw_motion_path, Sequence)
        and not isinstance(raw_motion_path, (str, bytes))
        and len(raw_motion_path) > 2
        else []
    )
    if len(path_refs) < 2:
        path_refs = [
            "scene.keypoints.start",
            "scene.keypoints.end",
        ]
    multi_segment_path = len(path_refs) > 2
    action_stages = [
        {
            "name": "action" if not multi_segment_path else f"action_path_{index + 1}",
            "description": (
                "execute the selected tool ref action along one segment of the "
                "VLM-grounded scene path under explicit axis constraints"
            ),
            "repeat": 1,
        }
        for index in range(len(path_refs) - 1)
    ]
    action_constraints: list[dict[str, Any]] = []
    for index, (start_ref, end_ref) in enumerate(zip(path_refs[:-1], path_refs[1:])):
        suffix = "" if not multi_segment_path else f"_{index + 1}"
        stage_name = "action" if not multi_segment_path else f"action_path_{index + 1}"
        # The path supplies positions. The explicit axis binding supplies orientation.
        # A bend in the path must not silently replace a VLM-selected surface normal.
        segment_axis_ref = action_axis["target_axis"]
        action_constraints.extend(
            [
                {
                    "id": f"action_axis_direction{suffix}",
                    "stage": stage_name,
                    "type": "align_axis",
                    "refs": {"source_axis": action_source_axis, "target_axis": segment_axis_ref},
                    "params": {"polarity": action_axis["polarity"], "min_cosine": 0.9},
                    "hard": True,
                    "weight": 1.0,
                    "source": "image_grounded_scene_adapter",
                },
                {
                    "id": f"action_contact_start{suffix}",
                    "stage": stage_name,
                    "type": "contact",
                    "refs": {"source": program_ir_contact_ref, "target": start_ref},
                    "params": {"tolerance_m": 0.008},
                    "hard": True,
                    "weight": 1.0,
                    "source": "image_grounded_scene_adapter",
                },
                {
                    "id": f"action_move_along_vlm_path{suffix}",
                    "stage": stage_name,
                    "type": "move_along",
                    "refs": {
                        "source": program_ir_contact_ref,
                        "start": start_ref,
                        "end": end_ref,
                    },
                    "params": {"path_family": "contact_follow", "profile": "linear"},
                    "hard": False,
                    "weight": 1.0,
                    "source": "image_grounded_scene_adapter",
                },
            ]
        )
    approach_constraints = [
        {
            "id": "approach_pre_contact_offset",
            "stage": "approach",
            "type": "pre_contact_offset",
            "refs": {
                "source": program_ir_contact_ref,
                "target": approach_offset["target"],
                "axis": approach_offset["axis"],
            },
            "params": {
                "distance_m": approach_offset["distance_m"],
                "direction": approach_offset["direction"],
            },
            "hard": True,
            "weight": 1.0,
            "source": "image_grounded_scene_adapter",
        },
    ]
    align_constraints.append(
        {
            "id": "align_axis_face_target",
            "stage": "align",
            "type": "align_axis",
            "refs": {
                "source_axis": approach_source_axis,
                "target_axis": approach_offset["axis"],
            },
            "params": {"polarity": "same", "min_cosine": 0.9},
            "hard": True,
            "weight": 1.0,
            "source": "image_grounded_scene_adapter",
        }
    )

    program_ir = {
        "schema_version": "v0.1",
        "program_id": f"{task_id}_image_grounded_mainline_program",
        "task_instruction": instruction,
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [
            {
                "name": "align",
                "description": (
                    "align the selected tool axis in free space before approaching "
                    "the target contact point"
                ),
                "repeat": 1,
            },
            {
                "name": "approach",
                "description": (
                    "approach the pre-contact target after free-space tool-axis alignment"
                ),
                "repeat": 1,
            },
            *action_stages,
        ],
        "constraints": [
            *align_constraints,
            *approach_constraints,
            *action_constraints,
        ],
        "objectives": [],
        "diagnostics": {
            "ast_checked": True,
            "executed": True,
            "source": "generated_from_toolframe_constraint_program",
            "source_ir_name": (
                str(program.get("metadata", {}).get("ir_name"))
                if isinstance(program.get("metadata"), Mapping)
                else "Tool-Scene Constraint IR"
            ),
            "constraint_target_types": [
                "PointTarget",
                "AxisTarget",
                "MoveAlongTarget",
                "OrientationTarget",
                "ClearanceTarget",
            ],
            "explicit_axis_bindings": {
                "align": approach_axis,
                "approach": {
                    "target": approach_offset["target"],
                    "axis": approach_offset["axis"],
                    "distance_m": approach_offset["distance_m"],
                    "direction": approach_offset["direction"],
                    "depends_on_stage": "align",
                },
                "action": action_axis,
                "approach_source_axis": approach_source_axis,
                "action_source_axis": action_source_axis,
                "approach_coupled_motion_axis": False,
                "align_before_approach": True,
                "motion_path_ref_count": len(path_refs),
                "motion_path_segment_count": max(0, len(path_refs) - 1),
            },
        },
    }
    _validate_image_grounded_program_ir_contract(program_ir)
    return program_ir


def _validate_image_grounded_program_ir_contract(program_ir: Mapping[str, Any]) -> None:
    constraints = program_ir.get("constraints")
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)):
        raise ImageGroundedMainlineAdapterError("generated ProgramIR has no constraints")
    contact_refs = set()
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            continue
        stage = str(constraint.get("stage") or "")
        if (stage == "action" or stage.startswith("action_path_")) and constraint.get("type") == "contact":
            refs = constraint.get("refs")
            if isinstance(refs, Mapping):
                contact_refs.add(str(refs.get("target") or ""))
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            continue
        stage = str(constraint.get("stage") or "")
        if not (stage in {"start", "contact", "action", "use"} or stage.startswith("action_path_")):
            continue
        if constraint.get("type") != "move_along":
            continue
        params = constraint.get("params")
        refs = constraint.get("refs")
        if not isinstance(params, Mapping) or not isinstance(refs, Mapping):
            raise ImageGroundedMainlineAdapterError("start/contact move_along must have refs and params")
        if str(params.get("path_family") or "contact_follow") != "contact_follow":
            continue
        start_ref = str(refs.get("start") or "")
        end_ref = str(refs.get("end") or "")
        valid_start = start_ref == "scene.keypoints.start" or start_ref.startswith(
            "scene.keypoints.path_"
        )
        if not valid_start:
            raise ImageGroundedMainlineAdapterError(
                "contact_follow move_along must start at a public start/path keypoint, not pre-contact"
            )
        if start_ref not in contact_refs:
            raise ImageGroundedMainlineAdapterError(
                "contact_follow move_along start must have a start/contact constraint"
            )
        valid_end = end_ref == "scene.keypoints.end" or end_ref.startswith(
            "scene.keypoints.path_"
        )
        if not valid_end:
            raise ImageGroundedMainlineAdapterError(
                "contact_follow move_along end must be a public end/path keypoint"
            )


def build_planner_program_stub(program: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(program.get("task_id") or "image_grounded_task")
    instruction = str(program.get("instruction") or "")
    return {
        "schema_version": "v0.1",
        "program_id": f"{task_id}_image_grounded_mainline_program",
        "task_instruction": instruction,
        "language": "python",
        "entrypoint": "build_program",
        "allowed_api_version": "parttoolgen_program_api_v0.3",
        "program_text": (
            "def build_program(ctx):\n"
            "    # ProgramIR is generated from image-grounded refs by the staged adapter.\n"
            "    return ctx.program()\n"
        ),
        "declared_symbol_table": {},
        "diagnostics": {
            "source": "toolframe.image_grounded.staged_mainline_adapter",
            "contains_objectives": False,
        },
    }


def build_mainline_inputs_from_image_grounded_program(
    program: Mapping[str, Any],
    *,
    task_data: Mapping[str, Any] | None = None,
    start_pose: Mapping[str, Any] | None = None,
    approach_offset_m: float = 0.03,
) -> dict[str, Any]:
    """Return artifacts ready for run_mainline_fixed_goal_chain."""

    derived = derive_scene_geometry_from_program(
        program,
        task_data=task_data,
        approach_offset_m=approach_offset_m,
    )
    selected_program = _preserve_vlm_selected_tool_refs(program)
    return {
        "observed_scene_graph": build_observed_scene_graph(
            selected_program,
            task_data=task_data,
            derived=derived,
        ),
        "functional_tool_parts": build_functional_tool_parts_from_tool_refs(selected_program),
        "grounding_candidates": build_grounding_candidates(selected_program, derived=derived),
        "target_part_graph": build_target_part_graph(selected_program, derived=derived),
        "tool_part_graph": None,
        "program_ir": build_program_ir_from_image_grounded_program(
            selected_program,
            approach_offset_m=approach_offset_m,
        ),
        "planner_program": build_planner_program_stub(selected_program),
        "derived_scene_geometry": derived,
        "selected_constraint_program": selected_program,
    }
