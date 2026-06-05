#!/usr/bin/env python3
"""Build tool-frame PartToolGen candidates from mesh part face labels.

SAMPart3D and P3SAM both emit face-label arrays for a mesh. This module keeps
that output in the offline tool-prior lane by turning mesh labels into stable
tool-frame part refs, geometric centers, surface patches, grasp candidates, and
functional frame candidates.
"""

from __future__ import annotations

import ast
import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AXES = ("x", "y", "z")
IDENTITY_AXES = {
    "x": [1.0, 0.0, 0.0],
    "y": [0.0, 1.0, 0.0],
    "z": [0.0, 0.0, 1.0],
}


@dataclass(frozen=True)
class ObjMesh:
    vertices: list[list[float]]
    faces: list[list[int]]


@dataclass(frozen=True)
class SegmentGeometry:
    label: int
    part_id: str
    face_indices: list[int]
    vertex_indices: list[int]
    face_centroids: list[list[float]]
    face_normals: list[list[float]]
    face_areas: list[float]
    centroid: list[float]
    center: list[float]
    extents: list[float]
    principal_axis_index: int
    shortest_axis_index: int
    face_count: int
    vertex_count: int
    original_face_count: int
    connected_component_index: int
    connected_component_count: int
    removed_disconnected_face_count: int


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not safe:
        safe = "candidate"
    if not re.match(r"^[A-Za-z_]", safe):
        safe = f"id_{safe}"
    return safe


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return [float(item) / norm for item in vector]


def _length(vector: list[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in vector))


def _axis_vector(index: int, sign: float = 1.0) -> list[float]:
    vector = [0.0, 0.0, 0.0]
    vector[index] = 1.0 if sign >= 0.0 else -1.0
    return vector


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _frame_from_normal(normal: list[float], preferred_x: list[float]) -> dict[str, list[float]]:
    z_axis = _unit(normal)
    x_seed = _unit(preferred_x)
    projection = [z_axis[i] * _dot(x_seed, z_axis) for i in range(3)]
    x_axis = [x_seed[i] - projection[i] for i in range(3)]
    if sum(value * value for value in x_axis) <= 1e-8:
        fallback = [0.0, 1.0, 0.0] if abs(z_axis[1]) < 0.9 else [1.0, 0.0, 0.0]
        projection = [z_axis[i] * _dot(fallback, z_axis) for i in range(3)]
        x_axis = [fallback[i] - projection[i] for i in range(3)]
    x_axis = _unit(x_axis)
    y_axis = _unit(_cross(z_axis, x_axis))
    return {"x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis}


def _transform_from_frame(frame: dict[str, list[float]], origin: list[float]) -> list[list[float]]:
    x_axis = frame["x_axis"]
    y_axis = frame["y_axis"]
    z_axis = frame["z_axis"]
    return [
        [x_axis[0], y_axis[0], z_axis[0], origin[0]],
        [x_axis[1], y_axis[1], z_axis[1], origin[1]],
        [x_axis[2], y_axis[2], z_axis[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mean(points: list[list[float]]) -> list[float]:
    return [sum(point[i] for point in points) / float(len(points)) for i in range(3)]


def _minmax(points: list[list[float]]) -> tuple[list[float], list[float]]:
    return (
        [min(point[i] for point in points) for i in range(3)],
        [max(point[i] for point in points) for i in range(3)],
    )


def _positive_extents(extents: list[float]) -> list[float]:
    return [max(float(item), 1e-6) for item in extents]


def _surface_point(geometry: SegmentGeometry, axis_index: int, sign: float) -> list[float]:
    point = list(geometry.center)
    point[axis_index] = float(geometry.center[axis_index]) + float(sign) * 0.5 * float(
        geometry.extents[axis_index]
    )
    return point


def _face_centroid_normal_area(mesh: ObjMesh, face_index: int) -> tuple[list[float], list[float], float]:
    face = mesh.faces[face_index]
    points = [mesh.vertices[index] for index in face]
    if len(points) == 3:
        edge_a = [points[1][i] - points[0][i] for i in range(3)]
        edge_b = [points[2][i] - points[0][i] for i in range(3)]
        normal_area2 = _cross(edge_a, edge_b)
        area = 0.5 * _length(normal_area2)
        centroid = _mean(points)
        normal = _unit(normal_area2) if area > 1e-12 else [0.0, 0.0, 1.0]
        return centroid, normal, area

    weighted_centroid = [0.0, 0.0, 0.0]
    normal_area2 = [0.0, 0.0, 0.0]
    area_sum = 0.0
    origin = points[0]
    for idx in range(1, len(points) - 1):
        tri = [origin, points[idx], points[idx + 1]]
        edge_a = [tri[1][axis] - tri[0][axis] for axis in range(3)]
        edge_b = [tri[2][axis] - tri[0][axis] for axis in range(3)]
        tri_normal_area2 = _cross(edge_a, edge_b)
        tri_area = 0.5 * _length(tri_normal_area2)
        if tri_area <= 1e-12:
            continue
        tri_centroid = _mean(tri)
        area_sum += tri_area
        normal_area2 = [normal_area2[axis] + tri_normal_area2[axis] for axis in range(3)]
        weighted_centroid = [
            weighted_centroid[axis] + tri_area * tri_centroid[axis]
            for axis in range(3)
        ]
    if area_sum <= 1e-12:
        return _mean(points), [0.0, 0.0, 1.0], 0.0
    centroid = [value / area_sum for value in weighted_centroid]
    normal = _unit(normal_area2) if _length(normal_area2) > 1e-12 else [0.0, 0.0, 1.0]
    return centroid, normal, area_sum


def _surface_center_for_axis_side(
    geometry: SegmentGeometry,
    axis_index: int,
    sign: float,
) -> tuple[list[float], list[float], int]:
    """Area-weighted mesh surface center on one side of a SAMPart3D part."""

    outward = _axis_vector(axis_index, sign)
    if not geometry.face_centroids:
        return _surface_point(geometry, axis_index, sign), outward, 0
    side_coords = [_dot(centroid, outward) for centroid in geometry.face_centroids]
    max_side = max(side_coords)
    normal_alignments = [_dot(normal, outward) for normal in geometry.face_normals]
    side_band = max(0.004, 0.18 * float(geometry.extents[axis_index]))
    selected = [
        idx
        for idx, side_coord in enumerate(side_coords)
        if side_coord >= max_side - side_band and normal_alignments[idx] >= -0.05
    ]
    if not selected:
        fallback_point = _surface_point(geometry, axis_index, sign)
        selected = sorted(
            range(len(geometry.face_centroids)),
            key=lambda idx: _length(
                [
                    geometry.face_centroids[idx][axis] - fallback_point[axis]
                    for axis in range(3)
                ]
            ),
        )[: min(32, len(geometry.face_centroids))]
    weights: list[float] = []
    for idx in selected:
        alignment_weight = min(max(0.35 + normal_alignments[idx], 0.05), 1.5)
        weights.append(max(float(geometry.face_areas[idx]), 1e-12) * alignment_weight)
    weight_sum = sum(weights)
    if weight_sum <= 1e-12:
        return _surface_point(geometry, axis_index, sign), outward, len(selected)
    center = [
        sum(weight * geometry.face_centroids[idx][axis] for weight, idx in zip(weights, selected))
        / weight_sum
        for axis in range(3)
    ]
    normal = [
        sum(weight * geometry.face_normals[idx][axis] for weight, idx in zip(weights, selected))
        / weight_sum
        for axis in range(3)
    ]
    if _length(normal) <= 1e-12:
        normal = outward
    else:
        normal = _unit(normal)
    return center, normal, len(selected)


def _grounding(
    *,
    observation_id: str,
    method: str,
    source_artifact: str,
    confidence: float,
    source: str = "sampart3d_tool_prior",
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "observation_id": observation_id,
        "method": method,
        "source_artifact": source_artifact,
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return {
        "confidence": float(max(0.0, min(1.0, confidence))),
        "source": source,
        "provenance": provenance,
    }


def load_obj_mesh(path: str | Path) -> ObjMesh:
    """Load the vertex and face topology needed for SAMPart3D face labels."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                face: list[int] = []
                for token in tokens:
                    raw_index = int(token.split("/")[0])
                    if raw_index < 0:
                        index = len(vertices) + raw_index
                    else:
                        index = raw_index - 1
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"face index {raw_index} out of range in {path}")
                    face.append(index)
                if len(face) >= 3 and len(set(face)) >= 3:
                    faces.append(face)
    if not vertices:
        raise ValueError(f"no vertices found in OBJ: {path}")
    if not faces:
        raise ValueError(f"no faces found in OBJ: {path}")
    return ObjMesh(vertices=vertices, faces=faces)


def load_npy_1d_int(path: str | Path) -> list[int]:
    """Load a simple 1-D integer .npy file without requiring NumPy at runtime."""

    npy_path = Path(path)
    with npy_path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not a NumPy .npy file: {path}")
        major, _minor = f.read(2)
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"unsupported .npy version in {path}")
        header = ast.literal_eval(f.read(header_len).decode("latin1").strip())
        descr = str(header.get("descr"))
        shape = tuple(header.get("shape") or ())
        if header.get("fortran_order") is not False:
            raise ValueError(f"fortran-order .npy arrays are unsupported: {path}")
        if len(shape) != 1:
            raise ValueError(f"expected a 1-D face label array in {path}, got shape {shape}")
        dtype_map = {
            "|i1": ("b", 1),
            "<i1": ("b", 1),
            "|u1": ("B", 1),
            "<u1": ("B", 1),
            "<i2": ("h", 2),
            "<u2": ("H", 2),
            "<i4": ("i", 4),
            "<u4": ("I", 4),
            "<i8": ("q", 8),
            "<u8": ("Q", 8),
        }
        if descr not in dtype_map:
            raise ValueError(f"unsupported integer dtype {descr!r} in {path}")
        code, width = dtype_map[descr]
        count = int(shape[0])
        payload = f.read(count * width)
        if len(payload) != count * width:
            raise ValueError(f"truncated .npy payload in {path}")
    return [int(item) for item in struct.unpack(f"<{count}{code}", payload)]


def _face_connected_components(mesh: ObjMesh, face_indices: list[int]) -> list[list[int]]:
    """Split a face set into mesh-topology connected components."""

    if not face_indices:
        return []
    face_set = set(face_indices)
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_i in face_indices:
        for vertex_i in mesh.faces[face_i]:
            vertex_to_faces[vertex_i].append(face_i)

    components: list[list[int]] = []
    visited: set[int] = set()
    for start in face_indices:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            face_i = stack.pop()
            component.append(face_i)
            for vertex_i in mesh.faces[face_i]:
                for neighbor in vertex_to_faces[vertex_i]:
                    if neighbor in face_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda item: (-len(item), item[0]))
    return components


def _component_bounds(mesh: ObjMesh, face_indices: list[int]) -> tuple[list[float], list[float]]:
    vertex_ids = sorted({idx for face_i in face_indices for idx in mesh.faces[face_i]})
    return _minmax([mesh.vertices[idx] for idx in vertex_ids])


def _bbox_gap(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> float:
    a_min, a_max = a
    b_min, b_max = b
    squared = 0.0
    for axis in range(3):
        gap = max(0.0, max(float(a_min[axis]) - float(b_max[axis]), float(b_min[axis]) - float(a_max[axis])))
        squared += gap * gap
    return math.sqrt(squared)


def _dominant_spatial_component_cluster(
    mesh: ObjMesh,
    components: list[list[int]],
) -> list[int]:
    """Keep the largest spatial cluster of disconnected components.

    Meshes such as brushes can have real functional regions made of many
    disconnected bristle components.  We therefore avoid keeping only the
    single largest topological component.  Instead, nearby components form a
    spatial cluster, and only remote islands with the same SAMPart3D label are
    filtered away.
    """

    if len(components) <= 1:
        return components[0] if components else []
    component_bounds = [_component_bounds(mesh, component) for component in components]
    all_points = [
        mesh.vertices[idx]
        for component in components
        for face_i in component
        for idx in mesh.faces[face_i]
    ]
    label_mins, label_maxs = _minmax(all_points)
    label_diag = _length([label_maxs[axis] - label_mins[axis] for axis in range(3)])
    merge_gap_m = max(0.006, 0.06 * label_diag)

    neighbors: list[list[int]] = [[] for _ in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _bbox_gap(component_bounds[i], component_bounds[j]) <= merge_gap_m:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited: set[int] = set()
    clusters: list[list[int]] = []
    for start in range(len(components)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append(cluster)
    best_cluster = max(
        clusters,
        key=lambda cluster: (sum(len(components[index]) for index in cluster), -min(cluster)),
    )
    return sorted(face_i for index in best_cluster for face_i in components[index])


def _segment_geometry(mesh: ObjMesh, labels: list[int], min_face_count: int) -> list[SegmentGeometry]:
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"SAMPart3D label count ({len(labels)}) does not match mesh face count ({len(mesh.faces)})"
        )
    faces_by_label: dict[int, list[int]] = defaultdict(list)
    for face_i, label in enumerate(labels):
        if int(label) < 0:
            continue
        faces_by_label[int(label)].append(face_i)

    geometries: list[SegmentGeometry] = []
    public_part_index = 0
    for label in sorted(faces_by_label):
        original_face_indices = faces_by_label[label]
        if len(original_face_indices) < min_face_count:
            continue
        components = _face_connected_components(mesh, original_face_indices)
        if not components:
            continue
        face_indices = _dominant_spatial_component_cluster(mesh, components)
        if len(face_indices) < min_face_count:
            continue
        removed_face_count = len(original_face_indices) - len(face_indices)
        vertex_ids = sorted({idx for face_i in face_indices for idx in mesh.faces[face_i]})
        points = [mesh.vertices[idx] for idx in vertex_ids]
        face_centroids: list[list[float]] = []
        face_normals: list[list[float]] = []
        face_areas: list[float] = []
        for face_i in face_indices:
            centroid, normal, area = _face_centroid_normal_area(mesh, face_i)
            face_centroids.append(centroid)
            face_normals.append(normal)
            face_areas.append(area)
        mins, maxs = _minmax(points)
        extents = _positive_extents([maxs[i] - mins[i] for i in range(3)])
        center = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
        principal_axis = max(range(3), key=lambda idx: extents[idx])
        shortest_axis = min(range(3), key=lambda idx: extents[idx])
        geometries.append(
            SegmentGeometry(
                label=label,
                part_id=f"part_{public_part_index:03d}",
                face_indices=face_indices,
                vertex_indices=vertex_ids,
                face_centroids=face_centroids,
                face_normals=face_normals,
                face_areas=face_areas,
                centroid=_mean(points),
                center=center,
                extents=extents,
                principal_axis_index=principal_axis,
                shortest_axis_index=shortest_axis,
                face_count=len(face_indices),
                vertex_count=len(vertex_ids),
                original_face_count=len(original_face_indices),
                connected_component_index=0,
                connected_component_count=len(components),
                removed_disconnected_face_count=removed_face_count,
            )
        )
        public_part_index += 1
    if not geometries:
        raise ValueError("SAMPart3D labels did not yield any usable mesh segments")
    return geometries


def _roles_for_geometry(geometry: SegmentGeometry, max_faces: int) -> list[str]:
    roles = ["tool_part_candidate", "functional_candidate"]
    sorted_extents = sorted(geometry.extents)
    elongation = sorted_extents[-1] / max(sorted_extents[1], 1e-6)
    if elongation >= 2.2:
        roles.append("grasp_candidate")
    if geometry.face_count <= max(8, int(0.05 * max_faces)):
        roles.append("fine_feature_candidate")
    roles.append("contact_patch")
    return roles


def _segment_confidence(geometry: SegmentGeometry, max_faces: int) -> float:
    if max_faces <= 0:
        return 0.5
    support = math.sqrt(float(geometry.face_count) / float(max_faces))
    return max(0.35, min(0.95, 0.45 + 0.45 * support))


def _surface_patch_record(
    *,
    geometry: SegmentGeometry,
    axis_index: int,
    sign: float,
    role_hint: str,
    observation_id: str,
    source_artifact: str,
    confidence: float,
    segmentation_mode: str = "sampart3d",
    grounding_source: str = "sampart3d_tool_prior",
) -> dict[str, Any]:
    center, normal, support_faces = _surface_center_for_axis_side(
        geometry,
        axis_index,
        sign,
    )
    return {
        "role_hint": role_hint,
        "center_xyz_tool": center,
        "normal_tool": normal,
        "support_faces": support_faces,
        **_grounding(
            observation_id=observation_id,
            method=f"{segmentation_mode}_mesh_surface_center",
            source_artifact=source_artifact,
            confidence=confidence,
            source=grounding_source,
        ),
    }


def _surface_patches_for_geometry(
    *,
    geometry: SegmentGeometry,
    observation_id: str,
    source_artifact: str,
    confidence: float,
    segmentation_mode: str = "sampart3d",
    grounding_source: str = "sampart3d_tool_prior",
) -> dict[str, dict[str, Any]]:
    patches: dict[str, dict[str, Any]] = {}
    for axis_index, axis_name in enumerate(AXES):
        role_suffix = "endpoint" if axis_index == geometry.principal_axis_index else "contact"
        role_hint = "endpoint_contact" if role_suffix == "endpoint" else "contact_patch"
        for sign, sign_name in ((-1.0, "negative"), (1.0, "positive")):
            patches[f"{axis_name}_{sign_name}_{role_suffix}"] = _surface_patch_record(
                geometry=geometry,
                axis_index=axis_index,
                sign=sign,
                role_hint=role_hint,
                observation_id=observation_id,
                source_artifact=source_artifact,
                confidence=confidence,
                segmentation_mode=segmentation_mode,
                grounding_source=grounding_source,
            )
    return patches


def _functional_part_record(
    *,
    geometry: SegmentGeometry,
    patch_id: str,
    origin: list[float],
    normal: list[float],
    role: str,
    observation_id: str,
    source_artifact: str,
    confidence: float,
    segmentation_mode: str = "sampart3d",
    grounding_source: str = "sampart3d_tool_prior",
) -> dict[str, Any]:
    axis_name = None
    for name in AXES:
        for polarity in ("negative", "positive"):
            if patch_id.startswith(f"{name}_{polarity}_"):
                axis_name = name
                break
        if axis_name is not None:
            break
    normal_axis_index = AXES.index(axis_name) if axis_name in AXES else None
    tangent_indices = [idx for idx in range(3) if idx != normal_axis_index] if normal_axis_index is not None else []
    if len(tangent_indices) == 2:
        long_index = max(tangent_indices, key=lambda idx: float(geometry.extents[idx]))
        short_index = min(tangent_indices, key=lambda idx: float(geometry.extents[idx]))
        frame = _frame_from_normal(normal, _axis_vector(long_index))
        tangent_long = frame["x_axis"]
        tangent_short = frame["y_axis"]
    else:
        preferred_x = _axis_vector(geometry.principal_axis_index)
        frame = _frame_from_normal(normal, preferred_x)
        tangent_long = frame["x_axis"]
        tangent_short = frame["y_axis"]
    frame_with_origin = {"origin": origin, **frame}
    return {
        "role": role,
        "center": {"xyz_tool": origin},
        "normal": {"vector_tool": normal},
        "axis": {"vector_tool": tangent_long},
        "tangent_long": {"vector_tool": tangent_long},
        "tangent_short": {"vector_tool": tangent_short},
        "frame": frame_with_origin,
        "T_tool_functional": _transform_from_frame(frame, origin),
        **_grounding(
            observation_id=observation_id,
            method=f"{segmentation_mode}_face_label_adapter",
            source_artifact=source_artifact,
            confidence=confidence,
            source=grounding_source,
            extra_provenance={
                "part_label": geometry.label,
                "patch_id": patch_id,
                "connected_component_index": geometry.connected_component_index,
                "connected_component_count": geometry.connected_component_count,
                "original_face_count": geometry.original_face_count,
                "kept_face_count": geometry.face_count,
                "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
            },
        ),
    }


def build_sampart3d_mesh_tool_prior(
    *,
    tool_id: str,
    source_mesh: str | Path,
    face_labels_path: str | Path,
    observation_id: str | None = None,
    min_face_count: int = 1,
    segmentation_mode: str = "sampart3d",
    provider_id: str | None = None,
) -> dict[str, Any]:
    """Build PartToolGen tool-prior artifacts from mesh face labels."""

    mesh = load_obj_mesh(source_mesh)
    labels = load_npy_1d_int(face_labels_path)
    geometries = _segment_geometry(mesh, labels, min_face_count=max(1, int(min_face_count)))
    segmentation_mode = _safe_id(segmentation_mode or "sampart3d")
    provider_id = provider_id or f"{segmentation_mode}_tool_prior_provider"
    grounding_source = f"{segmentation_mode}_tool_prior"
    capture_id = observation_id or f"{tool_id}_{segmentation_mode}_mesh_prior"
    labels_artifact = str(face_labels_path)
    source_artifacts = [str(source_mesh), labels_artifact]
    max_faces = max(geometry.face_count for geometry in geometries)

    parts: dict[str, Any] = {}
    functional_parts: dict[str, Any] = {}
    grasp_references: dict[str, Any] = {}
    grounding_parts: dict[str, Any] = {}
    grounding_points: dict[str, Any] = {}
    grounding_axes: dict[str, Any] = {}
    grounding_surfaces: dict[str, Any] = {}
    grounding_regions: dict[str, Any] = {}
    tool_mins, tool_maxs = _minmax(mesh.vertices)
    tool_center = [(tool_mins[i] + tool_maxs[i]) * 0.5 for i in range(3)]
    grounding_points["tool_center"] = {
        "label": "tool_center",
        "candidate_type": "whole_tool_center",
        "xyz_tool": tool_center,
        **_grounding(
            observation_id=capture_id,
            method="tool_mesh_bbox_center",
            source_artifact=str(source_mesh),
            confidence=1.0,
            source=grounding_source,
        ),
    }

    for geometry in geometries:
        confidence = _segment_confidence(geometry, max_faces)
        roles = _roles_for_geometry(geometry, max_faces)
        part_ref = f"tool.parts.{geometry.part_id}"
        surface_patches = _surface_patches_for_geometry(
            geometry=geometry,
            observation_id=capture_id,
            source_artifact=labels_artifact,
            confidence=confidence,
            segmentation_mode=segmentation_mode,
            grounding_source=grounding_source,
        )
        parts[geometry.part_id] = {
            "label": f"{segmentation_mode}_segment_{geometry.label}",
            "role_candidates": roles,
            "mask_ref": f"{segmentation_mode}:face_label:{geometry.label}",
            "point_count": int(geometry.vertex_count),
            "face_count": int(geometry.face_count),
            "original_face_count": int(geometry.original_face_count),
            "connected_component_count": int(geometry.connected_component_count),
            "removed_disconnected_face_count": int(geometry.removed_disconnected_face_count),
            "centroid_xyz_tool": geometry.centroid,
            "obb": {
                "center_xyz_tool": geometry.center,
                "extent_xyz": geometry.extents,
                "axes_tool": IDENTITY_AXES,
            },
            "surface_patches": surface_patches,
            **_grounding(
                observation_id=capture_id,
                method=f"{segmentation_mode}_face_label_adapter",
                source_artifact=labels_artifact,
                confidence=confidence,
                source=grounding_source,
                extra_provenance={
                    "part_label": geometry.label,
                    "connected_component_index": geometry.connected_component_index,
                    "connected_component_count": geometry.connected_component_count,
                    "original_face_count": geometry.original_face_count,
                    "kept_face_count": geometry.face_count,
                    "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    "component_filter": f"keep_dominant_spatial_cluster_per_{segmentation_mode}_label",
                },
            ),
        }
        grounding_parts[geometry.part_id] = {
            "label": f"{segmentation_mode}_segment_{geometry.label}",
            "candidate_type": f"{segmentation_mode}_part",
            "part_ref": part_ref,
            "mask_ref": f"{segmentation_mode}:face_label:{geometry.label}",
            "role_candidates": roles,
            "center_xyz_tool": geometry.center,
            "extent_xyz_tool": geometry.extents,
            **_grounding(
                observation_id=capture_id,
                method=f"{segmentation_mode}_face_label_adapter",
                source_artifact=labels_artifact,
                confidence=confidence,
                source=grounding_source,
                extra_provenance={
                    "part_label": geometry.label,
                    "connected_component_index": geometry.connected_component_index,
                    "connected_component_count": geometry.connected_component_count,
                    "original_face_count": geometry.original_face_count,
                    "kept_face_count": geometry.face_count,
                    "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    "component_filter": f"keep_dominant_spatial_cluster_per_{segmentation_mode}_label",
                },
            ),
        }
        grounding_regions[f"{geometry.part_id}_obb"] = {
            "label": f"{geometry.part_id}_obb",
            "candidate_type": "tool_part_region",
            "part_ref": part_ref,
            "center_xyz_tool": geometry.center,
            "extent_xyz_tool": geometry.extents,
            **_grounding(
                observation_id=capture_id,
                method=f"{segmentation_mode}_face_label_adapter",
                source_artifact=labels_artifact,
                confidence=confidence,
                source=grounding_source,
                extra_provenance={
                    "part_label": geometry.label,
                    "connected_component_index": geometry.connected_component_index,
                    "connected_component_count": geometry.connected_component_count,
                    "original_face_count": geometry.original_face_count,
                    "kept_face_count": geometry.face_count,
                    "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    "component_filter": f"keep_dominant_spatial_cluster_per_{segmentation_mode}_label",
                },
            ),
        }
        grounding_points[f"{geometry.part_id}_center"] = {
            "label": f"{geometry.part_id}_center",
            "candidate_type": "geometry_anchor",
            "part_ref": part_ref,
            "xyz_tool": geometry.center,
            **_grounding(
                observation_id=capture_id,
                method=f"{segmentation_mode}_geometry_anchor",
                source_artifact=labels_artifact,
                confidence=confidence,
                source=grounding_source,
                extra_provenance={
                    "part_label": geometry.label,
                    "connected_component_index": geometry.connected_component_index,
                    "connected_component_count": geometry.connected_component_count,
                    "original_face_count": geometry.original_face_count,
                    "kept_face_count": geometry.face_count,
                    "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    "component_filter": f"keep_dominant_spatial_cluster_per_{segmentation_mode}_label",
                },
            ),
        }
        for patch_id, patch in surface_patches.items():
            normal = patch["normal_tool"]
            origin = patch["center_xyz_tool"]
            functional_id = f"{geometry.part_id}_{patch_id}"
            role = "endpoint_contact_patch" if "endpoint" in patch_id else "contact_patch"
            grounding_points[functional_id] = {
                "label": f"{functional_id}_center",
                "candidate_type": "geometry_surface_center",
                "part_ref": part_ref,
                "xyz_tool": origin,
                "normal_tool": normal,
                **_grounding(
                    observation_id=capture_id,
                    method=f"{segmentation_mode}_mesh_surface_center",
                    source_artifact=labels_artifact,
                    confidence=confidence,
                    source=grounding_source,
                    extra_provenance={
                        "part_label": geometry.label,
                        "patch_id": patch_id,
                        "role": role,
                        "support_faces": patch.get("support_faces"),
                        "connected_component_index": geometry.connected_component_index,
                        "connected_component_count": geometry.connected_component_count,
                        "original_face_count": geometry.original_face_count,
                        "kept_face_count": geometry.face_count,
                        "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    },
                ),
            }
            functional_parts[functional_id] = _functional_part_record(
                geometry=geometry,
                patch_id=patch_id,
                origin=origin,
                normal=normal,
                role=role,
                observation_id=capture_id,
                source_artifact=labels_artifact,
                confidence=confidence,
                segmentation_mode=segmentation_mode,
                grounding_source=grounding_source,
            )
            grounding_surfaces[functional_id] = {
                "label": functional_id,
                "candidate_type": "tool_part_surface",
                "part_ref": part_ref,
                "point_xyz_tool": origin,
                "normal_tool": normal,
                **_grounding(
                    observation_id=capture_id,
                    method=f"{segmentation_mode}_surface_anchor",
                    source_artifact=labels_artifact,
                    confidence=confidence,
                    source=grounding_source,
                    extra_provenance={
                        "part_label": geometry.label,
                        "patch_id": patch_id,
                        "connected_component_index": geometry.connected_component_index,
                        "connected_component_count": geometry.connected_component_count,
                        "original_face_count": geometry.original_face_count,
                        "kept_face_count": geometry.face_count,
                        "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                    },
                ),
            }
        grasp_references[f"{geometry.part_id}_grasp_center"] = {
            "position_tool": geometry.center,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            **_grounding(
                observation_id=capture_id,
                method=f"{segmentation_mode}_geometry_grasp_candidate",
                source_artifact=labels_artifact,
                confidence=max(0.25, confidence - 0.15),
                source=grounding_source,
                extra_provenance={
                    "part_label": geometry.label,
                    "connected_component_index": geometry.connected_component_index,
                    "connected_component_count": geometry.connected_component_count,
                    "original_face_count": geometry.original_face_count,
                    "kept_face_count": geometry.face_count,
                    "removed_disconnected_face_count": geometry.removed_disconnected_face_count,
                },
            ),
        }

    tool_part_graph = {
        "schema_version": "v0.1",
        "tool_id": tool_id,
        "tool_frame": "tool",
        "parts": parts,
        "relations": [
            {
                "type": "part_of",
                "subject": f"tool.parts.{part_id}",
                "object": f"tool.{tool_id}",
                "confidence": float(parts[part_id]["confidence"]),
                "source": grounding_source,
            }
            for part_id in sorted(parts)
        ],
        "metadata": {
            "created_by": "parttoolgen.providers.tool_prior.sampart3d_mesh_candidates",
            "geometry_mode": "prior",
            "segmentation_mode": segmentation_mode,
            "notes": (
                f"{segmentation_mode} face labels are converted in the stable "
                "SimToolReal tool/asset frame."
            ),
        },
    }
    functional_tool_parts = {
        "schema_version": "v0.1",
        "tool_id": tool_id,
        "tool_frame": "tool",
        "functional_parts": functional_parts,
        "grasp_references": grasp_references,
        "metadata": {
            "created_by": "parttoolgen.providers.tool_prior.sampart3d_mesh_candidates",
            "geometry_mode": "prior",
            "notes": (
                "Functional refs are candidate contact/grasp frames derived "
                f"from {segmentation_mode} mesh segments."
            ),
        },
    }
    grounding_candidates = {
        "schema_version": "v0.1",
        "grounding_id": f"{capture_id}_tool_candidates",
        "observation_id": capture_id,
        "frames": {
            "tool": {
                "frame_type": "simtoolreal_asset_root",
                "description": "All *_tool fields are expressed in the stable SimToolReal asset/root frame.",
            }
        },
        "marks": {},
        "points": grounding_points,
        "regions": grounding_regions,
        "axes": grounding_axes,
        "surfaces": grounding_surfaces,
        "parts": grounding_parts,
        "metadata": {
            "created_by": "parttoolgen.providers.tool_prior.sampart3d_mesh_candidates",
            "grounding_mode": "model_grounded_with_prior",
            "anchor_mode": f"{segmentation_mode}_geometry_functional_frames",
            "tool_frame": "tool",
            "source_mesh": str(source_mesh),
            "face_labels": labels_artifact,
        },
    }
    provider_contract = {
        "schema_version": "v0.1",
        "provider_id": provider_id,
        "provider_type": "tool_prior",
        "source_kind": "MESH_PRIOR",
        "deployable": True,
        "oracle": False,
        "capabilities": ["tool_functional_parts"],
        "frame_contracts": ["simtoolreal_asset_root"],
        "provenance": {
            "method": f"{segmentation_mode}_face_labels_geometry_functional_frames",
            "source_artifacts": source_artifacts,
            "timestamp_or_capture_id": capture_id,
            "notes": (
                f"Offline tool mesh prior. {segmentation_mode} regions produce executable "
                "geometry functional point, axis, surface, and grasp candidates."
            ),
        },
        "metadata": {
            "segment_count": len(parts),
            "functional_candidate_count": len(functional_parts),
            "grasp_candidate_count": len(grasp_references),
        },
    }
    return {
        "tool_part_graph": tool_part_graph,
        "functional_tool_parts": functional_tool_parts,
        "grounding_candidates": grounding_candidates,
        "provider_contract": provider_contract,
    }
