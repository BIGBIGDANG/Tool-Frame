"""Pure SAMPart3D mesh projection utilities.

This module contains only geometry/projection helpers used by 5.15 visual
evidence and sanity checks. It deliberately has no removed visual-feature
route, keypoint proposal, or feature-clustering dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ToGo.providers.tool_prior.sampart3d_mesh_candidates import (
    load_npy_1d_int,
    load_obj_mesh,
)


SAMPart3D_RENDER_ROTATION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FaceGeometry:
    label: int
    part_id: str
    centroid_tool: np.ndarray
    centroid_render: np.ndarray
    normal_tool: np.ndarray
    normal_render: np.ndarray


def _cross_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


def _unit_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector.astype(np.float64) / norm


def _face_centroid_and_normal(vertices: list[list[float]], face: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray([vertices[index] for index in face], dtype=np.float64)
    centroid = pts.mean(axis=0)
    normal = _unit_np(_cross_np(pts[1] - pts[0], pts[2] - pts[0]))
    return centroid, normal


def face_geometry(
    *,
    source_mesh: str | Path,
    face_labels_path: str | Path | None,
    meta: dict[str, Any],
    projection_mode: str,
    min_part_face_count: int = 1,
    selected_part_ids: set[str] | None = None,
) -> list[FaceGeometry]:
    mesh = load_obj_mesh(source_mesh)
    labels = (
        load_npy_1d_int(face_labels_path)
        if face_labels_path is not None
        else [0 for _ in mesh.faces]
    )
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"face-label count ({len(labels)}) does not match mesh face count ({len(mesh.faces)})"
        )
    label_counts: dict[int, int] = {}
    for label in labels:
        if int(label) >= 0:
            label_counts[int(label)] = label_counts.get(int(label), 0) + 1
    scale = float(meta["scaling_factor"])
    offset = np.asarray(meta["mesh_offset"], dtype=np.float64)
    use_sampart3d_rotation = projection_mode == "sampart3d_glb"
    result: list[FaceGeometry] = []
    for face, label in zip(mesh.faces, labels):
        if int(label) < 0:
            continue
        if label_counts.get(int(label), 0) < max(1, int(min_part_face_count)):
            continue
        part_id = f"part_{int(label):03d}" if face_labels_path is not None else "whole_tool"
        if selected_part_ids is not None and part_id not in selected_part_ids:
            continue
        centroid_tool, normal_tool = _face_centroid_and_normal(mesh.vertices, face)
        if use_sampart3d_rotation:
            centroid_render_base = centroid_tool @ SAMPart3D_RENDER_ROTATION
            normal_render_base = normal_tool @ SAMPart3D_RENDER_ROTATION
        else:
            centroid_render_base = centroid_tool
            normal_render_base = normal_tool
        result.append(
            FaceGeometry(
                label=int(label),
                part_id=part_id,
                centroid_tool=centroid_tool,
                centroid_render=centroid_render_base * scale + offset,
                normal_tool=normal_tool,
                normal_render=_unit_np(normal_render_base),
            )
        )
    return result


def project_points(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    transform: list[list[float]],
    camera_angle_x: float,
    image_width: int,
    image_height: int,
    front_facing_cos: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(transform, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (homog @ w2c.T)[:, :3]
    rotation = w2c[:3, :3]
    normal_cam = normals @ rotation.T
    z = cam[:, 2]
    front = z < -1e-6
    tan_half = math.tan(float(camera_angle_x) * 0.5)
    x_ndc = (cam[:, 0] / np.maximum(-z, 1e-9)) / tan_half
    y_ndc = (cam[:, 1] / np.maximum(-z, 1e-9)) / tan_half
    u = (x_ndc + 1.0) * 0.5 * float(image_width)
    v = (1.0 - y_ndc) * 0.5 * float(image_height)
    view_to_camera = -cam
    view_to_camera_norm = np.linalg.norm(view_to_camera, axis=1, keepdims=True)
    view_to_camera_unit = view_to_camera / np.maximum(view_to_camera_norm, 1e-9)
    facing = np.sum(normal_cam * view_to_camera_unit, axis=1) >= float(front_facing_cos)
    inside = (
        front
        & facing
        & (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )
    pixels = np.stack([u, v], axis=1)
    return inside, pixels, cam


def projected_bounds_mask(
    *,
    pixels: np.ndarray,
    cam: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    return (
        (cam[:, 2] < -1e-6)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_height))
    )


def occlusion_visible_mask(
    *,
    candidate_mask: np.ndarray,
    pixels: np.ndarray,
    cam: np.ndarray,
    image_width: int,
    image_height: int,
    occlusion_window_px: int,
    occlusion_depth_tolerance: float,
    occlusion_depth_rel_tolerance: float,
) -> np.ndarray:
    projected = projected_bounds_mask(
        pixels=pixels,
        cam=cam,
        image_width=image_width,
        image_height=image_height,
    )
    candidate = candidate_mask & projected
    if not np.any(candidate):
        return candidate

    depth = -cam[:, 2]
    zbuffer = np.full((image_height, image_width), np.inf, dtype=np.float64)
    rounded = np.rint(pixels).astype(np.int64)
    for face_index in np.nonzero(projected)[0]:
        x = int(np.clip(rounded[face_index, 0], 0, image_width - 1))
        y = int(np.clip(rounded[face_index, 1], 0, image_height - 1))
        face_depth = float(depth[face_index])
        if face_depth < zbuffer[y, x]:
            zbuffer[y, x] = face_depth

    radius = max(0, int(occlusion_window_px))
    abs_tolerance = max(0.0, float(occlusion_depth_tolerance))
    rel_tolerance = max(0.0, float(occlusion_depth_rel_tolerance))
    visible = np.zeros_like(candidate, dtype=bool)
    for face_index in np.nonzero(candidate)[0]:
        x = int(np.clip(rounded[face_index, 0], 0, image_width - 1))
        y = int(np.clip(rounded[face_index, 1], 0, image_height - 1))
        x0 = max(0, x - radius)
        x1 = min(image_width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(image_height, y + radius + 1)
        nearest = float(np.min(zbuffer[y0:y1, x0:x1]))
        tolerance = max(abs_tolerance, rel_tolerance * float(depth[face_index]))
        visible[face_index] = math.isfinite(nearest) and float(depth[face_index]) <= nearest + tolerance
    return visible
