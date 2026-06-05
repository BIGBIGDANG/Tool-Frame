#!/usr/bin/env python3
"""Render 5.15-style SAMPart3D tool-part evidence.

This renderer is intentionally separate from the older anonymous region overlay
flow. It supports the paper-style two-stage assets:

  SAMPart3D parts -> whole-mesh anonymous part overview -> candidate part cards
  selected part -> bbox overview -> one highlighted bbox-face card per F marker

Legacy per-face/contact-choice paths are intentionally disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.providers.tool_prior.sampart3d_projection_geometry import (  # noqa: E402
    SAMPart3D_RENDER_ROTATION,
    project_points,
)
from parttoolgen.providers.tool_prior.sampart3d_mesh_candidates import (  # noqa: E402
    _dominant_spatial_component_cluster,
    _face_connected_components,
    load_obj_mesh,
)
from parttoolgen.providers.tool_prior.sampart3d_projection_helpers import (  # noqa: E402
    PALETTE,
    _font,
    _make_contact_sheet,
    _mesh_foreground_mask,
    _presentation_image,
    _project_render_points,
    _project_view_faces,
    _projection_context,
)
from parttoolgen.providers.tool_prior.visualize_sampart3d_region_overlays import (  # noqa: E402
    _draw_projected_faces,
    _face_indices_for_label,
    _part_rows,
    _project_mesh_vertices,
    _projection_artifact_from_geometry_prior,
    _select_views,
)


AXIS_NAMES = ("x", "y", "z")
BBOX_EDGES = [
    (a, b)
    for a, signs_a in enumerate(product((-1.0, 1.0), repeat=3))
    for b, signs_b in enumerate(product((-1.0, 1.0), repeat=3))
    if a < b and sum(sa != sb for sa, sb in zip(signs_a, signs_b)) == 1
]


def _filtered_face_indices_by_label(anchor_artifact: Mapping[str, Any], labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mirror provider-side disconnected-island filtering for visual evidence.

    Tool priors already derive OBBs and candidate contacts from filtered SAMPart3D
    labels.  Rendering must use the same kept faces, otherwise the VLM can see a
    stale part shape while choosing refs from the filtered geometry.
    """

    try:
        mesh = load_obj_mesh(anchor_artifact["source_mesh"])
    except Exception:
        return {}
    if len(mesh.faces) != len(labels):
        return {}
    result: dict[int, np.ndarray] = {}
    for label in sorted({int(value) for value in labels.tolist() if int(value) >= 0}):
        original = [index for index, value in enumerate(labels.tolist()) if int(value) == label]
        if not original:
            continue
        components = _face_connected_components(mesh, original)
        kept = _dominant_spatial_component_cluster(mesh, components) if components else []
        if kept:
            result[label] = np.asarray(sorted(kept), dtype=np.int64)
    return result


def _filtered_face_indices_for_label(
    *,
    context: Mapping[str, Any],
    labels: np.ndarray,
    inside: np.ndarray,
    label: int,
    face_count: int,
) -> np.ndarray:
    filtered = context.get("filtered_face_indices_by_label")
    if isinstance(filtered, Mapping) and int(label) in filtered:
        indices = np.asarray(filtered[int(label)], dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(inside))]
        return indices[inside[indices]]
    return _face_indices_for_label(
        labels=labels,
        inside=inside,
        label=label,
        face_count=face_count,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _crop_to_mask(image: Image.Image, mask: np.ndarray, *, pad: int) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image
    width, height = image.size
    left = max(0, int(xs.min()) - pad)
    right = min(width, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(height, int(ys.max()) + pad + 1)
    return image.crop((left, top, right, bottom))


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    scale = float(width) / float(max(image.width, 1))
    return image.resize((width, max(1, int(round(image.height * scale)))))


def _fit_image(image: Image.Image, *, max_width: int, max_height: int) -> Image.Image:
    scale = min(float(max_width) / float(max(image.width, 1)), float(max_height) / float(max(image.height, 1)), 1.0)
    return image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))))


def _draw_marker_circle(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    outline: tuple[int, int, int] = (20, 20, 20),
    fill: tuple[int, int, int] = (255, 255, 255),
    radius_min: int = 13,
    pad: int = 7,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    radius = max(radius_min, int(max(text_w, text_h) * 0.62) + pad)
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    draw.text((x - text_w * 0.5 - bbox[0], y - text_h * 0.5 - bbox[1]), text, fill=outline, font=font)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 10,
    gap: int = 7,
) -> None:
    left, top, right, bottom = box
    for x in range(left, right, dash + gap):
        draw.line((x, top, min(x + dash, right), top), fill=fill, width=width)
        draw.line((x, bottom, min(x + dash, right), bottom), fill=fill, width=width)
    for y in range(top, bottom, dash + gap):
        draw.line((left, y, left, min(y + dash, bottom)), fill=fill, width=width)
        draw.line((right, y, right, min(y + dash, bottom)), fill=fill, width=width)


def _draw_badge(image: Image.Image, text: str, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = 8
    pad_y = 5
    rect = (8, 8, 8 + text_w + 2 * pad_x, 8 + text_h + 2 * pad_y)
    draw.rounded_rectangle(rect, radius=4, fill=(255, 255, 255), outline=(30, 30, 30), width=2)
    draw.text((rect[0] + pad_x - bbox[0], rect[1] + pad_y - bbox[1]), text, fill=(20, 20, 20), font=font)


def _part_mask(
    *,
    image_size: tuple[int, int],
    mesh_geometry: Mapping[str, Any],
    vertex_front: np.ndarray,
    vertex_pixels: np.ndarray,
    face_indices: np.ndarray,
) -> np.ndarray:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    faces = mesh_geometry["faces"]
    for face_index in face_indices:
        if int(face_index) < 0 or int(face_index) >= len(faces):
            continue
        face = faces[int(face_index)]
        if not any(bool(vertex_front[index]) for index in face):
            continue
        polygon = [(float(vertex_pixels[index, 0]), float(vertex_pixels[index, 1])) for index in face]
        draw.polygon(polygon, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    return np.asarray(mask, dtype=np.uint8) > 0


def _isolated_part_full(
    *,
    raw: Image.Image,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
    part_label: int,
) -> tuple[Image.Image, np.ndarray] | None:
    projection = _project_view_faces(dict(context), view_index)
    vertex_front, vertex_pixels = _project_mesh_vertices(dict(context), view_index)
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    face_indices = _filtered_face_indices_for_label(
        context=context,
        labels=mesh_labels,
        inside=projection["inside"],
        label=part_label,
        face_count=len(mesh_faces),
    )
    if len(face_indices) == 0:
        return None
    foreground = _mesh_foreground_mask(mesh_geometry=dict(context["mesh_geometry"]), meta=dict(meta), view_index=view_index)
    base = _presentation_image(raw, reference=raw, foreground_mask=foreground, crop=False).convert("RGB")
    mask = _part_mask(
        image_size=base.size,
        mesh_geometry=context["mesh_geometry"],
        vertex_front=vertex_front,
        vertex_pixels=vertex_pixels,
        face_indices=face_indices,
    )
    if not mask.any():
        return None
    base_arr = np.asarray(base, dtype=np.uint8)
    isolated = np.full_like(base_arr, 255, dtype=np.uint8)
    isolated[mask] = base_arr[mask]
    image = Image.fromarray(isolated, mode="RGB")
    return image, mask


def _isolated_part_panel(
    *,
    raw: Image.Image,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
    part_label: int,
    marker: str,
    badge_font: ImageFont.ImageFont,
    crop_pad: int,
) -> Image.Image | None:
    isolated = _isolated_part_full(
        raw=raw,
        context=context,
        meta=meta,
        view_index=view_index,
        part_label=part_label,
    )
    if isolated is None:
        return None
    image, mask = isolated
    image = _crop_to_mask(image, mask, pad=crop_pad)
    _draw_badge(image, marker, badge_font)
    return image


def _isolated_part_crop(
    *,
    raw: Image.Image,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
    part_label: int,
    crop_pad: int,
) -> tuple[Image.Image, int] | None:
    isolated = _isolated_part_full(
        raw=raw,
        context=context,
        meta=meta,
        view_index=view_index,
        part_label=part_label,
    )
    if isolated is None:
        return None
    image, mask = isolated
    area = int(mask.sum())
    return _crop_to_mask(image, mask, pad=crop_pad), area


def _best_isolated_part_crop(
    *,
    render_paths: list[Path],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_indices: list[int],
    part_label: int,
    crop_pad: int,
) -> tuple[Image.Image, int] | None:
    best: tuple[Image.Image, int, int] | None = None
    for view_index in view_indices:
        raw = Image.open(render_paths[view_index]).convert("RGB")
        candidate = _isolated_part_crop(
            raw=raw,
            context=context,
            meta=meta,
            view_index=view_index,
            part_label=part_label,
            crop_pad=crop_pad,
        )
        if candidate is None:
            continue
        panel, area = candidate
        if best is None or area > best[1]:
            best = (panel, area, view_index)
    if best is None:
        return None
    return best[0], best[2]


def _tool_points_to_render(points_tool: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    return (
        np.asarray(points_tool, dtype=np.float64) @ np.asarray(SAMPart3D_RENDER_ROTATION, dtype=np.float64)
    ) * float(meta["scaling_factor"]) + np.asarray(meta["mesh_offset"], dtype=np.float64)


def _project_tool_points(
    *,
    points_tool: np.ndarray,
    meta: Mapping[str, Any],
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _project_render_points(
        points=_tool_points_to_render(points_tool, meta),
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=int(meta.get("width") or 512),
        image_height=int(meta.get("height") or 512),
    )


def _axis_vector(part: Mapping[str, Any], axis_name: str) -> np.ndarray:
    axes = part.get("obb", {}).get("axes_tool", {})
    vector = axes.get(axis_name) if isinstance(axes, Mapping) else None
    if not (isinstance(vector, list) and len(vector) == 3):
        index = AXIS_NAMES.index(axis_name)
        vector = [1.0 if axis == index else 0.0 for axis in range(3)]
    array = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm <= 1e-12:
        index = AXIS_NAMES.index(axis_name)
        array = np.asarray([1.0 if axis == index else 0.0 for axis in range(3)], dtype=np.float64)
        norm = 1.0
    return array / norm


def _obb_corners(part: Mapping[str, Any]) -> np.ndarray | None:
    obb = part.get("obb", {})
    center = obb.get("center_xyz_tool")
    extents = obb.get("extent_xyz")
    if not (isinstance(center, list) and len(center) == 3 and isinstance(extents, list) and len(extents) == 3):
        return None
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extents, dtype=np.float64)
    if not (np.isfinite(center_array).all() and np.isfinite(extent_array).all()):
        return None
    corners: list[np.ndarray] = []
    for signs in product((-1.0, 1.0), repeat=3):
        point = center_array.copy()
        for axis_index, axis_name in enumerate(AXIS_NAMES):
            point += float(signs[axis_index]) * 0.5 * float(extent_array[axis_index]) * _axis_vector(part, axis_name)
        corners.append(point)
    return np.stack(corners, axis=0)


def _obb_face_corners(part: Mapping[str, Any], *, axis_name: str, polarity: str) -> np.ndarray | None:
    obb = part.get("obb", {})
    center = obb.get("center_xyz_tool")
    extents = obb.get("extent_xyz")
    if axis_name not in AXIS_NAMES or polarity not in {"negative", "positive"}:
        return None
    if not (isinstance(center, list) and len(center) == 3 and isinstance(extents, list) and len(extents) == 3):
        return None
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extents, dtype=np.float64)
    if not (np.isfinite(center_array).all() and np.isfinite(extent_array).all()):
        return None
    fixed_axis_index = AXIS_NAMES.index(axis_name)
    varying_indices = [idx for idx in range(3) if idx != fixed_axis_index]
    fixed_sign = -1.0 if polarity == "negative" else 1.0
    corners: list[np.ndarray] = []
    for signs in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        point = center_array.copy()
        point += fixed_sign * 0.5 * float(extent_array[fixed_axis_index]) * _axis_vector(
            part, AXIS_NAMES[fixed_axis_index]
        )
        for sign, axis_index in zip(signs, varying_indices):
            point += sign * 0.5 * float(extent_array[axis_index]) * _axis_vector(part, AXIS_NAMES[axis_index])
        corners.append(point)
    return np.stack(corners, axis=0)


def _tool_vectors_to_render(vectors_tool: np.ndarray) -> np.ndarray:
    return np.asarray(vectors_tool, dtype=np.float64) @ np.asarray(SAMPart3D_RENDER_ROTATION, dtype=np.float64)


def _zbuffer_from_projection(
    *,
    pixels: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    depth = -cam[:, 2]
    zbuffer = np.full((image_height, image_width), np.inf, dtype=np.float64)
    projected = (
        mask
        & (cam[:, 2] < -1e-6)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_height))
    )
    rounded = np.rint(pixels).astype(np.int64)
    for index in np.nonzero(projected)[0]:
        x = int(np.clip(rounded[index, 0], 0, image_width - 1))
        y = int(np.clip(rounded[index, 1], 0, image_height - 1))
        value = float(depth[index])
        if value < zbuffer[y, x]:
            zbuffer[y, x] = value
    return zbuffer


def _visible_against_zbuffer(
    *,
    candidate_mask: np.ndarray,
    pixels: np.ndarray,
    cam: np.ndarray,
    zbuffer: np.ndarray,
    window_px: int,
    depth_tolerance: float,
    depth_rel_tolerance: float,
) -> np.ndarray:
    image_height, image_width = zbuffer.shape
    projected = (
        candidate_mask
        & (cam[:, 2] < -1e-6)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_height))
    )
    visible = np.zeros_like(projected, dtype=bool)
    if not np.any(projected):
        return visible
    depth = -cam[:, 2]
    rounded = np.rint(pixels).astype(np.int64)
    radius = max(0, int(window_px))
    for index in np.nonzero(projected)[0]:
        x = int(np.clip(rounded[index, 0], 0, image_width - 1))
        y = int(np.clip(rounded[index, 1], 0, image_height - 1))
        x0 = max(0, x - radius)
        x1 = min(image_width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(image_height, y + radius + 1)
        nearest = float(np.min(zbuffer[y0:y1, x0:x1]))
        tolerance = max(float(depth_tolerance), float(depth_rel_tolerance) * float(depth[index]))
        visible[index] = math.isfinite(nearest) and float(depth[index]) <= nearest + tolerance
    return visible


def _surface_contact_projection(
    *,
    contact_rows: list[dict[str, Any]],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not contact_rows:
        return np.zeros(0, dtype=bool), np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)
    points_tool = np.asarray([row["xyz_tool"] for row in contact_rows], dtype=np.float64)
    normals_tool = np.asarray([row["normal_tool"] for row in contact_rows], dtype=np.float64)
    points_render = _tool_points_to_render(points_tool, meta)
    normals_render = _tool_vectors_to_render(normals_tool)
    image_width = int(meta.get("width") or 512)
    image_height = int(meta.get("height") or 512)
    facing, pixels, cam = project_points(
        points=points_render,
        normals=normals_render,
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=image_width,
        image_height=image_height,
        front_facing_cos=-0.08,
    )
    blocker_mask, blocker_pixels, blocker_cam = project_points(
        points=np.asarray(context["points_render"], dtype=np.float64),
        normals=np.asarray(context["normals_render"], dtype=np.float64),
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=image_width,
        image_height=image_height,
        front_facing_cos=-1.0,
    )
    zbuffer = _zbuffer_from_projection(
        pixels=blocker_pixels,
        cam=blocker_cam,
        mask=blocker_mask,
        image_width=image_width,
        image_height=image_height,
    )
    visible = _visible_against_zbuffer(
        candidate_mask=facing,
        pixels=pixels,
        cam=cam,
        zbuffer=zbuffer,
        window_px=max(12, int(context.get("occlusion_window_px", 4))),
        depth_tolerance=max(0.03, float(context.get("occlusion_depth_tolerance", 0.02))),
        depth_rel_tolerance=max(0.05, float(context.get("occlusion_depth_rel_tolerance", 0.01))),
    )
    return facing, pixels, visible


def _candidate_patch_id(candidate_id: str, candidate: Mapping[str, Any]) -> str:
    provenance = candidate.get("provenance")
    patch_id = provenance.get("patch_id") if isinstance(provenance, Mapping) else None
    return str(patch_id or candidate_id)


def _axis_polarity_from_patch(patch_id: str) -> tuple[str | None, str | None]:
    for axis_name in AXIS_NAMES:
        for polarity in ("negative", "positive"):
            if patch_id.startswith(f"{axis_name}_{polarity}_"):
                return axis_name, polarity
    return None, None


def _contact_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    axis_name, _ = _axis_polarity_from_patch(str(row.get("patch_id") or ""))
    axis_index = AXIS_NAMES.index(str(axis_name)) if axis_name in AXIS_NAMES else len(AXIS_NAMES)
    polarity = str(row.get("polarity") or "")
    polarity_index = 0 if polarity == "negative" else 1 if polarity == "positive" else 2
    return axis_index, polarity_index, str(row.get("point_ref") or "")


def _contact_marker_table(
    *,
    part_id: str,
    part_ref: str,
    marker: str,
    grounding_candidates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    points = grounding_candidates.get("points", {})
    surfaces = grounding_candidates.get("surfaces", {})
    if not isinstance(points, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in sorted(points.items()):
        if not isinstance(candidate, Mapping) or candidate.get("part_ref") != part_ref:
            continue
        xyz = candidate.get("xyz_tool")
        normal = candidate.get("normal_tool")
        if not (isinstance(xyz, list) and len(xyz) == 3 and isinstance(normal, list) and len(normal) == 3):
            continue
        surface_ref = f"candidate.surfaces.{candidate_id}" if isinstance(surfaces, Mapping) and candidate_id in surfaces else None
        if surface_ref is None:
            continue
        patch_id = _candidate_patch_id(str(candidate_id), candidate)
        _, polarity = _axis_polarity_from_patch(patch_id)
        rows.append(
            {
                "part_marker": marker,
                "part_id": part_id,
                "part_ref": part_ref,
                "point_ref": f"candidate.points.{candidate_id}",
                "surface_ref": surface_ref,
                "polarity": polarity,
                "patch_id": patch_id,
                "role": candidate.get("role"),
                "xyz_tool": [float(value) for value in xyz],
                "normal_tool": [float(value) for value in normal],
            }
        )
    rows = sorted(rows, key=_contact_sort_key)
    for index, row in enumerate(rows, start=1):
        marker_id = f"F{index}"
        row["face_marker"] = marker_id
        row["contact_marker"] = marker_id
        row["selection_semantics"] = "contact_point_bbox_face"
        row["derived_point_ref"] = row["point_ref"]
        row["source_surface_ref"] = f"candidate.surfaces.{str(row['point_ref']).split('.')[-1]}"
    return rows


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    dx = ex - sx
    dy = ey - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 15.0
    spread = 8.0
    points = [
        (ex, ey),
        (ex - ux * head + px * spread, ey - uy * head + py * spread),
        (ex - ux * head - px * spread, ey - uy * head - py * spread),
    ]
    draw.polygon(points, fill=color)


def _draw_projected_obb(
    draw: ImageDraw.ImageDraw,
    *,
    part: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
    color: tuple[int, int, int] = (32, 120, 210),
    width: int = 3,
) -> np.ndarray | None:
    corners = _obb_corners(part)
    if corners is None:
        return None
    front, pixels = _project_tool_points(points_tool=corners, meta=meta, view_index=view_index)
    for a, b in BBOX_EDGES:
        if not (bool(front[a]) and bool(front[b])):
            continue
        draw.line(
            (
                float(pixels[a, 0]),
                float(pixels[a, 1]),
                float(pixels[b, 0]),
                float(pixels[b, 1]),
            ),
            fill=color,
            width=width,
        )
    return pixels


def _best_contact_view_indices(
    *,
    contact_rows: list[dict[str, Any]],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_indices: list[int],
    max_views: int,
) -> list[int]:
    if not contact_rows:
        return []
    scored: list[tuple[float, int]] = []
    for view_index in view_indices:
        _front, pixels, visible_mask = _surface_contact_projection(
            contact_rows=contact_rows,
            context=context,
            meta=meta,
            view_index=view_index,
        )
        visible = [pixels[index] for index, is_visible in enumerate(visible_mask) if bool(is_visible)]
        if not visible:
            continue
        min_separation = 0.0
        if len(visible) >= 2:
            min_separation = min(
                float(np.linalg.norm(a - b))
                for idx, a in enumerate(visible)
                for b in visible[idx + 1 :]
            )
        span = float(np.linalg.norm(np.max(visible, axis=0) - np.min(visible, axis=0))) if visible else 0.0
        score = len(visible) * 1000.0 + min_separation + 0.08 * span
        scored.append((score, view_index))
    if not scored:
        return view_indices[:max_views]
    scored.sort(reverse=True)
    selected: list[int] = []
    for _score, view_index in scored:
        if view_index in selected:
            continue
        selected.append(view_index)
        if len(selected) >= max_views:
            break
    return selected


def _svg_num(value: float) -> str:
    if not math.isfinite(float(value)):
        return "0"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _svg_rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"


def _svg_polygon(
    points: np.ndarray,
    *,
    fill: tuple[int, int, int],
    opacity: float,
    stroke: tuple[int, int, int] | None = None,
    stroke_width: float = 0.0,
) -> str:
    coords = " ".join(f"{_svg_num(x)},{_svg_num(y)}" for x, y in points[:, :2])
    stroke_attrs = ""
    if stroke is not None and stroke_width > 0.0:
        stroke_attrs = (
            f' stroke="{_svg_rgb(stroke)}" stroke-width="{_svg_num(stroke_width)}"'
            ' stroke-linejoin="round"'
        )
    return (
        f'<polygon points="{coords}" fill="{_svg_rgb(fill)}" '
        f'fill-opacity="{_svg_num(opacity)}"{stroke_attrs}/>'
    )


def _svg_line(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    color: tuple[int, int, int],
    width: float,
    opacity: float = 1.0,
) -> str:
    return (
        f'<line x1="{_svg_num(p0[0])}" y1="{_svg_num(p0[1])}" '
        f'x2="{_svg_num(p1[0])}" y2="{_svg_num(p1[1])}" '
        f'stroke="{_svg_rgb(color)}" stroke-width="{_svg_num(width)}" '
        f'stroke-opacity="{_svg_num(opacity)}" stroke-linecap="round"/>'
    )


def _shade_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(channel * factor)))) for channel in color)


def _mesh_base_color(label: int) -> tuple[int, int, int]:
    if int(label) == 2:
        return (18, 18, 18)
    return (136, 165, 176)


def _camera_face_depths(
    *,
    vertices_render: np.ndarray,
    faces: np.ndarray,
    meta: Mapping[str, Any],
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(meta["transforms"][view_index], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate([vertices_render, np.ones((len(vertices_render), 1), dtype=np.float64)], axis=1)
    cam = (homog @ w2c.T)[:, :3]
    depths = np.mean(cam[faces, 2], axis=1)
    view_dir = -c2w[:3, 2]
    norm = np.linalg.norm(view_dir)
    if norm > 1e-12:
        view_dir = view_dir / norm
    return depths, view_dir


def _face_normals_render(vertices_render: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices_render[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _project_tool_points_to_svg(
    *,
    points_tool: np.ndarray,
    meta: Mapping[str, Any],
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _project_tool_points(points_tool=points_tool, meta=meta, view_index=view_index)


def _svg_panel_crop(
    *,
    foreground: np.ndarray,
    extra_points: list[np.ndarray],
    image_size: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(foreground)
    width, height = image_size
    point_sets = [np.column_stack([xs, ys]).astype(np.float64)] if len(xs) else []
    point_sets.extend(points[:, :2] for points in extra_points if len(points))
    if not point_sets:
        return 0, 0, width, height
    pts = np.vstack(point_sets)
    left = max(0, int(math.floor(float(np.nanmin(pts[:, 0])))) - pad)
    right = min(width, int(math.ceil(float(np.nanmax(pts[:, 0])))) + pad + 1)
    top = max(0, int(math.floor(float(np.nanmin(pts[:, 1])))) - pad)
    bottom = min(height, int(math.ceil(float(np.nanmax(pts[:, 1])))) + pad + 1)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _svg_mesh_panel(
    *,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_index: int,
    part: Mapping[str, Any] | None = None,
    part_label: int | None = None,
    contact_rows: list[dict[str, Any]] | None = None,
    include_contact_region: bool = False,
    mesh_opacity: float = 0.96,
    selected_part_opacity: float = 0.28,
    pad: int = 30,
) -> dict[str, Any]:
    mesh_geometry = context["mesh_geometry"]
    vertices_render = np.asarray(mesh_geometry["vertices_render"], dtype=np.float64)
    faces = np.asarray(mesh_geometry["faces"], dtype=np.int64)
    labels = np.asarray(mesh_geometry["labels"], dtype=np.int64)
    vertex_front, vertex_pixels = _project_mesh_vertices(dict(context), view_index)
    foreground = _mesh_foreground_mask(mesh_geometry=dict(mesh_geometry), meta=dict(meta), view_index=view_index)
    depths, view_dir = _camera_face_depths(vertices_render=vertices_render, faces=faces, meta=meta, view_index=view_index)
    normals = _face_normals_render(vertices_render, faces)
    order = np.argsort(depths)
    extra_points: list[np.ndarray] = []
    face_polygons: list[np.ndarray] = []
    bbox_pixels: np.ndarray | None = None
    if part is not None:
        corners = _obb_corners(part)
        if corners is not None:
            _front, bbox_pixels = _project_tool_points_to_svg(points_tool=corners, meta=meta, view_index=view_index)
            extra_points.append(bbox_pixels)
    if include_contact_region and part is not None:
        for row in contact_rows or []:
            axis_name, polarity = _axis_polarity_from_patch(str(row.get("patch_id") or ""))
            if axis_name is None or polarity is None:
                continue
            face_corners = _obb_face_corners(part, axis_name=axis_name, polarity=polarity)
            if face_corners is None:
                continue
            _front, face_pixels = _project_tool_points_to_svg(
                points_tool=face_corners,
                meta=meta,
                view_index=view_index,
            )
            face_polygons.append(face_pixels)
            extra_points.append(face_pixels)

    left, top, right, bottom = _svg_panel_crop(
        foreground=foreground,
        extra_points=extra_points,
        image_size=(int(meta.get("width") or 512), int(meta.get("height") or 512)),
        pad=pad,
    )
    shift = np.asarray([left, top], dtype=np.float64)
    lines: list[str] = []
    for face_index in order:
        face = faces[int(face_index)]
        if not any(bool(vertex_front[index]) for index in face):
            continue
        points = vertex_pixels[face] - shift[None, :]
        label = int(labels[int(face_index)]) if int(face_index) < len(labels) else -1
        base = _mesh_base_color(label)
        light = 0.66 + 0.34 * abs(float(normals[int(face_index)] @ view_dir))
        fill = _shade_color(base, light)
        lines.append(_svg_polygon(points, fill=fill, opacity=mesh_opacity))
    if part_label is not None:
        selected_indices = np.nonzero(labels == int(part_label))[0]
        color = PALETTE[1 % len(PALETTE)]
        for face_index in selected_indices:
            face = faces[int(face_index)]
            if not any(bool(vertex_front[index]) for index in face):
                continue
            points = vertex_pixels[face] - shift[None, :]
            lines.append(
                _svg_polygon(
                    points,
                    fill=color,
                    opacity=selected_part_opacity,
                    stroke=color,
                    stroke_width=0.35,
                )
            )
    if bbox_pixels is not None:
        local_bbox = bbox_pixels - shift[None, :]
        for a, b in BBOX_EDGES:
            lines.append(_svg_line(local_bbox[a], local_bbox[b], color=(32, 120, 210), width=2.2, opacity=0.92))
    for face_pixels in face_polygons:
        local_face = face_pixels - shift[None, :]
        lines.append(
            _svg_polygon(
                local_face,
                fill=(255, 196, 35),
                opacity=0.46,
                stroke=(230, 106, 0),
                stroke_width=2.2,
            )
        )
    return {
        "width": int(right - left),
        "height": int(bottom - top),
        "lines": lines,
    }


def _write_svg_document(path: Path, *, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" version="1.1">',
        '<g shape-rendering="geometricPrecision">',
        *body,
        "</g>",
        "</svg>",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_contact_sheet(path: Path, panels: list[dict[str, Any]], *, margin: int = 28) -> None:
    if not panels:
        return
    width = sum(int(panel["width"]) for panel in panels) + margin * (len(panels) + 1)
    height = max(int(panel["height"]) for panel in panels) + 2 * margin
    body: list[str] = []
    x = margin
    for panel in panels:
        y = margin + (height - 2 * margin - int(panel["height"])) // 2
        body.append(f'<g transform="translate({_svg_num(x)},{_svg_num(y)})">')
        body.extend(panel["lines"])
        body.append("</g>")
        x += int(panel["width"]) + margin
    _write_svg_document(path, width=width, height=height, body=body)


def _write_svg_ghost_strip(path: Path, panel: dict[str, Any], *, stages: int = 4) -> None:
    item_w = int(panel["width"])
    item_h = int(panel["height"])
    margin = 46
    width = stages * item_w + (stages + 1) * margin
    height = item_h + 84
    body: list[str] = []
    for index in range(stages):
        x = margin + index * (item_w + margin)
        y = 36 + int((stages - 1 - index) * 7)
        angle = -18 + index * 10
        cx = item_w * 0.5
        cy = item_h * 0.5
        opacity = 0.34 + index * 0.07
        body.append(
            f'<g transform="translate({_svg_num(x)},{_svg_num(y)}) '
            f'rotate({_svg_num(angle)} {_svg_num(cx)} {_svg_num(cy)})" '
            f'opacity="{_svg_num(opacity)}">'
        )
        body.extend(panel["lines"])
        body.append("</g>")
    _write_svg_document(path, width=width, height=height, body=body)


def _export_ppt_vector_assets(
    *,
    output_root: Path,
    tool_id: str,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    part: Mapping[str, Any],
    part_label: int,
    contact_rows: list[dict[str, Any]],
    view_indices: list[int],
    contact_card_views: int,
) -> dict[str, Any]:
    if not contact_rows:
        return {}
    vector_root = output_root / tool_id / "ppt_vector_assets"
    vector_root.mkdir(parents=True, exist_ok=True)
    contact_svgs: list[dict[str, Any]] = []
    for row in contact_rows:
        face_marker = str(row.get("face_marker") or row.get("contact_marker"))
        contact_views = _best_contact_view_indices(
            contact_rows=[row],
            context=context,
            meta=meta,
            view_indices=view_indices,
            max_views=max(3, int(contact_card_views)),
        )
        contact_panels = [
            _svg_mesh_panel(
                context=context,
                meta=meta,
                view_index=view_index,
                part=part,
                part_label=part_label,
                contact_rows=[row],
                include_contact_region=True,
                mesh_opacity=0.96,
                selected_part_opacity=0.24,
                pad=34,
            )
            for view_index in contact_views
        ]
        contact_svg = vector_root / f"{tool_id}_part_contact_region_{face_marker}_no_markers.svg"
        _write_svg_contact_sheet(contact_svg, contact_panels, margin=28)
        contact_svgs.append(
            {
                "face_marker": face_marker,
                "path": str(contact_svg),
                "point_ref": row.get("point_ref"),
                "surface_ref": row.get("surface_ref"),
                "patch_id": row.get("patch_id"),
                "view_indices": [int(value) for value in contact_views],
            }
        )
    clean_panels = []
    for view_index in view_indices[:4]:
        panel = _svg_mesh_panel(
            context=context,
            meta=meta,
            view_index=view_index,
            part=None,
            part_label=None,
            contact_rows=[],
            include_contact_region=False,
            mesh_opacity=0.92,
            selected_part_opacity=0.0,
            pad=22,
        )
        clean_svg = vector_root / f"{tool_id}_clean_mesh_view_{int(view_index):02d}.svg"
        _write_svg_document(clean_svg, width=panel["width"], height=panel["height"], body=panel["lines"])
        clean_panels.append({"view_index": int(view_index), "path": str(clean_svg), **panel})
    ghost_strip = vector_root / f"{tool_id}_clean_mesh_subgoal_ghost_strip.svg"
    if clean_panels:
        _write_svg_ghost_strip(ghost_strip, clean_panels[0], stages=4)
    manifest = {
        "schema_version": "toolframe_ppt_vector_assets_v0",
        "tool_id": tool_id,
        "source_renderer": "render_sampart3d_part_role_evidence.py",
        "source_style": "5.15-style SAMPart3D projection geometry exported as SVG vector paths",
        "contact_region_svgs": contact_svgs,
        "subgoal_ghost_strip_svg": str(ghost_strip),
        "clean_mesh_svgs": [item["path"] for item in clean_panels],
        "notes": [
            "Contact-region SVG preserves bbox and highlighted face, but has no text, face-center dot, or normal arrow.",
            "Subgoal SVGs are clean mesh ghost assets with no bbox, highlighted face, marker, or arrow.",
            "These are vector exports from the vendored 5.15-style renderer, not simplified standalone mesh redraw scripts.",
        ],
    }
    _save_json(vector_root / "manifest.json", manifest)
    return manifest


def _make_part_choice_overview(
    entries: list[dict[str, Any]],
    *,
    item_width: int,
    item_height: int,
    marker_font: ImageFont.ImageFont,
) -> Image.Image:
    if not entries:
        return Image.new("RGB", (512, 128), (255, 255, 255))
    columns = min(max(1, len(entries)), 6)
    rows = int(math.ceil(len(entries) / float(columns)))
    margin = 18
    marker_h = 44
    cell_w = int(item_width)
    cell_h = int(item_height) + marker_h
    sheet = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * margin, rows * cell_h + (rows + 1) * margin),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    _draw_dashed_rectangle(draw, (6, 6, sheet.width - 7, sheet.height - 7), fill=(25, 25, 25), width=2)
    for index, entry in enumerate(entries):
        col = index % columns
        row = index // columns
        x0 = margin + col * (cell_w + margin)
        y0 = margin + row * (cell_h + margin)
        image = _fit_image(entry["image"], max_width=cell_w - 24, max_height=item_height - 12)
        paste_x = x0 + (cell_w - image.width) // 2
        paste_y = y0 + (item_height - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        _draw_marker_circle(
            draw,
            xy=(x0 + cell_w * 0.5, y0 + item_height + marker_h * 0.5),
            text=str(entry["marker"]),
            font=marker_font,
        )
    return sheet


def _whole_mesh_part_choice_overview(
    *,
    render_paths: list[Path],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    rows: list[dict[str, Any]],
    view_indices: list[int],
    panel_width: int,
    marker_font: ImageFont.ImageFont,
) -> Image.Image | None:
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    label_rows = [
        row
        for row in rows
        if row.get("part_label") is not None
    ]
    panels: list[Image.Image] = []
    for view_index in view_indices:
        raw = Image.open(render_paths[view_index]).convert("RGB")
        projection = _project_view_faces(dict(context), view_index)
        vertex_front, vertex_pixels = _project_mesh_vertices(dict(context), view_index)
        foreground = _mesh_foreground_mask(mesh_geometry=dict(context["mesh_geometry"]), meta=dict(meta), view_index=view_index)
        base = _presentation_image(raw, reference=raw, foreground_mask=foreground, crop=False).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        marker_points: list[tuple[tuple[float, float], str]] = []
        for row in label_rows:
            part_label = int(row["part_label"])
            indices = _filtered_face_indices_for_label(
                context=context,
                labels=mesh_labels,
                inside=projection["inside"],
                label=part_label,
                face_count=len(mesh_faces),
            )
            if len(indices) == 0:
                continue
            marker_index = max(0, int(row["marker"]) - 1)
            color = PALETTE[marker_index % len(PALETTE)]
            _draw_projected_faces(
                overlay_draw,
                mesh_geometry=dict(context["mesh_geometry"]),
                vertex_front=vertex_front,
                vertex_pixels=vertex_pixels,
                face_indices=indices,
                color=color,
                alpha=108,
                outline_alpha=220,
            )
            pixels = projection["pixels"][indices]
            marker_points.append(((float(pixels[:, 0].mean()), float(pixels[:, 1].mean())), str(row["marker"])))
        if not marker_points:
            continue
        composed = Image.alpha_composite(base, overlay).convert("RGB")
        draw = ImageDraw.Draw(composed)
        crop_mask = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
        crop_draw = ImageDraw.Draw(crop_mask)
        for xy, marker in marker_points:
            _draw_marker_circle(draw, xy=xy, text=marker, font=marker_font, fill=(255, 230, 74))
            x, y = xy
            crop_draw.rectangle((x - 44.0, y - 44.0, x + 44.0, y + 44.0), fill=255)
        crop_arr = np.asarray(crop_mask.filter(ImageFilter.MaxFilter(19)), dtype=np.uint8) > 0
        if crop_arr.any():
            composed = _crop_to_mask(composed, crop_arr, pad=80)
        panels.append(_fit_image(composed, max_width=panel_width, max_height=panel_width))
    if not panels:
        return None
    return _make_contact_sheet(panels, columns=min(3, len(panels)), margin=14)


def _selected_part_bbox_overview(
    *,
    render_paths: list[Path],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    part: Mapping[str, Any],
    part_label: int,
    marker: str,
    view_indices: list[int],
    panel_width: int,
    marker_font: ImageFont.ImageFont,
) -> Image.Image | None:
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    panels: list[Image.Image] = []
    marker_index = max(0, int(marker) - 1)
    part_color = PALETTE[marker_index % len(PALETTE)]
    for view_index in view_indices:
        raw = Image.open(render_paths[view_index]).convert("RGB")
        projection = _project_view_faces(dict(context), view_index)
        vertex_front, vertex_pixels = _project_mesh_vertices(dict(context), view_index)
        foreground = _mesh_foreground_mask(
            mesh_geometry=dict(context["mesh_geometry"]),
            meta=dict(meta),
            view_index=view_index,
        )
        face_indices = _filtered_face_indices_for_label(
            context=context,
            labels=mesh_labels,
            inside=projection["inside"],
            label=part_label,
            face_count=len(mesh_faces),
        )
        if len(face_indices) == 0:
            continue
        base = _presentation_image(raw, reference=raw, foreground_mask=foreground, crop=False).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        _draw_projected_faces(
            overlay_draw,
            mesh_geometry=dict(context["mesh_geometry"]),
            vertex_front=vertex_front,
            vertex_pixels=vertex_pixels,
            face_indices=face_indices,
            color=part_color,
            alpha=96,
            outline_alpha=220,
        )
        composed = Image.alpha_composite(base, overlay).convert("RGB")
        draw = ImageDraw.Draw(composed)
        crop_mask = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
        crop_draw = ImageDraw.Draw(crop_mask)
        part_pixels = projection["pixels"][face_indices]
        crop_draw.rectangle(
            (
                float(part_pixels[:, 0].min()),
                float(part_pixels[:, 1].min()),
                float(part_pixels[:, 0].max()),
                float(part_pixels[:, 1].max()),
            ),
            fill=255,
        )
        bbox_pixels = _draw_projected_obb(
            draw,
            part=part,
            meta=meta,
            view_index=view_index,
            color=(32, 120, 210),
            width=4,
        )
        if bbox_pixels is not None and len(bbox_pixels):
            crop_draw.rectangle(
                (
                    float(bbox_pixels[:, 0].min()),
                    float(bbox_pixels[:, 1].min()),
                    float(bbox_pixels[:, 0].max()),
                    float(bbox_pixels[:, 1].max()),
                ),
                fill=255,
            )
        center = np.mean(part_pixels, axis=0)
        _draw_marker_circle(
            draw,
            xy=(float(center[0]), float(center[1])),
            text=str(marker),
            font=marker_font,
            fill=(255, 230, 74),
        )
        crop_arr = np.asarray(crop_mask.filter(ImageFilter.MaxFilter(19)), dtype=np.uint8) > 0
        if crop_arr.any():
            composed = _crop_to_mask(composed, crop_arr, pad=80)
        panels.append(_fit_image(composed, max_width=panel_width, max_height=panel_width))
    if not panels:
        return None
    return _make_contact_sheet(panels, columns=min(3, len(panels)), margin=14)


def _single_face_highlight_card(
    *,
    render_paths: list[Path],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    part: Mapping[str, Any],
    part_label: int,
    marker: str,
    face_row: Mapping[str, Any],
    view_indices: list[int],
    panel_width: int,
    marker_font: ImageFont.ImageFont,
) -> Image.Image | None:
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    axis_name, polarity = _axis_polarity_from_patch(str(face_row.get("patch_id") or ""))
    face_corners = (
        _obb_face_corners(part, axis_name=axis_name, polarity=polarity)
        if axis_name is not None and polarity is not None
        else None
    )
    if face_corners is None:
        return None
    panels: list[Image.Image] = []
    extents = np.asarray(part.get("obb", {}).get("extent_xyz", [0.05, 0.05, 0.05]), dtype=np.float64)
    normal_length = max(float(np.max(extents)) * 0.28, 0.03)
    for view_index in view_indices:
        raw = Image.open(render_paths[view_index]).convert("RGB")
        projection = _project_view_faces(dict(context), view_index)
        foreground = _mesh_foreground_mask(
            mesh_geometry=dict(context["mesh_geometry"]),
            meta=dict(meta),
            view_index=view_index,
        )
        face_indices = _filtered_face_indices_for_label(
            context=context,
            labels=mesh_labels,
            inside=projection["inside"],
            label=part_label,
            face_count=len(mesh_faces),
        )
        if len(face_indices) == 0:
            continue
        face_front, face_pixels = _project_tool_points(
            points_tool=face_corners,
            meta=meta,
            view_index=view_index,
        )
        if int(np.count_nonzero(face_front)) < 3:
            continue
        base = _presentation_image(raw, reference=raw, foreground_mask=foreground, crop=False).convert("RGBA")
        composed = base.convert("RGB")
        draw = ImageDraw.Draw(composed)
        bbox_pixels = _draw_projected_obb(
            draw,
            part=part,
            meta=meta,
            view_index=view_index,
            color=(32, 120, 210),
            width=4,
        )
        face_overlay = Image.new("RGBA", composed.size, (0, 0, 0, 0))
        face_draw = ImageDraw.Draw(face_overlay)
        polygon = [(float(x), float(y)) for x, y in face_pixels]
        face_draw.polygon(polygon, fill=(255, 199, 48, 138), outline=(214, 132, 16, 255))
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            face_draw.line((*start, *end), fill=(214, 132, 16, 255), width=5)
        composed = Image.alpha_composite(composed.convert("RGBA"), face_overlay).convert("RGB")
        draw = ImageDraw.Draw(composed)

        center = np.mean(face_corners, axis=0)
        normal = np.asarray(face_row.get("normal_tool") or [0.0, 0.0, 1.0], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            normal = np.asarray(face_row.get("xyz_tool") or center, dtype=np.float64) - center
            norm = np.linalg.norm(normal)
        normal = normal / max(norm, 1e-12)
        arrow_points = np.vstack([center, center + normal * normal_length])
        arrow_front, arrow_pixels = _project_tool_points(
            points_tool=arrow_points,
            meta=meta,
            view_index=view_index,
        )
        if bool(arrow_front[0]) and bool(arrow_front[1]):
            _draw_arrow(
                draw,
                (float(arrow_pixels[0, 0]), float(arrow_pixels[0, 1])),
                (float(arrow_pixels[1, 0]), float(arrow_pixels[1, 1])),
                color=(220, 40, 28),
                width=5,
            )
        point = np.asarray(center, dtype=np.float64)
        point_front, point_pixels = _project_tool_points(
            points_tool=point.reshape(1, 3),
            meta=meta,
            view_index=view_index,
        )
        if bool(point_front[0]):
            px, py = float(point_pixels[0, 0]), float(point_pixels[0, 1])
        else:
            px, py = float(np.mean(face_pixels[:, 0])), float(np.mean(face_pixels[:, 1]))
        label_xy = (
            min(max(px + 24.0, 40.0), float(composed.width - 40)),
            min(max(py - 20.0, 40.0), float(composed.height - 40)),
        )
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 230, 74), outline=(25, 25, 25), width=2)
        _draw_marker_circle(
            draw,
            xy=label_xy,
            text=str(face_row.get("face_marker") or face_row.get("contact_marker")),
            font=marker_font,
            outline=(25, 25, 25),
            fill=(255, 255, 255),
            radius_min=11,
            pad=4,
        )

        crop_mask = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
        crop_draw = ImageDraw.Draw(crop_mask)
        part_pixels = projection["pixels"][face_indices]
        crop_draw.rectangle(
            (
                float(part_pixels[:, 0].min()),
                float(part_pixels[:, 1].min()),
                float(part_pixels[:, 0].max()),
                float(part_pixels[:, 1].max()),
            ),
            fill=255,
        )
        xs = face_pixels[:, 0]
        ys = face_pixels[:, 1]
        crop_draw.rectangle((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), fill=255)
        if bbox_pixels is not None and len(bbox_pixels):
            crop_draw.rectangle(
                (
                    float(bbox_pixels[:, 0].min()),
                    float(bbox_pixels[:, 1].min()),
                    float(bbox_pixels[:, 0].max()),
                    float(bbox_pixels[:, 1].max()),
                ),
                fill=255,
            )
        lx, ly = label_xy
        crop_draw.rectangle((min(px, lx) - 52.0, min(py, ly) - 52.0, max(px, lx) + 52.0, max(py, ly) + 52.0), fill=255)
        crop_arr = np.asarray(crop_mask.filter(ImageFilter.MaxFilter(19)), dtype=np.uint8) > 0
        if crop_arr.any():
            composed = _crop_to_mask(composed, crop_arr, pad=88)
        panels.append(_fit_image(composed, max_width=panel_width, max_height=panel_width))
    if not panels:
        return None
    return _make_contact_sheet(panels, columns=min(3, len(panels)), margin=12)


def render_tool(
    *,
    tool_root: Path,
    output_root: Path,
    prior_subdir: str,
    selected_contact_markers: set[str] | None,
    max_views: int,
    part_overview_views: int,
    contact_card_views: int,
    view_selection: str,
    panel_width: int,
    part_item_width: int,
    part_item_height: int,
    skip_face_cards: bool = False,
    hide_face_markers: bool = False,
    hide_face_normal_arrows: bool = False,
    hide_view_badges: bool = False,
    export_ppt_vector_assets: bool = False,
) -> dict[str, Any]:
    tool_id = tool_root.name
    prior_dir = tool_root / prior_subdir
    tool_part_graph = _load_json(prior_dir / "tool_part_graph.json")
    grounding_candidates = _load_json(prior_dir / "grounding_candidates.json")
    anchor_artifact = _projection_artifact_from_geometry_prior(
        tool_root=tool_root,
        prior_dir=prior_dir,
        tool_part_graph=tool_part_graph,
    )
    context = _projection_context(anchor_artifact)
    context = dict(context)
    context["filtered_face_indices_by_label"] = _filtered_face_indices_by_label(
        anchor_artifact,
        np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64),
    )
    meta = context["meta"]
    rows = _part_rows(tool_part_graph)
    render_paths = sorted(Path(anchor_artifact["render_dir"]).glob("render_*.webp"))
    view_indices = _select_views(len(render_paths), max_views=max_views, view_selection=view_selection)
    overview_view_indices = _select_views(
        len(render_paths),
        max_views=max(3, int(part_overview_views)),
        view_selection=view_selection,
    )

    tool_out = output_root / tool_id
    for managed_name in (
        "overview",
        "part_cards",
        "selected_part_bbox_overview",
        "selected_part_face_highlights",
        "contact_marker_overviews",
        "function_contact_choices",
        "contact_point_choices",
    ):
        managed_dir = tool_out / managed_name
        if managed_dir.exists():
            shutil.rmtree(managed_dir)
    overview_dir = tool_out / "overview"
    part_card_dir = tool_out / "part_cards"
    overview_dir.mkdir(parents=True, exist_ok=True)
    part_card_dir.mkdir(parents=True, exist_ok=True)

    marker_font = _font(size=22, bold=True)
    part_cards: dict[str, str] = {}
    selected_part_bbox_overviews: dict[str, str] = {}
    face_highlight_cards: list[dict[str, Any]] = []
    part_choice_overview: str | None = None
    isolated_part_choice_overview: str | None = None
    contact_marker_rows: list[dict[str, Any]] = []
    overview_entries: list[dict[str, Any]] = []
    best_view_by_part: dict[str, int] = {}
    ppt_vector_assets: dict[str, Any] = {}

    whole_mesh_overview = _whole_mesh_part_choice_overview(
        render_paths=render_paths,
        context=context,
        meta=meta,
        rows=rows,
        view_indices=overview_view_indices,
        panel_width=panel_width,
        marker_font=marker_font,
    )
    if whole_mesh_overview is not None:
        overview_path = overview_dir / f"{tool_id}_whole_mesh_part_choice_overview.png"
        whole_mesh_overview.save(overview_path)
        part_choice_overview = str(overview_path)

    label_by_part = {
        row["part_id"]: int(row["part_label"])
        for row in rows
        if row.get("part_label") is not None
    }
    parts = tool_part_graph.get("parts", {})
    for row in rows:
        part_id = row["part_id"]
        part = parts.get(part_id)
        part_label = label_by_part.get(part_id)
        if not isinstance(part, Mapping) or part_label is None:
            continue
        marker = str(row["marker"])
        render_part_card = selected_contact_markers is None or marker in selected_contact_markers
        render_contact_markers = selected_contact_markers is not None and marker in selected_contact_markers
        best_crop = _best_isolated_part_crop(
            render_paths=render_paths,
            context=context,
            meta=meta,
            view_indices=view_indices,
            part_label=part_label,
            crop_pad=72,
        )
        if best_crop is not None and render_part_card:
            panel, best_view = best_crop
            best_view_by_part[part_id] = best_view
            single = _fit_image(panel, max_width=panel_width, max_height=panel_width)
            single_card = Image.new("RGB", (single.width, single.height + 44), (255, 255, 255))
            single_card.paste(single, (0, 0))
            single_draw = ImageDraw.Draw(single_card)
            _draw_marker_circle(
                single_draw,
                xy=(single_card.width * 0.5, single.height + 22),
                text=marker,
                font=marker_font,
            )
            card_path = part_card_dir / f"candidate_{int(marker):03d}_card.png"
            single_card.save(card_path)
            part_cards[part_id] = str(card_path)
            overview_entries.append({"marker": marker, "image": panel})

        if not render_contact_markers:
            continue
        part_contact_rows = _contact_marker_table(
            part_id=part_id,
            part_ref=row["part_ref"],
            marker=marker,
            grounding_candidates=grounding_candidates,
        )
        contact_marker_rows.extend(part_contact_rows)
        contact_view_indices = _best_contact_view_indices(
            contact_rows=part_contact_rows,
            context=context,
            meta=meta,
            view_indices=view_indices,
            max_views=max(3, int(part_overview_views)),
        )
        selected_bbox_overview = _selected_part_bbox_overview(
            render_paths=render_paths,
            context=context,
            meta=meta,
            part=part,
            part_label=part_label,
            marker=marker,
            view_indices=contact_view_indices,
            panel_width=panel_width,
            marker_font=marker_font,
        )
        if selected_bbox_overview is not None:
            bbox_dir = tool_out / "selected_part_bbox_overview"
            bbox_dir.mkdir(parents=True, exist_ok=True)
            bbox_path = bbox_dir / f"candidate_{int(marker):03d}_bbox_overview.png"
            selected_bbox_overview.save(bbox_path)
            selected_part_bbox_overviews[part_id] = str(bbox_path)
        face_highlight_dir = tool_out / "selected_part_face_highlights"
        for face_row in part_contact_rows:
            face_marker = str(face_row.get("face_marker") or face_row.get("contact_marker"))
            face_view_indices = _best_contact_view_indices(
                contact_rows=[face_row],
                context=context,
                meta=meta,
                view_indices=view_indices,
                max_views=max(3, int(contact_card_views)),
            )
            face_card = _single_face_highlight_card(
                render_paths=render_paths,
                context=context,
                meta=meta,
                part=part,
                part_label=part_label,
                marker=marker,
                face_row=face_row,
                view_indices=face_view_indices,
                panel_width=panel_width,
                marker_font=marker_font,
            )
            if face_card is None:
                continue
            face_highlight_dir.mkdir(parents=True, exist_ok=True)
            face_path = face_highlight_dir / f"candidate_{int(marker):03d}_face_{face_marker}_highlight.png"
            face_card.save(face_path)
            face_highlight_cards.append(
                {
                    "part_id": part_id,
                    "part_marker": marker,
                    "face_marker": face_marker,
                    "path": str(face_path),
                    "point_ref": face_row.get("point_ref"),
                    "surface_ref": face_row.get("surface_ref"),
                    "patch_id": face_row.get("patch_id"),
                }
            )
        if export_ppt_vector_assets and part_contact_rows and not ppt_vector_assets:
            ppt_vector_assets = _export_ppt_vector_assets(
                output_root=output_root,
                tool_id=tool_id,
                context=context,
                meta=meta,
                part=part,
                part_label=part_label,
                contact_rows=part_contact_rows,
                view_indices=view_indices,
                contact_card_views=contact_card_views,
            )
    if overview_entries:
        overview = _make_part_choice_overview(
            overview_entries,
            item_width=part_item_width,
            item_height=part_item_height,
            marker_font=marker_font,
        )
        overview_path = overview_dir / f"{tool_id}_isolated_part_choice_overview.png"
        overview.save(overview_path)
        isolated_part_choice_overview = str(overview_path)

    return {
        "tool_id": tool_id,
        "part_count": len(rows),
        "view_count": len(view_indices),
        "part_choice_overview": part_choice_overview,
        "isolated_part_choice_overview": isolated_part_choice_overview,
        "part_cards": part_cards,
        "selected_part_bbox_overviews": selected_part_bbox_overviews,
        "face_highlight_cards": face_highlight_cards,
        "region_marker_table": rows,
        "contact_marker_table": contact_marker_rows,
        "selected_contact_markers": sorted(selected_contact_markers) if selected_contact_markers else None,
        "ppt_vector_assets": ppt_vector_assets,
    }


def _selected_markers_from_part_role_selection(path: str | None) -> dict[str | None, set[str]]:
    if not path:
        return {}
    data = _load_json(Path(path))
    marker = (
        data.get("selected_marker")
        or data.get("selected_function_marker")
        or data.get("selected_contact_point_marker")
        or (data.get("selected_part") or {}).get("marker")
    )
    if marker is None:
        raise ValueError(f"part-role selection does not contain a selected part marker: {path}")
    tool_id = data.get("tool_id")
    return {str(tool_id) if tool_id is not None else None: {str(marker)}}


def _selected_markers_from_args(
    *,
    part_role_selection: str | None,
    selected_part_markers: list[str] | None,
    tool_ids: list[str] | None,
) -> dict[str | None, set[str]]:
    selected = _selected_markers_from_part_role_selection(part_role_selection)
    if selected_part_markers:
        if tool_ids and len(tool_ids) > 1:
            raise ValueError("--selected-part-markers requires exactly one --tool-id when multiple tools are rendered")
        selected[tool_ids[0] if tool_ids else None] = {str(marker) for marker in selected_part_markers}
    return selected


def _write_index(output_root: Path, results: list[dict[str, Any]]) -> Path:
    rows: list[str] = []
    for result in results:
        tool_id = result["tool_id"]
        overview_path = result.get("part_choice_overview")
        overview_link = (
            f"<a href='{Path(str(overview_path)).relative_to(output_root)}'>overview</a>"
            if overview_path
            else ""
        )
        isolated_overview_path = result.get("isolated_part_choice_overview")
        isolated_overview_link = (
            f"<a href='{Path(str(isolated_overview_path)).relative_to(output_root)}'>isolated</a>"
            if isolated_overview_path
            else ""
        )
        card_links = " ".join(
            f"<a href='{Path(path).relative_to(output_root)}'>part {marker}</a>"
            for part_id, path in sorted(result.get("part_cards", {}).items())
            for marker in [
                next(row["marker"] for row in result["region_marker_table"] if row["part_id"] == part_id)
            ]
        )
        bbox_links = " ".join(
            f"<a href='{Path(path).relative_to(output_root)}'>bbox part {marker}</a>"
            for part_id, path in sorted(result.get("selected_part_bbox_overviews", {}).items())
            for marker in [
                next(row["marker"] for row in result["region_marker_table"] if row["part_id"] == part_id)
            ]
        )
        face_links = " ".join(
            f"<a href='{Path(str(card['path'])).relative_to(output_root)}'>{card['part_marker']}/{card['face_marker']}</a>"
            for card in result.get("face_highlight_cards", [])
            if isinstance(card, Mapping) and card.get("path")
        )
        rows.append(
            f"<tr><td>{tool_id}</td><td>{result['part_count']}</td><td>{overview_link}</td>"
            f"<td>{isolated_overview_link}</td><td>{card_links}</td>"
            f"<td>{bbox_links}</td><td>{face_links}</td></tr>"
        )
    html_path = output_root / "index.html"
    html_path.write_text(
        "<!doctype html><html><body>"
        "<table><thead><tr><th>tool</th><th>parts</th><th>whole mesh overview</th>"
        "<th>isolated overview</th><th>part cards</th><th>selected bbox</th><th>face highlights</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>\n",
        encoding="utf-8",
    )
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(ROOT / "artifacts" / "tool_prior" / "sampart3d_scale2"))
    parser.add_argument("--prior-subdir", default="tool_prior")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-id", action="append")
    parser.add_argument(
        "--part-role-selection",
        help=(
            "Part-role selection JSON. When provided, restrict isolated part cards "
            "and the structured contact marker table to its selected part marker."
        ),
    )
    parser.add_argument(
        "--selected-part-markers",
        nargs="*",
        help=(
            "Render isolated part cards only for these selected part markers. "
            "Use with a single --tool-id; --part-role-selection is preferred in the mainline."
        ),
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=16,
        help="View candidates used to choose clean isolated part-card views.",
    )
    parser.add_argument(
        "--part-overview-views",
        type=int,
        default=3,
        help="Number of whole-mesh views shown in one VLM part-choice overview image; minimum 3.",
    )
    parser.add_argument(
        "--contact-card-views",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--view-selection", choices=["all", "uniform", "first"], default="uniform")
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--part-item-width", type=int, default=210)
    parser.add_argument("--part-item-height", type=int, default=220)
    parser.add_argument(
        "--skip-face-cards",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hide-face-markers",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hide-face-normal-arrows",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hide-view-badges",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--export-ppt-vector-assets",
        action="store_true",
        help=(
            "Also export PPT-oriented SVG vector assets from the same 5.15-style "
            "projection geometry: contact-region SVGs and clean subgoal ghost SVGs."
        ),
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    tool_roots = sorted(path for path in run_root.iterdir() if (path / args.prior_subdir / "tool_part_graph.json").exists())
    if args.tool_id:
        wanted = set(args.tool_id)
        tool_roots = [path for path in tool_roots if path.name in wanted]
    if not tool_roots:
        raise RuntimeError("no tool_part_graph.json files were found")
    selected_markers_by_tool = _selected_markers_from_args(
        part_role_selection=args.part_role_selection,
        selected_part_markers=args.selected_part_markers,
        tool_ids=args.tool_id,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        render_tool(
            tool_root=tool_root,
            output_root=output_root,
            prior_subdir=args.prior_subdir,
            selected_contact_markers=(
                selected_markers_by_tool.get(tool_root.name)
                or selected_markers_by_tool.get(None)
            ),
            max_views=args.max_views,
            part_overview_views=args.part_overview_views,
            contact_card_views=args.contact_card_views,
            view_selection=args.view_selection,
            panel_width=args.panel_width,
            part_item_width=args.part_item_width,
            part_item_height=args.part_item_height,
            skip_face_cards=args.skip_face_cards,
            hide_face_markers=args.hide_face_markers,
            hide_face_normal_arrows=args.hide_face_normal_arrows,
            hide_view_badges=args.hide_view_badges,
            export_ppt_vector_assets=args.export_ppt_vector_assets,
        )
        for tool_root in tool_roots
    ]
    items: list[dict[str, Any]] = []
    region_marker_table: list[dict[str, Any]] = []
    contact_marker_table: list[dict[str, Any]] = []
    for result in results:
        tool_id = result["tool_id"]
        region_marker_table.extend({"tool_id": tool_id, **row} for row in result["region_marker_table"])
        contact_marker_table.extend({"tool_id": tool_id, **row} for row in result["contact_marker_table"])
        overview_path = result.get("part_choice_overview")
        if overview_path:
            items.append(
                {
                    "id": f"{tool_id}_whole_mesh_part_choice_overview",
                    "kind": "whole_mesh_part_choice_overview",
                    "media_type": "image",
                    "path": overview_path,
                    "description": (
                        f"Multi-view whole-mesh SAMPart3D part candidates for {tool_id}; "
                        "each part is shown as a colored overlay on the mesh, with the same "
                        "anonymous marker ID reused across views."
                    ),
                    "provider_id": "sampart3d_part_role_evidence",
                    "source_stage": "offline_tool_part_visualization",
                    "allowed_use": "planning_context",
                }
            )
        isolated_overview_path = result.get("isolated_part_choice_overview")
        if isolated_overview_path:
            items.append(
                {
                    "id": f"{tool_id}_isolated_part_choice_overview",
                    "kind": "isolated_part_choice_overview",
                    "media_type": "image",
                    "path": isolated_overview_path,
                    "description": f"Isolated SAMPart3D part candidates for {tool_id}; auxiliary debug view.",
                    "provider_id": "sampart3d_part_role_evidence",
                    "source_stage": "offline_tool_part_visualization",
                    "allowed_use": "debug_only",
                }
            )
        for part_id, path in sorted(result.get("part_cards", {}).items()):
            marker = next(row["marker"] for row in result["region_marker_table"] if row["part_id"] == part_id)
            items.append(
                {
                    "id": f"{tool_id}_part_candidate_{marker}_isolated_card",
                    "kind": "isolated_part_card",
                    "media_type": "image",
                    "path": path,
                    "part_marker": str(marker),
                    "description": f"Isolated SAMPart3D part candidate {marker} for {tool_id}; auxiliary debug view.",
                    "provider_id": "sampart3d_part_role_evidence",
                    "source_stage": "offline_tool_part_visualization",
                    "allowed_use": "debug_only",
                }
            )
        for part_id, path in sorted(result.get("selected_part_bbox_overviews", {}).items()):
            marker = next(row["marker"] for row in result["region_marker_table"] if row["part_id"] == part_id)
            items.append(
                {
                    "id": f"{tool_id}_part_candidate_{marker}_selected_part_bbox_overview",
                    "kind": "selected_part_bbox_overview",
                    "media_type": "image",
                    "path": path,
                    "part_marker": str(marker),
                    "description": (
                        f"5.15-style selected-part bbox overview for part candidate {marker} on {tool_id}."
                    ),
                    "provider_id": "sampart3d_part_role_evidence",
                    "source_stage": "offline_tool_part_visualization",
                    "allowed_use": "planning_context",
                }
            )
        for card in result.get("face_highlight_cards", []):
            if not isinstance(card, Mapping):
                continue
            marker = str(card.get("part_marker"))
            face_marker = str(card.get("face_marker"))
            path = str(card.get("path"))
            items.append(
                {
                    "id": f"{tool_id}_part_candidate_{marker}_face_{face_marker}_highlight",
                    "kind": "selected_part_face_highlight",
                    "media_type": "image",
                    "path": path,
                    "part_marker": marker,
                    "face_marker": face_marker,
                    "point_ref": card.get("point_ref"),
                    "surface_ref": card.get("surface_ref"),
                    "patch_id": card.get("patch_id"),
                    "description": (
                        f"5.15-style selected-part face evidence for {tool_id} part {marker}, face {face_marker}: "
                        "part bbox is drawn, only this bbox face is highlighted, and the contact point plus "
                        "surface-normal/motion-axis arrow are shown."
                    ),
                    "provider_id": "sampart3d_part_role_evidence",
                    "source_stage": "offline_tool_part_visualization",
                    "allowed_use": "planning_context",
                }
            )
    report = {"tools": results}
    visual_evidence = {
        "schema_version": "v0.1",
        "evidence_id": f"{output_root.name}_visual_evidence",
        "items": items,
        "metadata": {
            "evidence_kind": "sampart3d_part_overview_only",
            "contact_marker_table_scope": (
                "selected_part_only" if selected_markers_by_tool else "all_parts_precomputed"
            ),
            "selected_contact_point_markers_by_tool": {
                str(key): sorted(value)
                for key, value in selected_markers_by_tool.items()
                if key is not None
            },
            "selected_contact_point_markers_default": sorted(selected_markers_by_tool.get(None, set())),
            "part_role_selection_path": args.part_role_selection,
            "region_marker_table": region_marker_table,
            "contact_marker_table": contact_marker_table,
            "usage": (
                "The whole-mesh part-choice overview exposes anonymous SAMPart3D regions in full-tool context "
                "as one multi-view sheet with colored part overlays for contact-region selection. "
                "Isolated cards are auxiliary debug views. After a part is selected, the renderer emits a "
                "selected-part bbox overview plus one selected-part face-highlight card per F marker; each "
                "face card shows the bbox, one highlighted bbox face, the contact point, and the surface "
                "normal/motion-axis arrow. This renderer must not emit legacy per-face/contact-choice paths."
            ),
        },
    }
    _save_json(output_root / "part_role_evidence_report.json", report)
    _save_json(output_root / "visual_evidence.json", visual_evidence)
    html_path = _write_index(output_root, results)
    print(
        json.dumps(
            {
                "html": str(html_path),
                "visual_evidence": str(output_root / "visual_evidence.json"),
                "tool_count": len(results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
