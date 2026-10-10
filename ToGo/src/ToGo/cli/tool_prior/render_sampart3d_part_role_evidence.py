#!/usr/bin/env python3
"""Render P3SAM part overviews and selected-part face evidence for the shared VLM pipeline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "ToGo" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ToGo.pipeline.handoff import write_json as _write_json

from ToGo.providers.tool_prior.sampart3d_projection_geometry import SAMPart3D_RENDER_ROTATION
from ToGo.providers.tool_prior.sampart3d_mesh_candidates import (  # noqa: E402
    _dominant_spatial_component_cluster,
    _face_connected_components,
    load_obj_mesh,
)
from ToGo.providers.tool_prior.sampart3d_projection_helpers import (  # noqa: E402
    PALETTE,
    _font,
    _make_contact_sheet,
    _mesh_foreground_mask,
    _presentation_image,
    _project_render_points,
    _project_view_faces,
    _projection_context,
)
from ToGo.providers.tool_prior.visualize_sampart3d_region_overlays import (  # noqa: E402
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


def _clip_rgba_alpha_to_mask(overlay: Image.Image, mask: Image.Image) -> Image.Image:
    """Restrict an RGBA overlay to the visible pixels of a projected mesh mask."""

    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != overlay.size:
        mask = mask.resize(overlay.size)
    arr = np.asarray(overlay, dtype=np.uint8).copy()
    mask_arr = np.asarray(mask, dtype=np.uint16)
    arr[:, :, 3] = ((arr[:, :, 3].astype(np.uint16) * mask_arr) // 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


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


def _with_bottom_marker_label(
    image: Image.Image,
    *,
    text: str,
    font: ImageFont.ImageFont,
    band_height: int = 44,
) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + band_height), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    _draw_marker_circle(
        draw,
        xy=(image.width * 0.5, image.height + band_height * 0.5),
        text=text,
        font=font,
        outline=(25, 25, 25),
        fill=(255, 255, 255),
        radius_min=12,
        pad=4,
    )
    return canvas


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


def _bbox_axis_from_patch_id(patch_id: Any) -> list[float] | None:
    axis_name, polarity = _axis_polarity_from_patch(str(patch_id or ""))
    if axis_name is None or polarity is None:
        return None
    sign = -1.0 if polarity == "negative" else 1.0
    if axis_name == "x":
        return [sign, 0.0, 0.0]
    if axis_name == "y":
        return [0.0, sign, 0.0]
    if axis_name == "z":
        return [0.0, 0.0, sign]
    return None


def _bbox_face_area(part: Mapping[str, Any], patch_id: Any) -> float | None:
    axis_name, _ = _axis_polarity_from_patch(str(patch_id or ""))
    if axis_name not in AXIS_NAMES:
        return None
    extent = part.get("obb", {}).get("extent_xyz")
    if not (isinstance(extent, list) and len(extent) == 3):
        return None
    values = [max(float(item), 0.0) for item in extent]
    if axis_name == "x":
        return values[1] * values[2]
    if axis_name == "y":
        return values[0] * values[2]
    if axis_name == "z":
        return values[0] * values[1]
    return None


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
    part: Mapping[str, Any],
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
        axis_name, polarity = _axis_polarity_from_patch(patch_id)
        bbox_axis = _bbox_axis_from_patch_id(patch_id)
        bbox_area = _bbox_face_area(part, patch_id)
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
                "bbox_axis_tool": bbox_axis,
                "axis_role": "bbox_face_axis" if bbox_axis is not None else "mesh_patch_normal",
                "bbox_face_area_m2": bbox_area,
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


def _projected_part_mask(
    *,
    image_size: tuple[int, int],
    mesh_faces: np.ndarray,
    vertex_front: np.ndarray,
    vertex_pixels: np.ndarray,
    face_indices: np.ndarray,
    pad_px: int = 2,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for face_index in face_indices:
        vertices = mesh_faces[int(face_index)]
        if not np.all(vertex_front[vertices]):
            continue
        polygon = [(float(vertex_pixels[v, 0]), float(vertex_pixels[v, 1])) for v in vertices]
        draw.polygon(polygon, fill=255)
    if pad_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(pad_px * 2 + 1))
    return mask


def _best_working_pose_view_indices(
    *,
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    part_label: int,
    view_indices: list[int],
    max_views: int,
    render_paths: Sequence[Path] | None = None,
    prefer_part_lower: bool = False,
    prefer_dark_part_lower: bool = False,
) -> list[int]:
    """Prefer views where the whole tool appears in a stable working posture.

    Candidate face cards are easier for a VLM to interpret when the tool is
    laid out horizontally in image space.  This is especially important for
    motion-axis arrows: otherwise a tool-fixed axis can look like a world
    up/down instruction.
    """

    mesh_geometry = context["mesh_geometry"]
    mesh_faces = np.asarray(mesh_geometry["faces"], dtype=np.int64)
    mesh_labels = np.asarray(mesh_geometry["labels"], dtype=np.int64)
    scored: list[tuple[float, int]] = []
    for view_index in view_indices:
        foreground = _mesh_foreground_mask(
            mesh_geometry=dict(mesh_geometry),
            meta=dict(meta),
            view_index=view_index,
        )
        ys, xs = np.nonzero(foreground)
        if len(xs) == 0:
            continue
        whole_w = float(xs.max() - xs.min() + 1)
        whole_h = float(ys.max() - ys.min() + 1)
        whole_aspect = whole_w / max(whole_h, 1.0)

        projection = _project_view_faces(dict(context), view_index)
        face_indices = _filtered_face_indices_for_label(
            context=context,
            labels=mesh_labels,
            inside=projection["inside"],
            label=part_label,
            face_count=len(mesh_faces),
        )
        part_lower = 0.0
        dark_part_lower = 0.0
        if len(face_indices) == 0:
            part_aspect = 0.0
            part_area = 0.0
        else:
            part_pixels = projection["pixels"][face_indices]
            part_w = float(part_pixels[:, 0].max() - part_pixels[:, 0].min() + 1)
            part_h = float(part_pixels[:, 1].max() - part_pixels[:, 1].min() + 1)
            part_aspect = part_w / max(part_h, 1.0)
            part_area = part_w * part_h
            if prefer_part_lower:
                whole_center_y = 0.5 * (float(ys.min()) + float(ys.max()))
                part_center_y = 0.5 * (float(part_pixels[:, 1].min()) + float(part_pixels[:, 1].max()))
                part_lower = (part_center_y - whole_center_y) / max(whole_h, 1.0)
            if prefer_dark_part_lower and render_paths is not None and 0 <= view_index < len(render_paths):
                try:
                    image = Image.open(render_paths[view_index]).convert("RGB")
                    pixels = np.asarray(image)
                    px = np.clip(np.rint(part_pixels[:, 0]).astype(np.int64), 0, pixels.shape[1] - 1)
                    py = np.clip(np.rint(part_pixels[:, 1]).astype(np.int64), 0, pixels.shape[0] - 1)
                    samples = pixels[py, px].astype(np.float64)
                    brightness = samples.mean(axis=1)
                    if len(brightness) > 0:
                        cutoff = min(110.0, float(np.quantile(brightness, 0.35)))
                        dark = brightness <= cutoff
                        if np.count_nonzero(dark) >= 3:
                            whole_center_y = 0.5 * (float(ys.min()) + float(ys.max()))
                            dark_center_y = float(np.mean(part_pixels[dark, 1]))
                            dark_part_lower = (dark_center_y - whole_center_y) / max(whole_h, 1.0)
                except OSError:
                    dark_part_lower = 0.0
        score = min(whole_aspect, 5.0) * 1000.0 + min(part_aspect, 4.0) * 300.0 + np.log1p(part_area)
        if prefer_part_lower:
            score += part_lower * 1800.0
        if prefer_dark_part_lower:
            score += dark_part_lower * 2400.0
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


def _polygon_area_2d(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _best_bbox_face_view_indices(
    *,
    face_row: Mapping[str, Any],
    part: Mapping[str, Any],
    meta: Mapping[str, Any],
    view_indices: list[int],
    max_views: int,
) -> list[int]:
    axis_name, polarity = _axis_polarity_from_patch(str(face_row.get("patch_id") or ""))
    face_corners = (
        _obb_face_corners(part, axis_name=axis_name, polarity=polarity)
        if axis_name is not None and polarity is not None
        else None
    )
    if face_corners is None:
        return view_indices[:max_views]
    normal_tool = np.asarray(
        face_row.get("bbox_axis_tool") or face_row.get("normal_tool") or [0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(normal_tool))
    if norm <= 1e-12:
        return view_indices[:max_views]
    normal_render = (normal_tool / norm) @ SAMPart3D_RENDER_ROTATION
    normal_render = normal_render / max(float(np.linalg.norm(normal_render)), 1e-12)
    scored: list[tuple[float, int]] = []
    for view_index in view_indices:
        face_front, face_pixels = _project_tool_points(
            points_tool=face_corners,
            meta=meta,
            view_index=view_index,
        )
        if int(np.count_nonzero(face_front)) < 3:
            continue
        c2w = np.asarray(meta["transforms"][view_index], dtype=np.float64)
        # Projected front-facing tests use the vector from the point toward the
        # camera.  For the SAMPart3D camera transforms this is +camera Z in
        # world/render coordinates, not the camera's viewing direction.
        view_dir = c2w[:3, 2]
        view_norm = float(np.linalg.norm(view_dir))
        if view_norm <= 1e-12:
            continue
        view_dir = view_dir / view_norm
        facing = max(0.0, float(normal_render @ view_dir))
        area = _polygon_area_2d(face_pixels[np.asarray(face_front, dtype=bool)])
        span = float(np.linalg.norm(np.max(face_pixels, axis=0) - np.min(face_pixels, axis=0)))
        score = facing * 100000.0 + area + 0.05 * span
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


def _candidate_face_overview(
    *,
    render_paths: list[Path],
    context: Mapping[str, Any],
    meta: Mapping[str, Any],
    part: Mapping[str, Any],
    part_label: int,
    marker: str,
    contact_rows: list[dict[str, Any]],
    view_indices: list[int],
    panel_width: int,
    marker_font: ImageFont.ImageFont,
    show_contact_point: bool = True,
    show_motion_axis: bool = True,
    motion_axis_color: tuple[int, int, int] = (220, 40, 28),
    clean_contact_overlay: bool = False,
    contact_local_zoom: bool = False,
    hide_contact_bbox: bool = False,
    contact_outline_only: bool = False,
    contact_point_radius: float = 9.0,
    contact_crop_pad: int = 128,
) -> Image.Image | None:
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    extents = np.asarray(part.get("obb", {}).get("extent_xyz", [0.05, 0.05, 0.05]), dtype=np.float64)
    normal_length = max(float(np.max(extents)) * 0.28, 0.03)
    if not view_indices:
        return None

    anchored_view_indices = list(view_indices)
    anchor_view_indices: list[int] = []
    panels: list[Image.Image] = []
    for face_row in contact_rows:
        face_marker = str(face_row.get("face_marker") or face_row.get("contact_marker") or "")
        axis_name, polarity = _axis_polarity_from_patch(str(face_row.get("patch_id") or ""))
        face_corners = (
            _obb_face_corners(part, axis_name=axis_name, polarity=polarity)
            if axis_name is not None and polarity is not None
            else None
        )
        if face_corners is None:
            continue
        best_view_indices = _best_bbox_face_view_indices(
            face_row=face_row,
            part=part,
            meta=meta,
            view_indices=anchored_view_indices,
            max_views=3,
        )
        face_view_indices: list[int] = []
        # If an overview anchor face is requested, it must be tried before the
        # default first view; otherwise the default view can satisfy projection
        # checks and silently prevent the requested view from reaching tool_refs.
        for candidate_view_index in [*anchor_view_indices, *best_view_indices, *anchored_view_indices]:
            if candidate_view_index in face_view_indices:
                continue
            face_front, face_pixels = _project_tool_points(
                points_tool=face_corners,
                meta=meta,
                view_index=candidate_view_index,
            )
            if int(np.count_nonzero(face_front)) < 2 or not np.all(np.isfinite(face_pixels)):
                continue
            face_view_indices.append(candidate_view_index)
            break
        center = np.mean(face_corners, axis=0)
        normal = np.asarray(
            face_row.get("bbox_axis_tool") or face_row.get("normal_tool") or [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            normal = np.asarray(face_row.get("xyz_tool") or center, dtype=np.float64) - center
            norm = np.linalg.norm(normal)
        normal = normal / max(norm, 1e-12)
        contact_clean_mode = bool(clean_contact_overlay and not show_motion_axis)
        contact_zoom_mode = bool((contact_clean_mode or contact_local_zoom) and not show_motion_axis)
        color = (35, 205, 92) if contact_clean_mode else (245, 190, 45)
        view_panels: list[Image.Image] = []
        for panel_view_index in face_view_indices:
            face_front, face_pixels = _project_tool_points(
                points_tool=face_corners,
                meta=meta,
                view_index=panel_view_index,
            )
            if int(np.count_nonzero(face_front)) < 2 or not np.all(np.isfinite(face_pixels)):
                continue
            arrow_points = np.vstack([center, center + normal * normal_length])
            arrow_front, arrow_pixels = _project_tool_points(
                points_tool=arrow_points,
                meta=meta,
                view_index=panel_view_index,
            )
            point_front, point_pixels = _project_tool_points(
                points_tool=np.asarray(center, dtype=np.float64).reshape(1, 3),
                meta=meta,
                view_index=panel_view_index,
            )
            if bool(point_front[0]) and np.all(np.isfinite(point_pixels)):
                px, py = float(point_pixels[0, 0]), float(point_pixels[0, 1])
            else:
                px, py = float(np.mean(face_pixels[:, 0])), float(np.mean(face_pixels[:, 1]))
            panel_raw = Image.open(render_paths[panel_view_index]).convert("RGB")
            panel_projection = _project_view_faces(dict(context), panel_view_index)
            panel_foreground = _mesh_foreground_mask(
                mesh_geometry=dict(context["mesh_geometry"]),
                meta=dict(meta),
                view_index=panel_view_index,
            )
            vertex_front, vertex_pixels = _project_mesh_vertices(dict(context), panel_view_index)
            panel_face_indices = _filtered_face_indices_for_label(
                context=context,
                labels=mesh_labels,
                inside=panel_projection["inside"],
                label=part_label,
                face_count=len(mesh_faces),
            )
            panel_base = _presentation_image(
                panel_raw,
                reference=panel_raw,
                foreground_mask=panel_foreground,
                crop=False,
            ).convert("RGB")
            composed = panel_base.copy()
            draw = ImageDraw.Draw(composed)
            if not contact_clean_mode and not hide_contact_bbox:
                _draw_projected_obb(
                    draw,
                    part=part,
                    meta=meta,
                    view_index=panel_view_index,
                    color=(32, 120, 210),
                    width=3,
                )
            overlay = Image.new("RGBA", composed.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            polygon = [(float(x), float(y)) for x, y in face_pixels]
            fill_alpha = 0 if contact_outline_only else (150 if contact_clean_mode else 92)
            outline_alpha = 255 if contact_clean_mode else 240
            overlay_draw.polygon(polygon, fill=(*color[:3], fill_alpha), outline=(*color[:3], outline_alpha))
            for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                overlay_draw.line((*start, *end), fill=(*color[:3], outline_alpha), width=4 if contact_clean_mode else 5)
            panel_part_mask = _projected_part_mask(
                image_size=panel_raw.size,
                mesh_faces=np.asarray(mesh_faces, dtype=np.int64),
                vertex_front=vertex_front,
                vertex_pixels=vertex_pixels,
                face_indices=panel_face_indices,
                pad_px=2,
            )
            overlay = _clip_rgba_alpha_to_mask(overlay, panel_part_mask)
            composed = Image.alpha_composite(composed.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(composed)
            arrow_visible = bool(arrow_front[0]) and bool(arrow_front[1]) and np.all(np.isfinite(arrow_pixels))
            if show_motion_axis and arrow_visible:
                _draw_arrow(
                    draw,
                    (float(arrow_pixels[0, 0]), float(arrow_pixels[0, 1])),
                    (float(arrow_pixels[1, 0]), float(arrow_pixels[1, 1])),
                    color=motion_axis_color,
                    width=4,
                )
            if show_contact_point and not contact_clean_mode:
                radius = float(contact_point_radius)
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=(32, 235, 70),
                    outline=(10, 45, 15),
                    width=3,
                )
            crop_mask = Image.fromarray((panel_foreground.astype(np.uint8) * 255), mode="L")
            crop_draw = ImageDraw.Draw(crop_mask)
            if len(panel_face_indices) > 0:
                panel_part_pixels = panel_projection["pixels"][panel_face_indices]
                crop_draw.rectangle(
                    (
                        float(panel_part_pixels[:, 0].min()),
                        float(panel_part_pixels[:, 1].min()),
                        float(panel_part_pixels[:, 0].max()),
                        float(panel_part_pixels[:, 1].max()),
                    ),
                    fill=255,
                )
            panel_bbox_corners = None if (contact_clean_mode or hide_contact_bbox) else _obb_corners(part)
            if panel_bbox_corners is not None:
                _, panel_bbox_pixels = _project_tool_points(
                    points_tool=panel_bbox_corners,
                    meta=meta,
                    view_index=panel_view_index,
                )
                if np.all(np.isfinite(panel_bbox_pixels)):
                    crop_draw.rectangle(
                        (
                            float(panel_bbox_pixels[:, 0].min()),
                            float(panel_bbox_pixels[:, 1].min()),
                            float(panel_bbox_pixels[:, 0].max()),
                            float(panel_bbox_pixels[:, 1].max()),
                        ),
                        fill=255,
                    )
            xs = face_pixels[:, 0]
            ys = face_pixels[:, 1]
            crop_draw.rectangle((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), fill=255)
            if show_motion_axis and arrow_visible:
                crop_draw.rectangle(
                    (
                        float(arrow_pixels[:, 0].min()) - 20.0,
                        float(arrow_pixels[:, 1].min()) - 20.0,
                        float(arrow_pixels[:, 0].max()) + 20.0,
                        float(arrow_pixels[:, 1].max()) + 20.0,
                    ),
                    fill=255,
                )
            crop_arr = np.asarray(crop_mask.filter(ImageFilter.MaxFilter(19)), dtype=np.uint8) > 0
            if crop_arr.any():
                default_pad = 72 if contact_zoom_mode else 96
                composed = _crop_to_mask(composed, crop_arr, pad=max(default_pad, int(contact_crop_pad)))
            panel_extent = panel_width if contact_zoom_mode else max(120, panel_width // 2)
            view_panels.append(
                _fit_image(
                    composed,
                    max_width=max(120, panel_extent),
                    max_height=max(1, panel_extent),
                )
            )
        if not view_panels:
            continue
        face_sheet = _make_contact_sheet(view_panels, columns=len(view_panels), margin=8)
        panels.append(_with_bottom_marker_label(face_sheet, text=face_marker, font=marker_font))
    if not panels:
        return None
    return _make_contact_sheet(panels, columns=min(2, len(panels)), margin=14)


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
        "candidate_part_face_highlights",
        "candidate_contact_face_overviews",
        "candidate_axis_face_overviews",
        "candidate_face_cards",
        "selected_contact_point_highlights",
        "selected_motion_axis_highlights",
        "selected_contact",
        "selected_axis",
        "candidate_contact_point_highlights",
        "candidate_motion_axis_highlights",
        "contact_marker_overviews",
        "function_contact_choices",
        "contact_point_choices",
    ):
        managed_dir = tool_out / managed_name
        if managed_dir.exists():
            shutil.rmtree(managed_dir)
    overview_dir = tool_out / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    marker_font = _font(size=22, bold=True)
    candidate_contact_face_overviews: dict[str, str] = {}
    clean_contact_overlay = True
    contact_local_zoom = False
    hide_contact_bbox = False
    contact_outline_only = False
    try:
        contact_point_radius = float('16.0')
    except ValueError:
        contact_point_radius = 16.0
    try:
        contact_crop_pad = int('128')
    except ValueError:
        contact_crop_pad = 128
    part_choice_overview: str | None = None
    isolated_part_choice_overview: str | None = None
    contact_marker_rows: list[dict[str, Any]] = []
    overview_entries: list[dict[str, Any]] = []
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
            overview_entries.append({"marker": marker, "image": panel})

        if not render_contact_markers:
            continue
        part_contact_rows = _contact_marker_table(
            part_id=part_id,
            part_ref=row["part_ref"],
            part=part,
            marker=marker,
            grounding_candidates=grounding_candidates,
        )
        contact_marker_rows.extend(part_contact_rows)
        contact_view_indices = _best_working_pose_view_indices(
            context=context,
            meta=meta,
            part_label=part_label,
            view_indices=view_indices,
            max_views=max(3, int(part_overview_views)),
            render_paths=render_paths,
            prefer_part_lower=False,
            prefer_dark_part_lower=False,
        )
        candidate_face_view_indices = [
            *contact_view_indices,
            *[view_index for view_index in view_indices if view_index not in set(contact_view_indices)],
        ]
        contact_face_overview = _candidate_face_overview(
            render_paths=render_paths,
            context=context,
            meta=meta,
            part=part,
            part_label=part_label,
            marker=marker,
            contact_rows=part_contact_rows,
            view_indices=candidate_face_view_indices,
            panel_width=panel_width,
            marker_font=marker_font,
            show_contact_point=False,
            show_motion_axis=False,
            clean_contact_overlay=clean_contact_overlay,
            contact_local_zoom=contact_local_zoom,
            hide_contact_bbox=hide_contact_bbox,
            contact_outline_only=contact_outline_only,
            contact_point_radius=contact_point_radius,
            contact_crop_pad=contact_crop_pad,
        )
        if contact_face_overview is not None:
            contact_face_overview_dir = tool_out / "candidate_contact_face_overviews"
            contact_face_overview_dir.mkdir(parents=True, exist_ok=True)
            contact_face_overview_path = (
                contact_face_overview_dir / f"candidate_{int(marker):03d}_contact_faces_overview.png"
            )
            contact_face_overview.save(contact_face_overview_path)
            candidate_contact_face_overviews[part_id] = str(contact_face_overview_path)
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
        "candidate_contact_face_overviews": candidate_contact_face_overviews,
        "region_marker_table": rows,
        "contact_marker_table": contact_marker_rows,
        "selected_contact_markers": sorted(selected_contact_markers) if selected_contact_markers else None,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(ROOT / "assets" / "tool_prior"))
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
    parser.add_argument("--panel-width", type=int, default=int('300'))
    parser.add_argument("--part-item-width", type=int, default=int('210'))
    parser.add_argument("--part-item-height", type=int, default=int('220'))
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
    results = [render_tool(tool_root=tool_root, output_root=output_root, prior_subdir=args.prior_subdir,
        selected_contact_markers=selected_markers_by_tool.get(tool_root.name) or selected_markers_by_tool.get(None),
        max_views=args.max_views, part_overview_views=args.part_overview_views,
        contact_card_views=args.contact_card_views, view_selection=args.view_selection,
        panel_width=args.panel_width, part_item_width=args.part_item_width, part_item_height=args.part_item_height)
        for tool_root in tool_roots]
    items = []
    region_rows, contact_rows = [], []
    for result in results:
        tool_id = result["tool_id"]
        for key, kind in (("part_choice_overview", "whole_mesh_part_choice_overview"),
                          ("isolated_part_choice_overview", "isolated_part_choice_overview")):
            if result.get(key):
                items.append({"id": f"{tool_id}_{key}", "kind": kind, "path": result[key]})
        for part_id, image in result["candidate_contact_face_overviews"].items():
            row = next(row for row in result["region_marker_table"] if row["part_id"] == part_id)
            items.append({"id": f"{tool_id}_part_candidate_{row['marker']}_contact_overview",
                          "kind": "candidate_part_contact_overview", "path": image, "part_marker": str(row["marker"])})
        region_rows.extend(result["region_marker_table"])
        contact_rows.extend(result["contact_marker_table"])
    evidence = {"schema_version": "v0.1", "items": items,
                "metadata": {"region_marker_table": region_rows, "contact_marker_table": contact_rows}}
    _write_json(output_root / "visual_evidence.json", evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
