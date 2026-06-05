#!/usr/bin/env python3
"""Shared SAMPart3D render/projection helpers for visual evidence scripts."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from parttoolgen.providers.tool_prior.sampart3d_projection_geometry import (
    SAMPart3D_RENDER_ROTATION,
    face_geometry,
    occlusion_visible_mask,
    project_points,
)
from parttoolgen.providers.tool_prior.sampart3d_mesh_candidates import (
    load_npy_1d_int,
    load_obj_mesh,
)


PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (131, 56, 236),
    (255, 0, 110),
    (58, 134, 255),
    (107, 144, 128),
    (0, 109, 119),
    (157, 2, 8),
    (94, 96, 206),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(*, size: int = 12, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _border_background_mask(image: Image.Image, *, threshold: int = 8) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    dark = arr.max(axis=2) <= threshold
    h, w = dark.shape
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if dark[0, x]:
            q.append((0, x))
        if dark[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if dark[y, 0]:
            q.append((y, 0))
        if dark[y, w - 1]:
            q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        if visited[y, x] or not dark[y, x]:
            continue
        visited[y, x] = True
        if y > 0:
            q.append((y - 1, x))
        if y + 1 < h:
            q.append((y + 1, x))
        if x > 0:
            q.append((y, x - 1))
        if x + 1 < w:
            q.append((y, x + 1))
    return visited


def _foreground_crop_box(foreground: np.ndarray, *, pad: int = 140) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        return None
    h, w = foreground.shape
    left = max(0, int(xs.min()) - pad)
    right = min(w, int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(h, int(ys.max()) + pad + 1)
    return left, top, right, bottom


def _presentation_image(
    image: Image.Image,
    *,
    reference: Image.Image,
    foreground_mask: np.ndarray | None = None,
    crop: bool = True,
) -> Image.Image:
    if foreground_mask is None:
        foreground_mask = ~_border_background_mask(reference)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    arr[~foreground_mask] = np.asarray([248, 248, 248], dtype=np.uint8)
    result = Image.fromarray(arr, mode="RGB")
    result = ImageEnhance.Brightness(result).enhance(1.28)
    result = ImageEnhance.Contrast(result).enhance(1.12)
    crop_box = _foreground_crop_box(foreground_mask) if crop else None
    if crop_box is not None:
        result = result.crop(crop_box)
    return result


def _project_render_points(
    *,
    points: np.ndarray,
    transform: list[list[float]],
    camera_angle_x: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(transform, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (homog @ w2c.T)[:, :3]
    z = cam[:, 2]
    front = z < -1e-6
    tan_half = math.tan(float(camera_angle_x) * 0.5)
    x_ndc = (cam[:, 0] / np.maximum(-z, 1e-9)) / tan_half
    y_ndc = (cam[:, 1] / np.maximum(-z, 1e-9)) / tan_half
    u = (x_ndc + 1.0) * 0.5 * float(image_width)
    v = (1.0 - y_ndc) * 0.5 * float(image_height)
    return front, np.stack([u, v], axis=1)


def _mesh_render_geometry(anchor_artifact: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    mesh = load_obj_mesh(anchor_artifact["source_mesh"])
    face_labels = anchor_artifact.get("face_labels")
    labels = (
        load_npy_1d_int(face_labels)
        if face_labels is not None
        else np.zeros(len(mesh.faces), dtype=np.int64)
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if anchor_artifact.get("metadata", {}).get("projection_mode", "sampart3d_glb") == "sampart3d_glb":
        vertices = vertices @ SAMPart3D_RENDER_ROTATION
    vertices = vertices * float(meta["scaling_factor"]) + np.asarray(meta["mesh_offset"], dtype=np.float64)
    return {"vertices_render": vertices, "faces": mesh.faces, "labels": labels}


def _mesh_foreground_mask(
    *,
    mesh_geometry: dict[str, Any],
    meta: dict[str, Any],
    view_index: int,
) -> np.ndarray:
    image_width = int(meta.get("width") or 512)
    image_height = int(meta.get("height") or 512)
    front, pixels = _project_render_points(
        points=mesh_geometry["vertices_render"],
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=image_width,
        image_height=image_height,
    )
    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    labels = mesh_geometry["labels"]
    for face_index, face in enumerate(mesh_geometry["faces"]):
        if int(labels[face_index]) < 0:
            continue
        if not any(bool(front[index]) for index in face):
            continue
        polygon = [(float(pixels[index, 0]), float(pixels[index, 1])) for index in face]
        draw.polygon(polygon, fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(9))
    return np.asarray(mask, dtype=np.uint8) > 0


def _projection_context(anchor_artifact: dict[str, Any]) -> dict[str, Any]:
    render_dir = Path(anchor_artifact["render_dir"])
    meta = _load_json(render_dir / "meta.json")
    mesh_geometry = _mesh_render_geometry(anchor_artifact, meta)
    faces = face_geometry(
        source_mesh=anchor_artifact["source_mesh"],
        face_labels_path=anchor_artifact["face_labels"],
        meta=meta,
        projection_mode=anchor_artifact.get("metadata", {}).get("projection_mode", "sampart3d_glb"),
    )
    return {
        "meta": meta,
        "points_render": np.stack([face.centroid_render for face in faces], axis=0),
        "normals_render": np.stack([face.normal_render for face in faces], axis=0),
        "points_tool": np.stack([face.centroid_tool for face in faces], axis=0),
        "labels": np.asarray([face.label for face in faces], dtype=np.int64),
        "front_facing_cos": float(anchor_artifact.get("metadata", {}).get("front_facing_cos", 0.0)),
        "occlusion_filter": bool(anchor_artifact.get("metadata", {}).get("occlusion_filter", True)),
        "occlusion_window_px": int(anchor_artifact.get("metadata", {}).get("occlusion_window_px", 4)),
        "occlusion_depth_tolerance": float(anchor_artifact.get("metadata", {}).get("occlusion_depth_tolerance", 0.02)),
        "occlusion_depth_rel_tolerance": float(
            anchor_artifact.get("metadata", {}).get("occlusion_depth_rel_tolerance", 0.01)
        ),
        "mesh_geometry": mesh_geometry,
    }


def _project_view_faces(context: dict[str, Any], view_index: int) -> dict[str, Any]:
    meta = context["meta"]
    inside, pixels, cam = project_points(
        points=context["points_render"],
        normals=context["normals_render"],
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=int(meta.get("width") or 512),
        image_height=int(meta.get("height") or 512),
        front_facing_cos=float(context["front_facing_cos"]),
    )
    if bool(context.get("occlusion_filter", True)):
        inside = occlusion_visible_mask(
            candidate_mask=inside,
            pixels=pixels,
            cam=cam,
            image_width=int(meta.get("width") or 512),
            image_height=int(meta.get("height") or 512),
            occlusion_window_px=int(context["occlusion_window_px"]),
            occlusion_depth_tolerance=float(context["occlusion_depth_tolerance"]),
            occlusion_depth_rel_tolerance=float(context["occlusion_depth_rel_tolerance"]),
        )
    return {"inside": inside, "pixels": pixels}


def _panel_header(image: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    width, height = image.size
    header_h = 24
    panel = Image.new("RGB", (width, height + header_h), (245, 245, 245))
    panel.paste(image, (0, header_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), text, fill=(20, 20, 20), font=font)
    return panel


def _make_contact_sheet(images: list[Image.Image], *, columns: int = 2, margin: int = 12) -> Image.Image:
    if not images:
        return Image.new("RGB", (512, 128), (255, 255, 255))
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = int(math.ceil(len(images) / float(columns)))
    sheet = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * margin, rows * cell_h + (rows + 1) * margin),
        (255, 255, 255),
    )
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        sheet.paste(image, (x, y))
    return sheet
