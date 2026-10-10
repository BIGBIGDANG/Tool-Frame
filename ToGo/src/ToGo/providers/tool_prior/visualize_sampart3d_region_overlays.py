#!/usr/bin/env python3
"""Draw numbered SAMPart3D region overlays for VLM region selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ToGo.providers.tool_prior.sampart3d_projection_helpers import (
    PALETTE,
    _font,
    _make_contact_sheet,
    _mesh_foreground_mask,
    _panel_header,
    _presentation_image,
    _projection_context,
    _project_render_points,
    _project_view_faces,
)


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "ToGo" / "src").exists() and (parent / "ToGo" / "scripts").exists():
            return parent
    return current.parents[5]


ROOT = repo_root()

POINT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 68, 68),
    (255, 218, 36),
    (0, 190, 255),
    (184, 96, 255),
    (0, 218, 128),
    (255, 128, 0),
    (0, 235, 216),
    (255, 91, 190),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_color(label: int) -> tuple[int, int, int]:
    return POINT_PALETTE[int(label) % len(POINT_PALETTE)]


def _iter_source_artifacts(value: Any) -> list[str]:
    if isinstance(value, dict):
        artifacts: list[str] = []
        source = value.get("source_artifact")
        if isinstance(source, str):
            artifacts.append(source)
        for child in value.values():
            artifacts.extend(_iter_source_artifacts(child))
        return artifacts
    if isinstance(value, list):
        artifacts = []
        for child in value:
            artifacts.extend(_iter_source_artifacts(child))
        return artifacts
    return []


def _face_labels_from_part_graph(tool_part_graph: dict[str, Any]) -> Path | None:
    artifacts = [
        Path(path)
        for path in _iter_source_artifacts(tool_part_graph)
        if path.endswith(".npy") and "mesh_" in Path(path).name
    ]
    existing = [path for path in artifacts if path.exists()]
    if not existing:
        return None
    counts: dict[Path, int] = {}
    for path in existing:
        counts[path] = counts.get(path, 0) + 1
    return max(counts, key=lambda path: (counts[path], len(str(path))))


def _projection_artifact_from_geometry_prior(
    *,
    tool_root: Path,
    prior_dir: Path,
    tool_part_graph: dict[str, Any],
) -> dict[str, Any]:
    grounding_candidates = _load_json(prior_dir / "grounding_candidates.json")
    metadata = grounding_candidates.get("metadata") or {}
    source_mesh = metadata.get("source_mesh")
    face_labels = metadata.get("face_labels")
    if not source_mesh:
        raise RuntimeError(f"grounding_candidates.json is missing metadata.source_mesh: {prior_dir}")
    source_mesh = Path(source_mesh)
    if not source_mesh.is_absolute():
        source_mesh = ROOT / source_mesh
    if not face_labels:
        face_labels_path = _face_labels_from_part_graph(tool_part_graph)
        if face_labels_path is None:
            raise RuntimeError(f"could not infer SAMPart3D face labels for {tool_root.name}")
        face_labels = str(face_labels_path)
    face_labels = Path(face_labels)
    if not face_labels.is_absolute():
        face_labels = ROOT / face_labels
    render_dir = tool_root / "data_root" / tool_root.name
    if not (render_dir / "meta.json").exists():
        raise RuntimeError(f"SAMPart3D render metadata not found: {render_dir / 'meta.json'}")
    return {
        "source_mesh": str(source_mesh),
        "face_labels": str(face_labels),
        "render_dir": str(render_dir),
        "metadata": {
            "projection_mode": "sampart3d_glb",
            "front_facing_cos": 0.0,
            "occlusion_filter": True,
            "occlusion_window_px": 4,
            "occlusion_depth_tolerance": 0.02,
            "occlusion_depth_rel_tolerance": 0.01,
            "source": "sampart3d_geometry_prior",
        },
    }


def _draw_part_marker(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    width, height = image_size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    radius = max(13, int(max(text_w, text_h) * 0.62) + 7)
    x = min(max(float(xy[0]), radius + 3), max(radius + 3, width - radius - 3))
    y = min(max(float(xy[1]), radius + 3), max(radius + 3, height - radius - 3))
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, fill=(255, 210, 45), outline=(20, 20, 20), width=3)
    draw.text(
        (x - text_w * 0.5 - bbox[0], y - text_h * 0.5 - bbox[1]),
        text,
        fill=(15, 15, 15),
        font=font,
    )


def _part_rows(tool_part_graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for marker_index, (part_id, part) in enumerate(sorted(tool_part_graph.get("parts", {}).items()), start=1):
        rows.append(
            {
                "marker": str(marker_index),
                "part_id": part_id,
                "part_ref": f"tool.parts.{part_id}",
                "source_label": part.get("label"),
                "point_count": part.get("point_count"),
                "role_candidates": part.get("role_candidates", []),
                "part_label": part.get("provenance", {}).get("part_label"),
            }
        )
    return rows


def _select_views(total: int, *, max_views: int, view_selection: str) -> list[int]:
    if view_selection == "all" or max_views <= 0 or max_views >= total:
        return list(range(total))
    if view_selection == "first":
        return list(range(max_views))
    if max_views == 1:
        return [0]
    raw = np.linspace(0, total - 1, max_views)
    selected: list[int] = []
    for value in raw:
        index = int(round(float(value)))
        if index not in selected:
            selected.append(index)
    index = 0
    while len(selected) < max_views and index < total:
        if index not in selected:
            selected.append(index)
        index += 1
    return sorted(selected)


def _face_indices_for_label(
    *,
    labels: np.ndarray,
    inside: np.ndarray,
    label: int,
    face_count: int,
) -> np.ndarray:
    if len(labels) == face_count and len(inside) == face_count:
        return np.nonzero(inside & (labels == label))[0]
    # Fallback for unusual files with skipped labels: still show region support
    # with centroid markers rather than silently drawing wrong polygons.
    limit = min(len(labels), len(inside))
    return np.nonzero(inside[:limit] & (labels[:limit] == label))[0]


def _draw_projected_faces(
    draw: ImageDraw.ImageDraw,
    *,
    mesh_geometry: dict[str, Any],
    vertex_front: np.ndarray,
    vertex_pixels: np.ndarray,
    face_indices: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
    outline_alpha: int,
) -> None:
    faces = mesh_geometry["faces"]
    for face_index in face_indices:
        if face_index < 0 or face_index >= len(faces):
            continue
        face = faces[int(face_index)]
        if not any(bool(vertex_front[index]) for index in face):
            continue
        polygon = [(float(vertex_pixels[index, 0]), float(vertex_pixels[index, 1])) for index in face]
        draw.polygon(polygon, fill=(*color, alpha), outline=(*color, outline_alpha))


def _draw_point_cloud(
    image: Image.Image,
    *,
    pixels: np.ndarray,
    indices: np.ndarray,
    color: tuple[int, int, int],
    radius: float,
    alpha: int,
    outline: tuple[int, int, int] | None = (15, 15, 15),
    outline_width: float = 1.25,
    max_points: int = 0,
) -> None:
    if len(indices) == 0:
        return
    if max_points > 0 and len(indices) > max_points:
        step = int(math.ceil(len(indices) / float(max_points)))
        indices = indices[::step]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if outline is not None and outline_width > 0:
        outline_radius = radius + outline_width
        outline_alpha = min(230, max(40, int(alpha * 0.72)))
        for face_index in indices:
            x, y = pixels[int(face_index)]
            draw.ellipse(
                (x - outline_radius, y - outline_radius, x + outline_radius, y + outline_radius),
                fill=(*outline, outline_alpha),
            )
    for face_index in indices:
        x, y = pixels[int(face_index)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
    image.alpha_composite(overlay)


def _project_mesh_vertices(context: dict[str, Any], view_index: int) -> tuple[np.ndarray, np.ndarray]:
    meta = context["meta"]
    return _project_render_points(
        points=context["mesh_geometry"]["vertices_render"],
        transform=meta["transforms"][view_index],
        camera_angle_x=float(meta["camera_angle_x"]),
        image_width=int(meta.get("width") or 512),
        image_height=int(meta.get("height") or 512),
    )


def visualize_tool(
    *,
    tool_root: Path,
    output_dir: Path,
    prior_subdir: str,
    max_views: int,
    view_selection: str,
    style: str,
    point_radius: float,
    focus_point_radius: float,
    max_points_per_part: int,
    show_panel_headers: bool,
) -> dict[str, Any]:
    tool_id = tool_root.name
    prior_dir = tool_root / prior_subdir
    tool_part_graph = _load_json(prior_dir / "tool_part_graph.json")
    anchor_artifact = _projection_artifact_from_geometry_prior(
        tool_root=tool_root,
        prior_dir=prior_dir,
        tool_part_graph=tool_part_graph,
    )
    current_face_labels = _face_labels_from_part_graph(tool_part_graph)
    if current_face_labels is not None:
        anchor_artifact["face_labels"] = str(current_face_labels)
    context = _projection_context(anchor_artifact)
    meta = context["meta"]
    rows = _part_rows(tool_part_graph)
    marker_by_part = {row["part_id"]: str(row["marker"]) for row in rows}
    render_dir = Path(anchor_artifact["render_dir"])
    render_paths = sorted(render_dir.glob("render_*.webp"))
    view_indices = _select_views(len(render_paths), max_views=max_views, view_selection=view_selection)

    label_to_part_id = {
        int(row["part_label"]): row["part_id"]
        for row in rows
        if row.get("part_label") is not None
    }
    panels: list[Image.Image] = []
    focus_panels: dict[str, list[Image.Image]] = {row["part_id"]: [] for row in rows}
    visible_parts: set[str] = set()
    marker_font = _font(size=14, bold=True)
    header_font = _font(size=12, bold=True)
    mesh_faces = context["mesh_geometry"]["faces"]
    mesh_labels = np.asarray(context["mesh_geometry"]["labels"], dtype=np.int64)
    for view_index in view_indices:
        raw = Image.open(render_paths[view_index]).convert("RGB")
        projection = _project_view_faces(context, view_index)
        vertex_front, vertex_pixels = _project_mesh_vertices(context, view_index)
        foreground = _mesh_foreground_mask(mesh_geometry=context["mesh_geometry"], meta=meta, view_index=view_index)
        base = _presentation_image(raw, reference=raw, foreground_mask=foreground, crop=False).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        pixels = projection["pixels"]
        labels = context["labels"]
        inside = projection["inside"]
        for label in sorted(label_to_part_id):
            part_id = label_to_part_id[label]
            indices = _face_indices_for_label(
                labels=mesh_labels,
                inside=inside,
                label=label,
                face_count=len(mesh_faces),
            )
            if len(indices) == 0:
                continue
            visible_parts.add(part_id)
            color = PALETTE[label % len(PALETTE)]
            if style in {"surface", "both"}:
                _draw_projected_faces(
                    draw_overlay,
                    mesh_geometry=context["mesh_geometry"],
                    vertex_front=vertex_front,
                    vertex_pixels=vertex_pixels,
                    face_indices=indices,
                    color=color,
                    alpha=74 if style == "both" else 118,
                    outline_alpha=120 if style == "both" else 210,
                )
        composed = Image.alpha_composite(base, overlay).convert("RGB")
        composed_rgba = composed.convert("RGBA")
        if style in {"pointcloud", "both"}:
            for label in sorted(label_to_part_id):
                indices = np.nonzero(inside & (labels == label))[0]
                if len(indices) == 0:
                    continue
                color = _point_color(label)
                _draw_point_cloud(
                    composed_rgba,
                    pixels=pixels,
                    indices=indices,
                    color=color,
                    radius=point_radius,
                    alpha=235,
                    outline=(255, 255, 255),
                    outline_width=1.6,
                    max_points=max_points_per_part,
                )
                _draw_point_cloud(
                    composed_rgba,
                    pixels=pixels,
                    indices=indices,
                    color=color,
                    radius=max(1.0, point_radius * 0.58),
                    alpha=255,
                    outline=(15, 15, 15),
                    outline_width=0.7,
                    max_points=max_points_per_part,
                )
        composed = composed_rgba.convert("RGB")
        draw = ImageDraw.Draw(composed)
        for label in sorted(label_to_part_id):
            part_id = label_to_part_id[label]
            indices = np.nonzero(inside & (labels == label))[0]
            if len(indices) == 0:
                continue
            xy = pixels[indices].mean(axis=0)
            _draw_part_marker(
                draw,
                xy=(float(xy[0]), float(xy[1])),
                text=marker_by_part[part_id],
                image_size=composed.size,
                font=marker_font,
            )
        panel = _panel_header(composed, f"view {view_index:02d}", header_font) if show_panel_headers else composed
        panels.append(panel.resize((384, int(panel.height * (384 / panel.width)))))

        for label in sorted(label_to_part_id):
            part_id = label_to_part_id[label]
            indices = _face_indices_for_label(
                labels=mesh_labels,
                inside=inside,
                label=label,
                face_count=len(mesh_faces),
            )
            if len(indices) == 0:
                continue
            focus_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            focus_draw = ImageDraw.Draw(focus_overlay)
            for other_label in sorted(label_to_part_id):
                other_indices = _face_indices_for_label(
                    labels=mesh_labels,
                    inside=inside,
                    label=other_label,
                    face_count=len(mesh_faces),
                )
                if len(other_indices) == 0:
                    continue
                if style in {"surface", "both"}:
                    if other_label == label:
                        color = PALETTE[label % len(PALETTE)]
                        _draw_projected_faces(
                            focus_draw,
                            mesh_geometry=context["mesh_geometry"],
                            vertex_front=vertex_front,
                            vertex_pixels=vertex_pixels,
                            face_indices=other_indices,
                            color=color,
                            alpha=120 if style == "both" else 170,
                            outline_alpha=190 if style == "both" else 245,
                        )
                    else:
                        _draw_projected_faces(
                            focus_draw,
                            mesh_geometry=context["mesh_geometry"],
                            vertex_front=vertex_front,
                            vertex_pixels=vertex_pixels,
                            face_indices=other_indices,
                            color=(170, 170, 170),
                            alpha=20 if style == "both" else 30,
                            outline_alpha=32 if style == "both" else 45,
                        )
            focus_image = Image.alpha_composite(base, focus_overlay).convert("RGB")
            focus_rgba = focus_image.convert("RGBA")
            if style in {"pointcloud", "both"}:
                for other_label in sorted(label_to_part_id):
                    other_indices = np.nonzero(inside & (labels == other_label))[0]
                    if len(other_indices) == 0:
                        continue
                    if other_label == label:
                        color = _point_color(label)
                        _draw_point_cloud(
                            focus_rgba,
                            pixels=pixels,
                            indices=other_indices,
                            color=color,
                            radius=focus_point_radius + 1.4,
                            alpha=205,
                            outline=(255, 255, 255),
                            outline_width=1.8,
                            max_points=0,
                        )
                        _draw_point_cloud(
                            focus_rgba,
                            pixels=pixels,
                            indices=other_indices,
                            color=color,
                            radius=focus_point_radius,
                            alpha=255,
                            outline=(10, 10, 10),
                            outline_width=1.0,
                            max_points=0,
                        )
                    else:
                        _draw_point_cloud(
                            focus_rgba,
                            pixels=pixels,
                            indices=other_indices,
                            color=(175, 175, 175),
                            radius=max(1.2, point_radius * 0.55),
                            alpha=78,
                            outline=None,
                            max_points=max_points_per_part,
                        )
            focus_image = focus_rgba.convert("RGB")
            focus_marker_draw = ImageDraw.Draw(focus_image)
            marker_indices = np.nonzero(inside & (labels == label))[0]
            if len(marker_indices):
                xy = pixels[marker_indices].mean(axis=0)
                _draw_part_marker(
                    focus_marker_draw,
                    xy=(float(xy[0]), float(xy[1])),
                    text=marker_by_part[part_id],
                    image_size=focus_image.size,
                    font=marker_font,
                )
            focus_panel = (
                _panel_header(focus_image, f"region {marker_by_part[part_id]} view {view_index:02d}", header_font)
                if show_panel_headers
                else focus_image
            )
            focus_panels[part_id].append(focus_panel.resize((384, int(focus_panel.height * (384 / focus_panel.width)))))

    sheet = _make_contact_sheet(panels, columns=4, margin=10)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{tool_id}_sampart3d_region_overlay.png"
    sheet.save(overlay_path)
    focus_outputs: dict[str, str] = {}
    for part_id, part_panels in focus_panels.items():
        if not part_panels:
            continue
        focus_sheet = _make_contact_sheet(part_panels, columns=4, margin=10)
        focus_path = output_dir / f"{tool_id}_{part_id}_solo_region_overlay.png"
        focus_sheet.save(focus_path)
        focus_outputs[part_id] = str(focus_path)
    return {
        "tool_id": tool_id,
        "part_count": len(rows),
        "view_count": len(view_indices),
        "overlay": str(overlay_path),
        "part_focus_overlays": focus_outputs,
        "style": style,
        "region_marker_table": rows,
        "visible_parts": sorted(visible_parts),
    }
