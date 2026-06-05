#!/usr/bin/env python3
"""Draw VLM scene keypoints and motion arrow over the prompt image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    uv: tuple[float, float],
    fill: tuple[int, int, int],
    radius: int = 18,
) -> None:
    x, y = uv
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=(45, 45, 45),
        width=4,
    )


def _draw_center_dot(
    draw: ImageDraw.ImageDraw,
    *,
    uv: tuple[float, float],
    radius: int = 6,
) -> None:
    x, y = uv
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(15, 15, 15),
    )


def _projected_action_is_degenerate(
    keypoints: Mapping[str, Any],
    *,
    object_uv: tuple[float, float],
    end_uv: tuple[float, float],
    pixel_threshold: float = 6.0,
) -> bool:
    dx = float(end_uv[0]) - float(object_uv[0])
    dy = float(end_uv[1]) - float(object_uv[1])
    return (dx * dx + dy * dy) ** 0.5 <= pixel_threshold


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int] = (20, 20, 20),
    width: int = 4,
    dash: float = 14.0,
    gap: float = 9.0,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    t = 0.0
    while t < length:
        t1 = min(t + dash, length)
        p0 = (start[0] + ux * t, start[1] + uy * t)
        p1 = (start[0] + ux * t1, start[1] + uy * t1)
        draw.line((*p0, *p1), fill=fill, width=width)
        t += dash + gap


def _point(data: Mapping[str, Any], key: str) -> tuple[float, float]:
    row = data.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"missing {key}")
    return float(row["u"]), float(row["v"])


def _point_any(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, tuple[float, float]]:
    for key in keys:
        if isinstance(data.get(key), Mapping):
            return key, _point(data, key)
    raise ValueError(f"missing one of {keys}")


def _path_rows(data: Mapping[str, Any], key: str) -> list[tuple[str, tuple[float, float]]]:
    rows = data.get(key) or []
    if not isinstance(rows, list):
        return []
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        result.append((str(row.get("id") or f"{key}_{index}"), (float(row["u"]), float(row["v"]))))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-evidence", required=True)
    parser.add_argument("--scene-keypoints", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = _load_json(args.scene_evidence)
    keypoints = _load_json(args.scene_keypoints)
    prompt_images = evidence["metadata"]["prompt_images"]
    grounding_image_index = int(
        keypoints.get(
            "grounding_image_index",
            evidence.get("metadata", {}).get("grounding_image_index", 0),
        )
    )
    rgb_path = Path(prompt_images[grounding_image_index])
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    object_key, object_uv = _point_any(
        keypoints,
        ("start_point",),
    )
    path_uvs = _path_rows(keypoints, "path_points")
    end_source = "path_points" if path_uvs else "end_point"
    if path_uvs:
        end_uv = path_uvs[-1][1]
        waypoint_uvs = path_uvs[1:-1] if len(path_uvs) >= 2 else []
    elif isinstance(keypoints.get("end_point"), Mapping):
        end_uv = _point(keypoints, "end_point")
    else:
        waypoint_uvs = _path_rows(keypoints, "waypoints")
        if waypoint_uvs:
            end_source = "final_waypoint"
            end_uv = waypoint_uvs[-1][1]
            waypoint_uvs = waypoint_uvs[:-1]
        else:
            action_direction = keypoints.get("action_direction")
            end_source = "action_direction"
            if not isinstance(action_direction, Mapping):
                raise ValueError("missing end_point or waypoint path")
            else:
                start = action_direction.get("start")
                end = action_direction.get("end")
                if not isinstance(start, Mapping) or not isinstance(end, Mapping):
                    raise ValueError("action_direction must include start and end")
                end_uv = (float(end["u"]), float(end["v"]))
    action_direction = keypoints.get("action_direction")
    if isinstance(action_direction, Mapping):
        start = action_direction.get("start")
        end = action_direction.get("end")
        if not isinstance(start, Mapping) or not isinstance(end, Mapping):
            raise ValueError("action_direction must include start and end")
    if "waypoint_uvs" not in locals():
        waypoint_uvs = _path_rows(keypoints, "waypoints")
    draw_down_dot = False
    if end_source == "action_direction" and _projected_action_is_degenerate(
        keypoints,
        object_uv=object_uv,
        end_uv=end_uv,
    ):
        end_uv = object_uv
        waypoint_uvs = []
        draw_down_dot = True

    if not draw_down_dot:
        path = [object_uv, *(uv for _, uv in waypoint_uvs), end_uv]
        for start, end in zip(path, path[1:]):
            _draw_dashed_line(draw, start=start, end=end)
    marker_rows = [
        (object_uv, (232, 137, 190)),
        *[
            (uv, (111, 202, 232))
            for _, uv in waypoint_uvs
        ],
    ]
    if not draw_down_dot:
        marker_rows.append((end_uv, (103, 190, 111)))
    for uv, color in marker_rows:
        _draw_marker(draw, uv=uv, fill=color)
    if draw_down_dot:
        _draw_center_dot(draw, uv=object_uv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(json.dumps({"overlay": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
