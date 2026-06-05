"""Calibrated scene evidence rendering helpers for ToolFrame."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class PinholeCamera:
    """A calibrated pinhole camera with OpenCV-like image coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: tuple[tuple[float, float, float, float], ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def default_topdown_camera(
    *,
    width: int = 900,
    height: int = 700,
    camera_z: float = 1.2,
    focal_px: float = 900.0,
) -> PinholeCamera:
    """Return a top-down calibrated camera over the tabletop workspace."""

    return PinholeCamera(
        width=width,
        height=height,
        fx=focal_px,
        fy=focal_px,
        cx=width / 2.0,
        cy=height / 2.0,
        camera_to_world=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, camera_z),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _camera_matrices(camera: PinholeCamera) -> tuple[np.ndarray, np.ndarray]:
    camera_to_world = np.asarray(camera.camera_to_world, dtype=float)
    world_to_camera = np.linalg.inv(camera_to_world)
    return camera_to_world, world_to_camera


def project_world(camera: PinholeCamera, point_world: Sequence[float]) -> tuple[float, float, float]:
    """Project a 3D world point to `(u, v, depth)`."""

    _, world_to_camera = _camera_matrices(camera)
    point = np.asarray([float(point_world[0]), float(point_world[1]), float(point_world[2]), 1.0])
    camera_xyz = world_to_camera @ point
    depth = float(camera_xyz[2])
    if depth <= 1e-6:
        raise ValueError(f"point is behind camera: {point_world}")
    u = camera.fx * float(camera_xyz[0]) / depth + camera.cx
    v = camera.fy * float(camera_xyz[1]) / depth + camera.cy
    return u, v, depth


def _polygon_uv_depth(
    camera: PinholeCamera,
    points_world: Sequence[Sequence[float]],
) -> tuple[list[tuple[float, float]], float]:
    projected = [project_world(camera, point) for point in points_world]
    return [(u, v) for u, v, _ in projected], float(np.mean([depth for _, _, depth in projected]))


def _fill_depth_polygon(depth: np.ndarray, polygon: list[tuple[float, float]], value: float) -> None:
    mask = Image.new("L", (depth.shape[1], depth.shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)
    mask_array = np.asarray(mask) > 0
    depth[mask_array] = float(value)


def _box_top_corners(center: Sequence[float], size: Sequence[float]) -> list[tuple[float, float, float]]:
    cx, cy, cz = [float(v) for v in center[:3]]
    sx, sy, sz = [float(v) for v in size[:3]]
    z = cz + sz * 0.5
    return [
        (cx - sx * 0.5, cy - sy * 0.5, z),
        (cx + sx * 0.5, cy - sy * 0.5, z),
        (cx + sx * 0.5, cy + sy * 0.5, z),
        (cx - sx * 0.5, cy + sy * 0.5, z),
    ]


def _target_from_task(task_data: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    fixture = task_data.get("private_eval", {}).get("scene_fixture", {})
    targets = fixture.get("target_objects", []) if isinstance(fixture, Mapping) else []
    if targets and isinstance(targets[0], Mapping):
        pose = targets[0].get("pose_xyz")
        size = targets[0].get("size_xyz")
        if isinstance(pose, list) and len(pose) >= 3 and isinstance(size, list) and len(size) >= 3:
            return [float(v) for v in pose[:3]], [float(v) for v in size[:3]]
    state_bundle = task_data.get("execution", {}).get("state_bundle", {})
    actors = state_bundle.get("parttoolgen_benchmark_extra_actors", [])
    if actors and isinstance(actors[0], Mapping):
        pose_xyzw = actors[0].get("pose_xyzw")
        box = state_bundle.get("synthetic_target_box", {})
        size = box.get("size_xyz") if isinstance(box, Mapping) else None
        if isinstance(pose_xyzw, list) and len(pose_xyzw) >= 3 and isinstance(size, list):
            return [float(v) for v in pose_xyzw[:3]], [float(v) for v in size[:3]]
    return [0.12, -0.04, 0.5525], [0.045, 0.045, 0.045]


def _support_from_task(task_data: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    fixture = task_data.get("private_eval", {}).get("scene_fixture", {})
    support = fixture.get("support_surface", {}) if isinstance(fixture, Mapping) else {}
    pose = support.get("pose_xyz") if isinstance(support, Mapping) else None
    size = support.get("size_xyz") if isinstance(support, Mapping) else None
    if isinstance(pose, list) and len(pose) >= 3 and isinstance(size, list) and len(size) >= 3:
        return [float(v) for v in pose[:3]], [float(v) for v in size[:3]]
    return [0.0, 0.0, 0.38], [0.475, 0.4, 0.3]


def _draw_tool_hint(draw: ImageDraw.ImageDraw, camera: PinholeCamera, start_pose: Sequence[float]) -> None:
    if not isinstance(start_pose, Sequence) or len(start_pose) < 3:
        return
    x, y, z = [float(v) for v in start_pose[:3]]
    # A small schematic brush footprint. Tool mesh evidence remains the source
    # for function faces; this is only scene context.
    handle = _box_top_corners((x - 0.05, y, z), (0.13, 0.025, 0.018))
    head = _box_top_corners((x + 0.035, y, z), (0.075, 0.065, 0.018))
    for corners, fill in ((handle, (180, 42, 38)), (head, (36, 128, 120))):
        poly, _ = _polygon_uv_depth(camera, corners)
        draw.polygon(poly, fill=fill, outline=(25, 25, 25))


def render_topdown_scene_evidence(
    *,
    task_data: Mapping[str, Any],
    output_dir: Path,
    camera: PinholeCamera | None = None,
) -> dict[str, Any]:
    """Render a calibrated RGB-D top-down scene image for VLM grounding.

    The renderer may use private fixture geometry to create pixels, but the
    public artifact contains only image/depth/camera files and task language.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    camera = camera or default_topdown_camera()
    rgb = Image.new("RGB", (camera.width, camera.height), (244, 244, 242))
    depth = np.full((camera.height, camera.width), np.nan, dtype=np.float32)
    draw = ImageDraw.Draw(rgb)

    support_pose, support_size = _support_from_task(task_data)
    target_pose, target_size = _target_from_task(task_data)

    table_poly, table_depth = _polygon_uv_depth(camera, _box_top_corners(support_pose, support_size))
    draw.polygon(table_poly, fill=(224, 202, 162), outline=(115, 95, 68))
    _fill_depth_polygon(depth, table_poly, table_depth)

    target_poly, target_depth = _polygon_uv_depth(camera, _box_top_corners(target_pose, target_size))
    draw.polygon(target_poly, fill=(30, 85, 218), outline=(18, 28, 60))
    _fill_depth_polygon(depth, target_poly, target_depth)

    start_pose = task_data.get("execution", {}).get("simtoolreal_start_pose", [])
    _draw_tool_hint(draw, camera, start_pose)

    font = _font(20)
    small_font = _font(16)
    instruction = str(task_data.get("instruction", ""))
    draw.rectangle((12, 12, camera.width - 12, 62), fill=(255, 255, 255), outline=(205, 205, 205))
    draw.text((24, 24), instruction, font=font, fill=(25, 25, 25))
    draw.line((camera.width - 150, camera.height - 60, camera.width - 150, camera.height - 120), fill=(35, 35, 35), width=3)
    draw.polygon(
        [
            (camera.width - 150, camera.height - 130),
            (camera.width - 160, camera.height - 112),
            (camera.width - 140, camera.height - 112),
        ],
        fill=(35, 35, 35),
    )
    draw.text((camera.width - 205, camera.height - 150), "workspace forward", font=small_font, fill=(35, 35, 35))

    rgb_path = output_dir / "scene_rgb.png"
    depth_path = output_dir / "scene_depth.npy"
    camera_path = output_dir / "camera.json"
    evidence_path = output_dir / "scene_evidence.json"
    private_path = output_dir / "_private_render_manifest.json"

    rgb.save(rgb_path)
    np.save(depth_path, depth)
    camera_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": "toolframe_camera_v0",
                "model": "pinhole",
                **camera.to_json(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "toolframe_scene_evidence_v0",
        "task_id": task_data.get("task_id"),
        "instruction": task_data.get("instruction"),
        "items": [
            {
                "kind": "scene_rgb",
                "id": "scene_rgb_topdown",
                "path": str(rgb_path),
            },
            {
                "kind": "scene_depth",
                "id": "scene_depth_topdown",
                "path": str(depth_path),
            },
            {
                "kind": "camera",
                "id": "scene_camera_topdown",
                "path": str(camera_path),
            },
        ],
        "metadata": {
            "prompt_images": [str(rgb_path)],
            "render_backend": "calibrated_topdown_schematic",
            "public_contract": "VLM may use only task language and pixels; no target refs or axes are exposed.",
        },
    }
    evidence_path.write_text(__import__("json").dumps(evidence, indent=2) + "\n", encoding="utf-8")
    private_path.write_text(
        __import__("json").dumps(
            {
                "schema_version": "toolframe_private_scene_render_manifest_v0",
                "task_id": task_data.get("task_id"),
                "render_used_private_fixture_geometry": True,
                "not_for_method_input": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "scene_evidence": str(evidence_path),
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "camera": str(camera_path),
    }

