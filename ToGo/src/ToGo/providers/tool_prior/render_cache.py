"""Prepare calibrated software tool views from a mesh, without Blender or evaluation."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[5]

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")

def _unit(vector: Any) -> Any:
    import numpy as np

    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def _look_at_transform(eye: Any, target: Any = (0.0, 0.0, 0.0)) -> list[list[float]]:
    import numpy as np

    eye_np = np.asarray(eye, dtype=np.float64)
    target_np = np.asarray(target, dtype=np.float64)
    forward = _unit(target_np - eye_np)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _unit(np.cross(forward, world_up))
    if float(np.linalg.norm(right)) <= 1e-12:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    back = _unit(-forward)
    up = _unit(np.cross(back, right))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = back
    transform[:3, 3] = eye_np
    return transform.tolist()


def _project_vertices(
    *,
    vertices: Any,
    transform: list[list[float]],
    camera_angle_x: float,
    image_width: int,
    image_height: int,
) -> tuple[Any, Any, Any]:
    import numpy as np

    c2w = np.asarray(transform, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (homog @ w2c.T)[:, :3]
    z = cam[:, 2]
    front = z < -1e-6
    tan_half = math.tan(float(camera_angle_x) * 0.5)
    x_ndc = (cam[:, 0] / np.maximum(-z, 1e-9)) / tan_half
    y_ndc = (cam[:, 1] / np.maximum(-z, 1e-9)) / tan_half
    u = (x_ndc + 1.0) * 0.5 * float(image_width)
    v = (1.0 - y_ndc) * 0.5 * float(image_height)
    return front, np.stack([u, v], axis=1), cam


def _material_base_color(material: Any) -> Any:
    import numpy as np

    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        factor = getattr(material, "diffuse", None)
    if factor is None:
        return np.asarray([178, 178, 178], dtype=np.uint8)
    arr = np.asarray(factor, dtype=np.float64).reshape(-1)
    if len(arr) < 3:
        return np.asarray([178, 178, 178], dtype=np.uint8)
    if float(arr[:3].max()) <= 1.0:
        arr[:3] *= 255.0
    return np.clip(arr[:3], 0, 255).astype(np.uint8)


def _find_external_texture(source_mesh: Path, material_name: str | None) -> Path | None:
    texture_dir = source_mesh.parent / "textures"
    if not texture_dir.exists():
        return None
    textures = sorted(
        path
        for path in texture_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not textures:
        return None
    name = (material_name or "").lower()
    if name:
        for path in textures:
            if name in path.stem.lower() or path.stem.lower() in name:
                return path
    return textures[0]


def _face_colors_from_visual(visual: Any, faces: Any, source_mesh: Path) -> Any:
    import numpy as np
    from PIL import Image

    material = getattr(visual, "material", None)
    base = _material_base_color(material)
    uv = getattr(visual, "uv", None)
    image = getattr(material, "image", None) if material is not None else None
    texture_path = _find_external_texture(source_mesh, getattr(material, "name", None))
    if image is None and texture_path is not None:
        image = Image.open(texture_path).convert("RGB")
    if image is None or uv is None:
        return np.tile(base.reshape(1, 3), (len(faces), 1)).astype(np.uint8)
    image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = image_arr.shape[:2]
    uv_arr = np.asarray(uv, dtype=np.float64)
    colors = np.empty((len(faces), 3), dtype=np.uint8)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        face_uv = uv_arr[face].mean(axis=0)
        u = float(face_uv[0]) % 1.0
        v = float(face_uv[1]) % 1.0
        x = int(np.clip(round(u * (width - 1)), 0, width - 1))
        y = int(np.clip(round((1.0 - v) * (height - 1)), 0, height - 1))
        sampled = image_arr[y, x, :3].astype(np.float64)
        if float(base.max()) < 250.0:
            sampled = sampled * (base.astype(np.float64) / 255.0)
        colors[face_index] = np.clip(sampled, 0, 255).astype(np.uint8)
    return colors


def _load_material_visual_mesh(source_mesh: Path) -> tuple[Any, Any, Any, str] | None:
    import numpy as np
    import trimesh  # type: ignore

    from ToGo.providers.tool_prior.sampart3d_mesh_candidates import load_obj_mesh

    candidates: list[Path] = []
    if (source_mesh.parent / "material.mtl").exists() or source_mesh.with_suffix(".mtl").exists():
        candidates.append(source_mesh)
    visual_mesh = source_mesh.with_name(f"{source_mesh.stem}_visual.glb")
    if visual_mesh.exists():
        candidates.append(visual_mesh)
    if not candidates:
        return None
    selected_mesh = candidates[0]
    scene_or_mesh = trimesh.load(selected_mesh, force="scene")
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    vertex_offset = 0
    if isinstance(scene_or_mesh, trimesh.Scene):
        for node_name in scene_or_mesh.graph.nodes_geometry:
            transform, geometry_name = scene_or_mesh.graph[node_name]
            geom = scene_or_mesh.geometry[geometry_name]
            vertices = np.asarray(geom.vertices, dtype=np.float64)
            homog = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
            vertices = (homog @ np.asarray(transform, dtype=np.float64).T)[:, :3]
            faces = np.asarray(geom.faces, dtype=np.int64)
            vertices_all.append(vertices)
            faces_all.append(faces + vertex_offset)
            colors_all.append(_face_colors_from_visual(geom.visual, faces, source_mesh))
            vertex_offset += len(vertices)
    else:
        vertices_all.append(np.asarray(scene_or_mesh.vertices, dtype=np.float64))
        faces = np.asarray(scene_or_mesh.faces, dtype=np.int64)
        faces_all.append(faces)
        colors_all.append(_face_colors_from_visual(scene_or_mesh.visual, faces, source_mesh))
    if not vertices_all or not faces_all:
        return None
    vertices = np.vstack(vertices_all)
    faces = np.vstack(faces_all)
    colors = np.vstack(colors_all)
    source = load_obj_mesh(source_mesh)
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_mins = source_vertices.min(axis=0)
    source_maxs = source_vertices.max(axis=0)
    visual_mins = vertices.min(axis=0)
    visual_maxs = vertices.max(axis=0)
    source_extent = np.maximum(source_maxs - source_mins, 1e-9)
    visual_extent = np.maximum(visual_maxs - visual_mins, 1e-9)
    scale = float(np.max(source_extent) / np.max(visual_extent))
    source_center = (source_mins + source_maxs) * 0.5
    visual_center = (visual_mins + visual_maxs) * 0.5
    vertices = (vertices - visual_center) * scale + source_center
    return vertices, faces, colors.astype(np.uint8), str(selected_mesh)


def _render_software_mesh_views(
    *,
    source_mesh: Path,
    render_dir: Path,
    meta: dict[str, Any],
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from ToGo.providers.tool_prior.sampart3d_mesh_candidates import load_obj_mesh
    from ToGo.providers.tool_prior.sampart3d_projection_geometry import SAMPart3D_RENDER_ROTATION

    material_mesh = _load_material_visual_mesh(source_mesh)
    if material_mesh is None:
        mesh = load_obj_mesh(source_mesh)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        face_colors = np.full((len(faces), 3), 178, dtype=np.uint8)
        material_source = "source_obj_gray"
    else:
        vertices, faces, face_colors, material_source = material_mesh
    vertices_render = vertices @ SAMPart3D_RENDER_ROTATION
    vertices_render = vertices_render * float(meta["scaling_factor"]) + np.asarray(
        meta["mesh_offset"],
        dtype=np.float64,
    )
    image_width = int(meta.get("width") or 512)
    image_height = int(meta.get("height") or 512)
    camera_angle_x = float(meta["camera_angle_x"])
    light = _unit(np.asarray([0.2, -0.35, 1.0], dtype=np.float64))
    render_dir.mkdir(parents=True, exist_ok=True)
    for view_index, transform in enumerate(meta["transforms"]):
        front, pixels, cam = _project_vertices(
            vertices=vertices_render,
            transform=transform,
            camera_angle_x=camera_angle_x,
            image_width=image_width,
            image_height=image_height,
        )
        face_depths: list[tuple[float, int]] = []
        for face_index, face in enumerate(faces):
            if not any(bool(front[int(index)]) for index in face):
                continue
            depth = float(np.mean(-cam[face, 2]))
            if depth <= 0.0 or not math.isfinite(depth):
                continue
            face_depths.append((depth, face_index))
        image = Image.new("RGB", (image_width, image_height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for _, face_index in sorted(face_depths, reverse=True):
            face = faces[face_index]
            pts = vertices_render[face]
            normal = _unit(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            shade = 0.58 + 0.34 * max(0.0, float(np.dot(normal, light)))
            rgb = np.clip(np.asarray(face_colors[face_index], dtype=np.float64) * shade, 18, 242)
            color = tuple(int(value) for value in rgb.astype(np.uint8))
            polygon = [(float(pixels[int(index), 0]), float(pixels[int(index), 1])) for index in face]
            draw.polygon(polygon, fill=color, outline=(36, 36, 36))
        image.save(render_dir / f"render_{view_index:04d}.webp", quality=92)
    meta["material_render_source"] = material_source


def ensure_render_cache(tool_root: Path, prior_subdir: str) -> None:
    """Create SAMPart3D-style render cache for freshly imported external tools."""

    import numpy as np

    from ToGo.providers.tool_prior.sampart3d_mesh_candidates import load_obj_mesh
    from ToGo.providers.tool_prior.sampart3d_projection_geometry import SAMPart3D_RENDER_ROTATION

    prior_dir = tool_root / prior_subdir
    candidates_path = prior_dir / "grounding_candidates.json"
    if not candidates_path.exists():
        return
    grounding_candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    source_mesh_raw = (grounding_candidates.get("metadata") or {}).get("source_mesh")
    if not source_mesh_raw:
        return
    source_mesh = Path(str(source_mesh_raw))
    if not source_mesh.is_absolute():
        source_mesh = ROOT / source_mesh
    if not source_mesh.exists():
        return
    render_dir = tool_root / "data_root" / tool_root.name
    meta_path = render_dir / "meta.json"
    existing_renders = sorted(render_dir.glob("render_*.webp")) if render_dir.exists() else []
    if meta_path.exists() and len(existing_renders) >= 16:
        try:
            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}
        native_material_source = (
            source_mesh
            if (source_mesh.parent / "material.mtl").exists() or source_mesh.with_suffix(".mtl").exists()
            else source_mesh.with_name(f"{source_mesh.stem}_visual.glb")
        )
        if not native_material_source.exists() or existing_meta.get("material_render_source") == str(native_material_source):
            return

    mesh = load_obj_mesh(source_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ SAMPart3D_RENDER_ROTATION
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    scaling_factor = 1.0 / float(np.max(extent))
    center = (mins + maxs) * 0.5
    mesh_offset = (-center * scaling_factor).tolist()
    bbox_size = (extent * scaling_factor).tolist()
    radius = 1.65
    elevation = math.radians(25.0)
    horizontal = radius * math.cos(elevation)
    transforms: list[list[list[float]]] = []
    for index in range(16):
        azimuth = 2.0 * math.pi * float(index) / 16.0
        eye = (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            radius * math.sin(elevation),
        )
        transforms.append(_look_at_transform(eye))
    meta = {
        "camera_angle_x": 0.6911111611634243,
        "camera_lens": 50,
        "sensor_width": 36,
        "env_texture": "null",
        "bbox_size": bbox_size,
        "scaling_factor": scaling_factor,
        "mesh_offset": mesh_offset,
        "transforms": transforms,
        "width": 512,
        "height": 512,
        "generated_by": "ToGo.providers.tool_prior.render_cache",
    }
    render_dir.mkdir(parents=True, exist_ok=True)
    _render_software_mesh_views(source_mesh=source_mesh, render_dir=render_dir, meta=meta)
    _write_json(meta_path, meta)
    print(f"[tool-ref-pipeline] generated render cache: {render_dir}", flush=True)
