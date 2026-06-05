#!/usr/bin/env python3
"""Evaluate side-board HDU eraser success from functional contact coverage."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "parttoolgen").is_dir() and (parent / "simtoolreal").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [float(item) for item in value[:3]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def _unit(v: Sequence[float], name: str) -> list[float]:
    n = _norm(v)
    if n <= 1e-12:
        raise ValueError(f"{name} is near zero")
    return [float(item) / n for item in v[:3]]


def _sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _mul(v: Sequence[float], s: float) -> list[float]:
    return [float(v[i]) * float(s) for i in range(3)]


def _quat_to_matrix(q_xyzw: Sequence[float]) -> list[list[float]]:
    x, y, z, w = [float(item) for item in q_xyzw[:4]]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
    ]


def _rotate_vector(pose_xyzw: Sequence[float], vector_tool: Sequence[float]) -> list[float]:
    rot = _quat_to_matrix(pose_xyzw[3:7])
    x, y, z = [float(item) for item in vector_tool[:3]]
    return [
        rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
        rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
        rot[2][0] * x + rot[2][1] * y + rot[2][2] * z,
    ]


def _resolve_asset_path(asset_file: str | Path, *, repo_root: Path) -> Path:
    path = Path(asset_file)
    if path.is_absolute():
        return path
    if str(path).startswith("urdf/"):
        return repo_root / "simtoolreal/assets" / path
    return repo_root / path


def _load_obj_vertices_faces(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            face: list[int] = []
            for item in line.split()[1:]:
                token = item.split("/")[0]
                if token:
                    index = int(token)
                    face.append(index - 1 if index > 0 else len(vertices) + index)
            if len(face) >= 3:
                faces.append(face)
    return vertices, faces


def _resolve_visual_mesh_from_urdf(asset_path: Path) -> Path | None:
    if asset_path.suffix.lower() != ".urdf":
        return asset_path
    fallback = asset_path.with_suffix(".obj")
    try:
        root = ET.fromstring(asset_path.read_text(encoding="utf-8", errors="ignore"))
    except ET.ParseError:
        return fallback if fallback.exists() else None
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if mesh_path.is_absolute() and mesh_path.exists():
            return mesh_path
        local_path = asset_path.parent / mesh_path
        if local_path.exists():
            return local_path
    return fallback if fallback.exists() else None


def _selected_contact_row(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    def row_with_xyz(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        row = value.get("row")
        if isinstance(row, Mapping) and row.get("xyz_tool") is not None:
            return row
        if value.get("xyz_tool") is not None:
            return value
        return None

    direct = selection.get("selected_contact_row")
    row = row_with_xyz(direct)
    if row is not None:
        return row

    row = row_with_xyz(selection.get("contact_point"))
    if row is not None:
        return row

    active = selection.get("active_contact_selection") or selection.get(
        "active_contact_point_candidate_selection"
    )
    active_ref = None
    if isinstance(active, Mapping):
        active_ref = (
            active.get("active_contact_point_source_ref")
            or active.get("selected_contact_point_source_ref")
            or active.get("point_ref")
        )
    active_ref = active_ref or selection.get("selected_contact_point_source_ref") or selection.get(
        "selected_contact_point_ref"
    )

    for list_key in ("candidate_contact_rows", "contact_point_candidates"):
        rows = selection.get(list_key)
        if not isinstance(rows, list):
            continue
        for item in rows:
            row = row_with_xyz(item)
            if row is None:
                continue
            point_ref = row.get("point_ref") or (item.get("point_ref") if isinstance(item, Mapping) else None)
            if active_ref and point_ref == active_ref:
                return row
        for item in rows:
            row = row_with_xyz(item)
            if row is not None:
                return row
    raise ValueError("could not find selected flat-eraser contact row")


def _task_hdu_geometry(task: Mapping[str, Any]) -> dict[str, Any]:
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("task.metadata is required")
    center = _vec3(
        metadata.get("hdu_sticker_center_world") or metadata.get("hdu_ink_center_world"),
        "metadata.hdu_sticker_center_world",
    )
    size_yz_value = metadata.get("hdu_sticker_size_yz") or metadata.get("hdu_ink_size_yz")
    if not isinstance(size_yz_value, Sequence) or isinstance(size_yz_value, (str, bytes)) or len(size_yz_value) < 2:
        raise ValueError("metadata.hdu_sticker_size_yz is required")
    asset_file = metadata.get("hdu_sticker_asset_file") or metadata.get("hdu_ink_asset_file")
    if not isinstance(asset_file, str):
        raise ValueError("metadata.hdu_sticker_asset_file is required")
    return {
        "center_world": center,
        "size_yz": [float(size_yz_value[0]), float(size_yz_value[1])],
        "stroke_width_m": float(metadata.get("hdu_sticker_stroke_width_m") or metadata.get("hdu_ink_stroke_width_m") or 0.0),
        "asset_file": asset_file,
        "surface_normal_world": _vec3(metadata.get("public_whiteboard_surface_normal_world"), "metadata.public_whiteboard_surface_normal_world"),
    }


def _pixel_mapper(size_yz: Sequence[float], mask_width: int) -> tuple[int, int, Any]:
    size_y = float(size_yz[0])
    size_z = float(size_yz[1])
    width = int(mask_width)
    height = max(1, int(round(width * size_z / size_y)))
    half_y = 0.5 * size_y
    half_z = 0.5 * size_z

    def to_px(local_y: float, local_z: float) -> tuple[float, float]:
        x = (float(local_y) + half_y) / size_y * float(width - 1)
        y = (half_z - float(local_z)) / size_z * float(height - 1)
        return x, y

    return width, height, to_px


def _hdu_mask_from_asset(
    *,
    asset_path: Path,
    size_yz: Sequence[float],
    mask_width: int,
) -> Image.Image:
    mesh_path = _resolve_visual_mesh_from_urdf(asset_path)
    if mesh_path is None:
        raise ValueError(f"could not resolve HDU visual mesh from asset: {asset_path}")
    vertices, faces = _load_obj_vertices_faces(mesh_path)
    width, height, to_px = _pixel_mapper(size_yz, mask_width)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if vertices and faces:
        for face in faces:
            polygon = [to_px(vertices[index][1], vertices[index][2]) for index in face if 0 <= index < len(vertices)]
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=255)
    else:
        raise ValueError(f"HDU visual mesh has no drawable OBJ faces: {mesh_path}")
    return mask


def _tool_contact_footprint(
    *,
    tool_mesh_path: Path,
    tool_ref_selection: Mapping[str, Any],
    contact_face_band_m: float,
) -> dict[str, Any]:
    vertices, _ = _load_obj_vertices_faces(tool_mesh_path)
    if not vertices:
        raise ValueError(f"tool mesh has no vertices: {tool_mesh_path}")
    row = _selected_contact_row(tool_ref_selection)
    tip_tool = _vec3(row.get("xyz_tool"), "selected_contact_row.xyz_tool")
    normal_tool = _unit(_vec3(row.get("normal_tool"), "selected_contact_row.normal_tool"), "selected_contact_row.normal_tool")
    motion_row = tool_ref_selection.get("selected_motion_axis_row")
    if not isinstance(motion_row, Mapping):
        motion_axis = tool_ref_selection.get("motion_axis")
        if isinstance(motion_axis, Mapping) and isinstance(motion_axis.get("row"), Mapping):
            motion_row = motion_axis.get("row")
    preferred = None
    if isinstance(motion_row, Mapping) and motion_row.get("xyz_tool") is not None:
        preferred = _sub(_vec3(motion_row.get("xyz_tool"), "selected_motion_axis_row.xyz_tool"), tip_tool)
    if preferred is None or _norm(preferred) <= 1e-8:
        preferred = [1.0, 0.0, 0.0]
    preferred = _sub(preferred, _mul(normal_tool, _dot(preferred, normal_tool)))
    if _norm(preferred) <= 1e-8:
        preferred = [0.0, 0.0, 1.0]
        preferred = _sub(preferred, _mul(normal_tool, _dot(preferred, normal_tool)))
    tangent_u = _unit(preferred, "contact_face_tangent_u")
    tangent_v = _unit(_cross(normal_tool, tangent_u), "contact_face_tangent_v")
    near = [
        vertex
        for vertex in vertices
        if abs(_dot(_sub(vertex, tip_tool), normal_tool)) <= float(contact_face_band_m)
    ]
    if len(near) < 4:
        near = vertices
    u_values = [_dot(_sub(vertex, tip_tool), tangent_u) for vertex in near]
    v_values = [_dot(_sub(vertex, tip_tool), tangent_v) for vertex in near]
    u_min, u_max = min(u_values), max(u_values)
    v_min, v_max = min(v_values), max(v_values)
    corners = [
        _add(tip_tool, _add(_mul(tangent_u, u_min), _mul(tangent_v, v_min))),
        _add(tip_tool, _add(_mul(tangent_u, u_max), _mul(tangent_v, v_min))),
        _add(tip_tool, _add(_mul(tangent_u, u_max), _mul(tangent_v, v_max))),
        _add(tip_tool, _add(_mul(tangent_u, u_min), _mul(tangent_v, v_max))),
    ]
    return {
        "selected_contact_row": {
            "part_marker": row.get("part_marker"),
            "part_ref": row.get("part_ref"),
            "face_marker": row.get("face_marker") or row.get("contact_marker"),
            "point_ref": row.get("point_ref"),
            "surface_ref": row.get("surface_ref"),
            "xyz_tool": tip_tool,
            "normal_tool": normal_tool,
        },
        "tangent_u_tool": tangent_u,
        "tangent_v_tool": tangent_v,
        "u_range_m": [u_min, u_max],
        "v_range_m": [v_min, v_max],
        "corner_offsets_from_tip_tool": [_sub(corner, tip_tool) for corner in corners],
        "contact_face_band_m": float(contact_face_band_m),
        "near_vertex_count": len(near),
        "tool_mesh_vertex_count": len(vertices),
    }


def _world_to_hdu_local_yz(point_world: Sequence[float], center_world: Sequence[float]) -> tuple[float, float]:
    return float(point_world[1]) - float(center_world[1]), float(point_world[2]) - float(center_world[2])


def _count_on(mask: Image.Image) -> int:
    return sum(1 for value in mask.getdata() if int(value) > 0)


def evaluate_run(
    *,
    task_path: Path,
    tracking_samples_path: Path,
    tool_ref_selection_path: Path,
    tool_mesh_path: Path,
    output_dir: Path,
    min_erased_fraction: float,
    max_surface_distance_m: float,
    min_surface_normal_opposition_cos: float,
    contact_face_band_m: float,
    mask_width: int,
) -> dict[str, Any]:
    task = _load_json(task_path)
    if not isinstance(task, Mapping):
        raise ValueError(f"task must be a JSON object: {task_path}")
    hdu = _task_hdu_geometry(task)
    repo = _repo_root()
    hdu_asset_path = _resolve_asset_path(hdu["asset_file"], repo_root=repo)
    hdu_mask = _hdu_mask_from_asset(
        asset_path=hdu_asset_path,
        size_yz=hdu["size_yz"],
        mask_width=int(mask_width),
    )
    width, height = hdu_mask.size
    _, _, to_px = _pixel_mapper(hdu["size_yz"], width)
    coverage_mask = Image.new("L", (width, height), 0)
    coverage_draw = ImageDraw.Draw(coverage_mask)
    selection = _load_json(tool_ref_selection_path)
    if not isinstance(selection, Mapping):
        raise ValueError(f"tool-ref selection must be a JSON object: {tool_ref_selection_path}")
    footprint = _tool_contact_footprint(
        tool_mesh_path=tool_mesh_path,
        tool_ref_selection=selection,
        contact_face_band_m=float(contact_face_band_m),
    )
    tracking_samples = _load_json(tracking_samples_path)
    if not isinstance(tracking_samples, list):
        raise ValueError(f"tracking samples must be a JSON list: {tracking_samples_path}")
    center = [float(item) for item in hdu["center_world"]]
    surface_normal = _unit(hdu["surface_normal_world"], "surface_normal_world")
    accepted_samples: list[dict[str, Any]] = []
    rejected_near_plane = 0
    rejected_normal = 0
    for sample in tracking_samples:
        if not isinstance(sample, Mapping):
            continue
        pose = sample.get("actual_pose_xyzw")
        actual_contact = sample.get("actual_contact_point_world")
        sample_tip = sample.get("contact_point_tool")
        if not (
            isinstance(pose, Sequence)
            and not isinstance(pose, (str, bytes))
            and len(pose) >= 7
            and isinstance(actual_contact, Sequence)
            and not isinstance(actual_contact, (str, bytes))
            and len(actual_contact) >= 3
        ):
            continue
        if isinstance(sample_tip, Sequence) and not isinstance(sample_tip, (str, bytes)) and len(sample_tip) >= 3:
            tip_delta = math.sqrt(
                sum(
                    (float(sample_tip[i]) - footprint["selected_contact_row"]["xyz_tool"][i]) ** 2
                    for i in range(3)
                )
            )
            if tip_delta > 1e-4:
                continue
        contact_world = [float(item) for item in actual_contact[:3]]
        signed_distance = _dot(_sub(contact_world, center), surface_normal)
        if abs(signed_distance) > float(max_surface_distance_m):
            rejected_near_plane += 1
            continue
        normal_world = _unit(
            _rotate_vector([float(item) for item in pose[:7]], footprint["selected_contact_row"]["normal_tool"]),
            "surface_normal_world",
        )
        normal_opposition_cos = -_dot(normal_world, surface_normal)
        if normal_opposition_cos < float(min_surface_normal_opposition_cos):
            rejected_normal += 1
            continue
        polygon_px = []
        polygon_world = []
        for offset_tool in footprint["corner_offsets_from_tip_tool"]:
            offset_world = _rotate_vector([float(item) for item in pose[:7]], offset_tool)
            corner_world = _add(contact_world, offset_world)
            local_y, local_z = _world_to_hdu_local_yz(corner_world, center)
            polygon_px.append(to_px(local_y, local_z))
            polygon_world.append(corner_world)
        coverage_draw.polygon(polygon_px, fill=255)
        accepted_samples.append(
            {
                "step": sample.get("step"),
                "goal_index": sample.get("goal_index"),
                "stage": sample.get("stage"),
                "signed_surface_distance_m": signed_distance,
                "surface_normal_opposition_cos": normal_opposition_cos,
                "contact_point_world": contact_world,
                "footprint_corners_world": polygon_world,
            }
        )

    hdu_on = hdu_mask.point(lambda value: 255 if value > 0 else 0)
    erased_mask = Image.new("L", (width, height), 0)
    hdu_data = list(hdu_on.getdata())
    coverage_data = list(coverage_mask.getdata())
    erased_mask.putdata([255 if h and c else 0 for h, c in zip(hdu_data, coverage_data)])
    hdu_area_px = _count_on(hdu_on)
    erased_area_px = _count_on(erased_mask)
    erased_fraction = erased_area_px / hdu_area_px if hdu_area_px else 0.0
    passed = erased_fraction >= float(min_erased_fraction)
    overlay = Image.new("RGB", (width, height), (245, 248, 245))
    overlay_pixels = []
    erased_data = list(erased_mask.getdata())
    for hdu_value, coverage_value, erased_value in zip(hdu_data, coverage_data, erased_data):
        if erased_value:
            overlay_pixels.append((34, 180, 82))
        elif hdu_value:
            overlay_pixels.append((10, 45, 235))
        elif coverage_value:
            overlay_pixels.append((255, 185, 85))
        else:
            overlay_pixels.append((245, 248, 245))
    overlay.putdata(overlay_pixels)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdu_mask_path = output_dir / "hdu_pattern_mask.png"
    coverage_mask_path = output_dir / "eraser_contact_coverage_mask.png"
    erased_mask_path = output_dir / "erased_hdu_mask.png"
    overlay_path = output_dir / "erased_area_overlay.png"
    hdu_on.save(hdu_mask_path)
    coverage_mask.save(coverage_mask_path)
    erased_mask.save(erased_mask_path)
    overlay.save(overlay_path)
    metrics = {
        "schema_version": "toolframe_side_board_hdu_erase_success_v1",
        "task_success_name": "side_whiteboard_hdu_sticker_erase_success",
        "passed": bool(passed),
        "failure_reason": None if passed else "erased_area_fraction_below_threshold",
        "erased_area_fraction": erased_fraction,
        "erased_area_px": erased_area_px,
        "initial_hdu_area_px": hdu_area_px,
        "thresholds": {
            "min_erased_fraction": float(min_erased_fraction),
            "max_surface_distance_m": float(max_surface_distance_m),
            "min_surface_normal_opposition_cos": float(min_surface_normal_opposition_cos),
            "contact_face_band_m": float(contact_face_band_m),
        },
        "policy": {
            "uses": "flat_eraser tool.contact_point and selected contact-face footprint from public tool refs",
            "does_not_use": [
                "whole-tool root pose alone",
                "sticker collision",
                "hidden evaluator refs",
                "oracle trajectory",
                "tool_family fallback",
                "task_family fallback",
            ],
        },
        "task_path": str(task_path),
        "tracking_samples_path": str(tracking_samples_path),
        "tool_ref_selection_path": str(tool_ref_selection_path),
        "tool_mesh_path": str(tool_mesh_path),
        "hdu": {
            **hdu,
            "asset_path": str(hdu_asset_path),
        },
        "footprint": footprint,
        "sample_count": len(tracking_samples),
        "accepted_contact_sample_count": len(accepted_samples),
        "rejected_surface_distance_sample_count": rejected_near_plane,
        "rejected_normal_sample_count": rejected_normal,
        "accepted_samples_preview": accepted_samples[:24],
        "outputs": {
            "hdu_pattern_mask": str(hdu_mask_path),
            "eraser_contact_coverage_mask": str(coverage_mask_path),
            "erased_hdu_mask": str(erased_mask_path),
            "erased_area_overlay": str(overlay_path),
            "metrics": str(output_dir / "erased_area_metrics.json"),
        },
    }
    _write_json(output_dir / "erased_area_metrics.json", metrics)
    return metrics


def _run_seed_paths(batch_root: Path, seed: int) -> tuple[Path, Path, Path]:
    sid = f"{seed:03d}"
    return (
        batch_root / "tasks" / f"seed_{sid}_task.json",
        batch_root / "runs" / f"seed_{sid}" / "rollout" / "tracking_samples.json",
        batch_root / "runs" / f"seed_{sid}" / "hdu_erase_success",
    )


def evaluate_batch(args: argparse.Namespace) -> dict[str, Any]:
    batch_root = Path(args.batch_root)
    rows = []
    for seed in range(int(args.seed_start), int(args.seed_start) + int(args.trials)):
        task_path, tracking_path, output_dir = _run_seed_paths(batch_root, seed)
        row = {"seed": seed, "task_path": str(task_path), "tracking_samples_path": str(tracking_path)}
        if not task_path.exists() or not tracking_path.exists():
            row.update({"evaluated": False, "passed": False, "failure_reason": "missing_task_or_tracking_samples"})
        else:
            metrics = evaluate_run(
                task_path=task_path,
                tracking_samples_path=tracking_path,
                tool_ref_selection_path=Path(args.tool_ref_selection),
                tool_mesh_path=Path(args.tool_mesh),
                output_dir=output_dir,
                min_erased_fraction=float(args.min_erased_fraction),
                max_surface_distance_m=float(args.max_surface_distance_m),
                min_surface_normal_opposition_cos=float(args.min_surface_normal_opposition_cos),
                contact_face_band_m=float(args.contact_face_band_m),
                mask_width=int(args.mask_width),
            )
            row.update(
                {
                    "evaluated": True,
                    "passed": bool(metrics.get("passed")),
                    "erased_area_fraction": metrics.get("erased_area_fraction"),
                    "accepted_contact_sample_count": metrics.get("accepted_contact_sample_count"),
                    "failure_reason": metrics.get("failure_reason"),
                    "metrics_path": metrics.get("outputs", {}).get("metrics"),
                    "overlay_path": metrics.get("outputs", {}).get("erased_area_overlay"),
                }
            )
        rows.append(row)
    evaluated = [row for row in rows if row.get("evaluated")]
    passed = [row for row in evaluated if row.get("passed")]
    batch_summary = {
        "schema_version": "toolframe_side_board_hdu_erase_batch_summary_v1",
        "batch_root": str(batch_root),
        "num_seeds": int(args.trials),
        "evaluated_count": len(evaluated),
        "passed_count": len(passed),
        "task_success_rate": len(passed) / len(evaluated) if evaluated else None,
        "min_erased_fraction": float(args.min_erased_fraction),
        "rows": rows,
    }
    _write_json(batch_root / "hdu_erase_success_summary.json", batch_summary)
    summary_path = batch_root / "summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        if isinstance(summary, Mapping):
            updated = dict(summary)
            updated["side_board_hdu_erase_success"] = batch_summary
            _write_json(summary_path, updated)
    return batch_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--tracking-samples", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--tool-ref-selection",
        type=Path,
        default=Path("configs/tool_refs/flat_eraser/contact_point_selection.json"),
    )
    parser.add_argument(
        "--tool-mesh",
        type=Path,
        default=Path("simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj"),
    )
    parser.add_argument("--min-erased-fraction", type=float, default=0.70)
    parser.add_argument("--max-surface-distance-m", type=float, default=0.020)
    parser.add_argument("--min-surface-normal-opposition-cos", type=float, default=0.0)
    parser.add_argument("--contact-face-band-m", type=float, default=0.006)
    parser.add_argument("--mask-width", type=int, default=720)
    args = parser.parse_args()
    if args.batch_root is not None:
        summary = evaluate_batch(args)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        return 0
    if args.task is None or args.tracking_samples is None or args.output_dir is None:
        parser.error("single-run mode requires --task, --tracking-samples, and --output-dir")
    metrics = evaluate_run(
        task_path=args.task,
        tracking_samples_path=args.tracking_samples,
        tool_ref_selection_path=args.tool_ref_selection,
        tool_mesh_path=args.tool_mesh,
        output_dir=args.output_dir,
        min_erased_fraction=float(args.min_erased_fraction),
        max_surface_distance_m=float(args.max_surface_distance_m),
        min_surface_normal_opposition_cos=float(args.min_surface_normal_opposition_cos),
        contact_face_band_m=float(args.contact_face_band_m),
        mask_width=int(args.mask_width),
    )
    print(
        json.dumps(
            {
                "passed": metrics["passed"],
                "erased_area_fraction": metrics["erased_area_fraction"],
                "metrics": metrics["outputs"]["metrics"],
                "overlay": metrics["outputs"]["erased_area_overlay"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
