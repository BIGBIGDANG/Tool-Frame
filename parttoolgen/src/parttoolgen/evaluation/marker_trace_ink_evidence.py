#!/usr/bin/env python3
"""Render marker-trace ink evidence from the reviewed marker-tip trajectory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


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
    return sum(float(x) * float(y) for x, y in zip(a[:3], b[:3]))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float], name: str) -> list[float]:
    n = _norm(a)
    if n <= 1e-12:
        raise ValueError(f"{name} is near zero")
    return [float(x) / n for x in a[:3]]


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


def _transform_point(pose_xyzw: Sequence[float], point_tool: Sequence[float]) -> list[float]:
    rot = _quat_to_matrix(pose_xyzw[3:7])
    px, py, pz = [float(item) for item in pose_xyzw[:3]]
    x, y, z = [float(item) for item in point_tool[:3]]
    return [
        px + rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
        py + rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
        pz + rot[2][0] * x + rot[2][1] * y + rot[2][2] * z,
    ]


def _transform_vector(pose_xyzw: Sequence[float], vector_tool: Sequence[float]) -> list[float]:
    rot = _quat_to_matrix(pose_xyzw[3:7])
    x, y, z = [float(item) for item in vector_tool[:3]]
    return [
        rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
        rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
        rot[2][0] * x + rot[2][1] * y + rot[2][2] * z,
    ]


def _extract_tip_row(selection: Mapping[str, Any]) -> Mapping[str, Any]:
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
    active_ref = active_ref or selection.get("selected_contact_point_ref") or selection.get(
        "selected_contact_point_source_ref"
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
    raise ValueError("could not find selected marker-tip row with xyz_tool in tool-ref selection")


def _label_review_status(path: Path | None) -> str | None:
    if path is None:
        return None
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"tool label review must be a JSON object: {path}")
    return str(data.get("manual_review_status") or "")


def _task_metadata_vec3(path: Path | None, key: str) -> list[float] | None:
    if path is None:
        return None
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"task must be a JSON object: {path}")
    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    return [float(item) for item in value[:3]]


def _tip_world_for_sample(sample: Mapping[str, Any], tip_tool: Sequence[float]) -> list[float] | None:
    world = sample.get("actual_contact_point_world")
    sample_tip = sample.get("contact_point_tool")
    if (
        isinstance(world, Sequence)
        and not isinstance(world, (str, bytes))
        and isinstance(sample_tip, Sequence)
        and not isinstance(sample_tip, (str, bytes))
        and len(world) >= 3
        and len(sample_tip) >= 3
    ):
        delta = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(sample_tip[:3], tip_tool[:3])))
        if delta <= 1e-5:
            return [float(item) for item in world[:3]]
    pose = sample.get("actual_pose_xyzw")
    if isinstance(pose, Sequence) and not isinstance(pose, (str, bytes)) and len(pose) >= 7:
        return _transform_point([float(item) for item in pose[:7]], tip_tool)
    return None


def _tip_normal_world_for_sample(
    sample: Mapping[str, Any],
    normal_tool: Sequence[float] | None,
) -> list[float] | None:
    if normal_tool is None:
        return None
    pose = sample.get("actual_pose_xyzw")
    if isinstance(pose, Sequence) and not isinstance(pose, (str, bytes)) and len(pose) >= 7:
        return _unit(_transform_vector([float(item) for item in pose[:7]], normal_tool), "tip_normal_world")
    return None


def _world_to_saved_uv(point_world: Sequence[float], camera: Mapping[str, Any]) -> tuple[float, float] | None:
    c2w = camera.get("camera_to_world")
    if not (isinstance(c2w, list) and len(c2w) >= 4):
        raise ValueError("camera.camera_to_world is required")
    right = _vec3([c2w[0][0], c2w[1][0], c2w[2][0]], "camera right")
    down = _vec3([c2w[0][1], c2w[1][1], c2w[2][1]], "camera down")
    forward = _vec3([c2w[0][2], c2w[1][2], c2w[2][2]], "camera forward")
    pos = _vec3([c2w[0][3], c2w[1][3], c2w[2][3]], "camera position")
    rel = [float(point_world[i]) - pos[i] for i in range(3)]
    x_cam = _dot(rel, right)
    y_cam = _dot(rel, down)
    z_cam = _dot(rel, forward)
    if z_cam <= 1e-9:
        return None
    u = float(camera["fx"]) * x_cam / z_cam + float(camera["cx"])
    v = float(camera["fy"]) * y_cam / z_cam + float(camera["cy"])
    width = int(camera["width"])
    height = int(camera["height"])
    transform = camera.get("image_transform")
    if isinstance(transform, Mapping) and int(transform.get("rotation_degrees") or 0) == 180:
        u = float(width - 1) - u
        v = float(height - 1) - v
    return u, v


def _contact_gate(
    *,
    tip_world: Sequence[float],
    tip_normal_world: Sequence[float] | None,
    surface_point_world: Sequence[float],
    surface_normal_world: Sequence[float],
    max_contact_gap_m: float,
    max_penetration_m: float,
    min_normal_opposition_cos: float,
) -> tuple[bool, dict[str, Any]]:
    normal = _unit(surface_normal_world, "surface_normal_world")
    signed_gap = _dot([float(tip_world[i]) - float(surface_point_world[i]) for i in range(3)], normal)
    height_ok = -float(max_penetration_m) <= signed_gap <= float(max_contact_gap_m)
    normal_cos = None
    normal_ok = True
    if tip_normal_world is not None and min_normal_opposition_cos > 0.0:
        normal_cos = _dot(_unit(tip_normal_world, "tip_normal_world"), normal)
        normal_ok = normal_cos <= -float(min_normal_opposition_cos)
    return height_ok and normal_ok, {
        "signed_gap_m": signed_gap,
        "height_gate_passed": height_ok,
        "tip_normal_dot_surface_normal": normal_cos,
        "normal_gate_passed": normal_ok,
    }


def render_ink_evidence(
    *,
    base_rgb_path: Path,
    camera_path: Path,
    tracking_samples_path: Path,
    tool_ref_selection_path: Path,
    output_png_path: Path,
    output_metrics_path: Path,
    tool_label_review_path: Path | None,
    allow_pending_tool_labels: bool,
    surface_point_world: Sequence[float],
    surface_normal_world: Sequence[float],
    max_contact_gap_m: float,
    max_penetration_m: float,
    min_normal_opposition_cos: float,
    ink_width_px: int,
    ink_rgb: Sequence[int],
) -> dict[str, Any]:
    review_status = _label_review_status(tool_label_review_path)
    if review_status is not None and review_status.lower() != "approved" and not allow_pending_tool_labels:
        raise ValueError(
            "tool label review is not approved; pass --allow-pending-tool-labels for debug-only rendering"
        )

    selection = _load_json(tool_ref_selection_path)
    if not isinstance(selection, Mapping):
        raise ValueError("tool-ref selection must be a JSON object")
    tip_row = _extract_tip_row(selection)
    tip_tool = _vec3(tip_row.get("xyz_tool"), "selected_tip.xyz_tool")
    normal_tool = tip_row.get("normal_tool")
    tip_normal_tool = _unit(normal_tool, "selected_tip.normal_tool") if normal_tool is not None else None

    camera = _load_json(camera_path)
    tracking_samples = _load_json(tracking_samples_path)
    if not isinstance(tracking_samples, list):
        raise ValueError("tracking samples must be a JSON list")

    base = Image.open(base_rgb_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    ink_color = tuple(int(item) for item in ink_rgb[:3])
    contact_points: list[dict[str, Any]] = []
    previous_uv: tuple[float, float] | None = None
    previous_step: int | None = None
    ink_segments = 0
    ink_length_m = 0.0
    previous_world: list[float] | None = None
    gate_rows = []

    for sample in tracking_samples:
        if not isinstance(sample, Mapping):
            continue
        tip_world = _tip_world_for_sample(sample, tip_tool)
        if tip_world is None:
            continue
        tip_normal_world = _tip_normal_world_for_sample(sample, tip_normal_tool)
        passed, gate = _contact_gate(
            tip_world=tip_world,
            tip_normal_world=tip_normal_world,
            surface_point_world=surface_point_world,
            surface_normal_world=surface_normal_world,
            max_contact_gap_m=max_contact_gap_m,
            max_penetration_m=max_penetration_m,
            min_normal_opposition_cos=min_normal_opposition_cos,
        )
        step = int(sample.get("step", len(gate_rows)))
        gate_rows.append({"step": step, "tip_world": tip_world, "contact_passed": passed, **gate})
        if not passed:
            previous_uv = None
            previous_step = None
            previous_world = None
            continue
        uv = _world_to_saved_uv(tip_world, camera)
        if uv is None:
            previous_uv = None
            previous_step = None
            previous_world = None
            continue
        contact_points.append({"step": step, "tip_world": tip_world, "uv": [uv[0], uv[1]], **gate})
        if previous_uv is not None and previous_step is not None and step == previous_step + 1:
            draw.line([previous_uv, uv], fill=ink_color, width=int(ink_width_px))
            ink_segments += 1
            if previous_world is not None:
                ink_length_m += math.sqrt(sum((tip_world[i] - previous_world[i]) ** 2 for i in range(3)))
        previous_uv = uv
        previous_step = step
        previous_world = tip_world

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(output_png_path)
    metrics = {
        "schema_version": "toolframe_marker_trace_ink_evidence_v1",
        "policy": "ink is rendered only from the selected marker-tip/tool contact point ref while the contact gate passes",
        "base_rgb_path": str(base_rgb_path),
        "camera_path": str(camera_path),
        "tracking_samples_path": str(tracking_samples_path),
        "tool_ref_selection_path": str(tool_ref_selection_path),
        "tool_label_review_path": str(tool_label_review_path) if tool_label_review_path else None,
        "tool_label_review_status": review_status,
        "allow_pending_tool_labels": bool(allow_pending_tool_labels),
        "selected_tip": {
            "part_marker": tip_row.get("part_marker"),
            "part_ref": tip_row.get("part_ref"),
            "face_marker": tip_row.get("face_marker") or tip_row.get("contact_marker"),
            "point_ref": tip_row.get("point_ref"),
            "surface_ref": tip_row.get("surface_ref"),
            "xyz_tool": tip_tool,
            "normal_tool": tip_normal_tool,
        },
        "contact_gate": {
            "surface_point_world": [float(item) for item in surface_point_world[:3]],
            "surface_normal_world": _unit(surface_normal_world, "surface_normal_world"),
            "max_contact_gap_m": float(max_contact_gap_m),
            "max_penetration_m": float(max_penetration_m),
            "min_normal_opposition_cos": float(min_normal_opposition_cos),
        },
        "sample_count": len(tracking_samples),
        "evaluated_sample_count": len(gate_rows),
        "contact_sample_count": len(contact_points),
        "ink_segment_count": ink_segments,
        "ink_length_m": ink_length_m,
        "output_png_path": str(output_png_path),
        "contact_points": contact_points,
    }
    _write_json(output_metrics_path, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rgb", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--tracking-samples", required=True, type=Path)
    parser.add_argument("--tool-ref-selection", required=True, type=Path)
    parser.add_argument("--tool-label-review", type=Path)
    parser.add_argument("--allow-pending-tool-labels", action="store_true")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--surface-point-world", nargs=3, type=float)
    parser.add_argument("--surface-normal-world", nargs=3, type=float)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.003)
    parser.add_argument("--max-penetration-m", type=float, default=0.003)
    parser.add_argument("--min-normal-opposition-cos", type=float, default=0.0)
    parser.add_argument("--ink-width-px", type=int, default=5)
    parser.add_argument("--ink-rgb", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-metrics", required=True, type=Path)
    args = parser.parse_args()
    surface_point_world = (
        args.surface_point_world
        or _task_metadata_vec3(args.task, "public_support_surface_point_world")
        or _task_metadata_vec3(args.task, "whiteboard_front_surface_point_world")
        or _task_metadata_vec3(args.task, "whiteboard_center_world")
    )
    surface_normal_world = args.surface_normal_world or _task_metadata_vec3(
        args.task, "public_whiteboard_surface_normal_world"
    )
    if surface_point_world is None:
        raise ValueError(
            "--surface-point-world is required unless --task metadata.whiteboard_center_world is available"
        )
    if surface_normal_world is None:
        raise ValueError(
            "--surface-normal-world is required unless --task metadata.public_whiteboard_surface_normal_world is available"
        )
    metrics = render_ink_evidence(
        base_rgb_path=args.base_rgb,
        camera_path=args.camera,
        tracking_samples_path=args.tracking_samples,
        tool_ref_selection_path=args.tool_ref_selection,
        output_png_path=args.output_png,
        output_metrics_path=args.output_metrics,
        tool_label_review_path=args.tool_label_review,
        allow_pending_tool_labels=bool(args.allow_pending_tool_labels),
        surface_point_world=surface_point_world,
        surface_normal_world=surface_normal_world,
        max_contact_gap_m=float(args.max_contact_gap_m),
        max_penetration_m=float(args.max_penetration_m),
        min_normal_opposition_cos=float(args.min_normal_opposition_cos),
        ink_width_px=int(args.ink_width_px),
        ink_rgb=args.ink_rgb,
    )
    print(json.dumps({"output_png": metrics["output_png_path"], "ink_segment_count": metrics["ink_segment_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
