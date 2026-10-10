#!/usr/bin/env python3
"""SimToolReal fixed-goal safety checks.

These helpers inspect fixed-goal artifacts against table geometry. They do not
modify generated goals; failed checks are repair evidence for ProgramIR,
compiler, or path-planner changes upstream.
"""

from __future__ import annotations
import json

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_MIN_ROOT_TABLE_CLEARANCE_M = 0.01
DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M = 0.003


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _simtoolreal_root() -> Path:
    return _repo_root() / "simtoolreal"


def _resolve_simtoolreal_asset_path(asset_path: str) -> Path:
    path = Path(asset_path)
    if path.is_absolute():
        return path
    return _simtoolreal_root() / "assets" / path


def resolve_simtoolreal_tool_mesh_path(
    tool_id: str | None = None,
    *,
    explicit_path: str | Path | None = None,
) -> Path | None:
    """Resolve the SimToolReal OBJ mesh that defines the tool asset-root frame."""

    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = _repo_root() / path
        return path if path.exists() else None
    if not tool_id:
        return None
    matches = sorted(
        (_simtoolreal_root() / "assets" / "urdf" / "dextoolbench").rglob(
            f"{tool_id}.obj"
        )
    )
    if matches:
        exact_parent_matches = [path for path in matches if path.parent.name == tool_id]
        return exact_parent_matches[0] if exact_parent_matches else matches[0]
    return None


def load_obj_vertices(mesh_path: str | Path) -> list[tuple[float, float, float]]:
    """Load OBJ vertex positions without pulling in a rendering dependency."""

    vertices: list[tuple[float, float, float]] = []
    for line in Path(mesh_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        except ValueError:
            continue
    return vertices


def _quat_xyzw_to_matrix(q: list[float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = [float(item) for item in q]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w),
        (2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w),
        (2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y),
    )


def tool_mesh_z_bounds_for_pose(
    pose_xyzw: list[float],
    vertices_tool: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """Return world z bounds of the whole tool mesh under an asset-root pose."""

    if not vertices_tool:
        raise ValueError("tool mesh has no OBJ vertices")
    row_z = _quat_xyzw_to_matrix(pose_xyzw[3:7])[2]
    root_z = float(pose_xyzw[2])
    min_z = float("inf")
    max_z = float("-inf")
    for x, y, z in vertices_tool:
        world_z = root_z + row_z[0] * x + row_z[1] * y + row_z[2] * z
        min_z = min(min_z, world_z)
        max_z = max(max_z, world_z)
    return min_z, max_z


def _pose_object_to_xyzw(pose: dict[str, Any]) -> list[float]:
    return [
        *[float(item) for item in pose["position"][:3]],
        *[float(item) for item in pose["orientation_xyzw"][:4]],
    ]


def _pose_xyzw_to_object(pose_xyzw: list[float]) -> dict[str, list[float]]:
    return {
        "position": [float(item) for item in pose_xyzw[:3]],
        "orientation_xyzw": [float(item) for item in pose_xyzw[3:7]],
    }


def table_contact_start_pose_from_state_bundle(
    *,
    start_pose: dict[str, Any],
    state_bundle: dict[str, Any],
    tool_id: str | None = None,
    tool_mesh_path: str | Path | None = None,
    min_tool_table_clearance_m: float = DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
) -> dict[str, Any]:
    """Project a tool start pose onto the SimToolReal tabletop using mesh geometry.

    This is not simulator oracle state. It uses the table asset pose and the tool
    mesh's asset-root frame to choose a start root z whose lowest mesh vertex sits
    just above the tabletop. The xy pose and orientation stay controlled by the
    provided start pose.
    """

    table_top_z = table_top_z_from_state_bundle(state_bundle)
    if table_top_z is None:
        raise ValueError("state bundle does not resolve a table top z")
    resolved_mesh = resolve_simtoolreal_tool_mesh_path(
        tool_id,
        explicit_path=tool_mesh_path,
    )
    if resolved_mesh is None:
        raise ValueError("tool mesh could not be resolved for table-contact start pose")
    vertices = load_obj_vertices(resolved_mesh)
    pose_xyzw = _pose_object_to_xyzw(start_pose)
    mesh_min_z, _ = tool_mesh_z_bounds_for_pose(pose_xyzw, vertices)
    target_min_z = float(table_top_z) + float(min_tool_table_clearance_m)
    pose_xyzw[2] = float(pose_xyzw[2]) + target_min_z - mesh_min_z
    return _pose_xyzw_to_object(pose_xyzw)


def mesh_table_clearance_report(
    fixed_goals: dict[str, Any],
    *,
    table_top_z: float,
    tool_id: str | None = None,
    tool_mesh_path: str | Path | None = None,
    min_tool_table_clearance_m: float = DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
    start_min_tool_table_clearance_m: float = 0.0,
) -> dict[str, Any]:
    """Check whole-tool geometry, not only asset-root z, against the tabletop."""

    resolved_mesh = resolve_simtoolreal_tool_mesh_path(
        tool_id,
        explicit_path=tool_mesh_path,
    )
    report: dict[str, Any] = {
        "checked": False,
        "tool_id": tool_id,
        "tool_mesh_path": str(resolved_mesh) if resolved_mesh else None,
        "table_top_z": float(table_top_z),
        "min_tool_table_clearance_m": float(min_tool_table_clearance_m),
        "start_min_tool_table_clearance_m": float(start_min_tool_table_clearance_m),
        "violations": [],
        "pose_clearances": [],
    }
    if resolved_mesh is None:
        report["reason"] = "tool mesh not resolved"
        return report
    vertices = load_obj_vertices(resolved_mesh)
    report["checked"] = True
    report["vertex_count"] = len(vertices)
    poses = [("start_pose", fixed_goals["start_pose"], start_min_tool_table_clearance_m)]
    poses.extend(
        (f"goals[{idx}]", goal, min_tool_table_clearance_m)
        for idx, goal in enumerate(fixed_goals["goals"])
    )
    for name, pose, min_clearance in poses:
        mesh_min_z, mesh_max_z = tool_mesh_z_bounds_for_pose(pose, vertices)
        clearance = mesh_min_z - float(table_top_z)
        item = {
            "pose": name,
            "root_z": float(pose[2]),
            "mesh_min_z": mesh_min_z,
            "mesh_max_z": mesh_max_z,
            "clearance_m": clearance,
            "required_clearance_m": float(min_clearance),
        }
        report["pose_clearances"].append(item)
        if clearance + 1e-6 < float(min_clearance):
            report["violations"].append(item)
    report["passed"] = not report["violations"]
    return report


def largest_box_top_z_from_urdf(asset_path: str | Path) -> float | None:
    """Return local z of the largest box top in a simple table URDF."""

    path = Path(asset_path)
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    best_area = -1.0
    best_top_z: float | None = None
    for element in root.findall(".//collision") + root.findall(".//visual"):
        box = element.find("./geometry/box")
        if box is None:
            continue
        try:
            sx, sy, sz = [float(item) for item in box.attrib.get("size", "").split()]
        except ValueError:
            continue
        origin = element.find("./origin")
        origin_z = 0.0
        if origin is not None and origin.attrib.get("xyz"):
            try:
                origin_z = float(origin.attrib["xyz"].split()[2])
            except (IndexError, ValueError):
                origin_z = 0.0
        area = sx * sy
        if area > best_area:
            best_area = area
            best_top_z = origin_z + sz / 2.0
    return best_top_z


def largest_box_xy_bounds_from_urdf(asset_path: str | Path) -> dict[str, list[float]] | None:
    """Return local xy bounds of the largest collision/visual box in a simple table URDF."""

    path = Path(asset_path)
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    best_area = -1.0
    best_bounds: dict[str, list[float]] | None = None
    for element in root.findall(".//collision") + root.findall(".//visual"):
        box = element.find("./geometry/box")
        if box is None:
            continue
        try:
            sx, sy, _ = [float(item) for item in box.attrib.get("size", "").split()]
        except ValueError:
            continue
        origin = element.find("./origin")
        origin_x = 0.0
        origin_y = 0.0
        if origin is not None and origin.attrib.get("xyz"):
            try:
                parts = [float(item) for item in origin.attrib["xyz"].split()]
                origin_x = parts[0]
                origin_y = parts[1]
            except (IndexError, ValueError):
                origin_x = 0.0
                origin_y = 0.0
        area = sx * sy
        if area > best_area:
            best_area = area
            best_bounds = {
                "x": [origin_x - sx / 2.0, origin_x + sx / 2.0],
                "y": [origin_y - sy / 2.0, origin_y + sy / 2.0],
            }
    return best_bounds


def table_xy_bounds_from_state_bundle(state_bundle: dict[str, Any]) -> dict[str, list[float]] | None:
    """Compute tabletop world xy bounds from a SimToolReal capture state bundle."""

    table_pose = state_bundle.get("table_pose_xyzw")
    table_asset = state_bundle.get("table_asset")
    if not table_asset or not isinstance(table_pose, list) or len(table_pose) < 2:
        return None
    local_bounds = largest_box_xy_bounds_from_urdf(
        _resolve_simtoolreal_asset_path(str(table_asset))
    )
    if local_bounds is None:
        return None
    return {
        "x": [
            float(table_pose[0]) + float(local_bounds["x"][0]),
            float(table_pose[0]) + float(local_bounds["x"][1]),
        ],
        "y": [
            float(table_pose[1]) + float(local_bounds["y"][0]),
            float(table_pose[1]) + float(local_bounds["y"][1]),
        ],
    }


def table_top_z_from_state_bundle(state_bundle: dict[str, Any]) -> float | None:
    """Compute tabletop world z from a SimToolReal capture state bundle."""

    table_pose = state_bundle.get("table_pose_xyzw")
    table_asset = state_bundle.get("table_asset")
    if not table_asset or not isinstance(table_pose, list) or len(table_pose) < 3:
        return None
    local_top_z = largest_box_top_z_from_urdf(
        _resolve_simtoolreal_asset_path(str(table_asset))
    )
    if local_top_z is None:
        return None
    return float(table_pose[2]) + float(local_top_z)


def fixed_goal_floor_z(
    *,
    start_pose: list[float],
    table_top_z: float,
    min_root_table_clearance_m: float = DEFAULT_MIN_ROOT_TABLE_CLEARANCE_M,
) -> float:
    """Return the lowest root z allowed for policy-facing fixed goals."""

    _ = start_pose
    return float(table_top_z) + min_root_table_clearance_m


def load_state_bundle(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))
