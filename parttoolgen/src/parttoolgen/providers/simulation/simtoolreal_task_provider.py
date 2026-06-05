#!/usr/bin/env python3
"""Emit legacy provider artifacts from a SimToolReal task asset.

This is the simulation-benchmark counterpart of a real environment/config
provider: it reads the task fixture geometry that SimToolReal already loads
(for example a table with a target block) and exposes symbolic target points, surfaces,
axes, and regions to the normal PartToolGen planner/compiler path.

It deliberately does not read policy state, robot state, contact outcomes, or
goal poses. The output is marked as a simulation/environment-config provider so
experiments do not confuse this with RGB-D-only perception.

This provider is not the active 6.4 image-grounded method path. Current
brush/hammer/eraser runners use SimToolReal RGB-D evidence, VLM scene refs, and
public tool refs instead of language-derived task defaults from this module.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    validate_grounding_candidates,
    validate_observed_scene_graph,
    validate_target_part_graph,
)


PROVIDER_ID = "simtoolreal_task_environment_provider"
PROVIDER_SOURCE = "simtoolreal_environment_config"


@dataclass(frozen=True)
class BoxGeometry:
    box_id: str
    origin_local: list[float]
    size_xyz: list[float]
    material: str | None = None

    @property
    def volume(self) -> float:
        return self.size_xyz[0] * self.size_xyz[1] * self.size_xyz[2]

    @property
    def xy_area(self) -> float:
        return self.size_xyz[0] * self.size_xyz[1]


def _synthetic_target_box(
    *,
    state_bundle: dict[str, Any],
    support_box: BoxGeometry,
) -> BoxGeometry:
    """Return a task-provider target when the fixture URDF only contains a table.

    Several DexToolBench environment URDFs describe only the support surface; the
    movable target is introduced by the benchmark environment rather than as a
    static table child.  The online provider contract still needs target geometry,
    so this fallback represents the externally observed target as a small box in
    the task asset frame.
    """

    configured = state_bundle.get("synthetic_target_box")
    if isinstance(configured, Mapping):
        size = configured.get("size_xyz")
        origin = configured.get("origin_local")
        if isinstance(size, list) and len(size) == 3 and isinstance(origin, list) and len(origin) == 3:
            return BoxGeometry(
                box_id="synthetic_target",
                origin_local=[float(value) for value in origin],
                size_xyz=[float(value) for value in size],
                material="synthetic_target",
            )
    size_xyz = [0.05, 0.05, 0.05]
    top_z = float(support_box.origin_local[2]) + float(support_box.size_xyz[2]) / 2.0
    return BoxGeometry(
        box_id="synthetic_target",
        origin_local=[0.0, 0.0, top_z + size_xyz[2] / 2.0],
        size_xyz=size_xyz,
        material="synthetic_target",
    )


def _task_action_axis_local(task_instruction: str) -> tuple[str, list[float], float]:
    """Infer a generic task motion axis from task language."""

    text = task_instruction.lower()
    if "right to left" in text or "to the left" in text or "leftward" in text:
        return "scene_direction", [0.0, -1.0, 0.0], 0.08
    if "left to right" in text or "to the right" in text or "rightward" in text:
        return "scene_direction", [0.0, 1.0, 0.0], 0.08
    if "backward" in text or "to the back" in text or "toward the back" in text:
        return "scene_direction", [1.0, 0.0, 0.0], 0.08
    if "right" in text:
        return "scene_direction", [0.0, 1.0, 0.0], 0.08
    if "forward" in text:
        return "scene_direction", [-1.0, 0.0, 0.0], 0.08
    if "horizontal" in text:
        return "scene_direction", [1.0, 0.0, 0.0], 0.05
    if "vertical" in text:
        return "scene_direction", [0.0, 0.0, 1.0], 0.05
    if "serve" in text:
        return "scene_direction", [0.0, 1.0, 0.0], 0.10
    if "flip" in text:
        return "scene_direction", [1.0, 0.0, 0.0], 0.05
    return "scene_direction", [0.0, 0.0, -1.0], 0.035


def _configured_task_goal_distance(
    state_bundle: Mapping[str, Any],
    default_distance_m: float,
) -> float:
    for key in ("task_motion_distance_m", "target_motion_distance_m", "task_goal_distance_m"):
        value = state_bundle.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            distance = float(value)
            if distance < 0.0:
                raise ValueError(f"state_bundle.{key} must be non-negative")
            return distance
    return float(default_distance_m)


def _vec3(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a 3-number list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a 3-number list") from exc


def _configured_target_motion_options(
    state_bundle: Mapping[str, Any],
    *,
    default_distance_m: float,
) -> list[dict[str, Any]]:
    rows = state_bundle.get("public_target_motion_options")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError("state_bundle.public_target_motion_options must be a list")
    options: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"public_target_motion_options[{index}] must be an object")
        axis_id = row.get("axis_id") or row.get("id")
        if not isinstance(axis_id, str) or not axis_id:
            raise ValueError(f"public_target_motion_options[{index}].axis_id is required")
        if not axis_id.startswith("target_"):
            raise ValueError(
                f"public_target_motion_options[{index}].axis_id must start with 'target_' "
                "so it is exported as a target axis ref"
            )
        axis_local = _unit(_vec3(row.get("axis_local"), field=f"public_target_motion_options[{index}].axis_local"))
        distance_value = row.get("distance_m", default_distance_m)
        if isinstance(distance_value, bool) or not isinstance(distance_value, (int, float)):
            raise ValueError(f"public_target_motion_options[{index}].distance_m must be numeric")
        distance_m = float(distance_value)
        if distance_m <= 0.0:
            raise ValueError(f"public_target_motion_options[{index}].distance_m must be positive")
        pre_offset = row.get("pre_contact_offset_m", 0.08)
        if isinstance(pre_offset, bool) or not isinstance(pre_offset, (int, float)):
            raise ValueError(f"public_target_motion_options[{index}].pre_contact_offset_m must be numeric")
        prefix = axis_id
        if prefix.endswith("_axis"):
            prefix = prefix[: -len("_axis")]
        options.append(
            {
                "axis_id": axis_id,
                "axis_local": axis_local,
                "distance_m": distance_m,
                "pre_contact_offset_m": float(pre_offset),
                "contact_keypoint_id": row.get("contact_keypoint_id") or f"{prefix}_contact_point",
                "pre_contact_keypoint_id": row.get("pre_contact_keypoint_id") or f"{prefix}_pre_contact_point",
                "goal_keypoint_id": row.get("goal_keypoint_id") or f"{prefix}_goal_point",
                "label": row.get("label") or prefix,
            }
        )
    return options


def _box_support_distance_along_axis(box: BoxGeometry, axis_local: list[float]) -> float:
    return 0.5 * sum(abs(float(axis_local[i])) * float(box.size_xyz[i]) for i in range(3))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _simtoolreal_assets_root() -> Path:
    return _repo_root() / "simtoolreal" / "assets"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _resolve_asset_path(asset_path: str, *, state_bundle_path: Path | None = None) -> Path:
    path = Path(asset_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if state_bundle_path is not None:
            candidates.append((state_bundle_path.parent / path).resolve())
            candidates.append((state_bundle_path.parent.parent / path).resolve())
        candidates.append((_simtoolreal_assets_root() / path).resolve())
        candidates.append((_repo_root() / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve SimToolReal asset {asset_path!r}; searched {searched}")


def _parse_vec3(text: str | None, *, default: list[float] | None = None) -> list[float]:
    if not text:
        return list(default or [0.0, 0.0, 0.0])
    values = [float(item) for item in text.split()]
    if len(values) != 3:
        raise ValueError(f"expected 3 values, got {text!r}")
    return values


def _quat_normalize(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in q))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    return [float(x) / norm for x in q]


def _quat_rotate(q_xyzw: list[float], v: list[float]) -> list[float]:
    x, y, z, w = _quat_normalize(q_xyzw)
    vx, vy, vz = [float(item) for item in v]
    # q * v * q^-1, expanded to avoid extra dependencies.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _mul(v: list[float], scale: float) -> list[float]:
    return [float(item) * float(scale) for item in v]


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in v))
    if norm <= 1e-12:
        raise ValueError("zero vector")
    return [float(item) / norm for item in v]


def _transform_point(
    local_xyz: list[float],
    *,
    world_position: list[float],
    world_orientation_xyzw: list[float],
) -> list[float]:
    return _add(world_position, _quat_rotate(world_orientation_xyzw, local_xyz))


def _transform_vector(local_vec: list[float], *, world_orientation_xyzw: list[float]) -> list[float]:
    return _unit(_quat_rotate(world_orientation_xyzw, local_vec))


def _parse_urdf_boxes(asset_path: Path) -> list[BoxGeometry]:
    root = ET.parse(asset_path).getroot()
    boxes: list[BoxGeometry] = []
    for index, element in enumerate(root.findall(".//collision")):
        box = element.find("./geometry/box")
        if box is None:
            continue
        size = _parse_vec3(box.attrib.get("size"))
        origin = element.find("./origin")
        origin_xyz = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
        boxes.append(
            BoxGeometry(
                box_id=f"collision_box_{index}",
                origin_local=origin_xyz,
                size_xyz=size,
            )
        )
    if boxes:
        return boxes
    for index, element in enumerate(root.findall(".//visual")):
        box = element.find("./geometry/box")
        if box is None:
            continue
        size = _parse_vec3(box.attrib.get("size"))
        origin = element.find("./origin")
        origin_xyz = _parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
        material = element.find("./material")
        boxes.append(
            BoxGeometry(
                box_id=f"visual_box_{index}",
                origin_local=origin_xyz,
                size_xyz=size,
                material=material.attrib.get("name") if material is not None else None,
            )
        )
    return boxes


def _select_target_box(state_bundle: Mapping[str, Any], target_boxes: list[BoxGeometry]) -> BoxGeometry | None:
    """Select the target fixture box named by the public target-selection bridge.

    This remains provider-side geometry routing: a benchmark may use a VLM
    target-selection result to identify which anonymous target object is active,
    then write its fixture box id here.  The planner still receives only the
    resulting target geometry refs, not private color labels or actor state.
    """

    if not target_boxes:
        return None
    if isinstance(state_bundle.get("synthetic_target_box"), Mapping):
        return None
    configured = state_bundle.get("target_box_id")
    if isinstance(configured, str) and configured:
        for box in target_boxes:
            if box.box_id == configured:
                return box
        raise ValueError(f"state_bundle.target_box_id={configured!r} did not match any parsed target boxes")
    configured_origin = state_bundle.get("target_box_origin_local")
    if isinstance(configured_origin, list) and len(configured_origin) == 3:
        desired = [float(value) for value in configured_origin]
        return min(
            target_boxes,
            key=lambda box: sum((float(box.origin_local[i]) - desired[i]) ** 2 for i in range(3)),
        )
    return max(target_boxes, key=lambda item: item.volume)


def _grounding(provenance: dict[str, Any], confidence: float = 1.0) -> dict[str, Any]:
    return {
        "confidence": float(confidence),
        "source": PROVIDER_SOURCE,
        "provenance": dict(provenance),
    }


def _derived_provenance(
    provenance: dict[str, Any],
    *,
    derived_from: str,
    axis_ref: str,
    offset_m: float,
    semantic_role: str,
) -> dict[str, Any]:
    data = dict(provenance)
    data.update(
        {
            "derived_from": derived_from,
            "axis_ref": axis_ref,
            "offset_m": float(offset_m),
            "semantic_role": semantic_role,
        }
    )
    return data


def _candidate_base(
    *,
    label: str,
    candidate_type: str,
    provenance: dict[str, Any],
    confidence: float = 1.0,
    object_ref: str | None = None,
    part_ref: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "label": label,
        "candidate_type": candidate_type,
        **_grounding(provenance, confidence),
    }
    if object_ref:
        data["object_ref"] = object_ref
    if part_ref:
        data["part_ref"] = part_ref
    return data


def _box_face(
    box: BoxGeometry,
    *,
    sign_axis: tuple[int, float],
    table_position: list[float],
    table_orientation: list[float],
) -> tuple[list[float], list[float]]:
    axis_index, sign = sign_axis
    local_normal = [0.0, 0.0, 0.0]
    local_normal[axis_index] = sign
    local_point = list(box.origin_local)
    local_point[axis_index] += sign * box.size_xyz[axis_index] / 2.0
    return (
        _transform_point(
            local_point,
            world_position=table_position,
            world_orientation_xyzw=table_orientation,
        ),
        _transform_vector(local_normal, world_orientation_xyzw=table_orientation),
    )


def _box_center_world(
    box: BoxGeometry,
    *,
    table_position: list[float],
    table_orientation: list[float],
) -> list[float]:
    return _transform_point(
        box.origin_local,
        world_position=table_position,
        world_orientation_xyzw=table_orientation,
    )


def _artifact_ids(observation_id: str) -> dict[str, str]:
    return {
        "scene": f"{observation_id}_simtoolreal_task_scene",
        "grounding": f"{observation_id}_simtoolreal_task_grounding",
        "target_graph": f"{observation_id}_simtoolreal_target_part_graph",
    }


def build_simtoolreal_task_artifacts(
    *,
    state_bundle: dict[str, Any],
    state_bundle_path: str | Path,
    task_instruction: str,
    observation_id: str,
    task_asset_path: str | Path | None = None,
    target_label: str = "target",
) -> dict[str, Any]:
    """Build scene/candidate/target artifacts from a SimToolReal task fixture."""

    state_path = Path(state_bundle_path)
    table_pose = state_bundle.get("table_pose_xyzw")
    if not isinstance(table_pose, list) or len(table_pose) != 7:
        raise ValueError("state bundle must contain table_pose_xyzw with 7 numbers")
    table_position = [float(item) for item in table_pose[:3]]
    table_orientation = _quat_normalize([float(item) for item in table_pose[3:7]])

    raw_asset = str(task_asset_path or state_bundle.get("table_asset") or "")
    if not raw_asset:
        raise ValueError("task asset path is required; state bundle has no table_asset")
    resolved_asset = _resolve_asset_path(raw_asset, state_bundle_path=state_path)
    boxes = _parse_urdf_boxes(resolved_asset)
    if not boxes:
        raise ValueError(f"task asset has no box collision/visual geometry: {resolved_asset}")

    support_box = max(boxes, key=lambda item: item.xy_area)
    target_boxes = [box for box in boxes if box is not support_box]
    target_box = _select_target_box(state_bundle, target_boxes)
    if target_box is None:
        target_box = _synthetic_target_box(state_bundle=state_bundle, support_box=support_box)

    ids = _artifact_ids(observation_id)
    source_artifact = str(resolved_asset)
    provenance = {
        "observation_id": observation_id,
        "method": "simtoolreal_task_urdf_environment_provider",
        "source_artifact": source_artifact,
        "state_bundle": str(state_path),
        "real_world_replacement": (
            "calibrated fixture/environment configuration, external target tracker, "
            "or RGB-D reconstruction provider"
        ),
    }

    support_center = _box_center_world(
        support_box,
        table_position=table_position,
        table_orientation=table_orientation,
    )
    target_center = _box_center_world(
        target_box,
        table_position=table_position,
        table_orientation=table_orientation,
    )
    support_top, support_top_normal = _box_face(
        support_box,
        sign_axis=(2, 1.0),
        table_position=table_position,
        table_orientation=table_orientation,
    )
    target_top, target_top_normal = _box_face(
        target_box,
        sign_axis=(2, 1.0),
        table_position=table_position,
        table_orientation=table_orientation,
    )
    target_contact_point = list(target_top)
    target_surface_normal_local = [0.0, 0.0, 1.0]
    target_surface_normal = list(target_top_normal)
    configured_contact_point = state_bundle.get("synthetic_target_contact_point_local")
    if (
        isinstance(configured_contact_point, list)
        and len(configured_contact_point) == 3
    ):
        target_contact_point = _transform_point(
            [float(value) for value in configured_contact_point],
            world_position=table_position,
            world_orientation_xyzw=table_orientation,
        )
    configured_surface_normal = state_bundle.get("synthetic_target_surface_normal_local")
    if (
        isinstance(configured_surface_normal, list)
        and len(configured_surface_normal) == 3
    ):
        target_surface_normal_local = [float(value) for value in configured_surface_normal]
        target_surface_normal = _unit(
            _transform_vector(
                target_surface_normal_local,
                world_orientation_xyzw=table_orientation,
            )
        )
    target_bottom, target_bottom_normal = _box_face(
        target_box,
        sign_axis=(2, -1.0),
        table_position=table_position,
        table_orientation=table_orientation,
    )
    action_down_axis = _unit([-x for x in target_top_normal])
    _task_axis_id, _task_axis_local, task_goal_distance_m = _task_action_axis_local(task_instruction)
    task_goal_distance_m = _configured_task_goal_distance(state_bundle, task_goal_distance_m)
    public_motion_options = _configured_target_motion_options(
        state_bundle,
        default_distance_m=task_goal_distance_m,
    )
    local_axes = {
        "target_local_x_axis": [1.0, 0.0, 0.0],
        "target_local_y_axis": [0.0, 1.0, 0.0],
        "target_surface_normal": target_surface_normal_local,
        "target_action_down_axis": [0.0, 0.0, -1.0],
        "target_action_forward_axis": [-1.0, 0.0, 0.0],
        "target_action_right_axis": [0.0, 1.0, 0.0],
        "target_action_left_axis": [0.0, -1.0, 0.0],
        "target_action_horizontal_axis": [1.0, 0.0, 0.0],
        "target_rotation_vertical_axis": [0.0, 0.0, 1.0],
    }
    axis_labels = {
        "target_local_x_axis": "target local +X axis",
        "target_local_y_axis": "target local +Y axis",
        "target_surface_normal": "selected target contact face normal",
        "target_action_down_axis": "downward action axis",
        "target_action_forward_axis": "forward sweep axis (-X in table frame)",
        "target_action_right_axis": "rightward sweep axis (+Y in table frame)",
        "target_action_left_axis": "leftward sweep axis (-Y in table frame)",
        "target_action_horizontal_axis": "horizontal action axis (+X in table frame)",
        "target_rotation_vertical_axis": "vertical rotation axis",
    }
    for option in public_motion_options:
        axis_id = str(option["axis_id"])
        local_axes[axis_id] = list(option["axis_local"])
        axis_labels[axis_id] = str(option["label"])
    target_object_id = f"target_{target_label}"
    axes = {
        axis_id: {
            "frame": "world",
            "object_ref": target_object_id,
            "vector_world": _transform_vector(axis, world_orientation_xyzw=table_orientation),
            "label": axis_labels.get(axis_id, axis_id),
            **_grounding(provenance),
        }
            for axis_id, axis in local_axes.items()
    }

    surfaces = {
        "support_table_top_surface": {
            "frame": "world",
            "point_xyz_world": support_top,
            "normal_world": support_top_normal,
            **_grounding(provenance),
        },
        "target_contact_surface": {
            "frame": "world",
            "point_xyz_world": target_contact_point,
            "normal_world": target_surface_normal,
            **_grounding(provenance),
        },
        "target_bottom_surface": {
            "frame": "world",
            "point_xyz_world": target_bottom,
            "normal_world": target_bottom_normal,
            **_grounding(provenance),
        },
    }
    for axis_index, axis_name in enumerate(("x", "y")):
        for sign in (-1.0, 1.0):
            point, normal = _box_face(
                target_box,
                sign_axis=(axis_index, sign),
                table_position=table_position,
                table_orientation=table_orientation,
            )
            surface_id = f"target_{'pos' if sign > 0 else 'neg'}_{axis_name}_side_surface"
            surfaces[surface_id] = {
                "frame": "world",
                "point_xyz_world": point,
                "normal_world": normal,
                **_grounding(provenance),
            }

    keypoints = {
        "target_center": {
            "frame": "world",
            "object_ref": target_object_id,
            "xyz_world": target_center,
            **_grounding(provenance),
        },
        "target_contact_point": {
            "frame": "world",
            "object_ref": target_object_id,
            "xyz_world": target_contact_point,
            **_grounding(provenance),
        },
        "support_table_center": {
            "frame": "world",
            "object_ref": "support_table",
            "xyz_world": support_center,
            **_grounding(provenance),
        },
    }
    public_motion_option_refs: list[dict[str, Any]] = []
    for option in public_motion_options:
        axis_local = list(option["axis_local"])
        motion_axis_world = _unit(_transform_vector(axis_local, world_orientation_xyzw=table_orientation))
        surface_normal_local = _mul(axis_local, -1.0)
        surface_normal_world = _unit(_transform_vector(surface_normal_local, world_orientation_xyzw=table_orientation))
        face_distance = _box_support_distance_along_axis(target_box, axis_local)
        contact_point_local = _add(target_box.origin_local, _mul(surface_normal_local, face_distance))
        contact_point_world = _transform_point(
            contact_point_local,
            world_position=table_position,
            world_orientation_xyzw=table_orientation,
        )
        pre_contact_point_world = _add(
            contact_point_world,
            _mul(surface_normal_world, float(option["pre_contact_offset_m"])),
        )
        goal_point_world = _add(contact_point_world, _mul(motion_axis_world, float(option["distance_m"])))
        contact_key = str(option["contact_keypoint_id"])
        pre_contact_key = str(option["pre_contact_keypoint_id"])
        goal_key = str(option["goal_keypoint_id"])
        axis_ref = f"target.axes.{option['axis_id']}"
        label = str(option["label"])
        keypoints[contact_key] = {
            "frame": "world",
            "object_ref": target_object_id,
            "xyz_world": contact_point_world,
            "label": f"{label} contact point",
            **_grounding(
                _derived_provenance(
                    provenance,
                    derived_from="scene.regions.target_bbox",
                    axis_ref=axis_ref,
                    offset_m=face_distance,
                    semantic_role="direction_conditioned_target_contact_point",
                )
            ),
        }
        keypoints[pre_contact_key] = {
            "frame": "world",
            "object_ref": target_object_id,
            "xyz_world": pre_contact_point_world,
            "label": f"{label} pre-contact point",
            **_grounding(
                _derived_provenance(
                    provenance,
                    derived_from=f"scene.keypoints.{contact_key}",
                    axis_ref=axis_ref,
                    offset_m=float(option["pre_contact_offset_m"]),
                    semantic_role="direction_conditioned_pre_contact_point",
                )
            ),
        }
        keypoints[goal_key] = {
            "frame": "world",
            "object_ref": target_object_id,
            "xyz_world": goal_point_world,
            "label": f"{label} goal point",
            **_grounding(
                _derived_provenance(
                    provenance,
                    derived_from=f"scene.keypoints.{contact_key}",
                    axis_ref=axis_ref,
                    offset_m=float(option["distance_m"]),
                    semantic_role="direction_conditioned_goal_point",
                )
            ),
        }
        surface_id = str(option["axis_id"]).replace("_axis", "_contact_surface")
        surfaces[surface_id] = {
            "frame": "world",
            "point_xyz_world": contact_point_world,
            "normal_world": surface_normal_world,
            "label": f"{label} contact face",
            **_grounding(provenance),
        }
        public_motion_option_refs.append(
            {
                "axis_ref": axis_ref,
                "contact_ref": f"scene.keypoints.{contact_key}",
                "pre_contact_ref": f"scene.keypoints.{pre_contact_key}",
                "goal_ref": f"scene.keypoints.{goal_key}",
                "axis_local": axis_local,
                "axis_world": motion_axis_world,
                "label": label,
            }
        )

    regions = {
        "target_bbox": {
            "frame": "world",
            "object_ref": target_object_id,
            "center_xyz_world": target_center,
            "extent_xyz": target_box.size_xyz,
            "normal_world": target_top_normal,
            **_grounding(provenance),
        },
        "support_table_bbox": {
            "frame": "world",
            "object_ref": "support_table",
            "center_xyz_world": support_center,
            "extent_xyz": support_box.size_xyz,
            "normal_world": support_top_normal,
            **_grounding(provenance),
        },
    }

    scene = {
        "schema_version": "v0.1",
        "scene_id": ids["scene"],
        "frames": {
            "world": {
                "type": "world",
                "description": "SimToolReal world frame used by fixed goal states.",
            },
            "simtoolreal_task_asset": {
                "type": "asset_local",
                "parent": "world",
                "description": "Task fixture/table asset frame transformed by table_pose_xyzw.",
            },
        },
        "objects": {
            "support_table": {
                "category": "support_surface",
                "attributes": {
                    "center_xyz_world": support_center,
                    "extent_xyz": support_box.size_xyz,
                    "asset_path": source_artifact,
                },
                **_grounding(provenance),
            },
            target_object_id: {
                "category": target_label,
                "attributes": {
                    "center_xyz_world": target_center,
                    "extent_xyz": target_box.size_xyz,
                    "role": "target_object",
                    "collision_role": state_bundle.get("target_collision_role"),
                    "asset_path": source_artifact,
                },
                **_grounding(provenance),
            },
        },
        "keypoints": keypoints,
        "axes": axes,
        "regions": regions,
        "surfaces": surfaces,
        "relations": [
            {
                "type": "mounted_on",
                "subject": f"scene.objects.{target_object_id}",
                "object": "scene.objects.support_table",
                "confidence": 1.0,
                "source": PROVIDER_SOURCE,
            }
        ],
        "metadata": {
            "created_by": "parttoolgen.providers.simulation.simtoolreal_task_provider",
            "task_instruction": task_instruction,
            "source_system": "simtoolreal",
            "geometry_mode": "mixed",
            "motion_binding_mode": (
                "public_multi_direction" if len(public_motion_option_refs) > 1 else "single_direction"
            ),
            "public_target_motion_options": public_motion_option_refs,
            "observation_bundle": str(state_path),
            "notes": (
                "Simulation benchmark provider: target/support geometry is read from "
                "the SimToolReal task fixture. Real deployments must replace this with "
                "a calibrated environment, tracker, or reconstruction provider. "
                "Only simple box collision/visual geometry is parsed."
            ),
        },
    }

    candidates = {
        "schema_version": "v0.1",
        "grounding_id": ids["grounding"],
        "observation_id": observation_id,
        "frames": scene["frames"],
        "marks": {},
        "points": {
            key: {
                **_candidate_base(
                    label=key,
                    candidate_type="point",
                    provenance=provenance,
                    object_ref=(
                        f"scene.objects.{value.get('object_ref')}"
                        if value.get("object_ref") and not str(value.get("object_ref")).startswith("scene.")
                        else value.get("object_ref")
                    ),
                ),
                "xyz_world": value["xyz_world"],
            }
            for key, value in keypoints.items()
            if key.startswith("target_")
        },
        "regions": {
            "target_bbox": {
                **_candidate_base(
                    label=f"{target_label} bbox",
                    candidate_type="region",
                    provenance=provenance,
                    object_ref=f"scene.objects.target_{target_label}",
                    part_ref="target.parts.target_contact_region",
                ),
                "center_xyz_world": target_center,
                "extent_xyz": target_box.size_xyz,
            },
            "support_table_bbox": {
                **_candidate_base(
                    label="support table bbox",
                    candidate_type="region",
                    provenance=provenance,
                    object_ref="scene.objects.support_table",
                ),
                "center_xyz_world": support_center,
                "extent_xyz": support_box.size_xyz,
            },
        },
        "axes": {
            axis_id: {
                **_candidate_base(
                    label=str(axis.get("label") or axis_id),
                    candidate_type="axis",
                    provenance=provenance,
                    object_ref=f"scene.objects.target_{target_label}",
                    part_ref="target.parts.target_contact_region",
                ),
                "vector_world": axis["vector_world"],
                "origin_xyz_world": target_contact_point,
            }
            for axis_id, axis in axes.items()
            if axis_id.startswith("target_")
        },
        "surfaces": {
            surface_id: {
                **_candidate_base(
                    label=str(surface.get("label") or surface_id),
                    candidate_type="surface",
                    provenance=provenance,
                    object_ref=(
                        f"scene.objects.target_{target_label}"
                        if surface_id.startswith("target_")
                        else "scene.objects.support_table"
                    ),
                    part_ref=(
                        "target.parts.target_contact_region"
                        if surface_id == "target_contact_surface"
                        else None
                    ),
                ),
                "point_xyz_world": surface["point_xyz_world"],
                "normal_world": surface["normal_world"],
            }
            for surface_id, surface in surfaces.items()
        },
        "parts": {
            "target_contact_region": {
                **_candidate_base(
                    label=f"{target_label} contact region",
                    candidate_type="target_part",
                    provenance=provenance,
                    object_ref=f"scene.objects.target_{target_label}",
                    part_ref="target.parts.target_contact_region",
                ),
                "role_candidates": [
                    "contact_region",
                    "target_part",
                    "impact_target",
                    f"{target_label}_contact_side",
                ],
            }
        },
        "metadata": {
            "created_by": "parttoolgen.providers.simulation.simtoolreal_task_provider",
            "grounding_mode": "mixed",
            "task_instruction": task_instruction,
            "source_kind": "ENVIRONMENT_CONFIG",
        },
    }

    target_part_graph = {
        "schema_version": "v0.1",
        "graph_id": ids["target_graph"],
        "target_id": f"target_{target_label}",
        "frame": "world",
        "parts": {
            "target_contact_region": {
                "part_ref": "target.parts.target_contact_region",
                "label": f"{target_label} contact region",
                "object_ref": f"scene.objects.target_{target_label}",
                "role_candidates": [
                    "contact_region",
                    "target_part",
                    "impact_target",
                ],
                "surface_patches": {
                    "contact_patch": {
                        "patch_ref": "target.parts.target_contact_region.surface_patches.contact_patch",
                        "role_hint": "contact_region",
                        "center_xyz_world": target_contact_point,
                        "normal_world": target_surface_normal,
                        **_grounding(provenance),
                    }
                },
                **_grounding(provenance),
            }
        },
        "axes": {
            axis_id: {
                "axis_ref": f"target.axes.{axis_id}",
                "object_ref": f"scene.objects.target_{target_label}",
                "vector_world": axis["vector_world"],
                "label": axis.get("label") or axis_id,
                **_grounding(provenance),
            }
            for axis_id, axis in axes.items()
            if axis_id.startswith("target_")
        },
        "relations": [
            {
                "type": "surface_of",
                "subject": "target.parts.target_contact_region",
                "object": f"scene.objects.target_{target_label}",
                "source": PROVIDER_SOURCE,
                "confidence": 1.0,
            }
        ],
        "metadata": {
            "created_by": "parttoolgen.providers.simulation.simtoolreal_task_provider",
            "geometry_mode": "mixed",
            "segmentation_mode": "environment_config_box_geometry",
            "observation_bundle": str(state_path),
            "source_kind": "ENVIRONMENT_CONFIG",
        },
    }

    provider_contract = {
        "schema_version": "v0.1",
        "provider_id": PROVIDER_ID,
        "provider_type": "simulation",
        "source_kind": "ENVIRONMENT_CONFIG",
        "deployable": True,
        "oracle": False,
        "capabilities": [
            "target_geometry",
            "collision_geometry",
            "environment_setup",
        ],
        "frame_contracts": [
            "world_scene_frame",
            "target_local_frame",
        ],
        "provenance": {
            "method": "simtoolreal_task_urdf_environment_provider",
            "source_artifacts": [source_artifact, str(state_path)],
            "timestamp_or_capture_id": observation_id,
            "notes": (
                "Simulation implementation of a deployable-style environment provider; "
                "not actor/root-state oracle geometry."
            ),
        },
        "metadata": {
            "task_instruction": task_instruction,
            "target_label": target_label,
            "support_box": {
                "origin_local": support_box.origin_local,
                "size_xyz": support_box.size_xyz,
            },
            "target_box": {
                "origin_local": target_box.origin_local,
                "size_xyz": target_box.size_xyz,
                "synthetic": not bool(target_boxes),
            },
            "task_action_axis_ref": None,
            "task_goal_distance_m": task_goal_distance_m,
            "motion_binding_mode": (
                "public_multi_direction" if len(public_motion_option_refs) > 1 else "single_direction"
            ),
            "public_target_motion_options": public_motion_option_refs,
            "geometry_parser_scope": "box_collision_geometry_only",
        },
    }

    validate_observed_scene_graph(scene)
    validate_grounding_candidates(candidates)
    validate_target_part_graph(target_part_graph)
    artifacts: dict[str, Any] = {
        "observed_scene_graph": scene,
        "grounding_candidates": candidates,
        "target_part_graph": target_part_graph,
        "provider_contract": provider_contract,
    }
    return artifacts


def write_simtoolreal_task_artifacts(
    *,
    state_bundle_path: str | Path,
    output_dir: str | Path,
    task_instruction: str,
    observation_id: str | None = None,
    task_asset_path: str | Path | None = None,
    target_label: str = "target",
) -> dict[str, Any]:
    state_path = Path(state_bundle_path)
    state_bundle = _load_json(state_path)
    capture_id = observation_id or state_path.parent.name
    artifacts = build_simtoolreal_task_artifacts(
        state_bundle=state_bundle,
        state_bundle_path=state_path,
        task_instruction=task_instruction,
        observation_id=capture_id,
        task_asset_path=task_asset_path,
        target_label=target_label,
    )
    output = Path(output_dir)
    _write_json(output / "observed_scene_graph.json", artifacts["observed_scene_graph"])
    _write_json(output / "grounding_candidates.json", artifacts["grounding_candidates"])
    _write_json(output / "target_part_graph.json", artifacts["target_part_graph"])
    _write_json(output / "provider_contract.json", artifacts["provider_contract"])
    outputs = {
        "observed_scene_graph": str(output / "observed_scene_graph.json"),
        "grounding_candidates": str(output / "grounding_candidates.json"),
        "target_part_graph": str(output / "target_part_graph.json"),
        "provider_contract": str(output / "provider_contract.json"),
    }
    manifest = {
        "generated_by": "parttoolgen.providers.simulation.simtoolreal_task_provider",
        "outputs": outputs,
    }
    _write_json(output / "manifest.json", manifest)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--observation-id")
    parser.add_argument("--task-asset")
    parser.add_argument("--target-label", default="target")
    args = parser.parse_args()
    write_simtoolreal_task_artifacts(
        state_bundle_path=args.state_bundle,
        output_dir=args.output_dir,
        task_instruction=args.task,
        observation_id=args.observation_id,
        task_asset_path=args.task_asset,
        target_label=args.target_label,
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
