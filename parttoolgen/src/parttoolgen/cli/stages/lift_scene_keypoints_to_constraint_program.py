#!/usr/bin/env python3
"""Lift VLM scene pixels and emit a ToolFrame constraint program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.image_grounded import (  # noqa: E402
    CameraIntrinsics,
    ImageKeypoint,
    build_contact_motion_program,
    derive_axis,
    lift_keypoint,
)

DEFAULT_AXIS_IDS = {"support_surface_normal", "opposite_support_surface_normal"}


def _derive_axis_from_point_refs(
    start_position: Any,
    end_position: Any,
    *,
    primary_source: str,
    ordered_point_refs: list[Mapping[str, Any]] | None = None,
) -> tuple[tuple[float, float, float], str]:
    """Derive a scene direction from grounded point references."""

    try:
        return derive_axis(start_position, end_position), primary_source
    except ValueError as primary_error:
        points = ordered_point_refs or []
        for idx, (first, second) in enumerate(zip(points, points[1:])):
            first_position = first.get("position_world") if isinstance(first, Mapping) else None
            second_position = second.get("position_world") if isinstance(second, Mapping) else None
            if first_position is None or second_position is None:
                continue
            try:
                return derive_axis(first_position, second_position), f"ordered_point_refs_segment_{idx}_{idx + 1}"
            except ValueError:
                continue
        raise ValueError(
            "scene grounding did not provide a non-zero use-stage direction: "
            f"{primary_source} is zero-length and the grounded point refs have no non-zero segment"
        ) from primary_error


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _evidence_item_path(evidence: Mapping[str, Any], kind: str) -> Path:
    for item in evidence.get("items", []):
        if isinstance(item, Mapping) and item.get("kind") == kind:
            return Path(str(item["path"]))
    raise FileNotFoundError(f"scene evidence has no item kind={kind!r}")


def _evidence_item_path_for_grounding_image(
    evidence: Mapping[str, Any],
    keypoints: Mapping[str, Any],
    *,
    kind: str,
) -> Path:
    """Return the depth/camera paired with the image where VLM pixels live."""

    metadata = evidence.get("metadata", {})
    prompt_images = metadata.get("prompt_images", []) if isinstance(metadata, Mapping) else []
    grounding_image_index = int(
        keypoints.get(
            "grounding_image_index",
            metadata.get("grounding_image_index", 0) if isinstance(metadata, Mapping) else 0,
        )
    )
    if isinstance(prompt_images, list) and 0 <= grounding_image_index < len(prompt_images):
        grounding_rgb = Path(str(prompt_images[grounding_image_index]))
        rgb_item = next(
            (
                item
                for item in evidence.get("items", [])
                if isinstance(item, Mapping)
                and item.get("kind") == "scene_rgb"
                and Path(str(item.get("path"))) == grounding_rgb
            ),
            None,
        )
        if isinstance(rgb_item, Mapping):
            rgb_id = str(rgb_item.get("id", ""))
            for suffix in ("_rgb", ".rgb"):
                if rgb_id.endswith(suffix):
                    rgb_id = rgb_id[: -len(suffix)]
            preferred_ids = {
                "camera": [f"{rgb_id}_camera", f"{rgb_id}.camera"],
                "scene_depth": [f"{rgb_id}_depth", f"{rgb_id}.depth"],
            }.get(kind, [])
            for preferred_id in preferred_ids:
                for item in evidence.get("items", []):
                    if (
                        isinstance(item, Mapping)
                        and item.get("kind") == kind
                        and item.get("id") == preferred_id
                    ):
                        return Path(str(item["path"]))
    return _evidence_item_path(evidence, kind)


def _evidence_item_path_for_rgb_id(evidence: Mapping[str, Any], rgb_id: str, *, kind: str) -> Path:
    rgb_item = next(
        (
            item
            for item in evidence.get("items", [])
            if isinstance(item, Mapping)
            and item.get("kind") == "scene_rgb"
            and item.get("id") == rgb_id
        ),
        None,
    )
    if not isinstance(rgb_item, Mapping):
        raise FileNotFoundError(f"scene evidence has no scene_rgb item id={rgb_id!r}")
    stem = str(rgb_item.get("id", ""))
    for suffix in ("_rgb", ".rgb"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    preferred_ids = {
        "camera": [f"{stem}_camera", f"{stem}.camera"],
        "scene_depth": [f"{stem}_depth", f"{stem}.depth"],
    }.get(kind, [])
    for preferred_id in preferred_ids:
        for item in evidence.get("items", []):
            if (
                isinstance(item, Mapping)
                and item.get("kind") == kind
                and item.get("id") == preferred_id
            ):
                return Path(str(item["path"]))
    raise FileNotFoundError(f"scene evidence has no paired {kind} for rgb id={rgb_id!r}")


def _sample_depth(
    depth: np.ndarray,
    *,
    u: float,
    v: float,
    radii: tuple[int, ...] = (1, 3, 5, 9, 15),
    policy: str = "median",
) -> float:
    x = int(round(float(u)))
    y = int(round(float(v)))
    last_finite: np.ndarray | None = None
    for radius in radii:
        x0 = max(0, x - radius)
        x1 = min(depth.shape[1], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(depth.shape[0], y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch) & (patch > 0.0)]
        if finite.size:
            if policy != "median":
                last_finite = finite
                continue
            return float(np.median(finite))
    if last_finite is not None:
        if policy == "nearest":
            return float(np.min(last_finite))
        if policy == "p05":
            return float(np.percentile(last_finite, 5))
        raise ValueError(f"unsupported depth sample policy {policy!r}")
    raise ValueError(f"no valid depth near pixel ({u:.1f}, {v:.1f})")


def _camera_intrinsics(camera_json: Mapping[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(camera_json["width"]),
        height=int(camera_json["height"]),
        fx=float(camera_json["fx"]),
        fy=float(camera_json["fy"]),
        cx=float(camera_json["cx"]),
        cy=float(camera_json["cy"]),
    )


def _point_row(data: Mapping[str, Any], key: str) -> dict[str, float]:
    row = data.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"scene keypoint grounding is missing {key}")
    return {
        "u": float(row["u"]),
        "v": float(row["v"]),
        "confidence": float(row.get("confidence", 1.0)),
    }


def _point_row_any(data: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, dict[str, float]]:
    for key in keys:
        if key in data:
            return key, _point_row(data, key)
    raise ValueError(f"scene keypoint grounding is missing one of {keys}")


def _contact_height_row(data: Mapping[str, Any]) -> dict[str, float]:
    row = data.get("contact_height_point")
    if not isinstance(row, Mapping):
        raise ValueError("contact height grounding is missing contact_height_point")
    return {
        "u": float(row["u"]),
        "v": float(row["v"]),
        "confidence": float(row.get("confidence", 1.0)),
    }


def _action_direction_rows(data: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]] | None:
    row = data.get("action_direction")
    if not isinstance(row, Mapping):
        return None
    start = row.get("start")
    end = row.get("end")
    if not isinstance(start, Mapping) or not isinstance(end, Mapping):
        raise ValueError("action_direction must include start and end point objects")
    return (
        {
            "u": float(start["u"]),
            "v": float(start["v"]),
            "confidence": float(start.get("confidence", 1.0)),
        },
        {
            "u": float(end["u"]),
            "v": float(end["v"]),
            "confidence": float(end.get("confidence", 1.0)),
        },
    )


def _action_axis_3d_row(data: Mapping[str, Any]) -> dict[str, Any] | None:
    row = data.get("action_axis_3d")
    if isinstance(row, Mapping):
        raise ValueError(
            "scene_keypoints.action_axis_3d is no longer accepted; VLM scene "
            "grounding must provide 2D start/path/end point refs or choose a "
            "public default-axis candidate"
        )
    return None


def _direction_decision(data: Mapping[str, Any]) -> dict[str, Any]:
    row = data.get("direction_decision")
    if row is None:
        return {
            "source": "visible_action_direction",
            "selected_default_axis_id": None,
            "confidence": 1.0,
            "reason": "No direction_decision returned; using visible action point refs.",
            "source_defaulted": True,
        }
    if not isinstance(row, Mapping):
        raise ValueError("scene_keypoints.direction_decision must be an object")
    source = str(row.get("source") or "").strip().lower().replace("-", "_").replace(" ", "_")
    selected = row.get("selected_default_axis_id")
    if selected in {"", "null"}:
        selected = None
    if source == "default_axis_candidate":
        if not isinstance(selected, str) or selected not in DEFAULT_AXIS_IDS:
            raise ValueError(
                "direction_decision.selected_default_axis_id must be "
                f"one of {sorted(DEFAULT_AXIS_IDS)}"
            )
    elif source == "visible_action_direction":
        if selected not in {None, ""}:
            raise ValueError(
                "direction_decision.selected_default_axis_id must be null for visible_action_direction"
            )
    else:
        raise ValueError(
            "direction_decision.source must be visible_action_direction or default_axis_candidate"
        )
    confidence = row.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "source": source,
        "selected_default_axis_id": selected,
        "confidence": confidence_f,
        "reason": str(row.get("reason") or ""),
    }


def _axis_from_default_direction_decision(
    decision: Mapping[str, Any],
    *,
    support_surface_normal_world: list[float] | None,
) -> tuple[list[float] | None, str | None]:
    if decision.get("source") != "default_axis_candidate":
        return None, None
    selected = decision.get("selected_default_axis_id")
    if not isinstance(selected, str) or selected not in DEFAULT_AXIS_IDS:
        raise ValueError("default-axis direction decision is missing a valid selected_default_axis_id")
    if support_surface_normal_world is None:
        raise ValueError(
            "VLM selected a support-surface default axis, but no public support "
            "surface normal is available from args or task metadata"
        )
    normal = _unit_vector(support_surface_normal_world, name="support_surface_normal_world")
    if selected == "support_surface_normal":
        axis = normal
    else:
        axis = [-float(item) for item in normal]
    return axis, f"vlm_selected_default_axis:{selected}"


def _unit_vector(value: list[float], *, name: str) -> list[float]:
    arr = np.asarray([float(item) for item in value], dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be nonzero")
    return (arr / norm).astype(float).tolist()


def _path_point_rows(data: Mapping[str, Any]) -> list[dict[str, float | str]]:
    rows = data.get("path_points") or data.get("waypoints") or data.get("waypoint_points") or []
    if not isinstance(rows, list):
        raise ValueError("scene keypoint grounding path_points must be a list")
    parsed: list[dict[str, float | str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"waypoint {index} must be an object")
        parsed.append(
            {
                "id": str(row.get("id") or f"wp{index}"),
                "u": float(row["u"]),
                "v": float(row["v"]),
                "confidence": float(row.get("confidence", 1.0)),
            }
        )
    return parsed


def _resolve_tool_mesh(task_data: Mapping[str, Any] | None) -> str | None:
    if not task_data:
        return None
    value = task_data.get("execution", {}).get("simtoolreal_tool_mesh")
    return str(value) if isinstance(value, str) else None


def _task_metadata_vec3(task_data: Mapping[str, Any] | None, key: str) -> list[float] | None:
    metadata = task_data.get("metadata") if isinstance(task_data, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) < 3:
        return None
    return [float(item) for item in value[:3]]


def _project_keypoint_to_height(row: Mapping[str, Any], *, z_world: float, policy: str) -> dict[str, Any]:
    projected = dict(row)
    position = row.get("position_world")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("lifted keypoint has no position_world")
    projected["raw_position_world"] = [float(item) for item in position]
    projected["position_world"] = [float(position[0]), float(position[1]), float(z_world)]
    projected["z_policy"] = policy
    return projected


def _project_keypoint_to_contact_plane(
    row: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    normal_world: list[float],
    policy: str,
) -> dict[str, Any]:
    projected = dict(row)
    position = row.get("position_world")
    anchor_position = anchor.get("position_world")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("lifted keypoint has no position_world")
    if not isinstance(anchor_position, (list, tuple)) or len(anchor_position) != 3:
        raise ValueError("anchor keypoint has no position_world")
    normal = np.asarray(_unit_vector(normal_world, name="support_surface_normal_world"), dtype=float)
    point = np.asarray([float(item) for item in position], dtype=float)
    anchor_point = np.asarray([float(item) for item in anchor_position], dtype=float)
    corrected = point - normal * float(np.dot(point - anchor_point, normal))
    projected["raw_position_world"] = [float(item) for item in point]
    projected["position_world"] = corrected.astype(float).tolist()
    projected["surface_policy"] = policy
    projected["contact_plane_anchor_world"] = anchor_point.astype(float).tolist()
    projected["contact_plane_normal_world"] = normal.astype(float).tolist()
    return projected


def _distance_between(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    a_pos = a.get("position_world")
    b_pos = b.get("position_world")
    if not isinstance(a_pos, (list, tuple)) or len(a_pos) != 3:
        raise ValueError("first point has no position_world")
    if not isinstance(b_pos, (list, tuple)) or len(b_pos) != 3:
        raise ValueError("second point has no position_world")
    delta = np.asarray([float(b_pos[i]) - float(a_pos[i]) for i in range(3)], dtype=float)
    return float(np.linalg.norm(delta))


def _lifted_semantic_refs(
    *,
    keypoints: Mapping[str, Any],
    contact: Mapping[str, Any],
    end: Mapping[str, Any],
    direction: list[float],
    direction_source: str,
    motion_path: list[Mapping[str, Any]],
    action_distance_m: float | None,
    tool_motion_axis_target_ref: str,
    tool_motion_axis_relation: str,
) -> dict[str, Any]:
    """Package lifted VLM point refs without changing execution."""

    start_constraints = [
        {
            "type": "point_match",
            "tool_ref": "tool.contact_point",
            "scene_ref": "scene.start",
        }
    ]
    start_compiler_refs = {
        "point_ref": "scene.start",
    }
    start_ref: dict[str, Any] = {
        "role": "start_point",
        "point": dict(contact),
        "compiler_refs": start_compiler_refs,
        "constraints": start_constraints,
    }

    return {
        "schema_version": "toolframe_lifted_semantic_refs_v1",
        "source": "vlm_refs_lifted_through_rgbd",
        "raw_semantic_refs": keypoints.get("semantic_refs"),
        "start": start_ref,
        "use_stage": {
            "role": "constraint_preserving_use",
            "start": dict(contact),
            "end": dict(end),
            "axis": {
                "ref": "scene.direction",
                "vector_world": [float(item) for item in direction],
                "source": direction_source,
            },
            "path": [dict(item) for item in motion_path],
            "action_distance_m": action_distance_m,
            "tool_motion_axis_binding": {
                "ref": tool_motion_axis_target_ref,
                "relation": tool_motion_axis_relation,
                "source": "explicit_tool_motion_axis_binding",
            },
            "preserve": [
                "start.point_match",
                "action.tool_motion_axis_align",
            ],
            "compiler_refs": {
                "start_ref": "scene.start",
                "end_ref": "scene.end",
                "direction_ref": "scene.direction",
                "path_ref": "scene.motion_path",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-evidence", required=True)
    parser.add_argument("--scene-keypoints", required=True)
    parser.add_argument(
        "--contact-height-keypoints",
        default=None,
        help=(
            "Optional front/oblique VLM grounding JSON. Its point supplies only "
            "the contact z; top-down grounding still supplies xy and scene direction."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default=None, help="Optional task JSON for tool mesh metadata.")
    parser.add_argument(
        "--action-distance-m",
        type=float,
        default=None,
        help=(
            "Deprecated diagnostic/config value. Tool-contact travel distance is "
            "derived from a VLM end_point or final waypoint."
        ),
    )
    parser.add_argument(
        "--require-end-point",
        action="store_true",
        help=(
            "Require an executable use-stage endpoint. A VLM end_point satisfies "
            "this directly; if the VLM returns path_points with no end_point, "
            "the final path point is used as the endpoint while the complete "
            "path remains available as scene.motion_path."
        ),
    )
    parser.add_argument(
        "--action-direction-z-policy",
        choices=["anchor_height", "raw_depth"],
        default="raw_depth",
        help=(
            "How to lift action_direction arrow endpoints. Use anchor_height "
            "for tabletop XY arrows; use raw_depth for vertical actions such as "
            "hammering downward in a side/front view."
        ),
    )
    parser.add_argument(
        "--direction-source",
        choices=["path_geometry"],
        default="path_geometry",
        help=(
            "How to derive scene.direction. The current image-grounded mainline "
            "uses lifted VLM start/path/end point refs for every task family."
        ),
    )
    parser.add_argument(
        "--tool-motion-axis-target-ref",
        default="scene.direction",
        help=(
            "Scene-side direction ref for aligning tool.motion_axis during use-stage motion."
        ),
    )
    parser.add_argument(
        "--tool-motion-axis-relation",
        default="parallel",
        choices=[
            "parallel",
            "same",
            "anti_parallel",
            "antiparallel",
            "opposite",
        ],
        help="Relation between --tool-motion-axis-ref and --tool-motion-axis-target-ref.",
    )
    parser.add_argument(
        "--tool-motion-axis-ref",
        default="tool.motion_axis",
        help="Tool-side motion axis ref used for use-stage motion alignment.",
    )
    parser.add_argument(
        "--target-surface-normal-world",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional public observed/support-surface normal to write into "
            "scene_refs.target_surface_normal. This is a scene ref, not a "
            "benchmark target anchor correction."
        ),
    )
    parser.add_argument(
        "--support-surface-normal-world",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Public observed/support-surface normal. When --end-surface-policy "
            "is anchor_contact_plane, end/waypoints are projected onto the "
            "contact plane through start_point with this normal."
        ),
    )
    parser.add_argument(
        "--target-approach-axis-source",
        choices=["opposite_scene_direction"],
        default="opposite_scene_direction",
        help=(
            "Source for scene_refs.target_approach_axis. The current mainline uses "
            "the opposite VLM motion direction for every task family."
        ),
    )
    parser.add_argument(
        "--end-surface-policy",
        choices=["contact_height", "raw_depth", "anchor_contact_plane"],
        default="raw_depth",
        help=(
            "How to set executable end/waypoint surface coordinates. "
            "`anchor_contact_plane` keeps the start-point contact depth along "
            "the public support-surface normal, which is equivalent to keeping "
            "start.z on tabletop and keeping start's whiteboard-depth on side boards."
        ),
    )
    parser.add_argument(
        "--target-anchor-policy",
        choices=["raw_topdown"],
        default="raw_topdown",
        help=(
            "How to turn the top-down VLM start point into the executable "
            "contact anchor. The active image-grounded mainline only supports "
            "`raw_topdown`, which uses the RGB-D lifted VLM point directly."
        ),
    )
    parser.add_argument(
        "--depth-sample-policy",
        choices=["median", "nearest", "p05"],
        default="median",
        help=(
            "Depth sampling policy around VLM pixels. Use nearest for topdown "
            "RGB-D when a target point can lie on an object edge and the median "
            "window would snap to the tabletop."
        ),
    )
    args = parser.parse_args()
    tool_motion_axis_target_ref = args.tool_motion_axis_target_ref
    tool_motion_axis_relation = args.tool_motion_axis_relation

    evidence = _load_json(args.scene_evidence)
    keypoints = _load_json(args.scene_keypoints)
    task_data = _load_json(args.task) if args.task else None
    target_surface_normal_world = (
        [float(item) for item in args.target_surface_normal_world]
        if args.target_surface_normal_world is not None
        else None
    )
    support_surface_normal_world = (
        [float(item) for item in args.support_surface_normal_world]
        if args.support_surface_normal_world is not None
        else (
            target_surface_normal_world
            or _task_metadata_vec3(task_data, "public_support_surface_normal_world")
            or _task_metadata_vec3(task_data, "public_tabletop_surface_normal_world")
            or _task_metadata_vec3(task_data, "public_whiteboard_surface_normal_world")
            or _task_metadata_vec3(task_data, "public_target_surface_normal_world")
        )
    )
    target_surface_normal_world = (
        target_surface_normal_world
        if target_surface_normal_world is not None
        else support_surface_normal_world
    )
    contact_height_keypoints = (
        _load_json(args.contact_height_keypoints)
        if args.contact_height_keypoints
        else None
    )
    camera_json = _load_json(
        _evidence_item_path_for_grounding_image(evidence, keypoints, kind="camera")
    )
    depth = np.load(_evidence_item_path_for_grounding_image(evidence, keypoints, kind="scene_depth"))
    intrinsics = _camera_intrinsics(camera_json)
    camera_to_world = camera_json["camera_to_world"]

    anchor_key, object_point = _point_row_any(
        keypoints,
        ("start_point",),
    )
    action_direction = _action_direction_rows(keypoints)
    _action_axis_3d_row(keypoints)
    direction_decision = _direction_decision(keypoints)
    selected_default_axis, selected_default_axis_source = _axis_from_default_direction_decision(
        direction_decision,
        support_surface_normal_world=support_surface_normal_world,
    )
    end_point = None
    for end_key in ("end_point",):
        if isinstance(keypoints.get(end_key), Mapping):
            end_point = _point_row(keypoints, end_key)
            break
    path_point_rows = _path_point_rows(keypoints)
    has_canonical_path_points = isinstance(keypoints.get("path_points"), list)
    object_depth = _sample_depth(
        depth,
        u=object_point["u"],
        v=object_point["v"],
        policy=args.depth_sample_policy,
    )
    lifted_object = lift_keypoint(
        ImageKeypoint(
            anchor_key,
            object_point["u"],
            object_point["v"],
            "target_object_point",
            object_point["confidence"],
        ),
        depth_m=object_depth,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
    )
    lifted_contact_height = None
    if contact_height_keypoints is not None:
        contact_rgb_id = str(contact_height_keypoints.get("grounding_rgb_id") or "")
        if not contact_rgb_id:
            raise ValueError("contact height grounding is missing grounding_rgb_id")
        contact_camera_json = _load_json(
            _evidence_item_path_for_rgb_id(evidence, contact_rgb_id, kind="camera")
        )
        contact_depth = np.load(
            _evidence_item_path_for_rgb_id(evidence, contact_rgb_id, kind="scene_depth")
        )
        contact_intrinsics = _camera_intrinsics(contact_camera_json)
        contact_row = _contact_height_row(contact_height_keypoints)
        contact_depth_m = _sample_depth(
            contact_depth,
            u=contact_row["u"],
            v=contact_row["v"],
            policy=args.depth_sample_policy,
        )
        lifted_contact_height = lift_keypoint(
            ImageKeypoint(
                "contact_height_point",
                contact_row["u"],
                contact_row["v"],
                "target_contact_height",
                contact_row["confidence"],
            ),
            depth_m=contact_depth_m,
            intrinsics=contact_intrinsics,
            camera_to_world=contact_camera_json["camera_to_world"],
        )
    lifted_action_start = None
    lifted_action_end = None
    if action_direction is not None:
        direction_start, direction_end = action_direction
        if args.action_direction_z_policy == "anchor_height":
            # A tabletop action arrow is a 2D direction cue, not a request to
            # sample whatever surface happens to sit under the arrow endpoint.
            # Use the contacted object depth for both pixels so topdown arrows
            # lift to a stable XY direction even when the arrow ends on table.
            start_depth = object_depth
            end_depth = object_depth
        else:
            start_depth = _sample_depth(
                depth,
                u=direction_start["u"],
                v=direction_start["v"],
                policy=args.depth_sample_policy,
            )
            end_depth = _sample_depth(
                depth,
                u=direction_end["u"],
                v=direction_end["v"],
                policy=args.depth_sample_policy,
            )
        lifted_action_start = lift_keypoint(
            ImageKeypoint(
                "action_direction_start",
                direction_start["u"],
                direction_start["v"],
                "action_direction_start",
                direction_start["confidence"],
            ),
            depth_m=start_depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        lifted_action_end = lift_keypoint(
            ImageKeypoint(
                "action_direction_end",
                direction_end["u"],
                direction_end["v"],
                "action_direction_end",
                direction_end["confidence"],
            ),
            depth_m=end_depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
    lifted_end = None
    if end_point is not None:
        end_depth = _sample_depth(depth, u=end_point["u"], v=end_point["v"], policy=args.depth_sample_policy)
        lifted_end = lift_keypoint(
            ImageKeypoint(
                "end_point",
                end_point["u"],
                end_point["v"],
                "end_point",
                end_point["confidence"],
            ),
            depth_m=end_depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
    lifted_path_points = []
    for row in path_point_rows:
        path_point_depth = _sample_depth(
            depth,
            u=float(row["u"]),
            v=float(row["v"]),
            policy=args.depth_sample_policy,
        )
        lifted_path_points.append(
            lift_keypoint(
                ImageKeypoint(
                    str(row["id"]),
                    float(row["u"]),
                    float(row["v"]),
                    "path_point",
                    float(row["confidence"]),
                ),
                depth_m=path_point_depth,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
            )
        )
    path_point_jsons = [path_point.to_json() for path_point in lifted_path_points]
    path_end_json: dict[str, Any] | None = None
    motion_path_point_jsons = list(path_point_jsons)
    if lifted_end is None and motion_path_point_jsons:
        path_end_json = dict(motion_path_point_jsons[-1])
        path_end_json["keypoint_id"] = "end_point"
        path_end_json["role"] = "end_point_from_final_path_point"
        path_end_json["source"] = "vlm_final_path_point_rgbd_lifted"
        if not has_canonical_path_points:
            motion_path_point_jsons = motion_path_point_jsons[:-1]
    direction_source_label = ""
    object_json = lifted_object.to_json()
    contact_json = dict(object_json)
    contact_json["keypoint_id"] = "start_point"
    contact_json["role"] = "start_point_topdown_raw"
    contact_json["source"] = (
        "topdown_target_xy_plus_front_contact_z"
        if lifted_contact_height is not None
        else "topdown_target_xyz"
    )
    if lifted_contact_height is not None:
        object_position = object_json["position_world"]
        height_position = lifted_contact_height.to_json()["position_world"]
        contact_json["position_world"] = [
            float(object_position[0]),
            float(object_position[1]),
            float(height_position[2]),
        ]
        contact_json["role"] = "start_point_fused_xy_z"
        contact_json["topdown_object_position_world"] = object_position
        contact_json["front_contact_height_position_world"] = height_position

    if lifted_action_start is not None and lifted_action_end is not None:
        if args.action_direction_z_policy == "anchor_height":
            start_json = _project_keypoint_to_height(
                lifted_action_start.to_json(),
                z_world=float(lifted_object.position_world[2]),
                policy="action_anchor_height",
            )
            end_json = _project_keypoint_to_height(
                lifted_action_end.to_json(),
                z_world=float(lifted_object.position_world[2]),
                policy="action_anchor_height",
            )
        else:
            start_json = lifted_action_start.to_json()
            end_json = lifted_action_end.to_json()
        action_start_json = start_json
        action_end_json = end_json
        if lifted_end is not None:
            end_json = lifted_end.to_json()
            end_json["keypoint_id"] = "end_point"
            end_json["role"] = "end_point"
            end_json["source"] = "vlm_end_point_rgbd_lifted"
        elif path_end_json is not None:
            end_json = path_end_json
        else:
            if args.require_end_point:
                raise ValueError(
                    "scene grounding did not include an end point; this tool contact "
                    "path requires a VLM-marked use-stage endpoint or ordered point refs"
                )
            end_json = dict(contact_json)
            end_json["keypoint_id"] = "end_point"
            end_json["role"] = "start_equals_end_contact_only"
            end_json["source"] = "no_end_point_contact_only"
        if selected_default_axis is not None:
            direction_axis = selected_default_axis
            direction_source_label = str(selected_default_axis_source)
        else:
            direction_axis, direction_source_label = _derive_axis_from_point_refs(
                action_start_json["position_world"],
                action_end_json["position_world"],
                primary_source="action_direction_arrow_2d_lifted",
                ordered_point_refs=[contact_json, *motion_path_point_jsons, end_json],
            )
    elif lifted_end is not None:
        end_json = lifted_end.to_json()
        end_json["keypoint_id"] = "end_point"
        end_json["role"] = "end_point"
        end_json["source"] = "vlm_end_point_rgbd_lifted"
        direction_source_label = (
            str(selected_default_axis_source)
            if selected_default_axis is not None
            else "end_point_minus_start_point:raw_depth_pre_surface_policy"
        )
    else:
        raise ValueError("scene grounding needs either action_direction or end_point")
    end_surface_policy = args.end_surface_policy
    if end_surface_policy == "anchor_contact_plane":
        if support_surface_normal_world is None:
            raise ValueError(
                "--end-surface-policy anchor_contact_plane requires a public support "
                "surface normal from --support-surface-normal-world or task metadata"
            )
        end_json = _project_keypoint_to_contact_plane(
            end_json,
            anchor=contact_json,
            normal_world=support_surface_normal_world,
            policy=end_surface_policy,
        )
        path_point_jsons = [
            _project_keypoint_to_contact_plane(
                path_point,
                anchor=contact_json,
                normal_world=support_surface_normal_world,
                policy=end_surface_policy,
            )
            for path_point in path_point_jsons
        ]
        motion_path_point_jsons = [
            _project_keypoint_to_contact_plane(
                path_point,
                anchor=contact_json,
                normal_world=support_surface_normal_world,
                policy=end_surface_policy,
            )
            for path_point in motion_path_point_jsons
        ]
    elif end_surface_policy == "contact_height":
        contact_z = float(contact_json["position_world"][2])
        end_json = _project_keypoint_to_height(
            end_json,
            z_world=contact_z,
            policy=end_surface_policy,
        )
        path_point_jsons = [
            _project_keypoint_to_height(
                path_point,
                z_world=contact_z,
                policy=end_surface_policy,
            )
            for path_point in path_point_jsons
        ]
        motion_path_point_jsons = [
            _project_keypoint_to_height(
                path_point,
                z_world=contact_z,
                policy=end_surface_policy,
            )
            for path_point in motion_path_point_jsons
        ]
    if selected_default_axis is not None:
        direction_axis = selected_default_axis
        direction_source_label = str(selected_default_axis_source)
    elif lifted_end is not None:
        direction_axis, direction_source_label = _derive_axis_from_point_refs(
            contact_json["position_world"],
            end_json["position_world"],
            primary_source=f"end_point_minus_start_point:{end_surface_policy}",
            ordered_point_refs=[contact_json, *motion_path_point_jsons, end_json],
        )
    elif lifted_action_start is None or lifted_action_end is None:
        direction_axis, direction_source_label = _derive_axis_from_point_refs(
            contact_json["position_world"],
            end_json["position_world"],
            primary_source=f"end_point_minus_start_point:{end_surface_policy}",
            ordered_point_refs=[contact_json, *motion_path_point_jsons, end_json],
        )
    metric_action_distance_m = _distance_between(contact_json, end_json)
    action_distance_source = (
        "vlm_end_point_rgbd_projection"
        if lifted_end is not None
        else (
            "vlm_final_path_point_rgbd_projection"
            if path_end_json is not None
            else "no_end_point_contact_only"
        )
    )
    motion_path = (
        list(motion_path_point_jsons)
        if has_canonical_path_points and motion_path_point_jsons
        else [
            contact_json,
            *motion_path_point_jsons,
            end_json,
        ]
    )
    semantic_refs = _lifted_semantic_refs(
        keypoints=keypoints,
        contact=contact_json,
        end=end_json,
        direction=[float(item) for item in direction_axis],
        direction_source=direction_source_label,
        motion_path=motion_path,
        action_distance_m=metric_action_distance_m,
        tool_motion_axis_target_ref=tool_motion_axis_target_ref,
        tool_motion_axis_relation=tool_motion_axis_relation,
    )
    scene_refs = {
        "target_object": keypoints.get("target_object"),
        "object_point": lifted_object.to_json(),
        "start_point": contact_json,
        "start": contact_json,
        "end_point": end_json,
        "end": end_json,
        "direction": direction_axis,
        "motion_path": motion_path,
        "direction_source": direction_source_label,
        "direction_decision": direction_decision,
        "action_direction": {
            "start": lifted_action_start.to_json() if lifted_action_start is not None else None,
            "end": lifted_action_end.to_json() if lifted_action_end is not None else None,
        },
        "contact_height": lifted_contact_height.to_json() if lifted_contact_height is not None else None,
        "action_distance_m": metric_action_distance_m,
        "action_distance_source": (
            action_distance_source
        ),
        "tool_motion_axis_target_ref": tool_motion_axis_target_ref,
        "tool_motion_axis_relation": tool_motion_axis_relation,
        "semantic_refs": semantic_refs,
        "start_refs": semantic_refs["start"],
        "use_stage_refs": semantic_refs["use_stage"],
    }
    if target_surface_normal_world is not None:
        target_surface_normal = {
            "vector_world": _unit_vector(
                target_surface_normal_world,
                name="target_surface_normal_world",
            ),
            "source": (
                "public_observed_support_surface_prior"
                if args.target_surface_normal_world is not None or args.support_surface_normal_world is not None
                else "task.metadata.public_support_surface_normal_world"
            ),
        }
        scene_refs["target_surface_normal"] = target_surface_normal
        scene_refs["surface_normal"] = target_surface_normal
    target_approach_axis = _unit_vector(
        [-float(item) for item in direction_axis],
        name="target_approach_axis_from_negative_scene_direction",
    )
    target_approach_axis_source = "opposite_vlm_scene_direction"
    scene_refs["target_approach_axis"] = {
        "vector_world": target_approach_axis,
        "source": target_approach_axis_source,
    }
    scene_refs["direction"] = [float(item) for item in direction_axis]
    scene_refs["tool_motion_axis_binding"] = {
        "tool_axis_ref": args.tool_motion_axis_ref,
        "target_axis_ref": tool_motion_axis_target_ref,
        "relation": tool_motion_axis_relation,
        "source": direction_source_label,
        "role": "tool_motion_axis_to_scene_direction",
    }
    tool_refs = {
        "tool_mesh": _resolve_tool_mesh(task_data),
        "contact_point_face_id": "pending_tool_mesh_vlm_selection",
        "contact_point_ref": "tool.contact_point",
        "motion_axis_ref": args.tool_motion_axis_ref,
    }
    program = build_contact_motion_program(
        task_id=str(evidence.get("task_id") or keypoints.get("task_id") or "image_grounded_task"),
        instruction=str(evidence.get("instruction") or keypoints.get("instruction") or ""),
        tool_refs=tool_refs,
        scene_refs=scene_refs,
        tool_motion_axis_ref=args.tool_motion_axis_ref,
        contact_approach_axis_ref="target.axes.target_approach_axis",
        tool_motion_axis_target_ref=tool_motion_axis_target_ref,
        tool_motion_axis_relation=tool_motion_axis_relation,
        scene_motion_path_ref="scene.motion_path" if path_point_jsons else None,
    )
    output_dir = Path(args.output_dir)
    _write_json(
        output_dir / "lifted_refs.json",
        {
            "schema_version": "toolframe_lifted_refs_v0",
            "task_id": evidence.get("task_id"),
            "scene_evidence": str(Path(args.scene_evidence)),
            "scene_keypoints": str(Path(args.scene_keypoints)),
            "target_object": keypoints.get("target_object"),
            "object_point": lifted_object.to_json(),
            "start_point": contact_json,
            "start": contact_json,
            "contact_height": lifted_contact_height.to_json() if lifted_contact_height is not None else None,
            "end_point": end_json,
            "end": end_json,
            "action_endpoint": end_json,
            "raw_end": lifted_end.to_json() if lifted_end is not None else None,
            "action_direction": {
                "start": lifted_action_start.to_json() if lifted_action_start is not None else None,
                "end": lifted_action_end.to_json() if lifted_action_end is not None else None,
            },
            "path_points": path_point_jsons,
            "raw_path_points": [path_point.to_json() for path_point in lifted_path_points],
            "intermediate_path_points": (
                path_point_jsons[1:-1] if len(path_point_jsons) >= 2 else []
            ),
            "motion_path": motion_path,
            "direction": direction_axis,
            "direction_source": direction_source_label,
            "direction_decision": direction_decision,
            "semantic_refs": semantic_refs,
            "start_refs": semantic_refs["start"],
            "use_stage_refs": semantic_refs["use_stage"],
            "action_distance_m": metric_action_distance_m,
            "action_distance_source": (
                action_distance_source
            ),
            "action_direction_z_policy": args.action_direction_z_policy,
            "direction_source_policy": args.direction_source,
            "target_approach_axis": scene_refs.get("target_approach_axis"),
            "tool_motion_axis_target_ref": tool_motion_axis_target_ref,
            "tool_motion_axis_relation": tool_motion_axis_relation,
            "tool_motion_axis_ref": args.tool_motion_axis_ref,
            "target_surface_normal": scene_refs.get("target_surface_normal"),
            "end_surface_policy": end_surface_policy,
            "target_anchor_policy": args.target_anchor_policy,
            "depth_sample_policy": args.depth_sample_policy,
        },
    )
    _write_json(output_dir / "constraint_program.json", program.to_json())
    print(
        json.dumps(
            {
                "lifted_refs": str(output_dir / "lifted_refs.json"),
                "constraint_program": str(output_dir / "constraint_program.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
