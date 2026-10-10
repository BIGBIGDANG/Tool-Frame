"""Lift scene references using RGB-D or a supplied interaction plane."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ToGo.image_grounded import CameraIntrinsics, ImageKeypoint, build_contact_motion_program
from ToGo.image_grounded.lifting import lift_keypoint, unproject_pixel


def _unit(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected a finite 3D vector")
    length = float(np.linalg.norm(vector))
    if length < 1e-10:
        raise ValueError("zero-length direction")
    return vector / length


def lift_to_plane(*, u: float, v: float, intrinsics: CameraIntrinsics,
                  camera_to_world: Any, plane: Mapping[str, Any]) -> list[float]:
    """Intersect the calibrated pixel ray with the plane, preserving its pixel."""
    transform = np.asarray(camera_to_world, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    normal = _unit(plane["normal_world"])
    anchor = np.asarray(plane["point_world"], dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("plane point must be a finite 3D vector")
    ray = transform[:3, :3] @ unproject_pixel(u, v, 1.0, intrinsics)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-10:
        raise ValueError("pixel ray is parallel to the interaction plane")
    distance = float(np.dot(normal, anchor - transform[:3, 3])) / denominator
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("interaction plane is behind the camera")
    return (transform[:3, 3] + distance * ray).tolist()


def build_program(*, task: Mapping[str, Any], evidence_path: Path,
                  keypoints: Mapping[str, Any], tool_refs: Mapping[str, Any],
                  approach_offset_m: float) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text())
    images = evidence.get("metadata", {}).get("prompt_images", [])
    index = int(keypoints.get("grounding_image_index",
                             evidence.get("metadata", {}).get("grounding_image_index", 0)))
    if not 0 <= index < len(images):
        raise ValueError("grounding_image_index outside calibrated image list")
    rgb_item = next((item for item in evidence["items"]
                     if item.get("kind") == "scene_rgb" and item.get("path") == images[index]), None)
    if rgb_item is None:
        raise ValueError("grounding image has no matching scene_rgb item")
    prefix = str(rgb_item["id"])
    for suffix in ("_rgb", ".rgb"):
        if prefix.endswith(suffix):
            prefix = prefix[:-len(suffix)]
            break

    def paired(kind: str) -> Path:
        matches = [item for item in evidence["items"] if item.get("kind") == kind
                   and str(item["id"]).startswith((prefix + "_", prefix + "."))]
        if len(matches) != 1:
            raise ValueError(f"grounding image needs exactly one paired {kind} item")
        path = Path(matches[0]["path"])
        return path if path.is_absolute() else evidence_path.parent / path

    camera = json.loads(paired("camera").read_text())
    intrinsics = CameraIntrinsics.from_mapping(camera.get("intrinsics", camera))
    transform = camera["camera_to_world"]
    depth = np.load(paired("scene_depth"), allow_pickle=False)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth image dimensions differ from camera calibration")
    plane = dict(task["interaction_plane"]) if task.get("interaction_plane") else None
    if plane is not None and not plane.get("source"):
        raise ValueError("interaction_plane.source must identify the geometry source")
    roles = set(plane.get("point_roles", ["start", "end", "path"])) if plane else set()
    if roles - {"start", "end", "path"}:
        raise ValueError("interaction_plane.point_roles must use start, end, path")

    def pixel(point: Mapping[str, Any]) -> tuple[float, float]:
        uv = point.get("pixel_uv", [point.get("u"), point.get("v")])
        u, v = map(float, uv)
        unproject_pixel(u, v, 1.0, intrinsics)
        return u, v

    def depth_point(point: Mapping[str, Any], role: str) -> dict[str, Any]:
        u, v = pixel(point)
        row, col = min(int(round(v)), intrinsics.height - 1), min(int(round(u)), intrinsics.width - 1)
        value = float(depth[row, col])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid RGB-D sample at {role} pixel ({u}, {v})")
        result = lift_keypoint(ImageKeypoint(role, u, v, role), depth_m=value,
                               intrinsics=intrinsics, camera_to_world=transform).to_json()
        result["source"] = "calibrated_rgbd"
        return result

    if plane and plane.get("anchor_from_depth") == "start":
        start_pixel = keypoints.get("start_point")
        if not isinstance(start_pixel, Mapping):
            raise ValueError("missing start point for interaction-plane estimation")
        anchor = depth_point(start_pixel, "start")
        plane["point_world"] = anchor["position_world"]
        # Match 6.13's local depth-plane estimate, with the provided normal as orientation prior.
        u, v = pixel(start_pixel)
        samples = []
        for y in range(max(0, int(v) - 4), min(intrinsics.height, int(v) + 5)):
            for x in range(max(0, int(u) - 4), min(intrinsics.width, int(u) + 5)):
                value = float(depth[y, x])
                if np.isfinite(value) and value > 0 and abs(value - anchor["depth_m"]) < 0.02:
                    samples.append(depth_point({"u": x, "v": y}, "plane_sample")["position_world"])
        if len(samples) >= 6:
            points = np.asarray(samples)
            _, singular, vectors = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
            prior_normal = _unit(plane["normal_world"])
            candidate = vectors[-1]
            if singular[1] > 1e-6 and singular[-1] / singular[1] < 0.2:
                if np.dot(candidate, prior_normal) < 0:
                    candidate = -candidate
                if np.dot(candidate, prior_normal) > 0.5:
                    plane["normal_world"] = candidate.tolist()
                    plane["source"] = "local_rgbd_plane"

    def lift(point: Any, role: str) -> dict[str, Any]:
        if not isinstance(point, Mapping):
            raise ValueError(f"missing {role} point")
        u, v = pixel(point)
        if plane and role in roles:
            world = lift_to_plane(u=u, v=v, intrinsics=intrinsics,
                                  camera_to_world=transform, plane=plane)
            return {"position_world": world, "pixel_uv": [u, v], "source": plane["source"]}
        return depth_point(point, role)

    start = lift(keypoints.get("start_point"), "start")
    path = [lift(point, "path") for point in keypoints.get("path_points", [])]
    raw_end = keypoints.get("end_point")
    end = lift(raw_end, "end") if raw_end is not None else (path[-1] if path else dict(start))
    action = keypoints.get("action_direction") or {}
    axis_id = (keypoints.get("direction_decision") or {}).get("selected_default_axis_id")
    normal = plane.get("normal_world") if plane else task.get("metadata", {}).get("public_support_surface_normal_world")
    if axis_id in ("support_surface_normal", "opposite_support_surface_normal"):
        if normal is None:
            raise ValueError("VLM selected a surface axis without supplied plane geometry")
        direction = _unit(normal) * (-1 if axis_id.startswith("opposite") else 1)
        direction_source = "vlm_selected_interaction_plane_normal"
    elif action.get("start") is not None and action.get("end") is not None:
        a = lift(action["start"], "start")
        b = lift(action["end"], "start")
        direction = _unit(np.asarray(b["position_world"]) - a["position_world"])
        direction_source = "vlm_direction_arrow"
    else:
        points = [start, *path, end]
        direction = next((_unit(np.asarray(b["position_world"]) - a["position_world"])
                          for a, b in zip(points, points[1:])
                          if np.linalg.norm(np.asarray(b["position_world"]) - a["position_world"]) > 1e-5), None)
        if direction is None:
            raise ValueError("grounding must provide a nonzero direction or select a plane normal")
        direction_source = "vlm_grounded_path"
    motion_path = []
    for point in [start, *path, end]:
        if not motion_path or np.linalg.norm(np.asarray(point["position_world"]) - motion_path[-1]["position_world"]) > 1e-6:
            motion_path.append(point)
    refs = {"start": start, "end": end, "object_point": start,
            "direction": direction.tolist(), "direction_source": direction_source,
            "motion_path": motion_path}
    if normal is not None:
        refs["target_surface_normal"] = {"vector_world": _unit(normal).tolist(),
                                          "source": plane["source"] if plane else "provided_scene_geometry"}
    return build_contact_motion_program(task_id=str(task["task_id"]),
        instruction=str(task["instruction"]), tool_refs=tool_refs, scene_refs=refs,
        scene_contact_ref="scene.start", scene_action_end_ref="scene.end",
        scene_motion_path_ref="scene.motion_path" if len(motion_path) > 2 else None,
        tool_motion_axis_target_ref="scene.direction", tool_motion_axis_relation="parallel",
        pre_contact_distance_m=approach_offset_m).to_json()
