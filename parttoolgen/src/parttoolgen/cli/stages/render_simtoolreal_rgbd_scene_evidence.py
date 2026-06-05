#!/usr/bin/env python3
"""Render true SimToolReal RGB-D scene evidence for image-grounded planning."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
PARTTOOLGEN_ROOT = ROOT / "parttoolgen"
SRC_ROOT = PARTTOOLGEN_ROOT / "src"
SIMTOOLREAL_ROOT = ROOT / "simtoolreal"
ISAACGYM_ROOT = Path(
    os.environ.get(
        "PARTTOOLGEN_ISAACGYM",
        str(ROOT / "artifacts" / "third_party" / "isaacgym" / "python"),
    )
)
for path in (
    SRC_ROOT,
    PARTTOOLGEN_ROOT,
    SIMTOOLREAL_ROOT,
    SIMTOOLREAL_ROOT / "isaacgymenvs",
    ISAACGYM_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parttoolgen.execution.simtoolreal_rollout import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_RENDER_LIGHT_PRESET,
    DEFAULT_SOFT_LIGHT_INDICES,
    _benchmark_extra_actor_specs,
    _install_numpy_legacy_aliases,
    _install_pytorch3d_transform_fallback,
    apply_simtoolreal_render_lighting,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _task_state_bundle(task_data: Mapping[str, Any]) -> dict[str, Any]:
    state_bundle = task_data.get("execution", {}).get("state_bundle", {})
    if not isinstance(state_bundle, dict):
        raise ValueError("task.execution.state_bundle must be a mapping")
    return state_bundle


def _task_start_pose(task_data: Mapping[str, Any]) -> list[float]:
    start_pose = task_data.get("execution", {}).get("simtoolreal_start_pose")
    if not isinstance(start_pose, list) or len(start_pose) != 7:
        raise ValueError("task.execution.simtoolreal_start_pose must be a 7D xyzw pose")
    return [float(item) for item in start_pose]


def _default_device() -> str:
    # IsaacGym must be imported before torch, so do not probe torch here.
    return os.environ.get("PARTTOOLGEN_SIMTOOLREAL_DEVICE", "cuda")


def _normalize(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError(f"cannot normalize near-zero vector: {v}")
    return arr / norm


def _camera_to_world_from_look_at(
    *,
    position: Sequence[float],
    target: Sequence[float],
    up_hint: Sequence[float] = (0.0, 0.0, 1.0),
) -> list[list[float]]:
    """Build a camera-to-world transform using x-right, y-down, z-forward axes."""

    pos = np.asarray(position, dtype=float)
    forward = _normalize(np.asarray(target, dtype=float) - pos)
    up = _normalize(up_hint)
    if abs(float(np.dot(forward, up))) > 0.95:
        up = _normalize((0.0, 1.0, 0.0))
    right = _normalize(np.cross(forward, up))
    down = _normalize(np.cross(forward, right))
    matrix = np.eye(4, dtype=float)
    matrix[:3, 0] = right
    matrix[:3, 1] = down
    matrix[:3, 2] = forward
    matrix[:3, 3] = pos
    return matrix.tolist()


def _intrinsics_from_fov(*, width: int, height: int, horizontal_fov_deg: float) -> dict[str, float | int]:
    fov_rad = math.radians(float(horizontal_fov_deg))
    fx = float(width) / (2.0 * math.tan(fov_rad * 0.5))
    # IsaacGym uses square pixels; vertical FoV follows from aspect ratio.
    fy = fx
    return {
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
    }


def _rgba_to_rgb(color_image: Any, *, width: int, height: int) -> np.ndarray:
    array = np.asarray(color_image)
    if array.size != width * height * 4:
        array = array.reshape(height, width, -1)
    else:
        array = array.reshape(height, width, 4)
    return np.asarray(array[:, :, :3], dtype=np.uint8).copy()


def _depth_to_positive_m(depth_image: Any, *, width: int, height: int) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32).reshape(height, width)
    depth = depth.copy()
    # IsaacGym depth is negative distance along the camera ray; -inf means no hit.
    depth[~np.isfinite(depth)] = np.nan
    positive = -depth
    positive[positive <= 0.0] = np.nan
    return positive.astype(np.float32)


def _save_depth_vis(depth_m: np.ndarray, path: Path) -> None:
    valid = np.isfinite(depth_m)
    if not valid.any():
        Image.fromarray(np.zeros(depth_m.shape, dtype=np.uint8), mode="L").save(path)
        return
    near = float(np.nanpercentile(depth_m, 2))
    far = float(np.nanpercentile(depth_m, 98))
    if far <= near + 1e-6:
        far = near + 1.0
    scaled = (1.0 - np.clip((depth_m - near) / (far - near), 0.0, 1.0)) * 255.0
    scaled[~valid] = 0.0
    Image.fromarray(scaled.astype(np.uint8), mode="L").save(path)


def _camera_props_to_dict(props: Any) -> dict[str, Any]:
    return {
        "width": int(props.width),
        "height": int(props.height),
        "horizontal_fov_deg": float(getattr(props, "horizontal_fov", 75.0)),
        "near_plane": float(getattr(props, "near_plane", 0.0)),
        "far_plane": float(getattr(props, "far_plane", 0.0)),
    }


def _capture_camera(
    *,
    env: Any,
    camera_handle: Any,
    camera_props: Any,
    env_handle: Any,
    camera_id: str,
    output_dir: Path,
    position_world: Sequence[float] | None = None,
    look_at_world: Sequence[float] | None = None,
    rotate_180_image: bool = False,
    frame_convention: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from isaacgym import gymapi  # noqa: WPS433

    if hasattr(env, "render"):
        env.render()
    else:
        env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    width = int(camera_props.width)
    height = int(camera_props.height)
    rgb = _rgba_to_rgb(
        env.gym.get_camera_image(env.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR),
        width=width,
        height=height,
    )
    depth = _depth_to_positive_m(
        env.gym.get_camera_image(env.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH),
        width=width,
        height=height,
    )
    image_transform: dict[str, Any] = {"rotation_degrees": 0}
    if rotate_180_image:
        rgb = np.rot90(rgb, 2).copy()
        depth = np.rot90(depth, 2).copy()
        image_transform = {
            "rotation_degrees": 180,
            "reason": "robot_forward_topdown_view",
            "camera_to_world_adjustment": (
                "image x/y axes are negated after rotating RGB and depth by 180 degrees"
            ),
        }

    props = _camera_props_to_dict(camera_props)
    intrinsics = _intrinsics_from_fov(
        width=width,
        height=height,
        horizontal_fov_deg=float(props["horizontal_fov_deg"]),
    )
    if rotate_180_image:
        intrinsics["cx"] = float(width - 1) - float(intrinsics["cx"])
        intrinsics["cy"] = float(height - 1) - float(intrinsics["cy"])
    camera_json: dict[str, Any] = {
        "schema_version": "toolframe_simtoolreal_camera_v0",
        "model": "pinhole",
        "source": "SimToolReal IsaacGym camera sensor",
        **intrinsics,
        "isaacgym_camera_properties": props,
        "depth_encoding": {
            "saved_depth": "positive_meters",
            "isaacgym_raw_depth_sign": "negative_distance",
            "invalid_depth": "nan",
        },
    }
    if position_world is not None and look_at_world is not None:
        camera_json["position_world"] = [float(item) for item in position_world[:3]]
        camera_json["look_at_world"] = [float(item) for item in look_at_world[:3]]
        camera_to_world = _camera_to_world_from_look_at(
            position=position_world,
            target=look_at_world,
        )
        if rotate_180_image:
            matrix = np.asarray(camera_to_world, dtype=float)
            matrix[:3, 0] *= -1.0
            matrix[:3, 1] *= -1.0
            camera_to_world = matrix.tolist()
        camera_json["camera_to_world"] = camera_to_world
    camera_json["image_transform"] = image_transform
    if frame_convention is not None:
        camera_json["frame_convention"] = dict(frame_convention)

    rgb_path = output_dir / f"{camera_id}_rgb.png"
    depth_path = output_dir / f"{camera_id}_depth.npy"
    depth_vis_path = output_dir / f"{camera_id}_depth_vis.png"
    camera_path = output_dir / f"{camera_id}_camera.json"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(depth_path, depth)
    _save_depth_vis(depth, depth_vis_path)
    try:
        view_matrix = env.gym.get_camera_view_matrix(env.sim, env_handle, camera_handle)
        proj_matrix = env.gym.get_camera_proj_matrix(env.sim, env_handle, camera_handle)
        camera_json["isaacgym_view_matrix"] = np.asarray(view_matrix, dtype=float).tolist()
        camera_json["isaacgym_projection_matrix"] = np.asarray(proj_matrix, dtype=float).tolist()
    except Exception as exc:  # pragma: no cover - depends on IsaacGym build.
        camera_json["matrix_capture_warning"] = str(exc)
    _write_json(camera_path, camera_json)

    return {
        "camera_id": camera_id,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "depth_vis_path": str(depth_vis_path),
        "camera_path": str(camera_path),
        "width": width,
        "height": height,
        "valid_depth_fraction": float(np.isfinite(depth).mean()),
        "image_transform": image_transform,
        "frame_convention": dict(frame_convention) if frame_convention is not None else None,
    }


def _create_env(
    *,
    task_data: Mapping[str, Any],
    config_path: Path,
    device: str,
    headless: bool,
    log_path: Path,
) -> Any:
    _install_numpy_legacy_aliases()
    _install_pytorch3d_transform_fallback()
    from deployment.isaac.isaac_env import create_env  # noqa: WPS433

    state_bundle = _task_state_bundle(task_data)
    start_pose = _task_start_pose(task_data)
    table_pose = state_bundle.get("table_pose_xyzw")
    overrides: dict[str, Any] = {
        "task.env.resetPositionNoiseX": 0.0,
        "task.env.resetPositionNoiseY": 0.0,
        "task.env.resetPositionNoiseZ": 0.0,
        "task.env.randomizeObjectRotation": False,
        "task.env.resetDofPosRandomIntervalFingers": 0.0,
        "task.env.resetDofPosRandomIntervalArm": 0.0,
        "task.env.resetDofVelRandomInterval": 0.0,
        "task.env.tableResetZRange": 0.0,
        "task.env.objectName": str(state_bundle.get("object_name") or task_data.get("tool", {}).get("tool_id")),
        "task.env.numEnvs": 1,
        "task.env.envSpacing": 0.4,
        "task.env.capture_video": False,
        "task.env.useFixedGoalStates": True,
        "task.env.fixedGoalStates": [start_pose],
        "task.env.hideGoalObjectVisual": True,
        "task.env.goalObjectVisualOpacity": 0.0,
        "task.env.useActionDelay": False,
        "task.env.useObsDelay": False,
        "task.env.useObjectStateDelayNoise": False,
        "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
        "task.env.resetWhenDropped": False,
        "task.env.useFixedInitObjectPose": True,
        "task.env.objectStartPose": start_pose,
        "task.env.startArmHigher": False,
        "task.env.forceScale": 0.0,
        "task.env.torqueScale": 0.0,
        "task.env.linVelImpulseScale": 0.0,
        "task.env.angVelImpulseScale": 0.0,
        "task.env.forceProbRange": [0.0001, 0.0001],
        "task.env.torqueProbRange": [0.0001, 0.0001],
        "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
        "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
    }
    extra_actors = [dict(item) for item in _benchmark_extra_actor_specs(state_bundle)]
    if extra_actors:
        overrides["task.env.parttoolgenBenchmarkExtraActors"] = extra_actors
    if state_bundle.get("table_asset"):
        overrides["task.env.asset.table"] = str(state_bundle["table_asset"])
    if isinstance(table_pose, list) and len(table_pose) >= 3:
        overrides["task.env.tableResetZ"] = float(table_pose[2])

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            env = create_env(
                config_path=str(config_path),
                headless=headless,
                device=device,
                episode_length=12,
                overrides=overrides,
            )
            import torch  # noqa: WPS433

            env.step(torch.zeros((env.num_envs, env.num_acts), device=device))
    return env


def render_simtoolreal_rgbd_scene_evidence(
    *,
    task_path: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    device: str | None = None,
    headless: bool = True,
    topdown_width: int = 900,
    topdown_height: int = 700,
    topdown_fov_deg: float = 55.0,
    grounding_camera: str = "topdown",
    grounding_width: int | None = None,
    grounding_height: int | None = None,
    grounding_fov_deg: float = 52.0,
    rollout_width: int = 900,
    rollout_height: int = 506,
    rollout_fov_deg: float = 70.0,
    include_presentation_camera: bool = False,
    presentation_width: int = 1100,
    presentation_height: int = 620,
    presentation_fov_deg: float = 50.0,
    light_preset: str = DEFAULT_RENDER_LIGHT_PRESET,
    soft_light_indices: Sequence[int] = DEFAULT_SOFT_LIGHT_INDICES,
) -> dict[str, Any]:
    task_data = _load_json(task_path)
    task_metadata = task_data.get("metadata", {})
    if not isinstance(task_metadata, Mapping):
        task_metadata = {}
    public_scene_context = {
        key: task_metadata[key]
        for key in (
            "target_kind",
            "whiteboard_center_world",
            "public_support_surface_normal_world",
            "public_tabletop_surface_normal_world",
            "public_whiteboard_surface_normal_world",
            "public_target_surface_normal_world",
        )
        if key in task_metadata
    }
    support_normal = (
        public_scene_context.get("public_support_surface_normal_world")
        or public_scene_context.get("public_tabletop_surface_normal_world")
        or public_scene_context.get("public_whiteboard_surface_normal_world")
        or public_scene_context.get("public_target_surface_normal_world")
    )
    if support_normal is not None:
        public_scene_context["default_axis_candidates"] = [
            {
                "id": "support_surface_normal",
                "vector_source": "scene.support_surface_normal",
                "meaning": "move away from the support/contact surface",
            },
            {
                "id": "opposite_support_surface_normal",
                "vector_source": "-scene.support_surface_normal",
                "meaning": "move into or press toward the support/contact surface",
            },
        ]
    state_bundle = _task_state_bundle(task_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = device or _default_device()
    log_path = output_dir / "simtoolreal_render_log.txt"
    env = _create_env(
        task_data=task_data,
        config_path=config_path,
        device=resolved_device,
        headless=headless,
        log_path=log_path,
    )
    from isaacgym import gymapi  # noqa: WPS433

    lighting = apply_simtoolreal_render_lighting(
        env,
        light_preset,
        soft_light_indices=soft_light_indices,
    )

    env_handle = env.envs[getattr(env, "index_to_view", 0)]
    # Match the original SimToolReal rollout camera so the VLM height mark is
    # grounded on the same view used for rollout inspection.
    rollout_position = [0.0, -0.42, 0.72]
    rollout_target = [0.0, 0.02, 0.50]
    rollout_props = gymapi.CameraProperties()
    rollout_props.width = int(rollout_width)
    rollout_props.height = int(rollout_height)
    rollout_props.horizontal_fov = float(rollout_fov_deg)
    rollout_handle = env.gym.create_camera_sensor(env_handle, rollout_props)
    env.gym.set_camera_location(rollout_handle, env_handle, gymapi.Vec3(*rollout_position), gymapi.Vec3(*rollout_target))
    rollout = _capture_camera(
        env=env,
        camera_handle=rollout_handle,
        camera_props=rollout_props,
        env_handle=env_handle,
        camera_id="sim_rollout",
        output_dir=output_dir,
        position_world=rollout_position,
        look_at_world=rollout_target,
    )

    top_props = gymapi.CameraProperties()
    top_props.width = int(topdown_width)
    top_props.height = int(topdown_height)
    top_props.horizontal_fov = float(topdown_fov_deg)
    top_handle = env.gym.create_camera_sensor(env_handle, top_props)
    # Slight y offset avoids a singular look-at up vector while still behaving
    # like an overhead workspace camera. The view is centered on the table
    # rather than the target object so the VLM sees the full workspace.
    top_pos = gymapi.Vec3(0.0, -0.001, 1.42)
    top_target = gymapi.Vec3(0.0, 0.0, 0.53)
    env.gym.set_camera_location(top_handle, env_handle, top_pos, top_target)
    robot_forward_topdown_frame = {
        "name": "robot_forward_topdown",
        "image_up": "robot_forward_away_from_arm",
        "image_down": "toward_robot_arm",
        "image_right": "robot_right",
        "action_direction_rule": (
            "VLM arrows are interpreted in this frame: up means away from the arm, "
            "down means toward the arm, left/right are the arm's left/right."
        ),
    }
    topdown = _capture_camera(
        env=env,
        camera_handle=top_handle,
        camera_props=top_props,
        env_handle=env_handle,
        camera_id="sim_topdown",
        output_dir=output_dir,
        position_world=[top_pos.x, top_pos.y, top_pos.z],
        look_at_world=[top_target.x, top_target.y, top_target.z],
        rotate_180_image=True,
        frame_convention=robot_forward_topdown_frame,
    )

    grounding = topdown
    grounding_frame = robot_forward_topdown_frame
    if grounding_camera == "slight_oblique":
        grounding_props = gymapi.CameraProperties()
        grounding_props.width = int(grounding_width or topdown_width)
        grounding_props.height = int(grounding_height or topdown_height)
        grounding_props.horizontal_fov = float(grounding_fov_deg)
        grounding_handle = env.gym.create_camera_sensor(env_handle, grounding_props)
        grounding_pos = gymapi.Vec3(0.0, -0.24, 1.34)
        grounding_target = gymapi.Vec3(0.0, 0.0, 0.53)
        env.gym.set_camera_location(
            grounding_handle,
            env_handle,
            grounding_pos,
            grounding_target,
        )
        grounding_frame = {
            "name": "robot_forward_slight_oblique",
            "image_up": "mostly_robot_forward_away_from_arm",
            "image_down": "toward_robot_arm",
            "image_right": "robot_right",
            "action_direction_rule": (
                "VLM arrows are interpreted in this calibrated camera image. "
                "Horizontal arrows should follow the visible table plane; "
                "vertical/down-strike actions can be indicated by a point marker "
                "when the motion is into the image."
            ),
        }
        grounding = _capture_camera(
            env=env,
            camera_handle=grounding_handle,
            camera_props=grounding_props,
            env_handle=env_handle,
            camera_id="sim_grounding_oblique",
            output_dir=output_dir,
            position_world=[
                grounding_pos.x,
                grounding_pos.y,
                grounding_pos.z,
            ],
            look_at_world=[
                grounding_target.x,
                grounding_target.y,
                grounding_target.z,
            ],
            frame_convention=grounding_frame,
        )
    elif grounding_camera == "whiteboard_front":
        grounding_props = gymapi.CameraProperties()
        grounding_props.width = int(grounding_width or topdown_width)
        grounding_props.height = int(grounding_height or topdown_height)
        grounding_props.horizontal_fov = float(grounding_fov_deg)
        grounding_handle = env.gym.create_camera_sensor(env_handle, grounding_props)
        metadata = task_data.get("metadata", {}) if isinstance(task_data.get("metadata"), dict) else {}
        center = metadata.get("whiteboard_center_world") or [0.38, 0.0, 0.505]
        normal = metadata.get("public_whiteboard_surface_normal_world") or [-1.0, 0.0, 0.0]
        target = [
            float(center[0]),
            float(center[1]),
            float(center[2]) + 0.12,
        ]
        distance_m = 0.95
        grounding_pos = gymapi.Vec3(
            target[0] + float(normal[0]) * distance_m,
            target[1] + float(normal[1]) * distance_m,
            target[2] + float(normal[2]) * distance_m,
        )
        grounding_target = gymapi.Vec3(*target)
        env.gym.set_camera_location(
            grounding_handle,
            env_handle,
            grounding_pos,
            grounding_target,
        )
        grounding_frame = {
            "name": "whiteboard_front_vertical",
            "image_up": "world_z_up_on_vertical_whiteboard",
            "image_down": "world_z_down_on_vertical_whiteboard",
            "image_right": "approximately_along_whiteboard_width",
            "action_direction_rule": (
                "VLM arrows are interpreted in this calibrated front-facing "
                "whiteboard image. Points and waypoints should lie on the visible "
                "vertical whiteboard surface; up/down correspond to vertical "
                "writing motion on the board."
            ),
        }
        grounding = _capture_camera(
            env=env,
            camera_handle=grounding_handle,
            camera_props=grounding_props,
            env_handle=env_handle,
            camera_id="sim_grounding_whiteboard_front",
            output_dir=output_dir,
            position_world=[
                grounding_pos.x,
                grounding_pos.y,
                grounding_pos.z,
            ],
            look_at_world=[
                grounding_target.x,
                grounding_target.y,
                grounding_target.z,
            ],
            frame_convention=grounding_frame,
        )
    elif grounding_camera != "topdown":
        raise ValueError(
            f"unsupported grounding_camera {grounding_camera!r}; "
            "expected 'topdown', 'slight_oblique', or 'whiteboard_front'"
        )

    presentation = None
    if include_presentation_camera:
        presentation_props = gymapi.CameraProperties()
        presentation_props.width = int(presentation_width)
        presentation_props.height = int(presentation_height)
        presentation_props.horizontal_fov = float(presentation_fov_deg)
        presentation_handle = env.gym.create_camera_sensor(env_handle, presentation_props)
        presentation_pos = gymapi.Vec3(0.36, -0.58, 0.78)
        presentation_target = gymapi.Vec3(0.08, -0.04, 0.53)
        env.gym.set_camera_location(
            presentation_handle,
            env_handle,
            presentation_pos,
            presentation_target,
        )
        presentation = _capture_camera(
            env=env,
            camera_handle=presentation_handle,
            camera_props=presentation_props,
            env_handle=env_handle,
            camera_id="sim_presentation",
            output_dir=output_dir,
            position_world=[
                presentation_pos.x,
                presentation_pos.y,
                presentation_pos.z,
            ],
            look_at_world=[
                presentation_target.x,
                presentation_target.y,
                presentation_target.z,
            ],
        )

    items = [
        {"kind": "scene_rgb", "id": "sim_rollout_rgb", "path": rollout["rgb_path"]},
        {"kind": "scene_depth", "id": "sim_rollout_depth", "path": rollout["depth_path"]},
        {"kind": "depth_visualization", "id": "sim_rollout_depth_vis", "path": rollout["depth_vis_path"]},
        {"kind": "camera", "id": "sim_rollout_camera", "path": rollout["camera_path"]},
        {"kind": "scene_rgb", "id": "sim_topdown_rgb", "path": topdown["rgb_path"]},
        {"kind": "scene_depth", "id": "sim_topdown_depth", "path": topdown["depth_path"]},
        {"kind": "depth_visualization", "id": "sim_topdown_depth_vis", "path": topdown["depth_vis_path"]},
        {"kind": "camera", "id": "sim_topdown_camera", "path": topdown["camera_path"]},
    ]
    if grounding_camera in {"slight_oblique", "whiteboard_front"}:
        grounding_item_id = (
            "sim_grounding_oblique"
            if grounding_camera == "slight_oblique"
            else "sim_grounding_whiteboard_front"
        )
        items.extend(
            [
                {
                    "kind": "scene_rgb",
                    "id": f"{grounding_item_id}_rgb",
                    "path": grounding["rgb_path"],
                },
                {
                    "kind": "scene_depth",
                    "id": f"{grounding_item_id}_depth",
                    "path": grounding["depth_path"],
                },
                {
                    "kind": "depth_visualization",
                    "id": f"{grounding_item_id}_depth_vis",
                    "path": grounding["depth_vis_path"],
                },
                {
                    "kind": "camera",
                    "id": f"{grounding_item_id}_camera",
                    "path": grounding["camera_path"],
                },
            ]
        )
    if presentation is not None:
        items.extend(
            [
                {
                    "kind": "scene_rgb",
                    "id": "sim_presentation_rgb",
                    "path": presentation["rgb_path"],
                },
                {
                    "kind": "scene_depth",
                    "id": "sim_presentation_depth",
                    "path": presentation["depth_path"],
                },
                {
                    "kind": "depth_visualization",
                    "id": "sim_presentation_depth_vis",
                    "path": presentation["depth_vis_path"],
                },
                {
                    "kind": "camera",
                    "id": "sim_presentation_camera",
                    "path": presentation["camera_path"],
                },
            ]
        )

    camera_order = [
        "sim_rollout: original SimToolReal rollout inspection RGB-D camera",
        "sim_topdown: robot-forward overhead RGB-D camera for inspection",
    ]
    if grounding_camera == "slight_oblique":
        camera_order.append(
            "sim_grounding_oblique: slight-oblique calibrated RGB-D camera for VLM keypoint coordinates"
        )
    elif grounding_camera == "whiteboard_front":
        camera_order.append(
            "sim_grounding_whiteboard_front: front-facing vertical whiteboard RGB-D camera for VLM keypoint coordinates"
        )
    else:
        camera_order.append(
            "sim_topdown: robot-forward overhead RGB-D camera for VLM keypoint coordinates"
        )
    if presentation is not None:
        camera_order.append(
            "sim_presentation: oblique paper-style camera for inspection/video context"
        )

    capture_stats = {
        "sim_rollout": rollout,
        "sim_topdown": topdown,
    }
    if grounding_camera == "slight_oblique":
        capture_stats["sim_grounding_oblique"] = grounding
    if grounding_camera == "whiteboard_front":
        capture_stats["sim_grounding_whiteboard_front"] = grounding
    if presentation is not None:
        capture_stats["sim_presentation"] = presentation

    prompt_images = [rollout["rgb_path"], grounding["rgb_path"]]
    grounding_image_index = 1
    if presentation is not None:
        prompt_images.append(presentation["rgb_path"])

    evidence = {
        "schema_version": "toolframe_scene_evidence_v0",
        "task_id": task_data.get("task_id"),
        "instruction": task_data.get("instruction"),
        "items": items,
        "metadata": {
            "prompt_images": prompt_images,
            "presentation_images": (
                [presentation["rgb_path"]] if presentation is not None else []
            ),
            "grounding_image_index": grounding_image_index,
            "grounding_camera_id": grounding["camera_id"],
            "render_backend": "simtoolreal_isaacgym_rgbd",
            "device": resolved_device,
            "headless": bool(headless),
            "camera_order": camera_order,
            "grounding_frame_convention": grounding_frame,
            "grounding_rgb_style": "original_simtoolreal_rgb",
            "goal_object_visual_hidden": True,
            "public_scene_note": (
                "SimToolReal green goal/subgoal visual is hidden for public "
                "VLM scene evidence; VLM only sees the real scene, task "
                "objects, and benchmark environment actors."
            ),
            "lighting": lighting,
            "public_contract": (
                "VLM must use the configured grounding image and output pixel "
                "coordinates in that image. The grounding image frame convention "
                "is recorded in metadata.grounding_frame_convention."
            ),
            "public_scene_context": public_scene_context,
            "simtoolreal_log": str(log_path),
            "capture_stats": capture_stats,
        },
    }
    evidence_path = output_dir / "scene_evidence.json"
    _write_json(evidence_path, evidence)
    return {"scene_evidence": str(evidence_path), **evidence["metadata"]["capture_stats"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Tool-use task JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="SimToolReal policy config YAML.")
    parser.add_argument("--device", default=None)
    parser.set_defaults(headless=True)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--topdown-width", type=int, default=900)
    parser.add_argument("--topdown-height", type=int, default=700)
    parser.add_argument("--topdown-fov-deg", type=float, default=55.0)
    parser.add_argument(
        "--grounding-camera",
        choices=("topdown", "slight_oblique", "whiteboard_front"),
        default="topdown",
        help="RGB-D camera used as the VLM grounding prompt image.",
    )
    parser.add_argument("--grounding-width", type=int)
    parser.add_argument("--grounding-height", type=int)
    parser.add_argument("--grounding-fov-deg", type=float, default=52.0)
    parser.add_argument("--rollout-width", type=int, default=900)
    parser.add_argument("--rollout-height", type=int, default=506)
    parser.add_argument("--rollout-fov-deg", type=float, default=70.0)
    parser.add_argument(
        "--presentation-camera",
        dest="include_presentation_camera",
        action="store_true",
        help=(
            "Also save an oblique paper-style RGB-D camera view for inspection. "
            "This is not used as the grounding image."
        ),
    )
    parser.set_defaults(include_presentation_camera=False)
    parser.add_argument("--presentation-width", type=int, default=1100)
    parser.add_argument("--presentation-height", type=int, default=620)
    parser.add_argument("--presentation-fov-deg", type=float, default=50.0)
    parser.add_argument(
        "--light-preset",
        choices=("soft", "native"),
        default=DEFAULT_RENDER_LIGHT_PRESET,
        help=(
            "Render-only lighting preset. The default soft preset softens "
            "IsaacGym light indices 0 and 1."
        ),
    )
    parser.add_argument(
        "--soft-light-indices",
        default=",".join(str(item) for item in DEFAULT_SOFT_LIGHT_INDICES),
        help=(
            "Comma-separated IsaacGym light indices used by --light-preset soft. "
            "The canonical side-board evidence lighting is 0,2."
        ),
    )
    args = parser.parse_args()

    result = render_simtoolreal_rgbd_scene_evidence(
        task_path=Path(args.task),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        device=args.device,
        headless=bool(args.headless),
        topdown_width=int(args.topdown_width),
        topdown_height=int(args.topdown_height),
        topdown_fov_deg=float(args.topdown_fov_deg),
        grounding_camera=str(args.grounding_camera),
        grounding_width=args.grounding_width,
        grounding_height=args.grounding_height,
        grounding_fov_deg=float(args.grounding_fov_deg),
        rollout_width=int(args.rollout_width),
        rollout_height=int(args.rollout_height),
        rollout_fov_deg=float(args.rollout_fov_deg),
        include_presentation_camera=bool(args.include_presentation_camera),
        presentation_width=int(args.presentation_width),
        presentation_height=int(args.presentation_height),
        presentation_fov_deg=float(args.presentation_fov_deg),
        light_preset=str(args.light_preset),
        soft_light_indices=tuple(
            int(item.strip())
            for item in str(args.soft_light_indices).split(",")
            if item.strip()
        ),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
