#!/usr/bin/env python3
"""Render one calibrated RGB-D observation for the shared grounding pipeline."""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[5]
for path in (ROOT / "ToGo/src", ROOT / "simtoolreal", ROOT / "simtoolreal/rl_games",
             Path(os.environ.get("TOGO_ISAACGYM", str(ROOT / "isaacgym/python")))):
    sys.path.insert(0, str(path))
from ToGo.execution.runtime import (_install_numpy_legacy_aliases,
                                   DEFAULT_CONFIG, apply_simtoolreal_render_lighting)
from ToGo.execution.simtoolreal_rollout import environment_overrides
from ToGo.pipeline.handoff import read_json, write_json

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


def render(task: dict[str, Any], output: Path, config: Path, device: str,
           settings: dict[str, Any]) -> dict[str, Any]:
    if not config.is_file():
        raise FileNotFoundError(f"missing policy config: {config}; set TOGO_PRETRAINED_POLICY_ROOT")
    _install_numpy_legacy_aliases()
    from deployment.isaac.isaac_env import create_env
    from isaacgym import gymapi
    pose = task["execution"]["simtoolreal_start_pose"]
    overrides = environment_overrides(task, {"start_pose": pose, "goals": [pose]}, settings)
    env = create_env(config_path=str(config), headless=True, device=device,
                     episode_length=int(settings["max_steps"]) + 5, overrides=overrides)
    try:
        # GPU rigid-body transforms are not ready for cameras until the first
        # physics update. Rendering earlier exposes stale actor transforms.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        apply_simtoolreal_render_lighting(env, "soft")
        camera = task.get("camera", {"position_world": [0.0, -0.2, 1.7],
                                   "look_at_world": [0.0, -0.04, 0.53],
                                   "width": 900, "height": 700, "horizontal_fov_deg": 55.0})
        props = gymapi.CameraProperties()
        props.width = int(camera.get("width", 900))
        props.height = int(camera.get("height", 700))
        props.horizontal_fov = float(camera.get("horizontal_fov_deg", 55.0))
        props.enable_tensors = False
        position, target = camera["position_world"], camera["look_at_world"]
        handle = env.gym.create_camera_sensor(env.envs[0], props)
        if handle < 0:
            raise RuntimeError("IsaacGym could not create an RGB-D camera")
        env.gym.set_camera_location(handle, env.envs[0], gymapi.Vec3(*position), gymapi.Vec3(*target))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgb = _rgba_to_rgb(env.gym.get_camera_image(env.sim, env.envs[0], handle, gymapi.IMAGE_COLOR),
                          width=props.width, height=props.height)
        depth = _depth_to_positive_m(env.gym.get_camera_image(env.sim, env.envs[0], handle, gymapi.IMAGE_DEPTH),
                                     width=props.width, height=props.height)
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output / "scene_rgb.png")
        np.save(output / "scene_depth.npy", depth)
        calibration = {**_intrinsics_from_fov(width=props.width, height=props.height,
                                               horizontal_fov_deg=props.horizontal_fov),
                       "camera_to_world": _camera_to_world_from_look_at(position=position, target=target),
                       "depth_encoding": "positive_camera_z_meters"}
        write_json(output / "scene_camera.json", calibration)
        plane = task.get("interaction_plane")
        normal = plane.get("normal_world") if plane else None
        context = {"default_axis_candidates": [{"id": "support_surface_normal", "vector_world": normal},
                    {"id": "opposite_support_surface_normal", "vector_world": [-float(x) for x in normal]}]} if normal else {}
        evidence = {"schema_version": "toolframe_scene_evidence_v0", "task_id": task["task_id"],
                    "instruction": task["instruction"], "items": [
                    {"id": "scene_rgb", "kind": "scene_rgb", "path": "scene_rgb.png"},
                    {"id": "scene_depth", "kind": "scene_depth", "path": "scene_depth.npy"},
                    {"id": "scene_camera", "kind": "camera", "path": "scene_camera.json"}],
                    "metadata": {"prompt_images": ["scene_rgb.png"], "grounding_image_index": 0,
                                 "public_scene_context": context}}
        write_json(output / "scene_evidence.json", evidence)
        return evidence
    finally:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    render(read_json(args.task), args.output_dir.resolve(), args.config, args.device, read_json(args.settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
