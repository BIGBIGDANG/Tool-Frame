"""2D scene keypoint lifting for image-grounded tool planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel units."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "CameraIntrinsics":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )


@dataclass(frozen=True)
class ImageKeypoint:
    """A VLM-marked 2D point in a calibrated scene image."""

    keypoint_id: str
    u: float
    v: float
    role: str
    confidence: float | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LiftedKeypoint:
    """A lifted world-frame keypoint with traceable image evidence."""

    keypoint_id: str
    role: str
    pixel_uv: tuple[float, float]
    depth_m: float
    position_camera: tuple[float, float, float]
    position_world: tuple[float, float, float]
    confidence: float | None = None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _as_homogeneous_transform(camera_to_world: Sequence[Sequence[float]]) -> np.ndarray:
    transform = np.asarray(camera_to_world, dtype=float)
    if transform.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=float)
        transform = np.vstack([transform, bottom])
    if transform.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 4x4 or 3x4, got {transform.shape}")
    return transform


def unproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Unproject a pixel/depth sample into the camera frame."""

    if depth_m <= 0.0:
        raise ValueError(f"depth must be positive, got {depth_m}")
    if not (0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height):
        raise ValueError(
            f"pixel ({u:.2f}, {v:.2f}) outside image "
            f"{intrinsics.width}x{intrinsics.height}"
        )
    x = (float(u) - intrinsics.cx) * float(depth_m) / intrinsics.fx
    y = (float(v) - intrinsics.cy) * float(depth_m) / intrinsics.fy
    z = float(depth_m)
    return np.array([x, y, z], dtype=float)


def transform_point(transform: Sequence[Sequence[float]], point: Iterable[float]) -> np.ndarray:
    """Apply a homogeneous transform to a 3D point."""

    matrix = _as_homogeneous_transform(transform)
    xyz = np.asarray(tuple(point), dtype=float)
    if xyz.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {xyz.shape}")
    result = matrix @ np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    return result[:3]


def lift_keypoint(
    keypoint: ImageKeypoint,
    *,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_to_world: Sequence[Sequence[float]],
) -> LiftedKeypoint:
    """Lift one VLM 2D keypoint into world coordinates."""

    camera_xyz = unproject_pixel(keypoint.u, keypoint.v, depth_m, intrinsics)
    world_xyz = transform_point(camera_to_world, camera_xyz)
    return LiftedKeypoint(
        keypoint_id=keypoint.keypoint_id,
        role=keypoint.role,
        pixel_uv=(float(keypoint.u), float(keypoint.v)),
        depth_m=float(depth_m),
        position_camera=tuple(float(v) for v in camera_xyz),
        position_world=tuple(float(v) for v in world_xyz),
        confidence=keypoint.confidence,
    )


def derive_axis(
    start_world: Sequence[float],
    end_world: Sequence[float],
    *,
    min_norm: float = 1e-5,
) -> tuple[float, float, float]:
    """Return a unit direction from start to end in world coordinates."""

    start = np.asarray(start_world, dtype=float)
    end = np.asarray(end_world, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start_world and end_world must be 3D vectors")
    delta = end - start
    norm = float(np.linalg.norm(delta))
    if norm < min_norm:
        raise ValueError(f"axis endpoints are too close: norm={norm:.6g}")
    axis = delta / norm
    return tuple(float(v) for v in axis)

