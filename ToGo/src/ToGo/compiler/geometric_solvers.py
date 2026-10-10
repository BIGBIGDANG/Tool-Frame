#!/usr/bin/env python3
"""Low-level geometric solvers for whole-tool subgoal optimization.

These solvers do not produce hand, gripper, or robot poses. They only solve
functional-part goals that the outer compiler converts into whole-tool object
poses for SimToolReal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionalWaypoint:
    """A desired functional-part waypoint in world coordinates."""

    name: str
    functional_position: list[float]
    orientation_xyzw: list[float]
    duration_sec: float


def _normalize_quat(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in q))
    if norm == 0:
        raise ValueError("zero quaternion")
    return [float(x) / norm for x in q]


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in v))
    if norm == 0:
        raise ValueError("zero vector")
    return [float(x) / norm for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _quat_from_axis_angle(axis: list[float], angle_deg: float) -> list[float]:
    unit = _unit(axis)
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return _normalize_quat([unit[0] * s, unit[1] * s, unit[2] * s, math.cos(half)])


def quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return _normalize_quat(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate(q_xyzw: list[float], v: list[float]) -> list[float]:
    x, y, z, w = _normalize_quat(q_xyzw)
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def quat_between_vectors(source: list[float], target: list[float]) -> list[float]:
    """Return the minimum rotation that aligns source to target."""

    a = _unit(source)
    b = _unit(target)
    dot = max(-1.0, min(1.0, _dot(a, b)))
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -0.999999:
        fallback = [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]
        return _quat_from_axis_angle(_unit(_cross(a, fallback)), 180.0)
    axis = _cross(a, b)
    return _normalize_quat([axis[0], axis[1], axis[2], 1.0 + dot])


def solve_point_contact_waypoints(
    *,
    functional_region: dict[str, Any],
    target_point: list[float],
    target_axis: list[float],
    current_orientation_xyzw: list[float],
    precontact_distance: float,
    local_normal_tool: list[float] | None = None,
    precontact_direction: str = "opposite_axis",
    swing_angle_deg: float = 0.0,
    pivot_origin_tool: list[float] | None = None,
    alignment_threshold: float = 0.85,
) -> list[FunctionalWaypoint]:
    """Solve contact waypoints for a point/patch functional part.

    Hard constraints:
    - functional frame origin reaches target_point;
    - functional frame z-axis/surface normal aligns to target_axis.

    Soft policy:
    - choose the minimum rotation from current orientation that satisfies the
      normal alignment, leaving tangential roll free instead of hard-aligning a
      full arbitrary frame.
    """

    frame = functional_region["frame"]
    local_normal = list(local_normal_tool or frame.get("z_axis", [0, 0, -1]))
    current_normal = _unit(quat_rotate(current_orientation_xyzw, local_normal))
    axis = _unit(target_axis)
    delta = quat_between_vectors(current_normal, axis)
    goal_orientation = quat_mul(delta, current_orientation_xyzw)
    solved_normal = _unit(quat_rotate(goal_orientation, local_normal))
    if _dot(solved_normal, axis) < alignment_threshold:
        raise ValueError("point surface normal alignment failed")
    if swing_angle_deg < 0.0 or swing_angle_deg > 75.0:
        raise ValueError("point contact swing_angle_deg must be in [0, 75]")
    if precontact_direction not in {"opposite_axis", "along_axis"}:
        raise ValueError("precontact_direction must be opposite_axis or along_axis")

    point = [float(x) for x in target_point]
    precontact_sign = -1.0 if precontact_direction == "opposite_axis" else 1.0
    if pivot_origin_tool is not None and float(swing_angle_deg) > 0.0:
        functional_origin_tool = list(frame["origin"])
        pivot_origin_tool = [float(x) for x in pivot_origin_tool]
        contact_tool_position = _sub(
            point,
            quat_rotate(goal_orientation, functional_origin_tool),
        )
        pivot_world = _add(
            contact_tool_position,
            quat_rotate(goal_orientation, pivot_origin_tool),
        )
        lever_world = _sub(point, pivot_world)
        hinge_axis = _cross(lever_world, axis)
        if sum(value * value for value in hinge_axis) <= 1e-10:
            hinge_axis = quat_rotate(
                goal_orientation,
                list(frame.get("y_axis", [1, 0, 0])),
            )
        swing_axis_world = _unit(hinge_axis)
        backswing = _quat_from_axis_angle(swing_axis_world, -float(swing_angle_deg))
        precontact_orientation = quat_mul(backswing, goal_orientation)
        precontact = [
            point[i] + axis[i] * float(precontact_distance) * precontact_sign
            for i in range(3)
        ]
    else:
        swing_axis_world = _unit(
            quat_rotate(
                goal_orientation,
                list(frame.get("y_axis", [1, 0, 0])),
            )
        )
        backswing = _quat_from_axis_angle(swing_axis_world, -float(swing_angle_deg))
        precontact_orientation = quat_mul(backswing, goal_orientation)
        precontact = [
            point[i] + axis[i] * float(precontact_distance) * precontact_sign
            for i in range(3)
        ]
    return [
        FunctionalWaypoint(
            name="pre_contact",
            functional_position=precontact,
            orientation_xyzw=precontact_orientation,
            duration_sec=0.5,
        ),
        FunctionalWaypoint(
            name="contact",
            functional_position=point,
            orientation_xyzw=goal_orientation,
            duration_sec=0.25,
        ),
    ]
