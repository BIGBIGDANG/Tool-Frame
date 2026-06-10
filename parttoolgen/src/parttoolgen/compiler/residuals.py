#!/usr/bin/env python3
"""Numerical residual helpers for PartToolGen tool-pose optimization.

ProgramIR stays symbolic. This module is the first numerical layer: it knows
how to evaluate point, axis, clearance, and smoothness terms for candidate
whole-tool poses.
"""

from __future__ import annotations

import math
from typing import Any

from parttoolgen.compiler.geometric_solvers import quat_rotate


class ResidualError(ValueError):
    """Raised when a residual cannot be evaluated."""


def add(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def mul(v: list[float], scalar: float) -> list[float]:
    return [float(x) * float(scalar) for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def unit(v: list[float]) -> list[float]:
    magnitude = norm(v)
    if magnitude <= 1e-12:
        raise ResidualError("zero vector")
    return [float(x) / magnitude for x in v]


def normalize_quat(q: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(float(x) * float(x) for x in q))
    if magnitude <= 1e-12:
        raise ResidualError("zero quaternion")
    return [float(x) / magnitude for x in q]


def quat_angle_rad(q1: list[float], q2: list[float]) -> float:
    alignment = abs(sum(float(a) * float(b) for a, b in zip(normalize_quat(q1), normalize_quat(q2))))
    alignment = min(1.0, max(-1.0, alignment))
    return 2.0 * math.acos(alignment)


def quat_slerp(q1: list[float], q2: list[float], t: float) -> list[float]:
    a = normalize_quat(q1)
    b = normalize_quat(q2)
    alignment = sum(x * y for x, y in zip(a, b))
    if alignment < 0.0:
        b = [-x for x in b]
        alignment = -alignment
    alignment = min(1.0, max(-1.0, alignment))
    if alignment > 0.9995:
        return normalize_quat([a[i] + t * (b[i] - a[i]) for i in range(4)])
    theta_0 = math.acos(alignment)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - alignment * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return normalize_quat([s0 * a[i] + s1 * b[i] for i in range(4)])


def pose_point(pose: dict[str, Any], point_tool: list[float]) -> list[float]:
    return add(pose["position"], quat_rotate(pose["orientation_xyzw"], point_tool))


def pose_axis(pose: dict[str, Any], axis_tool: list[float]) -> list[float]:
    return unit(quat_rotate(pose["orientation_xyzw"], axis_tool))


def pose_for_tool_point_goal(
    *,
    orientation_xyzw: list[float],
    point_tool: list[float],
    point_world: list[float],
) -> dict[str, Any]:
    return {
        "position": sub(point_world, quat_rotate(orientation_xyzw, point_tool)),
        "orientation_xyzw": normalize_quat(orientation_xyzw),
    }


def weighted_position_average(candidates: list[tuple[list[float], float]]) -> list[float]:
    total = sum(max(float(weight), 0.0) for _, weight in candidates)
    if total <= 1e-12:
        total = float(len(candidates))
        weights = [1.0 for _ in candidates]
    else:
        weights = [max(float(weight), 0.0) for _, weight in candidates]
    result = [0.0, 0.0, 0.0]
    for position, weight in zip((item[0] for item in candidates), weights):
        for i in range(3):
            result[i] += float(position[i]) * float(weight) / total
    return result


def point_tracking_error_m(pose: dict[str, Any], point_tool: list[float], point_world: list[float]) -> float:
    return norm(sub(pose_point(pose, point_tool), point_world))


def axis_alignment_cost(
    pose: dict[str, Any],
    axis_tool: list[float],
    axis_world: list[float],
    *,
    polarity: str,
) -> float:
    predicted = pose_axis(pose, axis_tool)
    target = unit(axis_world)
    if polarity == "opposite":
        return 1.0 + dot(predicted, target)
    if polarity == "auto":
        return 1.0 - dot(predicted, target)
    return 1.0 - dot(predicted, target)


def clearance_margin_m(
    pose: dict[str, Any],
    point_tool: list[float],
    obstacle_world: list[float],
    min_distance_m: float,
) -> float:
    distance = norm(sub(pose_point(pose, point_tool), obstacle_world))
    return distance - float(min_distance_m)


def stage_cost(
    *,
    pose: dict[str, Any],
    point_targets: list[Any],
    axis_targets: list[Any],
    clearance_targets: list[Any],
) -> float:
    cost = 0.0
    for target in point_targets:
        error = point_tracking_error_m(pose, target.point_tool, target.point_world)
        cost += float(target.weight) * error * error
    for target in axis_targets:
        error = axis_alignment_cost(
            pose,
            target.vector_tool,
            target.vector_world,
            polarity=target.polarity,
        )
        cost += float(target.weight) * error * error
    for target in clearance_targets:
        margin = clearance_margin_m(
            pose,
            target.point_tool,
            target.obstacle_world,
            target.min_distance_m,
        )
        if margin < 0.0:
            cost += float(target.weight) * margin * margin
    return cost
