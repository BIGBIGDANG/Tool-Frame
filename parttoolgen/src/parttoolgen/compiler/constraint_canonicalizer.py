#!/usr/bin/env python3
"""Canonicalize ProgramIR constraints into numerical solver inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from parttoolgen.compiler.reference_resolver import ReferenceResolver
from parttoolgen.compiler.residuals import add, mul, norm, pose_point, sub, unit


class ConstraintCanonicalizationError(ValueError):
    """Raised when a ProgramIR constraint cannot be canonicalized."""


@dataclass(frozen=True)
class PointTarget:
    source_ref: str
    target_ref: str
    point_tool: list[float]
    point_world: list[float]
    weight: float
    constraint_id: str
    hard: bool = False
    tolerance_m: float = 0.0
    pre_contact_target_world: list[float] | None = None
    pre_contact_axis_ref: str | None = None
    pre_contact_axis_world: list[float] | None = None
    pre_contact_direction: str | None = None
    pre_contact_distance_m: float | None = None
    allows_contact_penetration: bool = False
    point_tool_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AxisTarget:
    source_ref: str
    target_ref: str
    vector_tool: list[float]
    vector_world: list[float]
    polarity: str
    weight: float
    constraint_id: str
    hard: bool = False
    min_cosine: float = -1.0
    vector_tool_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ClearanceTarget:
    moving_ref: str
    obstacle_ref: str
    point_tool: list[float]
    obstacle_world: list[float]
    min_distance_m: float
    weight: float
    constraint_id: str
    hard: bool = False


@dataclass(frozen=True)
class MoveAlongTarget:
    source_ref: str
    start_ref: str
    goal_ref: str
    point_tool: list[float]
    start_world: list[float]
    goal_world: list[float]
    weight: float
    constraint_id: str
    hard: bool = False
    point_tool_candidates: list[dict[str, Any]] = field(default_factory=list)
    profile: str = "linear"
    path_family: str = "contact_follow"
    swing_axis_ref: str | None = None
    swing_axis_world: list[float] | None = None
    swing_axis_tool: list[float] | None = None
    swing_angle_rad: float = 0.0
    swing_lift_m: float = 0.03


@dataclass(frozen=True)
class ToolOccupancyClearanceTarget:
    tool_part_ref: str
    obstacle_ref: str
    sample_points_tool: list[list[float]]
    obstacle_world: list[float] | None
    min_distance_m: float
    samples: str
    weight: float
    constraint_id: str
    hard: bool = False


@dataclass(frozen=True)
class CanonicalStageConstraints:
    stage_name: str
    source_ids: list[str]
    point_targets: list[PointTarget]
    axis_targets: list[AxisTarget]
    clearance_targets: list[ClearanceTarget]
    move_along_targets: list[MoveAlongTarget]
    tool_occupancy_clearance_targets: list[ToolOccupancyClearanceTarget] = field(default_factory=list)
    orientation_sample_points_tool: list[list[float]] = field(default_factory=list)


def _constraint_weight(constraint: dict[str, Any]) -> float:
    return float(constraint.get("weight", 1.0))


def _hard(constraint: dict[str, Any]) -> bool:
    return bool(constraint.get("hard", False))


def _point_target_from_constraint(
    *,
    constraint: dict[str, Any],
    resolver: ReferenceResolver,
    current_pose: dict[str, Any],
) -> PointTarget | None:
    ctype = constraint["type"]
    refs = constraint["refs"]
    weight = _constraint_weight(constraint)
    if ctype == "contact":
        source = resolver.tool_point(refs["source"])
        target = resolver.world_point(refs["target"])
        return PointTarget(
            source_ref=source.ref,
            target_ref=target.ref,
            point_tool=source.point_tool or [],
            point_tool_candidates=source.point_tool_candidates or [],
            point_world=target.point_world or [],
            weight=weight,
            constraint_id=constraint["id"],
            hard=_hard(constraint),
            tolerance_m=float(constraint["params"]["tolerance_m"]),
            allows_contact_penetration=True,
        )
    if ctype == "pre_contact_offset":
        source = resolver.tool_point(refs["source"])
        target = resolver.world_point(refs["target"])
        axis = resolver.world_vector(refs["axis"])
        direction = str(constraint["params"].get("direction", "opposite_axis"))
        distance_m = float(constraint["params"]["distance_m"])
        sign = -1.0 if direction == "opposite_axis" else 1.0
        offset = mul(axis.vector_world or [], distance_m * sign)
        return PointTarget(
            source_ref=source.ref,
            target_ref=target.ref,
            point_tool=source.point_tool or [],
            point_tool_candidates=source.point_tool_candidates or [],
            point_world=add(target.point_world or [], offset),
            weight=weight,
            constraint_id=constraint["id"],
            hard=_hard(constraint),
            tolerance_m=0.0,
            pre_contact_target_world=target.point_world or [],
            pre_contact_axis_ref=axis.ref,
            pre_contact_axis_world=axis.vector_world or [],
            pre_contact_direction=direction,
            pre_contact_distance_m=distance_m,
        )
    if ctype == "distance":
        a = resolver.point(refs["a"])
        b = resolver.point(refs["b"])
        if a.point_tool is not None and b.point_world is not None:
            source, target = a, b
        elif b.point_tool is not None and a.point_world is not None:
            source, target = b, a
        else:
            raise ConstraintCanonicalizationError(
                f"{constraint['id']} distance needs one tool point and one world point"
            )
        current = pose_point(current_pose, source.point_tool or [])
        direction = sub(current, target.point_world or [])
        if norm(direction) <= 1e-9:
            direction = [0.0, 0.0, 1.0]
        goal = add(target.point_world or [], mul(unit(direction), float(constraint["params"]["target_m"])))
        return PointTarget(
            source_ref=source.ref,
            target_ref=target.ref,
            point_tool=source.point_tool or [],
            point_tool_candidates=source.point_tool_candidates or [],
            point_world=goal,
            weight=weight,
            constraint_id=constraint["id"],
            hard=_hard(constraint),
            tolerance_m=float(constraint["params"]["tolerance_m"]),
        )
    return None


def _axis_target_from_constraint(
    *,
    constraint: dict[str, Any],
    resolver: ReferenceResolver,
) -> AxisTarget | None:
    if constraint["type"] == "align_vector_between_points":
        refs = constraint["refs"]
        source_start = resolver.tool_point(refs["source_start"])
        source_end = resolver.tool_point(refs["source_end"])
        target = resolver.world_vector(refs["target_axis"])
        vector_tool = sub(source_end.point_tool or [], source_start.point_tool or [])
        if len(vector_tool) != 3 or norm(vector_tool) <= 1e-9:
            raise ConstraintCanonicalizationError(
                f"{constraint['id']} align_vector_between_points needs distinct tool points"
            )
        return AxisTarget(
            source_ref=f"{source_start.ref}->{source_end.ref}",
            target_ref=target.ref,
            vector_tool=unit(vector_tool),
            vector_world=target.vector_world or [],
            vector_tool_candidates=[],
            polarity=str(constraint["params"].get("polarity", "same")),
            weight=_constraint_weight(constraint),
            constraint_id=constraint["id"],
            hard=_hard(constraint),
            min_cosine=float(constraint["params"]["min_cosine"]),
        )
    if constraint["type"] != "align_axis":
        return None
    refs = constraint["refs"]
    source = resolver.tool_vector(refs["source_axis"])
    target = resolver.world_vector(refs["target_axis"])
    return AxisTarget(
        source_ref=source.ref,
        target_ref=target.ref,
        vector_tool=source.vector_tool or [],
        vector_world=target.vector_world or [],
        vector_tool_candidates=source.vector_tool_candidates or [],
        polarity=str(constraint["params"].get("polarity", "same")),
        weight=_constraint_weight(constraint),
        constraint_id=constraint["id"],
        hard=_hard(constraint),
        min_cosine=float(constraint["params"]["min_cosine"]),
    )


def _clearance_from_constraint(
    *,
    constraint: dict[str, Any],
    resolver: ReferenceResolver,
) -> ClearanceTarget | None:
    if constraint["type"] != "clearance":
        return None
    refs = constraint["refs"]
    moving = resolver.tool_point(refs["moving"])
    obstacle = resolver.world_point(refs["obstacle"])
    return ClearanceTarget(
        moving_ref=moving.ref,
        obstacle_ref=obstacle.ref,
        point_tool=moving.point_tool or [],
        obstacle_world=obstacle.point_world or [],
        min_distance_m=float(constraint["params"]["min_distance_m"]),
        weight=_constraint_weight(constraint),
        constraint_id=constraint["id"],
        hard=_hard(constraint),
    )


def _move_along_from_constraint(
    *,
    constraint: dict[str, Any],
    resolver: ReferenceResolver,
) -> MoveAlongTarget | None:
    if constraint["type"] != "move_along":
        return None
    refs = constraint["refs"]
    source = resolver.tool_point(refs["source"])
    start = resolver.world_point(refs["start"])
    goal = resolver.world_point(refs["end"])
    params = constraint.get("params", {})
    profile = str(params.get("profile", "linear"))
    if profile not in {"linear", "swing_arc"}:
        raise ConstraintCanonicalizationError(
            f"{constraint['id']} move_along profile must be linear or swing_arc"
        )
    path_family = str(
        params.get("path_family") or ("swing" if profile == "swing_arc" else "contact_follow")
    )
    if path_family not in {"contact_follow", "slide", "swing"}:
        raise ConstraintCanonicalizationError(
            f"{constraint['id']} move_along path_family must be contact_follow, slide, or swing"
        )
    swing_axis_ref = refs.get("swing_axis")
    swing_axis_world = None
    swing_axis_tool = None
    swing_angle_rad = 0.0
    swing_lift_m = 0.0
    if profile == "swing_arc":
        if not isinstance(swing_axis_ref, str):
            raise ConstraintCanonicalizationError(
                f"{constraint['id']} swing_arc move_along requires swing_axis"
            )
        swing_axis = resolver.vector(swing_axis_ref)
        swing_axis_world = swing_axis.vector_world
        swing_axis_tool = swing_axis.vector_tool
        swing_angle_rad = float(params.get("swing_angle_rad", 0.55))
        swing_lift_m = float(params.get("swing_lift_m", 0.03))
    return MoveAlongTarget(
        source_ref=source.ref,
        start_ref=start.ref,
        goal_ref=goal.ref,
        point_tool=source.point_tool or [],
        point_tool_candidates=source.point_tool_candidates or [],
        start_world=start.point_world or [],
        goal_world=goal.point_world or [],
        weight=_constraint_weight(constraint),
        constraint_id=constraint["id"],
        hard=_hard(constraint),
        profile=profile,
        path_family=path_family,
        swing_axis_ref=swing_axis_ref,
        swing_axis_world=swing_axis_world,
        swing_axis_tool=swing_axis_tool,
        swing_angle_rad=swing_angle_rad,
        swing_lift_m=swing_lift_m,
    )


def _add_scaled(base: list[float], axes: list[list[float]], scales: list[float]) -> list[float]:
    result = list(base)
    for axis, scale in zip(axes, scales):
        result = add(result, mul(axis, scale))
    return result


def _tool_part_sample_points(
    *,
    resolved_entity: dict[str, Any],
    sample_mode: str,
) -> list[list[float]]:
    centroid = resolved_entity.get("centroid_xyz_tool") or resolved_entity.get("center_xyz_tool")
    samples: list[list[float]] = []
    if isinstance(centroid, list) and len(centroid) == 3:
        samples.append([float(item) for item in centroid])

    if sample_mode == "surface_patch_centers":
        patches = resolved_entity.get("surface_patches", {})
        if isinstance(patches, dict):
            for patch in patches.values():
                if not isinstance(patch, dict):
                    continue
                center = patch.get("center_xyz_tool")
                if isinstance(center, list) and len(center) == 3:
                    samples.append([float(item) for item in center])

    if sample_mode == "obb_corners":
        obb = resolved_entity.get("obb")
        if isinstance(obb, dict):
            center = obb.get("center_xyz_tool")
            extent = obb.get("extent_xyz")
            axes = obb.get("axes_tool")
            if (
                isinstance(center, list)
                and len(center) == 3
                and isinstance(extent, list)
                and len(extent) == 3
                and isinstance(axes, dict)
            ):
                axis_vectors: list[list[float]] = []
                for axis_key in ("x", "y", "z"):
                    axis = axes.get(axis_key)
                    if not (isinstance(axis, list) and len(axis) == 3):
                        axis_vectors = []
                        break
                    axis_vectors.append([float(item) for item in axis])
                if axis_vectors:
                    half_extents = [0.5 * float(item) for item in extent]
                    center_vec = [float(item) for item in center]
                    for sx in (-1.0, 1.0):
                        for sy in (-1.0, 1.0):
                            for sz in (-1.0, 1.0):
                                samples.append(
                                    _add_scaled(
                                        center_vec,
                                        axis_vectors,
                                        [
                                            sx * half_extents[0],
                                            sy * half_extents[1],
                                            sz * half_extents[2],
                                        ],
                                    )
                                )

    if samples:
        return samples
    raise ConstraintCanonicalizationError("tool occupancy clearance needs at least one tool-frame sample")


def _tool_occupancy_from_constraint(
    *,
    constraint: dict[str, Any],
    resolver: ReferenceResolver,
) -> ToolOccupancyClearanceTarget | None:
    if constraint["type"] != "tool_occupancy_clearance":
        return None
    refs = constraint["refs"]
    tool_part = resolver.resolve(refs["tool_part"])
    if tool_part.point_tool is None:
        raise ConstraintCanonicalizationError(
            f"{constraint['id']} tool_occupancy_clearance needs a tool-frame part"
        )
    obstacle = resolver.resolve(refs["obstacle"])
    sample_mode = str(constraint["params"]["samples"])
    return ToolOccupancyClearanceTarget(
        tool_part_ref=tool_part.ref,
        obstacle_ref=obstacle.ref,
        sample_points_tool=_tool_part_sample_points(
            resolved_entity=tool_part.entity,
            sample_mode=sample_mode,
        ),
        obstacle_world=obstacle.point_world,
        min_distance_m=float(constraint["params"]["min_distance_m"]),
        samples=sample_mode,
        weight=_constraint_weight(constraint),
        constraint_id=constraint["id"],
        hard=_hard(constraint),
    )


def canonicalize_stage_constraints(
    *,
    stage_name: str,
    constraints: list[dict[str, Any]],
    resolver: ReferenceResolver,
    current_pose: dict[str, Any],
) -> CanonicalStageConstraints:
    point_targets: list[PointTarget] = []
    axis_targets: list[AxisTarget] = []
    clearance_targets: list[ClearanceTarget] = []
    move_along_targets: list[MoveAlongTarget] = []
    tool_occupancy_clearance_targets: list[ToolOccupancyClearanceTarget] = []

    for constraint in constraints:
        point_target = _point_target_from_constraint(
            constraint=constraint,
            resolver=resolver,
            current_pose=current_pose,
        )
        if point_target is not None:
            point_targets.append(point_target)
        axis_target = _axis_target_from_constraint(
            constraint=constraint,
            resolver=resolver,
        )
        if axis_target is not None:
            axis_targets.append(axis_target)
        clearance_target = _clearance_from_constraint(
            constraint=constraint,
            resolver=resolver,
        )
        if clearance_target is not None:
            clearance_targets.append(clearance_target)
        move_along_target = _move_along_from_constraint(
            constraint=constraint,
            resolver=resolver,
        )
        if move_along_target is not None:
            move_along_targets.append(move_along_target)
        tool_occupancy_target = _tool_occupancy_from_constraint(
            constraint=constraint,
            resolver=resolver,
        )
        if tool_occupancy_target is not None:
            tool_occupancy_clearance_targets.append(tool_occupancy_target)

    return CanonicalStageConstraints(
        stage_name=stage_name,
        source_ids=[constraint["id"] for constraint in constraints],
        point_targets=point_targets,
        axis_targets=axis_targets,
        clearance_targets=clearance_targets,
        move_along_targets=move_along_targets,
        tool_occupancy_clearance_targets=tool_occupancy_clearance_targets,
        orientation_sample_points_tool=resolver.orientation_sample_points_tool(),
    )
