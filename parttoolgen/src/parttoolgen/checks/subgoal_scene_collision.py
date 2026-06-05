#!/usr/bin/env python3
"""Check ToolPoseSubgoals against scene object AABBs.

This is a policy-agnostic gate between the geometric compiler and SimToolReal:
compiler constraints may be satisfied while the whole tool body still intersects
a target or distractor object. SimToolReal should not be blamed for failing to
track such object goals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.adapters.simtoolreal_policy_priors import (  # noqa: E402
    _quat_xyzw_to_matrix,
    load_obj_vertices,
    resolve_simtoolreal_tool_mesh_path,
)
from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    load_structured_file,
    validate_observed_scene_graph,
    validate_target_part_graph,
    validate_tool_pose_subgoals,
)


NEAR_CONTACT_EFFECT_GOAL_GRAZING_M = 0.004


def _transform_point(
    pose: Mapping[str, Any],
    vertex: Sequence[float],
) -> tuple[float, float, float]:
    rot = _quat_xyzw_to_matrix([float(x) for x in pose["orientation_xyzw"]])
    pos = [float(x) for x in pose["position"]]
    x, y, z = [float(v) for v in vertex]
    return (
        pos[0] + rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
        pos[1] + rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
        pos[2] + rot[2][0] * x + rot[2][1] * y + rot[2][2] * z,
    )


def _aabb_clearance(
    point: Sequence[float],
    center: Sequence[float],
    extent: Sequence[float],
) -> float:
    half = [0.5 * float(v) for v in extent]
    outside_sq = 0.0
    inside_margins: list[float] = []
    for value, c, h in zip(point, center, half):
        delta = abs(float(value) - float(c))
        if delta > h:
            outside_sq += (delta - h) ** 2
        else:
            inside_margins.append(h - delta)
    if outside_sq > 0.0:
        return math.sqrt(outside_sq)
    return -min(inside_margins) if inside_margins else 0.0


def _scene_obstacles(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    obstacles: list[dict[str, Any]] = []
    objects = scene.get("objects", {})
    if not isinstance(objects, Mapping):
        return obstacles
    for object_id, obj in sorted(objects.items()):
        if not isinstance(obj, Mapping):
            continue
        category = str(obj.get("category", ""))
        attrs = obj.get("attributes", {})
        if not isinstance(attrs, Mapping):
            continue
        if category == "support_surface" or attrs.get("role") == "support_surface":
            continue
        if attrs.get("collision_role") == "dynamic_manipulable":
            continue
        center = attrs.get("center_xyz_world")
        extent = attrs.get("extent_xyz")
        if not (
            isinstance(center, list)
            and len(center) == 3
            and isinstance(extent, list)
            and len(extent) == 3
        ):
            continue
        obstacles.append(
            {
                "object_ref": f"scene.objects.{object_id}",
                "object_id": str(object_id),
                "category": category,
                "center_xyz_world": [float(v) for v in center],
                "extent_xyz": [float(v) for v in extent],
            }
        )
    return obstacles


def _object_ref_for_target(
    ref: str | None,
    *,
    scene: Mapping[str, Any],
    target_part_graph: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not ref:
        return None, None
    if ref.startswith("scene.objects."):
        return ref, None
    parts = ref.split(".")
    if len(parts) == 3 and parts[0] == "scene":
        collection, entity_id = parts[1], parts[2]
        entity = scene.get(collection, {}).get(entity_id)
        if isinstance(entity, Mapping):
            object_ref = entity.get("object_ref")
            if isinstance(object_ref, str):
                if object_ref.startswith("scene.objects."):
                    return object_ref, _semantic_role(entity, entity_id)
                return f"scene.objects.{object_ref}", _semantic_role(entity, entity_id)
            return None, _semantic_role(entity, entity_id)
    if len(parts) == 3 and parts[0] == "target" and target_part_graph:
        collection, entity_id = parts[1], parts[2]
        entity = target_part_graph.get(collection, {}).get(entity_id)
        if isinstance(entity, Mapping):
            object_ref = entity.get("object_ref")
            if isinstance(object_ref, str):
                return (
                    object_ref if object_ref.startswith("scene.objects.") else f"scene.objects.{object_ref}",
                    _semantic_role(entity, entity_id),
                )
            return None, _semantic_role(entity, entity_id)
    return None, None


def _semantic_role(entity: Mapping[str, Any], entity_id: str) -> str:
    provenance = entity.get("provenance", {})
    role = None
    if isinstance(provenance, Mapping):
        role = provenance.get("semantic_role")
    if not role:
        role = entity.get("role")
    if not role:
        roles = entity.get("role_candidates")
        if isinstance(roles, list):
            role = ",".join(str(item) for item in roles)
    return str(role or entity_id)


def _expected_contact_allows_penetration(expected_contact: Mapping[str, Any] | None) -> bool:
    """Return the compiler-declared contact allowance for a subgoal.

    5.15 does not infer semantic contact permission from stage names such as
    "strike" or "press". Only the ProgramIR compiler can mark a subgoal as an
    intentional contact goal after seeing an explicit contact primitive.
    """

    if not isinstance(expected_contact, Mapping):
        return False
    return expected_contact.get("allow_contact_penetration") is True


def _contact_semantic_role(goal: Mapping[str, Any]) -> str:
    source_constraints = goal.get("source_constraints")
    if isinstance(source_constraints, Sequence) and not isinstance(source_constraints, str):
        for constraint in source_constraints:
            if not isinstance(constraint, str):
                continue
            prefix = "source_semantic_role:"
            if constraint.startswith(prefix):
                return constraint[len(prefix) :]
    role = goal.get("semantic_role")
    return role if isinstance(role, str) else ""


def _goal_is_terminal_contact_anchor(goal: Mapping[str, Any]) -> bool:
    explicit_gate = goal.get("semantic_gate")
    if isinstance(explicit_gate, bool) and explicit_gate:
        return True
    return "terminal" in _contact_semantic_role(goal).lower()


def _allowed_contact_penetration_m(
    expected_contact: Mapping[str, Any] | None,
    *,
    default_m: float,
    target_role: str | None = None,
) -> float:
    """Resolve the permitted AABB penetration for declared effect contacts.

    ``target_contact_goal_point`` refs are public target refs used by tasks such
    as impact/push/sweep to encode an intended effect endpoint inside or beyond
    a target object. They are still verifier-facing geometry, not private
    oracle labels, so they can use a slightly larger endpoint tolerance than a
    passive contact surface while ordinary obstacles stay strict.
    """

    resolved = float(default_m)
    if isinstance(expected_contact, Mapping):
        configured = expected_contact.get("allowed_contact_penetration_m")
        if isinstance(configured, (int, float)) and not isinstance(configured, bool):
            resolved = max(resolved, float(configured))
    if target_role == "task_direction_contact_goal_point":
        resolved = max(resolved, 0.015)
    return resolved


def _pose_records(subgoals: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
        ("start_pose", subgoals["start_pose"], {})
    ]
    for goal in subgoals.get("goals", []):
        if not isinstance(goal, Mapping):
            continue
        pose = goal.get("pose_world_tool")
        if isinstance(pose, Mapping):
            records.append((f"goals[{goal.get('index')}]", pose, goal))
    return records


def check_subgoal_scene_collision(
    subgoals: dict[str, Any],
    scene: dict[str, Any],
    *,
    target_part_graph: dict[str, Any] | None = None,
    tool_id: str | None = None,
    tool_mesh_path: str | Path | None = None,
    min_clearance_m: float = 0.003,
    allowed_contact_penetration_m: float = 0.006,
) -> dict[str, Any]:
    validate_tool_pose_subgoals(subgoals)
    validate_observed_scene_graph(scene)
    if target_part_graph:
        validate_target_part_graph(target_part_graph)

    mesh_path = resolve_simtoolreal_tool_mesh_path(
        tool_id or subgoals.get("tool_id"),
        explicit_path=tool_mesh_path,
    )
    report: dict[str, Any] = {
        "schema_version": "v0.1",
        "passed": True,
        "checked": False,
        "tool_id": tool_id or subgoals.get("tool_id"),
        "tool_mesh_path": str(mesh_path) if mesh_path else None,
        "min_clearance_m": float(min_clearance_m),
        "allowed_contact_penetration_m": float(allowed_contact_penetration_m),
        "obstacles": [],
        "pose_clearances": [],
        "violations": [],
        "warnings": [],
    }
    if mesh_path is None:
        report["warnings"].append("tool mesh not resolved; scene collision was not checked")
        return report
    vertices = load_obj_vertices(mesh_path)
    obstacles = _scene_obstacles(scene)
    report["checked"] = True
    report["vertex_count"] = len(vertices)
    report["obstacles"] = obstacles
    if not vertices or not obstacles:
        return report

    for pose_name, pose, goal in _pose_records(subgoals):
        stage = str(goal.get("stage", pose_name))
        expected_contact = goal.get("expected_contact", {})
        target_ref = None
        if isinstance(expected_contact, Mapping):
            target_ref = expected_contact.get("scene_target")
        target_object_ref, target_role = _object_ref_for_target(
            str(target_ref) if isinstance(target_ref, str) else None,
            scene=scene,
            target_part_graph=target_part_graph,
        )
        world_vertices = [_transform_point(pose, vertex) for vertex in vertices]
        for obstacle in obstacles:
            min_clearance = min(
                _aabb_clearance(
                    vertex,
                    obstacle["center_xyz_world"],
                    obstacle["extent_xyz"],
                )
                for vertex in world_vertices
            )
            same_expected_object = target_object_ref == obstacle["object_ref"]
            expected_allows_penetration = _expected_contact_allows_penetration(
                expected_contact if isinstance(expected_contact, Mapping) else None
            )
            terminal_task_direction_contact = (
                target_role != "task_direction_contact_goal_point"
                or _goal_is_terminal_contact_anchor(goal)
            )
            contact_allowed = (
                same_expected_object
                and expected_allows_penetration
                and terminal_task_direction_contact
            )
            near_contact_effect_goal_grazing = (
                same_expected_object
                and not contact_allowed
                and expected_allows_penetration
                and target_role == "task_direction_contact_goal_point"
            )
            contact_penetration_m = _allowed_contact_penetration_m(
                expected_contact if isinstance(expected_contact, Mapping) else None,
                default_m=allowed_contact_penetration_m,
                target_role=target_role,
            )
            required = (
                -contact_penetration_m
                if contact_allowed
                else -NEAR_CONTACT_EFFECT_GOAL_GRAZING_M
                if near_contact_effect_goal_grazing
                else float(min_clearance_m)
            )
            record = {
                "pose": pose_name,
                "goal_index": goal.get("index"),
                "stage": stage,
                "object_ref": obstacle["object_ref"],
                "category": obstacle["category"],
                "clearance_m": min_clearance,
                "required_clearance_m": required,
                "allowed_contact_penetration_m": contact_penetration_m
                if contact_allowed
                else NEAR_CONTACT_EFFECT_GOAL_GRAZING_M
                if near_contact_effect_goal_grazing
                else 0.0,
                "contact_allowed": contact_allowed,
                "contact_penetration_scope": (
                    "expected_target_contact"
                    if contact_allowed
                    else "near_contact_effect_goal_grazing"
                    if near_contact_effect_goal_grazing
                    else "strict_clearance"
                ),
                "expected_target_ref": target_ref,
                "expected_target_role": target_role,
            }
            report["pose_clearances"].append(record)
            if min_clearance + 1e-9 < required:
                report["violations"].append(record)

    if report["violations"]:
        report["passed"] = False
        first = report["violations"][0]
        report["warnings"].append(
            f"{first['pose']} stage={first['stage']} tool mesh clearance "
            f"{first['clearance_m']:.4f}m against {first['object_ref']} is below "
            f"required {first['required_clearance_m']:.4f}m"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgoals", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--target-parts")
    parser.add_argument("--report")
    parser.add_argument("--tool-id")
    parser.add_argument("--tool-mesh")
    parser.add_argument("--min-clearance-m", type=float, default=0.003)
    parser.add_argument("--allowed-contact-penetration-m", type=float, default=0.006)
    args = parser.parse_args()
    report = check_subgoal_scene_collision(
        load_structured_file(args.subgoals),
        load_structured_file(args.scene),
        target_part_graph=(
            load_structured_file(args.target_parts) if args.target_parts else None
        ),
        tool_id=args.tool_id,
        tool_mesh_path=args.tool_mesh,
        min_clearance_m=args.min_clearance_m,
        allowed_contact_penetration_m=args.allowed_contact_penetration_m,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
