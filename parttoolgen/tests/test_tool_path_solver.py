from __future__ import annotations

from parttoolgen.compiler.tool_path_solver import (
    _goal_clearance_violations,
    _line_search_safe_pose,
    _segment_sample_check,
    solve_collision_safe_tool_path,
)
from parttoolgen.program.api import ProgramContext
from parttoolgen.program.sandbox import execute_planner_program, make_planner_program_artifact


def _grounding() -> dict:
    return {
        "confidence": 1.0,
        "source": "unit_test",
        "provenance": {
            "observation_id": "unit_test",
            "method": "unit_test",
            "source_artifact": "unit_test",
        },
    }


def _scene_with_path_limit() -> dict:
    return {
        "schema_version": "v0.1",
        "scene_id": "scene",
        "frames": {"world": {"type": "world"}},
        "objects": {
            "target": {
                **_grounding(),
                "category": "target_object",
            }
        },
        "keypoints": {
            "limit": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [1.0, 0.0, 0.0],
                "object_ref": "target",
            }
        },
        "axes": {},
        "regions": {},
        "surfaces": {},
        "relations": [],
        "metadata": {"created_by": "unit_test", "geometry_mode": "observed"},
    }


def _scene_with_block_obstacle() -> dict:
    scene = _scene_with_path_limit()
    scene["objects"]["target"]["attributes"] = {
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }
    scene["keypoints"]["limit"]["object_ref"] = "target"
    return scene


def _functional_part(center: list[float], normal: list[float]) -> dict:
    return {
        **_grounding(),
        "role": "contact_patch",
        "center": {"xyz_tool": center},
        "normal": {"vector_tool": normal},
        "axis": {"vector_tool": [1.0, 0.0, 0.0]},
        "frame": {
            "origin": center,
            "x_axis": [1.0, 0.0, 0.0],
            "y_axis": [0.0, 1.0, 0.0],
            "z_axis": [0.0, 0.0, 1.0],
        },
        "T_tool_functional": [
            [1.0, 0.0, 0.0, center[0]],
            [0.0, 1.0, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def _tool_for_path_objective() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "tool",
        "tool_frame": "tool",
        "functional_parts": {
            "tip": _functional_part([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        },
        "grasp_references": {
            "grasp": {
                **_grounding(),
                "position_tool": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "metadata": {"created_by": "unit_test", "geometry_mode": "prior"},
    }


def _subgoals_for_path_objective() -> dict:
    return {
        "schema_version": "v0.2",
        "subgoals_id": "path_objective_subgoals",
        "tool_id": "tool",
        "task_id": "path_objective_program",
        "world_frame": "world",
        "units": "meters",
        "quaternion_order": "xyzw",
        "start_pose": {"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        "goals": [
            {
                "index": 0,
                "stage": "move",
                "pose_world_tool": {
                    "position": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["move_path_objective"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.keypoints.limit",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": [2.0, 0.0, 0.0],
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            }
        ],
        "source_trace": {
            "observed_scene_graph": {"id": "scene", "path": "<in-memory>"},
            "grounding_candidates": {"id": "<none>", "path": "<none>"},
            "functional_tool_parts": {"id": "tool", "path": "<in-memory>"},
            "planner_program": {"id": "path_objective_program", "path": "<in-memory>"},
            "program_ir": {"id": "path_objective_program", "path": "<in-memory>"},
        },
        "sanity_checks": {
            "compiled": True,
            "path_objectives_declared": 1,
            "path_objectives_deferred": 1,
        },
        "metadata": {
            "generated_by": "unit_test",
            "source": "unit_test",
            "geometry_mode": "observed",
        },
    }


def _one_goal_subgoals(stage: str, position: list[float]) -> dict:
    return {
        "schema_version": "v0.2",
        "subgoals_id": f"{stage}_subgoals",
        "tool_id": "tool",
        "task_id": "collision_program",
        "world_frame": "world",
        "units": "meters",
        "quaternion_order": "xyzw",
        "start_pose": {"position": [2.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        "goals": [
            {
                "index": 0,
                "stage": stage,
                "pose_world_tool": {
                    "position": position,
                    "orientation_xyzw": [0, 0, 0, 1],
                },
                "duration_sec": 0.45,
                "tolerance": {"position_m": 0.02, "rotation_rad": 0.18},
                "source_constraints": ["collision_constraint"],
                "expected_contact": {
                    "tool_part": "tool.functional_parts.tip.center",
                    "scene_target": "scene.objects.target",
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                    "position_world": position,
                    "orientation_xyzw": [0, 0, 0, 1],
                },
            }
        ],
        "source_trace": {
            "observed_scene_graph": {"id": "scene", "path": "<in-memory>"},
            "grounding_candidates": {"id": "<none>", "path": "<none>"},
            "functional_tool_parts": {"id": "tool", "path": "<in-memory>"},
            "planner_program": {"id": "collision_program", "path": "<in-memory>"},
            "program_ir": {"id": "collision_program", "path": "<in-memory>"},
        },
        "sanity_checks": {"compiled": True},
        "metadata": {
            "generated_by": "unit_test",
            "source": "unit_test",
            "geometry_mode": "observed",
        },
    }


def _path_objective_program() -> tuple[dict, dict]:
    source = '''
def stay_before_limit(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    limit = refs.world_point("scene.keypoints.limit")
    return tip[0] - limit[0]

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    limit = ctx.ref("scene.keypoints.limit")
    stage = ctx.stage("move")
    stage.objective(stay_before_limit, kind="path", weight=1.0, refs=[tip, limit])
    return ctx.program()
'''
    planner = make_planner_program_artifact(
        raw_text=source,
        program_id="path_objective_program",
        task_instruction="move while staying before the limit",
        generated_by="unit_test",
    )
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "scene.keypoints.limit",
        },
    )
    program_ir = execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )
    return planner, program_ir


def test_line_search_truncates_segment_before_aabb_collision():
    goal = {
        "index": 0,
        "stage": "approach",
        "expected_contact": {"scene_target": "scene.objects.target"},
    }
    scene = {"objects": {"target": {"category": "block", "attributes": {}}}}
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }
    safe_pose, safe_fraction, violations = _line_search_safe_pose(
        start_pose={"position": [2.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        iterations=24,
    )

    assert 0.0 < safe_fraction < 1.0
    assert safe_pose["position"][0] >= 0.6
    assert violations[0]["object_ref"] == "scene.objects.target"
    assert violations[0]["contact_allowed"] is False


def test_segment_interior_uses_near_contact_nonpenetration_before_contact_endpoint():
    goal = {
        "index": 0,
        "stage": "contact",
        "expected_contact": {
            "scene_target": "scene.objects.target",
            "allow_contact_penetration": True,
        },
    }
    scene = {"objects": {"target": {"category": "block", "attributes": {}}}}
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }
    end_pose = {"position": [0.495, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]}

    endpoint_safe, endpoint_violations = _goal_clearance_violations(
        goal=goal,
        pose=end_pose,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
    )
    segment_safe, first_bad_fraction, segment_violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose=end_pose,
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=3,
    )

    assert endpoint_safe is True
    assert endpoint_violations == []
    assert segment_safe is True
    assert first_bad_fraction == 1.0
    assert segment_violations == []


def test_task_direction_contact_goal_allows_effect_penetration_endpoint():
    goal = {
        "index": 0,
        "stage": "engage",
        "expected_contact": {
            "scene_target": "scene.keypoints.target_contact_goal_point",
            "allow_contact_penetration": True,
        },
    }
    scene = {
        "objects": {"target": {"category": "block", "attributes": {}}},
        "keypoints": {
            "target_contact_goal_point": {
                "object_ref": "target",
                "provenance": {"semantic_role": "task_direction_contact_goal_point"},
            }
        },
    }
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }

    endpoint_safe, endpoint_violations = _goal_clearance_violations(
        goal=goal,
        pose={"position": [0.486, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
    )

    assert endpoint_safe is True
    assert endpoint_violations == []


def test_compiler_transition_cannot_inherit_contact_penetration_allowance():
    goal = {
        "index": 0,
        "stage": "contact_transition_1",
        "semantic_role": "generated_transition_waypoint",
        "expected_contact": {
            "scene_target": "scene.objects.target",
            "allow_contact_penetration": True,
        },
    }
    scene = {"objects": {"target": {"category": "block", "attributes": {}}}}
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }

    safe, first_bad_fraction, violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.495, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=2,
    )

    assert safe is False
    assert first_bad_fraction == 1.0
    assert violations[0]["contact_allowed"] is False


def test_terminal_transition_inherits_task_direction_contact_allowance():
    goal = {
        "index": 0,
        "stage": "contact_transition_1",
        "semantic_role": "generated_transition_waypoint",
        "source_constraints": [
            "source_semantic_role:swing_terminal",
        ],
        "expected_contact": {
            "scene_target": "scene.keypoints.target_contact_goal_point",
            "allow_contact_penetration": True,
        },
    }
    scene = {
        "objects": {"target": {"category": "block", "attributes": {}}},
        "keypoints": {
            "target_contact_goal_point": {
                "object_ref": "target",
                "provenance": {"semantic_role": "task_direction_contact_goal_point"},
            }
        },
    }
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }

    shallow_safe, _fraction, _violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.499, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=2,
    )
    deep_safe, first_bad_fraction, violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.495, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=2,
    )

    assert shallow_safe is True
    assert deep_safe is True
    assert first_bad_fraction == 1.0
    assert violations == []


def test_nonterminal_effect_waypoint_allows_only_small_grazing():
    goal = {
        "index": 0,
        "stage": "contact_transition_1",
        "semantic_role": "generated_transition_waypoint",
        "source_constraints": [
            "source_semantic_role:motion_swing_arc",
        ],
        "expected_contact": {
            "scene_target": "scene.keypoints.target_contact_goal_point",
            "allow_contact_penetration": True,
        },
    }
    scene = {
        "objects": {"target": {"category": "block", "attributes": {}}},
        "keypoints": {
            "target_contact_goal_point": {
                "object_ref": "target",
                "provenance": {"semantic_role": "task_direction_contact_goal_point"},
            }
        },
    }
    obstacle = {
        "object_ref": "scene.objects.target",
        "center_xyz_world": [0.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }

    shallow_safe, _fraction, _violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.498, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=2,
    )
    deep_safe, first_bad_fraction, violations = _segment_sample_check(
        start_pose={"position": [0.7, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        end_pose={"position": [0.495, 0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        goal=goal,
        obstacles=[obstacle],
        vertices_tool=[[0.0, 0.0, 0.0]],
        scene=scene,
        target_part_graph=None,
        min_clearance_m=0.1,
        swept_min_clearance_m=0.1,
        allowed_contact_penetration_m=0.006,
        samples=2,
    )

    assert shallow_safe is True
    assert deep_safe is False
    assert first_bad_fraction == 1.0
    assert violations[0]["contact_penetration_scope"] == "near_contact_effect_goal_grazing"


def test_path_gate_does_not_insert_holds_or_tracking_waypoints(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    scene = _scene_with_path_limit()
    scene["objects"]["target"]["attributes"] = {
        "center_xyz_world": [10.0, 0.0, 0.0],
        "extent_xyz": [1.0, 1.0, 1.0],
    }

    checked, report = solve_collision_safe_tool_path(
        _one_goal_subgoals("engage", [0.0, 0.0, 0.0]),
        scene,
        tool_mesh_path=mesh_path,
    )

    assert report["path_solver_mode"] == "diagnostic_no_goal_repair_v1"
    assert report["changed"] is False
    assert len(checked["goals"]) == 1
    assert checked["goals"][0]["stage"] == "engage"
    assert "tracking_densification" not in report
    assert "semantic_anchor_holds" not in report


def test_path_objective_reports_failure_without_truncating_segment(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    planner, program_ir = _path_objective_program()

    checked, report = solve_collision_safe_tool_path(
        _subgoals_for_path_objective(),
        _scene_with_path_limit(),
        program_ir=program_ir,
        planner_program=planner,
        functional_tool_parts=_tool_for_path_objective(),
        tool_mesh_path=mesh_path,
        path_objective_samples_per_segment=9,
    )

    assert report["path_objectives_checked"] is True
    assert report["path_objectives_passed"] is False
    assert report["passed"] is False
    assert report["changed"] is False
    assert report["path_objective_violations"]
    assert report["path_objective_failures"]
    assert checked["goals"][0]["pose_world_tool"]["position"] == [2.0, 0.0, 0.0]
    assert checked["sanity_checks"]["tool_path_solver"]["path_objectives_passed"] is False


def test_deferred_path_objective_without_program_ir_is_not_silent_success(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")

    _repaired, report = solve_collision_safe_tool_path(
        _subgoals_for_path_objective(),
        _scene_with_path_limit(),
        functional_tool_parts=_tool_for_path_objective(),
        tool_mesh_path=mesh_path,
    )

    assert report["path_objectives_checked"] is False
    assert report["path_objectives_passed"] is False
    assert report["passed"] is False
    assert "program_ir was not provided" in " ".join(report["warnings"])


def test_path_objectives_still_checked_with_empty_tool_mesh(tmp_path):
    mesh_path = tmp_path / "empty_tool.obj"
    mesh_path.write_text("", encoding="utf-8")
    planner, program_ir = _path_objective_program()

    _checked, report = solve_collision_safe_tool_path(
        _subgoals_for_path_objective(),
        _scene_with_path_limit(),
        program_ir=program_ir,
        planner_program=planner,
        functional_tool_parts=_tool_for_path_objective(),
        tool_mesh_path=mesh_path,
        path_objective_samples_per_segment=9,
    )

    assert report["path_objectives_checked"] is True
    assert report["path_objective_violations"]
    assert report["path_objective_failures"]
    assert "tool mesh had no vertices" in " ".join(report["warnings"])


def test_compiler_transition_collision_is_reported_without_generated_waypoint(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")

    checked, report = solve_collision_safe_tool_path(
        _one_goal_subgoals("approach_transition_1", [0.0, 0.0, 0.0]),
        _scene_with_block_obstacle(),
        tool_mesh_path=mesh_path,
    )

    assert report["changed"] is False
    assert report["passed"] is False
    assert report["path_failures"]
    assert "repairs" not in report
    assert len(checked["goals"]) == 1
    assert checked["goals"][0]["stage"] == "approach_transition_1"


def test_semantic_anchor_path_through_obstacle_reports_path_failure(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")

    checked, report = solve_collision_safe_tool_path(
        _one_goal_subgoals("approach", [-2.0, 0.0, 0.0]),
        _scene_with_block_obstacle(),
        tool_mesh_path=mesh_path,
    )

    assert report["after"]["passed"] is True
    assert report["after_segments"]["passed"] is False
    assert report["passed"] is False
    assert report["path_failures"]
    assert report["changed"] is False
    assert "repairs" not in report
    assert checked["goals"][-1]["stage"] == "approach"
    assert checked["goals"][-1]["pose_world_tool"]["position"] == [-2.0, 0.0, 0.0]


def test_semantic_anchor_endpoint_collision_reports_path_failure(tmp_path):
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")

    checked, report = solve_collision_safe_tool_path(
        _one_goal_subgoals("approach", [0.0, 0.0, 0.0]),
        _scene_with_block_obstacle(),
        tool_mesh_path=mesh_path,
    )

    assert report["changed"] is False
    assert report["semantic_goal_preserved"] is False
    assert report["passed"] is False
    assert report["path_failures"]
    assert checked["goals"][0]["pose_world_tool"]["position"] == [0.0, 0.0, 0.0]
