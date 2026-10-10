from __future__ import annotations

from ToGo.compiler.constraint_canonicalizer import MoveAlongTarget
from ToGo.compiler.residuals import norm, pose_point, quat_angle_rad, sub
from ToGo.compiler.geometric_solvers import quat_rotate
from ToGo.compiler.se3_optimizer import compile_move_along_waypoints
from ToGo.compiler.tool_pose_compiler import (
    ProgramIRCompileError,
    compile_program_ir_to_tool_pose_subgoals,
)
from ToGo.program.api import ProgramContext
from ToGo.program.sandbox import execute_planner_program, make_planner_program_artifact
from ToGo.validation.validate_parttoolgen import validate_program_ir


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


def _scene_for_objective() -> dict:
    return {
        "schema_version": "v0.1",
        "scene_id": "scene",
        "frames": {"world": {"type": "world"}},
        "objects": {
            "target": {
                **_grounding(),
                "category": "target_object",
                "attributes": {
                    "center_xyz_world": [1.0, 0.0, 0.0],
                    "extent_xyz": [0.1, 0.1, 0.1],
                },
            }
        },
        "keypoints": {
            "goal": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [1.0, 0.0, 0.0],
                "object_ref": "target",
            },
            "secret": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [2.0, 0.0, 0.0],
                "object_ref": "target",
            },
        },
        "axes": {
            "up": {
                **_grounding(),
                "frame": "world",
                "vector_world": [0.0, 0.0, 1.0],
                "object_ref": "target",
            }
        },
        "regions": {},
        "surfaces": {},
        "relations": [],
        "metadata": {"created_by": "unit_test", "geometry_mode": "observed"},
    }


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


def _tool_for_objective() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "tool",
        "tool_frame": "tool",
        "functional_parts": {
            "tip": _functional_part([0.2, 0.0, 0.0], [0.0, 0.0, 1.0]),
        },
        "grasp_references": {
            "grasp": {
                **_grounding(),
                "position_tool": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "metadata": {
            "created_by": "unit_test",
            "geometry_mode": "prior",
            "segmentation_mode": "mesh_prior_adapter",
        },
    }


def _tool_part_graph() -> dict:
    return {
        "schema_version": "v0.1",
        "tool_id": "tool",
        "tool_frame": "tool",
        "parts": {
            "part_001": {
                **_grounding(),
                "label": "selected tool part",
                "role_candidates": ["functional_region"],
                "mask_ref": "unit_test:part_001",
                "point_count": 8,
                "centroid_xyz_tool": [0.2, 0.0, 0.0],
                "obb": {
                    "center_xyz_tool": [0.2, 0.0, 0.0],
                    "extent_xyz": [0.1, 0.1, 0.1],
                    "axes_tool": {
                        "x": [1.0, 0.0, 0.0],
                        "y": [0.0, 1.0, 0.0],
                        "z": [0.0, 0.0, 1.0],
                    },
                },
            }
        },
        "relations": [],
        "metadata": {
            "created_by": "unit_test",
            "geometry_mode": "prior",
            "segmentation_mode": "mesh_prior_adapter",
        },
    }


def _planner_artifact(source: str) -> dict:
    return make_planner_program_artifact(
        raw_text=source,
        program_id="objective_program",
        task_instruction="place the tip on the goal",
        generated_by="unit_test",
    )


def _program_ir_from_source(source: str) -> tuple[dict, dict]:
    planner = _planner_artifact(source)
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "scene.keypoints.goal",
        },
    )
    return planner, execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )


def test_move_along_waypoints_include_declared_start_ref() -> None:
    move = MoveAlongTarget(
        source_ref="tool.functional_parts.face.center",
        start_ref="scene.keypoints.pre_contact",
        goal_ref="scene.keypoints.contact",
        point_tool=[0.0, 0.0, 0.0],
        start_world=[0.0, 0.0, 1.0],
        goal_world=[0.0, 0.0, 0.0],
        weight=1.0,
        constraint_id="move_constraint",
    )

    waypoints = compile_move_along_waypoints(
        move=move,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_pose={"position": [0.0, 0.0, 2.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        steps=3,
    )

    functional_positions = [
        pose_point(waypoint[0], [0.0, 0.0, 0.0]) for waypoint in waypoints
    ]
    assert norm(sub(functional_positions[0], [0.0, 0.0, 1.0])) < 1e-9
    assert norm(sub(functional_positions[-1], [0.0, 0.0, 0.0])) < 1e-9


def test_swing_arc_move_along_waypoints_rotate_before_impact() -> None:
    move = MoveAlongTarget(
        source_ref="tool.functional_parts.face.center",
        start_ref="scene.keypoints.pre_contact",
        goal_ref="scene.keypoints.contact",
        point_tool=[0.0, 0.0, 0.0],
        start_world=[0.0, 0.0, 1.0],
        goal_world=[0.0, 0.0, 0.0],
        weight=1.0,
        constraint_id="swing_constraint",
        profile="swing_arc",
        swing_axis_ref="target.axes.swing",
        swing_axis_world=[0.0, 1.0, 0.0],
        swing_angle_rad=0.6,
    )

    waypoints = compile_move_along_waypoints(
        move=move,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_pose={"position": [0.0, 0.0, 2.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        steps=4,
    )

    functional_positions = [
        pose_point(waypoint[0], [0.0, 0.0, 0.0]) for waypoint in waypoints
    ]
    assert norm(sub(functional_positions[0], [0.0, 0.0, 1.0])) < 1e-9
    assert norm(sub(functional_positions[-1], [0.0, 0.0, 0.0])) < 1e-9
    assert abs(functional_positions[2][0]) > 0.0
    assert quat_angle_rad(
        waypoints[2][0]["orientation_xyzw"],
        waypoints[0][0]["orientation_xyzw"],
    ) > 0.4
    assert quat_angle_rad(
        waypoints[0][0]["orientation_xyzw"],
        waypoints[-1][0]["orientation_xyzw"],
    ) < 1e-9
    for previous, current in zip(waypoints, waypoints[1:]):
        assert quat_angle_rad(
            previous[0]["orientation_xyzw"],
            current[0]["orientation_xyzw"],
        ) <= 0.43


def test_swing_arc_move_along_accepts_function_surface_axis() -> None:
    move = MoveAlongTarget(
        source_ref="tool.functional_parts.face.center",
        start_ref="scene.keypoints.pre_contact",
        goal_ref="scene.keypoints.contact",
        point_tool=[0.0, 0.0, 0.0],
        start_world=[0.0, 0.0, 1.0],
        goal_world=[0.0, 0.0, 0.0],
        weight=1.0,
        constraint_id="tool_frame_swing_constraint",
        profile="swing_arc",
        swing_axis_ref="candidate.surfaces.function_normal",
        swing_axis_tool=[0.0, 1.0, 0.0],
        swing_angle_rad=0.6,
    )

    waypoints = compile_move_along_waypoints(
        move=move,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_pose={"position": [0.0, 0.0, 2.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        steps=4,
    )

    functional_positions = [
        pose_point(waypoint[0], [0.0, 0.0, 0.0]) for waypoint in waypoints
    ]
    assert norm(sub(functional_positions[0], [0.0, 0.0, 1.0])) < 1e-9
    assert norm(sub(functional_positions[-1], [0.0, 0.0, 0.0])) < 1e-9
    assert abs(functional_positions[2][0]) > 0.0
    assert quat_angle_rad(
        waypoints[2][0]["orientation_xyzw"],
        waypoints[0][0]["orientation_xyzw"],
    ) > 0.4
    assert quat_angle_rad(
        waypoints[0][0]["orientation_xyzw"],
        waypoints[-1][0]["orientation_xyzw"],
    ) < 1e-9


def test_move_along_waypoints_mark_only_terminal_as_semantic_anchor() -> None:
    move = MoveAlongTarget(
        source_ref="tool.functional_parts.face.center",
        start_ref="scene.keypoints.pre_contact",
        goal_ref="scene.keypoints.contact",
        point_tool=[0.0, 0.0, 0.0],
        start_world=[0.0, 0.0, 1.0],
        goal_world=[0.0, 0.0, 0.0],
        weight=1.0,
        constraint_id="slide_constraint",
        path_family="slide",
    )

    waypoints = compile_move_along_waypoints(
        move=move,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_pose={"position": [0.0, 0.0, 1.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        steps=2,
    )

    assert len(waypoints) == 4
    assert all(waypoint.semantic_gate is False for waypoint in waypoints[:-1])
    assert waypoints[0].semantic_role == "motion_slide_start"
    assert waypoints[1].semantic_role == "motion_slide_path"
    assert waypoints[-1].semantic_role == "slide_terminal"
    assert waypoints[-1].semantic_gate is True


def test_zero_length_move_along_emits_single_terminal_goal() -> None:
    move = MoveAlongTarget(
        source_ref="tool.functional_parts.face.center",
        start_ref="scene.keypoints.start",
        goal_ref="scene.keypoints.end",
        point_tool=[0.0, 0.0, 0.0],
        start_world=[0.0, 0.0, 0.5],
        goal_world=[0.0, 0.0, 0.5],
        weight=1.0,
        constraint_id="point_action_constraint",
        path_family="contact_follow",
    )

    waypoints = compile_move_along_waypoints(
        move=move,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        current_pose={
            "position": [0.0, 0.0, 0.58],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        steps=3,
    )

    assert len(waypoints) == 1
    assert waypoints[0].target_ref == "scene.keypoints.end"
    assert waypoints[0].semantic_role == "contact_follow_terminal"
    assert waypoints[0].semantic_gate is True
    assert norm(sub(pose_point(waypoints[0].pose, [0.0, 0.0, 0.0]), [0.0, 0.0, 0.5])) < 1e-9


def test_compiler_keeps_selected_contact_point_even_when_alternative_is_cheaper() -> None:
    program_ir = {
        "schema_version": "v0.1",
        "program_id": "contact_candidate_unit",
        "task_instruction": "place a VLM-approved contact point on the goal",
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [{"name": "contact", "repeat": 1}, {"name": "action", "repeat": 1}],
        "constraints": [
            {
                "id": "contact_goal",
                "stage": "contact",
                "type": "contact",
                "refs": {"source": "tool.contact_point", "target": "scene.keypoints.goal"},
                "params": {"tolerance_m": 0.01},
                "hard": True,
                "weight": 1.0,
            },
            {
                "id": "action_move",
                "stage": "action",
                "type": "move_along",
                "refs": {
                    "source": "tool.contact_point",
                    "start": "scene.keypoints.goal",
                    "end": "scene.keypoints.secret",
                },
                "params": {"profile": "linear", "path_family": "contact_follow"},
                "hard": True,
                "weight": 1.0,
            },
        ],
        "objectives": [],
        "diagnostics": {"ast_checked": True, "executed": True, "source": "unit_test"},
    }
    tool = _tool_for_objective()
    function = tool["functional_parts"]["tip"]
    function["center"]["xyz_tool"] = [0.2, 0.0, 0.0]
    function["point_candidates"] = [
        {
            "candidate_id": "vlm_far_face",
            "point_tool": [0.2, 0.0, 0.0],
            "contact_point_face_marker": "F_far",
        },
        {
            "candidate_id": "vlm_near_face",
            "point_tool": [0.0, 0.0, 0.0],
            "contact_point_face_marker": "F_near",
        },
    ]

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=tool,
        program_ir=program_ir,
        planner_program=_planner_artifact("def build_program(ctx): return ctx.program()"),
        start_pose={
            "position": [1.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        initial_grasp_lift_m=0.0,
    )

    stage = subgoals["sanity_checks"]["compiler_stages"][0]
    assert stage["pose_selection"]["point_resolutions"] == []
    final_goal = subgoals["goals"][-1]
    assert final_goal["contact_point_goal"]["point_tool"] == [0.2, 0.0, 0.0]
    assert final_goal["contact_point_goal"]["position_world"] == [2.0, 0.0, 0.0]
    assert subgoals["sanity_checks"]["locked_tool_points"]["tool.contact_point"] == [0.2, 0.0, 0.0]


def test_pre_contact_offset_stage_is_not_semantic_gate() -> None:
    program_ir = {
        "schema_version": "v0.1",
        "program_id": "pre_contact_unit",
        "task_instruction": "move near the goal before using the tool",
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [{"name": "approach", "repeat": 1}],
        "constraints": [
            {
                "id": "approach_pre_contact_offset",
                "stage": "approach",
                "type": "pre_contact_offset",
                "refs": {
                    "source": "tool.functional_parts.tip.center",
                    "target": "scene.keypoints.goal",
                    "axis": "scene.axes.up",
                },
                "params": {"distance_m": 0.08, "direction": "along_axis"},
                "hard": True,
                "weight": 1.0,
            }
        ],
        "objectives": [],
        "diagnostics": {"ast_checked": True, "executed": True, "source": "unit_test"},
    }

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        planner_program=_planner_artifact("def build_program(ctx): return ctx.program()"),
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        initial_grasp_lift_m=0.0,
    )

    goal = subgoals["goals"][-1]
    assert goal["semantic_role"] == "pre_contact_anchor"
    assert goal["semantic_gate"] is False
    assert norm(sub(goal["contact_point_goal"]["position_world"], [1.0, 0.0, 0.08])) < 1e-9


def test_surface_normal_alignment_preserves_existing_roll_when_axis_already_aligned() -> None:
    planner = _planner_artifact(
        '''
def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    normal = ctx.ref("tool.functional_parts.tip.normal")
    goal = ctx.ref("scene.keypoints.goal")
    up = ctx.ref("scene.axes.up")
    stage = ctx.stage("contact")
    stage.constraint(ctx.constraints.contact(tip, goal))
    stage.constraint(ctx.constraints.align_axis(normal, up))
    return ctx.program()
'''
    )
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "tool.functional_parts.tip.normal",
            "scene.keypoints.goal",
            "scene.axes.up",
        },
    )
    program_ir = execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )
    start_orientation = [0.0, 0.0, 0.2955202067, 0.9553364891]

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": start_orientation,
        },
    )

    goal = subgoals["goals"][-1]
    assert quat_angle_rad(goal["pose_world_tool"]["orientation_xyzw"], start_orientation) < 1e-8
    assert norm(sub(goal["contact_point_goal"]["position_world"], [1.0, 0.0, 0.0])) < 1e-9


def test_compiler_inserts_free_space_axis_alignment_before_contact() -> None:
    planner = _planner_artifact(
        '''
def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    normal = ctx.ref("tool.functional_parts.tip.normal")
    goal = ctx.ref("scene.keypoints.goal")
    up = ctx.ref("scene.axes.up")
    stage = ctx.stage("contact")
    stage.constraint(ctx.constraints.contact(tip, goal))
    stage.constraint(ctx.constraints.align_axis(normal, up))
    return ctx.program()
'''
    )
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "tool.functional_parts.tip.normal",
            "scene.keypoints.goal",
            "scene.axes.up",
        },
    )
    program_ir = execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )
    tool = _tool_for_objective()
    tool["functional_parts"]["tip"] = _functional_part([0.2, 0.0, 0.0], [1.0, 0.0, 0.0])

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=tool,
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        initial_grasp_lift_m=0.08,
    )

    roles = [goal.get("semantic_role") for goal in subgoals["goals"]]
    align_index = roles.index("tool_axis_align")
    contact_index = roles.index("semantic_anchor")
    lift_index = roles.index("grasp_lift")
    assert lift_index < align_index < contact_index
    assert subgoals["sanity_checks"]["tool_axis_align_waypoints"] == 1
    assert subgoals["sanity_checks"]["tool_axis_align_clearance_waypoints"] == 0
    align_goal = subgoals["goals"][align_index]
    lift_goal = subgoals["goals"][lift_index]
    assert norm(sub(align_goal["pose_world_tool"]["position"], lift_goal["pose_world_tool"]["position"])) < 1e-9
    projection = subgoals["sanity_checks"]["table_bbox_clearance_projection"]
    assert projection["enabled"] is False
    assert projection["num_bbox_corners"] == 0
    assert norm(sub(subgoals["goals"][-1]["contact_point_goal"]["position_world"], [1.0, 0.0, 0.0])) < 1e-9


def test_coupled_axis_alignment_uses_secondary_axis_to_choose_roll() -> None:
    scene = _scene_for_objective()
    scene["axes"]["forward"] = {
        **_grounding(),
        "frame": "world",
        "vector_world": [1.0, 0.0, 0.0],
        "object_ref": "target",
    }
    program_ir = {
        "schema_version": "v0.1",
        "program_id": "coupled_axis_test",
        "task_instruction": "align primary axis and choose roll from secondary axis",
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [{"name": "align", "repeat": 1}],
        "constraints": [
            {
                "id": "primary_axis",
                "stage": "align",
                "type": "align_axis",
                "refs": {
                    "source_axis": "tool.functional_parts.tip.axis",
                    "target_axis": "scene.axes.forward",
                },
                "params": {"polarity": "same", "min_cosine": 0.95},
                "hard": True,
                "weight": 1.0,
                "source": "unit_test",
            },
            {
                "id": "secondary_roll_axis",
                "stage": "align",
                "type": "align_axis",
                "refs": {
                    "source_axis": "tool.functional_parts.tip.normal",
                    "target_axis": "scene.axes.up",
                },
                "params": {"polarity": "same", "min_cosine": 0.95},
                "hard": True,
                "weight": 1.0,
                "source": "unit_test",
            },
        ],
        "objectives": [],
        "diagnostics": {"ast_checked": True, "executed": True},
    }
    validate_program_ir(program_ir)

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=scene,
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [1.0, 0.0, 0.0, 0.0],
        },
    )

    goal_orientation = subgoals["goals"][-1]["pose_world_tool"]["orientation_xyzw"]
    primary_world = quat_rotate(goal_orientation, [1.0, 0.0, 0.0])
    secondary_world = quat_rotate(goal_orientation, [0.0, 0.0, 1.0])
    assert norm(sub(primary_world, [1.0, 0.0, 0.0])) < 1e-6
    assert norm(sub(secondary_world, [0.0, 0.0, 1.0])) < 1e-6
    resolution = subgoals["sanity_checks"]["compiler_stages"][0]["pose_selection"]["axis_resolutions"][0]
    assert resolution["coupled_axis_count"] == 2


def test_auto_axis_alignment_preserves_motion_axis_direction() -> None:
    program_ir = {
        "schema_version": "v0.1",
        "program_id": "auto_axis_test",
        "task_instruction": "touch the goal with the lower-rotation axis polarity",
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [{"name": "contact", "repeat": 1}],
        "constraints": [
            {
                "id": "place_tip",
                "stage": "contact",
                "type": "contact",
                "refs": {
                    "source": "tool.functional_parts.tip.center",
                    "target": "scene.keypoints.goal",
                },
                "params": {"tolerance_m": 0.01},
                "hard": True,
                "weight": 1.0,
                "source": "unit_test",
            },
            {
                "id": "auto_axis",
                "stage": "contact",
                "type": "align_axis",
                "refs": {
                    "source_axis": "tool.functional_parts.tip.normal",
                    "target_axis": "scene.axes.up",
                },
                "params": {"polarity": "auto", "min_cosine": 0.85},
                "hard": True,
                "weight": 1.0,
                "source": "unit_test",
            },
        ],
        "objectives": [],
        "diagnostics": {"ast_checked": True, "executed": True},
    }
    validate_program_ir(program_ir)
    start_orientation = [1.0, 0.0, 0.0, 0.0]

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": start_orientation,
        },
    )

    goal = subgoals["goals"][-1]
    assert quat_angle_rad(goal["pose_world_tool"]["orientation_xyzw"], start_orientation) > 3.0
    stage_errors = subgoals["sanity_checks"]["compiler_stages"][0]["constraint_errors"]
    assert stage_errors["auto_axis.axis_cosine"] == 1.0
    pose_selection = subgoals["sanity_checks"]["compiler_stages"][0]["pose_selection"]
    assert pose_selection["algorithm"] == "constrained_minimum_change_v0"
    assert pose_selection["axis_resolutions"][0]["requested_polarity"] == "auto"
    assert pose_selection["axis_resolutions"][0]["resolved_polarity"] == "same"


def test_axis_alignment_keeps_selected_axis_even_when_alternative_needs_less_rotation() -> None:
    scene = _scene_for_objective()
    scene["axes"]["side"] = {
        **_grounding(),
        "frame": "world",
        "vector_world": [0.0, 1.0, 0.0],
        "object_ref": "target",
    }
    tool = _tool_for_objective()
    tool["functional_parts"]["tip"]["axis_candidates"] = [
        {
            "candidate_id": "side_tangent_axis",
            "vector_tool": [0.0, 1.0, 0.0],
            "source": "unit_test",
        }
    ]
    program_ir = {
        "schema_version": "v0.1",
        "program_id": "axis_candidate_test",
        "task_instruction": "align using the most natural equivalent tool axis",
        "api_version": "parttoolgen_program_api_v0.3",
        "stages": [{"name": "align", "repeat": 1}],
        "constraints": [
            {
                "id": "candidate_axis",
                "stage": "align",
                "type": "align_axis",
                "refs": {
                    "source_axis": "tool.functional_parts.tip.axis",
                    "target_axis": "scene.axes.side",
                },
                "params": {"polarity": "same", "min_cosine": 0.95},
                "hard": True,
                "weight": 1.0,
                "source": "unit_test",
            },
        ],
        "objectives": [],
        "diagnostics": {"ast_checked": True, "executed": True},
    }
    validate_program_ir(program_ir)
    start_orientation = [0.0, 0.0, 0.0, 1.0]

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=scene,
        functional_tool_parts=tool,
        program_ir=program_ir,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": start_orientation,
        },
    )

    goal = subgoals["goals"][-1]
    assert abs(quat_angle_rad(goal["pose_world_tool"]["orientation_xyzw"], start_orientation) - 1.5707963267948966) < 1e-8
    stage = subgoals["sanity_checks"]["compiler_stages"][0]
    assert stage["constraint_errors"]["candidate_axis.axis_cosine"] == 1.0
    resolution = stage["pose_selection"]["axis_resolutions"][0]
    assert resolution["source_axis_candidate_count"] == 1
    assert resolution["selected_source_axis_candidate_id"] == "primary_axis"
    assert resolution["selected_source_axis_tool"] == [1.0, 0.0, 0.0]


def test_compiler_records_goal_pose_delta_metrics() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def build_program(ctx):
    stage = ctx.stage("contact")
    stage.constraint(ctx.constraints.contact(ctx.ref("tool.functional_parts.tip.center"), ctx.ref("scene.keypoints.goal")))
''',
    )

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        start_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        planner_program=planner,
    )

    sanity = subgoals["sanity_checks"]
    assert sanity["pose_delta_summary"]["max_translation_delta_m"] >= 0.0
    assert sanity["pose_delta_summary"]["max_rotation_delta_rad"] >= 0.0
    assert sanity["pose_delta_summary"]["net_translation_m"] >= 0.0
    assert sanity["pose_delta_summary"]["net_rotation_rad"] >= 0.0
    assert sanity["pose_delta_summary"]["translation_path_ratio"] >= 1.0
    assert sanity["pose_delta_summary"]["rotation_path_ratio"] >= 0.0
    assert sanity["goal_pose_deltas"][0]["goal_index"] == 0
    assert "translation_delta_m" in sanity["goal_pose_deltas"][0]


def test_stage_objective_optimizes_whole_tool_contact_point() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def tip_to_goal(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(tip_to_goal, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )

    goal = subgoals["goals"][-1]
    assert subgoals["sanity_checks"]["compiled_objectives"] == 1
    assert subgoals["sanity_checks"]["ignored_objectives"] == 0
    assert norm(sub(goal["contact_point_goal"]["position_world"], [1.0, 0.0, 0.0])) < 0.03
    stage_errors = subgoals["sanity_checks"]["compiler_stages"][0]["constraint_errors"]
    assert stage_errors["place_tip_objective_1.objective_residual"] < 0.03


def test_stage_objective_cannot_access_unlisted_refs() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def tip_to_secret(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    secret = refs.world_point("scene.keypoints.secret")
    return np.linalg.norm(tip - secret)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(tip_to_secret, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    try:
        compile_program_ir_to_tool_pose_subgoals(
            observed_scene_graph=_scene_for_objective(),
            functional_tool_parts=_tool_for_objective(),
            program_ir=program_ir,
            planner_program=planner,
            start_pose={
                "position": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
    except ProgramIRCompileError as exc:
        assert "outside objective whitelist" in str(exc)
    else:
        raise AssertionError("objective accessed a ref outside its whitelist")


def test_stage_objective_rejects_omitted_refs() -> None:
    try:
        _program_ir_from_source(
            '''
def tip_to_goal(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    stage = ctx.stage("place_tip")
    stage.objective(tip_to_goal, kind="subgoal", weight=20.0)
    return ctx.program()
'''
        )
    except Exception as exc:
        assert "objective refs must be an explicit non-empty list" in str(exc)
    else:
        raise AssertionError("objective without explicit refs was accepted")


def test_stage_path_objective_is_deferred_to_path_solver() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def tip_to_goal(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(tip_to_goal, kind="path", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )

    assert subgoals["sanity_checks"]["compiled_objectives"] == 0
    assert subgoals["sanity_checks"]["ignored_objectives"] == 0
    assert subgoals["sanity_checks"]["path_objectives_declared"] == 1
    assert subgoals["sanity_checks"]["path_objectives_deferred"] == 1
    assert subgoals["sanity_checks"]["path_objective_runtime"] == "deferred_to_tool_path_solver"


def test_program_ir_validation_rejects_invalid_objective_refs() -> None:
    _, program_ir = _program_ir_from_source(
        '''
def tip_to_goal(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(tip_to_goal, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )
    program_ir["objectives"][0]["refs"] = ["not_a_symbol_ref"]
    try:
        validate_program_ir(program_ir)
    except Exception as exc:
        assert "must be a symbol reference" in str(exc)
    else:
        raise AssertionError("invalid objective ref string was accepted")


def test_tool_occupancy_constraints_compile_to_path_metadata() -> None:
    source = '''
def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    pre = ctx.ref("scene.keypoints.goal")
    tool_part = ctx.ref("tool.parts.part_001")
    obstacle = ctx.ref("scene.objects.target")
    stage = ctx.stage("approach")
    stage.constraint(ctx.constraints.contact(tip, pre))
    stage.constraint(ctx.constraints.tool_occupancy_clearance(tool_part, obstacle, min_distance_m=0.006))
    return ctx.program()
'''
    planner = _planner_artifact(source)
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "tool.parts.part_001",
            "scene.keypoints.goal",
            "scene.objects.target",
        },
    )
    program_ir = execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )

    validate_program_ir(program_ir)
    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=_scene_for_objective(),
        functional_tool_parts=_tool_for_objective(),
        program_ir=program_ir,
        planner_program=planner,
        tool_part_graph=_tool_part_graph(),
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )

    path_constraints = subgoals["sanity_checks"]["compiler_stages"][0]["path_constraints"]
    assert path_constraints["tool_occupancy_clearance"][0]["tool_part_ref"] == "tool.parts.part_001"
    assert path_constraints["tool_occupancy_clearance"][0]["num_sample_points"] == 9


def test_stage_objective_cannot_use_numpy_file_api() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def unsafe_numpy_io(pose, refs):
    np.load("/tmp/should_not_be_loaded.npy")
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(unsafe_numpy_io, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    try:
        compile_program_ir_to_tool_pose_subgoals(
            observed_scene_graph=_scene_for_objective(),
            functional_tool_parts=_tool_for_objective(),
            program_ir=program_ir,
            planner_program=planner,
            start_pose={
                "position": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
    except ProgramIRCompileError as exc:
        assert "load" in str(exc)
    else:
        raise AssertionError("NumPy filesystem-capable API was available to objective code")


def test_stage_objective_cannot_call_array_file_methods() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def unsafe_array_method(pose, refs):
    np.asarray([1.0]).tofile("/tmp/should_not_be_written.bin")
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(unsafe_array_method, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    try:
        compile_program_ir_to_tool_pose_subgoals(
            observed_scene_graph=_scene_for_objective(),
            functional_tool_parts=_tool_for_objective(),
            program_ir=program_ir,
            planner_program=planner,
            start_pose={
                "position": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
    except ProgramIRCompileError as exc:
        assert "unsupported attribute tofile" in str(exc)
    else:
        raise AssertionError("objective was allowed to call an ndarray file method")


def test_stage_objective_cannot_bind_forbidden_array_method() -> None:
    planner, program_ir = _program_ir_from_source(
        '''
def unsafe_bound_method(pose, refs):
    writer = np.asarray([1.0]).tofile
    writer("/tmp/should_not_be_written.bin")
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    goal = refs.world_point("scene.keypoints.goal")
    return np.linalg.norm(tip - goal)

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    goal = ctx.ref("scene.keypoints.goal")
    stage = ctx.stage("place_tip")
    stage.objective(unsafe_bound_method, kind="subgoal", weight=20.0, refs=[tip, goal])
    return ctx.program()
'''
    )

    try:
        compile_program_ir_to_tool_pose_subgoals(
            observed_scene_graph=_scene_for_objective(),
            functional_tool_parts=_tool_for_objective(),
            program_ir=program_ir,
            planner_program=planner,
            start_pose={
                "position": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
    except ProgramIRCompileError as exc:
        assert "unsupported attribute tofile" in str(exc)
    else:
        raise AssertionError("objective was allowed to bind an ndarray file method")
