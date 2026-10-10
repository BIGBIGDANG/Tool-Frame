from __future__ import annotations
import math
from ToGo.image_grounded import CameraIntrinsics, ImageKeypoint, build_contact_motion_program, derive_axis, lift_keypoint

def test_lift_keypoint_identity_camera() -> None:
    intrinsics = CameraIntrinsics(width=640, height=480, fx=100.0, fy=100.0, cx=320.0, cy=240.0)
    lifted = lift_keypoint(
        ImageKeypoint("contact", 330.0, 250.0, "target_contact", 0.8),
        depth_m=2.0,
        intrinsics=intrinsics,
        camera_to_world=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert lifted.position_camera == (0.2, 0.2, 2.0)
    assert lifted.position_world == (0.2, 0.2, 2.0)


def test_derive_axis_normalizes_contact_to_goal() -> None:
    axis = derive_axis((1.0, 2.0, 3.0), (1.0, 5.0, 7.0))
    assert math.isclose(sum(v * v for v in axis), 1.0)
    assert axis == (0.0, 0.6, 0.8)


def test_contact_motion_program_has_expected_stages() -> None:
    program = build_contact_motion_program(
        task_id="unit",
        instruction="Move the target with the selected tool face.",
    )
    names = [stage.name for stage in program.stages]
    assert names == ["align", "approach", "action"]
    serialized = program.to_json()
    assert serialized["schema_version"] == "toolframe_constraint_program_v1"
    assert serialized["metadata"]["ir_name"] == "Tool-Scene Constraint IR"
    assert serialized["metadata"]["constraint_target_types"] == [
        "PointTarget",
        "AxisTarget",
        "MoveAlongTarget",
        "OrientationTarget",
        "ClearanceTarget",
    ]
    assert serialized["metadata"]["semantic_ref_model"] == "start_end_action_refs"
    assert serialized["metadata"]["stage_ref_roles"]["align"] == "tool_axis_alignment"
    assert serialized["metadata"]["stage_ref_roles"]["approach"] == "start_approach"
    assert serialized["metadata"]["axis_role_bindings"]["tool_motion_axis"]["target_axis_ref"] == "scene.direction"
    assert serialized["stages"][0]["subgoal_constraints"][0]["type"] == "axis_align"
    assert serialized["stages"][1]["subgoal_constraints"][0]["type"] == "point_offset"
    assert serialized["stages"][2]["subgoal_constraints"][0]["type"] == "point_match"
    assert serialized["stages"][2]["path_constraints"][0]["type"] == "move_along"
    assert program.stages[2].constraints[-1].constraint_type == "axis_align"


def test_motion_axis_binding_uses_scene_direction() -> None:
    program = build_contact_motion_program(
        task_id="unit",
        instruction="Move the selected tool contact point along the marked scene direction.",
        tool_motion_axis_ref="tool.motion_axis",
        tool_motion_axis_target_ref="scene.direction",
        tool_motion_axis_relation="parallel",
    )
    serialized = program.to_json()
    action = serialized["stages"][2]["path_constraints"]
    assert action[0]["params"]["direction_ref"] == "scene.direction"
    assert action[1]["params"]["target_axis_ref"] == "scene.direction"
    assert action[1]["params"]["tool_axis_ref"] == "tool.motion_axis"
