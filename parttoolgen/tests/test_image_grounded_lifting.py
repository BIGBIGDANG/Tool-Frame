from __future__ import annotations

import math
import importlib.util
from pathlib import Path

from parttoolgen.image_grounded import (
    CameraIntrinsics,
    ImageKeypoint,
    build_contact_motion_program,
    derive_axis,
    lift_keypoint,
)


def _load_lift_stage_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "cli" / "stages" / "lift_scene_keypoints_to_constraint_program.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_lift_stage", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lift_stage = _load_lift_stage_module()


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


def test_lift_stage_uses_ordered_point_refs_when_primary_direction_is_zero() -> None:
    axis, source = lift_stage._derive_axis_from_point_refs(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        primary_source="action_direction_arrow_2d_lifted",
        ordered_point_refs=[
            {"position_world": [0.0, 0.0, 0.0]},
            {"position_world": [0.0, 0.2, 0.0]},
            {"position_world": [0.1, 0.2, 0.0]},
        ],
    )

    assert axis == (0.0, 1.0, 0.0)
    assert source == "ordered_point_refs_segment_0_1"


def test_lift_stage_reports_grounding_error_when_no_direction_exists() -> None:
    try:
        lift_stage._derive_axis_from_point_refs(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            primary_source="action_direction_arrow_2d_lifted",
            ordered_point_refs=[
                {"position_world": [0.0, 0.0, 0.0]},
                {"position_world": [0.0, 0.0, 0.0]},
            ],
        )
    except ValueError as exc:
        assert "did not provide a non-zero use-stage direction" in str(exc)
    else:
        raise AssertionError("zero-length direction without path fallback should fail")


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


def test_default_axis_decision_uses_opposite_support_surface_normal() -> None:
    decision = lift_stage._direction_decision(
        {
            "direction_decision": {
                "source": "default_axis_candidate",
                "selected_default_axis_id": "opposite_support_surface_normal",
                "confidence": 0.8,
            }
        }
    )
    axis, source = lift_stage._axis_from_default_direction_decision(
        decision,
        support_surface_normal_world=[0.0, 0.0, 2.0],
    )

    assert axis == [0.0, 0.0, -1.0]
    assert source == "vlm_selected_default_axis:opposite_support_surface_normal"


def test_default_axis_decision_requires_public_support_normal() -> None:
    decision = lift_stage._direction_decision(
        {
            "direction_decision": {
                "source": "default_axis_candidate",
                "selected_default_axis_id": "support_surface_normal",
            }
        }
    )
    try:
        lift_stage._axis_from_default_direction_decision(
            decision,
            support_surface_normal_world=None,
        )
    except ValueError as exc:
        assert "no public support surface normal" in str(exc)
    else:
        raise AssertionError("default axis without support normal should fail")
