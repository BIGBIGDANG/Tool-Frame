from __future__ import annotations

import pytest

from ToGo.compiler.tool_pose_compiler import compile_program_ir_to_tool_pose_subgoals
from ToGo.image_grounded import (
    build_contact_motion_program,
    build_mainline_inputs_from_image_grounded_program,
)
from ToGo.image_grounded.staged_mainline_adapter import (
    ImageGroundedMainlineAdapterError,
    _validate_image_grounded_program_ir_contract,
)
from ToGo.validation.validate_parttoolgen import validate_program_ir


def _program(
    *,
    instruction: str = "Use the tool to move the block.",
    tool_motion_axis_ref: str | None = None,
    tool_motion_axis_target_ref: str | None = None,
    tool_motion_axis_relation: str | None = "parallel",
    include_stages: bool = True,
) -> dict:
    program = build_contact_motion_program(
        task_id="unit_image_grounded",
        instruction=instruction,
        tool_motion_axis_ref=tool_motion_axis_ref,
        tool_motion_axis_target_ref=tool_motion_axis_target_ref,
        tool_motion_axis_relation=tool_motion_axis_relation,
        tool_refs={
            "tool_id": "unit_brush",
            "contact_point_tool": [0.1, 0.0, 0.02],
            "motion_axis_tool": [0.0, 1.0, 0.0],
        },
        scene_refs={
            "object_point": {"position_world": [0.1, -0.05, 0.575]},
            "start": {"position_world": [0.1, -0.05, 0.575]},
            "end": {"position_world": [0.1, 0.05, 0.575]},
            "direction": [0.0, 1.0, 0.0],
        },
    ).to_json()
    if not include_stages:
        program.pop("stages")
    return program


def _start_end_use_program() -> dict:
    return build_contact_motion_program(
        task_id="unit_image_grounded",
        instruction="Use the tool to move the block.",
        scene_contact_ref="scene.start",
        scene_action_end_ref="scene.end",
        scene_direction_ref="scene.direction",
        tool_motion_axis_target_ref="scene.direction",
        tool_motion_axis_relation="parallel",
        tool_refs={
            "tool_id": "unit_brush",
            "contact_point_tool": [0.1, 0.0, 0.02],
            "motion_axis_tool": [0.0, 1.0, 0.0],
        },
        scene_refs={
            "object_point": {"position_world": [0.1, -0.05, 0.575]},
            "start": {"position_world": [0.1, -0.05, 0.575]},
            "end": {"position_world": [0.1, 0.05, 0.575]},
            "direction": [0.0, 1.0, 0.0],
        },
    ).to_json()


def _program_with_candidate_faces() -> dict:
    program = _program()
    program["tool_refs"]["contact_point_candidates_tool"] = [
        {
            "contact_point_tool": [0.1, 0.0, 0.02],
            "contact_point_face_marker": "F_negative_y",
            "contact_point_source_ref": "candidate.points.negative_y",
            "contact_point_surface_ref": "candidate.surfaces.negative_y",
        },
        {
            "contact_point_tool": [0.1, 0.0, 0.02],
            "contact_point_face_marker": "F_positive_y",
            "contact_point_source_ref": "candidate.points.positive_y",
            "contact_point_surface_ref": "candidate.surfaces.positive_y",
        },
    ]
    return program


def _task() -> dict:
    return {
        "execution": {
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0],
            }
        },
    }


def _constraint(program_ir: dict, constraint_id: str) -> dict:
    return next(item for item in program_ir["constraints"] if item["id"] == constraint_id)


def _stage_names(program_ir: dict) -> list[str]:
    return [stage["name"] for stage in program_ir["stages"]]


def _constraints_for_stage(program_ir: dict, stage_name: str) -> list[dict]:
    return [item for item in program_ir["constraints"] if item["stage"] == stage_name]


def _goal_stages(subgoals: dict) -> list[str]:
    return [goal["stage"] for goal in subgoals["goals"]]


def test_program_axis_align_to_motion_is_lowered_without_family_fallback() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program(
        ),
        task_data=_task(),
    )
    program_ir = inputs["program_ir"]
    validate_program_ir(program_ir)
    assert _stage_names(program_ir) == ["align", "approach", "action"]
    align = _constraint(program_ir, "align_axis_face_target")
    approach = _constraint(program_ir, "approach_pre_contact_offset")
    engage = _constraint(program_ir, "action_axis_direction")
    assert align["stage"] == "align"
    assert align["refs"]["target_axis"] == "target.axes.direction"
    assert align["params"]["polarity"] == "same"
    assert approach["stage"] == "approach"
    assert approach["type"] == "pre_contact_offset"
    assert approach["refs"]["target"] == "scene.keypoints.start"
    assert approach["refs"]["axis"] == "target.axes.direction"
    assert approach["params"]["direction"] == "opposite_axis"
    assert all(item["type"] != "align_axis" for item in _constraints_for_stage(program_ir, "approach"))
    assert engage["refs"]["target_axis"] == "target.axes.direction"
    assert engage["params"]["polarity"] == "same"


def test_configured_approach_offset_reaches_program_ir() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program(),
        task_data=_task(),
        approach_offset_m=0.12,
    )

    derived = inputs["derived_scene_geometry"]
    approach = _constraint(inputs["program_ir"], "approach_pre_contact_offset")
    assert derived["approach_offset_m"] == pytest.approx(0.12)
    assert derived["pre_start_point"] == pytest.approx([0.1, -0.17, 0.575])
    assert approach["params"]["distance_m"] == pytest.approx(0.12)
    assert (
        inputs["program_ir"]["diagnostics"]["explicit_axis_bindings"]["approach"][
            "distance_m"
        ]
        == pytest.approx(0.12)
    )


def test_compiled_image_grounded_stages_align_before_approach() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program(),
        task_data=_task(),
        start_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )

    subgoals = compile_program_ir_to_tool_pose_subgoals(
        observed_scene_graph=inputs["observed_scene_graph"],
        functional_tool_parts=inputs["functional_tool_parts"],
        program_ir=inputs["program_ir"],
        start_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        target_part_graph=inputs["target_part_graph"],
        tool_part_graph=inputs["tool_part_graph"],
        initial_grasp_lift_m=0.08,
        initial_grasp_lift_steps=1,
    )
    stages = _goal_stages(subgoals)

    assert "approach_tool_axis_align" not in stages
    assert "align" in stages
    assert stages.index("grasp_lift") < stages.index("align")
    assert stages.index("align") < stages.index("approach")
    assert stages.index("approach") < stages.index("action_waypoint_1")

    align_goal = next(goal for goal in subgoals["goals"] if goal["stage"] == "align")
    lift_goal = next(goal for goal in subgoals["goals"] if goal["stage"] == "grasp_lift")
    assert align_goal["pose_world_tool"]["position"] == pytest.approx(
        lift_goal["pose_world_tool"]["position"]
    )
    align_stage = next(
        stage for stage in subgoals["sanity_checks"]["compiler_stages"] if stage["name"] == "align"
    )
    axis_record = align_stage["pose_selection"]["axis_resolutions"][0]
    assert axis_record["axis_roll_resolution_priority"] == (
        "axis_alignment_tool_geometry_axis_line_distance_contact_target"
    )


def test_motion_axis_cannot_target_scene_surface_normal_metadata() -> None:
    program = _program(
        tool_motion_axis_ref="tool.motion_axis",
        tool_motion_axis_target_ref="target.axes.surface_normal",
        tool_motion_axis_relation="opposite",
    )
    program["scene_refs"] = {
        "object_point": {"position_world": [0.1, -0.05, 0.575]},
        "start": {"position_world": [0.1, -0.05, 0.575]},
        "end": {"position_world": [0.1, 0.05, 0.575]},
        "direction": [0.0, 1.0, 0.0],
        "target_surface_normal": {"vector_world": [0.0, 0.0, 1.0]},
    }
    with pytest.raises(ImageGroundedMainlineAdapterError, match="unsupported explicit target axis ref"):
        build_mainline_inputs_from_image_grounded_program(program, task_data=_task())


def test_start_end_use_refs_lower_without_family_fallback() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _start_end_use_program(),
        task_data=_task(),
    )
    program_ir = inputs["program_ir"]
    validate_program_ir(program_ir)
    align = _constraint(program_ir, "align_axis_face_target")
    engage = _constraint(program_ir, "action_axis_direction")
    assert align["refs"]["target_axis"] == "target.axes.direction"
    assert engage["refs"]["target_axis"] == "target.axes.direction"
    contact_patch = inputs["target_part_graph"]["parts"]["target_contact_region"]["surface_patches"][
        "contact_patch"
    ]
    assert contact_patch["center_xyz_world"] == [0.1, -0.05, 0.575]


def test_missing_staged_axis_constraints_fail_instead_of_guessing_family_defaults() -> None:
    with pytest.raises(Exception, match="no task-family fallback"):
        build_mainline_inputs_from_image_grounded_program(
            _program(include_stages=False),
            task_data=_task(),
        )


def test_missing_align_axis_constraint_fails_instead_of_borrowing_action_axis() -> None:
    program = _program()
    align = program["stages"][0]
    align["subgoal_constraints"] = [
        row for row in align["subgoal_constraints"] if row["type"] != "axis_align"
    ]
    align["path_constraints"] = [
        row for row in align["path_constraints"] if row["type"] != "axis_align"
    ]

    with pytest.raises(ImageGroundedMainlineAdapterError, match="no constraints for stage 'align'"):
        build_mainline_inputs_from_image_grounded_program(program, task_data=_task())


def test_start_axis_is_not_used_as_direction_fallback() -> None:
    program = _program()
    program["scene_refs"].pop("direction")
    program["scene_refs"]["start_axis"] = [0.0, 1.0, 0.0]
    with pytest.raises(Exception, match="scene_refs.direction"):
        build_mainline_inputs_from_image_grounded_program(program, task_data=_task())


def test_missing_surface_normal_uses_motion_axis_only_for_pre_contact_not_axis_binding() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program(),
        task_data=_task(),
    )
    derived = inputs["derived_scene_geometry"]
    assert derived["target_surface_normal"] is None
    assert derived["target_surface_normal_source"] == "not_required_for_direction_contact"
    assert derived["target_approach_axis"] == pytest.approx([0.0, -1.0, 0.0])
    assert derived["pre_start_point"] == pytest.approx([0.1, -0.08, 0.575])


def test_surface_normal_axis_target_is_rejected_instead_of_required() -> None:
    program = _program(
        tool_motion_axis_target_ref="target.axes.surface_normal",
        tool_motion_axis_relation="opposite",
    )
    with pytest.raises(Exception, match="unsupported explicit target axis ref"):
        build_mainline_inputs_from_image_grounded_program(program, task_data=_task())


def test_contact_point_candidates_are_preserved_without_compiler_side_face_selection() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program_with_candidate_faces(),
        task_data=_task(),
        start_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )
    tool_refs = inputs["selected_constraint_program"]["tool_refs"]
    assert "active_contact_point_candidate_selection" not in tool_refs
    assert "contact_point_face_marker" not in tool_refs
    assert tool_refs["contact_point_candidates_tool"][0]["contact_point_face_marker"] == "F_negative_y"
    assert tool_refs["contact_point_candidates_tool"][1]["contact_point_face_marker"] == "F_positive_y"


def test_contact_point_candidates_are_not_reselected_from_axis_relation() -> None:
    program = _program_with_candidate_faces()
    program["stages"][0]["subgoal_constraints"][0]["params"]["relation"] = "same"
    action_axis = program["stages"][2]["path_constraints"][1]["params"]
    action_axis["relation"] = "same"
    inputs = build_mainline_inputs_from_image_grounded_program(
        program,
        task_data=_task(),
        start_pose={"position": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    )
    tool_refs = inputs["selected_constraint_program"]["tool_refs"]
    assert "active_contact_point_candidate_selection" not in tool_refs
    assert "contact_point_face_marker" not in tool_refs


def test_target_object_is_not_static_obstacle_for_path_solver() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(
        _program(),
        task_data=_task(),
    )
    target_attrs = inputs["observed_scene_graph"]["objects"]["target_object"]["attributes"]
    assert target_attrs["collision_role"] == "dynamic_manipulable"


def test_tabletop_wipe_uses_vlm_direction_even_with_surface_normal_metadata() -> None:
    program = build_contact_motion_program(
        task_id="unit_tabletop_wipe",
        instruction="Use the eraser to wipe across the white HDU sticker.",
        contact_approach_axis_ref="target.axes.target_approach_axis",
        tool_motion_axis_target_ref="scene.direction",
        tool_motion_axis_relation="parallel",
        tool_refs={
            "tool_id": "unit_eraser",
            "contact_point_tool": [0.0, 0.0, 0.02],
            "motion_axis_tool": [1.0, 0.0, 0.0],
        },
        scene_refs={
            "object_point": {"position_world": [0.18, -0.08, 0.533]},
            "start": {"position_world": [0.08, -0.08, 0.533]},
            "end": {"position_world": [0.22, -0.08, 0.533]},
            "direction": [1.0, 0.0, 0.0],
            "target_surface_normal": {"vector_world": [0.0, 0.0, 1.0]},
            "target_approach_axis": {"vector_world": [0.0, 0.0, 1.0]},
        },
    ).to_json()
    task_data = {
        "execution": {
            "state_bundle": {
                "table_pose_xyzw": [0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0],
            }
        }
    }

    inputs = build_mainline_inputs_from_image_grounded_program(
        program,
        task_data=task_data,
    )

    derived = inputs["derived_scene_geometry"]
    assert derived["direction"] == [1.0, 0.0, 0.0]
    assert derived["target_surface_normal"] == [0.0, 0.0, 1.0]
    assert derived["pre_start_point"] == pytest.approx([0.05, -0.08, 0.533])
    assert derived["end_point"] == pytest.approx([0.22, -0.08, 0.533])
    assert derived["motion_distance_m"] == pytest.approx(0.14)
    align = _constraint(inputs["program_ir"], "align_axis_face_target")
    engage = _constraint(inputs["program_ir"], "action_axis_direction")
    assert align["refs"]["target_axis"] == "target.axes.direction"
    assert engage["refs"]["target_axis"] == "target.axes.direction"
    assert align["params"]["polarity"] == "same"
    assert engage["params"]["polarity"] == "same"
    move_along = _constraint(inputs["program_ir"], "action_move_along_vlm_path")
    assert move_along["refs"]["start"] == "scene.keypoints.start"
    assert move_along["refs"]["end"] == "scene.keypoints.end"


def test_generated_program_ir_contract_rejects_pre_contact_contact_follow_start() -> None:
    inputs = build_mainline_inputs_from_image_grounded_program(_program(), task_data=_task())
    program_ir = inputs["program_ir"]
    move_along = _constraint(program_ir, "action_move_along_vlm_path")
    move_along["refs"]["start"] = "scene.keypoints.target_pre_contact_point"
    with pytest.raises(ImageGroundedMainlineAdapterError, match="public start/path keypoint"):
        _validate_image_grounded_program_ir_contract(program_ir)
