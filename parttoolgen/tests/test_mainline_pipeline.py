from __future__ import annotations

from parttoolgen.pipeline.mainline import (
    _lift_tool_pose_subgoals_to_table_clearance,
    run_mainline_fixed_goal_chain,
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


def _scene() -> dict:
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
            "target": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [0.8, 0.0, 0.0],
                "object_ref": "target",
            },
            "limit": {
                **_grounding(),
                "frame": "world",
                "xyz_world": [1.0, 0.0, 0.0],
                "object_ref": "target",
            },
        },
        "axes": {},
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


def _tool() -> dict:
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


def _planner_and_program_ir() -> tuple[dict, dict]:
    source = '''
def stay_before_limit(pose, refs):
    tip = refs.tool_point(pose, "tool.functional_parts.tip.center")
    limit = refs.world_point("scene.keypoints.limit")
    return tip[0] - limit[0]

def build_program(ctx):
    tip = ctx.ref("tool.functional_parts.tip.center")
    target = ctx.ref("scene.keypoints.target")
    limit = ctx.ref("scene.keypoints.limit")
    stage = ctx.stage("move")
    stage.constraint(ctx.constraints.contact(tip, target))
    stage.objective(stay_before_limit, kind="path", weight=1.0, refs=[tip, limit])
    return ctx.program()
'''
    planner = make_planner_program_artifact(
        raw_text=source,
        program_id="mainline_path_objective",
        task_instruction="touch the target without crossing the limit",
        generated_by="unit_test",
    )
    context = ProgramContext(
        program_id=planner["program_id"],
        task_instruction=planner["task_instruction"],
        allowed_refs={
            "tool.functional_parts.tip.center",
            "scene.keypoints.target",
            "scene.keypoints.limit",
        },
    )
    program_ir = execute_planner_program(
        program_text=planner["program_text"],
        context=context,
    )
    return planner, program_ir


def test_mainline_chain_routes_deferred_path_objective_through_path_solver(tmp_path) -> None:
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
    planner, program_ir = _planner_and_program_ir()

    result = run_mainline_fixed_goal_chain(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        tool_mesh_path=mesh_path,
    )

    subgoals = result["tool_pose_subgoals"]
    solved = result["path_solved_subgoals"]
    path_report = result["tool_path_solver_report"]
    fixed_goals = result["simtoolreal_fixed_goals"]
    roles = [goal.get("semantic_role") for goal in subgoals["goals"]]
    assert "grasp_lift" in roles
    assert subgoals["sanity_checks"]["initial_grasp_lift_m"] == 0.08
    assert subgoals["sanity_checks"]["initial_grasp_lift_inserted"] is True
    assert subgoals["sanity_checks"]["path_objectives_deferred"] == 1
    assert path_report["path_objectives_checked"] is True
    assert path_report["passed"] is True
    assert solved["sanity_checks"]["tool_path_solver"]["path_objectives_checked"] is True
    assert fixed_goals["metadata"]["source_subgoals_id"] == solved["subgoals_id"]


def test_mainline_can_project_start_pose_to_table_contact(tmp_path) -> None:
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 -0.10\nv 0 0 0.10\n", encoding="utf-8")
    table_path = tmp_path / "table.urdf"
    table_path.write_text(
        """
<robot name="table">
  <link name="table">
    <collision>
      <origin xyz="0 0 0" />
      <geometry><box size="1 1 0.20" /></geometry>
    </collision>
  </link>
</robot>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    planner, program_ir = _planner_and_program_ir()

    result = run_mainline_fixed_goal_chain(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.50],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0],
            "table_asset": str(table_path),
        },
        tool_mesh_path=mesh_path,
        start_min_tool_table_clearance_m=0.003,
    )

    start_pose = result["tool_pose_subgoals"]["start_pose"]
    assert start_pose["position"] == [0.0, 0.0, 0.403]
    assert result["simtoolreal_fixed_goals"]["start_pose"] == [
        0.0,
        0.0,
        0.403,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["simtoolreal_fixed_goals"]["goals"][0] == result["simtoolreal_fixed_goals"]["start_pose"]
    assert result["simtoolreal_fixed_goals"]["metadata"]["start_goal_in_fixed_goals"] is True
    assert result["start_pose_report"]["mode"] == "mesh_table_contact_projection"


def test_mainline_prefers_current_start_pose_over_table_projection(tmp_path) -> None:
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 -0.10\nv 0 0 0.10\n", encoding="utf-8")
    table_path = tmp_path / "table.urdf"
    table_path.write_text(
        """
<robot name="table">
  <link name="table">
    <collision>
      <origin xyz="0 0 0" />
      <geometry><box size="1 1 0.20" /></geometry>
    </collision>
  </link>
</robot>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    planner, program_ir = _planner_and_program_ir()
    current_pose = {
        "position": [0.0, 0.0, 0.407],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    result = run_mainline_fixed_goal_chain(
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        program_ir=program_ir,
        planner_program=planner,
        start_pose={
            "position": [0.0, 0.0, 0.50],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        current_start_pose=current_pose,
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0],
            "table_asset": str(table_path),
        },
        tool_mesh_path=mesh_path,
    )

    assert result["tool_pose_subgoals"]["start_pose"] == current_pose
    assert result["simtoolreal_fixed_goals"]["start_pose"] == [
        0.0,
        0.0,
        0.407,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["start_pose_report"]["mode"] == "observed_current_object_pose"


def test_table_clearance_projection_preserves_contact_path_contact_anchor(
    tmp_path,
) -> None:
    mesh_path = tmp_path / "point_tool.obj"
    mesh_path.write_text("v 0 0 -0.10\nv 0 0 0.10\n", encoding="utf-8")
    table_path = tmp_path / "table.urdf"
    table_path.write_text(
        """
<robot name="table">
  <link name="table">
    <collision>
      <origin xyz="0 0 0" />
      <geometry><box size="1 1 0.20" /></geometry>
    </collision>
  </link>
</robot>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    subgoals = {
        "goals": [
            {
                "index": 0,
                "stage": "approach_tool_axis_align",
                "pose_world_tool": {
                    "position": [0.0, 0.0, 0.15],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                },
            },
            {
                "index": 1,
                "stage": "engage_waypoint_1",
                "pose_world_tool": {
                    "position": [0.0, 0.0, 0.15],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "contact_point_goal": {
                    "part_ref": "tool.functional_parts.tip.center",
                },
            },
        ]
    }

    adjusted, report = _lift_tool_pose_subgoals_to_table_clearance(
        subgoals,
        observed_scene_graph=_scene(),
        functional_tool_parts=_tool(),
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 1.0],
            "table_asset": str(table_path),
        },
        grounding_candidates=None,
        target_part_graph=None,
        tool_part_graph=None,
        tool_id=None,
        tool_mesh_path=mesh_path,
        min_tool_table_clearance_m=0.003,
        allowed_contact_table_penetration_m=0.30,
    )

    assert adjusted["goals"][0]["pose_world_tool"]["position"][2] == 0.403
    assert adjusted["goals"][1]["pose_world_tool"]["position"][2] == 0.15
    assert report["changed"] is True
    assert report["contact_preservation_active"] is True
    assert report["adjusted_goals"][0]["stage"] == "approach_tool_axis_align"
    assert report["skipped_contact_goals"][0]["stage"] == "engage_waypoint_1"
