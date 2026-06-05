from parttoolgen.adapters.simtoolreal_policy_priors import (
    mesh_table_clearance_report,
    table_contact_start_pose_from_state_bundle,
    tool_mesh_z_bounds_for_pose,
    load_obj_vertices,
)


def _write_tool_obj(path):
    path.write_text(
        "\n".join(
            [
                "v 0 0 -0.10",
                "v 0 0 0.10",
                "f 1 2 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_mesh_table_clearance_report_detects_goal_violation(tmp_path):
    mesh_path = tmp_path / "tool.obj"
    _write_tool_obj(mesh_path)
    fixed_goals = {
        "start_pose": [0.12, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0],
        "goals": [[0.02, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0]],
    }

    report = mesh_table_clearance_report(
        fixed_goals,
        table_top_z=0.0,
        tool_mesh_path=mesh_path,
        min_tool_table_clearance_m=0.01,
    )

    assert report["passed"] is False
    assert report["violations"][0]["pose"] == "goals[0]"


def test_mesh_table_clearance_report_does_not_modify_fixed_goals(tmp_path):
    mesh_path = tmp_path / "tool.obj"
    _write_tool_obj(mesh_path)
    fixed_goals = {
        "start_pose": [0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0],
        "goals": [[0.02, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0]],
    }
    original = {
        "start_pose": list(fixed_goals["start_pose"]),
        "goals": [list(goal) for goal in fixed_goals["goals"]],
    }

    report = mesh_table_clearance_report(
        fixed_goals,
        table_top_z=0.0,
        tool_mesh_path=mesh_path,
        min_tool_table_clearance_m=0.01,
    )

    assert report["passed"] is True
    assert fixed_goals == original


def test_table_contact_start_pose_projects_mesh_to_tabletop(tmp_path):
    mesh_path = tmp_path / "tool.obj"
    _write_tool_obj(mesh_path)
    table_path = tmp_path / "table.urdf"
    table_path.write_text(
        """
<robot name="table">
  <link name="table">
    <collision>
      <origin xyz="0 0 0" />
      <geometry><box size="1 1 0.10" /></geometry>
    </collision>
  </link>
</robot>
""".strip()
        + "\n",
        encoding="utf-8",
    )
    start_pose = {
        "position": [0.1, 0.2, 1.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }

    projected = table_contact_start_pose_from_state_bundle(
        start_pose=start_pose,
        state_bundle={
            "table_pose_xyzw": [0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 1.0],
            "table_asset": str(table_path),
        },
        tool_mesh_path=mesh_path,
        min_tool_table_clearance_m=0.003,
    )

    assert projected["position"][:2] == [0.1, 0.2]
    assert projected["orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    mesh_min_z, _ = tool_mesh_z_bounds_for_pose(
        [*projected["position"], *projected["orientation_xyzw"]],
        load_obj_vertices(mesh_path),
    )
    assert abs(mesh_min_z - 0.553) < 1e-9
