from __future__ import annotations

import json

import pytest

from parttoolgen.image_grounded import (
    build_contact_motion_program,
    build_tool_refs_from_axis_evidence,
    infer_axis_mesh_report_path,
    merge_tool_refs_into_constraint_program,
)
from parttoolgen.image_grounded.tool_refs import (
    ToolRefMergeError,
    build_tool_ref_candidates_from_contact_selection,
)


def _axis_binding() -> dict:
    return {
        "tool_id": "flat_eraser",
        "selected_contact_point_part_ref": "tool.parts.part_002",
        "selected_contact_point_source_ref": "candidate.points.part_002_z_negative_contact",
        "selected_contact_point_face_marker": "F1",
        "selected_contact_surface_ref": "candidate.surfaces.part_002_z_negative_contact",
        "selected_motion_axis_marker": "F4",
        "selected_motion_axis_face_marker": "F4",
        "selected_motion_axis_surface_ref": "candidate.surfaces.part_002_x_positive_contact",
        "candidate_contact_point_source_refs": [],
        "motion_axis": {
            "axis_marker": "U",
            "axis_role": "mesh_patch_normal",
            "axis_ref": "tool.motion_axis",
            "surface_ref": "candidate.surfaces.part_002_x_positive_contact",
        },
        "axis_candidate_table": [
            {
                "axis_marker": "U",
                "axis_role": "mesh_patch_normal",
                "axis_ref": "candidate.surfaces.part_002_x_positive_contact",
                "surface_ref": "candidate.surfaces.part_002_x_positive_contact",
                "vector_tool": [2.0, 0.0, 0.0],
            }
        ],
        "visual_evidence": {
            "axis_mesh_overlay": "axis_overlay.png",
            "prompt_images": ["whole_mesh_part_choice_overview.png"],
        },
    }


def _axis_mesh_report() -> dict:
    return {
        "axis_candidates": [
            {
                "axis_marker": "U",
                "axis_role": "mesh_patch_normal",
                "vector_tool": [2.0, 0.0, 0.0],
            }
        ],
        "contact_point_surface": {
            "frame_source": "selected_bbox_face",
            "patch_id": "z_negative_contact",
            "contact_point_xyz_tool": [0.16, 0.04, 0.0],
            "surface_center_xyz_tool": [0.16, 0.04, 0.0],
            "mesh_normal_tool": [0.0, 0.0, -1.0],
            "rectangle_corners_xyz_tool": [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.0, 0.1, 0.0],
            ],
            "long_edge_m": 0.2,
            "short_edge_m": 0.1,
        },
        "motion_axis_surface": {
            "patch_id": "x_positive_contact",
            "surface_ref": "candidate.surfaces.part_002_x_positive_contact",
            "surface_center_xyz_tool": [0.2, 0.04, 0.03],
            "motion_axis_tool": [1.0, 0.0, 0.0],
            "mesh_normal_tool": [1.0, 0.0, 0.0],
        },
    }


def _contact_selection() -> dict:
    rows = [
        {
            "contact_marker": "F1",
            "face_marker": "F1",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_z_negative_contact",
            "surface_ref": "candidate.surfaces.part_002_z_negative_contact",
            "patch_id": "z_negative_contact",
            "xyz_tool": [0.16, 0.04, 0.0],
            "normal_tool": [0.0, 0.0, -1.0],
            "source_contact_surface_ref": "candidate.surfaces.part_002_z_negative_contact",
        },
        {
            "contact_marker": "F2",
            "face_marker": "F2",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_z_positive_contact",
            "surface_ref": "candidate.surfaces.part_002_z_positive_contact",
            "patch_id": "z_positive_contact",
            "xyz_tool": [0.16, 0.04, 0.05],
            "normal_tool": [0.0, 0.0, 1.0],
            "source_contact_surface_ref": "candidate.surfaces.part_002_z_positive_contact",
        },
    ]
    return {
        "candidate_contact_point_face_markers": ["F1", "F2"],
        "candidate_contact_point_source_refs": [
            "candidate.points.part_002_z_negative_contact",
            "candidate.points.part_002_z_positive_contact",
        ],
        "candidate_contact_rows": rows,
        "contact_marker_table": rows,
        "visual_evidence": {"visual_evidence": "visual_evidence.json"},
    }


def _selector_style_contact_selection() -> dict:
    old = _contact_selection()
    rows = old["candidate_contact_rows"]
    return {
        "schema_version": "parttoolgen_contact_point_motion_axis_selection_v1",
        "contact_point_candidates": [
            {
                "marker": row["contact_marker"],
                "point_ref": row["point_ref"],
                "surface_ref": row["surface_ref"],
                "patch_id": row["patch_id"],
                "row": dict(row),
            }
            for row in rows
        ],
        "contact_point_marker_table": old["contact_marker_table"],
        "visual_evidence": old["visual_evidence"],
    }


def _flat_tangent_contact_selection() -> dict:
    rows = [
        {
            "contact_marker": "F1",
            "face_marker": "F1",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_x_negative_endpoint",
            "surface_ref": "candidate.surfaces.part_002_x_negative_endpoint",
            "patch_id": "x_negative_endpoint",
            "xyz_tool": [-0.1, 0.0, 0.0],
            "normal_tool": [-1.0, 0.0, 0.0],
        },
        {
            "contact_marker": "F2",
            "face_marker": "F2",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_x_positive_endpoint",
            "surface_ref": "candidate.surfaces.part_002_x_positive_endpoint",
            "patch_id": "x_positive_endpoint",
            "xyz_tool": [0.1, 0.0, 0.0],
            "normal_tool": [1.0, 0.0, 0.0],
        },
        {
            "contact_marker": "F3",
            "face_marker": "F3",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_y_negative_contact",
            "surface_ref": "candidate.surfaces.part_002_y_negative_contact",
            "patch_id": "y_negative_contact",
            "xyz_tool": [0.0, -0.02, 0.0],
            "normal_tool": [0.0, -1.0, 0.0],
        },
        {
            "contact_marker": "F4",
            "face_marker": "F4",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_y_positive_contact",
            "surface_ref": "candidate.surfaces.part_002_y_positive_contact",
            "patch_id": "y_positive_contact",
            "xyz_tool": [0.0, 0.02, 0.0],
            "normal_tool": [0.0, 1.0, 0.0],
        },
        {
            "contact_marker": "F5",
            "face_marker": "F5",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_z_negative_contact",
            "surface_ref": "candidate.surfaces.part_002_z_negative_contact",
            "patch_id": "z_negative_contact",
            "xyz_tool": [0.0, 0.0, -0.03],
            "normal_tool": [0.0, 0.0, -1.0],
        },
        {
            "contact_marker": "F6",
            "face_marker": "F6",
            "part_id": "part_002",
            "part_ref": "tool.parts.part_002",
            "point_ref": "candidate.points.part_002_z_positive_contact",
            "surface_ref": "candidate.surfaces.part_002_z_positive_contact",
            "patch_id": "z_positive_contact",
            "xyz_tool": [0.0, 0.0, 0.03],
            "normal_tool": [0.0, 0.0, 1.0],
        },
    ]
    return {
        "selected_contact_patch_id": "y_negative_contact",
        "selected_motion_axis_patch_id": "x_negative_endpoint",
        "selected_contact_point_part_ref": "tool.parts.part_002",
        "candidate_contact_point_face_markers": ["F3", "F4"],
        "candidate_contact_rows": [rows[2], rows[3]],
        "contact_marker_table": rows,
        "face_marker_table": rows,
    }


def test_axis_evidence_builds_canonical_tool_refs() -> None:
    refs = build_tool_refs_from_axis_evidence(_axis_binding(), _axis_mesh_report())
    assert refs["contact_point_tool"] == [0.16, 0.04, 0.0]
    assert refs["motion_axis_ref"] == "tool.motion_axis"
    assert refs["motion_axis_tool"] == [1.0, 0.0, 0.0]
    assert refs["contact_point_face_marker"] == "F1"
    assert refs["motion_axis_face_marker"] == "F4"
    assert refs["contact_point_surface_ref"] == "candidate.surfaces.part_002_z_negative_contact"
    assert refs["candidate_contact_point_source_refs"] == []


def test_axis_evidence_preserves_nested_motion_axis_candidates() -> None:
    binding = _axis_binding()
    binding["motion_axis"]["candidates"] = [
        {
            "axis_marker": "F3",
            "axis_role": "mesh_patch_normal",
            "surface_ref": "candidate.surfaces.part_002_y_negative_contact",
            "vector_tool": [0.0, -1.0, 0.0],
        },
        {
            "axis_marker": "F4",
            "axis_role": "mesh_patch_normal",
            "surface_ref": "candidate.surfaces.part_002_y_positive_contact",
            "vector_tool": [0.0, 1.0, 0.0],
        },
    ]

    refs = build_tool_refs_from_axis_evidence(binding, _axis_mesh_report())
    vectors = {tuple(candidate["vector_tool"]) for candidate in refs["motion_axis_candidates_tool"]}

    assert (0.0, -1.0, 0.0) in vectors
    assert (0.0, 1.0, 0.0) in vectors


def test_axis_evidence_requires_explicit_motion_axis_selection() -> None:
    binding = _axis_binding()
    binding.pop("selected_motion_axis_marker", None)
    binding.pop("selected_motion_axis_face_marker", None)
    binding.pop("selected_motion_axis_surface_ref", None)
    binding.pop("motion_axis", None)

    with pytest.raises(ToolRefMergeError, match="motion-axis selection is missing"):
        build_tool_refs_from_axis_evidence(binding, _axis_mesh_report())


def test_contact_candidates_require_explicit_motion_axis_selection() -> None:
    binding = _axis_binding()
    binding.pop("selected_motion_axis_marker", None)
    binding.pop("selected_motion_axis_face_marker", None)
    binding.pop("selected_motion_axis_surface_ref", None)
    binding.pop("motion_axis", None)

    program = build_contact_motion_program(
        task_id="eraser_wipe",
        instruction="Wipe the mark with a VLM-approved contact point candidate.",
        tool_refs={"tool_id": "existing_tool", "tool_mesh": "eraser.obj"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
    ).to_json()

    with pytest.raises(ToolRefMergeError, match="motion-axis selection is missing"):
        merge_tool_refs_into_constraint_program(
            program,
            binding,
            _axis_mesh_report(),
            contact_point_selection=_contact_selection(),
        )


def test_axis_evidence_accepts_nested_p3sam_binding_without_axis_candidates() -> None:
    binding = {
        "tool_id": "flat_eraser",
        "selected_part": {"part_ref": "tool.parts.part_002"},
        "contact_point": {
            "marker": "F1",
            "point_ref": "candidate.points.part_002_z_negative_contact",
            "surface_ref": "candidate.surfaces.part_002_z_negative_contact",
            "candidates": [
                {"point_ref": "candidate.points.part_002_z_negative_contact"},
                {"point_ref": "candidate.points.part_002_z_positive_contact"},
            ],
        },
        "motion_axis": {
            "marker": "F4",
            "ref": "tool.motion_axis",
            "surface_ref": "candidate.surfaces.part_002_x_positive_contact",
            "vector_tool": [1.0, 0.0, 0.0],
        },
    }
    report = _axis_mesh_report()
    report.pop("axis_candidates")
    refs = build_tool_refs_from_axis_evidence(binding, report)
    assert refs["contact_point_source_ref"] == "candidate.points.part_002_z_negative_contact"
    assert refs["contact_point_surface_ref"] == "candidate.surfaces.part_002_z_negative_contact"
    assert refs["motion_axis_ref"] == "tool.motion_axis"
    assert refs["source_motion_axis_ref"] == "candidate.surfaces.part_002_x_positive_contact"
    assert refs["candidate_contact_point_source_refs"] == [
        "candidate.points.part_002_z_negative_contact",
        "candidate.points.part_002_z_positive_contact",
    ]


def test_tool_refs_merge_preserves_existing_program_metadata() -> None:
    program = build_contact_motion_program(
        task_id="brush_forward",
        instruction="Sweep the block forward.",
        tool_refs={"tool_id": "existing_tool", "tool_mesh": "brush.obj"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
    ).to_json()
    merged = merge_tool_refs_into_constraint_program(program, _axis_binding(), _axis_mesh_report())
    assert merged["tool_refs"]["tool_id"] == "existing_tool"
    assert merged["tool_refs"]["tool_mesh"] == "brush.obj"
    assert merged["tool_refs"]["motion_axis_tool"] == [1.0, 0.0, 0.0]
    action = next(stage for stage in merged["stages"] if stage["name"] == "action")
    axis_align = next(
        row for row in action["path_constraints"] if row["type"] == "axis_align"
    )
    assert axis_align["params"]["tool_axis_ref"] == "tool.motion_axis"
    assert axis_align["params"]["target_axis_ref"] == "scene.direction"
    assert "source" not in axis_align["params"]


def test_program_can_bind_vlm_selected_motion_axis_for_contact_point_path() -> None:
    program = build_contact_motion_program(
        task_id="eraser_wipe",
        instruction="Wipe the mark with the selected tool point and motion axis.",
        tool_refs={"tool_id": "existing_tool", "tool_mesh": "eraser.obj"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
        tool_motion_axis_ref="tool.motion_axis",
        tool_motion_axis_target_ref="scene.direction",
        tool_motion_axis_relation="parallel",
    ).to_json()
    merged = merge_tool_refs_into_constraint_program(program, _axis_binding(), _axis_mesh_report())
    action = next(stage for stage in merged["stages"] if stage["name"] == "action")
    motion_axis = next(row for row in action["path_constraints"] if row["type"] == "axis_align")
    assert motion_axis["params"]["tool_axis_ref"] == "tool.motion_axis"
    assert motion_axis["params"]["target_axis_ref"] == "scene.direction"


def test_tool_ref_candidates_inherit_vlm_motion_axis() -> None:
    program = build_contact_motion_program(
        task_id="eraser_wipe",
        instruction="Wipe the mark with a VLM-approved contact point candidate.",
        tool_refs={"tool_id": "existing_tool", "tool_mesh": "eraser.obj"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
        tool_motion_axis_ref="tool.motion_axis",
        tool_motion_axis_target_ref="scene.direction",
    ).to_json()
    merged = merge_tool_refs_into_constraint_program(
        program,
        _axis_binding(),
        _axis_mesh_report(),
        contact_point_selection=_contact_selection(),
    )
    candidates = merged["tool_refs"]["contact_point_candidates_tool"]
    assert [candidate["contact_point_face_marker"] for candidate in candidates] == ["F1", "F2"]
    assert all(candidate["motion_axis_tool"] == [1.0, 0.0, 0.0] for candidate in candidates)
    assert all(candidate["motion_axis_face_marker"] == "F4" for candidate in candidates)
    assert merged["tool_refs"]["contact_point_candidate_selection_policy"] == (
        "compiler_optimizes_over_public_vlm_contact_candidates"
    )


def test_tool_ref_candidates_inherit_motion_axis_candidate_set() -> None:
    binding = _axis_binding()
    binding["motion_axis"]["candidates"] = [
        {
            "axis_marker": "F3",
            "axis_role": "mesh_patch_normal",
            "surface_ref": "candidate.surfaces.part_002_y_negative_contact",
            "vector_tool": [0.0, -1.0, 0.0],
        },
        {
            "axis_marker": "F4",
            "axis_role": "mesh_patch_normal",
            "surface_ref": "candidate.surfaces.part_002_y_positive_contact",
            "vector_tool": [0.0, 1.0, 0.0],
        },
    ]
    program = build_contact_motion_program(
        task_id="brush_sweep",
        instruction="Sweep with any VLM-approved equivalent brush face.",
        tool_refs={"tool_id": "existing_tool", "tool_mesh": "brush.obj"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
        tool_motion_axis_ref="tool.motion_axis",
        tool_motion_axis_target_ref="scene.direction",
    ).to_json()

    merged = merge_tool_refs_into_constraint_program(
        program,
        binding,
        _axis_mesh_report(),
        contact_point_selection=_contact_selection(),
    )

    for candidate in merged["tool_refs"]["contact_point_candidates_tool"]:
        vectors = {
            tuple(axis["vector_tool"])
            for axis in candidate["motion_axis_candidates_tool"]
        }
        assert (0.0, -1.0, 0.0) in vectors
        assert (0.0, 1.0, 0.0) in vectors


def test_tool_ref_candidates_accept_selector_contact_point_candidates() -> None:
    candidates = build_tool_ref_candidates_from_contact_selection(
        _selector_style_contact_selection(),
        _axis_binding(),
    )

    assert [candidate["contact_point_face_marker"] for candidate in candidates] == ["F1", "F2"]
    assert [candidate["contact_point_source_ref"] for candidate in candidates] == [
        "candidate.points.part_002_z_negative_contact",
        "candidate.points.part_002_z_positive_contact",
    ]
    assert all(
        candidate["visual_evidence"]["visual_evidence"] == "visual_evidence.json"
        for candidate in candidates
    )


def test_flat_bbox_tangent_motion_axes_are_kept_as_candidates() -> None:
    binding = _axis_binding()
    binding.update(
        {
            "selected_contact_point_part_ref": "tool.parts.part_002",
            "selected_contact_point_face_marker": "F3",
            "selected_contact_surface_ref": "candidate.surfaces.part_002_y_negative_contact",
            "selected_motion_axis_marker": "F1",
            "selected_motion_axis_face_marker": "F1",
            "selected_motion_axis_surface_ref": "candidate.surfaces.part_002_x_negative_endpoint",
        }
    )
    binding["motion_axis"] = {
        "axis_marker": "U",
        "axis_role": "contact_point_bbox_endpoint_axis",
        "axis_ref": "tool.motion_axis",
        "surface_ref": "candidate.surfaces.part_002_x_negative_endpoint",
    }
    binding["axis_candidate_table"] = [
        {
            "axis_marker": "U",
            "axis_role": "contact_point_bbox_endpoint_axis",
            "axis_ref": "candidate.surfaces.part_002_x_negative_endpoint",
            "surface_ref": "candidate.surfaces.part_002_x_negative_endpoint",
            "vector_tool": [-1.0, 0.0, 0.0],
        }
    ]
    report = _axis_mesh_report()
    report["axis_candidates"] = [
        {
            "axis_marker": "U",
            "axis_role": "contact_point_bbox_endpoint_axis",
            "vector_tool": [-1.0, 0.0, 0.0],
        }
    ]
    report["contact_point_surface"] = {
        **report["contact_point_surface"],
        "patch_id": "y_negative_contact",
        "contact_point_xyz_tool": [0.0, -0.02, 0.0],
    }

    program = build_contact_motion_program(
        task_id="eraser_wipe",
        instruction="Wipe a printed pattern with a flat eraser.",
        tool_refs={"tool_id": "flat_eraser"},
        scene_refs={
            "start": {"position_world": [0.0, 0.0, 0.0]},
            "end": {"position_world": [0.1, 0.0, 0.0]},
            "direction": [1.0, 0.0, 0.0],
        },
    ).to_json()
    merged = merge_tool_refs_into_constraint_program(
        program,
        binding,
        report,
        contact_point_selection=_flat_tangent_contact_selection(),
    )

    vectors = {
        tuple(candidate["vector_tool"])
        for candidate in merged["tool_refs"]["motion_axis_candidates_tool"]
    }
    assert vectors == {
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    }
    assert all(
        len(candidate["motion_axis_candidates_tool"]) == 4
        for candidate in merged["tool_refs"]["contact_point_candidates_tool"]
    )


def test_axis_report_path_can_be_inferred_from_overlay(tmp_path) -> None:
    axis_binding_path = tmp_path / "axis_binding.json"
    overlay_path = tmp_path / "axis_overlay.png"
    report_path = tmp_path / "axis_overlay_report.json"
    overlay_path.write_bytes(b"not an actual image")
    report_path.write_text(json.dumps(_axis_mesh_report()), encoding="utf-8")
    binding = _axis_binding()
    binding["visual_evidence"]["axis_mesh_overlay"] = str(overlay_path)
    axis_binding_path.write_text(json.dumps(binding), encoding="utf-8")
    assert infer_axis_mesh_report_path(binding, axis_binding_path=axis_binding_path) == report_path
