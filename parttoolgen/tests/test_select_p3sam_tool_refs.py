from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_select_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "parttoolgen" / "cli" / "tool_prior" / "select_p3sam_tool_refs.py"
    spec = importlib.util.spec_from_file_location("parttoolgen_select_p3sam_tool_refs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_refs = _load_select_module()


def test_motion_axis_uses_explicit_bbox_axis() -> None:
    axis, source = select_refs._axis_for_face_row(
        {"bbox_axis_tool": [1, 0, 0], "contact_marker": "F1"}
    )

    assert axis == [1.0, 0.0, 0.0]
    assert source == "bbox_axis_tool"


def test_motion_axis_fails_closed_without_axis_evidence() -> None:
    with pytest.raises(ValueError, match="no explicit tool-axis evidence"):
        select_refs._axis_for_face_row(
            {"contact_marker": "F9", "surface_ref": "candidate.surfaces.part_1"}
        )


def test_contact_rows_skip_internal_tool_surfaces() -> None:
    visual_evidence = {
        "metadata": {
            "contact_marker_table": [
                {
                    "tool_id": "tool_a",
                    "part_marker": "1",
                    "contact_marker": "F1",
                    "point_ref": "candidate.points.external",
                    "surface_ref": "candidate.surfaces.external",
                    "tool_surface_visibility": "external",
                },
                {
                    "tool_id": "tool_a",
                    "part_marker": "1",
                    "contact_marker": "F2",
                    "point_ref": "candidate.points.internal",
                    "surface_ref": "candidate.surfaces.internal",
                    "is_internal_tool_surface": True,
                },
            ]
        }
    }

    rows = select_refs._contact_rows(visual_evidence, tool_id="tool_a", part_marker="1")

    assert [row["contact_marker"] for row in rows] == ["F1"]


def test_reference_contact_selection_keeps_motion_axis_candidates() -> None:
    visual_evidence = {
        "metadata": {
            "contact_marker_table": [
                {
                    "tool_id": "red_brush",
                    "part_marker": "3",
                    "contact_marker": "F3",
                    "face_marker": "F3",
                    "point_ref": "candidate.points.part_002_y_negative_contact",
                    "surface_ref": "candidate.surfaces.part_002_y_negative_contact",
                    "patch_id": "y_negative_contact",
                    "normal_tool": [0.0, -1.0, 0.0],
                },
                {
                    "tool_id": "red_brush",
                    "part_marker": "3",
                    "contact_marker": "F4",
                    "face_marker": "F4",
                    "point_ref": "candidate.points.part_002_y_positive_contact",
                    "surface_ref": "candidate.surfaces.part_002_y_positive_contact",
                    "patch_id": "y_positive_contact",
                    "normal_tool": [0.0, 1.0, 0.0],
                },
            ]
        }
    }

    contact, axis, contact_candidates, axis_candidates, raw = select_refs._select_contact(
        visual_evidence=visual_evidence,
        tool_id="red_brush",
        task_instruction="sweep the block forward",
        part_marker="3",
        selection_source="reference",
        marker="F3",
        candidate_markers=["F3", "F4"],
        axis_marker="F4",
        candidate_axis_markers=["F3", "F4"],
        model="dummy",
        base_url=None,
        max_new_tokens=64,
    )

    assert contact["contact_marker"] == "F3"
    assert axis["contact_marker"] == "F4"
    assert [row["contact_marker"] for row in contact_candidates] == ["F3", "F4"]
    assert [row["contact_marker"] for row in axis_candidates] == ["F3", "F4"]
    assert raw["candidate_motion_axis_markers"] == ["F3", "F4"]


def test_contact_prompt_keeps_all_semantically_valid_working_faces() -> None:
    prompt = select_refs._contact_prompt(
        tool_id="flat_eraser",
        task_instruction="wipe the marked region",
        part_marker="1",
        rows=[
            {
                "contact_marker": "F3",
                "point_ref": "candidate.points.part_000_y_negative_contact",
                "surface_ref": "candidate.surfaces.part_000_y_negative_contact",
                "polarity": "negative",
                "patch_id": "y_negative_contact",
            },
            {
                "contact_marker": "F5",
                "point_ref": "candidate.points.part_000_z_negative_contact",
                "surface_ref": "candidate.surfaces.part_000_z_negative_contact",
                "polarity": "negative",
                "patch_id": "z_negative_contact",
            },
        ],
    )

    assert "green dot is a representative point" in prompt
    assert "red arrow is the candidate motion axis" in prompt
    assert "candidate face a person would normally use to touch the target" in prompt
    assert "which axis of the working part would be used to approach the target" in prompt
    assert "A marker may be useful as a contact point, a motion axis, both, or neither" in prompt
    assert "include all of them as candidates" in prompt


def test_part_prompt_images_include_whole_and_isolated_overviews(tmp_path: Path) -> None:
    whole = tmp_path / "tool/overview/whole.png"
    isolated = tmp_path / "tool/overview/isolated.png"
    selected_card = tmp_path / "tool/part_cards/candidate_002.png"
    for path in (whole, isolated, selected_card):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    visual_evidence = {
        "items": [
            {
                "id": "hammer_whole_mesh_part_choice_overview",
                "kind": "whole_mesh_part_choice_overview",
                "path": str(whole),
            },
            {
                "id": "hammer_isolated_part_choice_overview",
                "kind": "isolated_part_choice_overview",
                "path": str(isolated),
            },
            {
                "id": "hammer_part_candidate_2_isolated_card",
                "kind": "isolated_part_card",
                "part_marker": "2",
                "path": str(selected_card),
            },
        ]
    }

    part_images = select_refs._image_items(visual_evidence, tool_id="hammer")
    contact_images = select_refs._image_items(
        visual_evidence,
        tool_id="hammer",
        part_marker="2",
    )

    assert part_images == [whole, isolated]
    assert contact_images == [whole, isolated, selected_card]
