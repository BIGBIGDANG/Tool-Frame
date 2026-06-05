from __future__ import annotations

import json
from pathlib import Path

from parttoolgen.evaluation.tool_selection_evidence import (
    write_tool_selection_visual_evidence,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_tool_selection_evidence_copies_contact_and_axis_images(tmp_path: Path) -> None:
    tool_refs = tmp_path / "tool_refs/hammer"
    whole_part_image = tmp_path / "imgs/hammer_whole_mesh_part_choice_overview.png"
    isolated_part_image = tmp_path / "imgs/hammer_isolated_part_choice_overview.png"
    part_card_1 = tmp_path / "imgs/candidate_001_card.png"
    part_card_2 = tmp_path / "imgs/candidate_002_card.png"
    bbox_image = tmp_path / "imgs/bbox.png"
    contact_image = tmp_path / "imgs/contact_f3.png"
    axis_image = tmp_path / "imgs/axis_f1.png"
    for image in (
        whole_part_image,
        isolated_part_image,
        part_card_1,
        part_card_2,
        bbox_image,
        contact_image,
        axis_image,
    ):
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")

    _write_json(
        tool_refs / "part_selection.json",
        {
            "tool_id": "hammer",
            "selected_part": {"marker": "2", "part_ref": "tool.parts.head"},
            "visual_evidence": str(tmp_path.relative_to(tmp_path) / "tool_refs/hammer/evidence/01_part_selection/visual_evidence.json"),
        },
    )
    _write_json(
        tool_refs / "contact_point_selection.json",
        {
            "tool_id": "hammer",
            "selected_part": {"marker": "2", "part_ref": "tool.parts.head"},
            "contact_point": {"marker": "F3", "surface_ref": "candidate.surfaces.head_bottom"},
            "motion_axis": {"marker": "F1", "surface_ref": "candidate.surfaces.head_side"},
            "confidence": 0.8,
            "reason": "strike with the bottom face and align by the side axis",
            "visual_evidence": {
                "visual_evidence": "tool_refs/hammer/evidence/02_contact_marker_selection/visual_evidence.json"
            },
        },
    )
    _write_json(
        tool_refs / "axis_binding.json",
        {
            "tool_id": "hammer",
            "selected_part": {"marker": "2", "part_ref": "tool.parts.head"},
            "contact_point": {"marker": "F3", "surface_ref": "candidate.surfaces.head_bottom"},
            "motion_axis": {"marker": "F1", "surface_ref": "candidate.surfaces.head_side"},
            "confidence": 0.9,
            "reason": "axis follows the selected face normal",
        },
    )
    _write_json(tool_refs / "axis_mesh_overlay_report.json", {"tool_id": "hammer"})
    _write_json(
        tool_refs / "evidence/01_part_selection/visual_evidence.json",
        {
            "items": [
                {
                    "kind": "whole_mesh_part_choice_overview",
                    "path": str(whole_part_image.relative_to(tmp_path)),
                },
                {
                    "kind": "isolated_part_choice_overview",
                    "path": str(isolated_part_image.relative_to(tmp_path)),
                },
                {
                    "kind": "isolated_part_card",
                    "part_marker": "1",
                    "path": str(part_card_1.relative_to(tmp_path)),
                },
                {
                    "kind": "isolated_part_card",
                    "part_marker": "2",
                    "path": str(part_card_2.relative_to(tmp_path)),
                },
            ]
        },
    )
    _write_json(
        tool_refs / "evidence/02_contact_marker_selection/visual_evidence.json",
        {
            "items": [
                {
                    "kind": "selected_part_bbox_overview",
                    "part_marker": "2",
                    "path": str(bbox_image.relative_to(tmp_path)),
                },
                {
                    "kind": "selected_part_face_highlight",
                    "part_marker": "2",
                    "face_marker": "F3",
                    "path": str(contact_image.relative_to(tmp_path)),
                },
                {
                    "kind": "selected_part_face_highlight",
                    "part_marker": "2",
                    "face_marker": "F1",
                    "path": str(axis_image.relative_to(tmp_path)),
                },
            ]
        },
    )

    result = write_tool_selection_visual_evidence(
        output_root=tmp_path / "run/hammer_strike/down",
        tool_refs_dir=tool_refs,
        task_family="hammer_strike",
        split="down",
        repo_root=tmp_path,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["copied_item_count"] >= 6
    evidence_dir = tmp_path / "run/hammer_strike/down/tool_selection_visual_evidence"
    assert not (evidence_dir / "part_selection/isolated_part_choice_overview.png").exists()
    assert not (evidence_dir / "part_selection/whole_mesh_part_choice_overview.png").exists()
    assert (
        evidence_dir
        / "part_selection/hammer/overview/hammer_isolated_part_choice_overview.png"
    ).exists()
    assert (evidence_dir / "part_selection/hammer/part_cards/candidate_001_card.png").exists()
    assert (evidence_dir / "part_selection/hammer/part_cards/candidate_002_card.png").exists()
    assert (evidence_dir / "contact_motion_selection/hammer/selected_part_bbox_overview/bbox.png").exists()
    assert (evidence_dir / "contact_motion_selection/hammer/selected_part_face_highlights/contact_f3.png").exists()
    assert (evidence_dir / "contact_motion_selection/hammer/selected_part_face_highlights/axis_f1.png").exists()
    summary = json.loads((evidence_dir / "decision_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_markers"] == {
        "part": "2",
        "contact_point_face": "F3",
        "motion_axis_face": "F1",
    }
