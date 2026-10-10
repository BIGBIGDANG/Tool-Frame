"""Validate the shared VLM selection boundary without calling a model."""
import pytest

from ToGo.cli.tool_prior import select_p3sam_tool_refs as selector
from ToGo.pipeline.handoff import ROOT, read_json, write_json


def evidence(tmp_path):
    (tmp_path / "overview.png").write_bytes(b"test image")
    path = tmp_path / "evidence.json"
    write_json(path, {"items": [
        {"id": "unit_part", "kind": "whole_mesh_part_choice_overview", "path": "overview.png"},
        {"id": "unit_face", "part_marker": "P1", "kind": "candidate_part_contact_overview", "path": "overview.png"}],
        "metadata": {"region_marker_table": [
            {"tool_id": "unit", "marker": "P1", "part_ref": "tool.parts.part_000"}],
            "contact_marker_table": [
                {"tool_id": "unit", "part_marker": "P1", "contact_marker": "F1",
                 "point_ref": "tool.points.face1", "surface_ref": "tool.surfaces.face1",
                 "xyz_tool": [0.1, 0.2, 0.3], "normal_tool": [0, 0, 1]},
                {"tool_id": "unit", "part_marker": "P1", "contact_marker": "F2",
                 "point_ref": "tool.points.face2", "surface_ref": "tool.surfaces.face2",
                 "xyz_tool": [0, 0, 0], "normal_tool": [2, 0, 0]}]}})
    write_json(tmp_path / "part_selection.json", {"tool_id": "unit", "selected_marker": "P1"})
    return path


def select(tmp_path, monkeypatch, response):
    path = evidence(tmp_path)
    monkeypatch.setattr(selector, "_vlm_json", lambda *args: response)
    return selector.select(tool_id="unit", instruction="Use this tool", evidence_path=path,
        output=tmp_path / "output", stage="contact", part_selection=tmp_path / "part_selection.json", model=None)


def test_contact_and_separate_local_axis_are_preserved(tmp_path, monkeypatch):
    result = select(tmp_path, monkeypatch, {"selected_contact_point_marker": "F1", "selected_motion_axis_marker": "F2"})
    assert result["contact_point_tool"] == [0.1, 0.2, 0.3]
    assert result["motion_axis_tool"] == [1, 0, 0]
    assert "motion_axis_candidates_tool" not in result
    assert list((tmp_path / "output").iterdir()) == [tmp_path / "output/tool_refs.json"]


@pytest.mark.parametrize("contact,axis", [("", "F1"), ("F3", "F1"), ("F1", "F3")])
def test_invalid_contact_or_axis_selection_is_rejected(tmp_path, monkeypatch, contact, axis):
    with pytest.raises(ValueError, match="absent from valid"):
        select(tmp_path, monkeypatch, {"selected_contact_point_marker": contact, "selected_motion_axis_marker": axis})


def test_part_selection_uses_evidence_relative_images(tmp_path, monkeypatch):
    path = evidence(tmp_path)
    def response(prompt, images, instruction, model):
        assert images == [tmp_path / "overview.png"]
        return {"selected_marker": "1"}
    monkeypatch.setattr(selector, "_vlm_json", response)
    result = selector.select(tool_id="unit", instruction="Use this tool", evidence_path=path,
        output=tmp_path / "output", stage="part", part_selection=None, model=None)
    assert result["selected_marker"] == "P1"
    assert read_json(tmp_path / "output/part_selection.json") == result


@pytest.mark.parametrize("path", sorted((ROOT / "assets/tool_refs").glob("*.json")), ids=lambda p: p.stem)
def test_shipped_contact_geometry_matches_its_named_prior_reference(path):
    refs = read_json(path)
    prior = read_json(ROOT / "assets/tool_prior" / refs["tool_id"] / "tool_prior/grounding_candidates.json")
    reference = refs["contact_point_source_ref"]
    assert reference.startswith("candidate.points.")
    point = prior["points"][reference[len("candidate.points."):]]
    assert refs["contact_point_tool"] == pytest.approx(point["xyz_tool"])
    assert refs["contact_point_part_ref"] == point["part_ref"]
