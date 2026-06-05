from __future__ import annotations

import json
from pathlib import Path

from parttoolgen.evaluation.visual_evidence_package import (
    package_trial_visual_evidence,
    package_trial_visual_evidence_from_artifacts,
)


def test_visual_evidence_default_manifest_only_mode_does_not_duplicate_files(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    render_dir = trial_dir / "sim_rgbd"
    render_dir.mkdir(parents=True)
    source = render_dir / "sim_topdown_rgb.png"
    source.write_bytes(b"png")

    result = package_trial_visual_evidence(
        trial_dir=trial_dir,
        render_dir=render_dir,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["copy_files"] is False
    assert manifest["items"][0]["source"] == str(source)
    assert manifest["items"][0]["path"] == str(source)
    assert manifest["items"][0]["copied"] is False
    assert (trial_dir / "visual_evidence" / "manifest.json").exists()
    assert not (trial_dir / "visual_evidence" / "scene").exists()
    assert not (trial_dir / "visual_evidence" / "scene" / "sim_topdown_rgb.png").exists()


def test_visual_evidence_from_artifacts_default_does_not_duplicate_compile(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    compile_dir = trial_dir / "compile"
    compile_dir.mkdir(parents=True)
    compile_summary = compile_dir / "compile_summary.json"
    compile_summary.write_text("{}", encoding="utf-8")

    result = package_trial_visual_evidence_from_artifacts(
        trial_dir=trial_dir,
        artifacts={"compile_summary": str(compile_summary)},
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["copy_files"] is False
    assert manifest["items"][0]["source"] == str(compile_summary)
    assert manifest["items"][0]["path"] == str(compile_summary)
    assert manifest["items"][0]["copied"] is False
    assert not (trial_dir / "visual_evidence" / "compile").exists()


def test_visual_evidence_package_keeps_new_tool_refs_and_ignores_legacy_face_cards(tmp_path: Path) -> None:
    trial_dir = tmp_path / "trial"
    tool_refs_dir = tmp_path / "tool_refs"
    tool_refs_dir.mkdir()
    (tool_refs_dir / "tool_ref_pipeline_summary.json").write_text("{}", encoding="utf-8")
    (tool_refs_dir / "p3sam_tool_ref_two_stage_vlm_summary.json").write_text("{}", encoding="utf-8")

    legacy_dir = tool_refs_dir / "evidence" / "02_face_selection" / "contact_point_choices"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "candidate_001_F1_face.png").write_bytes(b"old")
    new_dir = tool_refs_dir / "evidence" / "02_contact_marker_selection" / "tool" / "selected_part_face_highlights"
    new_dir.mkdir(parents=True)
    (new_dir / "candidate_001_face_F1_highlight.png").write_bytes(b"new")

    result = package_trial_visual_evidence(
        trial_dir=trial_dir,
        tool_refs_dir=tool_refs_dir,
        copy_files=True,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    copied_paths = [item["path"] for item in manifest["items"]]
    assert any(path.endswith("tool_ref_pipeline_summary.json") for path in copied_paths)
    assert not any(path.endswith("p3sam_tool_ref_two_stage_vlm_summary.json") for path in copied_paths)
    assert (
        trial_dir
        / "visual_evidence/tool_refs/evidence/02_contact_marker_selection/tool/selected_part_face_highlights/candidate_001_face_F1_highlight.png"
    ).exists()
    assert not (
        trial_dir
        / "visual_evidence/tool_refs/evidence/02_face_selection/contact_point_choices/candidate_001_F1_face.png"
    ).exists()
