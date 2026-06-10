"""Record per-trial visual evidence references.

The canonical stage outputs live directly under a trial directory, for example
``sim_rgbd/``, ``vlm_scene_keypoints/``, ``compile/`` and ``rollout/``.  The
``visual_evidence/`` directory is therefore a manifest by default, not a second
copy of those stage trees.  Set ``copy_files=True`` only for an explicit
self-contained export.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp4", ".webm", ".gif"}
EVIDENCE_SUFFIXES = VISUAL_SUFFIXES | {".json"}
FORBIDDEN_TOOL_REF_EVIDENCE_DIRS = {
    "02_face_selection",
    "contact_point_choices",
    "face_selection",
    "function_contact_choices",
}
FORBIDDEN_TOOL_REF_EVIDENCE_FILE_PATTERNS = {
    "candidate_*_F*_face*.png",
    "candidate_*_F*_face*.jpg",
    "candidate_*_F*_face*.jpeg",
}


def _ignore_forbidden_tool_ref_evidence(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in FORBIDDEN_TOOL_REF_EVIDENCE_DIRS:
            ignored.add(name)
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in FORBIDDEN_TOOL_REF_EVIDENCE_FILE_PATTERNS):
            ignored.add(name)
    return ignored


def _copy_file(
    source: Path,
    destination_dir: Path,
    *,
    kind: str,
    items: list[dict[str, Any]],
    name: str | None = None,
    copy_files: bool = False,
) -> None:
    if not source.exists() or not source.is_file():
        return
    destination = destination_dir / (name or source.name)
    if copy_files:
        destination_dir.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    else:
        destination = source
    items.append(
        {
            "kind": kind,
            "source": str(source),
            "path": str(destination),
            "copied": bool(copy_files),
        }
    )


def _copy_dir_files(
    source_dir: Path | None,
    destination_dir: Path,
    *,
    kind: str,
    items: list[dict[str, Any]],
    suffixes: set[str] = EVIDENCE_SUFFIXES,
    copy_files: bool = False,
) -> None:
    if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
        return
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in suffixes:
            continue
        _copy_file(source, destination_dir, kind=kind, items=items, copy_files=copy_files)


def _copy_named_files(
    source_dir: Path | None,
    destination_dir: Path,
    *,
    names: Iterable[str],
    kind: str,
    items: list[dict[str, Any]],
    copy_files: bool = False,
) -> None:
    if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
        return
    for name in names:
        _copy_file(source_dir / name, destination_dir, kind=kind, items=items, copy_files=copy_files)


def _copy_tree_if_exists(
    source_dir: Path | None,
    destination_dir: Path,
    *,
    kind: str,
    items: list[dict[str, Any]],
    copy_files: bool = False,
    ignore: Any = None,
) -> None:
    if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
        return
    if source_dir.resolve() == destination_dir.resolve():
        return
    if copy_files:
        if destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.copytree(source_dir, destination_dir, ignore=ignore)
        path = destination_dir
    else:
        path = source_dir
    items.append(
        {
            "kind": kind,
            "source": str(source_dir),
            "path": str(path),
            "copied": bool(copy_files),
        }
    )


def package_trial_visual_evidence(
    *,
    trial_dir: Path,
    render_dir: Path | None = None,
    grounding_dir: Path | None = None,
    tool_refs_dir: Path | None = None,
    compile_dir: Path | None = None,
    rollout_dir: Path | None = None,
    copy_files: bool = False,
) -> dict[str, Any]:
    """Write a visual/debug evidence manifest for one trial.

    By default the manifest points to canonical stage outputs in the trial tree
    and does not duplicate files.  ``copy_files=True`` keeps the old
    self-contained export behavior for external sharing.
    """

    evidence_dir = trial_dir / "visual_evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []

    _copy_dir_files(
        render_dir,
        evidence_dir / "scene",
        kind="scene_rgbd_evidence",
        items=items,
        copy_files=copy_files,
    )
    _copy_dir_files(
        grounding_dir,
        evidence_dir / "grounding",
        kind="vlm_scene_grounding",
        items=items,
        copy_files=copy_files,
    )
    _copy_named_files(
        tool_refs_dir,
        evidence_dir / "tool_refs",
        names=[
            "part_selection.json",
            "contact_point_selection.json",
            "axis_binding.json",
            "axis_mesh_overlay_report.json",
            "p3sam_tool_ref_selection_summary.json",
            "tool_ref_pipeline_summary.json",
            "tool_part_graph.json",
        ],
        kind="tool_ref_grounding",
        items=items,
        copy_files=copy_files,
    )
    _copy_tree_if_exists(
        tool_refs_dir / "evidence" if tool_refs_dir is not None else None,
        evidence_dir / "tool_refs" / "evidence",
        kind="tool_ref_visual_evidence_package",
        items=items,
        copy_files=copy_files,
        ignore=_ignore_forbidden_tool_ref_evidence,
    )
    _copy_named_files(
        compile_dir,
        evidence_dir / "compile",
        names=[
            "compile_summary.json",
            "selected_constraint_program.json",
            "tool_pose_subgoals.json",
            "path_solved_tool_pose_subgoals.json",
            "simtoolreal_fixed_goals.json",
            "functional_tool_parts.json",
            "derived_scene_geometry.json",
            "table_projection_report.json",
        ],
        kind="compile_evidence",
        items=items,
        copy_files=copy_files,
    )
    _copy_dir_files(
        rollout_dir,
        evidence_dir / "rollout",
        kind="rollout_evidence",
        items=items,
        copy_files=copy_files,
    )
    _copy_tree_if_exists(
        rollout_dir / "subgoal_visualization" if rollout_dir is not None else None,
        evidence_dir / "rollout" / "subgoal_visualization",
        kind="subgoal_visualization_evidence",
        items=items,
        copy_files=copy_files,
    )

    manifest = {
        "schema_version": "toolframe_trial_visual_evidence_package_v1",
        "trial_dir": str(trial_dir),
        "evidence_dir": str(evidence_dir),
        "item_count": len(items),
        "copy_files": bool(copy_files),
        "items": items,
    }
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "evidence_dir": str(evidence_dir),
        "manifest": str(manifest_path),
        "item_count": len(items),
    }


def package_trial_visual_evidence_from_artifacts(
    *,
    trial_dir: Path,
    artifacts: dict[str, Any],
    tool_refs_dir: Path | None = None,
    copy_files: bool = False,
) -> dict[str, Any]:
    """Write an evidence manifest from artifact paths already recorded in a result."""

    def parent_of(key: str) -> Path | None:
        value = artifacts.get(key)
        if not value:
            return None
        path = Path(str(value))
        return path.parent if path.exists() else None

    return package_trial_visual_evidence(
        trial_dir=trial_dir,
        render_dir=parent_of("scene_evidence"),
        grounding_dir=parent_of("grounding_overlay"),
        tool_refs_dir=tool_refs_dir,
        compile_dir=parent_of("compile_summary"),
        rollout_dir=parent_of("rollout_summary"),
        copy_files=copy_files,
    )
