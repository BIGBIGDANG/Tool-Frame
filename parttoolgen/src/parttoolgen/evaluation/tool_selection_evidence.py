"""Run-level evidence package for shared tool-side VLM choices."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_path(value: Any, *, base_dir: Path, repo_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [repo_root / path, base_dir / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return repo_root / path


def _copy_file_record(source: Path | None, destination: Path, *, role: str) -> dict[str, Any]:
    record = {
        "role": role,
        "source": str(source) if source is not None else None,
        "path": str(destination),
        "exists": bool(source is not None and source.exists() and source.is_file()),
        "copied": False,
    }
    if source is not None and source.exists() and source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record["copied"] = True
    return record


def _marker_from_mapping(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _contact_selection_source(contact_selection: Mapping[str, Any]) -> str:
    raw_response = _as_mapping(contact_selection.get("raw_response"))
    contact_raw = _as_mapping(raw_response.get("contact_point_motion_axis"))
    raw_meta = _as_mapping(_as_mapping(contact_raw.get("_raw_response")).get("metadata"))
    client = _marker_from_mapping(raw_meta.get("client"))
    if client:
        return f"vlm:{client}"
    return "reference_or_cached"


def _raw_contact_motion_selection(contact_selection: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(_as_mapping(contact_selection.get("raw_response")).get("contact_point_motion_axis"))


def _first_visual_item_path(
    visual_evidence: Any,
    *,
    kind: str,
    base_dir: Path,
    repo_root: Path,
    part_marker: str | None = None,
    face_marker: str | None = None,
) -> Path | None:
    visual = _as_mapping(visual_evidence)
    for item in visual.get("items") or []:
        if not isinstance(item, Mapping) or item.get("kind") != kind:
            continue
        if part_marker is not None and str(item.get("part_marker") or "") != str(part_marker):
            continue
        if face_marker is not None and str(item.get("face_marker") or "") != str(face_marker):
            continue
        path = _resolve_path(item.get("path"), base_dir=base_dir, repo_root=repo_root)
        if path is not None and path.exists():
            return path
    return None


def _visual_item_paths(
    visual_evidence: Any,
    *,
    kind: str,
    base_dir: Path,
    repo_root: Path,
    part_marker: str | None = None,
) -> list[Path]:
    visual = _as_mapping(visual_evidence)
    paths: list[Path] = []
    seen: set[str] = set()
    for item in visual.get("items") or []:
        if not isinstance(item, Mapping) or item.get("kind") != kind:
            continue
        if part_marker is not None and str(item.get("part_marker") or "") != str(part_marker):
            continue
        path = _resolve_path(item.get("path"), base_dir=base_dir, repo_root=repo_root)
        if path is None or not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _copy_visual_item_records(
    paths: Iterable[Path],
    destination_dir: Path,
    *,
    role: str,
) -> list[dict[str, Any]]:
    return [
        _copy_file_record(source, destination_dir / source.name, role=role)
        for source in paths
    ]


def _copy_face_highlight_records(
    visual_evidence: Any,
    *,
    part_marker: str | None,
    face_markers: Iterable[Any],
    base_dir: Path,
    repo_root: Path,
    destination_dir: Path,
    role: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for marker in face_markers:
        text = str(marker).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source = _first_visual_item_path(
            visual_evidence,
            kind="selected_part_face_highlight",
            part_marker=part_marker,
            face_marker=text,
            base_dir=base_dir,
            repo_root=repo_root,
        )
        suffix = source.suffix if source is not None else ".png"
        name = f"{role}_{text}{suffix}"
        records.append(_copy_file_record(source, destination_dir / name, role=role))
    return records


def _visual_evidence_from_ref(
    *,
    tool_refs_dir: Path,
    ref: Any,
    fallback: Path,
    repo_root: Path,
) -> Any:
    visual_ref = _as_mapping(ref).get("visual_evidence")
    visual_path = _resolve_path(visual_ref, base_dir=tool_refs_dir, repo_root=repo_root)
    if visual_path is not None and visual_path.exists():
        return _load_json(visual_path)
    return _load_json(fallback)


def _copy_prompt_images(
    *,
    visual_ref: Mapping[str, Any],
    output_dir: Path,
    tool_refs_dir: Path,
    repo_root: Path,
    role_prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, image_ref in enumerate(visual_ref.get("prompt_images") or []):
        source = _resolve_path(image_ref, base_dir=tool_refs_dir, repo_root=repo_root)
        records.append(
            _copy_file_record(
                source,
                output_dir / f"{role_prefix}_{index:02d}{source.suffix if source is not None else '.png'}",
                role=f"{role_prefix}_{index:02d}",
            )
        )
    return records


def _copy_named_visual_refs(
    *,
    visual_ref: Mapping[str, Any],
    output_dir: Path,
    tool_refs_dir: Path,
    repo_root: Path,
    names: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, role in names:
        source = _resolve_path(visual_ref.get(key), base_dir=tool_refs_dir, repo_root=repo_root)
        suffix = source.suffix if source is not None else ".png"
        records.append(_copy_file_record(source, output_dir / f"{role}{suffix}", role=role))
    return records


def write_tool_selection_visual_evidence(
    *,
    output_root: Path,
    tool_refs_dir: Path,
    task_family: str | None = None,
    split: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Copy shared tool-ref choice images next to a task run's seed folders."""

    tool_refs_dir = tool_refs_dir if tool_refs_dir.is_absolute() else repo_root / tool_refs_dir
    output_dir = output_root / "tool_selection_visual_evidence"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    part_selection = _as_mapping(_load_json(tool_refs_dir / "part_selection.json"))
    contact_selection = _as_mapping(_load_json(tool_refs_dir / "contact_point_selection.json"))
    axis_binding = _as_mapping(_load_json(tool_refs_dir / "axis_binding.json"))
    axis_mesh_report = _as_mapping(_load_json(tool_refs_dir / "axis_mesh_overlay_report.json"))

    selected_part = _as_mapping(contact_selection.get("selected_part")) or _as_mapping(axis_binding.get("selected_part"))
    contact_point = _as_mapping(contact_selection.get("contact_point")) or _as_mapping(axis_binding.get("contact_point"))
    motion_axis = _as_mapping(contact_selection.get("motion_axis")) or _as_mapping(axis_binding.get("motion_axis"))

    tool_id = _marker_from_mapping(
        contact_selection.get("tool_id"),
        axis_binding.get("tool_id"),
        part_selection.get("tool_id"),
        axis_mesh_report.get("tool_id"),
    )
    part_marker = _marker_from_mapping(
        selected_part.get("marker"),
        contact_selection.get("selected_contact_point_part_marker"),
        axis_mesh_report.get("selected_contact_point_part_marker"),
    )
    contact_face = _marker_from_mapping(
        contact_point.get("marker"),
        contact_selection.get("selected_contact_point_face_marker"),
        contact_selection.get("selected_contact_marker"),
        axis_binding.get("selected_contact_point_face_marker"),
        axis_binding.get("selected_contact_marker"),
    )
    motion_face = _marker_from_mapping(
        motion_axis.get("marker"),
        motion_axis.get("axis_marker"),
        contact_selection.get("selected_motion_axis_face_marker"),
        axis_binding.get("selected_motion_axis_face_marker"),
        axis_binding.get("selected_motion_axis_marker"),
    )

    part_visual = _visual_evidence_from_ref(
        tool_refs_dir=tool_refs_dir,
        ref={"visual_evidence": part_selection.get("visual_evidence")},
        fallback=tool_refs_dir / "evidence/01_part_selection/visual_evidence.json",
        repo_root=repo_root,
    )
    contact_visual = _visual_evidence_from_ref(
        tool_refs_dir=tool_refs_dir,
        ref=_as_mapping(contact_selection.get("visual_evidence")),
        fallback=tool_refs_dir / "evidence/02_contact_marker_selection/visual_evidence.json",
        repo_root=repo_root,
    )
    if not isinstance(contact_visual, Mapping):
        contact_visual = _visual_evidence_from_ref(
            tool_refs_dir=tool_refs_dir,
            ref=_as_mapping(axis_binding.get("visual_evidence")),
            fallback=tool_refs_dir / "evidence/02_contact_marker_selection/visual_evidence.json",
            repo_root=repo_root,
        )

    records: list[dict[str, Any]] = []
    for json_name, role in (
        ("part_selection.json", "part_selection_json"),
        ("contact_point_selection.json", "contact_point_and_motion_axis_selection_json"),
        ("axis_binding.json", "axis_binding_json"),
        ("axis_mesh_overlay_report.json", "axis_mesh_overlay_report_json"),
        ("tool_ref_pipeline_summary.json", "tool_ref_pipeline_summary_json"),
        ("p3sam_tool_ref_selection_summary.json", "p3sam_tool_ref_selection_summary_json"),
        ("tool_part_graph.json", "tool_part_graph_json"),
    ):
        records.append(
            _copy_file_record(
                tool_refs_dir / json_name,
                output_dir / "selected_refs" / json_name,
                role=role,
            )
        )

    part_base = tool_refs_dir / "evidence/01_part_selection"
    contact_base = tool_refs_dir / "evidence/02_contact_marker_selection"
    tool_dir_name = tool_id or "tool"
    raw_contact_motion = _raw_contact_motion_selection(contact_selection)
    raw_vlm_contact_faces = raw_contact_motion.get("candidate_contact_point_markers")
    raw_vlm_motion_faces = raw_contact_motion.get("candidate_motion_axis_markers")
    vlm_contact_faces = raw_vlm_contact_faces if isinstance(raw_vlm_contact_faces, list) else []
    vlm_motion_faces = raw_vlm_motion_faces if isinstance(raw_vlm_motion_faces, list) else []
    whole_part_overview = _first_visual_item_path(
        part_visual,
        kind="whole_mesh_part_choice_overview",
        base_dir=part_base,
        repo_root=repo_root,
    )
    isolated_part_overview = _first_visual_item_path(
        part_visual,
        kind="isolated_part_choice_overview",
        base_dir=part_base,
        repo_root=repo_root,
    )
    records.extend(
        _copy_visual_item_records(
            [path for path in (whole_part_overview, isolated_part_overview) if path is not None],
            output_dir / "part_selection" / tool_dir_name / "overview",
            role="structured_part_choice_overview",
        )
    )
    records.extend(
        _copy_visual_item_records(
            _visual_item_paths(
                part_visual,
                kind="isolated_part_card",
                base_dir=part_base,
                repo_root=repo_root,
            ),
            output_dir / "part_selection" / tool_dir_name / "part_cards",
            role="structured_part_candidate_card",
        )
    )
    records.extend(
        _copy_visual_item_records(
            _visual_item_paths(
                contact_visual,
                kind="selected_part_bbox_overview",
                part_marker=part_marker,
                base_dir=contact_base,
                repo_root=repo_root,
            ),
            output_dir / "contact_motion_selection" / tool_dir_name / "selected_part_bbox_overview",
            role="structured_selected_part_bbox_overview",
        )
    )
    records.extend(
        _copy_visual_item_records(
            _visual_item_paths(
                contact_visual,
                kind="selected_part_face_highlight",
                part_marker=part_marker,
                base_dir=contact_base,
                repo_root=repo_root,
            ),
            output_dir / "contact_motion_selection" / tool_dir_name / "all_part_face_highlights",
            role="structured_all_part_face_highlight",
        )
    )
    records.extend(
        _copy_face_highlight_records(
            contact_visual,
            part_marker=part_marker,
            face_markers=vlm_contact_faces,
            base_dir=contact_base,
            repo_root=repo_root,
            destination_dir=(
                output_dir
                / "contact_motion_selection"
                / tool_dir_name
                / "vlm_contact_candidate_face_highlights"
            ),
            role="structured_vlm_contact_candidate_face_highlight",
        )
    )
    records.extend(
        _copy_face_highlight_records(
            contact_visual,
            part_marker=part_marker,
            face_markers=vlm_motion_faces,
            base_dir=contact_base,
            repo_root=repo_root,
            destination_dir=(
                output_dir
                / "contact_motion_selection"
                / tool_dir_name
                / "vlm_motion_axis_candidate_face_highlights"
            ),
            role="structured_vlm_motion_axis_candidate_face_highlight",
        )
    )

    axis_visual = _as_mapping(axis_binding.get("visual_evidence"))
    contact_visual_ref = _as_mapping(contact_selection.get("visual_evidence"))
    records.extend(
        _copy_named_visual_refs(
            visual_ref=axis_visual,
            output_dir=output_dir / "motion_axis",
            tool_refs_dir=tool_refs_dir,
            repo_root=repo_root,
            names=[
                ("axis_choice_card", "axis_choice_card"),
                ("axis_mesh_choice", "axis_mesh_choice"),
                ("axis_mesh_overlay", "axis_mesh_overlay"),
            ],
        )
    )
    records.extend(
        _copy_prompt_images(
            visual_ref=contact_visual_ref,
            output_dir=output_dir / "prompt_images",
            tool_refs_dir=tool_refs_dir,
            repo_root=repo_root,
            role_prefix="contact_selection_prompt_image",
        )
    )
    records.extend(
        _copy_prompt_images(
            visual_ref=axis_visual,
            output_dir=output_dir / "prompt_images",
            tool_refs_dir=tool_refs_dir,
            repo_root=repo_root,
            role_prefix="axis_binding_prompt_image",
        )
    )

    decision_summary = {
        "schema_version": "toolframe_tool_selection_decision_summary_v1",
        "task_family": task_family,
        "split": split,
        "tool_id": tool_id,
        "tool_refs_dir": str(tool_refs_dir),
        "selected_part": dict(selected_part),
        "selected_contact_point": dict(contact_point),
        "selected_motion_axis": dict(motion_axis),
        "selected_markers": {
            "part": part_marker,
            "contact_point_face": contact_face,
            "motion_axis_face": motion_face,
        },
        "candidate_markers": {
            "contact_point_faces": [
                str(item.get("marker"))
                for item in (
                    contact_point.get("candidates")
                    or _as_mapping(axis_binding.get("contact_point")).get("candidates")
                    or []
                )
                if isinstance(item, Mapping) and item.get("marker") is not None
            ],
            "motion_axis_faces": [
                str(item.get("axis_marker") or item.get("marker"))
                for item in (
                    motion_axis.get("candidates")
                    or _as_mapping(axis_binding.get("motion_axis")).get("candidates")
                    or []
                )
                if isinstance(item, Mapping)
                and (item.get("axis_marker") is not None or item.get("marker") is not None)
            ],
            "raw_vlm_contact_point_faces": raw_vlm_contact_faces,
            "raw_vlm_motion_axis_faces": raw_vlm_motion_faces,
            "motion_axis_candidates_missing": raw_vlm_motion_faces is None,
        },
        "rationale": {
            "part_selection": {
                "selection_source": part_selection.get("selection_source"),
                "raw_response": part_selection.get("raw_response"),
            },
            "contact_point_selection": {
                "selection_source": _contact_selection_source(contact_selection),
                "confidence": contact_selection.get("confidence"),
                "reason": contact_selection.get("reason"),
                "raw_response": contact_selection.get("raw_response"),
            },
            "axis_binding": {
                "confidence": axis_binding.get("confidence"),
                "reason": axis_binding.get("reason"),
                "raw_response": axis_binding.get("raw_response"),
            },
        },
    }
    _write_json(output_dir / "decision_summary.json", decision_summary)
    records.append(
        {
            "role": "decision_summary_json",
            "source": None,
            "path": str(output_dir / "decision_summary.json"),
            "exists": True,
            "copied": False,
        }
    )

    manifest = {
        "schema_version": "toolframe_run_tool_selection_visual_evidence_v1",
        "output_dir": str(output_dir),
        "tool_refs_dir": str(tool_refs_dir),
        "task_family": task_family,
        "split": split,
        "selection_semantics": "shared_by_all_seeds_in_this_task_run",
        "tool_selection_summary": decision_summary,
        "item_count": len(records),
        "copied_item_count": sum(1 for item in records if item.get("copied")),
        "items": records,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "evidence_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "decision_summary": str(output_dir / "decision_summary.json"),
        "item_count": len(records),
        "copied_item_count": manifest["copied_item_count"],
    }
