"""Tool-side contact-point/motion-axis refs for ToolFrame image-grounded planning."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ToGo.compiler.residuals import add, mul, norm, sub, unit


class ToolRefMergeError(ValueError):
    """Raised when tool-side evidence cannot be converted into ToolFrame refs."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolRefMergeError(f"{name} must be an object")
    return value


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ToolRefMergeError(f"{name} must be a 3-vector")
    return [float(item) for item in value]


def _path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or base_dir is None:
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path


def _existing_path(value: str | None, *, base_dir: Path | None = None) -> Path | None:
    if not value:
        return None
    path = _path(value, base_dir=base_dir)
    if path.exists():
        return path
    return None


def infer_axis_mesh_report_path(
    axis_binding: Mapping[str, Any],
    *,
    axis_binding_path: str | Path | None = None,
) -> Path | None:
    """Infer the mesh-overlay report path referenced by an axis-binding file."""

    base_dir = Path(axis_binding_path).expanduser().resolve().parent if axis_binding_path else None
    visual_evidence = _mapping(axis_binding.get("visual_evidence", {}), "axis_binding.visual_evidence")
    direct = _existing_path(str(visual_evidence.get("axis_mesh_overlay_report") or ""), base_dir=base_dir)
    if direct is not None:
        return direct

    for key in ("axis_mesh_overlay", "axis_mesh_choice"):
        image_path = _existing_path(str(visual_evidence.get(key) or ""), base_dir=base_dir)
        if image_path is None:
            continue
        candidates = [image_path.with_name(f"{image_path.stem}_report.json")]
        if image_path.stem.endswith("_mesh_choice"):
            candidates.append(
                image_path.with_name(f"{image_path.stem[:-len('_mesh_choice')]}_mesh_overlay_report.json")
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _select_axis_candidate(
    axis_binding: Mapping[str, Any],
    axis_mesh_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    motion_axis = _mapping(axis_binding.get("motion_axis") or {}, "axis_binding.motion_axis")
    marker = str(
        motion_axis.get("axis_marker")
        or motion_axis.get("marker")
        or axis_binding.get("selected_motion_axis_marker")
        or ""
    )
    candidates = axis_mesh_report.get("axis_candidates")
    if isinstance(candidates, list) and candidates:
        if not marker:
            available = [
                str(candidate.get("axis_marker") or candidate.get("marker") or "")
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ]
            raise ToolRefMergeError(
                "motion-axis selection is missing; axis_candidates require an "
                f"explicit selected motion-axis marker, available={available!r}"
            )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if str(candidate.get("axis_marker") or candidate.get("marker") or "") == marker:
                return candidate
        available = [
            str(candidate.get("axis_marker") or candidate.get("marker") or "")
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        raise ToolRefMergeError(
            "selected motion-axis marker is missing from axis candidates: "
            f"selected={marker!r}, available={available!r}"
        )

    motion_axis_surface = axis_mesh_report.get("motion_axis_surface")
    if isinstance(motion_axis_surface, Mapping):
        vector = motion_axis_surface.get("motion_axis_tool") or motion_axis_surface.get("mesh_normal_tool")
        if vector is None:
            raise ToolRefMergeError("axis_mesh_report.motion_axis_surface has no motion_axis_tool")
        return {
            "axis_marker": marker or motion_axis.get("marker") or motion_axis.get("axis_marker"),
            "axis_role": motion_axis_surface.get("normal_role")
            or motion_axis.get("role")
            or motion_axis.get("axis_role")
            or "mesh_patch_normal",
            "axis_ref": motion_axis_surface.get("motion_axis_ref") or motion_axis.get("ref"),
            "surface_ref": motion_axis_surface.get("surface_ref") or motion_axis.get("surface_ref"),
            "vector_tool": vector,
        }

    if motion_axis.get("vector_tool") is not None:
        return {
            "axis_marker": marker or motion_axis.get("marker") or motion_axis.get("axis_marker"),
            "axis_role": motion_axis.get("role") or motion_axis.get("axis_role") or "mesh_patch_normal",
            "axis_ref": motion_axis.get("ref") or motion_axis.get("axis_ref"),
            "surface_ref": motion_axis.get("surface_ref"),
            "vector_tool": motion_axis.get("vector_tool"),
        }

    raise ToolRefMergeError(
        "axis_mesh_report must provide axis_candidates, motion_axis_surface.motion_axis_tool, "
        "or axis_binding.motion_axis.vector_tool"
    )


def _axis_sign_from_patch_id(patch_id: Any) -> tuple[str, str] | None:
    text = str(patch_id or "")
    if not text:
        return None
    axis = text[0].lower()
    if axis not in {"x", "y", "z"}:
        return None
    if "negative" in text:
        return axis, "negative"
    if "positive" in text:
        return axis, "positive"
    return None


def _bbox_axis_from_patch_id(patch_id: Any) -> list[float] | None:
    parsed = _axis_sign_from_patch_id(patch_id)
    if parsed is None:
        return None
    axis, sign = parsed
    value = -1.0 if sign == "negative" else 1.0
    if axis == "x":
        return [value, 0.0, 0.0]
    if axis == "y":
        return [0.0, value, 0.0]
    if axis == "z":
        return [0.0, 0.0, value]
    return None


def _motion_axis_candidate_key(vector: Sequence[float]) -> tuple[float, float, float]:
    normalized = unit([float(item) for item in vector])
    return tuple(round(value, 6) for value in normalized)


def _motion_axis_candidate_from_vector(
    *,
    vector_tool: Any,
    candidate_id: str,
    source: str,
    axis_marker: Any = None,
    axis_role: Any = None,
    axis_ref: Any = None,
    surface_ref: Any = None,
    patch_id: Any = None,
) -> dict[str, Any]:
    vector = unit(_vec3(vector_tool, f"{source}.vector_tool"))
    return {
        "candidate_id": str(candidate_id),
        "vector_tool": vector,
        "axis_marker": str(axis_marker or "") or None,
        "axis_role": str(axis_role or "mesh_patch_normal"),
        "axis_ref": str(axis_ref or "") or None,
        "surface_ref": str(surface_ref or "") or None,
        "patch_id": str(patch_id or "") or None,
        "source": source,
    }


def _dedupe_motion_axis_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        vector = candidate.get("vector_tool")
        if vector is None:
            continue
        key = _motion_axis_candidate_key(_vec3(vector, "motion_axis_candidate.vector_tool"))
        if key in seen:
            continue
        seen.add(key)
        row = dict(candidate)
        row["vector_tool"] = list(key)
        deduped.append(row)
    return deduped


def _motion_axis_candidates_from_axis_evidence(
    axis_binding: Mapping[str, Any],
    axis_mesh_report: Mapping[str, Any],
    *,
    selected_candidate: Mapping[str, Any],
    selected_vector: Sequence[float],
) -> list[dict[str, Any]]:
    motion_axis_binding = axis_binding.get("motion_axis")
    motion_axis_mapping = motion_axis_binding if isinstance(motion_axis_binding, Mapping) else {}
    candidates: list[dict[str, Any]] = [
        _motion_axis_candidate_from_vector(
            vector_tool=list(selected_vector),
            candidate_id="selected_motion_axis",
            source="selected_motion_axis",
            axis_marker=selected_candidate.get("axis_marker")
            or selected_candidate.get("marker")
            or motion_axis_mapping.get("axis_marker")
            or motion_axis_mapping.get("marker"),
            axis_role=selected_candidate.get("axis_role")
            or motion_axis_mapping.get("axis_role")
            or motion_axis_mapping.get("role"),
            axis_ref=selected_candidate.get("axis_ref")
            or motion_axis_mapping.get("axis_ref")
            or motion_axis_mapping.get("ref"),
            surface_ref=selected_candidate.get("surface_ref")
            or motion_axis_mapping.get("surface_ref"),
            patch_id=selected_candidate.get("patch_id"),
        )
    ]
    for source_name, rows in (
        ("axis_mesh_report.axis_candidates", axis_mesh_report.get("axis_candidates")),
        ("axis_binding.axis_candidate_table", axis_binding.get("axis_candidate_table")),
        (
            "axis_binding.motion_axis.candidates",
            motion_axis_mapping.get("candidates") if isinstance(motion_axis_mapping, Mapping) else None,
        ),
    ):
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("vector_tool") is None:
                continue
            candidates.append(
                _motion_axis_candidate_from_vector(
                    vector_tool=row.get("vector_tool"),
                    candidate_id=str(
                        row.get("candidate_id")
                        or row.get("axis_marker")
                        or row.get("marker")
                        or row.get("surface_ref")
                        or row.get("axis_ref")
                        or f"{source_name}[{index}]"
                    ),
                    source=source_name,
                    axis_marker=row.get("axis_marker") or row.get("marker"),
                    axis_role=row.get("axis_role") or row.get("role"),
                    axis_ref=row.get("axis_ref") or row.get("ref"),
                    surface_ref=row.get("surface_ref"),
                    patch_id=row.get("patch_id"),
                )
            )
    if isinstance(motion_axis_binding, Mapping) and motion_axis_binding.get("vector_tool") is not None:
        candidates.append(
            _motion_axis_candidate_from_vector(
                vector_tool=motion_axis_binding.get("vector_tool"),
                candidate_id=str(
                    motion_axis_binding.get("candidate_id")
                    or motion_axis_binding.get("axis_marker")
                    or motion_axis_binding.get("marker")
                    or "axis_binding.motion_axis"
                ),
                source="axis_binding.motion_axis",
                axis_marker=motion_axis_binding.get("axis_marker") or motion_axis_mapping.get("marker"),
                axis_role=motion_axis_binding.get("axis_role") or motion_axis_mapping.get("role"),
                axis_ref=motion_axis_binding.get("axis_ref") or motion_axis_mapping.get("ref"),
                surface_ref=motion_axis_binding.get("surface_ref"),
                patch_id=motion_axis_binding.get("patch_id"),
            )
        )
    return _dedupe_motion_axis_candidates(candidates)


def _motion_axis_role_from_axis_binding(axis_binding: Mapping[str, Any]) -> str | None:
    motion_axis = axis_binding.get("motion_axis")
    if isinstance(motion_axis, Mapping):
        value = motion_axis.get("axis_role") or motion_axis.get("role")
        if value:
            return str(value)
    rows = axis_binding.get("axis_candidate_table")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                value = row.get("axis_role") or row.get("role")
                if value:
                    return str(value)
    return None


def _motion_axis_candidates_from_contact_selection(
    contact_selection: Mapping[str, Any],
    axis_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_contact_axis = _axis_sign_from_patch_id(
        contact_selection.get("selected_contact_patch_id")
        or axis_binding.get("selected_contact_patch_id")
        or axis_binding.get("selected_contact_point_patch_id")
    )
    selected_motion_axis = _axis_sign_from_patch_id(
        contact_selection.get("selected_motion_axis_patch_id")
        or axis_binding.get("selected_motion_axis_patch_id")
    )
    motion_axis_role = _motion_axis_role_from_axis_binding(axis_binding)
    if (
        selected_contact_axis is None
        or selected_motion_axis is None
        or selected_contact_axis[0] == selected_motion_axis[0]
        or motion_axis_role != "contact_point_bbox_endpoint_axis"
    ):
        return []

    rows = contact_selection.get("contact_marker_table")
    if not isinstance(rows, list):
        rows = contact_selection.get("face_marker_table")
    if not isinstance(rows, list):
        return []
    candidate_contact_rows = contact_selection.get("candidate_contact_rows")
    if not isinstance(candidate_contact_rows, list):
        candidate_contact_rows = []
    selected_part_ids = {
        str(row.get("part_id"))
        for row in candidate_contact_rows
        if isinstance(row, Mapping) and row.get("part_id") is not None
    }
    selected_part_refs = {
        str(row.get("part_ref"))
        for row in candidate_contact_rows
        if isinstance(row, Mapping) and row.get("part_ref") is not None
    }
    if not selected_part_ids and axis_binding.get("selected_contact_point_part_ref") is not None:
        selected_part_refs.add(str(axis_binding.get("selected_contact_point_part_ref")))

    candidates: list[dict[str, Any]] = []
    contact_axis = selected_contact_axis[0]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        if selected_part_ids and str(row.get("part_id")) not in selected_part_ids:
            continue
        if selected_part_refs and str(row.get("part_ref")) not in selected_part_refs:
            continue
        parsed = _axis_sign_from_patch_id(row.get("patch_id"))
        if parsed is None or parsed[0] == contact_axis:
            continue
        vector = _bbox_axis_from_patch_id(row.get("patch_id"))
        if vector is None:
            continue
        candidates.append(
            _motion_axis_candidate_from_vector(
                vector_tool=vector,
                candidate_id=str(
                    row.get("surface_ref")
                    or row.get("point_ref")
                    or row.get("derived_point_ref")
                    or row.get("face_marker")
                    or f"contact_marker_table[{index}]"
                ),
                source="contact_selection.same_part_tangent_bbox_axis",
                axis_marker=row.get("face_marker") or row.get("contact_marker"),
                axis_role="contact_point_bbox_endpoint_axis",
                axis_ref=row.get("surface_ref") or row.get("point_ref") or row.get("derived_point_ref"),
                surface_ref=row.get("surface_ref"),
                patch_id=row.get("patch_id"),
            )
        )
    return _dedupe_motion_axis_candidates(candidates)


def _selected_motion_axis_from_axis_binding(
    axis_binding: Mapping[str, Any],
) -> tuple[list[float], str, str | None, str | None] | None:
    selected_marker = str(axis_binding.get("selected_motion_axis_marker") or "")
    motion_axis = axis_binding.get("motion_axis")
    if isinstance(motion_axis, Mapping):
        selected_marker = str(motion_axis.get("axis_marker") or motion_axis.get("marker") or selected_marker or "")

    candidates = axis_binding.get("axis_candidate_table")
    if isinstance(candidates, list):
        ordered = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
        available = [
            str(candidate.get("axis_marker") or candidate.get("marker") or "")
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        if not selected_marker:
            raise ToolRefMergeError(
                "motion-axis selection is missing; axis_candidate_table requires "
                f"an explicit selected motion-axis marker, available={available!r}"
            )
        ordered = [
            candidate
            for candidate in ordered
            if str(candidate.get("axis_marker") or candidate.get("marker") or "") == selected_marker
        ]
        if not ordered:
            raise ToolRefMergeError(
                "selected motion-axis marker is missing from axis candidate table: "
                f"selected={selected_marker!r}, available={available!r}"
            )
        for candidate in ordered:
            vector = candidate.get("vector_tool")
            if vector is None:
                continue
            return (
                unit(_vec3(vector, "axis_binding.axis_candidate_table[].vector_tool")),
                str(candidate.get("axis_role") or "mesh_patch_normal"),
                str(candidate.get("axis_marker") or candidate.get("marker") or selected_marker or "") or None,
                str(candidate.get("surface_ref") or candidate.get("axis_ref") or "") or None,
            )

    if isinstance(motion_axis, Mapping) and motion_axis.get("vector_tool") is not None:
        motion_axis_marker = str(motion_axis.get("axis_marker") or motion_axis.get("marker") or "")
        if selected_marker and motion_axis_marker not in {"", selected_marker}:
            raise ToolRefMergeError(
                "axis_binding.motion_axis marker does not match selected motion-axis marker: "
                f"selected={selected_marker!r}, motion_axis={motion_axis_marker!r}"
            )
        return (
            unit(_vec3(motion_axis.get("vector_tool"), "axis_binding.motion_axis.vector_tool")),
            str(motion_axis.get("axis_role") or motion_axis.get("role") or "mesh_patch_normal"),
            str(motion_axis_marker or selected_marker or "") or None,
            str(motion_axis.get("surface_ref") or motion_axis.get("axis_ref") or motion_axis.get("ref") or "") or None,
        )
    return None


def _face_rows_for_same_part(row: Mapping[str, Any], rows: Sequence[Any]) -> list[Mapping[str, Any]]:
    part_id = row.get("part_id")
    part_ref = row.get("part_ref")
    result: list[Mapping[str, Any]] = []
    for candidate in rows:
        if not isinstance(candidate, Mapping):
            continue
        if part_id is not None and candidate.get("part_id") != part_id:
            continue
        if part_id is None and part_ref is not None and candidate.get("part_ref") != part_ref:
            continue
        if _axis_sign_from_patch_id(candidate.get("patch_id")) is None:
            continue
        if candidate.get("xyz_tool") is None or candidate.get("normal_tool") is None:
            continue
        result.append(candidate)
    return result


def _face_rectangle_from_bbox_rows(
    row: Mapping[str, Any],
    *,
    all_face_rows: Sequence[Any],
) -> list[list[float]] | None:
    """Recover a selected bbox face rectangle from the same part's six face rows.

    The VLM-visible face candidates store face centers/normals but not always
    corners.  For table clearance we only need conservative per-part face
    samples, so reconstruct the selected face using opposite face centers.
    """

    parsed = _axis_sign_from_patch_id(row.get("patch_id"))
    if parsed is None:
        return None
    face_axis, _face_sign = parsed
    rows = _face_rows_for_same_part(row, all_face_rows)
    centers: dict[tuple[str, str], list[float]] = {}
    axis_votes: dict[str, list[list[float]]] = {"x": [], "y": [], "z": []}
    for face_row in rows:
        face = _axis_sign_from_patch_id(face_row.get("patch_id"))
        if face is None:
            continue
        axis, sign = face
        center = _vec3(face_row.get("xyz_tool"), "face_row.xyz_tool")
        normal = unit(_vec3(face_row.get("normal_tool"), "face_row.normal_tool"))
        centers[(axis, sign)] = center
        axis_votes[axis].append(normal if sign == "positive" else mul(normal, -1.0))

    axes: dict[str, list[float]] = {}
    extents: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        votes = axis_votes[axis]
        if votes:
            summed = [0.0, 0.0, 0.0]
            for vote in votes:
                summed = add(summed, vote)
            axes[axis] = unit(summed)
        negative = centers.get((axis, "negative"))
        positive = centers.get((axis, "positive"))
        if negative is not None and positive is not None:
            extents[axis] = norm(sub(positive, negative))

    other_axes = [axis for axis in ("x", "y", "z") if axis != face_axis]
    if any(axis not in axes or axis not in extents for axis in other_axes):
        return None
    center = _vec3(row.get("xyz_tool"), "selected_face.xyz_tool")
    half_a = 0.5 * float(extents[other_axes[0]])
    half_b = 0.5 * float(extents[other_axes[1]])
    axis_a = axes[other_axes[0]]
    axis_b = axes[other_axes[1]]
    corners: list[list[float]] = []
    for sa, sb in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        corners.append(add(add(center, mul(axis_a, sa * half_a)), mul(axis_b, sb * half_b)))
    if any(any(not math.isfinite(value) for value in corner) for corner in corners):
        return None
    return corners


def build_tool_refs_from_axis_evidence(
    axis_binding: Mapping[str, Any],
    axis_mesh_report: Mapping[str, Any],
    *,
    axis_binding_path: str | Path | None = None,
    axis_mesh_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert contact-point/motion-axis evidence into canonical ToolFrame refs.

    The public ToolFrame boundary names are ``contact_point`` and
    ``motion_axis``.  Surface normals are retained only as mesh geometry
    metadata for local frame construction; they are not public tool-axis refs.
    """

    surface = _mapping(axis_mesh_report.get("contact_point_surface"), "axis_mesh_report.contact_point_surface")
    point = (
        surface.get("contact_point_xyz_tool")
        or surface.get("surface_center_xyz_tool")
        or surface.get("raw_mesh_patch_center_xyz_tool")
    )
    if point is None:
        raise ToolRefMergeError("axis_mesh_report.contact_point_surface has no contact point")
    axis_candidate = _select_axis_candidate(axis_binding, axis_mesh_report)
    motion_axis_vector = unit(_vec3(axis_candidate.get("vector_tool"), "axis_candidate.vector_tool"))
    motion_axis_source = str(axis_candidate.get("axis_role") or "mesh_patch_normal")
    motion_axis_candidates = _motion_axis_candidates_from_axis_evidence(
        axis_binding,
        axis_mesh_report,
        selected_candidate=axis_candidate,
        selected_vector=motion_axis_vector,
    )
    motion_axis_binding = _mapping(axis_binding.get("motion_axis") or {}, "axis_binding.motion_axis")
    contact_point_binding = _mapping(axis_binding.get("contact_point") or {}, "axis_binding.contact_point")
    selected_part = axis_binding.get("selected_part")
    selected_part_mapping = selected_part if isinstance(selected_part, Mapping) else {}
    visual_evidence = _mapping(axis_binding.get("visual_evidence", {}), "axis_binding.visual_evidence")
    tool_part_graph_path = None
    if axis_binding_path is not None:
        candidate_tool_parts = Path(axis_binding_path).expanduser().resolve().parent / "tool_part_graph.json"
        if candidate_tool_parts.exists():
            tool_part_graph_path = str(candidate_tool_parts)

    return {
        "contact_point_ref": "tool.contact_point",
        "contact_point_tool": _vec3(point, "contact_point_surface.contact_point_xyz_tool"),
        "motion_axis_ref": "tool.motion_axis",
        "motion_axis_tool": motion_axis_vector,
        "motion_axis_candidates_tool": motion_axis_candidates,
        "contact_point_face_id": axis_binding.get("selected_contact_surface_ref")
        or contact_point_binding.get("surface_ref")
        or surface.get("patch_id")
        or axis_binding.get("selected_contact_point_face_marker")
        or contact_point_binding.get("marker"),
        "contact_point_face_marker": axis_binding.get("selected_contact_point_face_marker")
        or contact_point_binding.get("marker"),
        "motion_axis_face_marker": axis_binding.get("selected_motion_axis_face_marker")
        or axis_binding.get("selected_motion_axis_marker")
        or motion_axis_binding.get("marker")
        or motion_axis_binding.get("axis_marker"),
        "motion_axis_surface_ref": axis_binding.get("selected_motion_axis_surface_ref")
        or motion_axis_binding.get("surface_ref")
        or axis_candidate.get("surface_ref"),
        "contact_point_part_ref": axis_binding.get("selected_contact_point_part_ref")
        or selected_part_mapping.get("part_ref"),
        "contact_point_source_ref": axis_binding.get("selected_contact_point_source_ref")
        or axis_binding.get("selected_contact_point_ref")
        or contact_point_binding.get("point_ref")
        or axis_binding.get("selected_contact_surface_ref")
        or contact_point_binding.get("surface_ref"),
        "contact_point_surface_ref": axis_binding.get("selected_contact_surface_ref")
        or contact_point_binding.get("surface_ref"),
        "motion_axis_marker": axis_candidate.get("axis_marker")
        or axis_candidate.get("marker")
        or motion_axis_binding.get("axis_marker")
        or motion_axis_binding.get("marker"),
        "motion_axis_role": motion_axis_source,
        "source_surface_ref": surface.get("source_surface_ref")
        or surface.get("normal_ref")
        or surface.get("surface_ref")
        or contact_point_binding.get("surface_ref"),
        "source_motion_axis_ref": axis_binding.get("selected_motion_axis_surface_ref")
        or motion_axis_binding.get("surface_ref")
        or axis_candidate.get("surface_ref")
        or axis_candidate.get("axis_ref"),
        "candidate_contact_point_source_refs": axis_binding.get("candidate_contact_point_source_refs")
        or [
            candidate.get("point_ref")
            for candidate in (contact_point_binding.get("candidates") or [])
            if isinstance(candidate, Mapping) and candidate.get("point_ref") is not None
        ],
        "surface_geometry_tool": {
            "frame_source": surface.get("frame_source"),
            "patch_id": surface.get("patch_id"),
            "surface_center": surface.get("surface_center_xyz_tool"),
            "rectangle_corners": surface.get("rectangle_corners_xyz_tool"),
            "long_edge_m": surface.get("long_edge_m"),
            "short_edge_m": surface.get("short_edge_m"),
            "motion_axis_tool": motion_axis_vector,
            "motion_axis_candidates_tool": motion_axis_candidates,
            "motion_axis_source": motion_axis_source,
        },
        "visual_evidence": {
            "axis_choice_card": visual_evidence.get("axis_choice_card"),
            "axis_mesh_choice": visual_evidence.get("axis_mesh_choice"),
            "axis_mesh_overlay": visual_evidence.get("axis_mesh_overlay"),
            "axis_mesh_overlay_report": visual_evidence.get("axis_mesh_overlay_report"),
            "prompt_images": visual_evidence.get("prompt_images") or [],
        },
        "source_axis_binding": str(axis_binding_path) if axis_binding_path else None,
        "source_axis_mesh_report": str(axis_mesh_report_path) if axis_mesh_report_path else None,
        "tool_part_graph_path": tool_part_graph_path,
    }


def _candidate_tool_ref_from_contact_row(
    row: Mapping[str, Any],
    *,
    axis_binding: Mapping[str, Any],
    all_face_rows: Sequence[Any] = (),
    visual_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    point = row.get("xyz_tool")
    if point is None:
        raise ToolRefMergeError("contact-point candidate row has no xyz_tool")
    motion_axis = _selected_motion_axis_from_axis_binding(axis_binding)
    face_marker = row.get("face_marker") or row.get("contact_marker")
    surface_ref = row.get("surface_ref")
    contact_ref = row.get("point_ref") or row.get("derived_point_ref")
    patch_id = row.get("patch_id") or surface_ref or face_marker
    rectangle_corners = _face_rectangle_from_bbox_rows(row, all_face_rows=all_face_rows)
    long_edge_m = None
    short_edge_m = None
    if rectangle_corners is not None and len(rectangle_corners) >= 4:
        edge_a = norm(sub(rectangle_corners[1], rectangle_corners[0]))
        edge_b = norm(sub(rectangle_corners[2], rectangle_corners[1]))
        long_edge_m = max(edge_a, edge_b)
        short_edge_m = min(edge_a, edge_b)
    evidence_ref: dict[str, Any] = {}
    if isinstance(visual_evidence, Mapping):
        if visual_evidence.get("visual_evidence"):
            evidence_ref["visual_evidence"] = visual_evidence.get("visual_evidence")
        if visual_evidence.get("prompt_images"):
            evidence_ref["prompt_images"] = visual_evidence.get("prompt_images")
    candidate_ref = {
        "candidate_id": str(surface_ref or contact_ref or patch_id or face_marker),
        "contact_point_ref": "tool.contact_point",
        "contact_point_tool": _vec3(point, "candidate_contact_row.xyz_tool"),
        "motion_axis_ref": "tool.motion_axis",
        "contact_point_face_id": surface_ref or patch_id or face_marker,
        "contact_point_face_marker": face_marker,
        "contact_point_part_ref": row.get("part_ref") or axis_binding.get("selected_contact_point_part_ref"),
        "contact_point_source_ref": contact_ref,
        "contact_point_surface_ref": surface_ref,
        "source_surface_ref": row.get("source_surface_ref") or row.get("surface_ref"),
        "motion_axis_face_marker": axis_binding.get("selected_motion_axis_face_marker")
        or axis_binding.get("selected_motion_axis_marker"),
        "motion_axis_surface_ref": axis_binding.get("selected_motion_axis_surface_ref"),
        "surface_geometry_tool": {
            "frame_source": row.get("selection_semantics") or "contact_point_candidate_row",
            "patch_id": patch_id,
            "surface_center": row.get("xyz_tool"),
            "rectangle_corners": rectangle_corners,
            "long_edge_m": long_edge_m,
            "short_edge_m": short_edge_m,
        },
        "visual_evidence": evidence_ref,
    }
    if motion_axis is not None:
        motion_axis_tool, motion_axis_source, motion_axis_marker, motion_axis_ref = motion_axis
        motion_axis_candidates = _dedupe_motion_axis_candidates(
            [
                _motion_axis_candidate_from_vector(
                    vector_tool=motion_axis_tool,
                    candidate_id=str(motion_axis_ref or motion_axis_marker or "selected_motion_axis"),
                    source="axis_binding.selected_motion_axis",
                    axis_marker=motion_axis_marker,
                    axis_role=motion_axis_source,
                    axis_ref=motion_axis_ref,
                    surface_ref=motion_axis_ref,
                )
            ]
        )
        candidate_ref.update(
            {
                "motion_axis_tool": motion_axis_tool,
                "motion_axis_candidates_tool": motion_axis_candidates,
                "motion_axis_marker": motion_axis_marker,
                "motion_axis_role": motion_axis_source,
                "source_motion_axis_ref": motion_axis_ref,
            }
        )
        candidate_ref["surface_geometry_tool"]["motion_axis_tool"] = motion_axis_tool
        candidate_ref["surface_geometry_tool"]["motion_axis_candidates_tool"] = motion_axis_candidates
        candidate_ref["surface_geometry_tool"]["motion_axis_source"] = motion_axis_source
    return candidate_ref


def build_tool_ref_candidates_from_contact_selection(
    contact_selection: Mapping[str, Any],
    axis_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert selected contact-point alternatives into executable tool refs."""

    rows = contact_selection.get("candidate_contact_rows")
    if not isinstance(rows, list):
        rows = []
        for candidate in contact_selection.get("contact_point_candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            row_data = candidate.get("row")
            row = dict(row_data) if isinstance(row_data, Mapping) else dict(candidate)
            marker = candidate.get("marker") or row.get("contact_marker") or row.get("face_marker")
            if marker is not None:
                row.setdefault("contact_marker", marker)
                row.setdefault("face_marker", marker)
            for key in ("point_ref", "surface_ref", "patch_id", "part_ref"):
                if candidate.get(key) is not None:
                    row.setdefault(key, candidate.get(key))
            rows.append(row)
    if not rows:
        return []
    all_face_rows = contact_selection.get("contact_marker_table")
    if not isinstance(all_face_rows, list):
        all_face_rows = contact_selection.get("contact_point_marker_table")
    if not isinstance(all_face_rows, list):
        all_face_rows = contact_selection.get("face_marker_table")
    if not isinstance(all_face_rows, list):
        all_face_rows = rows
    visual = contact_selection.get("visual_evidence")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate = _candidate_tool_ref_from_contact_row(
            row,
            axis_binding=axis_binding,
            all_face_rows=all_face_rows,
            visual_evidence=visual if isinstance(visual, Mapping) else None,
        )
        key = str(
            candidate.get("contact_point_source_ref")
            or candidate.get("contact_point_surface_ref")
            or candidate.get("contact_point_face_marker")
            or candidate.get("candidate_id")
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _sync_axis_role_constraints_from_tool_refs(program: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve staged axis constraints from merged canonical tool refs.

    Scene lifting can know that an axis has the role ``tool_motion_axis`` or
    ``motion_axis`` before tool-side VLM/P3SAM refs have been merged.  After
    merge, the motion-axis ref is authoritative and must be copied back into
    staged constraints so later lowering does not keep a placeholder.
    """

    tool_refs = _mapping(program.get("tool_refs", {}), "program.tool_refs")
    motion_axis_ref = tool_refs.get("motion_axis_ref")
    if not isinstance(motion_axis_ref, str) or not motion_axis_ref:
        raise ToolRefMergeError("merged tool_refs missing motion_axis_ref")
    role_to_tool_ref = {
        "tool_motion_axis": motion_axis_ref,
        "motion_axis": motion_axis_ref,
    }
    resolutions: list[dict[str, str]] = []
    stages = program.get("stages")
    if not isinstance(stages, list):
        return resolutions
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("name") or "")
        for section_name in ("subgoal_constraints", "path_constraints", "constraints"):
            constraints = stage.get(section_name)
            if not isinstance(constraints, list):
                continue
            for constraint in constraints:
                if not isinstance(constraint, dict) or constraint.get("type") != "axis_align":
                    continue
                params = constraint.get("params")
                if not isinstance(params, dict):
                    continue
                axis_role = str(params.get("axis_role") or "")
                resolved_ref = role_to_tool_ref.get(axis_role)
                if not resolved_ref:
                    continue
                previous_ref = str(params.get("tool_axis_ref") or "")
                if previous_ref == resolved_ref:
                    continue
                params["tool_axis_ref"] = resolved_ref
                resolutions.append(
                    {
                        "stage": stage_name,
                        "section": section_name,
                        "axis_role": axis_role,
                        "previous_tool_axis_ref": previous_ref,
                        "resolved_tool_axis_ref": resolved_ref,
                    }
                )
    return resolutions


def merge_tool_refs_into_constraint_program(
    program: Mapping[str, Any],
    axis_binding: Mapping[str, Any],
    axis_mesh_report: Mapping[str, Any],
    *,
    contact_point_selection: Mapping[str, Any] | None = None,
    axis_binding_path: str | Path | None = None,
    axis_mesh_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a constraint program enriched with canonical tool refs."""

    merged_program = deepcopy(dict(program))
    existing_tool_refs = dict(_mapping(merged_program.get("tool_refs", {}), "program.tool_refs"))
    observed_tool_refs = build_tool_refs_from_axis_evidence(
        axis_binding,
        axis_mesh_report,
        axis_binding_path=axis_binding_path,
        axis_mesh_report_path=axis_mesh_report_path,
    )
    candidate_tool_refs = (
        build_tool_ref_candidates_from_contact_selection(contact_point_selection, axis_binding)
        if contact_point_selection is not None
        else []
    )
    if contact_point_selection is not None:
        tangent_axis_candidates = _motion_axis_candidates_from_contact_selection(
            contact_point_selection,
            axis_binding,
        )
        if tangent_axis_candidates:
            combined_axis_candidates = _dedupe_motion_axis_candidates(
                [
                    *(
                        observed_tool_refs.get("motion_axis_candidates_tool")
                        if isinstance(observed_tool_refs.get("motion_axis_candidates_tool"), list)
                        else []
                    ),
                    *tangent_axis_candidates,
                ]
            )
            observed_tool_refs["motion_axis_candidates_tool"] = combined_axis_candidates
            surface_geometry = observed_tool_refs.get("surface_geometry_tool")
            if isinstance(surface_geometry, dict):
                surface_geometry["motion_axis_candidates_tool"] = combined_axis_candidates
            for candidate_ref in candidate_tool_refs:
                if not isinstance(candidate_ref, dict):
                    continue
                candidate_candidates = _dedupe_motion_axis_candidates(
                    [
                        *(
                            candidate_ref.get("motion_axis_candidates_tool")
                            if isinstance(candidate_ref.get("motion_axis_candidates_tool"), list)
                            else []
                        ),
                        *combined_axis_candidates,
                    ]
                )
                candidate_ref["motion_axis_candidates_tool"] = candidate_candidates
                candidate_surface = candidate_ref.get("surface_geometry_tool")
                if isinstance(candidate_surface, dict):
                    candidate_surface["motion_axis_candidates_tool"] = candidate_candidates
        observed_axis_candidates = (
            observed_tool_refs.get("motion_axis_candidates_tool")
            if isinstance(observed_tool_refs.get("motion_axis_candidates_tool"), list)
            else []
        )
        if observed_axis_candidates:
            for candidate_ref in candidate_tool_refs:
                if not isinstance(candidate_ref, dict):
                    continue
                candidate_candidates = _dedupe_motion_axis_candidates(
                    [
                        *(
                            candidate_ref.get("motion_axis_candidates_tool")
                            if isinstance(candidate_ref.get("motion_axis_candidates_tool"), list)
                            else []
                        ),
                        *observed_axis_candidates,
                    ]
                )
                candidate_ref["motion_axis_candidates_tool"] = candidate_candidates
                candidate_surface = candidate_ref.get("surface_geometry_tool")
                if isinstance(candidate_surface, dict):
                    candidate_surface["motion_axis_candidates_tool"] = candidate_candidates
    merged_program["tool_refs"] = {
        **existing_tool_refs,
        **observed_tool_refs,
        "tool_id": existing_tool_refs.get("tool_id")
        or axis_binding.get("tool_id")
        or merged_program.get("task_id")
        or "image_grounded_tool",
    }
    if candidate_tool_refs:
        merged_program["tool_refs"]["contact_point_candidates_tool"] = candidate_tool_refs
        merged_program["tool_refs"]["contact_point_candidate_selection_policy"] = (
            "selected_contact_point_is_fixed"
        )
    axis_role_resolutions = _sync_axis_role_constraints_from_tool_refs(merged_program)
    if axis_role_resolutions:
        metadata = merged_program.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["axis_role_tool_ref_resolution"] = {
                "source": "merge_tool_refs_into_constraint_program",
                "policy": "stage axis_role constraints use merged canonical tool refs",
                "resolutions": axis_role_resolutions,
            }
    return merged_program
