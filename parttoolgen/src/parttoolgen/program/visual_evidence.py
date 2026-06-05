#!/usr/bin/env python3
"""Visual evidence helpers for VLM planning and repair prompts.

Visual evidence is advisory context for choosing or repairing symbolic refs. It
must not be treated as a source of raw metric poses or controller goals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parttoolgen.validation.validate_parttoolgen import (
    validate_visual_evidence as _validate_visual_evidence,
)


def load_visual_evidence(path: str | Path | None) -> dict[str, Any] | None:
    """Load and validate a visual-evidence manifest."""

    if path is None:
        return None
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_visual_evidence(manifest)
    return manifest


def validate_visual_evidence(manifest: Any) -> None:
    """Validate the compact manifest used by planner and repair prompts."""

    _validate_visual_evidence(manifest)


def _matches_tool(item: dict[str, Any], preferred_tool_id: str | None) -> bool:
    if not preferred_tool_id:
        return False
    needle = preferred_tool_id.lower()
    haystack = " ".join(
        str(item.get(key, ""))
        for key in ("id", "path", "description", "provider_id")
    ).lower()
    return needle in haystack


def _rank_evidence_items(
    items: list[dict[str, Any]],
    *,
    preferred_tool_id: str | None,
    max_items: int,
) -> list[dict[str, Any]]:
    if not preferred_tool_id:
        return items[:max_items]
    tool_items = [item for item in items if _matches_tool(item, preferred_tool_id)]
    global_items = [
        item
        for item in items
        if item.get("id") in {"functional_frame_summary", "sampart3d_region_overlay_index"}
    ]
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*tool_items, *global_items]:
        item_id = str(item.get("id", ""))
        if item_id in seen:
            continue
        ranked.append(item)
        seen.add(item_id)
        if len(ranked) >= max_items:
            break
    return ranked or items[:max_items]


def _summarize_anchor_ref_table(
    metadata: dict[str, Any],
    *,
    preferred_tool_id: str | None = None,
    max_rows: int = 64,
) -> list[dict[str, Any]]:
    rows = metadata.get("anchor_ref_table")
    if not isinstance(rows, list):
        return []
    summary: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if preferred_tool_id and row.get("tool_id") != preferred_tool_id:
            continue
        compact = {
            key: row[key]
            for key in (
                "tool_id",
                "number",
                "anchor_id",
                "part_id",
                "part_ref",
                "point_ref",
                "axis_ref",
                "surface_ref",
                "part_semantic_label",
                "part_semantic_aliases",
                "view_index",
                "feature_cluster_id",
                "score",
            )
            if key in row
        }
        if compact:
            summary.append(compact)
        if len(summary) >= max_rows:
            break
    return summary


def _summarize_marker_table(
    metadata: dict[str, Any],
    *,
    key: str,
    preferred_tool_id: str | None = None,
    max_rows: int = 64,
) -> list[dict[str, Any]]:
    rows = metadata.get(key)
    if not isinstance(rows, list):
        return []
    allowed_keys = (
        "tool_id",
        "part_marker",
        "part_id",
        "part_ref",
        "contact_marker",
        "point_ref",
        "surface_ref",
        "polarity",
        "patch_id",
        "role",
        "positive_side",
        "negative_side",
    )
    summary: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if preferred_tool_id and row.get("tool_id") != preferred_tool_id:
            continue
        compact = {field: row[field] for field in allowed_keys if field in row}
        if compact:
            summary.append(compact)
        if len(summary) >= max_rows:
            break
    return summary


def summarize_visual_evidence(
    manifest: dict[str, Any] | None,
    *,
    allowed_uses: set[str] | tuple[str, ...] = ("planning_context",),
    preferred_tool_id: str | None = None,
    max_items: int = 8,
    max_anchor_rows: int = 64,
) -> dict[str, Any] | None:
    """Return the prompt-safe subset of a visual-evidence manifest."""

    if manifest is None:
        return None
    validate_visual_evidence(manifest)
    allowed_use_set = set(allowed_uses) - {"debug_only"}
    items = [
        item
        for item in manifest["items"]
        if item.get("allowed_use", "planning_context") in allowed_use_set
    ]
    if not items:
        return None
    items = _rank_evidence_items(
        items,
        preferred_tool_id=preferred_tool_id,
        max_items=max_items,
    )
    metadata = manifest.get("metadata", {})
    summary = {
        "schema_version": manifest["schema_version"],
        "evidence_id": manifest["evidence_id"],
        "usage_contract": (
            "Visual evidence may guide semantic disambiguation and repair, but "
            "the program must still use only listed symbol refs and bounded "
            "constraint parameters. Marker numbers are not refs; do not infer raw "
            "coordinates or poses from pixels."
        ),
        "items": [
            {
                key: item[key]
                for key in (
                    "id",
                    "kind",
                    "media_type",
                    "path",
                    "description",
                    "provider_id",
                    "source_stage",
                    "allowed_use",
                )
                if key in item
            }
            for item in items
        ],
    }
    if isinstance(metadata, dict):
        anchor_ref_table = _summarize_anchor_ref_table(
            metadata,
            preferred_tool_id=preferred_tool_id,
            max_rows=max_anchor_rows,
        )
        if anchor_ref_table:
            summary["numbered_anchor_refs"] = {
                "numbering_scope": metadata.get("numbering_scope", "per_tool"),
                "usage": (
                    "Use marker numbers only to find the exact listed refs. In "
                    "build_program(ctx), call ctx.ref(...) on refs from the symbol_table."
                ),
                "rows": anchor_ref_table,
            }
        contact_marker_table = _summarize_marker_table(
            metadata,
            key="contact_marker_table",
            preferred_tool_id=preferred_tool_id,
            max_rows=max_anchor_rows,
        )
        if contact_marker_table:
            summary["contact_marker_refs"] = {
                "usage": (
                    "C* markers identify exact candidate.points.* function contacts "
                    "and matching same-contact candidate.surfaces.* outgoing directions. "
                    "Use these rows to map a visible point+arrow marker to the symbolic "
                    "function contact and function-axis refs."
                ),
                "rows": contact_marker_table,
            }
    return summary


def first_prompt_image_path(
    manifest: dict[str, Any] | None,
    *,
    allowed_uses: set[str] | tuple[str, ...] = ("planning_context",),
    preferred_tool_id: str | None = None,
) -> str | None:
    """Return the first image path suitable for single-image VLM clients."""

    if manifest is None:
        return None
    validate_visual_evidence(manifest)
    allowed_use_set = set(allowed_uses) - {"debug_only"}
    image_items: list[dict[str, Any]] = []
    for item in manifest["items"]:
        if item.get("media_type") != "image":
            continue
        if item.get("allowed_use", "planning_context") not in allowed_use_set:
            continue
        image_items.append(item)
    if preferred_tool_id:
        for item in image_items:
            if _matches_tool(item, preferred_tool_id):
                return str(item["path"])
    for item in image_items:
        return str(item["path"])
    return None


def prompt_image_paths(
    manifest: dict[str, Any] | None,
    *,
    allowed_uses: set[str] | tuple[str, ...] = ("planning_context",),
    preferred_tool_id: str | None = None,
    max_images: int = 4,
) -> list[str]:
    """Return prompt images, preferring the requested tool's multiview overlays.

    The returned paths are marked visual context only. Marker numbers must still
    resolve through the visual-evidence table and the symbolic whitelist.
    """

    if manifest is None:
        return []
    validate_visual_evidence(manifest)
    allowed_use_set = set(allowed_uses) - {"debug_only"}
    image_items = [
        item
        for item in manifest["items"]
        if item.get("media_type") == "image"
        and item.get("allowed_use", "planning_context") in allowed_use_set
    ]
    ranked = _rank_evidence_items(
        image_items,
        preferred_tool_id=preferred_tool_id,
        max_items=max_images,
    )
    paths: list[str] = []
    seen: set[str] = set()
    for item in ranked:
        path = str(item.get("path", ""))
        if not path or path in seen:
            continue
        paths.append(path)
        seen.add(path)
        if len(paths) >= max_images:
            break
    return paths
