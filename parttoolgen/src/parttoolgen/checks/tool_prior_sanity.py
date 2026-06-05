#!/usr/bin/env python3
"""Sanity checks for SAMPart3D geometry-first tool-prior artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    load_structured_file,
    validate_functional_tool_parts,
    validate_grounding_candidates,
    validate_provider_contract,
    validate_tool_part_graph,
)


class ToolPriorSanityError(ValueError):
    """Raised when tool-prior artifacts are schema-valid but internally unsafe."""


EPS = 1e-6


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return Path(text)


def _load_json(path: Path) -> dict[str, Any]:
    return load_structured_file(path)


def _finite_vec(values: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ToolPriorSanityError(f"{name} must be a length-{length} list")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ToolPriorSanityError(f"{name} contains non-finite values")
    return result


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _require_unit_vector(values: Any, *, name: str, tolerance: float = 0.2) -> None:
    vector = _finite_vec(values, length=3, name=name)
    norm = _norm(vector)
    if abs(norm - 1.0) > tolerance:
        raise ToolPriorSanityError(f"{name} norm {norm:.4f} is not unit-like")


def _assert_no_removed_dino_route(
    *,
    outputs: dict[str, Any],
    grounding: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if outputs.get("dinov2_anchors") or grounding.get("metadata", {}).get("dino_anchors"):
        raise ToolPriorSanityError("5.15 tool priors must not include DINO anchor artifacts")
    if grounding.get("metadata", {}).get("anchor_mode") == "dinov2_multiview":
        raise ToolPriorSanityError("5.15 tool priors must use geometry functional-frame candidates")
    if contract.get("provenance", {}).get("method") == "sampart3d_face_labels_with_dinov2_anchors":
        raise ToolPriorSanityError("DINO-backed provider method is not part of the 5.15 mainline")
    if int(contract.get("metadata", {}).get("dino_anchor_count", 0)) != 0:
        raise ToolPriorSanityError("provider_contract.metadata.dino_anchor_count must be zero/absent")
    for candidate_id, value in grounding.get("points", {}).items():
        if value.get("source") == "dinov2_tool_prior" or "dinov2" in str(value.get("candidate_type", "")).lower():
            raise ToolPriorSanityError(f"removed DINO point candidate leaked into 5.15 prior: {candidate_id}")


def check_tool_prior_directory(prior_dir: str | Path) -> dict[str, Any]:
    prior_root = Path(prior_dir)
    graph = _load_json(prior_root / "tool_part_graph.json")
    functional = _load_json(prior_root / "functional_tool_parts.json")
    grounding = _load_json(prior_root / "grounding_candidates.json")
    contract = _load_json(prior_root / "provider_contract.json")
    manifest_path = prior_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {"outputs": {}}

    validate_tool_part_graph(graph)
    validate_functional_tool_parts(functional)
    validate_grounding_candidates(grounding)
    validate_provider_contract(contract)

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    _assert_no_removed_dino_route(outputs=outputs, grounding=grounding, contract=contract)

    parts = graph.get("parts", {}) if isinstance(graph.get("parts"), dict) else {}
    points = grounding.get("points", {}) if isinstance(grounding.get("points"), dict) else {}
    axes = grounding.get("axes", {}) if isinstance(grounding.get("axes"), dict) else {}
    surfaces = grounding.get("surfaces", {}) if isinstance(grounding.get("surfaces"), dict) else {}
    warnings: list[str] = []
    for part_id in parts:
        part_ref = f"tool.parts.{part_id}"
        part_points = [item for item in points.values() if item.get("part_ref") == part_ref]
        part_surfaces = [item for item in surfaces.values() if item.get("part_ref") == part_ref]
        if not part_points:
            raise ToolPriorSanityError(f"{part_ref} has no grounded point candidates")
        if not part_surfaces:
            raise ToolPriorSanityError(f"{part_ref} has no grounded surface candidates")

    for candidate_id, point in points.items():
        _finite_vec(point.get("xyz_tool"), length=3, name=f"points.{candidate_id}.xyz_tool")
        if point.get("candidate_type") == "whole_tool_center":
            continue
        if not str(point.get("part_ref", "")).startswith("tool.parts."):
            raise ToolPriorSanityError(f"points.{candidate_id} must belong to a tool part")
    for candidate_id, axis in axes.items():
        _require_unit_vector(axis.get("vector_tool"), name=f"axes.{candidate_id}.vector_tool")
    for candidate_id, surface in surfaces.items():
        _require_unit_vector(surface.get("normal_tool"), name=f"surfaces.{candidate_id}.normal_tool")

    return {
        "passed": True,
        "tool_id": graph.get("tool_id"),
        "anchor_mode": grounding.get("metadata", {}).get("anchor_mode"),
        "part_count": len(parts),
        "point_candidate_count": len(points),
        "axis_candidate_count": len(axes),
        "surface_candidate_count": len(surfaces),
        "warnings": warnings,
    }


def _iter_prior_dirs(run_root: Path) -> list[Path]:
    if (run_root / "tool_prior").exists():
        return [run_root / "tool_prior"]
    if (run_root / "tool_part_graph.json").exists():
        return [run_root]
    return sorted(path for path in run_root.glob("*/tool_prior") if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-prior-dir")
    parser.add_argument("--run-root")
    parser.add_argument("--report")
    args = parser.parse_args()
    if not args.tool_prior_dir and not args.run_root:
        parser.error("one of --tool-prior-dir or --run-root is required")

    prior_dirs = [Path(args.tool_prior_dir)] if args.tool_prior_dir else _iter_prior_dirs(Path(args.run_root))
    results = []
    for prior_dir in prior_dirs:
        result = check_tool_prior_directory(prior_dir)
        result["tool_prior_dir"] = str(prior_dir)
        results.append(result)

    report = {"passed": True, "tool_count": len(results), "results": results}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
