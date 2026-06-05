#!/usr/bin/env python3
"""Merge tool-side contact point and motion axis refs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.image_grounded import (
    infer_axis_mesh_report_path,
    merge_tool_refs_into_constraint_program,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path, help="Input constraint_program.json")
    parser.add_argument("--axis-binding", required=True, type=Path, help="VLM axis-binding JSON")
    parser.add_argument(
        "--axis-mesh-report",
        type=Path,
        help="Motion-axis mesh overlay report JSON. If omitted, infer it from axis_binding.visual_evidence.",
    )
    parser.add_argument(
        "--contact-point-selection",
        type=Path,
        help="Optional VLM contact-point selection JSON containing multiple executable contact point candidates.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Enriched constraint program path")
    args = parser.parse_args()

    program = _load_json(args.program)
    axis_binding = _load_json(args.axis_binding)
    report_path = args.axis_mesh_report
    if report_path is None:
        report_path = infer_axis_mesh_report_path(axis_binding, axis_binding_path=args.axis_binding)
    if report_path is None:
        raise FileNotFoundError(
            "Could not infer axis mesh report. Pass --axis-mesh-report explicitly."
        )
    axis_mesh_report = _load_json(report_path)
    contact_point_selection = (
        _load_json(args.contact_point_selection)
        if args.contact_point_selection is not None
        else None
    )
    enriched = merge_tool_refs_into_constraint_program(
        program,
        axis_binding,
        axis_mesh_report,
        contact_point_selection=contact_point_selection,
        axis_binding_path=args.axis_binding,
        axis_mesh_report_path=report_path,
    )
    _write_json(args.output, enriched)
    tool_refs = enriched["tool_refs"]
    print(
        "merged tool refs:",
        {
            "contact_point_face_marker": tool_refs.get("contact_point_face_marker"),
            "contact_point_tool": tool_refs.get("contact_point_tool"),
            "motion_axis_tool": tool_refs.get("motion_axis_tool"),
            "contact_point_candidate_count": len(
                tool_refs.get("contact_point_candidates_tool") or []
            ),
            "output": str(args.output),
        },
    )


if __name__ == "__main__":
    main()
