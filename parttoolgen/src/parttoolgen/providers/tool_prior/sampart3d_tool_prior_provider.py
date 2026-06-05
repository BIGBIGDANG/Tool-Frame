#!/usr/bin/env python3
"""Convert SAMPart3D mesh regions into PartToolGen tool priors.

The current mainline is real offline SAMPart3D part candidates plus deterministic
geometry surface/endpoint candidates. This adapter writes ToolPartGraph,
FunctionalToolParts, GroundingCandidates, and ProviderContract artifacts in the
stable SimToolReal tool/asset frame so downstream VLM binding and compiler
stages can bind task semantics to stable refs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    validate_functional_tool_parts,
    validate_grounding_candidates,
    validate_provider_contract,
    validate_tool_part_graph,
)
from parttoolgen.providers.tool_prior.sampart3d_mesh_candidates import (  # noqa: E402
    build_sampart3d_mesh_tool_prior,
)
from parttoolgen.program.part_semantic_labels import (  # noqa: E402
    load_tool_part_semantic_labels,
    merge_part_semantic_labels,
)


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def convert_sampart3d_label_file(
    *,
    tool_id: str,
    source_mesh: str | Path,
    face_labels_path: str | Path,
    output_dir: str | Path,
    observation_id: str | None = None,
    part_semantic_labels_path: str | Path | None = None,
    min_face_count: int = 1,
    segmentation_mode: str = "sampart3d",
    provider_id: str | None = None,
) -> dict[str, Any]:
    artifacts = build_sampart3d_mesh_tool_prior(
        tool_id=tool_id,
        source_mesh=source_mesh,
        face_labels_path=face_labels_path,
        observation_id=observation_id,
        min_face_count=min_face_count,
        segmentation_mode=segmentation_mode,
        provider_id=provider_id,
    )
    part_semantic_labels = load_tool_part_semantic_labels(part_semantic_labels_path)
    if part_semantic_labels is not None:
        merged_tool_part_graph, merged_grounding_candidates = merge_part_semantic_labels(
            tool_part_graph=artifacts["tool_part_graph"],
            semantic_labels=part_semantic_labels,
            grounding_candidates=artifacts["grounding_candidates"],
        )
        artifacts["tool_part_graph"] = merged_tool_part_graph
        artifacts["grounding_candidates"] = merged_grounding_candidates
        artifacts["provider_contract"]["provenance"].setdefault("source_artifacts", []).append(
            str(part_semantic_labels_path)
        )
        artifacts["provider_contract"].setdefault("metadata", {})["part_semantic_labels"] = {
            "label_id": part_semantic_labels["label_id"],
            "mode": part_semantic_labels["metadata"].get("mode"),
        }
    validate_tool_part_graph(artifacts["tool_part_graph"])
    validate_functional_tool_parts(artifacts["functional_tool_parts"])
    validate_grounding_candidates(artifacts["grounding_candidates"])
    validate_provider_contract(artifacts["provider_contract"])

    output = Path(output_dir)
    _write_json(output / "tool_part_graph.json", artifacts["tool_part_graph"])
    _write_json(output / "functional_tool_parts.json", artifacts["functional_tool_parts"])
    _write_json(output / "grounding_candidates.json", artifacts["grounding_candidates"])
    _write_json(output / "provider_contract.json", artifacts["provider_contract"])
    manifest_outputs = {
        "tool_part_graph": str(output / "tool_part_graph.json"),
        "functional_tool_parts": str(output / "functional_tool_parts.json"),
        "grounding_candidates": str(output / "grounding_candidates.json"),
        "provider_contract": str(output / "provider_contract.json"),
    }
    if part_semantic_labels_path is not None:
        manifest_outputs["tool_part_semantic_labels"] = str(Path(part_semantic_labels_path))
    _write_json(
        output / "manifest.json",
        {
            "generated_by": "parttoolgen.providers.tool_prior.sampart3d_tool_prior_provider",
            "outputs": manifest_outputs,
        },
    )
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build 5.15 SAMPart3D geometry functional-frame priors from "
            "--source-mesh and --face-labels."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--observation-id")
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--face-labels", required=True)
    parser.add_argument("--part-semantic-labels")
    parser.add_argument("--min-face-count", type=int, default=1)
    parser.add_argument("--segmentation-mode", default="sampart3d")
    parser.add_argument("--provider-id")
    args = parser.parse_args()
    convert_sampart3d_label_file(
        tool_id=args.tool_id,
        source_mesh=args.source_mesh,
        face_labels_path=args.face_labels,
        output_dir=args.output_dir,
        observation_id=args.observation_id,
        part_semantic_labels_path=args.part_semantic_labels,
        min_face_count=args.min_face_count,
        segmentation_mode=args.segmentation_mode,
        provider_id=args.provider_id,
    )
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
