#!/usr/bin/env python3
"""Render generic tool-reference visual evidence from a cached tool-part prior.

This is the mainline renderer entry point.  It consumes a backend-agnostic
tool-part prior directory containing files such as tool_part_graph.json,
grounding_candidates.json, source mesh metadata, face labels, and a render
cache.  The upstream segmentation backend may be SAMPart3D, P3SAM, manual
labels, or any future part-prior generator; this entry point intentionally does
not expose that backend in the pipeline command.

The implementation currently reuses the mature renderer module for
compatibility while the public pipeline depends only on this generic script.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "ToGo" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ToGo.cli.tool_prior.render_sampart3d_part_role_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
