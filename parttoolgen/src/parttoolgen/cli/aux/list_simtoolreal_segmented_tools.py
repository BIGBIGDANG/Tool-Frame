#!/usr/bin/env python3
"""List SimToolReal tools with local segmented tool-prior evidence."""

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

from parttoolgen.evaluation.simtoolreal_segmented_tools import (  # noqa: E402
    DEFAULT_SEGMENTED_TOOL_MANIFEST,
    build_segmented_tool_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_SEGMENTED_TOOL_MANIFEST)
    parser.add_argument("--require-local-ready", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    inventory = build_segmented_tool_inventory(repo_root=ROOT, manifest_path=args.manifest)
    if args.compact:
        rows = [
            {
                "tool_id": row["tool_id"],
                "category": row["category"],
                "ready": row["ready_for_tool_side_grounding"],
                "part_count": row["part_count"],
                "smoke_tasks": row["smoke_tasks"],
            }
            for row in inventory["tools"]
        ]
        payload = {
            "schema_version": inventory["schema_version"],
            "manifest_present": inventory["manifest_present"],
            "tool_count": inventory["tool_count"],
            "ready_tool_count": inventory["ready_tool_count"],
            "tools": rows,
        }
    else:
        payload = inventory
    print(json.dumps(payload, indent=2))
    if args.require_local_ready and inventory["ready_tool_count"] != inventory["tool_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
