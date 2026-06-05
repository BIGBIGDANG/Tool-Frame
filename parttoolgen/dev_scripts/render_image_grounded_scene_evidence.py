#!/usr/bin/env python3
"""Render calibrated scene evidence for image-grounded ToolFrame planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "parttoolgen" / "src").is_dir():
            return path
    raise RuntimeError("Could not locate ToolFrame repository root")


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.image_grounded.scene_evidence import render_topdown_scene_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Tool-use task JSON.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    task_data = json.loads(Path(args.task).read_text(encoding="utf-8"))
    result = render_topdown_scene_evidence(task_data=task_data, output_dir=Path(args.output_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
