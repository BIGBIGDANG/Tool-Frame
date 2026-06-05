#!/usr/bin/env python3
"""Compile a ToolFrame constraint program into ToolPoseSubgoals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "parttoolgen" / "src").is_dir():
            return path
    raise RuntimeError("Could not locate ToolFrame repository root")


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.adapters.simtoolreal_adapter import convert_subgoals_to_simtoolreal  # noqa: E402
from parttoolgen.image_grounded import compile_constraint_program_to_tool_pose_subgoals  # noqa: E402
from parttoolgen.validation.validate_parttoolgen import (  # noqa: E402
    validate_simtoolreal_fixed_goals,
    validate_tool_pose_subgoals,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True, help="ToolFrame constraint_program_v1 JSON.")
    parser.add_argument("--output", required=True, help="Output ToolPoseSubgoals JSON.")
    parser.add_argument(
        "--start-pose",
        default=None,
        help="Optional JSON pose object with position/orientation_xyzw.",
    )
    parser.add_argument(
        "--fixed-goals-output",
        default=None,
        help="Optional output path for SimToolReal fixed goals via the existing adapter.",
    )
    args = parser.parse_args()

    program_path = Path(args.program)
    program = _load_json(program_path)
    start_pose = _load_json(args.start_pose) if args.start_pose else None
    subgoals = compile_constraint_program_to_tool_pose_subgoals(
        program,
        start_pose=start_pose,
        program_path=str(program_path),
    )
    validate_tool_pose_subgoals(subgoals)
    _write_json(args.output, subgoals)

    result = {"tool_pose_subgoals": str(Path(args.output))}
    if args.fixed_goals_output:
        fixed_goals = convert_subgoals_to_simtoolreal(subgoals)
        validate_simtoolreal_fixed_goals(fixed_goals)
        _write_json(args.fixed_goals_output, fixed_goals)
        result["simtoolreal_fixed_goals"] = str(Path(args.fixed_goals_output))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
