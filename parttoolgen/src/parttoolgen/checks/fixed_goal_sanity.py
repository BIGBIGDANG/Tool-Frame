#!/usr/bin/env python3
"""Sanity checks for SimToolReal fixed-goal tool-pose lists."""

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

from parttoolgen.validation.validate_parttoolgen import (
    load_structured_file,
    validate_simtoolreal_fixed_goals,
)
from parttoolgen.adapters.simtoolreal_policy_priors import (  # noqa: E402
    DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
    mesh_table_clearance_report,
)


class FixedGoalSanityError(ValueError):
    """Raised when a fixed-goal list is schema-valid but unsafe to pass onward."""


EPS = 1e-6


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _quat_angle_deg(q1: list[float], q2: list[float]) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def check_simtoolreal_fixed_goals(
    data: dict[str, Any],
    *,
    max_translation_step: float = 0.25,
    max_rotation_step_deg: float = 75.0,
    min_z: float = 0.0,
    max_z: float = 1.5,
    table_top_z: float | None = None,
    min_root_table_clearance_m: float = 0.01,
    tool_id: str | None = None,
    tool_mesh_path: str | Path | None = None,
    min_tool_table_clearance_m: float = DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
) -> dict[str, Any]:
    validate_simtoolreal_fixed_goals(data)
    poses = [data["start_pose"], *data["goals"]]
    warnings: list[str] = []
    min_table_root_z = (
        table_top_z + min_root_table_clearance_m
        if table_top_z is not None
        else None
    )
    for i, pose in enumerate(poses):
        z = float(pose[2])
        if z < min_z:
            raise FixedGoalSanityError(f"pose {i} z={z:.4f} below min_z={min_z:.4f}")
        if z > max_z:
            raise FixedGoalSanityError(f"pose {i} z={z:.4f} above max_z={max_z:.4f}")
        if min_table_root_z is not None and z + EPS < min_table_root_z:
            raise FixedGoalSanityError(
                f"pose {i} z={z:.4f} below table clearance "
                f"{min_table_root_z:.4f} (table_top_z={table_top_z:.4f}, "
                f"min_root_table_clearance_m={min_root_table_clearance_m:.4f})"
            )
    for i, (prev_pose, pose) in enumerate(zip(poses, poses[1:]), start=1):
        translation = _dist(prev_pose[:3], pose[:3])
        rotation = _quat_angle_deg(prev_pose[3:7], pose[3:7])
        if translation > max_translation_step:
            raise FixedGoalSanityError(
                f"pose jump {i - 1}->{i} translation {translation:.4f} exceeds {max_translation_step:.4f}"
            )
        if rotation > max_rotation_step_deg:
            raise FixedGoalSanityError(
                f"pose jump {i - 1}->{i} rotation {rotation:.2f} deg exceeds {max_rotation_step_deg:.2f}"
            )
        if translation > 0.15:
            warnings.append(f"pose jump {i - 1}->{i} translation {translation:.4f} is high")
        if rotation > 45.0:
            warnings.append(f"pose jump {i - 1}->{i} rotation {rotation:.2f} deg is high")
    result = {
        "passed": True,
        "failures": [],
        "warnings": warnings,
        "num_goals": len(data["goals"]),
        "table_top_z": table_top_z,
        "min_root_table_clearance_m": min_root_table_clearance_m,
        "min_tool_table_clearance_m": min_tool_table_clearance_m,
    }
    if table_top_z is not None and (tool_id or tool_mesh_path):
        mesh_report = mesh_table_clearance_report(
            data,
            table_top_z=table_top_z,
            tool_id=tool_id,
            tool_mesh_path=tool_mesh_path,
            min_tool_table_clearance_m=min_tool_table_clearance_m,
        )
        result["mesh_table_clearance"] = mesh_report
        if mesh_report.get("checked") and not mesh_report.get("passed"):
            first = mesh_report["violations"][0]
            raise FixedGoalSanityError(
                f"{first['pose']} tool mesh clearance {first['clearance_m']:.4f} "
                f"is below required {first['required_clearance_m']:.4f} "
                f"(mesh_min_z={first['mesh_min_z']:.4f}, table_top_z={table_top_z:.4f})"
            )
        if not mesh_report.get("checked"):
            warnings.append("tool mesh clearance was requested but no mesh was resolved")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-goals", required=True)
    parser.add_argument("--report")
    parser.add_argument("--table-top-z", type=float)
    parser.add_argument("--min-root-table-clearance-m", type=float, default=0.01)
    parser.add_argument("--tool-id")
    parser.add_argument("--tool-mesh")
    parser.add_argument(
        "--min-tool-table-clearance-m",
        type=float,
        default=DEFAULT_MIN_TOOL_TABLE_CLEARANCE_M,
    )
    args = parser.parse_args()
    data = load_structured_file(args.fixed_goals)
    result = check_simtoolreal_fixed_goals(
        data,
        table_top_z=args.table_top_z,
        min_root_table_clearance_m=args.min_root_table_clearance_m,
        tool_id=args.tool_id,
        tool_mesh_path=args.tool_mesh,
        min_tool_table_clearance_m=args.min_tool_table_clearance_m,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
