"""Catalog SimToolReal tools that can use segmented tool-prior evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_SEGMENTED_TOOL_MANIFEST = (
    "artifacts/image_grounded/tool_ref_grounding/part_selection/"
    "all_segmented_tools_20260525/manifest.json"
)


@dataclass(frozen=True)
class SimToolRealToolSpec:
    tool_id: str
    category: str
    mesh: str
    urdf: str
    p3sam_prior: str
    environments: tuple[str, ...]
    trajectories: tuple[str, ...]
    smoke_tasks: tuple[str, ...]


SIMTOOLREAL_SEGMENTED_TOOL_SPECS: tuple[SimToolRealToolSpec, ...] = (
    SimToolRealToolSpec(
        tool_id="blue_brush",
        category="brush",
        mesh="simtoolreal/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/brush/blue_brush/blue_brush.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/blue_brush/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/brush/blue_brush/sweep_forward.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/brush/blue_brush/sweep_right.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/brush/blue_brush/sweep_forward.json",
            "simtoolreal/dextoolbench/trajectories/brush/blue_brush/sweep_right.json",
        ),
        smoke_tasks=("sweep_forward", "sweep_right"),
    ),
    SimToolRealToolSpec(
        tool_id="red_brush",
        category="brush",
        mesh="simtoolreal/assets/urdf/dextoolbench/brush/red_brush/red_brush.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/brush/red_brush/red_brush.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/red_brush/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/brush/red_brush/sweep_forward.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/brush/red_brush/sweep_right.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/brush/red_brush/sweep_forward.json",
            "simtoolreal/dextoolbench/trajectories/brush/red_brush/sweep_right.json",
        ),
        smoke_tasks=("sweep_forward", "sweep_right"),
    ),
    SimToolRealToolSpec(
        tool_id="flat_eraser",
        category="eraser",
        mesh="simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/flat_eraser/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/eraser/flat_eraser/wipe_c.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/eraser/flat_eraser/wipe_smile.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/eraser/flat_eraser/wipe_c.json",
            "simtoolreal/dextoolbench/trajectories/eraser/flat_eraser/wipe_smile.json",
        ),
        smoke_tasks=("wipe_c", "wipe_smile"),
    ),
    SimToolRealToolSpec(
        tool_id="handle_eraser",
        category="eraser",
        mesh="simtoolreal/assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/handle_eraser/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/eraser/handle_eraser/wipe_c.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/eraser/handle_eraser/wipe_smile.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/eraser/handle_eraser/wipe_c.json",
            "simtoolreal/dextoolbench/trajectories/eraser/handle_eraser/wipe_smile.json",
        ),
        smoke_tasks=("wipe_c", "wipe_smile"),
    ),
    SimToolRealToolSpec(
        tool_id="claw_hammer",
        category="hammer",
        mesh="simtoolreal/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/claw_hammer/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/hammer/claw_hammer/swing_down.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/hammer/claw_hammer/swing_side.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/hammer/claw_hammer/swing_down.json",
            "simtoolreal/dextoolbench/trajectories/hammer/claw_hammer/swing_side.json",
        ),
        smoke_tasks=("swing_down", "swing_side"),
    ),
    SimToolRealToolSpec(
        tool_id="mallet_hammer",
        category="hammer",
        mesh="simtoolreal/assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/mallet_hammer/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/hammer/mallet_hammer/swing_down.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/hammer/mallet_hammer/swing_side.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/hammer/mallet_hammer/swing_down.json",
            "simtoolreal/dextoolbench/trajectories/hammer/mallet_hammer/swing_side.json",
        ),
        smoke_tasks=("swing_down", "swing_side"),
    ),
    SimToolRealToolSpec(
        tool_id="sharpie_marker",
        category="marker",
        mesh="simtoolreal/assets/urdf/dextoolbench/marker/sharpie_marker/sharpie_marker.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/marker/sharpie_marker/sharpie_marker.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/sharpie_marker/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/marker/sharpie_marker/draw_smile.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/marker/sharpie_marker/write_c.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/marker/sharpie_marker/draw_smile.json",
            "simtoolreal/dextoolbench/trajectories/marker/sharpie_marker/write_c.json",
        ),
        smoke_tasks=("write_c", "draw_smile"),
    ),
    SimToolRealToolSpec(
        tool_id="staples_marker",
        category="marker",
        mesh="simtoolreal/assets/urdf/dextoolbench/marker/staples_marker/staples_marker.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/marker/staples_marker/staples_marker.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/staples_marker/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/marker/staples_marker/draw_smile.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/marker/staples_marker/write_c.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/marker/staples_marker/draw_smile.json",
            "simtoolreal/dextoolbench/trajectories/marker/staples_marker/write_c.json",
        ),
        smoke_tasks=("write_c", "draw_smile"),
    ),
    SimToolRealToolSpec(
        tool_id="long_screwdriver",
        category="screwdriver",
        mesh="simtoolreal/assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/long_screwdriver/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/screwdriver/long_screwdriver/spin_horizontal.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/screwdriver/long_screwdriver/spin_vertical.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/screwdriver/long_screwdriver/spin_horizontal.json",
            "simtoolreal/dextoolbench/trajectories/screwdriver/long_screwdriver/spin_vertical.json",
        ),
        smoke_tasks=("spin_horizontal", "spin_vertical"),
    ),
    SimToolRealToolSpec(
        tool_id="short_screwdriver",
        category="screwdriver",
        mesh="simtoolreal/assets/urdf/dextoolbench/screwdriver/short_screwdriver/short_screwdriver.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/screwdriver/short_screwdriver/short_screwdriver.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/short_screwdriver/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/screwdriver/short_screwdriver/spin_horizontal.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/screwdriver/short_screwdriver/spin_vertical.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/screwdriver/short_screwdriver/spin_horizontal.json",
            "simtoolreal/dextoolbench/trajectories/screwdriver/short_screwdriver/spin_vertical.json",
        ),
        smoke_tasks=("spin_horizontal", "spin_vertical"),
    ),
    SimToolRealToolSpec(
        tool_id="flat_spatula",
        category="spatula",
        mesh="simtoolreal/assets/urdf/dextoolbench/spatula/flat_spatula/flat_spatula.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/spatula/flat_spatula/flat_spatula.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/flat_spatula/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/spatula/flat_spatula/flip_over.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/spatula/flat_spatula/serve_plate.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/spatula/flat_spatula/flip_over.json",
            "simtoolreal/dextoolbench/trajectories/spatula/flat_spatula/serve_plate.json",
        ),
        smoke_tasks=("serve_plate", "flip_over"),
    ),
    SimToolRealToolSpec(
        tool_id="spoon_spatula",
        category="spatula",
        mesh="simtoolreal/assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.obj",
        urdf="simtoolreal/assets/urdf/dextoolbench/spatula/spoon_spatula/spoon_spatula.urdf",
        p3sam_prior="artifacts/tool_prior/p3sam/spoon_spatula/tool_prior",
        environments=(
            "simtoolreal/assets/urdf/dextoolbench/environments/spatula/spoon_spatula/flip_over.urdf",
            "simtoolreal/assets/urdf/dextoolbench/environments/spatula/spoon_spatula/serve_plate.urdf",
        ),
        trajectories=(
            "simtoolreal/dextoolbench/trajectories/spatula/spoon_spatula/flip_over.json",
            "simtoolreal/dextoolbench/trajectories/spatula/spoon_spatula/serve_plate.json",
        ),
        smoke_tasks=("serve_plate", "flip_over"),
    ),
)


def iter_simtoolreal_segmented_tool_specs() -> tuple[SimToolRealToolSpec, ...]:
    return SIMTOOLREAL_SEGMENTED_TOOL_SPECS


def get_simtoolreal_segmented_tool_spec(tool_id: str) -> SimToolRealToolSpec:
    for spec in SIMTOOLREAL_SEGMENTED_TOOL_SPECS:
        if spec.tool_id == tool_id:
            return spec
    raise KeyError(f"unknown SimToolReal segmented tool: {tool_id}")


def _resolve(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _load_manifest(manifest_path: Path) -> Mapping[str, Any] | None:
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else None


def _manifest_tool_entries(manifest: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    if not isinstance(manifest, Mapping):
        return entries
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        return entries
    for row in tools:
        if isinstance(row, Mapping) and isinstance(row.get("tool_id"), str):
            entries[str(row["tool_id"])] = row
    return entries


def build_segmented_tool_inventory(
    *,
    repo_root: str | Path,
    manifest_path: str | Path = DEFAULT_SEGMENTED_TOOL_MANIFEST,
) -> dict[str, Any]:
    root = Path(repo_root)
    manifest_abs = _resolve(root, str(manifest_path))
    manifest = _load_manifest(manifest_abs)
    manifest_entries = _manifest_tool_entries(manifest)
    rows: list[dict[str, Any]] = []
    for spec in SIMTOOLREAL_SEGMENTED_TOOL_SPECS:
        entry = manifest_entries.get(spec.tool_id)
        row = asdict(spec)
        row["mesh_exists"] = _resolve(root, spec.mesh).exists()
        row["urdf_exists"] = _resolve(root, spec.urdf).exists()
        row["p3sam_prior_exists"] = _resolve(root, spec.p3sam_prior).exists()
        row["manifest_entry_present"] = entry is not None
        row["part_count"] = entry.get("part_count") if isinstance(entry, Mapping) else None
        row["part_card_count"] = entry.get("part_card_count") if isinstance(entry, Mapping) else None
        row["overview_count"] = entry.get("overview_count") if isinstance(entry, Mapping) else None
        row["ready_for_tool_side_grounding"] = bool(
            row["mesh_exists"]
            and row["urdf_exists"]
            and row["p3sam_prior_exists"]
            and row["manifest_entry_present"]
        )
        rows.append(row)
    ready = [row for row in rows if row["ready_for_tool_side_grounding"]]
    return {
        "schema_version": "toolframe_simtoolreal_segmented_tool_inventory_v1",
        "manifest_path": str(manifest_abs),
        "manifest_present": manifest is not None,
        "tool_count": len(rows),
        "ready_tool_count": len(ready),
        "tools": rows,
    }


def tool_ids(specs: Iterable[SimToolRealToolSpec] = SIMTOOLREAL_SEGMENTED_TOOL_SPECS) -> list[str]:
    return [spec.tool_id for spec in specs]


__all__ = [
    "DEFAULT_SEGMENTED_TOOL_MANIFEST",
    "SIMTOOLREAL_SEGMENTED_TOOL_SPECS",
    "SimToolRealToolSpec",
    "build_segmented_tool_inventory",
    "get_simtoolreal_segmented_tool_spec",
    "iter_simtoolreal_segmented_tool_specs",
    "tool_ids",
]
