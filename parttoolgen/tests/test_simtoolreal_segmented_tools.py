from __future__ import annotations

from pathlib import Path

import pytest

from parttoolgen.evaluation.simtoolreal_segmented_tools import (
    build_segmented_tool_inventory,
    get_simtoolreal_segmented_tool_spec,
    tool_ids,
)


def test_catalog_covers_current_simtoolreal_segmented_tool_set() -> None:
    assert tool_ids() == [
        "blue_brush",
        "red_brush",
        "flat_eraser",
        "handle_eraser",
        "claw_hammer",
        "mallet_hammer",
        "sharpie_marker",
        "staples_marker",
        "long_screwdriver",
        "short_screwdriver",
        "flat_spatula",
        "spoon_spatula",
    ]


def test_catalog_entries_use_repo_relative_simtoolreal_and_p3sam_paths() -> None:
    spec = get_simtoolreal_segmented_tool_spec("flat_eraser")

    assert spec.category == "eraser"
    assert spec.mesh == "simtoolreal/assets/urdf/dextoolbench/eraser/flat_eraser/flat_eraser.obj"
    assert spec.p3sam_prior == "artifacts/tool_prior/p3sam/flat_eraser/tool_prior"
    assert "wipe_c" in spec.smoke_tasks


def test_local_segmented_inventory_when_manifest_is_available() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = (
        repo_root
        / "artifacts/image_grounded/tool_ref_grounding/part_selection/"
        "all_segmented_tools_20260525/manifest.json"
    )
    if not manifest.exists():
        pytest.skip("local P3SAM segmented-tool manifest is an ignored runtime artifact")

    inventory = build_segmented_tool_inventory(repo_root=repo_root, manifest_path=manifest)

    assert inventory["tool_count"] == 12
    assert inventory["ready_tool_count"] == 12
    assert all(row["ready_for_tool_side_grounding"] for row in inventory["tools"])
