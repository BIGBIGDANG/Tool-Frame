from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "parttoolgen" / "scripts"
CLI_DIR = ROOT / "parttoolgen" / "src" / "parttoolgen" / "cli"
CLI_RUNNERS_DIR = ROOT / "parttoolgen" / "src" / "parttoolgen" / "cli" / "runners"
INVENTORY_PATH = SCRIPTS_DIR / "script_inventory.json"
REGISTRY_PATH = ROOT / "artifacts" / "tool_use_benchmark" / "benchmark_tasks.json"
SCRIPT_REF_RE = re.compile(r"parttoolgen/scripts/([A-Za-z0-9_./-]+\.(?:py|sh))")
SCRIPT_CALL_RE = re.compile(r"_script\([\"']([^\"']+)[\"']\)")


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _script_key(path: Path) -> str:
    return path.relative_to(SCRIPTS_DIR).as_posix()


def _script_ref_key(value: str) -> str:
    prefix = "parttoolgen/scripts/"
    return value[len(prefix):] if value.startswith(prefix) else value


def test_script_inventory_covers_handoff_python_and_shell_scripts() -> None:
    inventory = _inventory()
    scripts = inventory["scripts"]
    actual = {
        _script_key(path)
        for path in SCRIPTS_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix in {".py", ".sh"}
            and "dev" not in path.relative_to(SCRIPTS_DIR).parts
        )
    }

    assert set(scripts) == actual


def test_script_inventory_entries_are_well_formed() -> None:
    inventory = _inventory()
    categories = set(inventory["policy"])

    for name, meta in inventory["scripts"].items():
        assert name.endswith((".py", ".sh"))
        assert not name.startswith("/")
        assert meta["category"] in categories
        assert isinstance(meta["stable_cli"], bool)
        assert isinstance(meta["do_not_move"], bool)
        assert isinstance(meta["description"], str)
        assert meta["description"].strip()


def test_script_inventory_public_entrypoints_are_intentional() -> None:
    scripts = _inventory()["scripts"]
    public = {
        name for name, meta in scripts.items()
        if meta["category"] == "public_entrypoint"
    }

    assert public == {
        "bootstrap_parttoolgen_env.sh",
        "parttoolgen_env.sh",
        "check_toolframe_env.py",
        "run_checks.sh",
        "run_image_grounded_task_spec.py",
        "run_p3sam_tool_ref_pipeline.py",
        "run_tool_use_benchmark.py",
    }
    assert all(scripts[name]["stable_cli"] is True for name in public)
    assert all(scripts[name]["do_not_move"] is True for name in public)
    assert all((SCRIPTS_DIR / name).stat().st_mode & 0o111 for name in public)


def test_script_inventory_keeps_debug_legacy_out_of_top_level_scripts() -> None:
    scripts = _inventory()["scripts"]
    assert all(meta["category"] != "debug_legacy" for meta in scripts.values())


def test_registry_runners_are_marked_internal_and_not_moved() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    runners = {
        str(task["runner"])
        for task in registry["tasks"].values()
        if task.get("runner")
    }

    assert runners
    for runner in runners:
        assert runner.startswith("parttoolgen/src/parttoolgen/cli/runners/")
        assert not runner.startswith("parttoolgen/scripts/")
        runner_path = ROOT / runner
        assert runner_path.exists()
        assert runner_path.parent == CLI_RUNNERS_DIR


def test_paper_release_suites_keep_headline_candidate_and_diagnostic_boundaries() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    suites = registry["suites"]

    assert set(suites["paper_release_6.4_headline"]) == {
        "brush_sweep/red_zone",
        "eraser_wipe/tabletop_wipe",
    }
    assert set(suites["legacy_directional_diagnostics"]) == {
        "brush_sweep/forward",
        "brush_sweep/left",
    }
    assert "brush_sweep/forward" not in suites["current_all_runnable"]
    assert "brush_sweep/left" not in suites["current_all_runnable"]
    assert "brush_sweep/forward" not in suites["paper_release_6.4_headline"]
    assert "brush_sweep/left" not in suites["paper_release_6.4"]
    assert "eraser_wipe/side_whiteboard_hdu_erase" not in suites["paper_release_6.4"]
    assert "screwdriver_insert/deep_socket" not in suites["paper_release_6.4"]
    assert "screwdriver_insert/deep_socket" in suites["diagnostic_stress_optional"]
    assert not any(task.endswith("_dense") for task in suites["paper_release_6.4"])


def test_script_inventory_protects_in_repo_script_references() -> None:
    scripts = _inventory()["scripts"]
    script_names = {}
    for key in scripts:
        script_names.setdefault(Path(key).name, []).append(key)
    cli_names = {}
    for path in CLI_DIR.rglob("*.py"):
        cli_names.setdefault(path.name, []).append(path.relative_to(ROOT).as_posix())
    checked_roots = [
        ROOT / "parttoolgen/scripts",
        ROOT / "artifacts/tool_use_benchmark",
    ]
    referenced: set[str] = set()
    helper_referenced: set[str] = set()
    ignored_dirs = {"__pycache__", ".pytest_cache"}

    for root in checked_roots:
        for path in root.rglob("*"):
            if not path.is_file() or any(part in ignored_dirs for part in path.parts):
                continue
            if path.suffix not in {".py", ".json", ".md", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            referenced.update(SCRIPT_REF_RE.findall(text))
            helper_referenced.update(SCRIPT_CALL_RE.findall(text))

    for name in helper_referenced:
        if name in scripts:
            referenced.add(name)
            continue
        matches = script_names.get(Path(name).name, [])
        if not matches:
            cli_matches = cli_names.get(Path(name).name, [])
            assert len(cli_matches) == 1, f"{name} resolved to {cli_matches}"
            continue
        assert len(matches) == 1, f"{name} resolved to {matches}"
        referenced.add(matches[0])
    missing = referenced - set(scripts)
    assert missing == set()
    for name in referenced:
        if name.endswith((".py", ".sh")):
            assert scripts[name]["do_not_move"] is True, name
