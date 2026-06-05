from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "parttoolgen/scripts/script_inventory.json"


def test_public_python_cli_help_smoke() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    scripts = [
        ROOT / "parttoolgen/scripts" / name
        for name, meta in inventory["scripts"].items()
        if meta["category"] == "public_entrypoint" and name.endswith(".py")
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        assert "usage:" in result.stdout.lower()


def test_public_shell_cli_help_smoke() -> None:
    scripts = [
        ROOT / "parttoolgen/scripts/run_checks.sh",
    ]
    for script in scripts:
        result = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        assert "usage:" in result.stdout.lower()


def test_top_level_toolframe_wrapper_help_smoke() -> None:
    script = ROOT / "bin/toolframe"
    assert script.exists()
    assert script.stat().st_mode & 0o111
    result = subprocess.run(
        [str(script), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "usage:" in result.stdout.lower()
    assert "benchmark" in result.stdout
    assert "inventory" in result.stdout
    assert "run-spec" in result.stdout


def test_top_level_toolframe_run_spec_help_smoke() -> None:
    script = ROOT / "bin/toolframe"
    result = subprocess.run(
        [str(script), "run-spec", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "usage:" in result.stdout.lower()
    assert "--spec" in result.stdout
    assert "--task" in result.stdout
