"""Shared IO and process helpers for ToolFrame pipeline runners."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root_from(path: str | Path) -> Path:
    """Find the ToolFrame repository root from a script or module path."""

    start = Path(path).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "parttoolgen" / "src").is_dir() and (
            candidate / "artifacts" / "tool_use_benchmark"
        ).is_dir():
            return candidate
    raise RuntimeError(f"Could not locate ToolFrame repository root from {path}")


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_env0(raw: bytes) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", errors="ignore")] = value.decode(
            "utf-8",
            errors="ignore",
        )
    return env


def toolframe_env(root: str | Path) -> dict[str, str]:
    """Return the environment produced by the project env wrapper."""

    root = Path(root)
    env_script = root / "parttoolgen/scripts/parttoolgen_env.sh"
    if not env_script.exists():
        return os.environ.copy()
    result = subprocess.run(
        ["bash", "-lc", 'source "$1" >/dev/null && env -0', "bash", str(env_script)],
        cwd=root,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"failed to source {env_script}:\n{stderr}")
    return read_env0(result.stdout)


def core_python(root: str | Path, env: dict[str, str] | None = None) -> str:
    env = env if env is not None else toolframe_env(root)
    return env.get("PARTTOOLGEN_CORE_PYTHON") or sys.executable


def subprocess_env(root: str | Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """Build the subprocess env used by unified ToolFrame runners."""

    root = Path(root)
    env = dict(env if env is not None else toolframe_env(root))
    entries = [
        str(root / "parttoolgen/src"),
        str(root / "parttoolgen"),
        str(root),
        str(root / "simtoolreal"),
        str(root / "simtoolreal/isaacgymenvs"),
        str(root / "simtoolreal/rl_games"),
    ]
    for key in ("PARTTOOLGEN_ISAACGYM", "PARTTOOLGEN_SIMTOOLREAL_SITE"):
        if env.get(key):
            entries.append(env[key])
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run_logged(
    command: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    log_path: str | Path | None = None,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command and optionally write a compact JSON process log."""

    proc = subprocess.run(
        command,
        cwd=Path(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if log_path is not None:
        write_json(
            log_path,
            {
                "command": command,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-8000:],
                "stderr_tail": proc.stderr[-8000:],
            },
        )
    if proc.returncode != 0 and not allow_failure:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    return proc

