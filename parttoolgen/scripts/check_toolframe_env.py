#!/usr/bin/env python3
"""Check the ToolFrame 6.4 runtime wiring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


CORE_IMPORTS = ("parttoolgen", "numpy", "parttoolgen.image_grounded")
ROLLOUT_IMPORTS = ("isaacgym", "isaacgymenvs", "rl_games")


def _exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _import_names_for_mode(mode: str) -> tuple[str, ...]:
    names = list(CORE_IMPORTS)
    if mode in {"rollout", "full"}:
        names.extend(ROLLOUT_IMPORTS)
    return tuple(names)


def _import_key(name: str) -> str:
    return "image_grounded" if name == "parttoolgen.image_grounded" else name


def _safe_find_spec(name: str) -> tuple[bool, str | None]:
    try:
        return importlib.util.find_spec(name) is not None, None
    except Exception as exc:  # pragma: no cover - depends on broken caller envs
        return False, f"{type(exc).__name__}: {exc}"


def _find_imports(mode: str) -> tuple[dict[str, bool], dict[str, str]]:
    imports: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for name in _import_names_for_mode(mode):
        ok, error = _safe_find_spec(name)
        key = _import_key(name)
        imports[key] = ok
        if error is not None:
            errors[key] = error
    return imports, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["core", "rollout", "vlm", "full"],
        default="core",
        help=(
            "core checks package/test imports; rollout also checks IsaacGym, "
            "SimToolReal, and policy paths; vlm also checks credentials; full checks all."
        ),
    )
    args = parser.parse_args()
    root = Path(os.environ.get("PARTTOOLGEN_ROOT", Path.cwd())).resolve()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    imports, import_errors = _find_imports(args.mode)
    checks = {
        "mode": args.mode,
        "python": sys.executable,
        "python_version": version,
        "python38": version == "3.8",
        "parttoolgen_root": str(root),
        "parttoolgen_src": os.environ.get("PARTTOOLGEN_SRC"),
        "core_venv": os.environ.get("PARTTOOLGEN_CORE_VENV"),
        "isaacgym": os.environ.get("PARTTOOLGEN_ISAACGYM"),
        "simtoolreal_site": os.environ.get("PARTTOOLGEN_SIMTOOLREAL_SITE"),
        "pretrained_policy_root": os.environ.get("PARTTOOLGEN_PRETRAINED_POLICY_ROOT"),
        "vlm_key_present": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("VLM_API_KEY")),
        "imports": imports,
        "import_errors": import_errors,
    }
    checks["paths_exist"] = {
        "root": root.exists(),
        "src": _exists(checks["parttoolgen_src"]),
        "core_venv": _exists(checks["core_venv"]),
        "isaacgym": _exists(checks["isaacgym"]),
        "simtoolreal_site": _exists(checks["simtoolreal_site"]),
        "pretrained_policy_root": _exists(checks["pretrained_policy_root"]),
    }
    required = [
        ("paths_exist.root", checks["paths_exist"]["root"]),
        ("paths_exist.src", checks["paths_exist"]["src"]),
        ("paths_exist.core_venv", checks["paths_exist"]["core_venv"]),
        ("python38", checks["python38"]),
        ("imports.parttoolgen", checks["imports"]["parttoolgen"]),
        ("imports.numpy", checks["imports"]["numpy"]),
        ("imports.image_grounded", checks["imports"]["image_grounded"]),
    ]
    if args.mode in {"rollout", "full"}:
        required.extend(
            [
                ("paths_exist.isaacgym", checks["paths_exist"]["isaacgym"]),
                ("paths_exist.simtoolreal_site", checks["paths_exist"]["simtoolreal_site"]),
                ("paths_exist.pretrained_policy_root", checks["paths_exist"]["pretrained_policy_root"]),
                ("imports.isaacgymenvs", checks["imports"]["isaacgymenvs"]),
                ("imports.rl_games", checks["imports"]["rl_games"]),
            ]
        )
    if args.mode in {"vlm", "full"}:
        required.append(("vlm_key_present", checks["vlm_key_present"]))
    missing_required = [name for name, ok in required if not ok]
    checks["missing_required"] = missing_required
    checks["required_ok"] = not missing_required
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["required_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
