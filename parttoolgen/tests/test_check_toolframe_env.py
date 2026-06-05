from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_env_check():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_toolframe_env.py"
    spec = importlib.util.spec_from_file_location("check_toolframe_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_check_import_scope_is_mode_specific() -> None:
    env_check = _load_env_check()

    assert env_check._import_names_for_mode("core") == (
        "parttoolgen",
        "numpy",
        "parttoolgen.image_grounded",
    )
    assert "isaacgym" in env_check._import_names_for_mode("rollout")
    assert "isaacgym" in env_check._import_names_for_mode("full")
    assert "isaacgym" not in env_check._import_names_for_mode("vlm")


def test_env_check_safe_find_spec_reports_errors(monkeypatch) -> None:
    env_check = _load_env_check()

    def broken_find_spec(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(env_check.importlib.util, "find_spec", broken_find_spec)

    ok, error = env_check._safe_find_spec("parttoolgen.image_grounded")
    assert ok is False
    assert error is not None
