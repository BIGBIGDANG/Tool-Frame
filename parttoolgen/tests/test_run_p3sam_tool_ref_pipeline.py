from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_pipeline():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_p3sam_tool_ref_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_p3sam_tool_ref_pipeline", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tool_id="flat_eraser",
        task_id="eraser_wipe",
        task_instruction="wipe printed text from the table",
        output_dir=tmp_path,
        selection_source="reference",
        model="dummy",
        max_new_tokens=64,
        base_url=None,
        part_marker="2",
        contact_point_marker="F3",
        motion_axis_marker="F1",
        candidate_contact_point_markers=["F2", "F3"],
        candidate_motion_axis_markers=["F1", "F4"],
    )


def test_p3sam_pipeline_defaults_use_portable_env_paths(tmp_path, monkeypatch) -> None:
    p3sam_root = tmp_path / "models/P3-SAM"
    ckpt_path = tmp_path / "weights/p3sam.safetensors"
    monkeypatch.setenv("PARTTOOLGEN_P3SAM_ROOT", str(p3sam_root))
    monkeypatch.setenv("PARTTOOLGEN_P3SAM_CKPT", str(ckpt_path))

    pipeline = _load_pipeline()

    assert pipeline.DEFAULT_P3SAM_ROOT == p3sam_root
    assert pipeline.DEFAULT_P3SAM_CKPT == ckpt_path


def test_p3sam_pipeline_selector_args_are_stage_scoped(tmp_path) -> None:
    pipeline = _load_pipeline()
    args = _selector_args(tmp_path)

    part_command = pipeline._select_command(
        args,
        tmp_path / "prior",
        tmp_path / "part_visual_evidence.json",
        stage="part",
    )
    assert "--part-marker" in part_command
    assert "--contact-point-marker" not in part_command
    assert "--motion-axis-marker" not in part_command
    assert "--candidate-contact-point-markers" not in part_command
    assert "--candidate-motion-axis-markers" not in part_command

    contact_command = pipeline._select_command(
        args,
        tmp_path / "prior",
        tmp_path / "contact_visual_evidence.json",
        stage="contact",
        part_selection=tmp_path / "part_selection.json",
    )
    assert "--part-selection" in contact_command
    assert "--contact-point-marker" in contact_command
    assert "--motion-axis-marker" in contact_command
    assert "--candidate-contact-point-markers" in contact_command
    assert "--candidate-motion-axis-markers" in contact_command


def test_p3sam_pipeline_resolves_internal_tool_prior_cli() -> None:
    pipeline = _load_pipeline()

    script = Path(pipeline._script("select_p3sam_tool_refs.py"))

    assert script.exists()
    assert script.relative_to(pipeline.ROOT).as_posix() == (
        "parttoolgen/src/parttoolgen/cli/tool_prior/select_p3sam_tool_refs.py"
    )


def test_p3sam_pipeline_env_uses_project_core_python(tmp_path, monkeypatch) -> None:
    pipeline = _load_pipeline()
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_TOOLFRAME_ENV_CACHE", None)
    core_python = tmp_path / "core_py38"
    env_script = tmp_path / "parttoolgen/scripts/parttoolgen_env.sh"
    env_script.parent.mkdir(parents=True)
    env_script.write_text(f"export PARTTOOLGEN_CORE_PYTHON='{core_python}'\n", encoding="utf-8")

    env = pipeline._pipeline_env()

    assert pipeline._pipeline_python(env) == str(core_python)


def test_p3sam_pipeline_summary_commands_are_portable(tmp_path, monkeypatch) -> None:
    pipeline = _load_pipeline()
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    script = tmp_path / "parttoolgen/src/parttoolgen/cli/tool_prior/select_p3sam_tool_refs.py"

    command = [
        str(tmp_path / ".venv/bin/python"),
        str(script),
        "--output-dir",
        "configs/tool_refs/flat_eraser",
    ]

    assert pipeline._portable_command(command) == [
        "${PARTTOOLGEN_CORE_PYTHON:-python}",
        "parttoolgen/src/parttoolgen/cli/tool_prior/select_p3sam_tool_refs.py",
        "--output-dir",
        "configs/tool_refs/flat_eraser",
    ]
