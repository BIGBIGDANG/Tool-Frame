"""Per-trial tool-reference helpers for benchmark runners."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def add_tool_ref_mode_args(parser: Any) -> None:
    parser.add_argument(
        "--tool-refs-mode",
        choices=["static", "vlm_per_seed"],
        default="static",
        help=(
            "static uses --tool-refs-dir. vlm_per_seed regenerates task-conditioned "
            "VLM tool refs into each trial directory before compilation."
        ),
    )
    parser.add_argument("--tool-ref-model", default=None)
    parser.add_argument("--tool-ref-base-url", default=None)
    parser.add_argument("--tool-ref-max-new-tokens", type=int, default=900)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--tool-ref-enable-thinking", dest="tool_ref_enable_thinking", action="store_true", default=None)
    thinking.add_argument("--tool-ref-disable-thinking", dest="tool_ref_enable_thinking", action="store_false")


def resolve_trial_tool_refs(
    *,
    repo_root: Path,
    python: str,
    args: Any,
    trial_dir: Path,
    static_tool_refs_dir: Path,
    tool_id: str,
    task_id: str,
    task_instruction: str,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Return static refs or generate VLM refs for this trial.

    Tool-ref regeneration intentionally uses only the tool-prior evidence and
    task language. Scene grounding remains a separate per-seed VLM stage.
    """

    if getattr(args, "tool_refs_mode", "static") != "vlm_per_seed":
        return static_tool_refs_dir

    output_dir = trial_dir / "tool_refs"
    script = repo_root / "parttoolgen/scripts/run_p3sam_tool_ref_pipeline.py"
    command = [
        python,
        str(script),
        "--tool-id",
        str(tool_id),
        "--task-id",
        str(task_id),
        "--task-instruction",
        str(task_instruction),
        "--output-dir",
        str(output_dir),
        "--selection-source",
        "vlm",
        "--max-new-tokens",
        str(getattr(args, "tool_ref_max_new_tokens", 900)),
    ]
    model = getattr(args, "tool_ref_model", None)
    if model:
        command.extend(["--model", str(model)])
    base_url = getattr(args, "tool_ref_base_url", None)
    if base_url:
        command.extend(["--base-url", str(base_url)])
    thinking = getattr(args, "tool_ref_enable_thinking", None)
    if thinking is True:
        command.append("--enable-thinking")
    elif thinking is False:
        command.append("--disable-thinking")

    output_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        subprocess.run(command, cwd=repo_root, env=env, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(command, cwd=repo_root, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    return output_dir


__all__ = ["add_tool_ref_mode_args", "resolve_trial_tool_refs"]
