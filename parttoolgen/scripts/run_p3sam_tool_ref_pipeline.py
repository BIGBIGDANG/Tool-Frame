#!/usr/bin/env python3
"""Unified P3SAM tool-ref pipeline.

This is the public tool-side entry point:

P3SAM/tool_prior -> standard part overview evidence -> task-conditioned
contact point and motion-axis refs.

The lower-level scripts remain importable/debuggable, but handoff users should
start here when they need to regenerate tool refs for a new tool/task pair.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT / "artifacts/tool_prior/p3sam"
DEFAULT_P3SAM_ROOT = Path(
    os.environ.get(
        "PARTTOOLGEN_P3SAM_ROOT",
        str(ROOT / "artifacts/models/Hunyuan3D-Part/P3-SAM"),
    )
)
_TOOLFRAME_ENV_CACHE: dict[str, str] | None = None
DEFAULT_P3SAM_CKPT = Path(
    os.environ.get(
        "PARTTOOLGEN_P3SAM_CKPT",
        str(DEFAULT_P3SAM_ROOT / "weights/p3sam/p3sam.safetensors"),
    )
)


def _script(name: str) -> str:
    scripts = ROOT / "parttoolgen/src/parttoolgen/cli"
    for subdir in ("tool_prior", "stages", "aux"):
        candidate = scripts / subdir / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"internal ToolFrame CLI module not found: {name}")


def _run(command: list[str], *, dry_run: bool, env: dict[str, str]) -> None:
    print("[tool-ref-pipeline]", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _read_env0(raw: bytes) -> dict[str, str]:
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


def _source_toolframe_env() -> dict[str, str]:
    global _TOOLFRAME_ENV_CACHE
    if _TOOLFRAME_ENV_CACHE is not None:
        return dict(_TOOLFRAME_ENV_CACHE)
    env_script = ROOT / "parttoolgen/scripts/parttoolgen_env.sh"
    if not env_script.exists():
        _TOOLFRAME_ENV_CACHE = os.environ.copy()
        return dict(_TOOLFRAME_ENV_CACHE)
    result = subprocess.run(
        ["bash", "-lc", 'source "$1" >/dev/null && env -0', "bash", str(env_script)],
        cwd=ROOT,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"failed to source {env_script}:\n{stderr}")
    _TOOLFRAME_ENV_CACHE = _read_env0(result.stdout)
    return dict(_TOOLFRAME_ENV_CACHE)


def _pipeline_env() -> dict[str, str]:
    env = _source_toolframe_env()
    entries = [
        str(ROOT / "parttoolgen/src"),
        str(ROOT / "parttoolgen"),
        str(ROOT),
        str(ROOT / "simtoolreal"),
        str(ROOT / "simtoolreal/isaacgymenvs"),
        str(ROOT / "simtoolreal/rl_games"),
    ]
    for key in ("PARTTOOLGEN_ISAACGYM", "PARTTOOLGEN_SIMTOOLREAL_SITE"):
        if env.get(key):
            entries.append(env[key])
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _pipeline_python(env: dict[str, str]) -> str:
    return env.get("PARTTOOLGEN_CORE_PYTHON") or sys.executable


def _portable_command(command: list[str]) -> list[str]:
    portable: list[str] = []
    for item in command:
        if item.endswith("/bin/python"):
            portable.append("${PARTTOOLGEN_CORE_PYTHON:-python}")
            continue
        path = Path(item)
        if path.is_absolute():
            try:
                portable.append(path.relative_to(ROOT).as_posix())
                continue
            except ValueError:
                pass
        portable.append(item)
    return portable


def _build_prior_command(args: argparse.Namespace, tool_root: Path, *, python: str) -> list[str]:
    if args.source_mesh is None:
        raise SystemExit("--source-mesh is required with --build-prior")
    command = [
        python,
        _script("build_p3sam_tool_prior.py"),
        "--tool-id",
        args.tool_id,
        "--source-mesh",
        str(args.source_mesh),
        "--output-root",
        str(tool_root),
        "--p3sam-root",
        str(args.p3sam_root),
        "--ckpt-path",
        str(args.ckpt_path),
        "--python",
        str(args.p3sam_python),
        "--point-num",
        str(args.point_num),
        "--prompt-num",
        str(args.prompt_num),
        "--prompt-bs",
        str(args.prompt_bs),
        "--threshold",
        str(args.threshold),
        "--seed",
        str(args.seed),
        "--parallel",
        str(args.parallel),
        "--clean-mesh",
        str(args.clean_mesh),
        "--min-face-count",
        str(args.min_face_count),
    ]
    if args.skip_p3sam_run:
        command.append("--skip-p3sam-run")
    if args.p3sam_raw_dir is not None:
        command.extend(["--p3sam-raw-dir", str(args.p3sam_raw_dir)])
    if args.observation_id:
        command.extend(["--observation-id", args.observation_id])
    return command


def _render_command(
    args: argparse.Namespace,
    evidence_dir: Path,
    *,
    part_selection: Path | None = None,
    python: str,
) -> list[str]:
    command = [
        python,
        _script("render_sampart3d_part_role_evidence.py"),
        "--run-root",
        str(args.run_root),
        "--tool-id",
        args.tool_id,
        "--prior-subdir",
        args.prior_subdir,
        "--output-dir",
        str(evidence_dir),
        "--view-selection",
        args.view_selection,
        "--part-overview-views",
        str(args.part_overview_views),
        "--max-views",
        str(args.max_views),
    ]
    if part_selection is not None:
        command.extend(["--part-role-selection", str(part_selection)])
    return command


def _select_command(
    args: argparse.Namespace,
    prior_dir: Path,
    visual_evidence: Path,
    *,
    stage: str,
    part_selection: Path | None = None,
    python: str | None = None,
) -> list[str]:
    if not args.task_id:
        raise SystemExit("--task-id is required unless --render-only is set")
    if not args.task_instruction:
        raise SystemExit("--task-instruction is required unless --render-only is set")
    command = [
        python or sys.executable,
        _script("select_p3sam_tool_refs.py"),
        "--tool-id",
        args.tool_id,
        "--task-id",
        args.task_id,
        "--task-instruction",
        args.task_instruction,
        "--p3sam-prior-dir",
        str(prior_dir),
        "--visual-evidence",
        str(visual_evidence),
        "--output-dir",
        str(args.output_dir),
        "--stage",
        stage,
        "--selection-source",
        args.selection_source,
        "--model",
        args.model,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    enable_thinking = getattr(args, "enable_thinking", None)
    if enable_thinking is True:
        command.append("--enable-thinking")
    elif enable_thinking is False:
        command.append("--disable-thinking")
    if stage != "part":
        for context_image in getattr(args, "context_image", None) or []:
            command.extend(["--context-image", str(context_image)])
    if part_selection is not None:
        command.extend(["--part-selection", str(part_selection)])
    if stage == "part" and args.part_marker:
        command.extend(["--part-marker", args.part_marker])
    if stage != "part" and args.contact_point_marker:
        command.extend(["--contact-point-marker", args.contact_point_marker])
    if stage != "part" and args.motion_axis_marker:
        command.extend(["--motion-axis-marker", args.motion_axis_marker])
    if stage != "part" and args.candidate_contact_point_markers:
        command.append("--candidate-contact-point-markers")
        command.extend(args.candidate_contact_point_markers)
    if stage != "part" and args.candidate_motion_axis_markers:
        command.append("--candidate-motion-axis-markers")
        command.extend(args.candidate_motion_axis_markers)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-id", required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--task-instruction")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--prior-subdir", default="tool_prior")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--part-evidence-dir",
        type=Path,
        help="Defaults to <output-dir>/evidence/01_part_selection.",
    )
    parser.add_argument(
        "--contact-evidence-dir",
        type=Path,
        help="Defaults to <output-dir>/evidence/02_contact_marker_selection.",
    )
    parser.add_argument("--build-prior", action="store_true")
    parser.add_argument("--source-mesh", type=Path)
    parser.add_argument("--p3sam-root", type=Path, default=DEFAULT_P3SAM_ROOT)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=DEFAULT_P3SAM_CKPT,
    )
    parser.add_argument("--p3sam-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-p3sam-run", action="store_true")
    parser.add_argument("--p3sam-raw-dir", type=Path)
    parser.add_argument("--point-num", type=int, default=100000)
    parser.add_argument("--prompt-num", type=int, default=400)
    parser.add_argument("--prompt-bs", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--clean-mesh", type=int, default=1)
    parser.add_argument("--min-face-count", type=int, default=1)
    parser.add_argument("--observation-id")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--view-selection", choices=["all", "uniform", "first"], default="uniform")
    parser.add_argument("--part-overview-views", type=int, default=3)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--selection-source", choices=["vlm", "reference"], default="vlm")
    parser.add_argument(
        "--context-image",
        type=Path,
        action="append",
        default=[],
        help="Optional scene/workspace image passed to contact and motion-axis selection.",
    )
    parser.add_argument("--part-marker")
    parser.add_argument("--contact-point-marker")
    parser.add_argument("--candidate-contact-point-markers", nargs="*")
    parser.add_argument("--motion-axis-marker")
    parser.add_argument("--candidate-motion-axis-markers", nargs="*")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL") or "gpt-5.4")
    parser.add_argument("--base-url", default=os.environ.get("VLM_API_BASE"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Pass enable_thinking=true to OpenAI-compatible providers that support it, such as Qwen.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Pass enable_thinking=false to OpenAI-compatible providers that support it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tool_root = args.run_root / args.tool_id
    prior_dir = tool_root / args.prior_subdir
    part_evidence_dir = args.part_evidence_dir or (args.output_dir / "evidence" / "01_part_selection")
    contact_evidence_dir = args.contact_evidence_dir or (
        args.output_dir / "evidence" / "02_contact_marker_selection"
    )
    part_visual_evidence = part_evidence_dir / "visual_evidence.json"
    contact_visual_evidence = contact_evidence_dir / "visual_evidence.json"
    part_selection = args.output_dir / "part_selection.json"
    env = _pipeline_env()
    pipeline_python = _pipeline_python(env)
    if args.p3sam_python == Path(sys.executable):
        args.p3sam_python = Path(pipeline_python)
    commands: list[list[str]] = []

    if args.build_prior:
        command = _build_prior_command(args, tool_root, python=pipeline_python)
        commands.append(command)
        _run(command, dry_run=args.dry_run, env=env)

    if not args.dry_run:
        if not args.build_prior and not prior_dir.exists():
            raise FileNotFoundError(f"tool prior dir does not exist: {prior_dir}")
        if args.skip_render and not part_visual_evidence.exists():
            raise FileNotFoundError(f"part visual evidence does not exist: {part_visual_evidence}")

    if not args.skip_render:
        command = _render_command(args, part_evidence_dir, python=pipeline_python)
        commands.append(command)
        _run(command, dry_run=args.dry_run, env=env)
    if not args.render_only:
        command = _select_command(
            args,
            prior_dir,
            part_visual_evidence,
            stage="part",
            python=pipeline_python,
        )
        commands.append(command)
        _run(command, dry_run=args.dry_run, env=env)
        command = _render_command(
            args,
            contact_evidence_dir,
            part_selection=part_selection,
            python=pipeline_python,
        )
        commands.append(command)
        _run(command, dry_run=args.dry_run, env=env)
        command = _select_command(
            args,
            prior_dir,
            contact_visual_evidence,
            stage="contact",
            part_selection=part_selection,
            python=pipeline_python,
        )
        commands.append(command)
        _run(command, dry_run=args.dry_run, env=env)

    summary = {
        "schema_version": "parttoolgen_p3sam_tool_ref_pipeline_v0",
        "tool_id": args.tool_id,
        "task_id": args.task_id,
        "run_root": str(args.run_root),
        "tool_root": str(tool_root),
        "prior_dir": str(prior_dir),
        "part_visual_evidence": str(part_visual_evidence),
        "contact_visual_evidence": str(contact_visual_evidence),
        "part_selection": str(part_selection),
        "output_dir": str(args.output_dir),
        "render_only": bool(args.render_only),
        "build_prior": bool(args.build_prior),
        "commands": [_portable_command(command) for command in commands],
    }
    if not args.dry_run:
        _write_json(args.output_dir / "tool_ref_pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
