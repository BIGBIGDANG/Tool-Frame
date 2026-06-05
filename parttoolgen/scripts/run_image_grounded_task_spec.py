#!/usr/bin/env python3
"""Run a unified ToolFrame task spec.

This is the spec-driven handoff path for new scenes/tasks. Current benchmark
configs run in compatibility mode through their registered family runner;
native `toolframe_task_spec_v1` files may also define explicit command stages
without adding another top-level task-family script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "parttoolgen/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.pipeline.runner_io import (  # noqa: E402
    core_python,
    load_json,
    run_logged,
    subprocess_env,
    toolframe_env,
    write_json,
)
from parttoolgen.pipeline.task_spec import build_plan, normalize_task_spec  # noqa: E402


DEFAULT_REGISTRY = ROOT / "artifacts/tool_use_benchmark/benchmark_tasks.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _default_run_id(registry: dict[str, Any] | None) -> str:
    fmt = str((registry or {}).get("default_run_id_format") or "%Y%m%d_%H%M%S")
    return datetime.now().strftime(fmt)


def _load_registry_task(
    *,
    registry_path: Path,
    task_name: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registry = load_json(registry_path)
    tasks = registry.get("tasks")
    if not isinstance(tasks, dict) or task_name not in tasks:
        raise SystemExit(f"Unknown registry task: {task_name}")
    task = tasks[task_name]
    if not isinstance(task, dict):
        raise SystemExit(f"Registry task is not an object: {task_name}")
    config_path = ROOT / str(task.get("config"))
    config = load_json(config_path)
    return registry, task, config_path


def _load_spec_from_args(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    provided = sum(bool(item) for item in (args.spec, args.task, args.suite))
    if provided != 1:
        raise SystemExit("Provide exactly one of --spec, --task, or --suite.")
    registry_path = args.registry if args.registry.is_absolute() else ROOT / args.registry
    if args.task:
        registry, task, config_path = _load_registry_task(
            registry_path=registry_path,
            task_name=args.task,
        )
        spec = normalize_task_spec(
            load_json(config_path),
            config_path=config_path.relative_to(ROOT),
            registry_task=task,
            task_name=args.task,
        )
        spec["pipeline"] = {"mode": "registry_launcher"}
        spec["registry_task"] = args.task
        spec["registry_path"] = _display_path(registry_path)
        return registry, spec
    if args.suite:
        registry = load_json(registry_path)
        suites = registry.get("suites")
        if not isinstance(suites, dict):
            raise SystemExit("Registry has no suites object.")
        for suite in args.suite:
            if suite not in suites:
                raise SystemExit(f"Unknown suite: {suite}")
        suite_id = "+".join(args.suite)
        spec = {
            "schema_version": "toolframe_task_spec_v1",
            "task_id": f"suite:{suite_id}",
            "task_family": "suite",
            "split": suite_id,
            "status": "registry_suite",
            "runnable": True,
            "instruction": f"Run registry suite(s): {suite_id}",
            "pipeline": {"mode": "registry_launcher"},
            "registry_suites": args.suite,
            "registry_path": _display_path(registry_path),
        }
        return registry, spec
    config_path = Path(args.spec)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    spec = normalize_task_spec(load_json(config_path), config_path=config_path.relative_to(ROOT))
    return None, spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="toolframe_task_spec_v1 or existing benchmark_config JSON.")
    parser.add_argument("--task", help="Registry task name such as brush_sweep/red_zone.")
    parser.add_argument("--suite", action="append", help="Registry suite name such as current_all_runnable.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--capture-video-every", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--allow-draft-smoke", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", default=None)
    parser.add_argument("--aggregate-output-root", default=None)
    parser.add_argument("--emit-spec", action="store_true")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--continue-on-task-error", action="store_true")
    parser.add_argument("--continue-on-stage-error", action="store_true")
    args = parser.parse_args()

    registry, spec = _load_spec_from_args(args)
    if args.emit_spec:
        print(json.dumps(spec, indent=2))
        return

    run_id = args.run_id or _default_run_id(registry)
    env = toolframe_env(ROOT)
    plan = build_plan(
        root=ROOT,
        spec=spec,
        python=core_python(ROOT, env),
        run_id=run_id,
        registry=registry,
        output_root=args.output_root,
        output_prefix=args.output_prefix,
        overrides=args,
    )
    if args.plan_json:
        write_json(args.plan_json, plan)

    print(json.dumps(plan, indent=2))
    if plan.get("command_error"):
        raise SystemExit(str(plan["command_error"]))
    if not args.skip_preflight and not (plan.get("preflight") or {}).get("passed"):
        if args.dry_run or args.preflight_only:
            return
        raise SystemExit(f"Preflight failed for {plan.get('task_id')}")
    mode = plan.get("mode")
    if args.dry_run or (args.preflight_only and mode != "registry_launcher"):
        return

    run_env = subprocess_env(ROOT, env)
    if mode in {"registry_launcher", "registry_runner"}:
        command = plan.get("command")
        if not isinstance(command, list):
            raise SystemExit(f"{mode} plan did not produce a command")
        raise SystemExit(subprocess.run(command, cwd=ROOT, env=run_env).returncode)
    if mode == "command_stages":
        output_root = Path(str(plan.get("output_root")))
        for stage in plan.get("commands") or []:
            command = stage.get("command")
            if not isinstance(command, list):
                raise SystemExit(f"stage has no command: {stage}")
            log_path = output_root / "logs" / f"{stage.get('index', 0):02d}_{stage.get('name', 'stage')}.json"
            try:
                run_logged(
                    [str(item) for item in command],
                    cwd=ROOT,
                    env=run_env,
                    log_path=log_path,
                    allow_failure=bool(stage.get("allow_failure")),
                )
            except subprocess.CalledProcessError as exc:
                if args.continue_on_stage_error:
                    continue
                raise SystemExit(exc.returncode) from exc
        return
    raise SystemExit(f"unsupported pipeline mode: {mode}")


if __name__ == "__main__":
    main()
