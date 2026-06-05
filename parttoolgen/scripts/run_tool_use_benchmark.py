#!/usr/bin/env python3
"""Unified ToolFrame benchmark launcher.

This wrapper keeps task selection, config paths, and output roots consistent
across benchmark families. Add new benchmark tasks to
`artifacts/tool_use_benchmark/benchmark_tasks.json`, then run them by task name.
"""

from __future__ import annotations

import argparse
import csv
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
    core_python as _pipeline_core_python,
    load_json as _pipeline_load_json,
    subprocess_env as _pipeline_subprocess_env,
    toolframe_env as _pipeline_toolframe_env,
    write_json as _pipeline_write_json,
)
from parttoolgen.pipeline.task_spec import (  # noqa: E402
    build_plan as _pipeline_build_plan,
    iter_config_path_values as _pipeline_iter_config_path_values,
    normalize_task_spec as _pipeline_normalize_task_spec,
)

DEFAULT_REGISTRY = ROOT / "artifacts/tool_use_benchmark/benchmark_tasks.json"
_TOOLFRAME_ENV_CACHE: dict[str, str] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return _pipeline_load_json(path)


def _write_json(path: Path, data: Any) -> None:
    _pipeline_write_json(path, data)


def _default_run_id(registry: dict[str, Any]) -> str:
    fmt = str(registry.get("default_run_id_format") or "%Y%m%d_%H%M%S")
    return datetime.now().strftime(fmt)


def _default_output_root(registry: dict[str, Any], task: dict[str, Any], run_id: str) -> str:
    template = str(
        registry.get("default_output_root_template")
        or "artifacts/image_grounded/{task_family}/{split}_{run_id}"
    )
    return template.format(
        task_family=task["task_family"],
        split=task["split"],
        run_id=run_id,
    )


def _output_root_for_task(
    *,
    registry: dict[str, Any],
    task: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> str:
    output_root = (
        args.output_root
        if args.output_root and len(args.task or []) == 1 and not args.suite
        else _default_output_root(registry, task, run_id)
    )
    if args.output_prefix:
        output_root = str(Path(args.output_prefix) / task["task_family"] / f"{task['split']}_{run_id}")
    return output_root


def _aggregate_root_for_args(args: argparse.Namespace, run_id: str) -> Path:
    if args.aggregate_output_root:
        return Path(args.aggregate_output_root)
    if args.output_root and len(args.task or []) == 1 and not args.suite:
        return Path(args.output_root)
    if args.output_prefix:
        return Path(args.output_prefix) / "benchmark_runs" / run_id
    return ROOT / "artifacts" / "image_grounded" / "benchmark_runs" / run_id


def _auto_cuda_device() -> str:
    """Pick the least occupied CUDA device for smoke runs."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "cuda:0"

    candidates: list[tuple[int, int, int]] = []
    for raw_line in result.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) < 3:
            continue
        try:
            index = int(fields[0])
            memory_used_mib = int(fields[1])
            utilization_pct = int(fields[2])
        except ValueError:
            continue
        candidates.append((memory_used_mib, utilization_pct, index))
    if not candidates:
        return "cuda:0"
    _, _, index = min(candidates)
    return f"cuda:{index}"


def _resolve_device_arg(args: argparse.Namespace) -> str | None:
    device = args.device
    if device in (None, ""):
        return None
    if str(device).lower() in {"auto", "cuda:auto"}:
        resolved = _auto_cuda_device()
        args.requested_device = device
        args.device = resolved
        return resolved
    args.requested_device = device
    return device


def _toolframe_env() -> dict[str, str]:
    global _TOOLFRAME_ENV_CACHE
    if _TOOLFRAME_ENV_CACHE is not None:
        return dict(_TOOLFRAME_ENV_CACHE)
    _TOOLFRAME_ENV_CACHE = _pipeline_toolframe_env(ROOT)
    return dict(_TOOLFRAME_ENV_CACHE)


def _core_python(env: dict[str, str] | None = None) -> str:
    return _pipeline_core_python(ROOT, env if env is not None else _toolframe_env())


def _task_names_for_args(registry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    tasks: list[str] = []
    if args.suite:
        suites = registry.get("suites", {})
        for suite in args.suite:
            if suite not in suites:
                raise SystemExit(f"Unknown suite: {suite}")
            if not suites[suite]:
                raise SystemExit(f"Suite {suite} has no registry-runnable tasks.")
            tasks.extend(str(item) for item in suites[suite])
    tasks.extend(args.task or [])
    if not tasks:
        raise SystemExit("Provide --task or --suite.")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in tasks:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _iter_config_path_values(value: Any, *, key: str = "", label: str = "config"):
    yield from _pipeline_iter_config_path_values(ROOT, value, key=key, label=label)


def _validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def add(level: str, code: str, message: str, *, task: str | None = None) -> None:
        issues.append(
            {
                "level": level,
                "code": code,
                "task": task,
                "message": message,
            }
        )

    tasks = registry.get("tasks")
    suites = registry.get("suites", {})
    if not isinstance(tasks, dict):
        add("error", "tasks_not_object", "registry field `tasks` must be an object")
        tasks = {}
    if not isinstance(suites, dict):
        add("error", "suites_not_object", "registry field `suites` must be an object")
        suites = {}

    for suite_name, members in sorted(suites.items()):
        if not isinstance(members, list):
            add("error", "suite_not_list", f"suite `{suite_name}` must be a list")
            continue
        for member in members:
            if member not in tasks:
                add(
                    "error",
                    "suite_unknown_task",
                    f"suite `{suite_name}` references unknown task `{member}`",
                )

    required_fields = ("task_family", "split", "status", "config", "runner", "runnable")
    for task_name, task in sorted(tasks.items()):
        if not isinstance(task, dict):
            add("error", "task_not_object", "task entry must be an object", task=task_name)
            continue
        for field in required_fields:
            if field not in task:
                add("error", "missing_task_field", f"missing task field `{field}`", task=task_name)
        family = task.get("task_family")
        split = task.get("split")
        if isinstance(family, str) and isinstance(split, str):
            expected_name = f"{family}/{split}"
            if expected_name != task_name:
                add(
                    "warning",
                    "task_name_mismatch",
                    f"task key should match task_family/split `{expected_name}`",
                    task=task_name,
                )
        if not isinstance(task.get("runnable"), bool):
            add("error", "runnable_not_bool", "`runnable` must be true or false", task=task_name)
        if task.get("runnable") is False and not task.get("blocked_reason"):
            add(
                "warning",
                "blocked_reason_missing",
                "non-runnable task should explain `blocked_reason`",
                task=task_name,
            )

        runner = task.get("runner")
        if isinstance(runner, str):
            if not (ROOT / runner).exists():
                add("error", "runner_missing", f"runner does not exist: {runner}", task=task_name)

        config_path: Path | None = None
        config = None
        config_rel = task.get("config")
        if isinstance(config_rel, str):
            config_path = ROOT / config_rel
            if not config_path.exists():
                add("error", "config_missing", f"config does not exist: {config_rel}", task=task_name)
            else:
                try:
                    config = _load_json(config_path)
                except Exception as exc:
                    add(
                        "error",
                        "config_invalid_json",
                        f"could not parse config JSON: {exc}",
                        task=task_name,
                    )
        if isinstance(config, dict):
            config_runner = config.get("runner")
            if isinstance(config_runner, str) and isinstance(runner, str) and config_runner != runner:
                add(
                    "error",
                    "runner_mismatch",
                    f"registry runner `{runner}` differs from config runner `{config_runner}`",
                    task=task_name,
                )
            if not config.get("schema_version"):
                add("warning", "config_schema_missing", "config lacks `schema_version`", task=task_name)
            if not isinstance(config.get("runner_args"), dict):
                add("error", "runner_args_not_object", "config `runner_args` must be an object", task=task_name)
            if not config.get("contract_status"):
                add("warning", "contract_status_missing", "config lacks `contract_status`", task=task_name)
            tool_refs_dir = (config.get("runner_args") or {}).get("tool_refs_dir")
            if (
                task.get("runnable") is True
                and isinstance(tool_refs_dir, str)
                and tool_refs_dir.startswith("artifacts/image_grounded/")
            ):
                add(
                    "warning",
                    "runtime_tool_refs_dir",
                    (
                        "runnable task uses generated artifacts/image_grounded tool refs; "
                        "publishable configs should prefer configs/tool_refs or a tracked manifest"
                    ),
                    task=task_name,
                )
        elif config_path is not None and config_path.exists():
            add("error", "config_not_object", "config JSON root must be an object", task=task_name)

    error_count = sum(1 for issue in issues if issue["level"] == "error")
    warning_count = sum(1 for issue in issues if issue["level"] == "warning")
    return {
        "schema_version": "toolframe_benchmark_registry_validation_v1",
        "task_count": len(tasks),
        "suite_count": len(suites),
        "error_count": error_count,
        "warning_count": warning_count,
        "passed": error_count == 0,
        "issues": issues,
    }


def _preflight_task(
    *,
    task_name: str,
    task: dict[str, Any],
    config_path: Path,
    config: dict[str, Any] | None,
    config_error: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, path: Path | None, *, required: bool = True) -> None:
        exists = bool(path is not None and path.exists())
        checks.append(
            {
                "name": name,
                "path": str(path) if path is not None else None,
                "required": bool(required),
                "exists": exists,
                "passed": bool(exists or not required),
            }
        )

    def add_status(name: str, passed: bool, *, detail: str | None = None) -> None:
        checks.append(
            {
                "name": name,
                "path": None,
                "required": True,
                "exists": None,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    add("benchmark_config", config_path)
    if config_error:
        add_status("benchmark_config_json", False, detail=config_error)
    runner = task.get("runner")
    add("runner", ROOT / str(runner) if runner else None)
    if isinstance(config, dict):
        config_runner = config.get("runner")
        if isinstance(config_runner, str) and runner and config_runner != runner:
            add_status(
                "registry_config_runner_match",
                False,
                detail=f"registry={runner} config={config_runner}",
            )
        for label, path in _iter_config_path_values(config):
            add(label, path)
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "task": task_name,
        "runnable": bool(task.get("runnable")),
        "status": task.get("status"),
        "passed": passed,
        "checks": checks,
    }


def _build_command(
    *,
    registry: dict[str, Any],
    task_name: str,
    task: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    env: dict[str, str] | None = None,
) -> list[str]:
    runner = task.get("runner")
    if (not task.get("runnable") and not args.allow_draft_smoke) or not runner:
        reason = task.get("blocked_reason") or "task is not marked runnable"
        raise SystemExit(f"{task_name} is not runnable through this launcher: {reason}")

    config_path = ROOT / str(task["config"])
    config = _load_json(config_path)
    try:
        config_ref = config_path.relative_to(ROOT)
    except ValueError:
        config_ref = config_path
    spec = _pipeline_normalize_task_spec(
        config,
        config_path=config_ref,
        registry_task=task,
        task_name=task_name,
    )
    plan = _pipeline_build_plan(
        root=ROOT,
        spec=spec,
        python=_core_python(env),
        run_id=run_id,
        registry=registry,
        output_root=(
            args.output_root
            if args.output_root and len(args.task or []) == 1 and not args.suite
            else None
        ),
        output_prefix=args.output_prefix,
        overrides=args,
    )
    command = plan.get("command")
    if not isinstance(command, list):
        raise SystemExit(plan.get("command_error") or f"Could not build command for {task_name}")
    return command


def _task_plan(
    *,
    registry: dict[str, Any],
    task_name: str,
    task: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    config_path = ROOT / str(task["config"])
    config: dict[str, Any] | None = None
    config_error: str | None = None
    try:
        config = _load_json(config_path)
    except Exception as exc:  # pragma: no cover - defensive for corrupt local configs
        config_error = str(exc)
    preflight = _preflight_task(
        task_name=task_name,
        task=task,
        config_path=config_path,
        config=config,
        config_error=config_error,
    )
    command: list[str] | None = None
    command_error: str | None = None
    if config is not None:
        try:
            env = _toolframe_env()
            command = _build_command(
                registry=registry,
                task_name=task_name,
                task=task,
                args=args,
                run_id=run_id,
                env=env,
            )
        except (RuntimeError, SystemExit) as exc:
            command_error = str(exc)
    return {
        "task": task_name,
        "task_family": task.get("task_family"),
        "split": task.get("split"),
        "status": task.get("status"),
        "runnable": bool(task.get("runnable")),
        "blocked_reason": task.get("blocked_reason"),
        "config": str(config_path),
        "runner": str(ROOT / str(task.get("runner"))) if task.get("runner") else None,
        "output_root": _output_root_for_task(registry=registry, task=task, args=args, run_id=run_id),
        "preflight": preflight,
        "config_error": config_error,
        "command": command,
        "command_error": command_error,
    }


def _summary_metrics(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {"summary_path": str(summary_path), "summary_exists": False}
    try:
        summary = _load_json(summary_path)
    except Exception as exc:  # pragma: no cover - defensive for corrupt runtime summaries
        return {"summary_path": str(summary_path), "summary_exists": True, "summary_error": str(exc)}
    trials = summary.get("trials") if isinstance(summary, dict) else None
    trial_count = summary.get("trial_count") if isinstance(summary, dict) else None
    if trial_count is None and isinstance(trials, list):
        trial_count = len(trials)
    pass_rate = None
    if isinstance(summary, dict):
        for key in ("pass_rate", "task_success_pass_rate", "success_rate"):
            if summary.get(key) is not None:
                pass_rate = summary.get(key)
                break
    return {
        "summary_path": str(summary_path),
        "summary_exists": True,
        "trial_count": trial_count,
        "pass_rate": pass_rate,
        "successes": summary.get("successes") if isinstance(summary, dict) else None,
        "success_metric": summary.get("success_metric") if isinstance(summary, dict) else None,
    }


def _write_aggregate(root: Path, run_id: str, records: list[dict[str, Any]], formats: set[str]) -> None:
    aggregate = {
        "schema_version": "toolframe_benchmark_launcher_summary_v1",
        "run_id": run_id,
        "task_count": len(records),
        "records": records,
    }
    if "json" in formats:
        _write_json(root / "launcher_summary.json", aggregate)
    rows = []
    for record in records:
        summary = record.get("summary") or {}
        rows.append(
            {
                "task": record.get("task"),
                "status": record.get("status"),
                "returncode": record.get("returncode"),
                "output_root": record.get("output_root"),
                "device": record.get("device"),
                "summary_path": summary.get("summary_path"),
                "trial_count": summary.get("trial_count"),
                "pass_rate": summary.get("pass_rate"),
                "success_metric": summary.get("success_metric"),
                "error": record.get("error"),
            }
        )
    if "csv" in formats:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "launcher_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task",
                    "status",
                    "returncode",
                    "output_root",
                    "device",
                    "summary_path",
                    "trial_count",
                    "pass_rate",
                    "success_metric",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
    if "md" in formats:
        lines = [
            "# ToolFrame Benchmark Launcher Summary",
            "",
            f"- run_id: `{run_id}`",
            f"- tasks: {len(records)}",
            "",
            "| task | status | returncode | device | trials | pass_rate | summary |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                "| {task} | {status} | {returncode} | {device} | {trial_count} | {pass_rate} | {summary_path} |".format(
                    **{key: "" if value is None else str(value).replace("|", "/") for key, value in row.items()}
                )
            )
        root.mkdir(parents=True, exist_ok=True)
        (root / "launcher_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_formats(value: str) -> set[str]:
    formats = {part.strip() for part in value.split(",") if part.strip()}
    unknown = formats - {"json", "csv", "md", "none"}
    if unknown:
        raise SystemExit(f"Unknown aggregate format(s): {', '.join(sorted(unknown))}")
    if "none" in formats:
        return set()
    return formats


def _subprocess_env() -> dict[str, str]:
    return _pipeline_subprocess_env(ROOT, _toolframe_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--task", action="append", help="Task name such as brush_sweep/red_zone.")
    parser.add_argument("--suite", action="append", help="Suite name from benchmark_tasks.json.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None, help="Only valid for a single --task run.")
    parser.add_argument("--output-prefix", default=None, help="Override root prefix for multi-task runs.")
    parser.add_argument("--aggregate-output-root", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--capture-video-every", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--continue-on-task-error", action="store_true")
    parser.add_argument("--allow-draft-smoke", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--validate-registry",
        action="store_true",
        help="Validate registry/config consistency, then exit unless tasks were also provided.",
    )
    parser.add_argument("--aggregate", default="json,md")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--list", action="store_true", help="List registered tasks and suites, then exit.")
    parser.add_argument("--describe", action="append", help="Print registry/config/preflight details for a task.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = _load_json(args.registry)
    run_id = args.run_id or _default_run_id(registry)
    resolved_device = _resolve_device_arg(args)
    if getattr(args, "requested_device", None) != resolved_device and resolved_device is not None:
        print(f"[toolframe] resolved device: {args.requested_device} -> {resolved_device}")
    tasks = registry.get("tasks", {})
    if args.validate_registry:
        validation = _validate_registry(registry)
        print(json.dumps(validation, indent=2))
        if not validation["passed"]:
            raise SystemExit(2)
        if not (args.list or args.describe or args.task or args.suite):
            return
    if args.list:
        print(
            json.dumps(
                {
                    "schema_version": registry.get("schema_version"),
                    "suites": registry.get("suites", {}),
                    "tasks": {
                        name: {
                            "status": task.get("status"),
                            "runnable": bool(task.get("runnable")),
                            "config": task.get("config"),
                            "runner": task.get("runner"),
                            "blocked_reason": task.get("blocked_reason"),
                        }
                        for name, task in sorted(tasks.items())
                    },
                },
                indent=2,
            )
        )
        return
    if args.describe:
        descriptions = []
        for task_name in args.describe:
            if task_name not in tasks:
                raise SystemExit(f"Unknown task: {task_name}")
            descriptions.append(
                _task_plan(
                    registry=registry,
                    task_name=task_name,
                    task=tasks[task_name],
                    args=args,
                    run_id=run_id,
                )
            )
        print(json.dumps({"run_id": run_id, "tasks": descriptions}, indent=2))
        return

    aggregate_formats = _aggregate_formats(args.aggregate)
    for task_name in _task_names_for_args(registry, args):
        if task_name not in tasks:
            raise SystemExit(f"Unknown task: {task_name}")
    selected_plans = [
        _task_plan(
            registry=registry,
            task_name=task_name,
            task=tasks[task_name],
            args=args,
            run_id=run_id,
        )
        for task_name in _task_names_for_args(registry, args)
    ]
    if args.plan_json:
        _write_json(args.plan_json, {"run_id": run_id, "tasks": selected_plans})
    aggregate_records: list[dict[str, Any]] = []
    for plan in selected_plans:
        command = plan.get("command")
        if plan.get("command_error"):
            if args.dry_run or args.preflight_only:
                aggregate_records.append(
                    {
                        "task": plan["task"],
                        "status": "not_runnable",
                        "returncode": None,
                        "output_root": plan.get("output_root"),
                        "device": args.device,
                        "command": None,
                        "preflight": plan.get("preflight"),
                        "error": str(plan["command_error"]),
                        "summary": _summary_metrics(Path(str(plan.get("output_root"))) / "summary.json"),
                    }
                )
                print("\n[toolframe] task:", plan["task"])
                print("[toolframe] not runnable:", plan["command_error"])
                continue
            raise SystemExit(str(plan["command_error"]))
        if not isinstance(command, list):
            raise SystemExit(f"Could not build command for {plan['task']}")
        preflight = plan.get("preflight") or {}
        if not args.skip_preflight and not preflight.get("passed"):
            failed = [
                check for check in preflight.get("checks", [])
                if not check.get("passed")
            ]
            message = f"Preflight failed for {plan['task']}: {json.dumps(failed, indent=2)}"
            if args.continue_on_task_error:
                aggregate_records.append(
                    {
                        "task": plan["task"],
                        "status": "preflight_failed",
                        "returncode": None,
                        "output_root": plan.get("output_root"),
                        "device": args.device,
                        "error": message,
                        "summary": _summary_metrics(Path(str(plan.get("output_root"))) / "summary.json"),
                    }
                )
                print("\n[toolframe] task:", plan["task"])
                print("[toolframe] preflight failed; continuing because --continue-on-task-error was set")
                continue
            raise SystemExit(message)
        print("\n[toolframe] task:", plan["task"])
        print("[toolframe] command:", " ".join(command))
        output_root = Path(str(plan["output_root"]))
        record = {
            "task": plan["task"],
            "status": "planned" if args.dry_run or args.preflight_only else "running",
            "returncode": None,
            "output_root": str(output_root),
            "device": args.device,
            "command": command,
            "preflight": preflight,
            "summary": _summary_metrics(output_root / "summary.json"),
        }
        if args.dry_run or args.preflight_only:
            aggregate_records.append(record)
            continue
        proc = subprocess.run(command, cwd=ROOT, env=_subprocess_env())
        record["returncode"] = proc.returncode
        record["status"] = "completed" if proc.returncode == 0 else "failed"
        record["summary"] = _summary_metrics(output_root / "summary.json")
        aggregate_records.append(record)
        if proc.returncode != 0 and not args.continue_on_task_error:
            if aggregate_formats:
                _write_aggregate(_aggregate_root_for_args(args, run_id), run_id, aggregate_records, aggregate_formats)
            raise SystemExit(proc.returncode)
    if aggregate_formats:
        aggregate_root = _aggregate_root_for_args(args, run_id)
        _write_aggregate(aggregate_root, run_id, aggregate_records, aggregate_formats)
        print("[toolframe] launcher summary:", aggregate_root)


if __name__ == "__main__":
    main()
