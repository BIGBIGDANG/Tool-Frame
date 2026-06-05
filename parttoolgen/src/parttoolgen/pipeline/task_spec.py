"""Unified task-spec planning for image-grounded ToolFrame runs.

The first version intentionally keeps two execution modes:

``registry_runner``
    Compatibility mode for current benchmark configs. It builds the same
    command shape as the benchmark registry launcher, so existing task-family
    runners keep their reviewed behavior.

``command_stages``
    Spec-driven stage mode for new scenes and experiments that can be expressed
    as a list of script/module commands without adding another family runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


RUNNER_ARG_EXCLUDE = {
    "output_root",
    "scratch_output_policy",
    "source_task_status",
    "tool_refs_policy",
}
ROOT_REL_PATH_ARG_KEYS = {
    "review_file",
    "script",
    "source_mesh",
    "source_task",
    "tool_mesh",
    "tool_refs_dir",
    "tool_start_pose_source",
}
SIMTOOLREAL_ASSET_ARG_SUFFIXES = ("_asset",)
SIMTOOLREAL_ASSET_KEYS = {"asset"}


def kebab(name: str) -> str:
    return name.replace("_", "-")


def append_cli_arg(command: list[str], key: str, value: Any) -> None:
    if value is None or key in RUNNER_ARG_EXCLUDE:
        return
    flag = f"--{kebab(key)}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, list):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def resolve_runner_arg_path(root: str | Path, key: str, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    root = Path(root)
    raw = Path(value)
    if raw.is_absolute():
        return raw
    if key in ROOT_REL_PATH_ARG_KEYS:
        return root / raw
    if any(key.endswith(suffix) for suffix in SIMTOOLREAL_ASSET_ARG_SUFFIXES):
        return root / "simtoolreal" / "assets" / raw
    if key in SIMTOOLREAL_ASSET_KEYS:
        if value.startswith("urdf/"):
            return root / "simtoolreal" / "assets" / raw
        if value.startswith(("artifacts/", "configs/", "parttoolgen/", "simtoolreal/")):
            return root / raw
    return None


def iter_config_path_values(root: str | Path, value: Any, *, key: str = "", label: str = "config"):
    if isinstance(value, dict):
        for child_key, child_value in sorted(value.items()):
            child_label = f"{label}.{child_key}" if label else child_key
            yield from iter_config_path_values(
                root,
                child_value,
                key=child_key,
                label=child_label,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_label = f"{label}[{index}]"
            yield from iter_config_path_values(root, item, key=key, label=child_label)
        return
    path = resolve_runner_arg_path(root, key, value)
    if path is not None:
        yield label, path


def normalize_task_spec(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    registry_task: dict[str, Any] | None = None,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Normalize a benchmark config or a native task spec to one shape."""

    if config.get("schema_version") == "toolframe_task_spec_v1":
        spec = dict(config)
        if config_path is not None:
            spec.setdefault("config_path", str(config_path))
        return spec

    registry_task = registry_task or {}
    family = (
        registry_task.get("task_family")
        or config.get("benchmark")
        or config.get("task_family")
        or "unknown"
    )
    split = registry_task.get("split") or config.get("variant") or config.get("split") or "default"
    runner = registry_task.get("runner") or config.get("runner")
    runner_args = dict(config.get("runner_args") or {})
    return {
        "schema_version": "toolframe_task_spec_v1",
        "task_id": task_name or f"{family}/{split}",
        "task_family": family,
        "split": split,
        "status": registry_task.get("status") or config.get("runner_status") or config.get("manual_review_status"),
        "runnable": bool(registry_task.get("runnable", bool(runner))),
        "blocked_reason": registry_task.get("blocked_reason"),
        "instruction": config.get("instruction") or config.get("description") or "",
        "config_path": str(config_path) if config_path is not None else None,
        "runner": runner,
        "runner_args": runner_args,
        "pipeline": {"mode": "registry_runner"},
        "scene": config.get("scene_initialization") or {},
        "tool": {
            "tool_refs_dir": runner_args.get("tool_refs_dir"),
            "tool_mesh": (config.get("scene_initialization") or {}).get("tool_mesh"),
        },
        "mainline_params": config.get("mainline_params")
        or config.get("mainline_params_pending_review")
        or {},
        "acceptance": config.get("acceptance") or {},
        "contract_status": config.get("contract_status") or {},
    }


def default_output_root(
    *,
    registry: dict[str, Any] | None,
    spec: dict[str, Any],
    run_id: str,
) -> str:
    template = str(
        (registry or {}).get("default_output_root_template")
        or "artifacts/image_grounded/{task_family}/{split}_{run_id}"
    )
    return template.format(
        task_family=spec.get("task_family") or "unknown",
        split=spec.get("split") or "default",
        run_id=run_id,
    )


def output_root_for_spec(
    *,
    registry: dict[str, Any] | None,
    spec: dict[str, Any],
    run_id: str,
    output_root: str | None = None,
    output_prefix: str | None = None,
) -> str:
    if output_root:
        return output_root
    if output_prefix:
        return str(
            Path(output_prefix)
            / str(spec.get("task_family") or "unknown")
            / f"{spec.get('split') or 'default'}_{run_id}"
        )
    return default_output_root(registry=registry, spec=spec, run_id=run_id)


def apply_runner_overrides(
    runner_args: dict[str, Any],
    *,
    trials: int | None = None,
    seed_start: int | None = None,
    device: str | None = None,
    capture_video_every: int | None = None,
    resume: bool = False,
    skip_rollout: bool = False,
) -> dict[str, Any]:
    args = dict(runner_args)
    if trials is not None:
        args["trials"] = trials
    if seed_start is not None:
        args["seed_start"] = seed_start
    if device is not None:
        args["device"] = device
    if capture_video_every is not None:
        args["capture_video_every"] = capture_video_every
    if resume:
        args["resume"] = True
    if skip_rollout:
        args["skip_rollout"] = True
    return args


def runner_command(
    *,
    root: str | Path,
    spec: dict[str, Any],
    python: str,
    output_root: str,
    runner_args: dict[str, Any],
) -> list[str]:
    root = Path(root)
    runner = spec.get("runner")
    if not isinstance(runner, str) or not runner:
        raise ValueError("registry_runner spec requires `runner`")
    config_path = spec.get("config_path")
    command = [
        python,
        str(root / runner),
    ]
    if config_path:
        command.extend(["--benchmark-config", str(root / str(config_path)) if not Path(str(config_path)).is_absolute() else str(config_path)])
    command.extend(["--output-root", output_root])
    for key, value in runner_args.items():
        if key == "output_root":
            continue
        append_cli_arg(command, key, value)
    return command


def _format_value(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [_format_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, variables) for key, item in value.items()}
    return value


def stage_command(
    *,
    root: str | Path,
    stage: dict[str, Any],
    python: str,
    output_root: str,
    variables: dict[str, str] | None = None,
) -> list[str]:
    root = Path(root)
    variables = {
        "root": str(root),
        "output_root": output_root,
        "python": python,
        **(variables or {}),
    }
    if "command" in stage:
        raw = _format_value(stage["command"], variables)
        if not isinstance(raw, list):
            raise ValueError("stage `command` must be a list")
        return [str(item) for item in raw]
    script = stage.get("script")
    module = stage.get("module")
    if script:
        script_path = Path(str(script))
        script_text = script_path.as_posix()
        if script_path.is_absolute() or not script_text.startswith("parttoolgen/src/parttoolgen/cli/"):
            raise ValueError(
                "stage `script` must be a repo-relative internal CLI path under "
                "parttoolgen/src/parttoolgen/cli/; prefer `module` for new specs"
            )
        command = [python, str(root / str(script))]
    elif module:
        command = [python, "-m", str(module)]
    else:
        raise ValueError("command_stages stage requires `script`, `module`, or `command`")
    args = _format_value(stage.get("args") or {}, variables)
    if isinstance(args, dict):
        for key, value in args.items():
            append_cli_arg(command, key, value)
    elif isinstance(args, list):
        command.extend(str(item) for item in args)
    else:
        raise ValueError("stage `args` must be an object or list")
    return command


def registry_launcher_command(
    *,
    root: str | Path,
    spec: dict[str, Any],
    python: str,
    run_id: str,
    output_root: str | None,
    output_prefix: str | None,
    overrides: argparse.Namespace | None = None,
) -> list[str]:
    root = Path(root)
    command = [
        python,
        str(root / "parttoolgen/scripts/run_tool_use_benchmark.py"),
        "--registry",
        str(root / str(spec.get("registry_path") or "artifacts/tool_use_benchmark/benchmark_tasks.json")),
        "--run-id",
        run_id,
    ]
    task = spec.get("registry_task")
    suites = spec.get("registry_suites") or []
    if task:
        command.extend(["--task", str(task)])
    for suite in suites:
        command.extend(["--suite", str(suite)])
    if output_root:
        command.extend(["--output-root", output_root])
    if output_prefix:
        command.extend(["--output-prefix", output_prefix])
    for key in (
        "trials",
        "seed_start",
        "device",
        "capture_video_every",
        "aggregate",
        "aggregate_output_root",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            append_cli_arg(command, key, value)
    for key in (
        "resume",
        "skip_rollout",
        "allow_draft_smoke",
        "continue_on_task_error",
        "preflight_only",
        "dry_run",
    ):
        if bool(getattr(overrides, key, False)):
            append_cli_arg(command, key, True)
    return command


def preflight_spec(root: str | Path, spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
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

    config_path = spec.get("config_path")
    add("task_spec_or_config", root / str(config_path) if config_path else None, required=bool(config_path))
    mode = (spec.get("pipeline") or {}).get("mode") or "registry_runner"
    if mode == "registry_launcher":
        add(
            "registry",
            root / str(spec.get("registry_path") or "artifacts/tool_use_benchmark/benchmark_tasks.json"),
        )
        config_path = spec.get("config_path")
        if config_path:
            add("benchmark_config", root / str(config_path))
        runner = spec.get("runner")
        if runner:
            add("runner", root / str(runner))
    elif mode == "registry_runner":
        runner = spec.get("runner")
        add("runner", root / str(runner) if runner else None)
        for label, path in iter_config_path_values(root, spec.get("runner_args") or {}):
            add(f"runner_args.{label}", path)
    elif mode == "command_stages":
        for index, stage in enumerate((spec.get("pipeline") or {}).get("stages") or []):
            script = stage.get("script")
            if script:
                add(f"pipeline.stages[{index}].script", root / str(script))
    else:
        checks.append(
            {
                "name": "pipeline.mode",
                "path": None,
                "required": True,
                "exists": None,
                "passed": False,
                "detail": f"unsupported mode: {mode}",
            }
        )
    return {
        "schema_version": "toolframe_task_spec_preflight_v1",
        "task_id": spec.get("task_id"),
        "mode": mode,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
    }


def build_plan(
    *,
    root: str | Path,
    spec: dict[str, Any],
    python: str,
    run_id: str,
    registry: dict[str, Any] | None = None,
    output_root: str | None = None,
    output_prefix: str | None = None,
    overrides: argparse.Namespace | None = None,
) -> dict[str, Any]:
    root = Path(root)
    output = output_root_for_spec(
        registry=registry,
        spec=spec,
        run_id=run_id,
        output_root=output_root,
        output_prefix=output_prefix,
    )
    mode = (spec.get("pipeline") or {}).get("mode") or "registry_runner"
    command: list[str] | None = None
    commands: list[dict[str, Any]] = []
    command_error: str | None = None
    try:
        if mode == "registry_launcher":
            command = registry_launcher_command(
                root=root,
                spec=spec,
                python=python,
                run_id=run_id,
                output_root=output_root,
                output_prefix=output_prefix,
                overrides=overrides,
            )
        elif mode == "registry_runner":
            runner_args = apply_runner_overrides(
                spec.get("runner_args") or {},
                trials=getattr(overrides, "trials", None),
                seed_start=getattr(overrides, "seed_start", None),
                device=getattr(overrides, "device", None),
                capture_video_every=getattr(overrides, "capture_video_every", None),
                resume=bool(getattr(overrides, "resume", False)),
                skip_rollout=bool(getattr(overrides, "skip_rollout", False)),
            )
            command = runner_command(
                root=root,
                spec=spec,
                python=python,
                output_root=output,
                runner_args=runner_args,
            )
        elif mode == "command_stages":
            for index, stage in enumerate((spec.get("pipeline") or {}).get("stages") or []):
                commands.append(
                    {
                        "index": index,
                        "name": stage.get("name") or f"stage_{index:02d}",
                        "allow_failure": bool(stage.get("allow_failure", False)),
                        "command": stage_command(
                            root=root,
                            stage=stage,
                            python=python,
                            output_root=output,
                        ),
                    }
                )
        else:
            raise ValueError(f"unsupported pipeline mode: {mode}")
    except Exception as exc:  # noqa: BLE001 - report plan failures as data.
        command_error = str(exc)
    return {
        "schema_version": "toolframe_task_spec_plan_v1",
        "run_id": run_id,
        "task_id": spec.get("task_id"),
        "task_family": spec.get("task_family"),
        "split": spec.get("split"),
        "status": spec.get("status"),
        "runnable": bool(spec.get("runnable", True)),
        "mode": mode,
        "output_root": output,
        "config_path": spec.get("config_path"),
        "runner": spec.get("runner"),
        "preflight": preflight_spec(root, spec),
        "command": command,
        "commands": commands,
        "command_error": command_error,
    }
