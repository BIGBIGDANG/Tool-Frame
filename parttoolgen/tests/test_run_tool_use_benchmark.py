from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_launcher():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_tool_use_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_tool_use_benchmark", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    data = {
        "task": ["demo/one"],
        "suite": None,
        "output_root": None,
        "output_prefix": None,
        "aggregate_output_root": None,
        "trials": 1,
        "seed_start": None,
        "device": None,
        "capture_video_every": None,
        "resume": False,
        "skip_rollout": False,
        "allow_draft_smoke": False,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_benchmark_launcher_builds_default_output_and_preflight(tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_TOOLFRAME_ENV_CACHE", None)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    asset = tmp_path / "simtoolreal/assets/urdf/demo_asset.urdf"
    asset.parent.mkdir(parents=True)
    asset.write_text("", encoding="utf-8")
    config = tmp_path / "configs/demo.json"
    config.parent.mkdir()
    launcher._write_json(
        config,
        {
            "schema_version": "demo_config_v1",
            "runner": "runner.py",
            "runner_args": {
                "trials": 5,
                "block_asset": "urdf/demo_asset.urdf",
                "socket_size_xyz": [0.048, 0.048, 0.055],
            }
        },
    )
    registry = {
        "default_output_root_template": "runs/{task_family}/{split}_{run_id}",
        "tasks": {},
    }
    task = {
        "task_family": "demo",
        "split": "one",
        "status": "fixed/current",
        "runnable": True,
        "config": "configs/demo.json",
        "runner": "runner.py",
    }
    core_python = tmp_path / "core_py38"
    env_script = tmp_path / "parttoolgen/scripts/parttoolgen_env.sh"
    env_script.parent.mkdir(parents=True)
    env_script.write_text(f"export PARTTOOLGEN_CORE_PYTHON='{core_python}'\n", encoding="utf-8")

    plan = launcher._task_plan(
        registry=registry,
        task_name="demo/one",
        task=task,
        args=_args(),
        run_id="run001",
    )

    assert plan["output_root"] == "runs/demo/one_run001"
    assert plan["preflight"]["passed"] is True
    check_names = [check["name"] for check in plan["preflight"]["checks"]]
    assert not any(name.startswith("runner_arg:socket_size_xyz") for name in check_names)
    assert plan["command"][0] == str(core_python)
    assert plan["command"][2:6] == [
        "--benchmark-config",
        str(config),
        "--output-root",
        "runs/demo/one_run001",
    ]
    assert "--trials" in plan["command"]
    assert "1" in plan["command"]


def test_benchmark_launcher_aggregate_outputs(tmp_path) -> None:
    launcher = _load_launcher()
    records = [
        {
            "task": "demo/one",
            "status": "completed",
            "returncode": 0,
            "output_root": "runs/demo/one",
            "device": "cuda:1",
            "summary": {
                "summary_path": "runs/demo/one/summary.json",
                "trial_count": 2,
                "pass_rate": 0.5,
            },
        }
    ]

    launcher._write_aggregate(tmp_path, "run001", records, {"json", "csv", "md"})

    assert (tmp_path / "launcher_summary.json").exists()
    assert (tmp_path / "launcher_summary.csv").exists()
    assert "demo/one" in (tmp_path / "launcher_summary.md").read_text(encoding="utf-8")


def test_benchmark_launcher_auto_device_picks_least_occupied_gpu(monkeypatch) -> None:
    launcher = _load_launcher()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, 22000, 100\n1, 18, 0\n2, 2048, 3\n",
            stderr="",
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    args = _args(device="auto")

    assert launcher._resolve_device_arg(args) == "cuda:1"
    assert args.requested_device == "auto"
    assert args.device == "cuda:1"


def test_benchmark_launcher_registry_validation_catches_handoff_errors(tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/demo.json"
    config.parent.mkdir()
    launcher._write_json(
        config,
        {
            "schema_version": "demo_config_v1",
            "runner": "other_runner.py",
            "runner_args": {},
            "contract_status": {"status": "reviewed"},
        },
    )
    registry = {
        "suites": {"broken_suite": ["demo/one", "demo/missing"]},
        "tasks": {
            "demo/one": {
                "task_family": "demo",
                "split": "one",
                "status": "fixed/current",
                "runnable": True,
                "config": "configs/demo.json",
                "runner": "runner.py",
            },
        },
    }

    validation = launcher._validate_registry(registry)

    codes = {issue["code"] for issue in validation["issues"]}
    assert validation["passed"] is False
    assert "suite_unknown_task" in codes
    assert "runner_mismatch" in codes


def test_benchmark_launcher_preflight_reports_config_parse_errors(tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/bad.json"
    config.parent.mkdir()
    config.write_text("{bad json", encoding="utf-8")
    registry = {
        "default_output_root_template": "runs/{task_family}/{split}_{run_id}",
        "tasks": {},
    }
    task = {
        "task_family": "demo",
        "split": "bad",
        "status": "fixed/current",
        "runnable": True,
        "config": "configs/bad.json",
        "runner": "runner.py",
    }

    plan = launcher._task_plan(
        registry=registry,
        task_name="demo/bad",
        task=task,
        args=_args(),
        run_id="run001",
    )

    assert plan["preflight"]["passed"] is False
    assert plan["config_error"]
    assert any(check["name"] == "benchmark_config_json" for check in plan["preflight"]["checks"])


def test_benchmark_launcher_main_dry_run_writes_plan_and_summary(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/demo.json"
    config.parent.mkdir()
    launcher._write_json(
        config,
        {
            "schema_version": "demo_config_v1",
            "runner": "runner.py",
            "runner_args": {"trials": 5},
            "contract_status": {"status": "reviewed"},
        },
    )
    registry = tmp_path / "registry.json"
    launcher._write_json(
        registry,
        {
            "schema_version": "toolframe_benchmark_task_registry_v1",
            "default_output_root_template": "runs/{task_family}/{split}_{run_id}",
            "suites": {},
            "tasks": {
                "demo/one": {
                    "task_family": "demo",
                    "split": "one",
                    "status": "fixed/current",
                    "runnable": True,
                    "config": "configs/demo.json",
                    "runner": "runner.py",
                }
            },
        },
    )
    plan_json = tmp_path / "plan.json"
    aggregate_root = tmp_path / "aggregate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tool_use_benchmark.py",
            "--registry",
            str(registry),
            "--task",
            "demo/one",
            "--run-id",
            "run001",
            "--dry-run",
            "--aggregate",
            "json",
            "--aggregate-output-root",
            str(aggregate_root),
            "--plan-json",
            str(plan_json),
        ],
    )

    launcher.main()

    output = capsys.readouterr().out
    assert "demo/one" in output
    assert plan_json.exists()
    assert (aggregate_root / "launcher_summary.json").exists()


def test_benchmark_launcher_does_not_forward_config_metadata(tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/marker.json"
    config.parent.mkdir()
    launcher._write_json(
        config,
        {
            "schema_version": "marker_config_v1",
            "runner": "runner.py",
            "runner_args": {
                "source_task": "artifacts/image_grounded/marker/task.json",
                "source_task_status": "metadata_not_cli",
            },
            "contract_status": {"status": "draft"},
        },
    )
    registry = {"default_output_root_template": "runs/{task_family}/{split}_{run_id}"}
    task = {
        "task_family": "marker_trace",
        "split": "write_l",
        "status": "draft",
        "runnable": True,
        "config": "configs/marker.json",
        "runner": "runner.py",
    }

    command = launcher._build_command(
        registry=registry,
        task_name="marker_trace/write_l",
        task=task,
        args=_args(task=["marker_trace/write_l"]),
        run_id="run001",
        env={},
    )

    assert "--source-task" in command
    assert "--source-task-status" not in command


def test_benchmark_launcher_command_matches_task_spec_plan(tmp_path, monkeypatch) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_TOOLFRAME_ENV_CACHE", None)
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/demo.json"
    config.parent.mkdir()
    launcher._write_json(
        config,
        {
            "schema_version": "demo_config_v1",
            "benchmark": "demo",
            "variant": "one",
            "runner": "runner.py",
            "runner_args": {"trials": 5, "source_task_status": "metadata_not_cli"},
            "contract_status": {"status": "reviewed"},
        },
    )
    core_python = tmp_path / "core_py38"
    env_script = tmp_path / "parttoolgen/scripts/parttoolgen_env.sh"
    env_script.parent.mkdir(parents=True)
    env_script.write_text(f"export PARTTOOLGEN_CORE_PYTHON='{core_python}'\n", encoding="utf-8")
    registry = {
        "default_output_root_template": "runs/{task_family}/{split}_{run_id}",
        "tasks": {},
    }
    task = {
        "task_family": "demo",
        "split": "one",
        "status": "fixed/current",
        "runnable": True,
        "config": "configs/demo.json",
        "runner": "runner.py",
    }
    args = _args(task=["demo/one"], trials=2, skip_rollout=True)

    launcher_plan = launcher._task_plan(
        registry=registry,
        task_name="demo/one",
        task=task,
        args=args,
        run_id="run001",
    )

    from parttoolgen.pipeline.task_spec import build_plan, normalize_task_spec

    spec = normalize_task_spec(
        launcher._load_json(config),
        config_path="configs/demo.json",
        registry_task=task,
        task_name="demo/one",
    )
    spec_plan = build_plan(
        root=tmp_path,
        spec=spec,
        python=str(core_python),
        run_id="run001",
        registry=registry,
        overrides=args,
    )

    assert launcher_plan["output_root"] == spec_plan["output_root"]
    assert launcher_plan["command"] == spec_plan["command"]
