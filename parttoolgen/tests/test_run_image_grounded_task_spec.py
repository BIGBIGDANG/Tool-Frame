from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "parttoolgen/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.pipeline.task_spec import build_plan, normalize_task_spec  # noqa: E402
from parttoolgen.pipeline.runner_io import load_json, write_json  # noqa: E402


def _args(**overrides):
    data = {
        "trials": 1,
        "seed_start": None,
        "device": None,
        "capture_video_every": None,
        "resume": False,
        "skip_rollout": True,
        "allow_draft_smoke": False,
        "continue_on_task_error": False,
        "preflight_only": False,
        "dry_run": False,
        "aggregate": None,
        "aggregate_output_root": None,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_run_spec_registry_task_delegates_to_registry_launcher(tmp_path) -> None:
    runner = tmp_path / "parttoolgen/src/parttoolgen/cli/runners/family_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("print('runner')\n", encoding="utf-8")
    config = tmp_path / "configs/demo.json"
    write_json(
        config,
        {
            "schema_version": "demo_config_v1",
            "benchmark": "demo",
            "variant": "one",
            "runner": "parttoolgen/src/parttoolgen/cli/runners/family_runner.py",
            "runner_args": {"trials": 5},
            "contract_status": {"status": "reviewed"},
        },
    )
    registry = {
        "default_output_root_template": "runs/{task_family}/{split}_{run_id}",
        "tasks": {
            "demo/one": {
                "task_family": "demo",
                "split": "one",
                "status": "fixed/current",
                "runnable": True,
                "config": "configs/demo.json",
                "runner": "parttoolgen/src/parttoolgen/cli/runners/family_runner.py",
            }
        },
    }
    spec = normalize_task_spec(
        config=load_json(config),
        config_path="configs/demo.json",
        registry_task=registry["tasks"]["demo/one"],
        task_name="demo/one",
    )
    spec["pipeline"] = {"mode": "registry_launcher"}
    spec["registry_task"] = "demo/one"
    spec["registry_path"] = "registry.json"
    write_json(tmp_path / "registry.json", registry)

    plan = build_plan(
        root=tmp_path,
        spec=spec,
        python="/python",
        run_id="run001",
        registry=registry,
        overrides=_args(preflight_only=True),
    )

    assert plan["preflight"]["passed"] is True
    assert plan["command"][1].endswith("parttoolgen/scripts/run_tool_use_benchmark.py")
    assert str(runner) not in plan["command"]
    assert "--task" in plan["command"]
    assert "demo/one" in plan["command"]
    assert "--preflight-only" in plan["command"]


def test_run_spec_compat_runner_filters_metadata_args(tmp_path) -> None:
    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')\n", encoding="utf-8")
    spec = {
        "schema_version": "toolframe_task_spec_v1",
        "task_id": "demo/one",
        "task_family": "demo",
        "split": "one",
        "config_path": "config.json",
        "runner": "runner.py",
        "runner_args": {
            "trials": 5,
            "source_task_status": "metadata_only",
            "tool_refs_policy": "metadata_only",
        },
        "pipeline": {"mode": "registry_runner"},
    }
    write_json(tmp_path / "config.json", {"schema_version": "demo", "runner_args": {}})

    plan = build_plan(
        root=tmp_path,
        spec=spec,
        python="/python",
        run_id="run001",
        overrides=_args(trials=2),
    )

    assert plan["preflight"]["passed"] is True
    assert "--trials" in plan["command"]
    assert "2" in plan["command"]
    assert "--source-task-status" not in plan["command"]
    assert "--tool-refs-policy" not in plan["command"]


def test_run_spec_command_stages_substitute_output_root(tmp_path) -> None:
    stage_script = tmp_path / "parttoolgen/src/parttoolgen/cli/stages/stage.py"
    stage_script.parent.mkdir(parents=True)
    stage_script.write_text("print('stage')\n", encoding="utf-8")
    spec = {
        "schema_version": "toolframe_task_spec_v1",
        "task_id": "demo/staged",
        "task_family": "demo",
        "split": "staged",
        "pipeline": {
            "mode": "command_stages",
            "stages": [
                {
                    "name": "stage",
                    "script": "parttoolgen/src/parttoolgen/cli/stages/stage.py",
                    "args": {
                        "input_path": "{output_root}/input.json",
                        "enabled": True,
                        "values": [1, 2],
                    },
                }
            ],
        },
    }

    plan = build_plan(
        root=tmp_path,
        spec=spec,
        python="/python",
        run_id="run001",
        output_root="runs/demo/staged",
        overrides=_args(),
    )

    command = plan["commands"][0]["command"]
    assert plan["preflight"]["passed"] is True
    assert command[:2] == ["/python", str(stage_script)]
    assert "--input-path" in command
    assert "runs/demo/staged/input.json" in command
    assert "--enabled" in command
    assert command[-2:] == ["1", "2"]
