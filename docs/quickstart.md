# Quickstart

This quickstart starts from a fresh ToolFrame checkout. It explains the
maintained entrypoints and where outputs should go; it does not require
checking generated evidence into git.

## 1. Enter the workspace

```bash
cd ToolFrame_6.4
bin/toolframe bootstrap-env
bin/toolframe env-check
```

The bootstrap creates `.venv_parttoolgen_py38` under the repo and installs the
core Python 3.8 dependencies from the root `requirements.txt`.

Manual install follows the same paper-code convention:

```bash
python -m pip install -r requirements.txt
```

If your workstation needs local paths for IsaacGym, SimToolReal packages,
pretrained policies, or VLM credentials, put them in `.parttoolgen.local.env`.
Do not commit that file. Full rollout runs need those runtime artifacts; the
compiler and unit-test path should not depend on an external checkout.

Optional full-chain layers are split out so clean machines do not install GPU
or model packages by accident:

```bash
python -m pip install -r requirements-rollout.txt
python3.10 -m pip install --target .planner_vlm_py310_pkgs -r requirements-vlm.txt
```

Use the VLM target only for local/open planner models that need those packages;
API-backed planners usually only need credentials in `.parttoolgen.local.env`.

## 2. Pick a minimal smoke

Use the thin public wrapper. It is the handoff boundary between actual run
commands and the lower-level registry runners/stage scripts:

```bash
bin/toolframe list
bin/toolframe benchmark --task brush_sweep/red_zone --trials 1 --device cuda:0
bin/toolframe run-spec --task brush_sweep/red_zone --preflight-only
bin/toolframe preflight-headline
bin/toolframe preflight-paper
bin/toolframe preflight-all
```

These runners follow the public evidence path:

```text
RGB-D scene evidence
  -> VLM scene refs
  -> RGB-D lifting
  -> public tool refs
  -> whole-tool SE(3) compile
  -> SimToolReal fixed-goal rollout
  -> task-level success check
```

For side-board marker writing and side-board HDU erasing, use the registered
suites instead of calling marker/eraser internals directly:

```bash
bin/toolframe benchmark --suite current_side_board --trials 1 --device cuda:0
```

For paper-review evidence regeneration, use the named ToolFrame 6.4 suites:

```bash
bin/toolframe paper-headline --trials 20 --device cuda:0 --continue-on-task-error
bin/toolframe paper-release --trials 20 --device cuda:0 --continue-on-task-error
```

`paper-headline` is the currently reviewed headline subset. `paper-release`
also includes provisional candidates for review, but excludes draft side-board
smoke tasks and stress variants.

## 3. Keep outputs in the right place

Use these output roots:

```text
artifacts/image_grounded/<task_family>/   publishable benchmark evidence roots
test000/                                  local scratch and interactive review
```

Do not place runtime images, VLM JSON outputs, videos, or rollout summaries
under `artifacts/tool_use_benchmark/`. That directory is for lightweight
contracts, configs, metrics, and task matrices.

## 4. Interpret success

The headline metric is task-specific success, not raw whole-tool tracking.
Module rates for summaries are:

```text
functional_grounding_rate
scene_grounding_rate
subgoal_validity_rate
policy_execution_rate
task_success_rate
```

See `artifacts/tool_use_benchmark/evaluation_metrics.md` for definitions and
failure attribution.

## 5. Before publishing changes

For source/config edits, run checks only after the relevant task contract or
code change is ready:

```bash
git diff --check
bin/toolframe check --quick
```

Run Isaac/rollout smokes only when validating a specific benchmark contract,
environment asset, or runner change.
