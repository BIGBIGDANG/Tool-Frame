# Unified benchmark input/output layout

Date: 2026-05-31

This document defines the stable layout used by benchmark configs and the
unified launcher. The goal is that adding a new benchmark starts from one
unified task spec, then synchronizes the compatibility config/registry records
needed by the current runners, rather than writing a new path convention or a
new family-specific top-level script.

## Source-of-truth inputs

New task and scene work should start from a `toolframe_task_spec_v1` file or a
registry task normalized through `bin/toolframe run-spec`. During the migration,
reviewed split configs remain in:

Each benchmark split owns one directory:

```text
artifacts/tool_use_benchmark/<task_family>/<split>/
```

Required files:

```text
benchmark_config.json        # or benchmark_config.draft.json while provisional
contract_draft.json          # required for new/provisional tasks
README.md                    # optional split-specific notes
```

The task must also be registered in:

```text
artifacts/tool_use_benchmark/benchmark_tasks.json
```

The registry/config files are the current compatibility layer for old runner
backends. They should be treated as generated or synchronized handoff records
once a split has a reviewed unified spec.

## Runtime outputs

The unified launcher writes outputs under:

```text
artifacts/image_grounded/<task_family>/<split>_<run_id>/
```

Expected subdirectories, when the underlying runner reaches each stage:

```text
scene_grounding/
tool_refs/
compile/
rollout/
visual_evidence/
logs/
```

Task runners may create task-specific nested seed directories, but the root
above is the common run boundary used for cleanup, summaries, and publication
review.

Scene direction is recorded as public method evidence. The VLM should use a
visible `action_direction` when the use direction is visible in the grounding
image. For point/depth actions with no reliable visible endpoint, a task may
expose two support-surface candidates, `support_surface_normal` and
`opposite_support_surface_normal`; the VLM selection is written as
`direction_decision`, and the lifter converts only that explicit selection into
`scene.direction`. The compiler must not guess a default world axis.

## Scratch outputs

Interactive previews and quick review runs stay under:

```text
test000/<purpose>_<run_id>/
```

Scratch roots must not be referenced by publishable benchmark configs.

## Unified launcher

Dry-run one task:

```bash
bin/toolframe benchmark \
  --task brush_sweep/red_zone \
  --trials 1 \
  --dry-run
```

Run one task:

```bash
bin/toolframe benchmark \
  --task brush_sweep/red_zone \
  --trials 20 \
  --device cuda:0
```

Run the current tabletop suite:

```bash
bin/toolframe benchmark \
  --suite current_tabletop \
  --trials 20 \
  --device cuda:0
```

Run every current launcher-runnable task, including provisional side-board /
marker smoke tasks:

```bash
bin/toolframe run-all \
  --trials 1 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

Regenerate the reviewed ToolFrame 6.4 headline subset:

```bash
bin/toolframe preflight-headline
bin/toolframe paper-headline \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

Regenerate the broader ToolFrame 6.4 paper-review candidate suite:

```bash
bin/toolframe preflight-paper
bin/toolframe paper-release \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

`paper_release_6.4` is the current evidence-generation suite for paper review.
It is not a blanket claim that every included task is already headline-ready;
use each task's `status` and contract review notes when building final
statistics tables. Draft side-board smoke tasks and stress variants remain in
their diagnostic suites.

Run only the side-board smoke-review tasks:

```bash
bin/toolframe benchmark \
  --suite current_side_board \
  --trials 1 \
  --device cuda:0 \
  --continue-on-task-error
```

Useful handoff commands:

```bash
bin/toolframe list
bin/toolframe benchmark --validate-registry
bin/toolframe benchmark --describe brush_sweep/red_zone
bin/toolframe run-spec --task brush_sweep/red_zone --preflight-only
bin/toolframe run-spec --suite current_all_runnable --preflight-only
bin/toolframe run-spec --suite paper_release_6.4_headline --preflight-only
bin/toolframe run-spec --suite paper_release_6.4 --preflight-only
bin/toolframe preflight-all \
  --continue-on-task-error \
  --aggregate json,csv,md
```

Use `--output-prefix test000/some_review_root` for explicit local scratch runs.
Use `--plan-json <path>` to save the exact runner commands before launching a
suite. Use `--aggregate-output-root <path>` to keep launcher summaries separate
from per-task runtime evidence.

## Adding a new task or scene

1. Write a `toolframe_task_spec_v1` that names the tool, scene/environment,
   public evidence policy, mainline params, rollout policy, and success checker.
2. Validate and inspect it with `bin/toolframe run-spec --spec <spec> --preflight-only`.
3. For current compatibility runners, synchronize the reviewed split directory,
   `benchmark_config.json`/`benchmark_config.draft.json`, contract stub, and
   `benchmark_tasks.json` entry from that spec.
4. If it is a new family, add it to the appropriate suite only after the
   contract and smoke evidence are reviewed.

The task language, public scene refs, tool refs, success predicates,
diagnostics, thresholds, and allowed labels must be explicit in the contract.
Do not add a new `run_image_grounded_<family>_eval.py` as the first step; extend
the unified spec/executor boundary first, then add a backend only if the task
cannot be expressed by existing stage adapters and success plugins.
