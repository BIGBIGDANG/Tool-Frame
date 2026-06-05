# Tool-Use Benchmark Notes

This directory is the human-readable benchmark/config layer for the active
ToolFrame 6.4 image-grounded mainline. It contains task contracts, benchmark
configuration stubs, review status, and environment notes. It is not a runtime
output directory.

Runtime VLM evidence, tool-ref exports, compiler outputs, rollout summaries,
and videos belong under `artifacts/image_grounded/<task_family>/` when a run is
ready to be treated as benchmark evidence. Interactive scratch outputs stay
outside this tree and are not referenced by publishable configs.

## Status Vocabulary

```text
fixed/current       maintained task/config path with reviewed success semantics
current/provisional runnable mainline or hard task, but some labels, thresholds, or publish smoke remain pending
draft               documented task/environment proposal; no final statistics
stress              intentionally harder diagnostic variant, not the headline split
blocked             missing asset, physics, or success contract
deprecated          historical branch; do not promote without fresh review
```

A task is publishable only when its contract states the public scene refs, tool
refs, compiler constraints, required success predicates, diagnostic-only
predicates, and thresholds. Tool-label contracts are evaluator-side audit
metadata; pending or sparse labels must stay visible in docs/configs, but they
are not method inputs and do not replace executed-effect success metrics.

## Environment Isolation

| Environment id | Scope | Public evidence | Used by |
|---|---|---|---|
| `tabletop_wood_table` | Table-top tasks only | established tabletop/topdown RGB-D views | `brush_sweep`, `eraser_wipe/tabletop_wipe`, `hammer_strike/down`, `screwdriver_insert/*` |
| `side_left_vertical_whiteboard_soft02` | Side-board tasks only | `whiteboard_front` RGB-D view with `soft02` lighting | `marker_trace/write_l`, `marker_trace/write_c`, side-board eraser/hammer/screwdriver drafts |

The side-board whiteboard is a shared environment asset, not a task target.
Task targets such as an HDU sticker, side nail, or side socket must remain
separate from the shared whiteboard environment.

## Current Task Status

| Task family / split | Status | Success-check status | Main config/contract |
|---|---|---|---|
| `brush_sweep/red_zone` | fixed/current | Reviewed target-zone final-occupancy predicate; red-brush tool labels remain diagnostic audit metadata. | `brush_sweep/red_zone/benchmark_config.json` |
| `brush_sweep/forward`, `brush_sweep/left` | diagnostic/legacy | Directional-displacement splits kept for attribution and regression checks, not headline statistics. | `brush_sweep/{forward,left}/benchmark_config.json` |
| `brush_sweep/blue_forward`, `brush_sweep/blue_left` | same_difficulty_variant/pending_tool_refs | Same brush-sweep difficulty and success predicate as red-brush splits; blue-brush DexToolBench asset exists, mainline tool refs pending. | `brush_sweep/blue_*/benchmark_config.json` |
| `eraser_wipe/tabletop_wipe` | fixed/current | Reviewed flat-eraser tabletop contact-path success; flat-eraser labels approved. | `eraser_wipe/tabletop_wipe/benchmark_config.json` |
| `hammer_strike/down` | current/provisional | Semantic contact/strike-direction success; hammer labels pending. | `hammer_strike/down/benchmark_config.json` |
| `hammer_strike/down_mallet` | same_difficulty_variant/pending_tool_refs | Same tabletop down-strike contract as claw-hammer; mallet DexToolBench asset exists, mainline tool refs pending. | `hammer_strike/down_mallet/benchmark_config.json` |
| `screwdriver_insert/wide_socket` | current/provisional | Measured tip radial alignment and insertion depth; long-screwdriver labels pending. | `screwdriver_insert/wide_socket/benchmark_config.json` |
| `screwdriver_insert/deep_socket` | stress/provisional | Strict socket stress split; not the headline split. | `screwdriver_insert/deep_socket/benchmark_config.json` |
| `screwdriver_insert/*_dense` | stress/provisional | Optional diagnostic dense-subgoal variants; excluded from the default current suite. | `screwdriver_insert/*_dense/benchmark_config.json` |
| `marker_trace/write_l` | current/provisional hard task | Human accepted the task scope as a hard benchmark candidate; VLM-judge success and marker-tip labels remain provisional. | `marker_trace/write_l/contract_draft.json` |
| `marker_trace/write_c` | current/provisional hard task | Same marker success gate as `write_l`, with curved path refs; marker-tip labels remain provisional. | `marker_trace/write_c/contract_draft.json` |
| `eraser_wipe/side_whiteboard_hdu_erase` | draft/smoke_review | Registry-runnable smoke task; not approved for headline statistics. | `eraser_wipe/side_whiteboard_hdu_erase/benchmark_config.json` |
| side hammer, side screwdriver | draft | Contracts are documented but not approved for statistics. | `side_board/README.md` |

Detailed per-task contract summaries are in `task_contracts.md`.
Unified input/output conventions and one-command launcher usage are in
`benchmark_io.md`. The task registry is `benchmark_tasks.json`; use
`bin/toolframe run-all` or
`bin/toolframe benchmark --suite current_all_runnable` to launch the default
current mainline suite through the public wrapper. Optional diagnostic stress
variants stay in `diagnostic_stress_optional` and are not part of the default
statistics path.

## ToolFrame 6.4 Paper-Release Candidate Suite

`paper_release_6.4_headline` is the reviewed headline subset for the current
ToolFrame 6.4 tree:

```text
brush_sweep/red_zone
eraser_wipe/tabletop_wipe
```

`paper_release_6.4` is the broader paper-review evidence-generation suite. It
adds current provisional candidates while intentionally excluding draft
side-board smoke tasks, dense diagnostic splits, and same-difficulty tool-shape
variants whose tool refs are still pending:

```text
brush_sweep/red_zone
eraser_wipe/tabletop_wipe
brush_sweep/left
hammer_strike/down
screwdriver_insert/wide_socket
marker_trace/write_l
marker_trace/write_c
```

Task-level `status` still controls what can be reported as headline
statistics.

Use:

```bash
bin/toolframe preflight-headline
bin/toolframe paper-headline \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md

bin/toolframe preflight-paper
bin/toolframe paper-release \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

Before a task moves into the final ICRA/RA-L headline table, its status should
be `fixed/current` or have an explicit review note explaining which provisional
labels, thresholds, or smoke artifacts were accepted. `paper_release_6.4` is
therefore a reproducible evidence-generation suite first, and a final
statistics suite only after those per-task reviews are closed.

Side-board HDU erase remains in `current_side_board` for smoke review, and
`screwdriver_insert/deep_socket` remains in `diagnostic_stress_optional`.

## Current Asset Scope

Use built-in SimToolReal tools and reviewed benchmark target assets for the
current public benchmark release:

```text
simtoolreal/assets/urdf/dextoolbench/
simtoolreal/assets/urdf/parttoolgen_benchmark/
```

External ToolUse imports are intentionally out of scope for this release. Do
not treat converted external assets, old trace-guide assets, or historical run
roots as benchmark inputs. Revisit external assets only after a separate asset
release decision.

Same-difficulty tool-shape variants should start from existing SimToolReal /
DexToolBench OBJ+URDF assets, then become runnable only after matching
`configs/tool_refs/<tool_id>/` files and a smoke run are available. Current
pending examples are `blue_brush` for brush sweep and `mallet_hammer` for
hammer down-strike.

## Mainline Method Boundary

Allowed method inputs:

```text
public RGB-D scene evidence
shared task-generic VLM scene refs: start_point, optional path_points, explicit end_point, scene.direction
RGB-D lifting
public/P3SAM/task-conditioned VLM tool refs
whole-tool SE(3) compiler subgoals
SimToolReal fixed-goal rollout
explicit task-level success checks
```

Forbidden method inputs:

```text
hidden evaluator target refs
oracle target axes
oracle trajectories
pre-rendered target traces or guide paths
VLM-provided robot trajectories
tool_family or task_family fallback
primitive fallback
repair-loop fallback
automatic global-axis retries
seed-specific shortcuts
```

## Review Files

`tool_label_review/` stores human-review staging files for evaluator-side audit
labels. Candidate refs from VLM/P3SAM artifacts are not ground truth for method
selection. Approved `allowed_*` lists, when present, are diagnostic checks for
post-hoc attribution only; task pass/fail is governed by the task's executed
effect metric.

## Where New Work Goes

Start new task/scene work from a unified `toolframe_task_spec_v1`. During the
compatibility period, keep synchronized benchmark configs and human-facing
contracts in this directory:

```text
artifacts/tool_use_benchmark/<task_family>/<split>/
```

Keep benchmark run evidence outside this directory:

```text
artifacts/image_grounded/<task_family>/
```

Do not place videos, RGB-D renders, rollout outputs, or generated previews under
`artifacts/tool_use_benchmark/`.
