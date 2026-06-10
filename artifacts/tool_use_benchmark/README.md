# Tool-Use Benchmark Notes

This directory is the source-of-truth benchmark/config layer. It contains
registered task configs and human-readable status notes. It is not a runtime
output directory.

Runtime evidence belongs under:

```text
artifacts/image_grounded/...
artifacts/test008/...
```

Do not place rollout videos, RGB-D renders, VLM raw responses, or temporary
experiment outputs under `artifacts/tool_use_benchmark/`.

## Current Experiment Scope

The current mainline experiment is full end-to-end execution:

```text
VLM tool refs
  + VLM scene refs
  + RGB-D lifting
  + tool-centric compiler
  + SimToolReal rollout
  -> final task success
  -> manual failure attribution
```

We currently report final E2E success and manually attribute failures. Fixed
refs, oracle refs, dense module-accuracy tables, and detailed ablations are not
part of the active experiment protocol.

## Current Suite

Use `current_all_runnable` for the current E2E pass:

```text
brush_sweep/off_table_red
brush_sweep/off_table_blue
eraser_wipe/tabletop_wipe
eraser_wipe/tabletop_wipe_handle
eraser_wipe/side_whiteboard_hdu_erase
hammer_strike/down
hammer_strike/down_mallet
screwdriver_insert/wide_socket
screwdriver_insert/wide_socket_short
screwdriver_insert/deep_socket
marker_trace/write_l
marker_trace/write_c
```

Legacy brush directional/red-zone splits, dense screwdriver stress variants,
and unapproved side-board drafts are diagnostic only.

## Status Vocabulary

```text
current/e2e         maintained current full end-to-end task/config path
current/provisional runnable current task, but thresholds or evidence still need review
draft               documented task/environment proposal; not for statistics
stress              harder diagnostic variant; not part of the current E2E table
legacy             historical branch; do not promote without fresh review
blocked             missing asset, physics, or success contract
```

## Current Task Status

| Task family / split | Current use | Notes |
| --- | --- | --- |
| `brush_sweep/off_table_red`, `brush_sweep/off_table_blue` | Current E2E | Sweep block off table; red and blue brushes are tool-shape variants. |
| `eraser_wipe/tabletop_wipe`, `eraser_wipe/tabletop_wipe_handle` | Current E2E | Tabletop raised-board wipe; flat and handle erasers are tool-shape variants. |
| `eraser_wipe/side_whiteboard_hdu_erase` | Current E2E smoke/review | Included for broad E2E coverage, but review evidence carefully. |
| `hammer_strike/down`, `hammer_strike/down_mallet` | Current E2E | Tabletop down-strike; claw hammer and mallet are tool-shape variants. |
| `screwdriver_insert/wide_socket`, `screwdriver_insert/wide_socket_short` | Current E2E | Tabletop insertion with long and short screwdriver variants. |
| `screwdriver_insert/deep_socket` | Current E2E stress/review | Included as a harder current insertion candidate. |
| `marker_trace/write_l`, `marker_trace/write_c` | Current E2E hard tasks | Side-board marker tasks; final success uses marker evidence and human review. |
| `brush_sweep/red_zone`, `brush_sweep/forward`, `brush_sweep/left`, `brush_sweep/blue_forward`, `brush_sweep/blue_left` | Legacy diagnostics | Retired from the current E2E table. |
| `screwdriver_insert/*_dense` | Diagnostic stress | Excluded from the current E2E table. |
| side hammer / side screwdriver drafts | Draft | Keep for future scene expansion, not current statistics. |

## Method Boundary

Allowed method inputs:

```text
task language
public RGB-D scene evidence
public tool mesh / P3SAM part evidence
VLM-selected tool refs from tool-only evidence
VLM-selected scene refs from per-seed scene evidence
RGB-D lifted refs
compiler-generated whole-tool SE(3) subgoals
```

Forbidden method inputs:

```text
hidden evaluator target refs
oracle trajectories
oracle target axes
task-family or tool-family fallback refs
scene-image hints for tool-ref selection
repair-loop or retry shortcuts
seed-specific prompt hacks
automatic private snapping to target centers
```

## Running

Preflight:

```bash
bin/toolframe preflight-current
```

Current E2E run:

```bash
bin/toolframe current-e2e \
  --trials 10 \
  --seed-start 0 \
  --device cuda:0 \
  --capture-video-every 1 \
  --continue-on-task-error \
  --tool-ref-model gpt-5.5 \
  --tool-ref-base-url "$VLM_API_BASE" \
  --output-prefix artifacts/test008/e2e_10seed_gpt55_20260607 \
  --aggregate json,csv,md
```

For multi-GPU runs, split the task list into groups but keep the same protocol.

## Manual Attribution

After the run completes, inspect each failed trial and assign the primary label:

```text
reference
compilation
execution
artifact
needs_review
```

Manual attribution should use the tool-ref evidence, scene grounding overlay,
lifted refs, compiler outputs, rollout videos, and task success report. The
attribution label explains the failure; it is not a method input.

## Where New Work Goes

Add new task configs under:

```text
artifacts/tool_use_benchmark/<task_family>/<split>/
```

Add new tool refs under:

```text
configs/tool_refs/<tool_id>/
```

Generated experiment evidence must stay outside this config tree.
