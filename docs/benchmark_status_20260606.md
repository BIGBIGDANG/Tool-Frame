# Benchmark Status And Cleanup Notes

This file is a cleanup/status note, not a final result table.

## Current Stage

The current stage is:

```text
Full E2E benchmark -> final task success -> manual failure attribution
```

The current run target is `current_all_runnable` with 10 seeds per task. The
result root should live under:

```text
artifacts/test008/
```

Use `docs/current_pipeline.md` and
`docs/paper_pipeline_and_benchmark_6.4.md` as the active protocol.

## Current Mainline Tasks

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

## Historical Evidence

Earlier smoke and 20-seed folders under `test001` through `test007` are
development evidence. They may be useful for debugging but should not be cited
as current paper statistics.

Examples of stale or diagnostic roots:

```text
artifacts/test003/mainline_20seed_20260605/
artifacts/test004/
artifacts/test005/
artifacts/test006/
artifacts/test007/
```

These folders include useful episodes, tool-ref prompt experiments, compiler
cleanup checks, and scene-context experiments. They do not define the current
method.

## Retired Or Diagnostic Branches

Do not include these in the current E2E success table:

```text
brush_sweep/red_zone
brush_sweep/forward
brush_sweep/left
brush_sweep/blue_forward
brush_sweep/blue_left
screwdriver_insert/*_dense
side-board hammer/screwdriver draft tasks
scene-context tool-ref selection experiments
fixed/oracle-ref baseline experiments
```

They can stay in the registry or artifacts only as diagnostics, historical
provenance, or future-work seeds.

## Cleanup Rules

- Remove obviously wrong generated paths such as nested repository roots.
- Keep source/config/docs/tests small and reproducible.
- Keep full rollout videos and raw RGB-D evidence out of git unless a compact
  evidence package is explicitly promoted.
- Do not promote a partial launcher run to a result table.
- If a failure is ambiguous, mark it `needs_review` instead of forcing a module
  label.
