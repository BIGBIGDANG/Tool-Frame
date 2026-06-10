# Next Agent Handoff

Read `AGENTS.md` first and follow the collaboration rules.

## Current Priority

Do not expand the benchmark or add new baselines before the current mainline is
clean. The active priority is:

```text
clean code/docs/configs
  -> run current_all_runnable full E2E
  -> manually audit failures
  -> report final task success + coarse attribution
```

## Mainline Method

The only active method path is:

```text
task language
  + public tool mesh / P3SAM tool evidence
  + public SimToolReal RGB-D scene evidence
    -> VLM tool refs from tool-only evidence
    -> VLM scene refs from per-seed scene evidence
    -> RGB-D lifting
    -> tool-centric compiler
    -> whole-tool SE(3) subgoals
    -> SimToolReal rollout
    -> task success check
    -> manual failure attribution
```

Do not pass the scene image into the tool-ref VLM prompt in the current
mainline. Do not use hidden simulator target centers, oracle axes, oracle
trajectories, fixed global-axis fallbacks, tool-family fallbacks, repair loops,
or seed-specific prompt hacks.

## Current Suite

Use `current_all_runnable`:

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

Retired red-zone/directional brush tasks, dense screwdriver tasks, scene-context
tool-ref experiments, fixed/oracle-ref experiments, and unapproved side-board
drafts are not part of the current result.

## Safety Check Before Touching Outputs

```bash
pgrep -af 'run_tool_use_benchmark.py|run_image_grounded|run_p3sam_tool_ref_pipeline|render_sampart3d|run_vlm_scene_keypoint|compile_image_grounded|simtoolreal_rollout'
```

Do not edit, move, or delete active output roots.

## Canonical Commands

```bash
bin/toolframe env-check --mode core
bin/toolframe preflight-current
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
```

Current full E2E run:

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

For multi-GPU runs, split tasks into groups but keep the same protocol.

## Manual Attribution

Each failed trial gets one primary label:

```text
reference
compilation
execution
artifact
needs_review
```

Use tool-ref evidence, scene overlays, lifted refs, compiler outputs, rollout
videos, and task success reports. Do not turn attribution labels into method
inputs.

## Cleanup Rules

- Do not run `git add .`.
- Do not commit `.parttoolgen.local.env`, VLM keys, rollout videos, raw RGB-D
  dumps, large generated roots, or temporary `test00*` artifacts.
- `artifacts/tool_use_benchmark/` is config/contract source, not runtime output.
- `artifacts/test008/` is the current E2E experiment root.
- `test001` through `test007` are development evidence unless a reviewed report
  explicitly promotes a compact artifact.
- Remove obviously wrong generated paths such as nested repository roots.

## Verification

For source/docs/config changes:

```bash
git diff --check
bin/toolframe check --quick
bin/toolframe preflight-current
```

Run full rollouts only after the method/docs/config state matches the current
protocol.
