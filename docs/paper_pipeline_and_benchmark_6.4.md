# ToolFrame 6.4 Paper Pipeline And Benchmark Protocol

This note freezes the current paper-development protocol. The current priority
is not a large baseline matrix; it is a clean, reproducible full end-to-end
benchmark with human failure attribution.

## Claim Under Test

ToolFrame uses visual-language grounded tool and scene references with a
tool-centric compiler to generate whole-tool SE(3) subgoals executable by a
pretrained SimToolReal policy.

The current experiment asks:

```text
Can the full chain complete diverse tool-use tasks, and where does it fail?
```

## Method Path

```text
registered benchmark config
  -> task.json / method_task.json / run_config.json
  -> SimToolReal RGB-D scene evidence
  -> VLM tool-ref selection from tool-only P3SAM evidence
  -> VLM scene-ref grounding from the current scene image
  -> RGB-D lifting into public 3D scene refs
  -> compiler-generated whole-tool SE(3) subgoals
  -> SimToolReal fixed-goal rollout
  -> task success check
  -> manual failure attribution
```

Tool-ref selection and scene grounding are separate. The current mainline does
not pass the scene image into the tool-ref VLM prompt.

## Current Results To Report

Report final end-to-end task success:

```text
E2E success = task outcome achieved AND expected functional tool interaction
```

Do not count accidental or artifact-driven outcomes as success. Examples that
remain failures include hand contact completing the task, a dropped tool hitting
the target, a non-functional tool part causing the effect, or mesh penetration.

For failed trials, manually assign the primary failure source:

| Label | Meaning |
| --- | --- |
| `reference` | VLM tool refs or scene refs are insufficient for the intended interaction. |
| `compilation` | Refs are plausible, but the compiler cannot generate feasible subgoals. |
| `execution` | Subgoals are plausible, but grasping, tracking, contact, or dynamics fail. |
| `artifact` | Simulation artifact, unintended success/failure, hand collision, drop, or penetration. |
| `needs_review` | Evidence is ambiguous. |

This is the current reporting table:

```text
Task | Tool variant | Scene layout | Seeds | E2E success | Main failure sources
```

Fixed-ref baselines, oracle-ref upper bounds, detailed grounding acceptability,
and compiler diagnostic tables are optional future additions. They are not part
of the current mainline result protocol.

## Current Task Suite

Use `current_all_runnable` for the current full E2E pass:

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

The suite intentionally excludes retired brush red-zone/directional tasks,
dense screwdriver stress tasks, and unapproved side-board expansion drafts.

## Task-Level Success Criteria

The exact evaluator code owns thresholds, but the human review standard is:

| Task family | Expected functional interaction |
| --- | --- |
| Brush sweep | Brush working face contacts the block and sweeps it off the table. |
| Eraser wipe | Eraser functional surface contacts and covers the target mark region. |
| Hammer strike | Hammer head / selected striking region contacts the target. |
| Screwdriver insert | Screwdriver tip/axis enters the socket region. |
| Marker trace | Marker tip contacts the board and leaves the intended mark. |

Each trial should be reviewed with both the rollout outcome and the functional
interaction evidence in mind.

## Running The Current E2E Experiment

From the repository root:

```bash
bin/toolframe env-check --mode rollout
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
```

Then run the suite with per-seed VLM tool refs:

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

For multi-GPU runs, split the task list into groups but keep the same run
protocol and output prefix. Do not cite partial runs.

## Artifact Boundary

Commit source, configs, schemas, tests, and compact reviewed tool-ref configs.
Do not commit raw rollout videos, RGB-D dumps, temporary `test00*` outputs, VLM
raw responses, or local credentials.

Development outputs from `test001` through `test007` are debugging evidence.
`test008` is the current E2E experiment root. A run becomes paper evidence only
after its launcher summaries and manual attribution table are reviewed.

## Future Extensions

If the E2E result needs stronger analysis, add one extension at a time:

```text
1. Reference acceptability by manual review.
2. Oracle tool/scene refs as an upper bound.
3. Fixed/default refs as a simple baseline.
4. Compiler diagnostic metrics.
```

Do not reintroduce these as required current-stage metrics until the E2E suite
and manual attribution process are stable.
