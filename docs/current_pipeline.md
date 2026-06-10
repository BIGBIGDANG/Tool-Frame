# Current Pipeline

This document is the current ToolFrame 6.4 mainline contract. It intentionally
keeps the method and experiment protocol simple: run the full image-grounded
tool-use chain, report final task success, and manually audit failures.

## Mainline Method

The only current method path is:

```text
task language
  + public tool mesh / P3SAM part evidence
  + public SimToolReal RGB-D scene evidence
    -> VLM selects tool refs from tool-only evidence
    -> VLM marks scene refs from the per-seed RGB-D scene
    -> RGB-D lifting converts scene pixels to 3D refs
    -> tool-centric compiler generates whole-tool SE(3) subgoals
    -> SimToolReal executes the fixed-goal rollout
    -> evaluator checks task success and humans audit failures
```

Tool refs are selected per trial from stable P3SAM/tool-prior evidence. The
tool-ref VLM input does not include the task scene image in the current
mainline. Scene refs are grounded separately from the per-seed RGB-D scene.

## Current Experiment Protocol

The current paper-development experiment is full end-to-end success only:

```text
Full E2E success = VLM tool refs + VLM scene refs + lifting + compiler + rollout
```

We do not currently report fixed-ref baselines, oracle-ref baselines, dense
module accuracy tables, or automatic per-module scores. Failed trials are
manually attributed from artifacts and videos using coarse labels:

```text
Reference failure    VLM tool refs or scene refs are not sufficient.
Compilation failure  Refs are plausible but compiler cannot produce feasible subgoals.
Execution failure    Subgoals are plausible but grasp/tracking/contact fails.
Artifact             Accidental success/failure, hand collision, drop, or simulation artifact.
Needs review         Evidence is ambiguous.
```

Task success requires both the task outcome and the expected functional tool
interaction. A result caused by hand contact, a dropped tool, a non-functional
tool part, or a simulation artifact is not counted as success.

## Current Mainline Suite

The current suite is `current_all_runnable`:

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

Legacy brush directional/red-zone tasks, dense screwdriver stress tasks, and
draft side-board expansion tasks are not part of the current E2E result table.
They may remain in the registry only as diagnostics or archived development
branches.

## Canonical Commands

Use the thin public wrapper:

```bash
cd <clone>/ToolFrame_6.4
bin/toolframe env-check --mode core
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
```

For the current full E2E experiment, provide the VLM endpoint/key through the
local environment and run the current suite with per-seed VLM tool refs:

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

When running multiple GPUs, split the suite into groups but keep the same
arguments and output prefix.

## Method Boundary

Allowed method inputs:

```text
task language
public tool mesh / P3SAM part evidence
public RGB-D scene images and depth
VLM-selected tool refs
VLM-selected scene refs
RGB-D lifted 3D refs
compiler-generated whole-tool SE(3) subgoals
```

Forbidden method inputs:

```text
hidden simulator target refs
oracle target axes
oracle trajectories
task-family or tool-family fallback refs
scene-image hints for tool-ref selection
repair-loop or retry shortcuts
seed-specific prompt hacks
fixed global-axis fallback unless exposed as a public VLM choice
automatic snapping to private target centers
```

## Runtime Artifacts

Each trial should contain:

```text
sim_rgbd/                       public RGB-D scene evidence
tool_refs/                      per-trial VLM tool-ref selection evidence
vlm_scene_keypoints/             VLM scene refs and overlay
lifted_program/                  lifted refs and ProgramIR
compile/                         whole-tool subgoals and fixed goals
rollout/                         rollout summary/videos
trial_result.json                compact per-trial result
```

Launcher summaries live under the run's `benchmark_runs/` directory. Do not
cite partial runs. Test roots `test001` through `test007` are development
evidence unless explicitly promoted by a reviewed report.
