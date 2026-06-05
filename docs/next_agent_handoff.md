# Next Agent Handoff

Read `AGENTS.md` first and follow the Agent Collaboration Rules.

## Current State

- Workspace: repository root, normally `<your clone>/ToolFrame_6.4`.
- Canonical workflow: see `docs/current_pipeline.md`.
- Active task families: `brush_sweep`, `hammer_strike`, `eraser_wipe`,
  `screwdriver_insert`, marker letter-writing (`marker_trace/write_l`, `marker_trace/write_c`), and draft side-board tasks.
- Active method boundary: public SimToolReal RGB-D scene evidence,
  task-generic VLM scene refs, RGB-D lifting, public/P3SAM/task-conditioned VLM
  tool refs, whole-tool SE(3) subgoals, SimToolReal fixed-goal rollout, and
  explicit task-level success checks.
- Prohibited method inputs: hidden evaluator target refs, oracle target axes,
  oracle trajectories, `tool_family`/`task_family` fallback, primitive
  fallback, repair-loop fallback, automatic global-axis retries, and
  seed-specific shortcuts.

## Active Output Safety Check

Before touching generated output roots under `artifacts/image_grounded/**` or
local scratch roots under `test000/`, check for active runners:

```bash
ps -eo pid,ppid,stat,etime,cmd | \
  rg 'run_seed20|run_smoke_left_tasks_parallel|run_image_grounded|simtoolreal_rollout|run_vlm_scene_keypoint_grounding'
```

Do not edit, move, stage, or delete active run roots while they are active.

## Task Status

| Status | Item | Notes |
|---|---|---|
| Current | Brush sweep forward/left | Success is directional target displacement; tracking is diagnostic. |
| Current | Hammer down | Success is support-direction/compiler/semantic contact; target displacement is not required. |
| Current/reviewed | Eraser wipe | Flat-eraser label direction reviewed; strict bottom-surface coverage and visual disappearance are diagnostic. |
| Current/provisional | Screwdriver insert wide socket | Corrected success uses compiled terminal socket-goal check plus measured tip XY/radial error and inside-socket depth; terminal endpoint error is diagnostic. |
| Current/hard | Marker trace write L/C | Side-board whiteboard tasks under `artifacts/tool_use_benchmark/marker_trace/write_l/` and `artifacts/tool_use_benchmark/marker_trace/write_c/`; use the shared left-side vertical side-task environment asset `simtoolreal/assets/urdf/parttoolgen_benchmark/marker_trace_whiteboard/left_side_vertical_whiteboard_002x0475x101.urdf` and canonical preview `artifacts/image_grounded/marker_trace/path_whiteboard_preview_staples_seed_000/sim_rgbd_left_vertical_wall_wide_front_aligned/`; switched to `staples_marker` after topdown review showed `sharpie_marker` is capped; use the generic brush/eraser-style VLM scene-ref prompt so the VLM marks start/optional-waypoints/end public refs on the blank side whiteboard via `whiteboard_front`; ink evidence must be rendered only from the reviewed marker-tip contact ref via `python -m parttoolgen.evaluation.marker_trace_ink_evidence`; final success is VLM-judged produced ink legibility, while rollout/contact metrics are diagnostic. |
| Draft/smoking | Side-board HDU eraser | Shared side-board whiteboard, `soft02` lighting, natural task language: "Use this eraser to erase the whole HDU pattern on the whiteboard." Target is a visual-only HDU sticker/handwriting decal. Human proposed erase >=70%; final implementation must measure flat-eraser functional contact coverage on the whiteboard, not whole-tool root pose or collision with the sticker. |
| Draft/smoking | Side-board hammer/screwdriver | Reuse the shared side-board whiteboard plus separate hammer target/socket actors. Keep side-board assets isolated from tabletop tasks and use `whiteboard_front` evidence. Success contracts are still pending review. |
| Removed/deprecated | Marker trace line/path-waypoint drafts | Superseded by marker letter-writing hard tasks; old runtime outputs may remain only as historical scratch evidence. |
| Blocked | Knife line cut | Needs valid mesh and cut/separation contract. |
| Deferred | Spatula | Wait for clearer task/object contract. |

## Benchmark Expansion Objective

The next main work item is benchmark expansion, not single-seed repair. Build a
candidate matrix from the current public-release asset sources:

1. segmented SimToolReal tools under `simtoolreal/assets/urdf/dextoolbench/`;
2. the P3SAM package reported by
   `bin/toolframe inventory --compact`;
3. small reviewed benchmark assets under
   `simtoolreal/assets/urdf/parttoolgen_benchmark/`.

Imported external ToolUse assets are intentionally omitted from the current
public release. Reintroduce them only after a separate asset/license review and
human-reviewed benchmark contracts.

For each feasible tool/object/task combination, record the proposed public
scene evidence, task language, required scene refs, tool refs, compiler
constraint type, rollout runner/config path, and a human-reviewed success
contract. Only promote a task from candidate to benchmark after the contract is
approved and a small smoke shows coherent evidence, compilation, rollout, and
videos.

## Success-Check Policy

Every new task needs a task-specific benchmark contract before large statistics.
For each task, define and get human approval for:

1. visible public scene refs available to the VLM,
2. tool refs and part-face grounding source,
3. required task-level success predicates,
4. diagnostic-only predicates,
5. thresholds/tolerances,
6. allowed functional part/face/motion-axis labels.

Do not invent labels or thresholds. Smoke runs before review are debugging
evidence, not final pass rates.

## Cleanup Guidance

- Do not run `git add .`.
- Leave generated run outputs under `artifacts/image_grounded/**` uncommitted.
- Avoid staging heavy/runtime artifacts such as imported third-party asset
  roots, `artifacts/tool_prior/**`, and generated preview outputs unless
  explicitly requested.
- Keep benchmark configs under `artifacts/tool_use_benchmark/`.
- Keep run outputs under `artifacts/image_grounded/<task_family>/`.
- `test000/` is allowed as an interactive scratch/review root when the human
  explicitly asks for it; do not treat it as a benchmark config root.
- `visual_evidence/` should normally be manifest-only.

Low-risk stale roots to remove only after confirming they are not active and
their summary values are no longer needed:

```text
artifacts/image_grounded/brush_sweep/forward_original_rgb/
artifacts/image_grounded/hammer_strike/down_original_rgb_smoke/
artifacts/image_grounded/eraser_wipe/tabletop_wipe_original_rgb_smoke/
artifacts/image_grounded/eraser_wipe/tabletop_wipe_seed20_original_rgb/
artifacts/image_grounded/eraser_wipe/tabletop_wipe_seed20_2500steps_original_rgb/
artifacts/image_grounded/eraser_wipe/tabletop_wipe_seed20_2000steps_nonoverlap_original_rgb/
artifacts/image_grounded/screwdriver_insert/scene_preview_seed_000/
artifacts/image_grounded/vlm_prompt_unified_scene_grounding_smoke/
```

Keep `vlm_prompt_unified_scene_grounding_smoke_v2/` as the current generic
prompt smoke evidence unless a newer prompt audit supersedes it.

## Verification

Before committing source/docs/config changes, run:

```bash
git diff --check
bin/toolframe check --quick
bin/toolframe preflight-headline
bin/toolframe preflight-paper
bin/toolframe preflight-all
```

## Prompt For The Next Coding Agent

```text
Read AGENTS.md first and follow the Agent Collaboration Rules.

You are working in the ToolFrame 6.4 repository root.

Start by checking for active runners before touching generated output roots.
Do not edit, move, stage, or delete active output roots while they are active.

Continue the image-grounded tabletop and side-board benchmark only within the method boundary:
public RGB-D scene evidence, task-generic VLM scene refs, RGB-D lifting,
public/P3SAM/task-conditioned VLM tool refs, whole-tool SE(3) compiler
subgoals, SimToolReal fixed-goal rollout, and explicit task-level success
checks. Do not use hidden evaluator target refs, oracle target axes, oracle
trajectories, tool_family/task_family fallback, primitive fallback, repair-loop
fallback, automatic global-axis retries, or seed-specific shortcuts.

Prefer documentation/contract consolidation and small, verifiable fixes over
broad refactors. For any proposed next task, first write the success contract
and get human approval for scene refs, tool refs, required predicates,
diagnostic predicates, thresholds, and tool labels.

The next priority is to finish smoke/review bookkeeping for the retained
tabletop and side-board tasks, then expand the benchmark from built-in
SimToolReal segmented tools and small reviewed benchmark assets. External
ToolUse assets are outside the current public release unless the project owner
approves a separate asset/license review.

Produce a candidate task matrix before implementing new tasks. For each
candidate, specify public scene evidence, natural task language, VLM scene refs,
tool-side refs, compiler constraints, runner/config/output paths, success
contract, diagnostic metrics, and why the task is or is not ready. Keep configs
under artifacts/tool_use_benchmark/<task_family>/<split>/ and outputs under
artifacts/image_grounded/<task_family>/ unless the human explicitly asks for a
temporary `test000/` run; do not place runtime videos/evidence under
artifacts/tool_use_benchmark/.
```
