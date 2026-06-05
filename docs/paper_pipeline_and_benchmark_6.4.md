# ToolFrame 6.4 Paper Pipeline And Benchmark Protocol

This document freezes the current paper-facing ToolFrame 6.4 mainline: what the
method consumes, what each stage writes, which benchmark tasks are ready for
statistics, and which generated artifacts may be cited as results.

## Method Boundary

ToolFrame 6.4 is an image-grounded tool-use pipeline. The method may use public
task language, public RGB-D scene evidence, public tool meshes, and reviewed
tool-part evidence. It must not use hidden evaluator target refs, oracle target
axes, oracle trajectories, tool-family or task-family fallback, primitive
fallback, repair-loop fallback, or seed-specific shortcuts.

The end-to-end path is:

```text
toolframe_task_spec_v1 or registered benchmark config
  -> task.json / method_task.json / run_config.json
  -> scene_evidence/scene_evidence.json
  -> vlm_scene_keypoints/scene_keypoints.json + grounding_overlay.png
  -> lifted_program/lifted_refs.json + constraint_program.json
  -> lifted_program/constraint_program_with_tool_refs.json
  -> compile/simtoolreal_fixed_goals.json
  -> rollout/rollout_summary.json
  -> visual_evidence/manifest.json
  -> summary.json + summary.md
```

## Pipeline Stages

| Paper panel | Stage | Inputs | Outputs | Boundary |
| --- | --- | --- | --- | --- |
| Fig. A | Tool-centric prior extraction | Tool mesh, P3SAM/SAMPart3D part evidence, task text, reviewed visual evidence | `configs/tool_refs/<tool_id>/{part_selection,contact_point_selection,axis_binding,tool_part_graph}.json` plus evidence cards | Produces only `tool.contact_point` and `tool.motion_axis`. The VLM selects from part/marker evidence; it does not invent coordinates. |
| Fig. B | Scene key-point grounding | Public RGB-D images, depth, camera files, and natural task language | `vlm_scene_keypoints/scene_keypoints.json`, overlay PNGs, `lifted_program/lifted_refs.json` | The VLM marks sparse scene refs: `start_point`, optional `path_points`, `end_point`, and `scene.direction` when needed. It does not output robot trajectories or hidden 3D targets. |
| Fig. C | Geometry-constrained optimizer | Lifted scene refs, reviewed tool refs, method task, start state | `compile/*`, `tool_pose_subgoals.json`, diagnostic path-gate report, `simtoolreal_fixed_goals.json` | The compiler solves whole-tool `SE(3)` subgoals. It may optimize over VLM-valid contact-point and motion-axis candidates, but it does not search outside those public candidates, guess missing axes from families, or insert corridor/tracking/recovery repair waypoints. |
| Fig. D | Policy rollout and evaluation | Fixed goals, observed scene graph, rollout config, task contract | Rollout videos/reports, `task_success_check`, `failure_attribution`, compact summaries | The SimToolReal policy executes fixed goals. Success checks are evaluator-side contracts and never feed back into method inputs. |

The supported public constraint vocabulary is intentionally small:
`point_match`, `point_offset`, `axis_align`, `move_along`,
`orientation_offset_about_axis`, and `clearance`.

## Tool-Reference Evidence Contract

The tool-reference path is two-stage:

1. The VLM selects a functional part from segmented part evidence.
2. Only the selected part is shown for contact-point and motion-axis selection,
   with its part bounding box and separate face highlights available as
   review evidence.

Do not reintroduce legacy per-face contact-choice cards such as
`contact_point_choices/candidate_*_F*_face.png`. Tool labels are audit/evaluator
metadata; they are not method inputs.

Current reviewed or active tool refs live in `configs/tool_refs/`, especially:

```text
claw_hammer/
flat_eraser/
long_screwdriver/
red_brush/
staples_marker/
```

## Benchmark Suites

Use the registered suites instead of ad hoc task-specific launch scripts.

| Suite | Purpose | Tasks |
| --- | --- | --- |
| `paper_release_6.4_headline` | Current headline statistics subset | `brush_sweep/red_zone`, `eraser_wipe/tabletop_wipe` |
| `paper_release_6.4` | Broader paper-review evidence-generation suite | Headline tasks plus `hammer_strike/down`, `screwdriver_insert/wide_socket`, `marker_trace/write_l`, `marker_trace/write_c` |
| `legacy_directional_diagnostics` | Legacy brush regression checks, not mainline statistics | `brush_sweep/forward`, `brush_sweep/left` |
| `diagnostic_stress_optional` | Harder stress diagnostics, not headline | `screwdriver_insert/deep_socket`, dense screwdriver variants |
| `current_side_board` | Side-board smoke/review tasks | Marker trace and side-board draft tasks |

Pending same-difficulty tool variants such as blue-brush sweep and mallet
strike remain excluded from statistics until their reviewed `configs/tool_refs`
bindings exist.

## Task Status For Reporting

| Task | Status | Final success contract |
| --- | --- | --- |
| `brush_sweep/red_zone` | `fixed/current` | Final stable target-block xy center inside the visible red outline zone. Strict tracking and directional displacement are diagnostics. |
| `eraser_wipe/tabletop_wipe` | `fixed/current` | `eraser_wipe_contact_coverage`: path-gate report, lifted start/end refs on the visible raised tabletop HDU board, and flat-eraser bottom contact-footprint coverage >= 0.3 of the target long-axis extent. Visual disappearance, strict tracking, policy-goal pass, and semantic execution pass are diagnostics. |
| `brush_sweep/forward`, `brush_sweep/left` | `diagnostic/legacy` | Directional-displacement variants retained for attribution/regression, not current smoke or paper statistics. |
| `hammer_strike/down` | `current/provisional` | VLM strike direction, path-gate report, and rollout semantic contact. Nail displacement is not required. |
| `screwdriver_insert/wide_socket` | `current/provisional_headline_split` | Tip path insertion with radial alignment and insertion-depth checks. |
| `marker_trace/write_l`, `marker_trace/write_c` | `current/provisional_hard_task` | VLM judge over marker-tip ink evidence; contact/tracking metrics are diagnostic. |
| Side-board eraser/hammer/screwdriver | `draft` or `smoke_review` | Useful for smoke and contract design, not paper statistics. |

## Metrics Table

Paper tables should report:

```text
task
contract_status
N
functional_grounding_rate
scene_grounding_rate
subgoal_validity_rate
policy_execution_rate
task_success_rate
```

Each rate is `passed_trials / evaluated_trials`. `task_success_rate` is the
only final task metric. The other module rates explain where failures occur:
tool grounding, scene grounding, compiler/subgoal generation, or policy
execution. Missing artifacts count as failures at the first missing evidence
stage. Ambiguous evidence should be marked `needs_review`.

## Running The Paper Suites

From the repository root:

```bash
bin/toolframe env-check --mode rollout
bin/toolframe preflight-headline
bin/toolframe paper-headline \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

For broader candidate evidence:

```bash
bin/toolframe preflight-paper
bin/toolframe paper-release \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

For a single registered task:

```bash
bin/toolframe benchmark --task brush_sweep/red_zone --trials 1 --device cuda:0
bin/toolframe benchmark --task brush_sweep/red_zone --trials 20 --device cuda:0
```

Useful non-rollout checks:

```bash
bin/toolframe list
bin/toolframe benchmark --validate-registry
bin/toolframe run-spec --suite paper_release_6.4_headline --preflight-only
bin/toolframe run-spec --suite paper_release_6.4 --preflight-only
bin/toolframe check --quick
```

Current benchmark configs cap rollouts at 1500 max steps for the paper-facing
tasks, except explicit stress or legacy diagnostics documented in their configs.

## Result And Artifact Boundaries

Use three release layers:

| Layer | Examples | Policy |
| --- | --- | --- |
| `release-source` | `bin/toolframe`, `parttoolgen/src/`, schemas, tests, `configs/tool_refs/`, `artifacts/tool_use_benchmark/` | Commit and treat as reproducible source of truth. |
| `release-evidence` | Compact reviewed tool-ref evidence, selected `launcher_summary.{json,csv,md}`, selected manifests | Cite only after review; keep compact. |
| `local-generated` | RGB-D dumps, raw VLM responses, rollout videos, compiler working dirs, temporary summaries | Do not treat as source. Keep out of release unless a specific compact evidence package is approved. |

Final paper statistics must come from completed `launcher_summary.{json,csv,md}`
files under `reports/paper_release_6.4/` and the corresponding per-task
`summary.json` files. Do not cite partial per-task summaries while a launcher is
still running.

`reports/mainline_6.4/` is a migrated provenance/audit snapshot from earlier
5.21/test002 reconstruction. It is useful for checking evidence layout and
historical debugging, but it is not the ToolFrame 6.4 paper-release result
table.

## Before Adding New Tasks

Freeze the above protocol first, then add tasks through the same extension
points:

1. Build or review `configs/tool_refs/<new_tool_id>/`.
2. Add the task config under `artifacts/tool_use_benchmark/<family>/<split>/`.
3. Register it in `benchmark_tasks.json`.
4. Run `bin/toolframe run-spec --suite <suite> --preflight-only`.
5. Run a 1-trial smoke with full evidence and videos.
6. Review task labels, thresholds, and failure attribution before any
   multi-trial statistics.
