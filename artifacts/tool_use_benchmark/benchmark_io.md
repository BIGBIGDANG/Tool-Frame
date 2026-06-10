# Current Benchmark Input/Output Contract

Date: 2026-06-07

This file describes the current ToolFrame 6.4 benchmark I/O. The active
experiment is full E2E success with manual failure attribution. Historical
paper/headline suites, red-zone brush variants, dense screwdriver diagnostics,
and scene-context tool-ref trials are not part of the current result protocol.

## Source Inputs

Benchmark definitions live under:

```text
artifacts/tool_use_benchmark/<task_family>/<split>/benchmark_config.json
artifacts/tool_use_benchmark/benchmark_tasks.json
```

The active suite is:

```text
current_all_runnable
```

Each current task config defines public task language, tool id, scene asset
policy, seeded scene variation, rollout policy, and success checker settings.
It must not contain hidden evaluator target refs, oracle axes, oracle
trajectories, repair/fallback shortcuts, or seed-specific prompt hacks.

## Runtime Root

Current E2E runs should write to an explicit experiment root, for example:

```text
artifacts/test008/e2e_10seed_gpt55_20260607/
```

If no explicit prefix is given, task-family runners may write under:

```text
artifacts/image_grounded/<task_family>/<split>_<run_id>/
```

Do not write runtime evidence into `artifacts/tool_use_benchmark/`.

## Module I/O

### Registry Launcher

Input:

```text
benchmark_tasks.json
benchmark_config.json
CLI flags: suite/task, trials, seeds, device, output root, tool-ref mode
```

Output:

```text
expanded runner commands
per-task run roots
aggregate summaries: JSON / CSV / Markdown
```

Main entry:

```bash
bin/toolframe current-e2e --trials 10 --device cuda:0 --continue-on-task-error
```

For single-task debugging:

```bash
bin/toolframe benchmark \
  --task brush_sweep/off_table_red \
  --trials 1 \
  --device cuda:0 \
  --tool-refs-mode vlm_per_seed
```

### Scene Setup And Evidence

Input:

```text
task config
seed id
tool start policy
target/object sampler
SimToolReal scene/asset definitions
```

Output:

```text
sim_rgbd/
method_task.json
run_config.json
scene/object state metadata
```

The RGB-D scene evidence is public method input for scene grounding. Tool start
poses should place tools on the table near the hand unless a reviewed task
contract says otherwise.

### Tool-Reference Selection

Input:

```text
tool mesh / P3SAM part evidence
selected tool part overview
candidate face/contact/axis evidence images
task language
```

Output:

```text
tool_selection_visual_evidence/
tool_refs/ or selected_refs/
part_selection.json
contact_point_selection.json
axis_binding.json
p3sam_tool_ref_selection_summary.json
```

Current mainline selection is per trial with `--tool-refs-mode vlm_per_seed`.
The tool-ref prompt uses tool-only evidence; it does not receive the task scene
image. The VLM may select multiple plausible contact-point and motion-axis
candidates, and the compiler consumes the valid set rather than a single
hard-coded face.

### Scene-Reference Grounding

Input:

```text
task language
per-seed RGB-D scene evidence
public default-axis candidates when explicitly exposed by the scene
```

Output:

```text
vlm_scene_keypoints/
scene_keypoints.json
scene overlay images
```

The VLM marks task-relevant scene points and action direction references from
the visible scene. The compiler must not infer private target centers or hidden
world-axis fallbacks.

### RGB-D Lifting

Input:

```text
scene_keypoints.json
camera intrinsics/extrinsics
depth image
task config
```

Output:

```text
lifted_program/
constraint_program.json
lift report / diagnostics
```

The lifted refs are public 3D scene refs. They are the target geometry used by
the compiler.

### Tool-Centric Compiler

Input:

```text
lifted scene refs
VLM-selected tool contact candidates
VLM-selected tool motion-axis candidates
tool part graph / part bbox
scene support surface metadata
```

Output:

```text
compile/tool_pose_subgoals.json
compile/tool_path_solver_report.json
compile/tool_part_graph.json
compile/target_part_graph.json
compile/simtoolreal_fixed_goals.json
optional subgoal-only visualization video
```

Current compiler invariants:

```text
1. All VLM-valid contact and motion-axis candidates participate in selection.
2. Align stage rotates the tool so the selected motion axis follows the task
   use direction; translation is left to approach/use stages.
3. Approach moves the selected contact point toward the pre-contact point.
4. Use moves the selected contact point through the lifted scene endpoint and
   waypoint sequence.
5. Every generated subgoal must keep the relevant tool part bbox above the
   support surface; this should be handled during generation, not as a hidden
   post-hoc repair.
6. No same/opposite axis polarity fallback may reverse the intended use
   direction unless that choice is explicitly represented in the public refs.
```

### Rollout

Input:

```text
simtoolreal_fixed_goals.json
task scene assets
tool start pose
pretrained SimToolReal policy/runtime
```

Output:

```text
rollout/
rollout_summary.json
videos / frame logs when enabled
benchmark_extra_actor_summary.json when applicable
```

The rollout is execution evidence, not a source of oracle refs.

### Success Check And Manual Attribution

Input:

```text
rollout summary
task-specific evaluator outputs
videos and visual evidence
compile/tool-ref/scene-ref artifacts
```

Output:

```text
trial_result.json
aggregate summary.json / summary.csv / summary.md
manual failure attribution table
```

A trial succeeds only when the task outcome is achieved through the expected
functional tool interaction. Manual attribution uses:

```text
reference
compilation
execution
artifact
needs_review
```

## Scratch And Legacy Policy

Use `test000/` for interactive previews and short review runs. Use
`artifacts/test00*/` for named experiment batches. Historical outputs from
`test001` through `test007` are development evidence unless a reviewed report
explicitly promotes them.

Legacy registry tasks may remain for regression checks, but current paper
statistics should cite only `current_all_runnable` runs launched with the
current full E2E protocol.
