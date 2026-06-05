# Marker Trace Benchmark

This directory defines the current marker-writing side-board hard tasks. They
are kept as difficult benchmarks: the full image-grounded chain has a formal
one-command runner, but low-level execution can still produce fragmented ink.

## Current Tasks

```text
write_l/contract_draft.json
write_l/benchmark_config.draft.json
write_c/contract_draft.json
write_c/benchmark_config.draft.json
```

| Task | Status | Success-check status |
|---|---|---|
| `marker_trace/write_l` | current/provisional hard task | VLM judge over produced marker-tip ink is the intended gate; marker labels and publish smoke remain pending. |
| `marker_trace/write_c` | current/provisional hard task | Same gate as `write_l`, with curved path refs and higher execution difficulty. |

Both tasks use the SimToolReal `staples_marker` and the shared left-side
vertical blank whiteboard environment:

```text
urdf/parttoolgen_benchmark/marker_trace_whiteboard/left_side_vertical_whiteboard_002x0475x101.urdf
```

For side-task public scene evidence, use the renderer's `whiteboard_front`
grounding camera with the canonical side-board `soft02` lighting. Topdown images
are layout sanity checks only.

## Runner

The marker tasks are available through the unified benchmark launcher:

```bash
bin/toolframe benchmark \
  --suite current_side_board \
  --trials 1 \
  --seed-start 0 \
  --device cuda:0 \
  --capture-video-every 1
```

The task-specific runner is:

```text
parttoolgen/src/parttoolgen/cli/runners/run_image_grounded_marker_trace_eval.py
```

This runner is an internal implementation detail used by the benchmark
registry. It regenerates public `whiteboard_front` RGB-D evidence, VLM scene refs, lifted
3D refs, compiled whole-tool SE(3) subgoals, SimToolReal rollout videos,
marker-tip ink evidence, and VLM-judge results under:

```text
artifacts/image_grounded/marker_trace/
```

## Scene Ref Policy

The shared task-generic VLM scene-ref interface applies:

```text
start_point
optional waypoint points
end_point
scene.direction when applicable
```

These scene refs are task intent and compilation refs, not a robot trajectory.
There is no pre-rendered trace, guide path, or visible target stroke in the
initial whiteboard scene.

## Success Policy

Marker-writing success is decided by a VLM judge from produced marker-tip ink
evidence. The VLM judge consumes publishable per-run artifacts such as:

```text
scene_grounding/scene_keypoints_overlay.png
ink_evidence/ink_trace_overlay.png
rollout/rollout_side_board_front_render.mp4
rollout/rollout_summary.json
ink_evidence/ink_trace_metrics.json
```

VLM judge labels are:

```text
success
failure
```

A success requires visible marker-tip ink on the whiteboard that the judge
recognizes as the requested letter, without severe off-surface artifacts or
non-tip ink artifacts.

## Diagnostic Values

Automatic values are diagnostic only unless a later reviewed contract promotes
them:

```text
pipeline stage status
policy_goal_passed
tracking_execution_passed
semantic_execution_passed
reached_goal_count / num_goals
marker-tip contact_sample_count
ink_segment_count
ink_length_m
side-board video availability
```

Ink evidence must come only from the reviewed `tool.contact_point` marker tip /
writing end. The renderer must not draw ink from the tool center, gripper,
handle, cap, shaft, or whole-tool pose alone. Pending marker labels require an
explicit debug flag and must not be reported as final benchmark evidence.

Run outputs, VLM evidence, tool refs, summaries, and videos should live under:

```text
artifacts/image_grounded/marker_trace/
```
