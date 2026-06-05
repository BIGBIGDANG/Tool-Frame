# Benchmark Evaluation Metrics Reference

Date: 2026-05-31

This document defines compact module-level metrics for image-grounded ToolFrame
benchmark summaries. It does not approve any task contract, threshold, label, or
success predicate by itself.

## Main Reporting Table

Use one row per task or split:

```text
task | contract_status | N | functional_grounding_rate | scene_grounding_rate | subgoal_validity_rate | policy_execution_rate | task_success_rate
```

Each rate is computed over trials or seeds:

```text
rate = number_of_trials_that_pass_the_metric / number_of_trials_evaluated
```

If a trial is missing required artifacts, count it as not passed for the first
module whose evidence is missing. Keep detailed reasons in per-trial diagnostics.

## Contract Status

Use one of these values in summaries:

```text
reviewed
provisional
draft
stress
blocked
```

A pending tool-label review, pending threshold, or missing publish smoke should
keep the task `provisional` or `draft`, even if the chain can run.

## Module Metrics

### `functional_grounding_rate`

Functional grounding measures the tool-side grounding module. A trial passes
when the tool-side functional refs required by the task contract are present,
public, task-conditioned, and reviewable:

```text
tool.contact_point
tool.motion_axis
tool.motion_axis
```

For final benchmark statistics, allowed tool part, face, contact-point, and
motion-axis labels must come from the human-reviewed task contract. Pending
labels may be reported for smoke/debug runs, but they are not approved final
benchmark evidence.

### `scene_grounding_rate`

Scene grounding measures VLM scene marking and RGB-D lifting. A trial passes
when public RGB-D evidence supports the task-required scene refs and the lifted
3D refs are valid for compilation. Examples include start points, waypoints,
end points, target surfaces, and public `scene.direction` when applicable.

This module must not use hidden evaluator targets, oracle axes, oracle
trajectories, simulator target centers, or seed-specific shortcuts.

### `subgoal_validity_rate`

Subgoal validity measures the compiler module. A trial passes when generated
whole-tool SE(3) subgoals are task-consistent and should complete the task if
executed accurately. The task contract should define the exact check. Common
requirements include:

```text
ordered subgoals exist
tool.contact_point is driven toward the lifted scene refs
tool.motion_axis is aligned to the scene direction before and during contact
tool.motion_axis satisfies the scene.direction constraint when applicable
the final subgoal reaches the task-relevant end/contract region
```

Path, clearance, and collision checks are diagnostics or gates only when the
reviewed task contract explicitly says so.

### `policy_execution_rate`

Policy execution measures whether the fixed SimToolReal policy can execute the
compiled subgoals well enough to produce required rollout evidence. Strict
whole-tool pose tracking is diagnostic by default.

### `task_success_rate`

Task success is the final task-specific benchmark metric. A trial passes only
when the reviewed or provisional task success contract says it passes. The four
module rates above explain where failures occur.

## Failure Attribution

Assign each failed trial to the first module that lacks valid evidence:

```text
functional_grounding_failure
scene_grounding_failure
subgoal_validity_failure
policy_execution_failure
task_success_failure
needs_review
```

Use `needs_review` when artifact evidence is ambiguous or the task contract has
not approved required labels, thresholds, or predicates.

## Optional Diagnostics

Keep detailed diagnostics out of the main table unless a paper figure or debug
report needs them:

```text
tracking_pass_rate
tracking_task_disagreement_rate
success_despite_tracking_failure_rate
failure_despite_tracking_success_rate
primary_task_effect_p50
primary_task_effect_p90
```

Primary effect summaries by task:

```text
brush_sweep: final target-block xy occupancy inside the red outline target zone
hammer_strike: semantic contact and strike-direction consistency
eraser_wipe: semantic wipe/contact coverage
screwdriver_insert: tip radial error and measured insertion depth
marker_trace: VLM-judged recognizability of produced marker-tip ink
side_whiteboard_hdu_erase: erased HDU area fraction from functional eraser contact
```

Do not invent thresholds. Thresholds must be approved in the task-specific
success contract before final benchmark statistics.
