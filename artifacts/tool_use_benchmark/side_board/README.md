# Side-Board Environment Notes

This directory records shared human-facing notes for side-board task drafts.
It is not a runtime-output directory.

## Fixed Shared Environment

All current side-board tasks use the same shared environment unless the human
explicitly approves a change:

```text
environment id = side_left_vertical_whiteboard_soft02
asset = urdf/parttoolgen_benchmark/marker_trace_whiteboard/left_side_vertical_whiteboard_002x0475x101.urdf
whiteboard_center_world = [0.38, 0.0, 0.505]
public_whiteboard_surface_normal_world = [-1.0, 0.0, 0.0]
public evidence camera = whiteboard_front
public evidence lighting = soft02 = --light-preset soft --soft-light-indices 0,2
```

The whiteboard is a shared environment asset. Task targets stay separate:
marker trace uses a blank writable surface, eraser wipe uses a visual-only HDU
pattern, hammer strike uses a visible side strike target, and screwdriver insert
uses a visible side socket.

Environment metadata and task metadata must stay separated. The shared
environment may expose calibrated public facts such as `whiteboard_center_world`
and `public_whiteboard_surface_normal_world`. Task-specific public action-axis metadata fields are not
allowed in method-facing task JSON. If a side task needs an action direction,
the VLM marks public scene refs from natural task language and RGB-D evidence;
the compiler derives `scene.direction` and aligns `tool.motion_axis` to it.

## Current Side-Board Tasks

| Task | Status | Contract status | Notes |
|---|---|---|---|
| `marker_trace/write_l` | current/provisional hard task | Formal runner available; VLM-judge success accepted as the intended gate, but marker labels and publish smoke remain pending. | Blank initial whiteboard; produced ink only from marker-tip contact. |
| `marker_trace/write_c` | current/provisional hard task | Formal runner available; same gate as `write_l`, with curved path refs. | This is the hard curved-writing split. |
| `eraser_wipe/side_whiteboard_hdu_erase` | draft/smoke_review | Formal runner available; proposed 70% erased-area threshold needs publish smoke review. | Visual-only HDU pattern; no collision or hidden target oracle. |
| `hammer_strike/side_nail_strike` | draft | Pending side-board nail strike success contract. | Do not use color-target wording or task-specific public action-axis metadata. |
| `screwdriver_insert/side_socket_insert` | draft | Pending side-board socket insertion success contract. | Must stay separate from tabletop insertion configs. |

## Output Policy

This directory must not contain run outputs, videos, rendered RGB-D evidence,
rollout summaries, or generated preview images. Publishable run evidence should
be regenerated under `artifacts/image_grounded/<task_family>/` after a task
contract is approved. Local scratch runs may exist outside this tree, but they
are not benchmark configs and should not be referenced here.

## Success-Check Status

Marker letter writing uses VLM-judged marker-tip ink legibility as the semantic
success gate. Rollout/contact metrics are diagnostic.

The side-board HDU eraser task has a human-proposed headline threshold: erase at
least 70% of the visible HDU pattern. The checker measures covered HDU area from
flat-eraser `tool.contact_point` / contact-face footprint samples on the
whiteboard surface. Whole-tool root pose alone, sticker collision, hidden
evaluator refs, oracle trajectories, `tool_family` fallback, and `task_family`
fallback are out of scope.

Side hammer and side screwdriver still need reviewed task-specific success
contracts before statistics.
