# Benchmark Expansion Candidate Matrix

Date: 2026-05-31

This document is the human-facing staging matrix for expanding the image-grounded
ToolFrame benchmark. It is not a result table and does not approve any new task
contract.

Metric reporting reference: `artifacts/tool_use_benchmark/evaluation_metrics.md`.
Detailed current contract summary: `artifacts/tool_use_benchmark/task_contracts.md`.

## Review Rules

- Keep method inputs inside the current boundary: public RGB-D scene evidence,
  task-generic VLM scene refs, RGB-D lifting, public/P3SAM/task-conditioned VLM
  tool refs, whole-tool SE(3) subgoals, SimToolReal fixed-goal rollout, and
  explicit task-level success checks.
- Do not use hidden evaluator target refs, oracle target axes, oracle
  trajectories, pre-rendered trace guides, `tool_family` or `task_family`
  fallback, primitive fallback, repair-loop fallback, automatic global-axis
  retries, or seed-specific shortcuts.
- Every new task must keep `success_review_status = pending_human_review` until
  the required predicates, diagnostic predicates, thresholds, tolerances, and
  allowed contact-point/motion-axis labels are reviewed.
- Smoke runs before review are debugging evidence only. Promotion requires an
  approved contract plus coherent evidence, compilation, rollout, and videos.
- Keep configs under `artifacts/tool_use_benchmark/<task_family>/<split>/` and
  runtime outputs under `artifacts/image_grounded/<task_family>/`.

## Current Release Scope

Current release scope is built-in SimToolReal tools and reviewed benchmark
target assets:

```text
simtoolreal/assets/urdf/dextoolbench/
simtoolreal/assets/urdf/parttoolgen_benchmark/
```

External ToolUse imports are omitted from the current public release. Old trace
guide assets are omitted from marker trace. If either category returns, it needs
a fresh candidate row and pending contract review.

## Current Side-Board Drafts

All side-board rows share `side_left_vertical_whiteboard_soft02`: a left-side
vertical blank whiteboard, `whiteboard_front` public evidence, and `soft02`
lighting.

| Candidate | Task target asset | Task language | Current status | Contract status |
|---|---|---|---|---|
| `marker_trace/write_l` | Blank side-board whiteboard surface; produced ink is post-rollout evidence only. | Use this marker to write the letter L on the whiteboard. | Current hard side-board task candidate. Uses generic VLM start/optional-waypoint/end refs, marker-tip ink rendering, and VLM judge success from produced ink legibility. | Provisional: task scope accepted, marker-tip labels and publish smoke pending. |
| `marker_trace/write_c` | Same blank side-board whiteboard surface. | Use this marker to write the letter C on the whiteboard. | Current hard curved-writing candidate. | Provisional: same as `write_l`. |
| `eraser_wipe/side_whiteboard_hdu_erase` | Visual-only HDU sticker/decal target on the side whiteboard. | Use this eraser to erase the whole HDU pattern on the whiteboard. | Draft side-board wipe task. | Draft: proposed erased-area threshold needs final review. |
| `hammer_strike/side_nail_strike` | Reused visible strike target on side board. | Use this hammer to hit the visible nail on the side board. | Draft side-board strike task. | Pending side-board nail strike success contract. |
| `screwdriver_insert/side_socket_insert` | Reused visible side socket. | Use this screwdriver to insert its tip into the side socket. | Draft side-board insertion task. | Pending side-board socket insertion success contract. |

## SimToolReal Segmented Tool Inventory

Inventory sources:

```text
simtoolreal/assets/urdf/dextoolbench/
bin/toolframe inventory --compact
```

The compact P3SAM catalog covers these ready tool families:

| Family | Tools | Existing task coverage | Status |
|---|---|---|---|
| `brush` | `blue_brush`, `red_brush` | Brush sweep contracts exist for red brush; blue brush remains a candidate variant. | Current/provisional. |
| `eraser` | `flat_eraser`, `handle_eraser` | Flat-eraser tabletop wipe is reviewed; side-board HDU erase is draft; handle eraser remains candidate. | Mixed reviewed/draft. |
| `hammer` | `claw_hammer`, `mallet_hammer` | Claw-hammer down strike exists; side-board nail strike is draft; mallet remains candidate. | Provisional/draft. |
| `marker` | `sharpie_marker`, `staples_marker` | Staples-marker `write_l` and `write_c` are hard side-board candidates; sharpie was not used because the visible asset appeared capped. | Provisional hard task. |
| `screwdriver` | `long_screwdriver`, `short_screwdriver` | Long-screwdriver tabletop insertion exists; side socket is draft; short screwdriver remains candidate. | Provisional/draft. |
| `spatula` | `flat_spatula`, `spoon_spatula` | No active approved task. | Deferred. |

## Candidate Task Matrix

Readiness legend:

| Status | Meaning |
|---|---|
| `current_reference` | Active/current task useful as a pattern, not a new expansion row. |
| `review_next` | Good next contract-review candidate; no large stats before review. |
| `smoke_after_review` | Needs contract review, then a small public evidence/compile/rollout smoke. |
| `needs_assets` | Needs a target object, public evidence asset, tool-ref evidence, or config scaffold. |
| `blocked` | Missing task contract, invalid/unclear mesh, or ambiguous success semantics. |

| Candidate | Source assets | Public scene evidence | Task language draft | VLM scene refs to require/review | Tool refs to require/review | Compiler constraint type | Config/output policy | Draft success-check for human review | Diagnostic-only candidates | Readiness |
|---|---|---|---|---|---|---|---|---|---|---|
| `brush_sweep/red_forward_left` | `red_brush`; visible movable block | Table-top RGB-D with visible block and brush | Sweep the target block in the requested direction using the brush. | Target start point, optional end point, `scene.direction`. | Brush contact point/motion-axis refs; red-brush labels pending. | Planar contact-and-sweep whole-tool SE(3) path. | Existing configs under `brush_sweep/`; outputs under `artifacts/image_grounded/brush_sweep/`. | Target displacement along requested direction; thresholds are config fields and should be reviewed before publication. | Tracking error, endpoint error, contact path margin. | `current_reference` |
| `brush_sweep/blue_forward_left` | `blue_brush`; visible movable block | Same table-top setup with blue brush | Sweep the target block in the requested direction using the blue brush. | Same as red brush. | Blue-brush refs pending under `configs/tool_refs/blue_brush`. | Same sweep path. | Config scaffolds exist under `brush_sweep/blue_forward` and `brush_sweep/blue_left`; outputs under `artifacts/image_grounded/brush_sweep/`. | Same predicate family as red brush. | Same as red brush. | `smoke_after_review` |
| `eraser_wipe/flat_tabletop_wipe` | `flat_eraser`; visible raised white HDU tabletop board | Table-top RGB-D with visible raised white HDU board and eraser | Use this eraser to erase the whole HDU pattern on the table. | Start/end refs on visible board surface, optional `scene.direction`. | Flat-eraser contact refs approved; `handle_eraser` refs are generated as an optional morphology candidate. | Surface contact use-stage from VLM-marked point refs over the raised board target surface. | Existing config under `eraser_wipe/tabletop_wipe/`; outputs under `artifacts/image_grounded/eraser_wipe/`. | Reviewed contact-coverage success: flat-eraser bottom footprint covers >=30% of target long-axis extent. | Visual disappearance, tracking, semantic execution. | `current_reference` |
| `eraser_wipe/side_whiteboard_hdu_erase` | `flat_eraser`; visual-only HDU pattern; shared whiteboard | `whiteboard_front` RGB-D with visible HDU pattern | Use this eraser to erase the whole HDU pattern on the whiteboard. | Public refs over the visible HDU pattern. | Flat-eraser refs approved for tabletop; side-board reuse pending review. | Whiteboard surface contact wipe. | Draft config should live under `eraser_wipe/side_whiteboard_hdu_erase/`; outputs under `artifacts/image_grounded/eraser_wipe/`. | Proposed erased area fraction >=70%; pending final review. | Contact continuity, tracking, visual overlay. | `review_next` |
| `hammer_strike/down` | `claw_hammer`; visible nail | Table-top RGB-D with visible nail | Hit the visible nail with the hammer. | Start point/contact point on nail; VLM strike direction. | Hammer labels pending. | Point contact strike path. | Existing config under `hammer_strike/down/`; outputs under `artifacts/image_grounded/hammer_strike/`. | Semantic contact/strike direction; target displacement diagnostic. | Tracking, rotation error, target motion. | `current_reference` |
| `hammer_strike/down_mallet` | `mallet_hammer`; visible nail | Same table-top setup with visible nail | Hit the visible nail with the mallet hammer. | Same as claw-hammer down strike. | Mallet refs pending under `configs/tool_refs/mallet_hammer`. | Same point contact strike path. | Config scaffold exists under `hammer_strike/down_mallet`; outputs under `artifacts/image_grounded/hammer_strike/`. | Same semantic contact/strike-direction predicate as claw-hammer down strike. | Tracking, rotation error, target motion. | `smoke_after_review` |
| `hammer_strike/side_nail_strike` | `claw_hammer`; side-board nail target | `whiteboard_front` RGB-D with visible nail | Use this hammer to hit the visible nail on the side board. | Public nail contact refs and direction refs. | Hammer labels pending. | Side-board point contact strike. | Draft config should live under `hammer_strike/side_nail_strike/`; outputs under `artifacts/image_grounded/hammer_strike/`. | Pending side-board strike contract. | Tracking, target motion, semantic contact. | `review_next` |
| `screwdriver_insert/wide_socket` | `long_screwdriver`; visible wide socket | Table-top RGB-D with public socket | Insert the screwdriver tip into the visible socket. | Socket opening start point and public insertion direction. | Long-screwdriver labels pending. | Whole-tool terminal inside-socket path. | Existing config under `screwdriver_insert/wide_socket/`; outputs under `artifacts/image_grounded/screwdriver_insert/`. | Compiled terminal socket goal plus measured tip radial alignment and inside-socket depth. | Terminal endpoint error, strict tracking. | `current_reference` |
| `screwdriver_insert/side_socket_insert` | `long_screwdriver`; side-board socket | `whiteboard_front` RGB-D with visible side socket | Use this screwdriver to insert its tip into the side socket. | Socket opening and public insertion direction/default axis. | Long-screwdriver labels pending. | Side-board terminal inside-socket path. | Draft config should live under `screwdriver_insert/side_socket_insert/`; outputs under `artifacts/image_grounded/screwdriver_insert/`. | Pending side-socket insertion contract. | Radial error, inside-socket depth, tracking. | `review_next` |
| `marker_trace/write_l` | `staples_marker`; blank side whiteboard | `whiteboard_front` RGB-D of blank whiteboard | Use this marker to write the letter L on the whiteboard. | Start point, optional path_points, end point on blank surface. | Marker-tip labels pending. | Ordered marker-tip path through lifted refs. | Draft contract/config under `marker_trace/write_l/`; outputs under `artifacts/image_grounded/marker_trace/`. | VLM judge recognizability of produced marker-tip ink as L. | Contact sample count, ink length, segment count, tracking. | `review_next` |
| `marker_trace/write_c` | `staples_marker`; blank side whiteboard | Same as `write_l` | Use this marker to write the letter C on the whiteboard. | Start point, optional path_points, end point on blank surface. | Marker-tip labels pending. | Ordered curved marker-tip path. | Draft contract/config under `marker_trace/write_c/`; outputs under `artifacts/image_grounded/marker_trace/`. | VLM judge recognizability of produced marker-tip ink as C. | Same as `write_l`. | `review_next` |
| `spatula_lift_or_serve/flat` | `flat_spatula`; visible thin object/plate target | Needs public RGB-D scene | Slide the spatula under the visible object, then lift or serve it. | Object edge/start point, target support/end area. | Spatula blade/contact labels pending. | Wedge-under plus lift/transport subgoals. | Proposed config under a future spatula task family. | Pending object support/lift/transfer contract. | Slip, lift height, final pose. | `needs_assets` |
| `spatula_lift_or_scoop/spoon` | `spoon_spatula`; scoopable object/container | Needs public RGB-D scene | Scoop or serve the visible target using the spoon spatula. | Object start point, optional container/end point. | Spoon bowl/contact labels pending. | Scoop/transport path. | Proposed config under a future spatula task family. | Pending scoop/transfer contract. | Retention, spill, tracking. | `needs_assets` |

## Omitted External Asset Notes

External ToolUse-derived assets are not part of the current public benchmark
release. Do not list them as ready candidate rows, do not rely on converted
external URDF paths, and do not mix their targets with the SimToolReal current
benchmark unless a future asset-release decision approves that scope.

## Success-Check Review Template

For each promoted candidate, create or update a review record with these fields:

| Field | Required human decision |
|---|---|
| `visible_public_scene_refs` | Which VLM-visible target, start, end, direction, path, or fixture refs are allowed. |
| `tool_refs_and_grounding_source` | Which P3SAM/VLM part and face refs are allowable for contact point and motion-axis semantics. |
| `required_task_level_predicates` | Which predicates can make a seed pass. |
| `diagnostic_only_predicates` | Which metrics can explain failures but cannot make a seed pass by themselves. |
| `thresholds_tolerances` | Numeric pass/fail tolerances, if any. Leave pending until reviewed. |
| `allowed_labels` | Functional part, face, point, and motion-axis labels. Leave pending until reviewed. |
| `smoke_gate` | Evidence that public RGB-D, VLM refs, lifting, tool refs, compilation, rollout, and videos are coherent. |

## Suggested Review Order

1. Finalize marker `write_l`/`write_c` label review and publish smoke policy.
2. Review side-board HDU eraser success checker and threshold.
3. Review side-board hammer and screwdriver contracts only after public refs and environment separation are clean.
4. Expand table-top variants such as blue brush, handle eraser, short screwdriver, and mallet after label review.
5. Keep spatula deferred until object state and success semantics are clear.
