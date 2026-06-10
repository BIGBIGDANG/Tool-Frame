# Task Contract Summary

Date: 2026-05-31

This file summarizes the current benchmark contracts in a way a new reader can
follow without opening every JSON file. It is a contract index, not a result
report. If a field is marked pending, it must not be reported as approved.

## Status Legend

```text
reviewed            human-reviewed enough for the named success predicate
provisional         runnable or accepted as a candidate, but labels, thresholds, smoke, or statistics are pending
draft               documented proposal only
stress              diagnostic harder variant, not the headline benchmark split
diagnostic_only     useful for attribution, not a pass/fail gate
```

## Table-Top Tasks

| Task | Environment | Natural task language | Scene refs | Tool refs | Compiler constraint | Required success | Diagnostics | Status |
|---|---|---|---|---|---|---|---|---|
| `brush_sweep/off_table_red`, `brush_sweep/off_table_blue` | `tabletop_wood_table` | Sweep the target block off the table using the brush; for `off_table_red`, use the side of the brush bristles to sweep the object. | Public block start/end refs and `scene.direction` from topdown RGB-D; no hidden target-zone center. | `tool.contact_point`, `tool.motion_axis`; red/blue brush refs from public tool-ref evidence. | Planar contact sweep with `align -> approach -> action`, where approach starts behind the first contact point along negative VLM motion direction. Table-bbox clearance is handled during subgoal generation, not as a post-hoc contact-point rewrite. | Final dynamic block center leaves the table bounds or drops below the tabletop. | Tracking, contact-path details, directional displacement, table-bbox clearance report. | Reviewed/current smoke-passed brush split; 20-seed statistics must be regenerated after the 2026-06-06 compiler update. |
| `brush_sweep/red_zone` | `tabletop_wood_table` | Retired target-zone sweep. | Public block refs and visual target-zone refs. | Same brush refs. | Same sweep compiler. | Retired red-zone occupancy predicate, retained only for attribution/regression diagnostics. | Tracking, final distance to zone, contact-path details. | Legacy/retired. |
| `brush_sweep/forward`, `brush_sweep/left` | `tabletop_wood_table` | Directional brush-sweep diagnostics. | Public target start/end refs and `scene.direction` from topdown RGB-D. | Same red-brush refs; labels pending. | Same sweep compiler. | Directional displacement, retained only for attribution/regression diagnostics. | Tracking, final distance to planned endpoint, contact-path details. | Diagnostic/legacy. |
| `eraser_wipe/tabletop_wipe` | `tabletop_wood_table` | Use this eraser to erase the whole HDU pattern on the table. | Public start/end refs on the visible raised white tabletop HDU board and `scene.direction` when applicable. | Flat-eraser contact part and faces approved in `tool_label_review/eraser_wipe__flat_eraser.json`; `handle_eraser` refs are available as an optional morphology candidate. | Surface contact use-stage from VLM-marked point refs over the raised board target surface. | Diagnostic path gate, lifted start/end refs on the visible board surface, and flat-eraser bottom contact-footprint coverage >= 30% of the target long-axis extent. | Visual disappearance, pose tracking, policy-goal pass, semantic execution pass, rotation error. | Reviewed/current. |
| `hammer_strike/down` | `tabletop_wood_table` | Hit the visible nail with the hammer. | Public start point and VLM motion direction for the point action; support normals are diagnostic/projection evidence only. | Claw-hammer striking face and motion-axis refs pending review. | Contact strike subgoals aligned to public VLM motion direction. | Semantic contact/strike-direction success; target displacement is not required. | Target displacement, path-gate report, strict tracking, rotation error. | Provisional. |
| `screwdriver_insert/wide_socket` | `tabletop_wood_table` | Insert the screwdriver tip into the visible socket. | Public socket opening start point and VLM insertion direction. | Long-screwdriver tip/contact and shaft-axis refs pending review. | Whole-tool terminal inside-socket goal plus insertion path into the public socket aperture. | Compiled terminal socket-goal check, semantic insertion path, measured tip radial alignment, measured inside-socket depth. | Terminal endpoint error, path-gate report, strict tracking, policy-goal pass. | Provisional headline split. |
| `screwdriver_insert/deep_socket` | `tabletop_wood_table` | Insert the screwdriver tip into the stricter socket. | Same family as wide socket. | Same long-screwdriver refs pending review. | Same insertion compiler with stricter aperture. | Same as wide socket, with stricter radial tolerance. | Same as wide socket. | Stress/provisional. |
| `screwdriver_insert/*_dense` | `tabletop_wood_table` | Optional dense-subgoal diagnostics for screwdriver insertion. | Same family as the corresponding non-dense split. | Same long-screwdriver refs pending review. | Same insertion compiler with tighter controller diagnostic thresholds. | Same required success family as the corresponding non-dense split. | Dense/step-size sensitivity only; excluded from default statistics. | Optional diagnostic/stress. |

## Side-Board Tasks

All side-board tasks use the shared `side_left_vertical_whiteboard_soft02`
environment from `references/environment_registry.md` and
`side_board/README.md`. The public evidence camera is `whiteboard_front`.

| Task | Target asset role | Natural task language | Scene refs | Tool refs | Compiler constraint | Required success | Diagnostics | Status |
|---|---|---|---|---|---|---|---|---|
| `marker_trace/write_l` | Blank writable whiteboard; produced ink is post-rollout evidence only. | Use this marker to write the letter L on the whiteboard. | Public `start_point`, optional path_points, and `end_point` on the whiteboard. | Marker-tip contact point/motion-axis refs pending review. | Ordered marker-tip path through lifted public refs. | VLM judge recognizes produced marker-tip ink as L. | Contact sample count, ink length, segment count, tracking, semantic execution. | Provisional hard task. |
| `marker_trace/write_c` | Blank writable whiteboard; produced ink is post-rollout evidence only. | Use this marker to write the letter C on the whiteboard. | Public `start_point`, optional path_points, and `end_point` on the whiteboard. | Marker-tip contact point/motion-axis refs pending review. | Ordered curved marker-tip path through lifted public refs. | VLM judge recognizes produced marker-tip ink as C. | Same as `write_l`. | Provisional hard task. |
| `eraser_wipe/side_whiteboard_hdu_erase` | Visual-only HDU sticker/decal on the side whiteboard. | Use this eraser to erase the whole HDU pattern on the whiteboard. | Public refs over the visible HDU pattern. | Flat-eraser functional contact refs are approved for tabletop; side-board reuse still needs contract review. | Surface wipe over the whiteboard pattern. | Draft proposal: erased area fraction from functional eraser contact, proposed threshold 70%. | Tracking, contact continuity, visual overlays. | Draft. |
| `hammer_strike/side_nail_strike` | Visible side-board nail/strike target. | Use this hammer to hit the visible nail on the side board. | Public start/end refs or contact point on the visible nail. | Hammer refs pending review. | Side-board contact strike aligned from public refs and environment geometry. | Pending side-board nail strike contract. | Tracking, semantic contact, target motion. | Draft. |
| `screwdriver_insert/side_socket_insert` | Visible side-board socket. | Use this screwdriver to insert its tip into the side socket. | Public socket opening and VLM/default insertion direction. | Long-screwdriver refs pending review. | Side-board terminal inside-socket goal plus insertion path into the public socket aperture. | Pending side-socket insertion contract. | Radial error, inside-socket depth, tracking. | Draft. |

## Marker Trace Contract Notes

Marker trace is intentionally retained as a hard side-board task. Its success
is not a path-template score, OCR score, or strict tracking score. The pass/fail
semantic gate is a VLM judge over produced marker-tip ink evidence. Ink may be
rendered only from reviewed marker-tip contact samples on the whiteboard.

The task scope was accepted by the human as a hard benchmark candidate. The
success check remains provisional until marker tool labels and a publishable
smoke are reviewed.

## Tool-Label Audit Boundary

Tool labels in `tool_label_review/` are evaluator-side audit metadata. They are
not method inputs, compiler-side selection rules, or task success gates. The
method chooses tool refs from public/P3SAM/VLM evidence; task pass/fail is
decided by executed effects such as displacement, insertion radial/depth,
surface coverage, semantic contact, or produced ink. Filled `allowed_*` lists
only support post-hoc diagnostics and failure attribution.

## External And Historical Assets

External ToolUse assets, old trace-guide assets, and historical scratch run
roots are not part of the current publishable benchmark contract set. If they
are revisited, add a new candidate row and mark every success-check field as
pending until reviewed.
