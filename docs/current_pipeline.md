# Current Pipeline

This is the clean image-grounded ToolFrame path. It keeps the working whole-tool
solver pieces in the current repo, but removes benchmark-only target refs and
tool-family fallbacks from the method path.

```bash
cd <your clone>/ToolFrame_6.4
bin/toolframe env-check --mode core
```

## One-Command Benchmarks

The current runnable brush mainline is the red-brush block-sweep benchmark. Each seed
generates a true SimToolReal RGB-D scene, asks the VLM to identify the target
object and mark start/end refs on the overhead image. In the current
scaffold these refs are represented by VLM-marked start/end pixels plus
the scene direction. The runner lifts those pixels to 3D, merges public
red-brush tool-face refs,
compiles whole-tool SE(3) subgoals, and runs SimToolReal fixed-goal rollout.

```bash
bin/toolframe benchmark --task brush_sweep/red_zone
```

Legacy directional brush variants are kept only for regression attribution:

```bash
bin/toolframe benchmark --suite legacy_directional_diagnostics --preflight-only
```

The current hammer smoke uses the same image-grounded chain with public
`claw_hammer` tool refs. The shared scene-ref interface provides the public
start point and VLM motion direction for this point-action task. Support
surface normals are retained only as projection/diagnostic evidence.

```bash
bin/toolframe benchmark --task hammer_strike/down
```

The current tabletop eraser benchmark uses the `flat_eraser` and a visible
raised white tabletop board containing the blue HDU text. The board top is the
target surface and includes the text footprint. Deterministic seeds spread both
eraser start pose/yaw and board pose/yaw over the table-safe workspace. The tool
is sampled at a random relative angle and non-overlapping distance from the board, with table-safety
rejection. The VLM prompt stays task-generic and natural: it marks public scene
refs from the RGB-D evidence without task-specific endpoint hints. The VLM owns
the start point, optional path points, end point, and scene.direction from the
RGB-D evidence; public flat-eraser refs come from task-conditioned P3SAM/VLM
part then face grounding.

```bash
bin/toolframe benchmark --task eraser_wipe/tabletop_wipe
```

The current screwdriver expansion task is `screwdriver_insert/wide_socket`.
It uses the registered `long_screwdriver`, a public movable tabletop guide
socket with a visible open top and four collision walls around the aperture.
The VLM grounds the socket center plus a VLM-decided insertion scene.direction,
and task-conditioned P3SAM/VLM screwdriver tip/axis refs provide the tool side.
This is a fresh image-grounded insertion task, not the legacy SimToolReal
screwdriver spin task. The stricter `deep_socket` configs remain method-stress
smokes; `wide_socket` is the current evaluable insertion variant.

```bash
bin/toolframe benchmark --task screwdriver_insert/wide_socket
```

Screwdriver insertion is currently best treated as a smoke or method stress
test. The chain can generate semantic insertion evidence, but strict tip
alignment/depth pass is not yet stable enough to promote to headline benchmark
statistics without another human-reviewed contract and review pass.

The current side-board expansion uses the shared left-side vertical whiteboard
environment documented in the `isaac-environment` skill registry. Side-board
tasks use `whiteboard_front` RGB-D evidence and the fixed `soft02` lighting
(`--light-preset soft --soft-light-indices 0,2`). The current side-board draft
tasks are marker letter writing on a blank whiteboard, HDU handwriting/sticker
erasure, side-board nail hammer strike, and side socket screwdriver insertion.
These are still smoke/review tasks until their task-level success contracts and
tool labels are approved. The shared side-board environment may expose public
calibration facts such as the whiteboard surface normal; task-specific action
axes must not be embedded in method-facing task JSON.

The drivers write:

- `run_config.json`: fixed camera/compiler/rollout parameters;
- `<task_id>/vlm_scene_keypoints/topdown_grounding_overlay.png`: VLM evidence;
- `<task_id>/lifted_program/lifted_refs.json`: lifted 3D start/end refs,
  optional waypoint refs, the public scene direction, and `direction_decision`;
- `<task_id>/compile/simtoolreal_fixed_goals.json`: executable goals;
- `<task_id>/rollout/rollout_summary.json`: rollout tracking result;
- `<task_id>/rollout/rollout_render.mp4` and
  `<task_id>/rollout/rollout_topdown_render.mp4`: rollout videos when capture
  is enabled;
- `<task_id>/visual_evidence/manifest.json`: per-seed evidence index. Eraser
  seed benchmarks use manifest-only references so large RGB-D, compile, and
  video artifacts are not duplicated inside `visual_evidence/`;
- `summary.json` and `summary.md`: pass rates and per-seed metrics.

Benchmark summary tables should use the compact module metrics defined in
`artifacts/tool_use_benchmark/evaluation_metrics.md`: functional grounding,
scene grounding, subgoal validity, policy execution, and task success. Strict
whole-tool tracking is a diagnostic by default, not a headline success metric,
unless a human-reviewed task contract explicitly requires it.

Scene direction is explicit evidence. The VLM normally returns a visible
`action_direction`; when a point/depth action has no reliable visible endpoint,
it may instead select one of the public support-surface default-axis candidates:
`support_surface_normal` or `opposite_support_surface_normal`. The lifter writes
that selected candidate into `scene.direction` and records the source. The
compiler does not infer a hidden world-axis fallback.

Brush summaries use the red-zone occupancy success check: the dynamic block's
final xy center must lie inside the visible red outline target zone. Strict
tracking and directional displacement are diagnostics. Hammer summaries use the
VLM target/axis, path-gate report, and semantic contact gates; the nail can be
fixed or hard to visibly displace, so strike success is judged from contact
evidence rather than block-style task displacement. Eraser summaries use
`eraser_wipe_contact_coverage`: path-gate report, lifted start/end refs on the
visible tabletop HDU board, and rollout samples showing the flat-eraser bottom
contact footprint covering at least 30% of the target long-axis extent. Visual
mark disappearance, strict tracking, policy-goal pass, and semantic execution pass are
diagnostics.
Screwdriver summaries use `screwdriver_insert_tip_path`: path-gate report,
rollout semantic insertion path, measured tip XY/radial alignment to the
VLM-grounded socket center, and measured inside-socket depth along the public
insertion axis. The terminal endpoint error is diagnostic; it is not the
radial alignment metric. Tracking and policy-goal pass remain diagnostics.

Task success checks are benchmark contracts for reporting and statistics. They
are not hidden method inputs, automatic training signals, or a place to encode
seed-specific repairs. A new task can be used for large statistics only after a
human-reviewed contract defines required predicates, diagnostic-only predicates,
tool labels, and thresholds. Smoke runs before that review are debugging
evidence, not final pass rates.

The scene-grounding prompt is task-generic. It always asks the VLM to decide
whether `start_point`, a distinct `end_point`, sparse `waypoints`, and
`scene.direction` are applicable from the task language and public image
evidence. The runner does not pre-label a task as contact-only or decide in
advance whether a 3D direction exists. If a task requires a use-stage endpoint
and the VLM returns waypoints but no `end_point`, lifting treats the final
waypoint as the endpoint and keeps earlier waypoints as ordered path refs. Task
contracts may later require or reject these fields during validation, but the
VLM selection/marking stage makes the semantic judgment.

The next benchmark expansion pass should start from built-in SimToolReal
segmented tools under `simtoolreal/assets/urdf/dextoolbench/` plus the local
P3SAM segmented-tool catalog. Imported external ToolUse assets are intentionally
omitted from the current public release; revisit them only through a separate
license, asset, and benchmark-contract review. The output of the expansion pass
should be a candidate matrix with task language, public scene evidence, VLM
scene refs, tool refs, compiler constraint type, runner/config/output paths,
success contract, diagnostics, and readiness status. Do not promote a
candidate to benchmark statistics until the task contract and tool labels have
been human-reviewed.

Failure attribution is reported as coarse module labels, with detailed reasons
kept under per-layer diagnostics. The primary labels are: `success`,
`vlm_tool_selection_error`, `vlm_scene_marking_error`, `rgbd_lifting_error`,
`compiler_subgoal_error`, `policy_tracking_failure`, and `needs_review`. The
attribution code should only use a module label when the artifact evidence can
support it; ambiguous cases stay `needs_review` rather than being forced into a
specific module. Tool-side labels are task contracts kept in evaluator artifacts
only, not method inputs. These labels must be human-reviewed against the
segmented tool, face evidence, and task semantics before use; pending contracts
record selected refs for inspection but do not treat any part or face as ground
truth. The human-review staging files live under
`artifacts/tool_use_benchmark/tool_label_review/`.
The currently approved eraser contract is: functional part marker `4`; contact
faces are the non-internal faces among `F3`/`F4`; other non-internal faces may
serve as motion-axis refs. Do not invent equivalent labels for brush, hammer,
screwdriver, marker, knife, or any newly introduced tool without the same human
review.
Tool-ref evidence generation marks and removes faces that are internal to the
tool before the VLM contact/motion-axis prompt, so an occluded or tool-internal
segmentation face is not a selectable ref.

Tool-ref and presentation evidence work, including hammer transparent PPT
exports, lives under `artifacts/image_grounded/tool_ref_grounding/`. Candidate
figures, selected-part bbox overviews, selected-part face highlights, and PPT
crops should preserve the 5.15-style raster evidence; do not replace them with
simplified local mesh/SVG redraws. F markers and highlighted bbox faces are
still required for VLM contact/motion-axis selection, but they should appear
only after the part is selected, alongside structured marker tables.
Per-face/contact-choice figures such as
`contact_point_choices/candidate_*_F*_face.png` must not be generated. These
evidence exports support inspection and prompting, but they do not change the
method boundary above.

Executable target refs come only from VLM pixels lifted through RGB-D. Simulator
target boxes/centers are private benchmark data and are not used to snap,
replace, or face-center the VLM-marked start point before compilation.

The current Tool-Scene Constraint IR records public start/end refs plus the
scene direction. The compiler consumes those refs together with the tool-side
functional point and `tool.motion_axis`, then emits backend fixed-goal fields
for SimToolReal without treating those backend fields as task primitives.
If the VLM marks a visible `end_point`, lifting derives both the use direction
and metric action distance from RGB-D projection. If no distinct visible
endpoint exists for a contact-only task, the lifted use endpoint defaults to the start point, records
`action_distance_m = 0`, and preserves only the grounded scene direction.
For screwdriver insertion, the VLM provides the public scene direction and the
runner converts that axis plus the benchmark insertion-depth contract into the
executable end point; no hidden socket center or legacy spin trajectory is used as a
method input.

## Canonical Workflow Contract

Use this sequence for every active tabletop and side-board task:

```text
toolframe_task_spec_v1 or normalized registry benchmark config
  -> spec validation and execution plan
  -> generated method-facing task.json/method_task.json
  -> SimToolReal RGB-D scene evidence
  -> generic VLM scene grounding prompt
  -> RGB-D lifting into scene refs
  -> task-conditioned public tool refs
  -> constraint program with explicit stage refs
  -> whole-tool SE(3) compiler
  -> SimToolReal fixed-goal rollout
  -> task-specific success check and failure attribution
```

The public entrypoint for the spec layer is:

```bash
bin/toolframe run-spec --task brush_sweep/red_zone --preflight-only
bin/toolframe run-spec --spec artifacts/tool_use_benchmark/<family>/<split>/task_spec.json
```

The VLM scene-grounding prompt is shared across tasks. Task-specific content
belongs in task language, public candidate refs, and the task-level success
contract. Do not create a new runner path by adding hidden target refs,
tool/task-family switches, primitive fallbacks, seed-specific corrections, or
automatic global-axis retries.

Large statistics require a reviewed success contract. A smoke run is useful for
debugging, but it is not a headline benchmark result until its required
predicates, diagnostic-only predicates, thresholds, and tool labels have been
checked with a human reviewer.

## Artifact Retention

Keep benchmark configs in `artifacts/tool_use_benchmark/` and runtime outputs
in `artifacts/image_grounded/<task_family>/`. Current mainline outputs are:

```text
brush_sweep/red_zone_unified_prompt_eval/
hammer_strike/down_unified_prompt_eval/
eraser_wipe/tabletop_wipe_seed20_current_success_2500steps_original_rgb/
screwdriver_insert/wide_socket_seed20_current_success_eval/
marker_trace/path_whiteboard_preview_staples_seed_000/  # shared side-board environment preview, not a path task
vlm_prompt_unified_scene_grounding_smoke_v2/
```

During interactive review, the human may ask to keep temporary smoke outputs
under `test000/` instead of `artifacts/image_grounded/`. Treat `test000/` as a
scratch review area, not a benchmark config, method input, or publishable result
root. Publish-facing docs and configs should not cite `test000/` paths as
canonical evidence.

Older smoke/debug roots can be removed after their summaries are no longer
needed for comparison. Never delete an output root while its runner or rollout
process is active. Per-trial `visual_evidence/` directories should be
manifest-only unless a task explicitly needs copied static review assets.

The current benchmark fixes the brush close to the robot arm and randomizes the
target block position. This intentionally isolates image-grounded scene-ref
selection from arbitrary grasp stabilization. For direction-specific smoke tests,
`--start-orientation-mode perpendicular_motion` yaws the brush so its long axis
is perpendicular to the requested motion axis and the handle side faces the
robot arm when possible. The compiler inserts the shared `grasp_lift` execution
phase before the image-grounded tool-use path. This is only a
grasp-stabilization transition: later alignment/contact stages may descend to
their solved contact height as long as all tool-part OBB corners stay above the
table.

## Fixed Mainline Parameters

The drivers fix the settings that are currently known to work:

```text
scene evidence:
  true SimToolReal RGB-D
  robot-forward topdown camera for tabletop tasks
  whiteboard_front camera for side-board tasks
  original SimToolReal RGB for VLM grounding
  tabletop tasks use soft01 SimToolReal lighting, shared by evidence and rollout
  side-board tasks use soft02 lighting, shared by evidence and rollout
  other SimToolReal visual settings stay native

scene lifting:
  depth_sample_policy = nearest
  action_direction_z_policy = anchor_height
  tool_motion_axis_target_ref = scene.direction

brush forward:
  action_distance_m = 0.12
  tool_motion_axis_relation = parallel
  project_start_pose_to_table = false
  max_translation_step_m = 0.035
  max_rotation_step_rad = 0.30
  min_clearance_m = 0.003
  swept_min_clearance_m = 0.006
  allowed_contact_penetration_m = 0.006

rollout:
  force_no_reset = true
  max_steps = 6500
  success_steps = 3

hammer down:
  direction_source = vlm_scene_direction
  end_surface_policy = anchor_contact_plane
  tool_motion_axis_relation = opposite
  endpoint_policy = start_equals_end_when_no_visible_endpoint
  max_translation_step_m = 0.035
  max_rotation_step_rad = 0.30
  min_clearance_m = 0.003
  swept_min_clearance_m = 0.006
  allowed_contact_penetration_m = 0.006
  max_steps = 1800
  success_steps = 3

eraser wipe:
  target_approach_axis_source = opposite_scene_direction
  end_surface_policy = anchor_contact_plane
  tool_motion_axis_relation = opposite
  tool_motion_axis_relation = parallel
  max_translation_step_m = 0.035
  max_rotation_step_rad = 0.30
  rollout_max_tracking_error_m = 0.018
  rollout_max_rotation_error_rad = 0.28
  initial_grasp_lift_m = 0.08
  initial_grasp_lift_steps = 3
  max_steps = 2500
  success_steps = 3

screwdriver insert:
  direction_source = vlm_scene_direction
  end_depth_policy = task_contract_socket_inside_goal_along_vlm_scene_direction
  terminal_inside_depth_m = 0.035
  success_min_inside_depth_m = 0.005
  wide_socket_inner_diameter_m = 0.024
  wide_socket_tip_radial_tolerance_m = 0.012
  target_approach_axis_source = opposite_scene_direction
  tool_motion_axis_relation = same
  tool_motion_axis_relation = same
  max_translation_step_m = 0.025
  max_rotation_step_rad = 0.25
  rollout_max_tracking_error_m = 0.018
  rollout_max_rotation_error_rad = 0.28
  initial_grasp_lift_m = 0.08
  initial_grasp_lift_steps = 3
  max_steps = 1600
  success_steps = 3
```

Task configs do not add default intermediate semantic subgoals. The compiler
uses explicit VLM start/end refs and any explicit VLM `path_points`; absent
those path points, a contact path is represented by its required anchors only.
Any controller smoothing must remain controller-only diagnostic behavior and
must not be drawn or reported as VLM evidence.

The compiler applies the main table safety rule directly at subgoal generation
time: every tool-part OBB corner must satisfy `corner_z >= table_top_z`. This
is the primary anti-penetration mechanism. The orientation naturalness cost also uses all
tool-part OBB corners when a `tool_part_graph` is available, rather than a
single whole-tool AABB.

## Manual Chain

The one-command benchmark expands to the following chain:

```text
render_simtoolreal_rgbd_scene_evidence.py
  -> run_vlm_scene_keypoint_grounding.py
  -> visualize_scene_keypoint_grounding.py
  -> lift_scene_keypoints_to_constraint_program.py
  -> merge_tool_refs_into_constraint_program.py
  -> compile_image_grounded_with_mainline.py
  -> parttoolgen.execution.simtoolreal_rollout
```

The red-brush tool refs are stored inside this branch:

```text
configs/tool_refs/red_brush/
  axis_binding.json
  axis_mesh_overlay_report.json
  contact_point_selection.json
```

Hammer refs use the same format:

```text
configs/tool_refs/claw_hammer/
  axis_binding.json
  axis_mesh_overlay_report.json
  contact_point_selection.json
```

Flat-eraser refs use the same publishable mainline tool-ref format:

```text
configs/tool_refs/flat_eraser/
  axis_binding.json
  axis_mesh_overlay_report.json
  contact_point_selection.json
```

Long-screwdriver insertion expects the same task-conditioned output format:

```text
configs/tool_refs/long_screwdriver/
  axis_binding.json
  axis_mesh_overlay_report.json
  contact_point_selection.json
```

Hammer uses the same generic `move_along`/point-contact constraint vocabulary
as brush; the point-action `scene.direction` comes from VLM scene grounding,
while the hammer functional face refs remain public tool refs.

The direct constraint compiler also supports a generic
`orientation_offset_about_axis` constraint for future rotational geometry
smokes: after anchoring the active start point and direction, the whole tool can rotate around a
grounded axis while preserving the active start point. This is a
geometric SE(3) constraint, not a task primitive.

The current method does not use `tool_family`, `task_family`, hidden target
axes, or fallback target refs. The VLM supplies scene `start_point`, optional `waypoints`, `end_point`, and
`scene.direction` when the image/task supports a direction. Tool-side VLM refs
provide `tool.contact_point` and `tool.motion_axis`;
the compiler solves whole-tool SE(3) subgoals under those public refs.

When a tool-side VLM call returns a small set of semantically equivalent
function-face candidates, the compiler may choose the active face by satisfying
the explicit point/axis constraints while minimizing whole-tool geometry motion
from the current pose. This is a deterministic tie-break inside VLM-proposed
public refs, not a search over hidden task families, oracle target axes, or
unseen evaluator geometry. Once the compiler selects a contact point for a
source ref such as `tool.contact_point`, later approach/use stages keep that
same point instead of switching faces mid-task.
