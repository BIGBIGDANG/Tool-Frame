# ToolFrame Architecture

ToolFrame is an image-grounded, tool-centric constraint planner. The VLM does
not emit robot trajectories. It grounds refs and proposes staged constraints;
the compiler solves whole-tool SE(3) subgoals.

## Inputs

Public method inputs:

- task language;
- scene RGB-D image or calibrated multi-view render;
- camera intrinsics/extrinsics;
- public tool refs under `configs/tool_refs/<tool_id>/`;
- optional robot/gripper state needed for execution.

Private benchmark-only inputs:

- simulator object ids and true states;
- success predicates;
- oracle contact annotations;
- hidden evaluation masks.

## Pipeline

```text
1. Render/observe scene evidence.
2. VLM marks task-conditioned scene refs. The current scaffold records them as
   `start_point`, optional `waypoints`, `end_point`, and `scene.direction`
   for the executable chain.
3. Lift 2D keypoints through depth and camera calibration.
4. Derive or read `scene.direction` from VLM scene grounding.
5. Build explicit staged tool-frame constraints.
6. Merge public tool refs and VLM-valid tool-side candidate sets.
7. Compile constraints to whole-tool SE(3) subgoals.
8. Execute through SimToolReal fixed-goal policy.
9. Score with private metrics.
```

## Core Representation

The tool side is represented as:

```text
contact_point_face_id
contact_point_tool = face center in tool frame
contact_point_candidates_tool = VLM-valid function-face contact candidates
motion_axis_candidates_tool   = VLM-valid/default-public motion-axis candidates
```

Tool-side evidence is normalized before planning:

```text
axis_binding.json + contact_motion_axis_mesh_overlay_report.json
  -> merge_tool_refs_into_constraint_program.py
  -> constraint_program_v1.tool_refs
```

The scene side is represented by one generic scene-ref interface:

```text
start_point_2d  -> start_point_3d
waypoints_2d    -> optional intermediate path points lifted through RGB-D
end_point_2d    -> end_point_3d
scene.direction -> scene-side use/path direction for the tool contact point
```

When the VLM marks a visible `end_point`, the executable use-stage path is
derived from RGB-D projection: `start_point_3d -> end_point_3d`. If the endpoint
is not image-separable, such as a vertical hammer strike in a top-down image, the
VLM returns `end_point == start_point` and grounds `scene.direction` separately.
The compiler then approaches from the derived pre-contact point and finishes at
the explicit endpoint; it does not invent an implicit action distance constant.
When tool-side grounding returns multiple semantically valid contact points or
motion axes, all of those public VLM-valid candidates are carried into the
compiler. The optimizer chooses the active candidate by explicit geometric
constraints and minimum-change cost; it does not search hidden tool faces or
task-family fallbacks. The selected contact point is locked per source ref for
the rest of the compiled task so approach and use stages do not switch faces.

`start_point_3d` is the lifted VLM pixel. It must not be replaced by simulator
object centers or private target-box face centers. Hidden target geometry is
only for benchmark scoring and scene construction.

Each stage has two constraint groups:

```text
subgoal_constraints: define the terminal whole-tool SE(3) pose for this stage
path_constraints:    define what must remain true while reaching that pose
```

The public method stages are intentionally limited to:

```text
align -> approach -> action
```

Conceptually, `align` fixes the selected tool motion axis at the lifted pose,
`approach` moves the selected contact point to the pre-contact offset opposite
the first VLM motion direction, and `action` follows the grounded use path.

Backend-specific preparation such as SimToolReal settling or grasp lifting is
added by the execution adapter and is not part of the method-stage abstraction.

The compiler solves tool poses `T_world_tool` such that:

```text
T_world_tool * contact_point_tool ~= start point
R_world_tool * motion_axis_tool ~= selected VLM scene direction
```

ToolFrame does not infer action semantics from `tool_family` or `task_family`.
Axis bindings must be explicit in the constraint program, for example:

```text
axis_align(tool.motion_axis -> scene.direction)
move_along(tool.contact_point, start_point -> end_point)
```

Missing staged axis constraints are treated as an error, not as a cue to guess a
brush/hammer default.

## Constraint Types

The first clean DSL should stay small:

- `point_match`: align a tool ref point with a scene ref point.
- `point_offset`: place a tool ref point at a fixed offset from a scene ref.
- `axis_align`: align or anti-align a tool axis to a scene axis.
- `move_along`: move a tool ref along a lifted scene direction or path.
- `orientation_offset_about_axis`: rotate the whole tool about a grounded axis
  while preserving the active point/axis relation.
- `clearance`: keep an explicitly requested diagnostic clearance relation.

Names like sweep, strike, scrape, and hook are task-language interpretations,
not compiler primitives.

## VLM Scaffold Policy

The VLM may provide sparse image-space waypoints when the language implies a
curved or multi-step path. These are path hints, not executable robot goals:

```text
scene.motion_path = [start_point, wp0, ..., end_point]
```

The compiler preserves the sparse public anchors by default. Any extra
transition waypoint is a compiler transition for staged alignment/approach, not
a task semantic waypoint or path-repair artifact.

## Execution Bridge

The current clean bridge is:

```text
constraint_program_v1
  -> compile_image_grounded_with_mainline.py
  -> staged ProgramIR lowering
  -> mainline whole-tool SE(3) compiler plus diagnostic path gate
  -> ToolPoseSubgoals v0.2
  -> simtoolreal_adapter.convert_subgoals_to_simtoolreal
  -> simtoolreal_fixed_goals.json
```

The bridge keeps whole-tool SE(3) solving, diagnostic path checks, and
SimToolReal fixed-goal export. Family templates, repair-loop fallback,
tracking/corridor waypoint insertion, and benchmark-only target-axis defaults
are not part of the clean method path.
