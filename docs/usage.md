# Commands and inputs

Run commands from the repository root. `python run.py --help` lists available
commands. Use a new output directory for each experiment; a failed replan can
leave files from an earlier successful run in a reused directory.

## Plan and execute separately

For CPU planning with the included observation, use the [quick start](../README.md#quick-start-cpu-planning).
For online tool selection and scene grounding, configure the
[runtime and VLM](reproducibility.md), then run:

```bash
python run.py plan \
  --task examples/mallet_hammer_strike.json \
  --output outputs/mallet_online_plan --device cuda:0

python run.py execute \
  --task examples/mallet_hammer_strike.json \
  --fixed-goals outputs/mallet_online_plan/fixed_goals.json \
  --output outputs/mallet_execution --device cuda:0
```

`run` combines planning and execution. To use your own or cached grounding,
provide all three inputs:

```bash
python run.py plan \
  --task examples/mallet_hammer_strike.json \
  --scene-evidence examples/observations/mallet_hammer_strike/scene_evidence.json \
  --scene-keypoints examples/observations/mallet_hammer_strike/scene_keypoints.json \
  --tool-refs assets/tool_refs/mallet_hammer.json \
  --output outputs/mallet_cached_plan
```

Cached tool references must match the tool and intended operation. Grounding
annotations must match the paired scene observation; tool references alone
do not replace scene grounding. Execute goals only with the task/tool that
generated them; the current goal format does not enforce that association.

## Outputs

| File | Contents |
| --- | --- |
| `constraint_program.json` | Staged geometric program |
| `tool_pose_subgoals.json` | Solved poses, residuals, and warnings |
| `fixed_goals.json` | Subdivided control goals |
| `execution_state.json` | Control status, steps, goal progress, and final tool pose |

`completed` means the controller advanced through the goals. `timeout` means
the step budget was exhausted. Neither state measures the physical task effect.

## Task configuration

The nine `examples/*.json` files use a common format:

- `task_id`: experiment identifier.
- `instruction`: language description of the operation.
- `tool`: tool ID and geometric prior directory.
- `execution`: simulation resources, initial tool pose, and scene objects.
- `metadata.public_support_surface_point_world` and
  `metadata.public_support_surface_normal_world`: declared support geometry.
- `interaction_plane`: optional plane and keypoint roles for ray-plane lifting.

Scene actor poses construct the simulator scene; they do not replace image
keypoints in the planner. Positions and depth use meters. Poses use
`[x,y,z,qx,qy,qz,qw]`. All tasks read `settings.json`; pass `--settings` for an
alternative shared configuration.

For example, a horizontal interaction plane can be declared as:

```json
{
  "point_world": [0, 0, 0.53],
  "normal_world": [0, 0, 1],
  "point_roles": ["start", "end", "path"],
  "source": "provided_scene_geometry"
}
```

Selected point roles use camera-ray intersection with that plane; other roles
use depth lifting. For insertion, `point_roles:["start"]` can constrain the entry
while retaining depth for the end point. `anchor_from_depth:"start"` can replace
`point_world` to derive the plane anchor from the start depth and declared normal.

## RGB-D and keypoints

`scene_evidence.json` pairs RGB, depth, and calibration from one view. Paths are
relative to that JSON or absolute:

```json
{
  "instruction": "Use this tool to act on the visible target.",
  "items": [
    {"id":"scene_rgb", "kind":"scene_rgb", "path":"rgb.png"},
    {"id":"scene_depth", "kind":"scene_depth", "path":"depth.npy"},
    {"id":"scene_camera", "kind":"camera", "path":"camera.json"}
  ],
  "metadata":{"prompt_images":["rgb.png"], "grounding_image_index":0}
}
```

Depth is positive camera-Z depth in meters, shaped `[height,width]`.
Calibration contains `width,height,fx,fy,cx,cy,camera_to_world`, with camera X
right, Y down, and Z forward.

`scene_keypoints.json` contains `start_point:{u,v}`, optional
`end_point:{u,v}`, and ordered `path_points:[{u,v},…]`. Direction can come from
`action_direction:{start:{u,v},end:{u,v}}`, a nonzero path segment, or
`direction_decision.selected_default_axis_id` set to `support_surface_normal`
or `opposite_support_surface_normal`. A single-point operation requires an
explicit direction or normal choice.

## New tools

The eight prepared tools can use `assets/tool_prior/` directly. To rebuild the
mallet prior from included face labels, without a segmentation model:

```bash
python run.py prepare-tool \
  --tool-id mallet_hammer \
  --mesh simtoolreal/assets/urdf/dextoolbench/hammer/mallet_hammer/mallet_hammer.obj \
  --face-labels assets/tool_prior/mallet_hammer/face_labels/p3sam_face_labels.npy \
  --output outputs/rebuilt_prior
```

Face labels must be a one-dimensional integer array, one entry per original
mesh triangle. Output contains a mesh copy, labels, priors, and tool views.
Keep generated priors inside the repository's ignored `outputs/` if moving the
whole project; priors generated outside the root retain absolute resource paths.

For a new mesh, the P3SAM adapter expects a separately installed environment:

```bash
python run.py prepare-tool \
  --tool-id my_tool --mesh /path/to/my_tool.obj \
  --p3sam-root /path/to/P3SAM \
  --p3sam-python /path/to/p3sam-env/bin/python \
  --output outputs/my_tool_prior
```

The adapter invokes `demo/auto_mask.py` with mesh cleaning disabled to preserve
face indexing and expects `weights/p3sam/p3sam.safetensors`, plus P3SAM's own
dependencies and Sonata resources. Fresh-mesh inference is not covered by the
current validation. It is separate from the IsaacGym Python 3.8 environment.

Set `tool.tool_prior_dir` to the generated `tool_prior/`. For simulation, also
provide a matching URDF and register physical resources and grasp dimensions
in `simtoolreal/dextoolbench/objects.py`; prior generation supplies geometry only.
Tool references can be selected independently with:

```bash
python run.py tool-ref \
  --task examples/mallet_hammer_strike.json \
  --output outputs/mallet_tool_refs
```
