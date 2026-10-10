# Architecture

The planner represents an operation using tool-local geometric references
and scene references grounded in calibrated images. Simulation actor poses
define the scene, while image grounding supplies the planner's target points.

## Processing stages

1. Mesh face labels define tool parts. A geometric prior contains part graphs,
   contact candidates, and local motion axes.
2. A VLM selects a working part, then its contact face and motion axis using
   rendered tool evidence. These references can also be supplied as cached JSON.
3. Scene grounding provides image keypoints and direction decisions. Calibrated
   depth or ray-plane intersection lifts them into world coordinates.
4. The adapter builds a staged constraint program. The compiler solves whole-tool
   poses for approach, contact, and action; path checks and subdivision produce
   fixed goals.
5. SimToolReal executes the goals with a pretrained recurrent policy. Geometric
   planning and low-level control remain separate.

## Code map

Paths below are relative to `ToGo/src/ToGo/`.

| Module | Responsibility |
| --- | --- |
| `pipeline/handoff.py` | Public CLI and stage orchestration used by `run.py` |
| `pipeline/prepare_tool.py` | Segmentation adapter and prior generation |
| `pipeline/lifting.py` | RGB-D pairing and interaction-plane lifting |
| `cli/tool_prior/` | Tool evidence rendering and reference selection |
| `cli/stages/` | Simulation observations and scene grounding |
| `image_grounded/staged_mainline_adapter.py` | Tool/scene references to staged constraints |
| `compiler/`, `checks/`, `adapters/` | Pose solving, path subdivision, and surface constraints |
| `program/`, `constraints/`, `validation/` | Program representation, model calls, and input contracts |
| `providers/` | Geometry priors and tool-view caches |
| `execution/` | Fixed-goal execution and runtime integration |

`simtoolreal/` contains the adapted simulation backend and policy inference
dependencies. It preserves the 140-dimensional observation and 29-dimensional
action conventions, normalization, and recurrent state. The policy also uses
its original additional scalar input of 50. The included backend is scoped to
simulation inference; use upstream SimToolReal for training and real-world deployment.

`settings.json` supplies shared planning and control parameters.
`assets/tool_prior/`, `assets/tool_refs/`, and `examples/` provide prepared
geometry, cached selections, and scene configurations. `ToGo/tests/` contains
CPU software regressions.

## Geometry conventions

Tool meshes, face labels, part graphs, contact references, and execution URDFs
must share the same tool-local coordinate frame and meter units. Face labels
index the original mesh triangles. Pose arrays use `[x,y,z,qx,qy,qz,qw]`.

Compiler support constraints require zero penetration below the table plane.
The path check's `allowed_contact_penetration_m` is a separate contact-stage
tolerance; it does not relax the compiler constraint. Solver residuals and
warnings are retained in the output.
