# Environment Registry

This registry records the benchmark environments that agents and runners should
use by default. Keep it lightweight and human-readable. Runtime renders, videos,
rollout summaries, and generated previews belong outside this registry.

| Environment id | Status | Task families | Asset path | Public evidence camera | Notes |
|---|---|---|---|---|---|
| `tabletop_wood_table` | fixed/current | `brush_sweep`, `hammer_strike`, `eraser_wipe/tabletop_wipe`, `screwdriver_insert` | `urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf` | tabletop/topdown task evidence cameras | Shared table-top benchmark environment. Task targets remain task-specific assets. |
| `side_left_vertical_whiteboard_soft02` | fixed/current | `marker_trace`, side-board eraser/hammer/screwdriver drafts | `urdf/parttoolgen_benchmark/marker_trace_whiteboard/left_side_vertical_whiteboard_002x0475x101.urdf` | `whiteboard_front` | Blank vertical whiteboard on the table-left side, `soft02` lighting (`--light-preset soft --soft-light-indices 0,2`). Produced ink is post-rollout evidence only. |

## Side-Board Canonical Metadata

```json
{
  "whiteboard_center_world": [0.38, 0.0, 0.505],
  "public_whiteboard_surface_normal_world": [-1.0, 0.0, 0.0],
  "target_kind": "left_side_vertical_blank_writable_whiteboard_surface"
}
```

Use the side-board whiteboard actor pattern documented in the
`isaac-environment` skill and `artifacts/tool_use_benchmark/side_board/README.md`.
Do not introduce task-specific action axes, hidden evaluator refs, oracle
trajectories, or pre-rendered trace guides as environment assets.

## Isolation Rules

Table-top tasks and side-board tasks must use separate environment ids, cameras,
and target assets. Reusing a task target across environments requires an explicit
contract note. The shared side-board whiteboard is never a task-specific target;
it is the support surface for side-board tasks.
