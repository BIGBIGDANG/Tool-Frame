# Brush Sweep Benchmark Configs

This folder records the current table-top brush-sweep benchmark configs. Run
outputs belong under `artifacts/image_grounded/brush_sweep/`.

## Environment

Use the shared table-top environment:

```text
environment id = tabletop_wood_table
table asset = urdf/parttoolgen_benchmark/tabletop_polyhaven_wood_table/tabletop_polyhaven_wood_table.urdf
public evidence = tabletop/topdown RGB-D task evidence
```

The local runtime environment is described in `environment.json`.

## Splits

| Split | Status | Config | Success-check status |
|---|---|---|---|
| `red_zone` | fixed/current | `red_zone/benchmark_config.json` | Final stable blue-block xy center inside the visual-only red outline zone; tracking is diagnostic. |
| `forward` | diagnostic/legacy | `forward/benchmark_config.json` | Directional target displacement retained for regression checks, not headline statistics. |
| `left` | diagnostic/legacy | `left/benchmark_config.json` | Leftward displacement variant retained for regression checks, not headline statistics. |

## Contract Summary

Brush sweep uses public RGB-D evidence, shared scene refs, and task-conditioned
brush tool refs. The compiler drives `tool.contact_point` through the public
contact path and aligns `tool.motion_axis` to `scene.direction` when applicable.

The headline task effect is final block occupancy inside the public red outline
target zone. Directional-displacement splits remain runnable diagnostics.
Strict rollout tracking, endpoint error, and contact-path details are diagnostic
unless a reviewed contract says otherwise.

The current red-zone split uses the 50 g stable sweep block with actor friction
0.8 so success is dominated by controlled planar sweeping rather than a
lightweight block flying off the table.

The configs do not use hidden benchmark target centers, task-family fallbacks,
tool-family fallbacks, or automatic global-axis retries.
