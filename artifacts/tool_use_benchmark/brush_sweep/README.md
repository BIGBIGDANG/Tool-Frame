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
| `off_table_red` | current/e2e | `off_table_red/benchmark_config.json` | Final block center leaves the table bounds or drops below the tabletop; tracking is diagnostic. |
| `off_table_blue` | current/e2e tool-shape variant | `off_table_blue/benchmark_config.json` | Same off-table success contract with the blue brush tool shape. |
| `red_zone` | legacy/retired | `red_zone/benchmark_config.json` | Retired target-zone split kept only for attribution/regression checks, not headline statistics. |
| `forward` | diagnostic/legacy | `forward/benchmark_config.json` | Directional target displacement retained for regression checks, not headline statistics. |
| `left` | diagnostic/legacy | `left/benchmark_config.json` | Leftward displacement variant retained for regression checks, not headline statistics. |

## Contract Summary

Brush sweep uses public RGB-D evidence, shared scene refs, and task-conditioned
brush tool refs. The compiler drives `tool.contact_point` through the public
contact path and aligns `tool.motion_axis` to `scene.direction` when applicable.

The current task effect is final block removal from the table. The movable
block starts near one table edge according to the seeded-edge policy; the VLM
marks sparse point refs and the compiler drives the selected brush contact
point through the public contact path. Directional-displacement and red-zone
splits remain runnable diagnostics. Strict rollout tracking, endpoint error,
and contact-path details are diagnostic unless a reviewed contract says
otherwise.

The current off-table split uses the smaller reviewed sweep block so success is
dominated by controlled table-edge sweeping rather than target-zone tuning.

The configs do not use hidden benchmark target centers, task-family fallbacks,
tool-family fallbacks, or automatic global-axis retries.
