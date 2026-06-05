# Hammer Strike Benchmark Configs

This folder records the current table-top hammer strike smoke connected to the
image-grounded mainline. Run outputs belong under
`artifacts/image_grounded/hammer_strike/`.

## Current Task

| Split | Environment | Status | Success-check status |
|---|---|---|---|
| `down` | `tabletop_wood_table` | current/provisional | Semantic contact and strike-direction success; claw-hammer labels pending. |

Config:

```text
down/benchmark_config.json
```

The method-facing inputs are natural task language, public SimToolReal RGB-D,
VLM-marked scene refs, and public hammer tool refs. For this point-action task,
the executable contact direction comes from VLM scene grounding; surface normals
are projection/diagnostic evidence only.

The task does not require nail/target displacement. Target displacement,
strict tracking, and rotation error are diagnostics.
