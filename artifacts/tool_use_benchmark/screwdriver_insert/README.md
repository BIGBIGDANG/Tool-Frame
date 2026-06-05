# Screwdriver Insert Benchmark

This directory records the current table-top screwdriver insertion configs. Run
outputs belong under `artifacts/image_grounded/screwdriver_insert/`.

## Splits

| Split | Status | Success-check status |
|---|---|---|
| `wide_socket` | current/provisional headline split | Compiled terminal socket-goal check, tip radial alignment, and measured inside-socket depth; long-screwdriver labels pending. |
| `deep_socket` | stress/provisional | Same contract family with stricter radial tolerance. |
| `wide_socket_dense` | optional diagnostic/stress | Dense-subgoal diagnostic variant of the wide socket; excluded from the default current suite. |
| `deep_socket_dense` | optional diagnostic/stress | Dense-subgoal diagnostic variant of the strict socket; excluded from the default current suite. |
| `side_socket_insert` | draft/not runnable | Side-board socket insert split; blocked until the screwdriver runner supports side-board scene mode and validated side-socket actor pose. |

The target socket is a real public insertion fixture, not a painted marker. It
has visible side walls, an open aperture, and enough vertical depth to measure
whether the screwdriver tip enters the hole.

A seed passes only when the diagnostic path gate, compiled terminal socket goal,
semantic insertion evidence, measured tip radial alignment within the public
socket tolerance, and measured inside-socket depth exceed the reviewed threshold.
The terminal endpoint error, strict tracking, and policy-goal pass are diagnostics.

The legacy SimToolReal screwdriver spin task is not part of this benchmark.
