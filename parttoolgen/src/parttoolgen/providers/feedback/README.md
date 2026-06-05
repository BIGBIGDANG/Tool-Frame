# Feedback Providers

This package is legacy/support code for normalizing execution evidence such as
tracking error, contact proxies, force/torque, tactile signals, visual checks,
drop events, and task-success estimates.

The active ToolFrame 6.4 image-grounded tabletop benchmark does not use a
repair-loop fallback as a method step. Failed semantic contact, tracking, or
task-success evidence is reported for diagnostics and failure attribution; it
must not silently trigger ReKep-style re-solving, RoboTwin-style visual repair,
or hidden target-ref substitution in the mainline benchmark.

Simulation-only internal signals must be marked as such when used for
evaluation or debugging.
