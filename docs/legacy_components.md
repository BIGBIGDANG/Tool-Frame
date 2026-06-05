# Legacy components

ToolFrame still contains some V0 symbolic planner modules because they are
useful for debugging and historical comparisons. They are not the public
interface for the current image-grounded benchmark.

## Legacy V0 symbolic affordance stack

The following modules and schemas use `function_*` terminology:

```text
parttoolgen/src/parttoolgen/program/affordance_binding.py
parttoolgen/src/parttoolgen/program/given_function_region_planner.py
parttoolgen/src/parttoolgen/checks/affordance_binding_sanity.py
parttoolgen/schemas/affordance_binding.schema.json
```

Treat these as legacy/debug components. Current benchmark tasks should use:

```text
scene.start
scene.end
scene.waypoints
scene.direction
tool.contact_point
tool.motion_axis
tool.motion_axis
```

Do not introduce new benchmark configs or contracts that expose
`function_keypoint_ref`, `function_axis_ref`, or
`selected_function_contact_ref` as the primary public method interface.
Do not use older `engagement` wording in new human-facing contracts; use
`start`, `waypoint`, `end`, and `direction` for scene refs, and
`contact_point` and `motion_axis` for tool refs.

## Acceptable compatibility fields

Some readers and adapters still accept older `start_point` / `end_point`
aliases or legacy report fields so historical artifacts remain inspectable.
Those compatibility paths should be documented as legacy and should not be used
to define new benchmark contracts.
