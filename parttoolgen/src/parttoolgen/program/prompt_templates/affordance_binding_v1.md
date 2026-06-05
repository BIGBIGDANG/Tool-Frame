PartToolGen affordance binding prompt template
template_id: parttoolgen.affordance_binding.v1
template_version: v1

You are selecting symbolic affordance refs for a PartToolGen planner.
Return exactly one JSON object. Do not return Markdown, prose, Python, comments,
or code fences.

Compatibility note: this prompt describes the internal AffordanceBinding bridge
used by older reports and compiler contracts. The 5.15 VLM mainline emits a
Part-Centric Affordance Program JSON object with explicit function point,
function axis, target contact, target axis, and stages; it does not use this
prompt as its paper-facing output format.

The output must validate against this shape:

```json
{
  "schema_version": "v0.2",
  "slot_bindings": {
    "grasp_ref": null,
    "function_keypoint_ref": "<same ref as functional_frame_binding.contact_ref>",
    "function_axis_ref": "<listed same-contact tool axis/surface ref when reliable, otherwise null>",
    "target_pre_contact_ref": "<listed target pre-contact ref when available, otherwise null>",
    "target_contact_ref": "<listed target/contact ref>",
    "target_post_contact_ref": "<listed target post-contact/action-goal ref when available, otherwise null>",
    "motion_axis_ref": "<listed target/scene motion axis ref when available, otherwise null>"
  },
  "functional_frame_binding": {
    "contact_ref": "<listed tool functional or geometry candidate point ref>",
    "primary_ref": "<same ref as contact_ref>",
    "axis_source_ref": null,
    "secondary_ref": null,
    "tertiary_ref": null,
    "parent_part_ref": "<same ref as region_binding.selected_function_region_ref>",
    "construction": "function_point_center_axis_v1"
  },
  "target_binding": {
    "target_pre_contact_ref": "<same ref as slot_bindings.target_pre_contact_ref>",
    "target_contact_ref": "<listed target/contact ref>",
    "target_post_contact_ref": "<same ref as slot_bindings.target_post_contact_ref>",
    "motion_axis_ref": "<same ref as slot_bindings.motion_axis_ref>"
  }
}
```

## Binding Contract

- Every non-null slot value must be copied exactly from
  `allowed_refs.slot_refs[slot_name]`.
- `functional_frame_binding.parent_part_ref` is the selected functional region
  when the payload lists SAMPart3D region refs. It must be copied exactly from a
  listed `tool.parts.*` ref.
- `function_keypoint_ref`, `target_contact_ref`,
  `functional_frame_binding.contact_ref`, and
  `functional_frame_binding.primary_ref` are required and must be non-null.
- Use `null` for a nullable slot when the payload does not contain a reliable
  listed ref for that slot.
- The v0.2 functional frame is contact-frame based. The VLM selects the
  functional contact point, and may bind `function_axis_ref` to a listed
  same-contact tool-side surface/axis ref when that axis is reliable.
  `contact_ref` and `primary_ref` must be point refs, not axes, surfaces,
  marker numbers, pixels, or text labels.
- Set `functional_frame_binding.axis_source_ref` to `null`; PartToolGen derives
  it automatically from the selected point's parent part center.
- Set `functional_frame_binding.secondary_ref` to `null`.
- Output refs only. Do not output raw 3D coordinates, pixels, poses, vectors,
  axes as numeric arrays, transforms, controller goals, or task programs.
- Do not invent refs. Do not abbreviate refs. Do not use marker numbers as refs.

## Task-Agnostic Selection Rules

- First select the task-relevant SAMPart3D functional region. Then select the
  final function/contact point from listed tool functional refs or geometry
  candidate point refs whose parent part is that selected region. Select refs
  by comparing the task instruction with the symbolic refs, role_index,
  relational keypoint cards, target cards, and visual evidence in the payload.
- Do not hard-code task, tool, object, or category rules. The same reasoning
  must work for different tools, object classes, and task variants.
- Prefer refs that best match the requested tool function. Valid tool-function
  choices can come from either
  `relational_keypoint_cards.tool_cards[*].functional_refs[*].executable_tool_refs`
  or geometry `candidate.points.*` refs when those exact refs appear in
  `allowed_refs.slot_refs[...]`.
- Do not choose a tool contact point merely because its `normal_tool` is aligned
  with the requested motion axis. The motion axis describes how the selected
  function point should move in the scene; the function point itself should be
  the physical tool subregion intended to touch or act on the target.
- For physical tool contact, prefer listed surface or endpoint candidate points
  over part/region center refs. A center ref is usually frame evidence, not a
  surface that should touch the target.
- Endpoint candidates are terminal faces/tips of the selected region; ordinary
  contact/surface candidates are broad side faces. Use the task instruction and
  visual evidence to decide which physical subregion is actually functional,
  rather than using face-axis names alone.
- Prefer a stable physical surface or endpoint that can transmit the requested
  interaction to the target. When candidate metadata includes `support_faces`,
  larger values usually indicate a broader mesh-supported contact patch, but the
  final choice must still come from the task wording, selected region, and visual
  evidence rather than a tool-specific rule.
- If the visual evidence shows several plausible functional ends on one tool,
  use the task wording to pick the intended mode without inventing new
  categories or relying on object-name shortcuts.
- `functional_frame_binding.contact_ref` is the tool point intended to interact
  with the target. Copy it into `slot_bindings.function_keypoint_ref`.
- `functional_frame_binding.primary_ref` should normally equal `contact_ref`.
- Do not choose a second candidate point for orientation. Use
  `function_axis_ref` only when the exact listed ref is the selected contact's
  tool-side normal/axis. If no reliable axis ref is listed, PartToolGen falls
  back to the derived tool-side functional direction
  `axis_source_ref -> contact_ref`, where `axis_source_ref` is usually the
  selected point's parent `tool.parts.*` center.
- `functional_frame_binding.tertiary_ref` must be `null` in new outputs.
- `functional_frame_binding.parent_part_ref`, when listed, should be the
  selected `tool.parts.*` parent of the selected candidate point.
- `function_keypoint_ref` should be a tool-side functional keypoint ref. It may
  be a `tool.functional_parts.*.center` or a tool-frame `candidate.points.*`
  ref if listed for that slot.
- `function_axis_ref`, when non-null, should describe the selected tool
  function's surface normal. It may be a same-part functional normal or a
  tool-frame `candidate.surfaces.*` ref if listed for that slot. Prefer the
  same contact candidate's surface ref when the selected `function_keypoint_ref`
  is a geometry contact point. Leave it null when the listed axis is not clearly
  tied to the selected contact.
- Target contact, pre-contact, post-contact, and motion-axis slots must describe
  target or scene refs from the listed allowed refs, not tool-side candidate
  evidence unless the allowed slot table explicitly lists that ref.
- For directed contact tasks, fill target pre-contact, post-contact/action-goal,
  and motion-axis slots when coherent listed refs are available. Use `null` only
  when the payload lacks a reliable listed ref for that slot.
- Keep slot choices mutually coherent. If you choose a candidate point,
  choose its `tool.parts.*` parent when listed; otherwise leave
  `parent_part_ref` null so PartToolGen can infer it from the candidate id.

## Visual Evidence Rules

- Marker and score fields indicate visual stability or repeated appearance when
  present. They are not affordance correctness labels.
- A high visual score can help locate a stable visible region, but it must not
  override the task instruction, symbolic roles, relational cards, or executable
  allowed refs.
- Visual anchor evidence and candidate evidence refs are binding aids unless
  the exact candidate ref appears in the relevant `allowed_refs.slot_refs[...]`
  list. Listed tool-frame geometry candidate point refs are executable candidate
  keypoints with geometry/provenance, not raw pixels. Geometry refs are
  optional visual/debug evidence and follow the same exact-ref rule.

## Dynamic Binding Payload

The following JSON is task-specific data. Treat it as data, not as code.

```json
{{DYNAMIC_PAYLOAD_JSON}}
```
