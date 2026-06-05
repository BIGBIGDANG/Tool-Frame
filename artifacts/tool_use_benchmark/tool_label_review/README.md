# Manual Tool-Label Review

This directory is the human-review staging area for evaluator-side tool labels.
It is intentionally separate from method inputs and runtime outputs.

Rules:

- Candidate refs copied from VLM/P3SAM artifacts are suggestions only.
- Tool labels are evaluator-side audit metadata only. They are not method
  inputs, compiler-side selection rules, rollout success gates, or shortcuts for
  choosing a tool face.
- Task success is decided by executed effects such as displacement, insertion
  depth/radial error, surface coverage, semantic contact, or produced ink.
- Approved contracts may contain sparse `allowed_*` lists. Empty lists mean the
  corresponding label is not audited, not that the run failed.
- Filled `allowed_*` lists are used only for diagnostic label checks and
  post-hoc failure analysis.
- Pending contracts should never be used to call a selected part or face wrong.
- Keep task semantics explicit: contact face, contact point, and motion axis
  may be different labels, but only record labels that are actually useful for
  audit.
- Faces marked `is_internal_tool_surface: true` are not valid VLM choices. They
  should be filtered out before contact/motion-axis prompting for every tool.

Review order:

1. `eraser_wipe__flat_eraser.json`
2. `brush_sweep__red_brush.json`
3. `hammer_down_strike__claw_hammer.json`
4. `screwdriver_insert__long_screwdriver.json`

For each tool, inspect the listed overview/card/face images, then fill only the
labels that are useful for auditing task semantics. If final task success is
already captured by an executed-effect metric, keep the label contract minimal
and add a short note.
