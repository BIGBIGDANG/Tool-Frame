# PartToolGen Schemas

The schema directory defines the fixed interfaces between providers, VLM
program synthesis, the compiler, and SimToolReal adapters.

## Current Mainline Interfaces

- `affordance_binding.schema.json`: compiler-compatibility binding artifact
  produced after a validated Part-Centric Affordance Program. It is not the
  paper-facing VLM output format in 5.15.
- `planner_program.schema.json` and `program_ir.schema.json`: restricted
  planner program and compiler IR.
- `tool_pose_subgoals.schema.json`, `compiler_report.schema.json`, and
  `simtoolreal_fixed_goals.schema.json`: task-agnostic compiler output and
  unchanged SimToolReal fixed-goal adapter interface.
- `toolframe_task_spec.schema.json`: unified spec-first task/scene runner
  contract. Current benchmark configs normalize into this shape; new task or
  scene experiments should start here before adding family-specific code.
- `tool_part_graph.schema.json`, `functional_tool_parts.schema.json`,
  `grounding_candidates.schema.json`, and `visual_evidence.schema.json`:
  offline SAMPart3D region and geometry functional-frame artifacts.
- `provider_contract.schema.json`, `observed_scene_graph.schema.json`, and
  `target_part_graph.schema.json`: online task/target/provider contracts.

## Compatibility Notes

Historical affordance binding and evaluation-suite artifacts may still validate
for report reproducibility. They are not the active image-grounded task shape.
New experiments should follow the current public-evidence contract documented in
`docs/current_pipeline.md`, then let PartToolGen emit any compatibility binding
internally.
