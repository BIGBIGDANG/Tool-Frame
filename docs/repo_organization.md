# Repository organization

This repository separates reviewed source/configuration from generated benchmark
evidence. Keep these boundaries stable so benchmark agents do not confuse
runtime outputs with method inputs.

## Commit-ready source and configuration

- `AGENTS.md` and `docs/`: collaboration rules, current pipeline notes, handoff
  state, environment notes, and repository governance.
- `pyproject.toml`: editable-install metadata plus pytest/ruff configuration.
- `bin/`: thin public handoff wrappers. Normal users and CI start here.
- `parttoolgen/src/parttoolgen/`: importable PartToolGen package code,
  including internal CLI modules under `parttoolgen.cli`.
- `parttoolgen/scripts/`: small public handoff scripts only. Normal benchmark
  runs should enter through `bin/` or the registry launcher.
- `parttoolgen/schemas/`: JSON schemas for task/config/IR artifacts.
- `parttoolgen/tests/`: tests that cover reviewed source behavior.
  Tests are grouped by ownership in `parttoolgen/tests/README.md`; do not remove
  legacy/backend tests unless the compatibility module they cover is removed in
  the same reviewed change.
- `configs/tool_refs/`: reviewed public tool reference bindings, including
  `contact_point` and `motion_axis` naming.
- `artifacts/tool_use_benchmark/`: human-facing benchmark contracts, task
  drafts, evaluation metric definitions, and manifest-style documentation.
- `simtoolreal/assets/urdf/parttoolgen_benchmark/`: small reviewed benchmark
  assets such as side-board whiteboards, sockets, stickers, patches, and other
  task/environment URDFs.
- `TOOLFRAME_VENDORING.md`: vendored backend, external runtime dependency, and
  release-data boundary.

## Generated evidence and local scratch

- `artifacts/image_grounded/tool_ref_grounding/`: compact reviewed tool-selection
  evidence that may be packaged with a release.
- Other `artifacts/image_grounded/` roots: benchmark run outputs, RGB-D
  evidence, VLM markings, rendered previews, rollout videos, summaries, and
  visual packages. Keep local unless a specific evidence artifact is approved.
- `test000/`: interactive scratch, smoke runs, review renders, temporary
  rollouts, and local comparison outputs. This root is ignored and should not
  be committed.
- `artifacts/tool_prior/` and `artifacts/visualizations/`: compact evidence
  candidates; package separately from source if included in a public release.
  Third-party/imported asset roots remain local unless the project owner
  explicitly approves a dedicated data/asset commit.
- `reports/`: compact audit/provenance reports. Regenerate
  `reports/paper_release_6.4/` from `bin/toolframe paper-headline` or
  `bin/toolframe paper-release`; do not treat historical report snapshots as
  source of truth.

## External asset decision boundary

Imported external ToolUse assets are not part of the current public release.
If they are revisited later, treat them as a separate asset/license decision
with their own benchmark contracts and reviewed redistribution plan. Do not mix
large imported assets into normal source/config commits.

## Current benchmark task families

- Table-top tasks: `brush_sweep`, `hammer_strike`, `eraser_wipe`, and
  `screwdriver_insert`.
- Side-board tasks: shared side whiteboard environment with task-specific
  targets, currently including marker letter writing (`write_l`, `write_c`) and
  side-board eraser wiping drafts.
- Paper suites: `paper_release_6.4_headline` is the reviewed headline subset;
  `paper_release_6.4` is the broader candidate evidence-generation suite.

## Cleanup policy

- Prefer deleting stale `test000/` scratch roots once a newer named output root
  supersedes them.
- Do not delete `artifacts/tool_use_benchmark/**` contracts/configs as runtime
  cleanup.
- Do not use hidden evaluator references, oracle target axes, oracle
  trajectories, primitive fallback, repair-loop fallback, or seed-specific
  shortcuts as committed method inputs.
- Do not run `git add .`; stage reviewed file groups explicitly when the user
  asks for a commit.
