# ToolFrame 6.4

ToolFrame 6.4 is an image-grounded tool-use benchmark and compiler stack for
turning public RGB-D scene evidence plus tool evidence into whole-tool SE(3)
subgoals for SimToolReal execution.

The core pipeline is:

```text
natural task language
  + public RGB-D scene evidence
  + public tool mesh / part evidence
    -> VLM scene refs: start_point, optional waypoints, end_point, scene.direction
    -> RGB-D lifting to 3D scene refs
    -> VLM/P3SAM tool refs: tool.contact_point, tool.motion_axis, valid candidates
    -> whole-tool SE(3) constraint compilation
    -> SimToolReal fixed-goal rollout
    -> task-level success checks and failure attribution
```

This branch focuses on public, image-grounded evidence. It deliberately avoids
hidden evaluator target refs, oracle target axes, oracle trajectories,
tool-family/task-family fallback, primitive fallback, repair-loop fallback, and
seed-specific shortcuts.

## Repository layout

```text
docs/                         method, environment, handoff, and repo docs
bin/toolframe                 thin public command wrapper
parttoolgen/src/parttoolgen/   compiler/runtime Python package
parttoolgen/src/parttoolgen/cli/ internal runner/stage/tool-prior CLIs
parttoolgen/scripts/           small public handoff scripts only
parttoolgen/schemas/           JSON schemas for tasks and ProgramIR
parttoolgen/tests/             unit tests for maintained behavior
configs/tool_refs/             reviewed public tool reference bindings
artifacts/tool_use_benchmark/  benchmark contracts, configs, matrices, metrics
artifacts/image_grounded/tool_ref_grounding/ reviewed tool-selection evidence
artifacts/tool_prior/          compact P3SAM tool-prior packages
simtoolreal/                   local SimToolReal backend copy
test000/                       local scratch/review outputs; ignored
```

See `docs/repo_organization.md` for the commit and artifact boundaries.

## Quick start

From a fresh clone, build the project-local Python 3.8 environment first:

```bash
cd <your clone>/ToolFrame_6.4
bin/toolframe bootstrap-env
bin/toolframe env-check
```

The bootstrap creates `.venv_parttoolgen_py38` inside this repository. The
environment wrapper uses that local venv unless `PARTTOOLGEN_CORE_VENV` is set
explicitly. It may read optional local runtime paths from
`.parttoolgen.local.env`, but those are workstation overrides only. Keep
secrets, model endpoints, pretrained policy paths, and large runtime artifacts
out of git.

Core compiler/tests are reproducible from the checked-in requirements. Full
SimToolReal/IsaacGym rollouts additionally require IsaacGym binaries,
pretrained policy weights, GPU-compatible torch packages, and VLM credentials;
package those as explicit local artifacts or documented downloads, not hidden
absolute paths.

For manual installs, use the root requirements files:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-rollout.txt  # optional
python3.10 -m pip install --target .planner_vlm_py310_pkgs -r requirements-vlm.txt  # optional local VLM
```

See `docs/quickstart.md` for the recommended first smoke commands and expected
output locations.

## Reproduce Results

Use the public wrapper for all reproduction paths:

```bash
bin/toolframe bootstrap-env
bin/toolframe env-check --mode core
bin/toolframe check --quick
```

For artifact/config validation without rollout:

```bash
bin/toolframe env-check --mode rollout
bin/toolframe preflight-current
bin/toolframe benchmark --validate-registry
```

For full SimToolReal rollout statistics, provide IsaacGym, policy weights, GPU
runtime packages, and VLM credentials through `.parttoolgen.local.env`, then run
the current full E2E suite:

```bash
bin/toolframe current-e2e \
  --trials 10 \
  --seed-start 0 \
  --device cuda:0 \
  --capture-video-every 1 \
  --continue-on-task-error \
  --tool-ref-model gpt-5.5 \
  --tool-ref-base-url "$VLM_API_BASE" \
  --output-prefix artifacts/test008/e2e_10seed_gpt55_20260607 \
  --aggregate json,csv,md
```

Expected runtime outputs are regenerated under
`artifacts/image_grounded/<task_family>/<split>_<run_id>/` unless an explicit
`--output-prefix` is provided. Do not commit rollout videos, temporary test
roots, local virtual environments, model checkpoints, or private credentials.

For new tools, use `bin/toolframe tool-ref` to produce reviewed
`configs/tool_refs/<tool_id>/` bindings. For new scenes or task variants, start
from a unified task spec and inspect it with:

```bash
bin/toolframe run-spec --spec <task_spec.json> --preflight-only
```

Current benchmark configs normalize into this same spec layer; the
family-specific runners are compatibility backends, not the extension API.
The current paper-development result is full E2E success plus manual failure
attribution, not fixed/oracle-ref baselines or dense module-accuracy tables.

## Current benchmark scope

Current runnable benchmark suite:

```text
current_all_runnable
```

It contains brush sweep, eraser wipe, hammer strike, screwdriver insert, and
marker trace variants. Legacy brush red-zone/directional splits, dense
screwdriver stress variants, and unapproved side-board drafts are diagnostic
only.

Benchmark configs and contract drafts live under
`artifacts/tool_use_benchmark/`. Generated RGB-D evidence, VLM outputs, rollout
videos, and smoke artifacts belong under `artifacts/image_grounded/` or local
`test000/`, not in benchmark config directories.

## Important docs

- `AGENTS.md`: collaboration rules for coding agents.
- `docs/paper_pipeline_and_benchmark_6.4.md`: paper-facing pipeline,
  benchmark-suite protocol, metrics, and result citation boundary.
- `docs/current_pipeline.md`: current runnable pipeline and method boundary.
- `docs/quickstart.md`: setup and first smoke-run guide.
- `docs/architecture.md`: representation, compiler, and execution bridge.
- `docs/environment.md`: local environment wrapper and runtime artifact policy.
- `docs/repo_organization.md`: publish/commit boundaries and cleanup policy.
- `docs/legacy_components.md`: legacy V0 modules that are retained for
  debugging but are not the current image-grounded benchmark interface.
- `TOOLFRAME_VENDORING.md`: vendored SimToolReal/runtime asset boundary.
- `artifacts/tool_use_benchmark/README.md`: benchmark task/config status.
- `artifacts/tool_use_benchmark/evaluation_metrics.md`: compact reporting
  metrics and failure attribution policy.

## Verification before publishing changes

Use the project wrapper first:

```bash
bin/toolframe env-check --mode core
```

For normal handoff use, stay at the thin public wrapper:

```bash
bin/toolframe list
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
bin/toolframe benchmark --task brush_sweep/off_table_red --trials 1 --device cuda:0
bin/toolframe tool-ref --help
```

Recommended pre-commit checks for source/config changes:

```bash
git diff --check
bin/toolframe check --quick
```

Use `bin/toolframe check --full` before larger releases; it adds
the full pytest suite on top of the core-only quick checks.

Use `bin/toolframe benchmark --list`, `--validate-registry`,
`--describe <task>`, and `--preflight-only` to inspect registered benchmark
tasks before launching rollouts. The family-specific experiment scripts are
registered runners behind this launcher, not the preferred handoff interface.

The GitHub Actions workflow `.github/workflows/core-ci.yml` is intentionally
core-only: it bootstraps Python 3.8, validates the benchmark registry, compiles
maintained entrypoints, and runs lightweight tests. It does not require
IsaacGym, policy weights, VLM credentials, or GPU access.

Run expensive Isaac/rollout smokes only when validating a specific benchmark
contract or environment change.

## Artifact and asset policy

- Commit source code, schemas, tests, lightweight benchmark configs, and small
  reviewed benchmark assets.
- Do not commit `test000/` scratch outputs.
- Do not commit generated run evidence under `artifacts/image_grounded/**`
  unless it is part of the reviewed `tool_ref_grounding` evidence set or a
  specific manifest/evidence artifact is explicitly approved.
- Imported external ToolUse bulk assets are intentionally outside the current
  public release. Treat any future inclusion as a separate license, asset, and
  benchmark-contract decision.

## License note

The repository includes a local SimToolReal backend copy under `simtoolreal/`
with its own license file. Confirm project-level licensing before public
release.
