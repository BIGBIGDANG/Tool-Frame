# Validation results

Recorded on 2026-10-10 using fixed grounding inputs and the pretrained
SimToolReal policy, without VLM calls or policy training. The runtime used
Python 3.8.20, PyTorch 2.4.1+cu121, IsaacGym, and NVIDIA RTX 4090 D GPUs.

## What was checked

| Check | Result | Evidence |
| --- | --- | --- |
| Software regressions | 107 passed in a fresh environment | Run `python -m pytest -q` after installing development dependencies |
| Official policy checkpoint | SHA256 matches the official archive | [Checkpoint comparison](official_weight_parity.json) |
| Recurrent policy inference | Across 128 fixed observations, actions and LSTM states have maximum absolute difference 0 versus the original player | [Policy comparison](policy_parity.json) |
| Prepared tool geometry | Face labels, part geometry, selected contacts, and selected axes match the pre-refactor implementation for eight tools | [Geometry comparison](geometry_parity.json) |
| Goal compilation | All nine identical geometric-program inputs produce identical goal pose arrays versus the pre-refactor compiler | [Compiler comparison](compiler_comparison.json) |
| Published example inputs | All nine cached examples compile on CPU and reproduce the goals used in the recorded GPU run | [Fixture reproduction](fixture_reproduction.json) |
| Prior rebuilding | Eight prepared tools rebuild from included face labels | [Prior rebuild results](prior_rebuild.json) |
| Runtime independence | Fresh environment and relocated project run rendering, prior/view generation, and mallet execution; 26/26 goals in 445 steps | [Run summary](reproduction_result.json) |

Policy comparison includes zero, random, and scaled observations; it is a
finite numerical check, not a proof for every possible observation. These
comparisons use a frozen pre-refactor implementation. The repository provides
comparison summaries, not the full historical baseline, observation sequences,
or all diagnostic outputs needed to regenerate those comparisons.

## Nine-scene GPU execution

Each scene used shared `settings.json`, cached tool selections, and fixed
manual image annotations. All nine compiled and started policy execution.

| Scene | Control status | Goal progress | Steps |
| --- | --- | --- | --- |
| Blue brush sweep | completed | 36/36 | 971 |
| Claw hammer strike | completed | 25/25 | 358 |
| Tabletop eraser | completed | 42/42 | 543 |
| Whiteboard eraser | completed | 30/30 | 730 |
| Long screwdriver insertion | completed | 21/21 | 299 |
| Mallet strike | completed | 26/26 | 396 |
| Whiteboard marker | completed | 30/30 | 745 |
| Red brush sweep | timeout | 33/36 | 1500 |
| Short screwdriver insertion | timeout | 21/23 | 1500 |

The [execution summary](controller_runs.json) contains the task IDs
and recorded control states. All nine paired RGB-D/keypoint fixtures are included
under `examples/observations/<task_id>/`, together with scene configurations and
cached tool references. See [example observations](../../examples/observations/README.md)
to compile them without model calls.

**These are single-run controller outcomes, not task-success rates.**
There is no task-effect evaluator in this release. Goal completion does not
establish successful sweeping, insertion, striking, or surface marking/erasure.
The runs do not establish robustness across seeds or scene variations.

## Backend and configuration

The model is the official pretrained SimToolReal checkpoint. The recorded
configuration adds `resetWhenTableForceTooHigh:true` and
`parttoolgenBenchmarkExtraActors:[]` under `task.env`; the network configuration
is unchanged. See [configuration diff](policy_config.diff) and
[runtime setup](../reproducibility.md).

The inference integration preserves native observation/action conversion,
normalization, and LSTM state. Simulation code is scoped to fixed-goal inference,
with training/logging dependencies removed. Numerical inference and compiler
comparisons do not imply identical complete rollouts to upstream or historical
runners. Shared settings, ray-plane lifting, explicit-axis handling, endpoint
calculation, and execution orchestration differ from earlier experiment code.

## Current limitations

- Online VLM selection/grounding and new-mesh P3SAM inference have not been
  validated here. The eight prepared tools can be used without rerunning P3SAM.
- Red brush and short screwdriver control sequences time out in the recorded run.
- Goal files do not enforce association with their source task/tool; keep them
  paired. Reusing a failed planning directory can leave older goal files.
- Priors generated outside the project root retain absolute paths. Generate
  inside ignored `outputs/` when relocating the whole project.
