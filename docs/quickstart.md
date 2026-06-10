# Quickstart

This quickstart starts from a fresh ToolFrame checkout and follows the current
full end-to-end mainline.

## 1. Environment

```bash
cd ToolFrame_6.4
bin/toolframe bootstrap-env
bin/toolframe env-check
```

The bootstrap creates `.venv_parttoolgen_py38` under the repo and installs the
core Python 3.8 dependencies from `requirements.txt`.

Optional rollout/VLM layers:

```bash
python -m pip install -r requirements-rollout.txt
python3.10 -m pip install --target .planner_vlm_py310_pkgs -r requirements-vlm.txt
```

Workstation-specific IsaacGym paths, policy weights, VLM endpoints, and API
keys belong in `.parttoolgen.local.env`. Do not commit that file.

## 2. Inspect The Current Suite

```bash
bin/toolframe list
bin/toolframe preflight-current
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
```

The current suite is `current_all_runnable`. It is the active E2E benchmark
scope for brush sweep, eraser wipe, hammer strike, screwdriver insert, and
marker trace tasks.

## 3. Run A Small Smoke

```bash
bin/toolframe benchmark \
  --task brush_sweep/off_table_red \
  --trials 1 \
  --device cuda:0 \
  --capture-video-every 1 \
  --continue-on-task-error \
  --tool-refs-mode vlm_per_seed \
  --tool-ref-model gpt-5.5 \
  --tool-ref-base-url "$VLM_API_BASE"
```

This runs the current method path:

```text
tool-only VLM tool refs
  -> per-seed VLM scene refs
  -> RGB-D lifting
  -> whole-tool SE(3) compiler
  -> SimToolReal rollout
  -> task success check
```

## 4. Run The Current E2E Suite

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

For multiple GPUs, split tasks into groups but keep the same protocol and
output prefix.

## 5. Outputs

Keep runtime outputs outside benchmark config directories:

```text
artifacts/test008/                       current E2E experiment roots
artifacts/image_grounded/<task_family>/  selected benchmark evidence roots
test000/                                 local scratch/review outputs
```

Do not place runtime images, raw VLM JSON, rollout videos, or summaries under
`artifacts/tool_use_benchmark/`.

## 6. Interpret Results

The current result is final E2E task success. A trial succeeds only when the
task outcome is achieved through the expected functional tool interaction.

Manual failure attribution labels:

```text
reference
compilation
execution
artifact
needs_review
```

Fixed/oracle-ref baselines, dense module rates, and detailed compiler metrics
are optional future analyses, not the current quickstart protocol.

## 7. Before Publishing Changes

```bash
git diff --check
bin/toolframe check --quick
bin/toolframe preflight-current
```

Run expensive Isaac/rollout smokes only when validating a specific benchmark
contract, environment asset, or runner change.
