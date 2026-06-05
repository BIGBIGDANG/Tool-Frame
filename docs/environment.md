# Environment

ToolFrame 6.4 is the active source tree and experiment workspace. Core Python
dependencies should be recreated from the checked-in bootstrap script and
requirements. Large runtime dependencies should come from ToolFrame-local
ignored artifacts when present. Older 5.15 paths are optional local runtime
fallback dependencies only when explicitly configured in `.parttoolgen.local.env`;
do not use them as the source/config workspace.

```text
ToolFrame_6.4/                          active source, configs, tool refs, assets
ToolFrame_6.4/artifacts/...             preferred local runtime artifacts
PartCentricAffordanceProgram_5.15/       fallback runtime env/IsaacGym/weights only
```

The public handoff command is:

```bash
bin/toolframe bootstrap-env
bin/toolframe env-check --mode core
```

For a paper-code style manual install, the root requirements files are:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-rollout.txt  # optional GPU rollout layer
# requirements-vlm.txt is for a separate local planner environment/target.
```

`requirements.txt` is the default lightweight install target. The rollout and
VLM files are intentionally separate because they pull GPU/model/runtime
dependencies that many readers do not need for compiler or benchmark-contract
inspection.

Internally, `bin/toolframe` delegates environment setup/checks to the
lower-level scripts in `parttoolgen/scripts/`. Those scripts are implementation
details for the public wrapper and direct use should be limited to debugging.

It sets:

- `PARTTOOLGEN_ROOT` to `ToolFrame_6.4`;
- `PARTTOOLGEN_SRC` to `ToolFrame_6.4/parttoolgen/src`;
- `PARTTOOLGEN_CORE_VENV` to `ToolFrame_6.4/.venv_parttoolgen_py38` when it
  exists, otherwise the explicitly configured fallback Python 3.8 env;
- `PARTTOOLGEN_ISAACGYM` to `ToolFrame_6.4/artifacts/third_party/isaacgym/python`
  when it exists, otherwise the configured fallback IsaacGym install.
- `PARTTOOLGEN_SIMTOOLREAL_SITE` to the project-local SimToolReal site package
  path when it exists, otherwise the configured fallback site package path.

Run the environment check:

```bash
bin/toolframe env-check --mode core
```

Use `--mode rollout` to additionally require IsaacGym, SimToolReal import
paths, and pretrained policy roots. Use `--mode vlm` to require VLM
credentials, and `--mode full` before launching a complete RGB-D/VLM/rollout
benchmark suite.

The project-local venv is intentionally ignored by git. Recreate it with:

```bash
bin/toolframe bootstrap-env
```

To install optional layers into the same venv during bootstrap:

```bash
PARTTOOLGEN_INSTALL_ROLLOUT=1 bin/toolframe bootstrap-env
```

The root `pyproject.toml` also supports editable source installs:

```bash
pip install -e .
```

Use the requirements files for dependency installation; editable install only
registers the `parttoolgen` package from `parttoolgen/src`.

By default this installs the core compiler/test dependencies only. Full
SimToolReal/IsaacGym rollout requires separately packaged runtime artifacts:
IsaacGym binaries, pretrained policy weights, GPU-compatible torch packages,
and VLM credentials. Those should be supplied through project-local ignored
artifact directories or explicit `.parttoolgen.local.env` overrides.

Local/open VLM planner packages should be installed into an isolated planner
environment or target directory:

```bash
python3.10 -m pip install --target .planner_vlm_py310_pkgs -r requirements-vlm.txt
```

Secrets are not copied. If a workstation needs a legacy VLM env file or runtime root, set it explicitly in `.parttoolgen.local.env`.

Active benchmark scripts and SAMPart3D/P3SAM visual-evidence renderers are kept
inside this repository. Agents should not call scripts from another workspace
checkout. Large runtime artifacts such as Python environments, IsaacGym,
pretrained policies, and model weights may still be provided through explicit
environment variables while they are being packaged locally.

Preferred local paths for newly packaged runtime artifacts:

```text
PARTTOOLGEN_CORE_VENV=$TOOLFRAME_ROOT/.venv_parttoolgen_py38
PARTTOOLGEN_ISAACGYM=$TOOLFRAME_ROOT/artifacts/third_party/isaacgym/python
PARTTOOLGEN_SIMTOOLREAL_SITE=$TOOLFRAME_ROOT/.venv_simtoolreal/lib/python3.8/site-packages
PARTTOOLGEN_P3SAM_ROOT=$TOOLFRAME_ROOT/artifacts/models/Hunyuan3D-Part/P3-SAM
```

If a runtime dependency still lives outside this repo, set it in
`.parttoolgen.local.env`; do not hard-code external workspace paths inside
benchmark scripts. Imported external asset sets are outside the current public
release unless the project owner approves a separate asset/license review.

For CI or a clean reproducibility check, skip workstation overrides:

```bash
PARTTOOLGEN_SKIP_LOCAL_ENV=1 bin/toolframe env-check --mode core
```

For ordinary source/config changes, use the core-only quality gate:

```bash
bin/toolframe check --quick
```

Use `bin/toolframe check --full` before release-style changes when the full
unit-test suite should run locally.
