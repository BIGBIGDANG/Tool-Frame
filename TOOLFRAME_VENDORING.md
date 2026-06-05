# ToolFrame Vendoring And Runtime Assets

This repository keeps a small ToolFrame handoff layer plus vendored runtime
backends needed by the current SimToolReal rollout path.

## Vendored Backend

`simtoolreal/` is treated as a vendored backend. The current ToolFrame runners
import it through `parttoolgen/scripts/parttoolgen_env.sh`, which adds the
following source roots to `PYTHONPATH`:

```text
simtoolreal/
simtoolreal/isaacgymenvs/
simtoolreal/rl_games/
```

Do not delete these runtime source trees while the SimToolReal fixed-goal
rollout backend is active:

```text
simtoolreal/isaacgymenvs/
simtoolreal/rl_games/rl_games/
simtoolreal/assets/urdf/dextoolbench/
simtoolreal/assets/urdf/parttoolgen_benchmark/
```

Local install byproducts are not source and should stay out of releases:

```text
simtoolreal/*.egg-info/
simtoolreal/rl_games/*.egg-info/
simtoolreal/train_dir/
```

## External Runtime Dependencies

These are machine-local runtime dependencies, not redistributable ToolFrame
source files:

```text
IsaacGym
pretrained rollout policies
VLM/API credentials
SAMPart3D/P3SAM model weights
large RGB-D, rollout, and video outputs
```

Configure local paths through `.parttoolgen.local.env`, using
`.parttoolgen.local.env.example` as the template.

## Evidence Bundles

Source-of-truth benchmark definitions live in:

```text
artifacts/tool_use_benchmark/
configs/tool_refs/
```

Optional evidence/data bundles live in:

```text
artifacts/tool_prior/
artifacts/image_grounded/tool_ref_grounding/
artifacts/visualizations/
```

Only publish selected compact evidence with manifests and checksums. Full
rollout videos, RGB-D dumps, raw VLM responses, and temporary reports should be
regenerated or distributed separately from the source release.

## Reference Code

`references/ReKep/` is reference-only. It is not a ToolFrame entrypoint and is
not imported by the public `bin/toolframe` commands.

Historical reports under `reports/mainline_6.4/` are provenance snapshots from
earlier reconstruction work. Regenerate paper results through
`bin/toolframe paper-release` before reporting success rates.
