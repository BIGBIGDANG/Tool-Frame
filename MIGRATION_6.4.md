# ToolFrame 6.4 Migration Notes

This folder is the cleaned 2026-06-04 handoff of the ToolFrame mainline.
It keeps the code, configs, benchmark definitions, current tool refs, and
small reproducibility evidence needed to run and review the current pipeline.

## Kept

- `parttoolgen/`: source, package CLIs, public scripts, tests, schemas, and Python 3.8 requirements.
- `simtoolreal/`: local rollout adapter code and benchmark assets needed by the current tasks.
- `configs/tool_refs/`: publishable current tool refs for `claw_hammer`, `red_brush`,
  `flat_eraser`, `long_screwdriver`, and `staples_marker`.
- `artifacts/tool_use_benchmark/`: benchmark contracts/configs and tool-label review files.
- `artifacts/tool_prior/`: compact P3SAM tool-prior packages used to regenerate tool refs.
- `artifacts/visualizations/`: small reference visualizations only.
- `reports/mainline_6.4/`: audit table plus the five-tool tool-selection evidence
  structure check with real overview/card/face-highlight images.
- `.parttoolgen.local.env.example`: template only; local secrets and machine paths were not copied.

## Excluded

- `.git/`, virtual environments, cache directories, and Python bytecode.
- Local private env files such as `.parttoolgen.local.env`.
- `artifacts/external/`, including the 20G external ToolUse archive.
- Historical `artifacts/image_grounded/*` rollout outputs.
- Large `test000/`, `test001/`, and `test002/` runtime trees, except the small copied
  review evidence under `reports/mainline_6.4/`.

## Verify

Create a local environment:

```bash
bash parttoolgen/scripts/bootstrap_parttoolgen_env.sh
source parttoolgen/scripts/parttoolgen_env.sh
parttoolgen/scripts/run_checks.sh --quick
```

During migration, the copied tree was checked with the existing 5.21 Python 3.8
environment:

```bash
bin/toolframe check --quick
bin/toolframe check --full
```

Latest local result after the 6.4 cleanup: quick `44 passed`; full `44 passed`
plus `248 passed`.
