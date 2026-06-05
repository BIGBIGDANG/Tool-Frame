# Scripts

`parttoolgen/scripts` is now a small public handoff layer. It should not contain
task-family runners, VLM stages, tool-prior stages, local demos, or one-off
experiment scripts.

Use `bin/toolframe` first:

```text
bin/toolframe bootstrap-env
bin/toolframe env-check
bin/toolframe check --quick
bin/toolframe preflight-headline
bin/toolframe preflight-paper
bin/toolframe benchmark --suite current_all_runnable --preflight-only --skip-rollout
bin/toolframe tool-ref --help
bin/toolframe run-spec --task brush_sweep/red_zone --preflight-only
```

The only stable files in this directory are listed in
`script_inventory.json`:

```text
bootstrap_parttoolgen_env.sh
parttoolgen_env.sh
check_toolframe_env.py
run_checks.sh
run_image_grounded_task_spec.py
run_p3sam_tool_ref_pipeline.py
run_tool_use_benchmark.py
```

Internal command-line modules live in the package instead:

```text
parttoolgen/src/parttoolgen/cli/runners/      registry task-family runners
parttoolgen/src/parttoolgen/cli/stages/       RGB-D, VLM, lift, merge, compile stages
parttoolgen/src/parttoolgen/cli/tool_prior/   P3SAM evidence and tool-ref selection stages
parttoolgen/src/parttoolgen/cli/aux/          inventory and reporting helpers
parttoolgen/dev_scripts/                      local demos and debug-only commands
```

Do not add a new file here for a new task, scene, or tool. Add registry/spec
data and package code, then invoke it through `bin/toolframe benchmark`,
`bin/toolframe run-spec`, or `bin/toolframe tool-ref`.

Marker-trace ink rendering, marker VLM judging, and side-board HDU erase
success checking are package modules under `parttoolgen.evaluation`; runners
call them with `python -m ...` instead of shim scripts.

Candidate part cards, selected-part bbox overviews, selected-part face
highlights, and presentation crops should still come from the 5.15-style
renderer in `parttoolgen.cli.tool_prior.render_sampart3d_part_role_evidence`.
Do not reintroduce simplified local mesh/SVG exporters or per-face candidate
choice figures.
