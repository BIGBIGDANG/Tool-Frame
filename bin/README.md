# Public ToolFrame Entrypoints

`bin/` is the handoff layer. New users, CI, and benchmark operators should
start here instead of calling task-family runners or stage scripts directly.

```bash
bin/toolframe list
bin/toolframe preflight-current
bin/toolframe current-e2e --trials 10 --device cuda:0 --continue-on-task-error
bin/toolframe benchmark --task brush_sweep/off_table_red --trials 1 --device cuda:0
bin/toolframe tool-ref --help
bin/toolframe inventory --compact
bin/toolframe run-spec --task brush_sweep/off_table_red --preflight-only
bin/toolframe check --quick
```

The wrapper is intentionally thin. It delegates to stable public scripts under
`parttoolgen/scripts/`, and those public scripts call the lower-level registry
runners, stage CLIs, compiler modules, and evaluation modules. Do not put
pipeline logic in `bin/`; add logic to package modules or the appropriate
runner/stage, then expose it through the existing wrapper command.

Current E2E experiments should use `current-e2e` or the explicit equivalent
`benchmark --suite current_all_runnable --tool-refs-mode vlm_per_seed`. Single
task debugging should use `benchmark --task ...` with the same method flags.

New task/scene work should start from a unified task spec and `run-spec`.
Family-specific runners are compatibility backends for current benchmark
splits, not the extension API for new task families. Historical paper/headline
suite aliases may remain in the registry for old reports, but they are not the
current public run path.
