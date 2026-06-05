# Public ToolFrame Entrypoints

`bin/` is the handoff layer. New users, CI, and benchmark operators should
start here instead of calling task-family runners or stage scripts directly.

```bash
bin/toolframe list
bin/toolframe preflight-headline
bin/toolframe paper-headline --trials 20 --device auto --continue-on-task-error
bin/toolframe preflight-paper
bin/toolframe paper-release --trials 20 --device auto --continue-on-task-error
bin/toolframe preflight-all
bin/toolframe benchmark --task brush_sweep/red_zone --trials 1 --device auto
bin/toolframe run-all --trials 1 --device auto --continue-on-task-error
bin/toolframe tool-ref --help
bin/toolframe inventory --compact
bin/toolframe run-spec --task brush_sweep/red_zone --preflight-only
bin/toolframe check --quick
```

The wrapper is intentionally thin. It delegates to stable public scripts under
`parttoolgen/scripts/`, and those public scripts call the lower-level registry
runners, stage CLIs, compiler modules, and evaluation modules. Do not put
pipeline logic in `bin/`; add logic to package modules or the appropriate
runner/stage, then expose it through the existing wrapper command.

New task/scene work should start from a unified task spec and `run-spec`.
Family-specific runners are compatibility backends for current benchmark
splits, not the extension API for new task families.

Use `paper-headline` for the currently reviewed headline subset,
`paper-release` for the broader ToolFrame 6.4 paper-review candidate suite, and
`run-all`/`preflight-all` for engineering smoke over every current non-legacy
launcher-runnable task. Legacy directional brush checks are available through
the `legacy_directional_diagnostics` suite when regression attribution needs
them.
