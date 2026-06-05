# Mainline 6.4 Historical Audit

This directory is a migrated audit snapshot from earlier ToolFrame 5.21/test002
work. It is useful for checking provenance and tool-selection evidence layout,
but it is not the ToolFrame 6.4 paper-release result table.

Known limitations:

- Many summary/video paths point to historical `test002` roots.
- Some entries keep old absolute workstation paths for provenance.
- The table mixes fixed/current, provisional, draft, and prompt/asset-debug
  runs.

Regenerate publishable results from the current registry instead:

```bash
bin/toolframe paper-headline \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md \
  --aggregate-output-root reports/paper_release_6.4/headline
```
