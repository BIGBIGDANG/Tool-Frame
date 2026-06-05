# Reports

`reports/` stores compact audit/provenance reports, not benchmark source of
truth and not final paper success-rate tables.

Current contents:

- `mainline_6.4/`: migrated historical audit tables and tool-selection evidence
  checks. These summarize earlier 5.21/test002 experiments and should be used
  only for provenance while rebuilding ToolFrame 6.4.

For ICRA/RA-L statistics, regenerate a fresh report from the registered
ToolFrame 6.4 headline subset:

```bash
bin/toolframe preflight-headline
bin/toolframe paper-headline \
  --trials 20 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md \
  --aggregate-output-root reports/paper_release_6.4/headline
```

Use `bin/toolframe paper-release` for the broader candidate evidence suite.

Do not copy rollout videos, RGB-D dumps, raw VLM responses, or compiler working
directories into `reports/`; reports should contain compact summaries,
manifests, tables, and review notes only.
