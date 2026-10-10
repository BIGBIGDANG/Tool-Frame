# Contributing

For bug reports, include the command, task configuration, Python/dependency
versions, and relevant output JSON or traceback. For GPU issues, include the
GPU, driver, and IsaacGym version. Remove API keys before sharing logs.

Use Python 3.8 and run the software checks from the repository root:

```bash
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Keep changes focused and use shared `settings.json` rather than task-name
branches. Preserve calibrated RGB-D pairing, geometry units, and the backend's
observation/action conventions. Describe any changes to these contracts.

For experiments, record input annotations, settings, runtime versions, and
control outcomes. Assess task effects separately from controller completion.
Keep generated runs in ignored `outputs/`; SDKs, weights, and credentials stay
outside source commits. GPU setup is in [reproducibility](docs/reproducibility.md).
