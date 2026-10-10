# Working on Tool-Frame / ToGo

Read AGENTS.md first and follow these rules before editing or running experiments.

- Make direct changes for verified requirements. Avoid duplicate validation,
  speculative fallbacks, new compatibility layers, or unrelated refactors.
- Use `run.py` and shared `settings.json`. Do not branch methods on task names
  or introduce evaluator targets as planning inputs.
- Preserve SimToolReal observation/action conventions and geometric constraints.
- Use Python 3.8. Install CPU dependencies with
  `python -m pip install -r requirements-dev.txt`; verify software changes with
  `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q`.
- For GPU execution and external resources, follow NON_VLM_REPRO.md.
- Keep generated outputs in ignored `outputs/` or outside the repository.
  Do not commit SDKs, weights, credentials, virtual environments, or logs.
- A completed control sequence is not proof of task-effect success. Keep
  reported verification and unverified behavior distinct.
- Include this file, concrete success criteria, and expected verification in
  any task handed to another coding agent.
