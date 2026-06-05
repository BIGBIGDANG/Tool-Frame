# Tests

The test suite is intentionally broader than the public CLI surface. Do not
delete tests just because the user-facing entrypoint is now `bin/toolframe`.

Recommended ownership groups:

- `test_cli_*`, `test_run_*`, and `test_scripts_inventory.py`: public wrapper,
  registry launcher, spec runner, and script-boundary checks.
- `test_image_grounded_*` and `test_mainline_pipeline.py`: image-grounded
  refs, lifting, staged mainline, and current runner compatibility.
- `test_tool_pose_compiler.py`, `test_tool_path_solver.py`, and
  `test_simtoolreal_adapter.py`: compiler/path-solver/backend contracts.
- `test_task_success_checks.py`, `test_failure_attribution.py`, and
  reporting/evaluation tests: task success and benchmark audit logic.
- `test_affordance_*`, `test_given_function_region_planner.py`, and repair
  prompt tests: legacy/backend compatibility. Keep them passing, but do not use
  them as the extension API for new image-grounded tasks.

Future physical reorganization can move these files into subdirectories, but
only after `run_checks.sh` and any CI path filters are updated in the same
change.

