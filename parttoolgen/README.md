# ToolFrame Runtime Package

This package contains the ToolFrame runtime and image-grounded interfaces used
by the current benchmark path.

The canonical image-grounded tabletop and side-board workflow plus method
boundary are recorded in `../docs/current_pipeline.md`. New benchmark work
should follow that document rather than older private-eval or repair-loop
interfaces.

Use the public wrapper for normal commands:

```bash
bin/toolframe env-check --mode core
```

For direct debugging of package scripts, source the lower-level environment
wrapper first:

```bash
source parttoolgen/scripts/parttoolgen_env.sh
```

New mainline code should enter through:

```text
parttoolgen.image_grounded
```

Legacy-compatible compiler, checks, evaluation, and SimToolReal modules remain
available while the 5.21 mainline is migrated. Modules that mention
`private_eval`, provider target boxes, or repair-loop feedback are legacy
evaluation/support surfaces only; they are not active method inputs for the
image-grounded benchmark.

## Checks

```bash
bin/toolframe env-check --mode core
python -m pytest parttoolgen/tests/test_image_grounded_lifting.py
```
