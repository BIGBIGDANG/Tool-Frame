# PartToolGen Providers

This package is the home for provider adapters. Providers produce grounded
artifacts with explicit provenance; they do not write planner programs and they
do not directly command SimToolReal.

Subpackages:

```text
perception/
  Legacy/baseline perception adapters: RGB-D/SAM, pose trackers, registration,
  mask trackers, keypoints, regions, surfaces, and target geometry.

tool_prior/
  Tool mesh/URDF/CAD/SAMPart3D or calibrated functional-part priors.

environment/
  Simulation or robot execution setup, object asset names, table pose,
  reset poses, and rollout scene configuration.

simulation/
  SimToolReal environment/task config providers. They may expose benchmark task
  symbols and must mark any privileged oracle output as debug.

feedback/
  Execution feedback adapters for tracking error, contacts, force/tactile
  signals, visual verification, and failure reports.
```

Existing code in `parttoolgen.perception` remains in place for compatibility.
New code should enter this package when it represents a provider boundary rather
than a low-level RGB-D utility.
