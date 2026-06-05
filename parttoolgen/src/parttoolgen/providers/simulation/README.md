# Simulation Providers

Simulation providers expose benchmark environment/config symbols for
experiments. They can supply deployable-style artifacts such as target
geometry, support surfaces, task labels, and collision geometry from configured
task assets.

Outputs must distinguish:

```text
simulated_perception_provider
  Used to evaluate planning with provider-style inputs.

simulator_oracle_debug
  Privileged upper-bound/debug data, not a deployable main result.
```
