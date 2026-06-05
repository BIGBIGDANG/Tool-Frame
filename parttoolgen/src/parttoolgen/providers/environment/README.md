# Environment Providers

Environment providers own setup data for simulation or robot execution:
object names, reset poses, table pose, target assets, randomization settings,
and visualization options.

They may use simulator state for setup and logging, but their outputs must not
be mixed into perception provenance unless explicitly marked as simulator debug
or upper-bound evidence.
