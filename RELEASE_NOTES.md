# ToGo 6.13 research preview

This version replaces the previous Tool-Frame placeholder implementation with
the prepared ToGo handoff. Repository history is retained; the previous CLI and
benchmark workflow are replaced by `run.py`, shared `settings.json`, and CPU CI.

Verified before publication:

- 107 software regression tests pass in a fresh Python 3.8 environment.
- SimToolReal pretrained model bytes match the official download. For 128
  fixed observations, native and simplified players produce identical actions
  and LSTM states. The local configuration retains two original environment
  fields beyond the official download configuration.
- For nine identical compiler inputs, original and handoff goal poses match.
- A complete local directory copy runs fresh GPU rendering and native policy
  execution; the mallet example reaches 26/26 control goals in 445 steps.
- Nine GPU scenarios compile and start; seven control sequences complete.
  Red brush and short screwdriver time out. Control completion does not
  establish task-effect success.

Online VLM and fresh-mesh P3SAM segmentation have not been accepted in this
validation. The eight prepared tools do not require running P3SAM again.
Use a fresh output directory for each run and pair goals with the matching
task/tool. Standalone relocation of priors generated outside the project root
still requires regeneration.

SDK and policy files are excluded from Git. Follow
[NON_VLM_REPRO.md](NON_VLM_REPRO.md) for runtime setup, and
[docs/validation/REPORT.md](docs/validation/REPORT.md) for detailed evidence.
The project root license and some asset attribution remain unresolved, as
recorded in [LICENSES.md](LICENSES.md) and [THIRD_PARTY.md](THIRD_PARTY.md).
