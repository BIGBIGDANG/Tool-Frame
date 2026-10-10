# Runtime setup and reproduction

The [CPU quick start](../README.md#quick-start-cpu-planning) runs planning with
included RGB-D and manual keypoints. GPU simulation additionally requires
IsaacGym and the official SimToolReal pretrained policy. This implementation
uses the legacy IsaacGym runtime, not upstream SimToolReal's newer Isaac Sim setup.

## Environment

The tested runtime uses Linux, Python 3.8.20, PyTorch 2.4.1+cu121, and an NVIDIA
RTX 4090 D. A compatible NVIDIA driver and C++ compiler are required. Start
from an activated Python 3.8 environment in the repository root:

```bash
python -m pip install -r requirements-runtime.txt
```

`requirements-dev.txt` adds CPU tests. `requirements-tested.txt` records the
complete tested environment, including development dependencies; install it
instead of the smaller requirements files to use those exact package versions.

Obtain IsaacGym Preview 4 from [NVIDIA](https://developer.nvidia.com/isaac-gym)
and extract it outside the repository. The upstream
[IsaacGym installation guide](https://github.com/tylerlum/simtoolreal/blob/main/docs/isaacgym_installation.md)
describes the legacy SDK. Point the runtime at its Python directory:

```bash
export TOGO_ISAACGYM=/path/to/isaacgym/python
export TOGO_RUNTIME_PYTHON="$(command -v python)"
```

IsaacGym needs a loadable `libpython3.8.so`. If the Python library directory
is not on the dynamic linker search path, add it for the current shell:

```bash
export LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The first run compiles `gymtorch`; `ninja` is included in the runtime requirements.
If planning and simulation use separate environments, `TOGO_RUNTIME_PYTHON`
or `--runtime-python` selects the simulation interpreter.

## Pretrained policy

Download the archive used by the official
[SimToolReal downloader](https://github.com/tylerlum/simtoolreal/blob/main/download_pretrained_policy.py):

```bash
curl --fail --location \
  https://download.cs.stanford.edu/juno/simtoolreal/pretrained_policy.zip \
  --output /tmp/simtoolreal-pretrained.zip
unzip /tmp/simtoolreal-pretrained.zip -d .
export TOGO_PRETRAINED_POLICY_ROOT="$PWD/pretrained_policy"
```

The extracted directory must contain `model.pth` and `config.yaml`.
The validated model SHA256 is:

```text
4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9
```

The recorded experiments use the official configuration with two additional
environment fields, leaving the network unchanged. To reproduce that configuration:

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
path = Path(os.environ["TOGO_PRETRAINED_POLICY_ROOT"]) / "config.yaml"
cfg = yaml.safe_load(path.read_text())
cfg["task"]["env"].update(resetWhenTableForceTooHigh=True, parttoolgenBenchmarkExtraActors=[])
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

SDKs, policy weights, and segmentation models are external resources. The
repository contains prepared tool priors, physical scene assets, and nine
complete RGB-D/keypoint examples. `python run.py doctor` reports dependencies
and resource locations for its current interpreter; it does not run a GPU test.

## Execute without VLM calls

With the runtime configured, the following uses the included observation,
manual keypoints, and cached tool references:

```bash
python run.py run \
  --task examples/mallet_hammer_strike.json \
  --scene-evidence examples/observations/mallet_hammer_strike/scene_evidence.json \
  --scene-keypoints examples/observations/mallet_hammer_strike/scene_keypoints.json \
  --tool-refs assets/tool_refs/mallet_hammer.json \
  --output outputs/mallet_cached_run --device cuda:0
```

A fresh-environment GPU run reached 26/26 control goals in 445 steps. This is
a recorded controller outcome, not a guarantee of identical physics or a
measurement of strike success.

The other eight scenes have matching cached inputs under
`examples/observations/<task_id>/`. Use the matching scene JSON and tool reference;
[example observations](../examples/observations/README.md) includes a batch planning command.

To regenerate the scene RGB-D before running, first render:

```bash
python ToGo/src/ToGo/cli/stages/render_simtoolreal_rgbd_scene_evidence.py \
  --task examples/mallet_hammer_strike.json \
  --output-dir outputs/mallet_observation --device cuda:0
```

Then use `outputs/mallet_observation/scene_evidence.json` for `--scene-evidence`.
The fixed pixels apply to the example's camera and object layout; annotate new
keypoints if either changes.

## Online grounding

Set `VLM_MODEL` and `OPENAI_API_KEY` (or `VLM_API_KEY`) for an image-capable model
supporting JSON output. `VLM_API_BASE` selects a compatible endpoint. Omit the
three cached-input flags from `run` to use online selection and grounding.
Online behavior is not part of the recorded validation.

## Reporting experiments

Record the commit, task JSON, shared settings, RGB-D calibration, grounding
annotations, tool references, dependency versions, and policy hash. Evaluate
task effects separately from goal progression. The
[validation page](validation/README.md) reports the available checks and their scope.
