# Fixed scene observations

Each task directory contains paired simulation RGB-D, camera calibration, and
manual pixel annotations used in the recorded nine-scene run. The annotations
identify their source in `fixture_source`; they are not VLM predictions.

The images and keypoints are inputs for studying lifting, geometry compilation,
and policy execution. They do not establish task success or evaluate online grounding.

## Contents

| Task ID | Tool reference |
| --- | --- |
| `blue_brush_sweep_block` | `blue_brush.json` |
| `red_brush_sweep_block` | `red_brush.json` |
| `claw_hammer_strike` | `claw_hammer.json` |
| `mallet_hammer_strike` | `mallet_hammer.json` |
| `handle_eraser_tabletop` | `handle_eraser.json` |
| `handle_eraser_whiteboard` | `handle_eraser.json` |
| `long_screwdriver_insert` | `long_screwdriver.json` |
| `short_screwdriver_insert` | `short_screwdriver.json` |
| `marker_whiteboard` | `staples_marker.json` |

Each directory has `scene_evidence.json`, `scene_rgb.png`, `scene_depth.npy`,
`scene_camera.json`, and `scene_keypoints.json`. Tool references live under
`assets/tool_refs/`; task configurations live under `examples/`.

## Compile all nine scenes on CPU

From the repository root with core dependencies installed:

```bash
for task in examples/*.json; do
  task_id=$(basename "$task" .json)
  tool_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tool"]["tool_id"])' "$task")
  python run.py plan \
    --task "$task" \
    --scene-evidence "examples/observations/$task_id/scene_evidence.json" \
    --scene-keypoints "examples/observations/$task_id/scene_keypoints.json" \
    --tool-refs "assets/tool_refs/$tool_id.json" \
    --output "outputs/cached_plans/$task_id" || exit 1
done
```

For GPU execution, follow [runtime setup](../../docs/reproducibility.md) and
execute each generated goal file with its matching task JSON. Use a fresh
output directory for reruns. If rendering a new camera/object layout, annotate
new keypoints rather than reusing the fixed pixels.
