# Tool Prior Providers

Tool prior providers define contact-point parts in a stable tool frame. Sources may
include URDF/CAD/mesh annotations, SAMPart3D regions, offline segmentation, or
calibrated human-approved priors.

For SimToolReal execution, tool priors must state whether their frame is the
SimToolReal asset/root frame or another registered tool frame.

## Responsibility Boundary

The tool-prior lane is responsible for tool-side geometry refs only:

```text
tool mesh / segmentation
-> tool part refs
-> candidate contact points
-> candidate surface normals / axes
-> visual evidence for choosing those refs
```

It must not introduce task primitives, tool families, task families, hidden
target refs, fallback repair loops, or executable robot goals. Scene refs,
action refs, constraint programs, SE(3) compilation, and SimToolReal rollouts
belong downstream.

## Mesh-Label Tool Prior Export

The generic mesh-label provider exports each segmentation region plus its part
center and six surface-center contact candidates in the stable SimToolReal asset
frame. These candidate refs are exposed as `candidate.points.*` plus matching
`candidate.surfaces.*` refs.

```bash
python -m parttoolgen.providers.tool_prior.sampart3d_tool_prior_provider \
  --tool-id claw_hammer \
  --source-mesh simtoolreal/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.obj \
  --face-labels artifacts/tool_prior/sampart3d_scale2/claw_hammer/face_labels/mesh_2.0.npy \
  --output-dir artifacts/tool_prior/sampart3d_scale2/claw_hammer/tool_prior
```

It writes `tool_part_graph.json`, `functional_tool_parts.json`,
`grounding_candidates.json`, `provider_contract.json`, and `manifest.json`
under `tool_prior/`.

The same builder is used for P3SAM by setting `segmentation_mode=p3sam`.

## P3SAM-First Tool-Prior Pipeline

The clean image-grounded pipeline uses P3SAM as a tool-side provider. This
provider lane builds structured tool geometry refs; the public visualization
lane should render only 5.15-style part-overview evidence.

```text
P3SAM auto_mask on tool mesh
-> p3sam_face_labels.npy
-> ToolPartGraph / FunctionalToolParts / GroundingCandidates
-> render 5.15-style part visual evidence
-> VLM selects contact-point part marker
-> render selected-part bbox face evidence with highlighted F markers
-> VLM selects F contact/motion-axis markers
-> axis_binding.json + contact_point_selection.json
-> merge into image-grounded constraint program
```

Part-overview evidence can be precomputed for every tool because it is purely
tool-side geometry. Functional point and tool-axis labels remain
task-conditioned: the same segmented tool may choose different refs for a
different instruction, so those refs are exported only after the VLM sees the
task wording.

The key rule is that public tool-side figures use 5.15-style part overview,
part-card, selected-part bbox overview, and selected-part face-highlight
evidence. F markers and highlighted bbox faces are kept because the VLM needs
them for task-conditioned contact/motion-axis selection, but they appear only
after the part is selected. Legacy per-face/contact-choice images such as
`contact_point_choices/candidate_*_F*_face.png` must not be generated.

For handoff users, prefer the unified wrapper:

```bash
python parttoolgen/scripts/run_p3sam_tool_ref_pipeline.py \
  --tool-id long_screwdriver \
  --task-id screwdriver_insert_deep_socket \
  --task-instruction "insert the screwdriver tip into the socket" \
  --run-root $TOOLFRAME_ROOT/artifacts/tool_prior/p3sam \
  --output-dir $TOOLFRAME_ROOT/configs/tool_refs/long_screwdriver
```

### 1. Build P3SAM Tool Prior

```bash
source parttoolgen/scripts/parttoolgen_env.sh
CUDA_VISIBLE_DEVICES=1 "$PARTTOOLGEN_CORE_PYTHON" \
  parttoolgen/src/parttoolgen/cli/tool_prior/build_p3sam_tool_prior.py \
  --tool-id long_screwdriver \
  --source-mesh simtoolreal/assets/urdf/dextoolbench/screwdriver/long_screwdriver/long_screwdriver.obj \
  --output-root artifacts/tool_prior/p3sam/long_screwdriver_20260524 \
  --observation-id long_screwdriver_p3sam_auto_mask_20260524 \
  --prompt-bs 8 \
  --parallel 0
```

This produces:

```text
artifacts/tool_prior/p3sam/<tool_id>_<date>/p3sam_raw/
artifacts/tool_prior/p3sam/<tool_id>_<date>/face_labels/p3sam_face_labels.npy
artifacts/tool_prior/p3sam/<tool_id>_<date>/tool_prior/
```

If the P3SAM raw output already exists, use `--skip-p3sam-run --p3sam-raw-dir
...` to rebuild only the ToolFrame prior.

### 2. Render Part-Overview Evidence

The first VLM prompt should show anonymous part candidates in full-tool
context. After the VLM selects the part, the second prompt should show only the
selected part's highlighted bbox faces and F markers. Neither prompt should
contain one-image-per-face candidate cards.

```bash
python parttoolgen/src/parttoolgen/cli/tool_prior/render_sampart3d_part_role_evidence.py \
  --run-root $TOOLFRAME_ROOT/artifacts/tool_prior/p3sam_render_ready \
  --tool-id long_screwdriver \
  --prior-subdir tool_prior \
  --output-dir $TOOLFRAME_ROOT/configs/tool_refs/long_screwdriver/evidence/01_part_selection \
  --view-selection uniform \
  --part-overview-views 3 \
  --skip-face-cards
```

The rendered `visual_evidence.json` contains a `region_marker_table`. The VLM
chooses one `selected_marker` from that table.

### 3. VLM Selects Functional Part

The part selection prompt should receive the task wording and the overview
image, and return only:

```json
{
  "selected_marker": "1",
  "confidence": 0.0,
  "reason": "short visual/task reason"
}
```

### 4. VLM Selects Function Point / Motion Axis And Exports Refs

The selected-part prompt should return:

```json
{
  "selected_contact_point_marker": "F1",
  "selected_motion_axis_marker": "F2",
  "confidence": 0.0,
  "reason": "short visual/task reason"
}
```

The exporter writes:

```text
axis_binding.json
axis_mesh_overlay_report.json
contact_point_selection.json
tool_part_graph.json
p3sam_tool_ref_selection_summary.json
evidence/
```

`evidence/` is the human-facing package for one task-conditioned VLM run:
`01_part_selection/` holds the part overview and part cards, and
`02_contact_marker_selection/` holds the selected part bbox plus one highlighted
face image per F marker. The JSON refs consumed downstream live at the tool-ref
root. The original `artifacts/visualizations` directory remains only a renderer
cache/provenance path.

`selection_source` must be `vlm` when the VLM made the choice. Use
`selection_source=reference` only for smoke tests or debugging, and keep those
artifacts separate.

These P3SAM/VLM refs are method inputs. Evaluator labels and success thresholds
belong to separate human-reviewed benchmark contracts under
`artifacts/tool_use_benchmark/tool_label_review/`; do not infer ground-truth
labels from candidate refs alone.

### 5. Merge Into The Constraint Program

Tool refs are merged into the image-grounded program after scene grounding and
lifting:

```bash
"$PARTTOOLGEN_CORE_PYTHON" parttoolgen/src/parttoolgen/cli/stages/merge_tool_refs_into_constraint_program.py \
  --program artifacts/image_grounded/<run>/<task>/lifted_program/constraint_program.json \
  --axis-binding artifacts/image_grounded/tool_ref_grounding/p3sam_tool_refs/<tool_vlm_run>/axis_binding.json \
  --axis-mesh-report artifacts/image_grounded/tool_ref_grounding/p3sam_tool_refs/<tool_vlm_run>/axis_mesh_overlay_report.json \
  --contact-point-selection artifacts/image_grounded/tool_ref_grounding/p3sam_tool_refs/<tool_vlm_run>/contact_point_selection.json \
  --output artifacts/image_grounded/tool_ref_grounding/p3sam_tool_refs/<tool_vlm_run>/constraint_program_with_p3sam_tool_refs.json
```

The compiler then consumes the selected tool point and selected tool normal as
ordinary point/axis constraints. No tool-specific primitive is created by the
tool-prior lane.
