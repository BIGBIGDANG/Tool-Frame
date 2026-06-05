# Tool-Ref Grounding Outputs

This directory stores task-conditioned tool-side grounding results for the
image-grounded mainline.

Static segmentation/tool priors live under:

```text
artifacts/tool_prior/
```

Benchmark/task configs live under:

```text
artifacts/tool_use_benchmark/
```

The outputs here are run artifacts: VLM part selections, exported tool refs,
and merged constraint programs used by compiler smokes.

Current kept outputs:

- `part_selection/all_segmented_tools_20260525/`
  Unified P3SAM segmented-tool evidence package for SimToolReal tool functional
  part selection. It contains per-tool overview images, isolated part cards,
  `manifest.json`, and `visual_evidence.json`.
- `red_brush/p3sam_refs_20260524/`
  Red-brush P3SAM/VLM tool-ref grounding evidence.
- `flat_eraser/tabletop_wipe_forward_seed_000_tool_refs/`
  Flat-eraser task-conditioned refs for tabletop wipe. The VLM-selected
  contact point/face and motion-axis face are kept as public tool refs for
  eraser seed benchmarks.
- `long_screwdriver/screwdriver_insert_deep_socket_tool_refs/`
  Long-screwdriver task-conditioned tip/contact and motion-axis refs for the
  deep-socket insertion smoke.
- `ppt_exports/`
  Presentation/evidence export artifacts, including hammer transparent
  VLM/P3SAM/SAMPart3D comparison figures. These are for inspection and slides,
  not benchmark configs. PPT-ready candidate figures should preserve the
	  5.15-style part-overview raster evidence layout from
	  `render_sampart3d_part_role_evidence.py`; post-processing may crop or remove
	  background, but should not redraw the mesh with simplified local SVG/mesh
	  exporters. Per-face/contact-choice figures such as
	  `contact_point_choices/candidate_*_F*_face.png` must not be generated.
  Subgoal ghost figures live under
  `ppt_exports/claw_hammer_vlm_original_transparent/subgoal_ghost_clean_mesh/`
  and intentionally use clean 5.15/SAMPart3D render images with no bbox,
  highlighted face, markers, or arrows.

Tool-side grounding follows this order:

```text
P3SAM tool parts
  -> precomputed part-overview evidence
  -> task-conditioned VLM function-part selection
  -> selected-part bbox face evidence with highlighted F markers
  -> task-conditioned VLM contact/motion-axis selection
  -> ToolFrame tool refs consumed by merge_tool_refs_into_constraint_program.py
```

Functional points/faces are task-conditioned refs, not task-agnostic global
labels. The VLM chooses the functional part from task wording and 5.15-style
part-overview visual evidence. Only after that part is selected do we render
its highlighted bbox faces and F markers for contact/motion-axis selection.
Do not use one-image-per-face rendered cards.

Evaluator-side tool labels live under
`artifacts/tool_use_benchmark/tool_label_review/` and require human approval
before they can be used for statistics. The eraser wipe contract is currently
approved; other tool/task contracts should be treated as pending unless their
review files explicitly say otherwise. Candidate generation excludes
internal/occluded tool surfaces before the VLM contact or motion-axis prompt.
