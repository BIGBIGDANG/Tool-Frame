# Tool-Prior Artifacts

This directory stores reusable static tool-prior provider outputs. It is not a
rollout or benchmark-output directory.

## P3SAM

`p3sam/` contains local tool segmentation and part-prior outputs used to build
ToolFrame tool refs. The full files are runtime artifacts and are ignored by
git; this README documents the expected local layout.

Current useful entries:

- `p3sam/blue_brush/`
- `p3sam/claw_hammer/`
- `p3sam/flat_eraser/`
- `p3sam/flat_spatula/`
- `p3sam/handle_eraser/`
- `p3sam/long_screwdriver/`
- `p3sam/mallet_hammer/`
- `p3sam/red_brush/`
- `p3sam/red_brush_rebuilt_20260524/`: cleaned red-brush P3SAM tool-prior
  rebuild used by the kept tool-ref evidence in
  `artifacts/image_grounded/tool_ref_grounding/red_brush/p3sam_refs_20260524/`.
- `p3sam/sharpie_marker/`
- `p3sam/short_screwdriver/`
- `p3sam/spoon_spatula/`
- `p3sam/staples_marker/`

Task-conditioned VLM part/face selections belong under
`artifacts/image_grounded/tool_ref_grounding/`.
