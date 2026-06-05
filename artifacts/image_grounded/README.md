# Image-Grounded Artifacts

Date: 2026-06-04

This cleaned 6.4 tree keeps compact reviewed tool-selection evidence under
`tool_ref_grounding/`. Full rollout runs should be regenerated from
`artifacts/tool_use_benchmark/benchmark_tasks.json` and written to new
`artifacts/image_grounded/<task_family>/...` roots.

## Present In This Tree

| Root | Status | Notes |
|---|---|---|
| `tool_ref_grounding/` | reviewed compact evidence | P3SAM/VLM part, contact-point, and motion-axis evidence packages. |
| `benchmark_runs/` | generated launcher output root | May be empty or ignored; not source of truth. |

Historical run summaries in `reports/mainline_6.4/` are provenance/audit
material, not final paper statistics.

## Tool-Ref Grounding Policy

Candidate part cards, selected-part bbox overviews, selected-part face
highlights, and presentation crops should come from
`parttoolgen.cli.tool_prior.render_sampart3d_part_role_evidence`.

Do not reintroduce simplified mesh/SVG redraws or per-face candidate choice
figures such as `contact_point_choices/candidate_*_F*_face.png` for the current
VLM selection chain.
