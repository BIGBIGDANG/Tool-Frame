# Artifacts

This directory separates benchmark source-of-truth files from generated
experimental evidence. Most experiment outputs are intentionally ignored by git;
commit only compact manifests, contracts, configs, and documentation unless the
project owner explicitly approves a data/evidence release.

See `experiment_inventory.md` for the current local keep/archive/cleanup
classification of existing run roots.

## Artifact taxonomy

| Root | Category | Commit policy | Contents |
|---|---|---|---|
| `tool_use_benchmark/` | Benchmark contracts and configs | Commit | Human-reviewed task definitions, candidate matrices, metric definitions, environment notes, and config JSON files. |
| `image_grounded/tool_ref_grounding/` | Reviewed tool-selection evidence | Optional evidence release | Compact P3SAM/VLM part, contact, and motion-axis evidence packages. |
| `image_grounded/<other roots>/` | Run evidence and history | Ignore by default | RGB-D evidence, VLM scene markings, lifted refs, compiler outputs, rollout summaries, videos, visual packages, and smoke/debug roots. |
| `tool_prior/` | Compact reusable priors | Optional evidence release | P3SAM/SAMPart3D tool-part priors used to regenerate tool refs; package separately from source if size matters. |
| `visualizations/` | Inspection figures | Optional figure/evidence release | Reviewed figures for debugging, slides, and human review. Prefer regenerating when possible. |
| `external/` | Imported third-party asset cache | Do not commit | External ToolUse and other imported raw assets. These are out of scope for the current public release. |
| `../test000/` | Local scratch outside artifacts | Do not commit | Interactive previews, temporary smoke runs, comparison renders, and ad-hoc review outputs. |

## Promotion flow

1. Develop interactively under `test000/` when the human asks for scratch
   previews or rapid comparisons.
2. Store benchmark-family runtime evidence under
   `artifacts/image_grounded/<task_family>/` once a run is worth preserving
   locally.
3. Copy only the reviewed conclusions, contract status, thresholds, and compact
   references into `artifacts/tool_use_benchmark/`.
4. Never move videos, RGB-D arrays, rollout folders, VLM raw responses, or
   compiler work directories into `artifacts/tool_use_benchmark/`.

## Release categories

- `release-source`: source code, scripts, schemas, docs, small reviewed URDF/OBJ
  benchmark assets, and `artifacts/tool_use_benchmark/**`.
- `release-evidence`: selected compact `artifacts/tool_prior/**`,
  `artifacts/image_grounded/tool_ref_grounding/**`, and reviewed figures.
- `release-manifest-only`: README files and compact manifests describing local
  evidence roots, without the heavy outputs themselves.
- `local-evidence`: non-selected generated roots under `artifacts/image_grounded/**`,
  `artifacts/tool_prior/**`, and `artifacts/visualizations/**`.
- `local-scratch`: `test000/**`.
- `out-of-scope`: `artifacts/external/**` and external ToolUse-derived outputs.

## Naming guidance for new run roots

Use descriptive, stable roots:

```text
artifacts/image_grounded/<task_family>/<task_or_scene>_<contract_or_smoke>_<seed_or_seed20>/
test000/<short_purpose>_<YYYYMMDD_or_counter>/
```

Avoid names that encode temporary implementation details such as CUDA device,
old prompt variants, or one-off debugging guesses unless the run is explicitly a
short-lived scratch root.

## Cleanup policy

- Keep the latest current run for each active benchmark family while it is still
  used for summaries or figures.
- Keep task-conditioned tool-ref evidence required by active configs.
- Keep `vlm_prompt_unified_scene_grounding_smoke_v2/` until a newer generic
  prompt audit supersedes it.
- Archive or delete one-off previews, failed smoke roots, duplicate videos, and
  old prompt/debug runs after their conclusion has been copied into
  `tool_use_benchmark/` or a handoff.
- Before deleting any output root, confirm no corresponding runner or rollout
  process is active.

Do not place source configs, prompts, reusable code, or benchmark contracts in
runtime evidence directories.
