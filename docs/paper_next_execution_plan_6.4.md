# ToolFrame 6.4 Paper Next Execution Plan

This plan distills the external paper-outline feedback and the follow-up
multi-agent review into the next executable steps for ToolFrame 6.4.

## Decision Summary

The feedback is directionally useful, but the paper claim must be calibrated.
Do not claim that ToolFrame already proves robust 4-5 task cross-tool
generalization. The current defensible claim is:

```text
ToolFrame provides an auditable tool-centric reference interface that can be
instantiated across multiple tool classes and scene families; final statistics
are reported only for reviewed headline contracts, while candidate and
diagnostic tasks expose extension coverage and failure modes.
```

This keeps the paper honest while still preserving the central idea:

```text
The tool defines the frame; the hand follows.
```

## Current Facts To Use

Historical frozen run before the red-zone brush split became the current
headline:

```text
reports/paper_release_6.4/headline/paper_headline_clean_renderfix_20260604_2110/
```

Launcher summary:

| Task | N | Pass rate | Result role |
| --- | ---: | ---: | --- |
| `brush_sweep/forward` | 20 | 0.05 | Historical directional-displacement evidence; current headline brush split is `brush_sweep/red_zone`. |
| `eraser_wipe/tabletop_wipe` | 20 | 0.15 | Frozen headline evidence with approved flat-eraser labels. |

Both tasks have `compile_path_pass_rate = 1.0`, so the historical bottleneck is
not the diagnostic path gate. Brush mostly fails the directional task-effect
predicate. Eraser mostly fails execution/contact-coverage on the target patch,
with a small number of RGB-D/VLM grounding failures.

Fresh per-trial audit table for this run:

```text
reports/paper_release_6.4/headline/paper_headline_clean_renderfix_20260604_2110/benchmark_audit_table.json
reports/paper_release_6.4/headline/paper_headline_clean_renderfix_20260604_2110/benchmark_audit_table.csv
reports/paper_release_6.4/headline/paper_headline_clean_renderfix_20260604_2110/benchmark_audit_table.md
```

The candidate queue at:

```text
reports/paper_release_6.4/candidates/paper_candidates_remaining_smoke_20260604_214523/
```

contains only queue metadata. It is not a completed result.

## Recommendations To Adopt

### Adopt Now

1. Freeze and report the headline result table from completed
   `launcher_summary.{json,csv,md}` files only.
2. Build a five-tool tool-ref status table:
   `red_brush`, `flat_eraser`, `claw_hammer`, `long_screwdriver`,
   `staples_marker`.
3. Keep headline, candidate, diagnostic, and draft tasks separated in every
   paper table.
4. Use failure attribution as a main result, not just a debugging note.
5. Add a boundary audit before any baseline or ablation result is promoted.

### Adopt With Downgraded Claim

1. Use hammer, screwdriver, and marker as candidate evidence unless their
   contracts and label reviews are closed.
2. Use side-board marker as a hard candidate/qualitative scene-family example,
   not as a headline statistic yet.
3. Use direct-waypoint or end-effector baselines only after proving they use the
   same public evidence boundary and do not smuggle in hidden target refs,
   primitives, or family defaults.

### Do Not Adopt Yet

1. Do not promote 4-5 tasks to headline merely to make the paper look larger.
2. Do not report `paper_release_6.4` as a single final success-rate suite.
3. Do not treat `reports/mainline_6.4/` or candidate queue metadata as final
   ToolFrame 6.4 results.
4. Do not use pending tool-label files as ground truth for calling a VLM
   selected part or face wrong.

## P0 Execution

### 1. Lock The Paper Claim

Update the paper outline around this claim:

```text
ToolFrame separates public visual grounding, tool-centric geometry references,
whole-tool SE(3) compilation, fixed-goal policy rollout, and evaluator-side
task contracts. Its value is auditability and attribution, not currently high
end-to-end success.
```

The paper should state that the interface is instantiated across five tools and
two scene families, but final statistics are limited to reviewed or explicitly
marked headline contracts.

### 2. Add Paper Result Reports

Create a public command or package CLI, preferably exposed as:

```bash
bin/toolframe paper-results
```

It should read a completed `launcher_summary.json` plus referenced per-task
`summary.json` files and write:

```text
paper_results.json
paper_results.csv
paper_results.md
failure_counts_by_task.csv
```

Required columns:

```text
task
tool
scene
contract_status
N
functional_grounding_rate
scene_grounding_rate
subgoal_validity_rate
policy_execution_rate
task_success_rate
primary_failure_category
review_note
```

### 3. Expose Run Auditing

Expose and improve the existing audit-table builder:

```bash
bin/toolframe audit-runs
```

The command should support explicit summary paths so it does not accidentally
scan historical output roots. It should produce per-trial layer status and
failure counts suitable for a stacked-bar figure.

Important follow-up: brush currently records `perception=unknown` and
`tool_grounding=not_recorded` in the audit table. Normalize brush fields or the
paper module-rate table will be misleading.

### 4. Add Paper Status And Tool-Ref Audit

Add:

```bash
bin/toolframe paper-status
bin/toolframe tool-ref-audit
```

`paper-status` should read the registry, task configs, task contracts, and tool
label reviews, then write a task-tier table:

```text
headline
candidate
diagnostic
draft
pending_same_difficulty_variant
```

`tool-ref-audit` should write:

```text
Tool | Functional geometry | contact_point source | motion_axis source |
review status | used tasks | evidence paths
```

Current expected status:

| Tool | Review status |
| --- | --- |
| `flat_eraser` | approved |
| `red_brush` | pending |
| `claw_hammer` | pending |
| `long_screwdriver` | pending |
| `staples_marker` | pending |

### 5. Local Environment Handoff

The 6.4 handoff path is now:

```bash
bin/toolframe bootstrap-env
bin/toolframe env-check --mode core
```

The checked-in `requirements.txt` recreates the lightweight Python 3.8 core
environment under `.venv_parttoolgen_py38`. Optional rollout and VLM layers live
in `requirements-rollout.txt` and `requirements-vlm.txt`. Workstation-specific
runtime paths, such as IsaacGym binaries, SimToolReal site packages, pretrained
policy roots, and VLM credentials, belong in the ignored `.parttoolgen.local.env`
file rather than in commands or source code.

## P1 Execution

### 6. Rerun Candidate Smoke Cleanly

The existing candidate queue only contains queue metadata. Rerun the candidate
smoke explicitly after the headline run:

```bash
bin/toolframe current-e2e \
  --trials 1 \
  --device cuda:0 \
  --continue-on-task-error \
  --aggregate json,csv,md
```

Report candidate tasks as evidence-generation or diagnostic rows, not headline
statistics.

### 7. Boundary-Safe Ablations

Prioritize ablations that preserve the public-evidence boundary:

| Ablation | Reporting role |
| --- | --- |
| no tool motion axis | Shows the compiler does not silently guess family axes. |
| remove selected contact point | Shows explicit functional contact geometry is necessary. |
| public human-clicked scene refs | Scene-grounding upper-bound diagnostic; public RGB-D clicks only. |
| strict tracking vs task-effect success | Shows why task contracts should not be replaced by pose tracking. |

Delay direct waypoint/end-effector baselines until the interface is specified
well enough to avoid hidden refs, primitive fallback, and unfair controller
differences.

### 8. Figure Material

Prepare a compact figure manifest rather than copying large raw outputs:

```text
Figure 2: five tools functional refs overview
Figure 3: tabletop and side-board scene grounding overlays
Figure 4: rollout snapshots for headline/candidate tasks
Figure 5: failure attribution stacked bar
```

This can become:

```bash
bin/toolframe figure-manifest
```

## P2 Execution

Only after P0/P1 are stable:

1. Promote hammer/screwdriver/marker tasks by closing contracts and label
   reviews.
2. Add same-difficulty tool variants such as blue brush or mallet only after
   their `configs/tool_refs/<tool_id>/` packages exist.
3. Add new task families only through `artifacts/tool_use_benchmark/` configs
   and the registry, not one-off scripts.

## Paper Writing Consequence

The current RA-L outline should be revised toward this framing:

```text
Main result: a reviewed headline table plus module/failure attribution.
Coverage result: five tool-ref packages and candidate task evidence.
Diagnostic result: ablations that show contact point, motion axis, and public
scene grounding are necessary.
```

This is weaker than claiming broad success, but much more publishable: reviewers
can see exactly what the method used, what it did not use, and where it failed.
