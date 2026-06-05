# ToolFrame 6.4 RA-L Outline Skill Audit

This audit follows the local `paper-self-review` and `academic-writing-latex`
skills. The goal is not to judge final acceptance, but to check whether the
advisor-facing outline already has the structure of a credible RA-L manuscript
and where the current skeleton still needs tightening.

## Verdict

**Current status: advisor-review ready, not yet submission ready.**

The outline now has a recognizable RA-L/IEEE paper shell: title, author block,
abstract, IEEE keywords, introduction, related work, formulation, method,
benchmark, experiments, limitations, conclusion, and bibliography placeholder.
The main story is coherent and should be written from first principles:
**recover the hidden functional mechanism, bind it to the scene mechanism, and
let the hand follow tool-centric subgoals**.

Recommended writing rule for all future revisions:

- Start with the bottom-level challenge: why category labels, visual appearance,
  or direct VLM trajectories are insufficient.
- Explain the mechanism: contact point, motion axis, scene effect references,
  frame alignment, and task-contract effect.
- State the solution: sparse public evidence selection, tool-centric
  references, geometry-constrained whole-tool SE(3) compilation, fixed-goal
  dexterous rollout, and failure attribution.

The strongest part is the method boundary. The outline correctly separates:

- VLM evidence selection from robot trajectory generation.
- Public RGB-D scene evidence from hidden evaluator references.
- Public tool-part evidence from oracle axes or task-family fallback.
- Geometry-constrained SE(3) compilation from policy execution.
- Final task success from module-level failure attribution.

The main missing part is not writing structure, but evidence maturity: final
frozen headline statistics, baselines, and complete task-tier labeling.

## Structure Review

**Pass with minor revision.**

The paper has the expected RA-L components:

- Abstract: problem, method, boundary, evaluation plan, result placeholder.
- Introduction: motivation, problem, main idea, contributions.
- Related work: VLM planning, tool affordance, constraint-based manipulation,
  benchmarks.
- Method: four-stage pipeline matching the requested storyline.
- Benchmark and experiments: task tiers, metrics, figures, tables.
- Limitations: explicit and credible.

Recommended improvement:

- Make the abstract more result-shaped once final numbers exist. RA-L reviewers
  expect at least one concrete statement such as `ToolFrame improves task
  success over TODO baselines on TODO headline tasks`, but only after frozen
  statistics are available.
- Move some advisor-facing TODO notes into comments when converting to a real
  submission draft.

## Logic Consistency Review

**Pass.**

The logic chain is clear:

1. Tool use requires functional geometry.
2. VLMs can select sparse visual references, but should not produce trajectories.
3. Tool refs and scene refs can be compiled into whole-tool SE(3) subgoals.
4. A fixed-goal dexterous policy can execute those subgoals.
5. Task contracts measure final success and attribute failures.

The most important logical risk is the term **dexterous manipulation**. It is
appropriate if the paper clearly shows dexterous hand execution through
SimToolReal fixed-goal rollout. It should not overclaim general in-hand
adaptation or open-ended dexterous tool use.

## Evidence Boundary Review

**Strong.**

The outline explicitly forbids:

- hidden evaluator target refs,
- oracle target axes,
- oracle trajectories,
- task-family fallback,
- primitive fallback,
- repair-loop fallback,
- seed-specific shortcuts.

This is a strong reviewer-facing point because it makes the evaluation boundary
clean. Keep this boundary visible in the Introduction, Method, and Experiments.

## Method Reproducibility Review

**Promising but needs more implementation detail in the full paper.**

The outline names the key artifacts:

- `tool.contact_point`
- `tool.motion_axis`
- lifted scene start/path/end refs
- `tool_pose_subgoals.json`
- `simtoolreal_fixed_goals.json`
- task-contract summaries

Needed in the full draft:

- Exact coordinate frames and transforms.
- Exact optimizer variables and constraints.
- How tool functional refs are reviewed or frozen.
- How RGB-D lifting handles missing/noisy depth.
- What constitutes subgoal validity.
- How task contracts decide success and first failure.

## Experiment Readiness Review

**Not submission ready until frozen result tables exist.**

Current outline correctly uses TODO placeholders and does not treat historical
`reports/mainline_6.4/` as final success statistics.

For advisor review, the current task tiering is acceptable:

- Headline/current: brush sweep forward; tabletop eraser wipe.
- Candidate/provisional: brush left, hammer down, screwdriver wide socket,
  marker trace L/C.
- Diagnostic/stress: screwdriver deep socket and dense variants.
- Draft/smoke: side-board eraser, hammer, screwdriver.

Before submission, the main table should be filled only from frozen
`launcher_summary.{json,csv,md}` and matching per-task summaries.

## Figure and Table Review

**Pass as a plan; needs final assets.**

The proposed Figure 1 is strong and maps well to the method:

- A. Tool-Centric Prior Extraction
- B. Scene Key-Points Grounding
- C. Geometry-Constrained Optimizer
- D. Policy Rollout and Evaluation

Recommended figure text edits:

- Replace `VLM Planner` with `VLM Ref Grounder` or `VLM Scene Ref Selector`.
- Replace `Generate a motion trajectory` with `Select sparse scene refs`.
- Replace `Pretrained Controlling Policy` with `SimToolReal Fixed-Goal Policy`.
- Add a small boundary note: `No hidden refs / no oracle axes / no VLM trajectories`.

## Writing Clarity Review

**Good for advisor review.**

The hook is memorable:

> The tool defines the frame; the hand follows.

This is a good title-level and figure-caption-level message. In the final paper,
pair it with more precise language: "functional tool geometry defines the
contact and motion frames used to compile whole-tool SE(3) subgoals."

## Blocking Items Before Submission

- TODO: Final headline result table from frozen launcher summaries.
- TODO: Baseline definitions and results.
- TODO: Ablation results, even if small.
- TODO: Complete task-tier table with N, status, and contract maturity.
- TODO: Bibliography with verified robotics/VLM/tool-use citations.
- TODO: Final figure assets at RA-L-readable size.
- TODO: Ensure no result claims are copied from historical migration snapshots.

## Recommended Next Draft Move

Use the bilingual RA-L-style outline as the advisor-facing document. It gives
the advisor both the formal paper structure and the Chinese project explanation
needed to quickly help with positioning, title, method framing, and experiments.
