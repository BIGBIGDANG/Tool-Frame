# Agent Collaboration Rules

These rules apply to every coding agent working in this workspace. Read this
file before planning, editing, running experiments, or writing prompts for
another workspace agent.

## Core Rules

1. Think first, then code.
   State assumptions clearly. If an important requirement is ambiguous and local
   context cannot resolve it, ask before changing code. Do not guess.

2. Simplicity first.
   Make the smallest change that solves the verified problem. Prefer existing
   project patterns over new abstractions. Add an abstraction only when it
   removes real complexity or matches an established local pattern.

3. Surgical modifications.
   Do not touch code unrelated to the requirement. Every changed line must trace
   back to the current task or to a required verification fix. Never restore old
   probes, legacy task families, fallback logic, or unrelated refactors unless
   explicitly requested.

4. Goal-driven execution.
   Before coding, turn vague instructions into concrete success criteria. After
   coding, verify those criteria with the smallest reliable checks, tests,
   summaries, or artifacts.

## Workspace Defaults

- Work from the repository root, normally the `ToolFrame_6.4` checkout.
- Before running Python, tests, IsaacGym, SimToolReal, or VLM scripts, source:

```bash
source parttoolgen/scripts/parttoolgen_env.sh
```

- Use `rg` for code search.
- Use `apply_patch` for manual file edits.
- Do not revert user changes or unrelated dirty worktree changes.
- Keep benchmark configs under `artifacts/tool_use_benchmark/`.
- Keep run outputs under `artifacts/image_grounded/<task_family>/...`.
- Do not put rollout, RGB-D evidence, videos, or VLM runtime outputs into
  `artifacts/tool_use_benchmark/`.

## ToolFrame Method Boundary

- Use public RGB-D scene evidence, VLM-selected scene refs, public tool refs,
  P3SAM/VLM tool grounding, and whole-tool SE(3) subgoals.
- Do not use hidden evaluator target refs, oracle target axes, oracle
  trajectories, `tool_family` or `task_family` fallbacks, primitive fallback
  templates, repair-loop fallback, or automatic global-axis retries.
- Keep SimToolReal backend priors separate from task semantics. For example,
  `grasp_lift` is an execution prior; `pre_contact` is a task approach ref and
  should be derived from the motion-axis rule unless the task contract says
  otherwise.

## Prompt Requirement

Every prompt or handoff to another workspace coding agent must include:

```text
Read AGENTS.md first and follow the Agent Collaboration Rules.
```

If a prompt asks an agent to edit code, also include the concrete success
criteria and the verification commands or artifacts expected for that task.
