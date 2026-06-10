# Eraser Wipe Benchmark

This directory holds eraser wipe contracts and configs. Run outputs belong under
`artifacts/image_grounded/eraser_wipe/`.

## Current Tasks

| Task | Environment | Status | Success-check status |
|---|---|---|---|
| `tabletop_wipe` | `tabletop_wood_table` | current/e2e | Reviewed flat-eraser contact-coverage success; flat-eraser labels approved. |
| `side_whiteboard_hdu_erase` | `side_left_vertical_whiteboard_soft02` | draft/smoke_review | Formal runner available; erased-area contract uses proposed 70% threshold and needs publish smoke review before statistics. |

## Table-Top Wipe Contract

Config:

```text
tabletop_wipe/benchmark_config.json
```

The table-top task uses a visible raised white tabletop board containing the
blue HDU text and the SimToolReal `flat_eraser`. The board top is the target
surface and includes the HDU text footprint; it is a small tabletop whiteboard.
Tool-side refs come from the local P3SAM segmented-tool package through the
required task-conditioned sequence: part selection first, then selected-part
face selection.

The reviewed success check is `eraser_wipe_contact_coverage`: compiler path
solve, lifted start/end refs on the visible target patch surface, and rollout
samples showing the flat-eraser bottom contact footprint covering at least 30%
of the visible tabletop HDU board long-axis extent. This is a contact-coverage
success check, not a visual erasure check. Visual mark disappearance, rollout
tracking, policy-goal pass, semantic execution pass, and rotation error are
diagnostic.

## Side-Board HDU Draft

Natural task language:

```text
Use this eraser to erase the whole HDU pattern on the whiteboard.
```

The HDU pattern is a visual-only sticker/decal target. It should not become a
collision target or a hidden success oracle. The intended success checker
measures erased area from the selected flat-eraser `tool.contact_point` and
contact-face footprint against the HDU pattern on the whiteboard surface.

This side-board task is draft until the visible public refs, checker artifacts,
threshold, and reuse of flat-eraser labels are reviewed.

Config:

```text
side_whiteboard_hdu_erase/benchmark_config.json
side_whiteboard_hdu_erase/contract_draft.json
```

Runner:

```text
parttoolgen/src/parttoolgen/cli/runners/run_image_grounded_side_board_hdu_erase_eval.py
```

This task-specific runner is an internal implementation detail used by the
benchmark registry.

Unified smoke command:

```bash
bin/toolframe benchmark \
  --task eraser_wipe/side_whiteboard_hdu_erase \
  --trials 1 \
  --seed-start 0 \
  --device cuda:0 \
  --capture-video-every 1
```

Each seed writes a per-trial audit manifest:

```text
seed_<NNN>/evidence_manifest.json
```

The manifest groups public scene evidence, VLM scene selections,
`tool.contact_point` / `tool.motion_axis` selections, lifted 3D refs, compile
artifacts, rollout evidence, and the HDU erased-area success output. This is the
preferred review entry point for seed-level failure attribution.
