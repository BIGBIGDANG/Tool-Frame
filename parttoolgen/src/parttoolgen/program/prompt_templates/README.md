# PartToolGen Prompt Templates

Current mainline templates keep semantic choice and metric execution separated.

## Mainline

- `affordance_binding_v1.md`: asks the VLM to bind a SAMPart3D functional
  region, then select exactly one tool-side functional/contact point ref from
  that region's geometry candidate table, an optional same-contact
  tool-side axis/surface ref, and target refs from the scene/target table.
- Part-Centric Affordance Program JSON: asks the VLM to choose exact
  selected-region candidate point/axis/surface refs, target refs, and bounded
  interaction/motion primitive stages. PartToolGen lowers this into restricted
  `build_program(ctx)`.

The current automatic mainline uses SAMPart3D regions plus geometry candidate
points. The VLM may inspect multi-view marked images, but executable refs still
come from declared tool-frame refs. Removed visual-feature whole-tool and raw
VLM pixel choices are ablation/debug signals, not default executable
interfaces.

## Functional Frame Contract

For internal compatibility bindings, `functional_frame_binding` uses:

```text
contact_ref = VLM-selected contact point
primary_ref = same as contact_ref
motion_axis_ref = listed same-contact tool motion axis selected by PCAP
axis_source_ref = null in the VLM output, derived by PartToolGen only for old reports
secondary_ref = null
construction = contact_point_motion_axis_v1
```

Part-Centric Affordance Program output must select `motion_axis_ref`
explicitly. The VLM should not choose a second keypoint to define the axis, and
5.15 mainline runs fail closed rather than deriving a missing axis.

## Removed From 5.15

Direct VLM-authored Python programs, non-null `secondary_ref` functional-frame
bindings, raw VLM clicks, and template-only stage selection are not shipped as
5.15 default routes.
