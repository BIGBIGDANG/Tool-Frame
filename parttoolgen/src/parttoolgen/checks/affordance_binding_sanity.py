#!/usr/bin/env python3
"""Legacy V0 sanity checks for symbolic affordance bindings.

This checker supports the older symbolic affordance-binding stack. The current
image-grounded benchmark contracts use reviewed ``tool.contact_point``,
``tool.motion_axis``, and ``tool.motion_axis`` refs instead of public
``function_*`` fields. Keep this module for legacy/debug validation only.

The gate in this module checks whether selected refs are resolvable, typed,
geometrically usable, and supported by grounding/evidence metadata. It is
deliberately independent of any specific tool or task success criterion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from parttoolgen.compiler.reference_resolver import (
    ReferenceResolutionError,
    ReferenceResolver,
    ResolvedRef,
)


AFFORDANCE_BINDING_REPORT_VERSION = "v0.1"
EPS = 1e-9

GeometryExpectation = Mapping[str, Any]
SymbolLookup = Callable[[str], Any]


_POINT_LIKE_KINDS = {"point", "region", "surface", "part", "object", "grasp", "frame"}
_AXIS_LIKE_KINDS = {"axis", "surface", "region", "part", "frame"}
_PHYSICAL_SLOT_NAMES = {
    "source",
    "moving",
    "tool_contact",
    "tool_contact_point",
    "tool_contact_axis",
    "tool_surface_normal",
    "function_point",
    "function_axis",
    "function_keypoint_ref",
    "function_axis_ref",
    "functional_point",
    "functional_axis",
}

_CONSTRAINT_REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "contact": ("source", "target"),
    "distance": ("a", "b"),
    "align_axis": ("source_axis", "target_axis"),
    "align_vector_between_points": ("source_start", "source_end", "target_axis"),
    "pre_contact_offset": ("source", "target", "axis"),
    "clearance": ("moving", "obstacle"),
    "move_along": ("source", "start", "end"),
    "tool_occupancy_clearance": ("tool_part", "obstacle"),
}

_CONSTRAINT_SLOT_EXPECTATIONS: dict[str, dict[str, dict[str, Any]]] = {
    "contact": {
        "source": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "target": {"geometry": "point", "frame": "world"},
    },
    "distance": {
        "a": {"geometry": "point"},
        "b": {"geometry": "point"},
    },
    "align_axis": {
        "source_axis": {"geometry": "axis", "frame": "tool", "physical_contact": True},
        "target_axis": {"geometry": "axis", "frame": "world"},
    },
    "align_vector_between_points": {
        "source_start": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "source_end": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "target_axis": {"geometry": "axis", "frame": "world"},
    },
    "pre_contact_offset": {
        "source": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "target": {"geometry": "point", "frame": "world"},
        "axis": {"geometry": "axis", "frame": "world"},
    },
    "clearance": {
        "moving": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "obstacle": {"geometry": "point", "frame": "world"},
    },
    "move_along": {
        "source": {"geometry": "point", "frame": "tool", "physical_contact": True},
        "start": {"geometry": "point", "frame": "world"},
        "end": {"geometry": "point", "frame": "world"},
    },
    "tool_occupancy_clearance": {
        "tool_part": {"frame": "tool"},
        "obstacle": {"frame": "world"},
    },
}

_AFFORDANCE_SLOT_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "grasp_ref": {"geometry": "point", "frame": "tool"},
    "function_keypoint_ref": {
        "geometry": "point",
        "frame": "tool",
        "physical_contact": True,
    },
    "function_axis_ref": {
        "geometry": "axis",
        "frame": "tool",
        "physical_contact": True,
    },
    "contact_ref": {
        "geometry": "point",
        "frame": "tool",
        "physical_contact": True,
    },
    "primary_ref": {
        "geometry": "point",
        "frame": "tool",
        "physical_contact": True,
    },
    "axis_source_ref": {
        "geometry": "point",
        "frame": "tool",
    },
    "secondary_ref": {
        "geometry": "point",
        "frame": "tool",
    },
    "tertiary_ref": {
        "geometry": "point",
        "frame": "tool",
        "physical_contact": True,
    },
    "target_pre_contact_ref": {"geometry": "point", "frame": "world"},
    "target_contact_ref": {"geometry": "point", "frame": "world"},
    "target_post_contact_ref": {"geometry": "point", "frame": "world"},
    "motion_axis_ref": {"geometry": "axis", "frame": "world"},
}

_AFFORDANCE_REQUIRED_SLOTS = ("function_keypoint_ref", "target_contact_ref")


@dataclass(frozen=True)
class BindingRef:
    slot: str
    ref: str
    category: str = "slot"
    promoted: bool = False


@dataclass(frozen=True)
class RefSnapshot:
    ref: str
    kind: str | None
    frame: str | None
    entity: Mapping[str, Any]
    point_world: list[float] | None = None
    vector_world: list[float] | None = None
    point_tool: list[float] | None = None
    vector_tool: list[float] | None = None
    confidence: float | None = None
    source: str | None = None
    provenance: Mapping[str, Any] | None = None


class _ReportBuilder:
    def __init__(self, *, binding_id: str | None = None) -> None:
        self.binding_id = binding_id
        self.checks: list[dict[str, Any]] = []
        self.failed_refs: set[str] = set()
        self.warning_refs: set[str] = set()
        self.repair_hints: list[str] = []

    def add(
        self,
        name: str,
        *,
        passed: bool,
        severity: str,
        message: str,
        ref: str | None = None,
        slot: str | None = None,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "message": message,
        }
        if self.binding_id:
            record["binding_id"] = self.binding_id
        if ref:
            record["ref"] = ref
        if slot:
            record["slot"] = slot
        if details:
            record["details"] = dict(details)
        self.checks.append(record)
        if passed:
            return
        if severity == "error" and ref:
            self.failed_refs.add(ref)
        elif severity == "warning" and ref:
            self.warning_refs.add(ref)
        if hint and hint not in self.repair_hints:
            self.repair_hints.append(hint)

    def report(self) -> dict[str, Any]:
        result = {
            "schema_version": AFFORDANCE_BINDING_REPORT_VERSION,
            "passed": not any(
                not check["passed"] and check["severity"] == "error"
                for check in self.checks
            ),
            "checks": self.checks,
            "failed_refs": sorted(self.failed_refs),
            "warning_refs": sorted(self.warning_refs),
            "repair_hints": self.repair_hints,
        }
        if self.binding_id:
            result["binding_id"] = self.binding_id
        return result


def check_affordance_binding_sanity(
    binding: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    resolver: ReferenceResolver | Any | None = None,
    symbol_table: Mapping[str, Any] | None = None,
    symbol_lookup: SymbolLookup | None = None,
    required_slots: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    slot_expectations: Mapping[str, GeometryExpectation] | None = None,
    visual_evidence: Mapping[str, Any] | None = None,
    confidence_warning_threshold: float = 0.25,
    degeneracy_epsilon_m: float = 1e-6,
) -> dict[str, Any]:
    """Check one binding or a sequence of bindings.

    The function is pure: callers provide a resolver, symbol table, or lookup
    callback, and receive a JSON-serializable report. Candidate refs are allowed
    as evidence, but executable tool contact slots must use
    ``tool.functional_parts.*``, an explicitly promoted ref, or a tool-frame
    candidate from the offline tool prior with part provenance.
    """

    if isinstance(binding, Sequence) and not isinstance(binding, (str, bytes, bytearray, Mapping)):
        return _aggregate_reports(
            [
                check_affordance_binding_sanity(
                    item,
                    resolver=resolver,
                    symbol_table=symbol_table,
                    symbol_lookup=symbol_lookup,
                    required_slots=required_slots,
                    slot_expectations=slot_expectations,
                    visual_evidence=visual_evidence,
                    confidence_warning_threshold=confidence_warning_threshold,
                    degeneracy_epsilon_m=degeneracy_epsilon_m,
                )
                for item in binding
            ]
        )
    if not isinstance(binding, Mapping):
        raise TypeError("binding must be a mapping or sequence of mappings")

    return _check_single_binding(
        binding,
        resolver=resolver,
        symbol_table=symbol_table,
        symbol_lookup=symbol_lookup,
        required_slots=required_slots,
        slot_expectations=slot_expectations,
        visual_evidence=visual_evidence,
        confidence_warning_threshold=confidence_warning_threshold,
        degeneracy_epsilon_m=degeneracy_epsilon_m,
    )


def check_program_ir_affordance_bindings(
    program_ir: Mapping[str, Any],
    *,
    resolver: ReferenceResolver | Any | None = None,
    symbol_table: Mapping[str, Any] | None = None,
    symbol_lookup: SymbolLookup | None = None,
    visual_evidence: Mapping[str, Any] | None = None,
    confidence_warning_threshold: float = 0.25,
    degeneracy_epsilon_m: float = 1e-6,
) -> dict[str, Any]:
    """Run the binding gate over ProgramIR constraints.

    This helper only applies generic geometric slot expectations implied by the
    constraint API: point vs axis and tool vs world frame. It does not encode
    table, clearance, or task-specific affordance semantics.
    """

    bindings: list[dict[str, Any]] = []
    for constraint in program_ir.get("constraints", []):
        if not isinstance(constraint, Mapping):
            continue
        constraint_type = str(constraint.get("type", ""))
        refs = constraint.get("refs", {})
        if not isinstance(refs, Mapping):
            refs = {}
        bindings.append(
            {
                "binding_id": str(constraint.get("id") or constraint_type or "constraint"),
                "type": constraint_type,
                "refs": dict(refs),
                "required_slots": _CONSTRAINT_REQUIRED_SLOTS.get(constraint_type, ()),
            }
        )

    expectations: dict[str, GeometryExpectation] = {}
    for value in _CONSTRAINT_SLOT_EXPECTATIONS.values():
        expectations.update(value)
    report = check_affordance_binding_sanity(
        bindings,
        resolver=resolver,
        symbol_table=symbol_table,
        symbol_lookup=symbol_lookup,
        slot_expectations=expectations,
        visual_evidence=visual_evidence,
        confidence_warning_threshold=confidence_warning_threshold,
        degeneracy_epsilon_m=degeneracy_epsilon_m,
    )
    report["program_id"] = program_ir.get("program_id")
    return report


def check_program_ir_affordance_contract(
    program_ir: Mapping[str, Any],
    affordance_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that ProgramIR uses the selected functional binding.

    ``check_program_ir_affordance_bindings`` validates individual constraint
    slots geometrically. This gate checks the cross-constraint contract created
    by AffordanceBinding: the planner must actually use the bound tool contact
    point, target contact, directed-motion refs, and derived point-pair
    orientation when those refs are available. The check is deliberately
    task-agnostic; it only reasons over declared symbolic slots.
    """

    report = _ReportBuilder(binding_id=_binding_id(affordance_binding))
    constraints = [
        constraint
        for constraint in program_ir.get("constraints", [])
        if isinstance(constraint, Mapping)
    ]
    objectives = [
        objective
        for objective in program_ir.get("objectives", [])
        if isinstance(objective, Mapping)
    ]
    refs = _affordance_contract_refs(affordance_binding)
    contact_ref = refs.get("contact_ref") or refs.get("function_keypoint_ref")
    function_axis_ref = refs.get("function_axis_ref")
    axis_source_ref = refs.get("axis_source_ref")
    target_contact_ref = refs.get("target_contact_ref")
    target_pre_contact_ref = refs.get("target_pre_contact_ref")
    target_post_contact_ref = refs.get("target_post_contact_ref")
    motion_axis_ref = refs.get("motion_axis_ref")
    frame_binding = affordance_binding.get("functional_frame_binding")
    frame_construction = (
        frame_binding.get("construction")
        if isinstance(frame_binding, Mapping)
        else None
    )
    requires_explicit_function_axis = (
        affordance_binding.get("binding_mode") == "selected_region_to_target"
        or frame_construction == "selected_region_contact_patch_v1"
    )

    if contact_ref:
        used = _constraint_ref_used(
            constraints,
            contact_ref,
            types={"contact", "pre_contact_offset", "move_along"},
            slots={"source"},
        ) or _objective_ref_used(objectives, contact_ref)
        report.add(
            "bound_function_contact_used",
            passed=used,
            severity="error",
            message=(
                "ProgramIR uses the bound tool contact/function point"
                if used
                else "ProgramIR does not use the bound tool contact/function point"
            ),
            ref=contact_ref,
            slot="contact_ref",
            hint=(
                "Use functional_frame_binding.contact_ref as the tool-side "
                "source in contact, pre_contact_offset, and move_along constraints."
            ),
        )

    if contact_ref and target_contact_ref:
        matched = _has_constraint(
            constraints,
            "contact",
            {"source": contact_ref, "target": target_contact_ref},
        ) or _has_constraint(
            constraints,
            "move_along",
            {"source": contact_ref, "goal": target_contact_ref},
        ) or _has_objective_with_refs(
            objectives,
            {contact_ref, target_contact_ref},
        )
        report.add(
            "bound_target_contact_used",
            passed=matched,
            severity="error",
            message=(
                "ProgramIR contacts or optimizes toward the bound target contact ref"
                if matched
                else "ProgramIR never contacts or optimizes toward the bound target contact ref"
            ),
            ref=target_contact_ref,
            slot="target_contact_ref",
            hint=(
                "Add contact(contact_ref, target_contact_ref), move_along(..., "
                "goal=target_contact_ref), or a registered subgoal objective over "
                "both refs, for the engagement stage instead of only moving near "
                "pre-contact or object centers."
            ),
        )

    if requires_explicit_function_axis:
        bound = bool(function_axis_ref and motion_axis_ref)
        report.add(
            "explicit_function_axis_bound",
            passed=bound,
            severity="error",
            message=(
                "AffordanceBinding provides an explicit contact axis and scene direction"
                if bound
                else "Selected-region mainline binding is missing function_axis_ref or motion_axis_ref"
            ),
            ref=function_axis_ref or motion_axis_ref or contact_ref,
            slot="function_axis_ref,motion_axis_ref",
            hint=(
                "Selected-region contact-frame planning must bind a derived "
                "surface/surface normal as function_axis_ref and a scene direction "
                "axis before ProgramIR is accepted."
            ),
            details={
                "function_axis_ref": function_axis_ref,
                "motion_axis_ref": motion_axis_ref,
                "binding_mode": affordance_binding.get("binding_mode"),
                "frame_construction": frame_construction,
            },
        )

    if function_axis_ref and motion_axis_ref:
        matched = _has_constraint(
            constraints,
            "align_axis",
            {
                "source_axis": function_axis_ref,
                "target_axis": motion_axis_ref,
            },
        ) or _has_objective_with_refs(
            objectives,
            {function_axis_ref, motion_axis_ref},
        )
        report.add(
            "explicit_function_axis_aligned",
            passed=matched,
            severity="error",
            message=(
                "ProgramIR aligns the bound tool contact axis"
                if matched
                else "ProgramIR omits the bound tool contact-axis alignment"
            ),
            ref=function_axis_ref,
            slot="function_axis_ref",
            hint=(
                "When function_axis_ref and motion_axis_ref are available, add "
                "align_axis(function_axis_ref, motion_axis_ref, polarity=...), "
                "or a registered subgoal objective over those refs, before rollout."
            ),
            details={
                "function_axis_ref": function_axis_ref,
                "motion_axis_ref": motion_axis_ref,
            },
        )
    elif (
        not requires_explicit_function_axis
        and contact_ref
        and axis_source_ref
        and motion_axis_ref
    ):
        matched = _has_constraint(
            constraints,
            "align_vector_between_points",
            {
                "source_start": axis_source_ref,
                "source_end": contact_ref,
                "target_axis": motion_axis_ref,
            },
        ) or _has_objective_with_refs(
            objectives,
            {axis_source_ref, contact_ref, motion_axis_ref},
        )
        report.add(
            "derived_function_axis_aligned",
            passed=matched,
            severity="error",
            message=(
                "ProgramIR aligns the derived tool functional direction"
                if matched
                else "ProgramIR omits the derived tool functional-direction alignment"
            ),
            ref=contact_ref,
            slot="functional_frame_binding",
            hint=(
                "When axis_source_ref, contact_ref, and motion_axis_ref are "
                "available, add align_vector_between_points(axis_source_ref, "
                "contact_ref, motion_axis_ref, polarity=...), or a registered "
                "subgoal objective over those refs, before rollout."
            ),
            details={
                "axis_source_ref": axis_source_ref,
                "contact_ref": contact_ref,
                "motion_axis_ref": motion_axis_ref,
            },
        )

    if contact_ref and target_pre_contact_ref and target_post_contact_ref:
        if target_pre_contact_ref != target_post_contact_ref:
            matched = _has_directed_motion_path(
                constraints,
                source_ref=contact_ref,
                start_ref=target_pre_contact_ref,
                goal_ref=target_post_contact_ref,
            ) or _has_objective_with_refs(
                objectives,
                {contact_ref, target_pre_contact_ref, target_post_contact_ref},
            )
            report.add(
                "bound_directed_motion_used",
                passed=matched,
                severity="error",
                message=(
                    "ProgramIR uses the bound directed-motion refs"
                    if matched
                    else "ProgramIR omits the bound directed-motion relation"
                ),
                ref=contact_ref,
                slot="target_pre_contact_ref,target_post_contact_ref",
                hint=(
                    "Add move_along constraints that connect target_pre_contact_ref "
                    "to target_post_contact_ref for contact_ref, either as one edge "
                    "or as a staged path such as pre_contact -> contact -> "
                    "post_contact. A registered path/subgoal objective over those "
                    "refs is also valid."
                ),
                details={
                    "target_pre_contact_ref": target_pre_contact_ref,
                    "target_post_contact_ref": target_post_contact_ref,
                    "target_contact_ref": target_contact_ref,
                },
            )

    result = report.report()
    result["program_id"] = program_ir.get("program_id")
    result["contract_refs"] = refs
    return result


def _check_single_binding(
    binding: Mapping[str, Any],
    *,
    resolver: ReferenceResolver | Any | None,
    symbol_table: Mapping[str, Any] | None,
    symbol_lookup: SymbolLookup | None,
    required_slots: Iterable[str] | Mapping[str, Iterable[str]] | None,
    slot_expectations: Mapping[str, GeometryExpectation] | None,
    visual_evidence: Mapping[str, Any] | None,
    confidence_warning_threshold: float,
    degeneracy_epsilon_m: float,
) -> dict[str, Any]:
    binding_id = _binding_id(binding)
    report = _ReportBuilder(binding_id=binding_id)
    slot_refs = _extract_slot_refs(binding)
    secondary_refs = _extract_secondary_refs(binding)
    evidence_refs = _extract_evidence_refs(binding) | _visual_evidence_refs(visual_evidence)
    promoted_refs = _extract_promoted_refs(binding)
    required = _required_slots_for_binding(binding, required_slots)
    expectations = {**_AFFORDANCE_SLOT_EXPECTATIONS, **dict(slot_expectations or {})}

    for slot in required:
        if slot not in {item.slot for item in slot_refs}:
            report.add(
                "required_slot_present",
                passed=False,
                severity="error",
                message=f"binding is missing required slot {slot}",
                slot=slot,
                hint="Fill every declared required slot with an exact symbol-table ref before compiling.",
            )
        else:
            report.add(
                "required_slot_present",
                passed=True,
                severity="info",
                message=f"required slot {slot} is present",
                slot=slot,
            )

    if not slot_refs and not secondary_refs:
        report.add(
            "binding_has_refs",
            passed=False,
            severity="error",
            message="binding does not contain any executable refs",
            hint="Bind at least one slot to a symbolic ref listed in the symbol table.",
        )
        return report.report()

    snapshots: dict[str, RefSnapshot] = {}
    for binding_ref in [*slot_refs, *secondary_refs]:
        promoted = binding_ref.promoted or binding_ref.ref in promoted_refs
        snapshot = _resolve_ref(
            binding_ref.ref,
            resolver=resolver,
            symbol_table=symbol_table,
            symbol_lookup=symbol_lookup,
        )
        if isinstance(snapshot, str):
            report.add(
                "ref_resolves",
                passed=False,
                severity="error",
                message=snapshot,
                ref=binding_ref.ref,
                slot=binding_ref.slot,
                hint="Use a ref exactly listed in the symbol table, or promote the intended candidate into a functional ref first.",
            )
            continue

        snapshots[binding_ref.ref] = snapshot
        report.add(
            "ref_resolves",
            passed=True,
            severity="info",
            message=f"{binding_ref.ref} resolves",
            ref=binding_ref.ref,
            slot=binding_ref.slot,
        )
        _check_expected_geometry(
            report,
            binding_ref=binding_ref,
            snapshot=snapshot,
            expectation=_expectation_for_slot(binding_ref.slot, expectations),
            promoted=promoted,
        )
        _check_grounding_metadata(
            report,
            binding_ref=binding_ref,
            snapshot=snapshot,
            evidence_refs=evidence_refs,
            confidence_warning_threshold=confidence_warning_threshold,
        )

    _check_point_axis_compatibility(report, slot_refs=slot_refs, snapshots=snapshots)
    _check_constraint_frame_consistency(report, binding=binding, slot_refs=slot_refs, snapshots=snapshots)
    _check_secondary_ref_degeneracy(
        report,
        slot_refs=slot_refs,
        secondary_refs=secondary_refs,
        snapshots=snapshots,
        degeneracy_epsilon_m=degeneracy_epsilon_m,
    )
    _check_point_pair_degeneracy(
        report,
        slot_refs=slot_refs,
        snapshots=snapshots,
        degeneracy_epsilon_m=degeneracy_epsilon_m,
    )
    _check_evidence_only_refs(
        report,
        evidence_refs=evidence_refs,
        resolver=resolver,
        symbol_table=symbol_table,
        symbol_lookup=symbol_lookup,
    )
    return report.report()


def _affordance_contract_refs(binding: Mapping[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}

    def add(slot: str, value: Any) -> None:
        ref, _ = _ref_from_value(value)
        if not ref:
            nested = sorted(_collect_refs(value))
            if len(nested) == 1:
                ref = nested[0]
        if ref:
            refs[slot] = ref

    for container_key in (
        "slot_bindings",
        "slot_refs",
        "slots",
        "functional_frame_binding",
        "target_binding",
    ):
        container = binding.get(container_key)
        if isinstance(container, Mapping):
            for slot, value in container.items():
                add(str(slot), value)
    return refs


def _has_constraint(
    constraints: Sequence[Mapping[str, Any]],
    constraint_type: str,
    required_refs: Mapping[str, str],
) -> bool:
    for constraint in constraints:
        if constraint.get("type") != constraint_type:
            continue
        refs = constraint.get("refs", {})
        if not isinstance(refs, Mapping):
            continue
        if all(refs.get(slot) == ref for slot, ref in required_refs.items()):
            return True
    return False


def _has_directed_motion_path(
    constraints: Sequence[Mapping[str, Any]],
    *,
    source_ref: str,
    start_ref: str,
    goal_ref: str,
) -> bool:
    """Return true when move_along constraints connect start to goal.

    Some functional motions are naturally staged, e.g. pre_contact -> contact ->
    post_contact.  The contract only needs the selected function point to have a
    declared directed path through the bound refs; it should not require that
    path to be a single move_along edge.
    """

    edges: dict[str, set[str]] = {}
    for constraint in constraints:
        if constraint.get("type") != "move_along":
            continue
        refs = constraint.get("refs", {})
        if not isinstance(refs, Mapping):
            continue
        if refs.get("source") != source_ref:
            continue
        start = refs.get("start")
        end = refs.get("end")
        if isinstance(start, str) and isinstance(end, str) and start and end:
            edges.setdefault(start, set()).add(end)
    if goal_ref in edges.get(start_ref, set()):
        return True
    seen = {start_ref}
    frontier = [start_ref]
    while frontier:
        current = frontier.pop()
        for nxt in edges.get(current, set()):
            if nxt == goal_ref:
                return True
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def _constraint_ref_used(
    constraints: Sequence[Mapping[str, Any]],
    ref: str,
    *,
    types: set[str],
    slots: set[str],
) -> bool:
    for constraint in constraints:
        if constraint.get("type") not in types:
            continue
        refs = constraint.get("refs", {})
        if not isinstance(refs, Mapping):
            continue
        if any(refs.get(slot) == ref for slot in slots):
            return True
    return False


def _objective_refs(objective: Mapping[str, Any]) -> set[str]:
    refs = objective.get("refs", [])
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
        return set()
    return {str(ref) for ref in refs if isinstance(ref, str) and ref}


def _objective_ref_used(objectives: Sequence[Mapping[str, Any]], ref: str) -> bool:
    return any(ref in _objective_refs(objective) for objective in objectives)


def _has_objective_with_refs(
    objectives: Sequence[Mapping[str, Any]],
    required_refs: set[str],
) -> bool:
    if not required_refs:
        return False
    return any(required_refs.issubset(_objective_refs(objective)) for objective in objectives)


def _aggregate_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failed_refs: set[str] = set()
    warning_refs: set[str] = set()
    repair_hints: list[str] = []
    for report in reports:
        checks.extend(report.get("checks", []))
        failed_refs.update(str(ref) for ref in report.get("failed_refs", []))
        warning_refs.update(str(ref) for ref in report.get("warning_refs", []))
        for hint in report.get("repair_hints", []):
            if isinstance(hint, str) and hint not in repair_hints:
                repair_hints.append(hint)
    return {
        "schema_version": AFFORDANCE_BINDING_REPORT_VERSION,
        "passed": not any(
            not check.get("passed", False) and check.get("severity") == "error"
            for check in checks
        ),
        "checks": checks,
        "failed_refs": sorted(failed_refs),
        "warning_refs": sorted(warning_refs),
        "repair_hints": repair_hints,
    }


def _binding_id(binding: Mapping[str, Any]) -> str | None:
    for key in ("binding_id", "id", "constraint_id", "name"):
        value = binding.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _looks_like_ref(value: Any) -> bool:
    if not isinstance(value, str) or "." not in value:
        return False
    return value.startswith(("scene.", "tool.", "target.", "candidate."))


def _ref_from_value(value: Any) -> tuple[str | None, bool]:
    if _looks_like_ref(value):
        return str(value), False
    if not isinstance(value, Mapping):
        return None, False
    for key in ("ref", "symbol_ref", "value"):
        ref = value.get(key)
        if _looks_like_ref(ref):
            return str(ref), bool(value.get("promoted") or value.get("physical"))
    return None, False


def _extract_slot_refs(binding: Mapping[str, Any]) -> list[BindingRef]:
    refs: list[BindingRef] = []
    seen: set[tuple[str, str]] = set()

    def add(slot: str, value: Any) -> None:
        ref, promoted = _ref_from_value(value)
        if not ref:
            return
        key = (slot, ref)
        if key in seen:
            return
        refs.append(BindingRef(slot=slot, ref=ref, category="slot", promoted=promoted))
        seen.add(key)

    for container_key in (
        "refs",
        "slots",
        "slot_bindings",
        "slot_refs",
        "binding",
        "affordance_slots",
        "functional_frame_binding",
        "target_binding",
    ):
        container = binding.get(container_key)
        if isinstance(container, Mapping):
            for slot, value in container.items():
                direct_ref, _ = _ref_from_value(value)
                if direct_ref:
                    add(str(slot), value)
                    continue
                nested_refs = _collect_refs(value)
                if nested_refs:
                    for ref in sorted(nested_refs):
                        add(str(slot), ref)
                else:
                    add(str(slot), value)
    for slot, value in binding.items():
        if slot in {
            "refs",
            "slots",
            "slot_bindings",
            "slot_refs",
            "binding",
            "affordance_slots",
            "functional_frame_binding",
            "target_binding",
            "evidence",
            "evidence_refs",
            "visual_evidence",
            "visual_evidence_refs",
            "candidate_evidence_refs",
            "visual_anchor_evidence",
            "secondary_refs",
            "secondary_keypoint_refs",
            "secondary_slots",
            "promoted_refs",
            "required_slots",
            "slot_expectations",
            "selected_refs",
            "allowed_refs",
            "all_refs",
        }:
            continue
        add(str(slot), value)
    return refs


def _extract_secondary_refs(binding: Mapping[str, Any]) -> list[BindingRef]:
    refs: list[BindingRef] = []
    seen: set[str] = set()
    for key in ("secondary_refs", "secondary_keypoint_refs"):
        for ref in _collect_refs(binding.get(key)):
            if ref not in seen:
                refs.append(BindingRef(slot=key, ref=ref, category="secondary"))
                seen.add(ref)
    secondary_slots = binding.get("secondary_slots")
    if isinstance(secondary_slots, Mapping):
        for slot, value in secondary_slots.items():
            ref, promoted = _ref_from_value(value)
            if ref and ref not in seen:
                refs.append(
                    BindingRef(
                        slot=str(slot),
                        ref=ref,
                        category="secondary",
                        promoted=promoted,
                    )
                )
                seen.add(ref)
    return refs


def _extract_evidence_refs(binding: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in (
        "evidence",
        "evidence_refs",
        "visual_evidence",
        "visual_evidence_refs",
        "candidate_evidence_refs",
        "visual_anchor_evidence",
    ):
        refs.update(_collect_refs(binding.get(key)))
    return refs


def _extract_promoted_refs(binding: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("promoted_refs", "promoted_physical_refs", "physical_refs"):
        refs.update(_collect_refs(binding.get(key)))
    return refs


def _collect_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if _looks_like_ref(value):
        refs.add(str(value))
        return refs
    if isinstance(value, Mapping):
        for child in value.values():
            refs.update(_collect_refs(child))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            refs.update(_collect_refs(child))
    return refs


def _required_slots_for_binding(
    binding: Mapping[str, Any],
    required_slots: Iterable[str] | Mapping[str, Iterable[str]] | None,
) -> tuple[str, ...]:
    if isinstance(required_slots, Mapping):
        binding_type = str(binding.get("type", ""))
        return tuple(str(slot) for slot in required_slots.get(binding_type, ()))
    if required_slots is not None:
        return tuple(str(slot) for slot in required_slots)
    declared = binding.get("required_slots")
    if isinstance(declared, Iterable) and not isinstance(declared, (str, bytes, bytearray)):
        return tuple(str(slot) for slot in declared)
    binding_type = str(binding.get("type", ""))
    if "slot_bindings" in binding or "slot_refs" in binding:
        return _AFFORDANCE_REQUIRED_SLOTS
    return _CONSTRAINT_REQUIRED_SLOTS.get(binding_type, ())


def _expectation_for_slot(
    slot: str,
    slot_expectations: Mapping[str, GeometryExpectation],
) -> dict[str, Any]:
    if slot in slot_expectations:
        return dict(slot_expectations[slot])
    lowered = slot.lower()
    expectation: dict[str, Any] = {}
    if "axis" in lowered or "normal" in lowered:
        expectation["geometry"] = "axis"
    elif any(token in lowered for token in ("point", "center", "contact", "source", "target", "moving", "obstacle", "start", "goal")):
        expectation["geometry"] = "point"
    if lowered.startswith(("tool", "source", "moving", "function", "functional")):
        expectation["frame"] = "tool"
    elif lowered.startswith(("target", "scene", "obstacle", "start", "goal")) or lowered in {"axis", "target_axis"}:
        expectation["frame"] = "world"
    if any(lowered == name or lowered.startswith(f"{name}_") for name in _PHYSICAL_SLOT_NAMES):
        expectation["physical_contact"] = True
    return expectation


def _resolve_ref(
    ref: str,
    *,
    resolver: ReferenceResolver | Any | None,
    symbol_table: Mapping[str, Any] | None,
    symbol_lookup: SymbolLookup | None,
) -> RefSnapshot | str:
    if symbol_table is not None and not _symbol_table_contains(symbol_table, ref):
        return f"{ref} is not listed in the provided symbol table"
    try:
        if resolver is not None:
            resolved = resolver.resolve(ref)
            return _snapshot_from_resolved(resolved)
        if symbol_lookup is not None:
            resolved = symbol_lookup(ref)
            if resolved is None:
                return f"{ref} could not be resolved by symbol_lookup"
            return _snapshot_from_any(ref, resolved)
        if symbol_table is not None:
            return RefSnapshot(ref=ref, kind=None, frame=None, entity={})
        return f"{ref} has no resolver, symbol_lookup, or symbol_table"
    except (ReferenceResolutionError, ValueError, TypeError) as exc:
        return f"{ref} failed to resolve: {exc}"


def _symbol_table_contains(symbol_table: Mapping[str, Any], ref: str) -> bool:
    found = False

    def visit(value: Any) -> None:
        nonlocal found
        if found:
            return
        if value == ref:
            found = True
            return
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(symbol_table)
    return found


def _snapshot_from_resolved(resolved: ResolvedRef) -> RefSnapshot:
    return RefSnapshot(
        ref=resolved.ref,
        kind=resolved.kind,
        frame=resolved.frame,
        entity=resolved.entity,
        point_world=resolved.point_world,
        vector_world=resolved.vector_world,
        point_tool=resolved.point_tool,
        vector_tool=resolved.vector_tool,
        confidence=resolved.confidence,
        source=resolved.source,
        provenance=resolved.provenance,
    )


def _snapshot_from_any(ref: str, value: Any) -> RefSnapshot:
    if isinstance(value, ResolvedRef):
        return _snapshot_from_resolved(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{ref} resolver returned unsupported value")
    entity = value.get("entity") if isinstance(value.get("entity"), Mapping) else value
    kind = value.get("kind") if isinstance(value.get("kind"), str) else _infer_kind(entity)
    frame = value.get("frame") if isinstance(value.get("frame"), str) else _infer_frame(entity)
    return RefSnapshot(
        ref=ref,
        kind=kind,
        frame=frame,
        entity=entity,
        point_world=_vec_or_none(
            value.get("point_world")
            or entity.get("xyz_world")
            or entity.get("center_xyz_world")
            or entity.get("point_xyz_world")
        ),
        vector_world=_vec_or_none(value.get("vector_world") or entity.get("vector_world") or entity.get("normal_world")),
        point_tool=_vec_or_none(
            value.get("point_tool")
            or entity.get("xyz_tool")
            or entity.get("center_xyz_tool")
            or entity.get("point_xyz_tool")
            or entity.get("position_tool")
        ),
        vector_tool=_vec_or_none(value.get("vector_tool") or entity.get("vector_tool") or entity.get("normal_tool")),
        confidence=_confidence(entity),
        source=str(entity.get("source")) if isinstance(entity.get("source"), str) else None,
        provenance=entity.get("provenance") if isinstance(entity.get("provenance"), Mapping) else None,
    )


def _infer_kind(entity: Mapping[str, Any]) -> str | None:
    if any(key in entity for key in ("vector_world", "vector_tool")):
        return "axis"
    if any(key in entity for key in ("normal_world", "normal_tool")):
        return "surface"
    if any(key in entity for key in ("xyz_world", "xyz_tool", "center_xyz_world", "center_xyz_tool", "position_tool")):
        return "point"
    return None


def _infer_frame(entity: Mapping[str, Any]) -> str | None:
    if any(key in entity for key in ("xyz_tool", "center_xyz_tool", "point_xyz_tool", "vector_tool", "normal_tool", "position_tool")):
        return "tool"
    if any(key in entity for key in ("xyz_world", "center_xyz_world", "point_xyz_world", "vector_world", "normal_world")):
        return "world"
    frame = entity.get("frame")
    return str(frame) if isinstance(frame, str) else None


def _confidence(entity: Mapping[str, Any]) -> float | None:
    value = entity.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _vec_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) != 3:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _check_expected_geometry(
    report: _ReportBuilder,
    *,
    binding_ref: BindingRef,
    snapshot: RefSnapshot,
    expectation: Mapping[str, Any],
    promoted: bool,
) -> None:
    expected_geometry = expectation.get("geometry")
    expected_frame = expectation.get("frame")
    if expected_geometry == "point":
        has_point = _point_for_frame(snapshot, expected_frame) is not None
        kind_ok = snapshot.kind is None or snapshot.kind in _POINT_LIKE_KINDS
        report.add(
            "slot_geometry_type",
            passed=bool(has_point and kind_ok),
            severity="error",
            message=(
                f"{binding_ref.slot} expects a point-capable ref"
                if not has_point or not kind_ok
                else f"{binding_ref.slot} has point geometry"
            ),
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Choose a point-capable ref for point/contact slots.",
        )
    elif expected_geometry == "axis":
        vector = _vector_for_frame(snapshot, expected_frame)
        has_axis = vector is not None
        kind_ok = snapshot.kind is None or snapshot.kind in _AXIS_LIKE_KINDS
        finite_nonzero = _finite_nonzero(vector)
        report.add(
            "slot_geometry_type",
            passed=bool(has_axis and kind_ok and finite_nonzero),
            severity="error",
            message=(
                f"{binding_ref.slot} expects a finite nonzero axis/vector ref"
                if not has_axis or not kind_ok or not finite_nonzero
                else f"{binding_ref.slot} has finite nonzero axis geometry"
            ),
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Choose an axis/normal ref with a finite nonzero vector for axis slots.",
        )
    if expected_frame is not None:
        frame_ok = snapshot.frame in {expected_frame, None}
        if expected_geometry == "point":
            frame_ok = _point_for_frame(snapshot, expected_frame) is not None
        elif expected_geometry == "axis":
            frame_ok = _vector_for_frame(snapshot, expected_frame) is not None
        report.add(
            "slot_frame_type",
            passed=bool(frame_ok),
            severity="error",
            message=(
                f"{binding_ref.slot} expects a {expected_frame}-frame ref"
                if not frame_ok
                else f"{binding_ref.slot} uses {expected_frame}-frame geometry"
            ),
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Use tool-frame refs for tool slots and world-frame refs for scene/target slots.",
        )
    if expectation.get("physical_contact") and _is_candidate_ref(binding_ref.ref):
        allowed = promoted or _explicitly_promoted(snapshot)
        report.add(
            "candidate_not_physical_contact",
            passed=allowed,
            severity="error",
            message=(
                f"{binding_ref.ref} is a candidate/evidence ref without executable tool-prior grounding"
                if not allowed
                else f"{binding_ref.ref} is executable tool-prior candidate geometry"
            ),
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint=(
                "Use tool.functional_parts.* or tool-frame SAMPart3D geometry candidate "
                "point/axis/surface refs with tool part provenance for executable tool contact slots."
            ),
        )


def _point_for_frame(snapshot: RefSnapshot, frame: Any) -> list[float] | None:
    if frame == "tool":
        return snapshot.point_tool
    if frame == "world":
        return snapshot.point_world
    return snapshot.point_tool or snapshot.point_world


def _vector_for_frame(snapshot: RefSnapshot, frame: Any) -> list[float] | None:
    if frame == "tool":
        return snapshot.vector_tool
    if frame == "world":
        return snapshot.vector_world
    return snapshot.vector_tool or snapshot.vector_world


def _finite_nonzero(values: list[float] | None) -> bool:
    if values is None:
        return False
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        return False
    return math.sqrt(sum(float(value) * float(value) for value in values)) > EPS


def _is_candidate_ref(ref: str) -> bool:
    return ref.startswith("candidate.")


def _explicitly_promoted(snapshot: RefSnapshot) -> bool:
    entity = snapshot.entity
    provenance = snapshot.provenance or {}
    for container in (entity, provenance):
        if container.get("promoted") is True:
            return True
        if container.get("promoted_to_physical_ref") is True:
            return True
        promoted_to = container.get("promoted_to_ref")
        if isinstance(promoted_to, str) and promoted_to.startswith("tool.functional_parts."):
            return True
    if _is_tool_prior_candidate(snapshot):
        return True
    return False


def _is_tool_prior_candidate(snapshot: RefSnapshot) -> bool:
    if not snapshot.ref.startswith("candidate."):
        return False
    parts = snapshot.ref.split(".")
    if len(parts) < 3 or parts[1] not in {"points", "axes", "surfaces"}:
        return False
    if not (
        snapshot.frame == "tool"
        or snapshot.point_tool is not None
        or snapshot.vector_tool is not None
    ):
        return False
    entity = snapshot.entity
    provenance = snapshot.provenance or {}
    part_ref = entity.get("part_ref") or provenance.get("part_ref")
    if not (isinstance(part_ref, str) and part_ref.startswith("tool.parts.")):
        return False
    text_fields: list[str] = []
    for container in (entity, provenance):
        for key in (
            "source",
            "source_stage",
            "provider_id",
            "candidate_type",
            "method",
            "generator",
            "label",
            "source_anchor_id",
        ):
            value = container.get(key)
            if isinstance(value, str):
                text_fields.append(value.lower())
    return any(
        token in value
        for value in text_fields
        for token in (
            "sampart3d",
            "tool_prior",
            "geometry_anchor",
            "geometry_surface",
            "functional_frame",
        )
    )


def _check_grounding_metadata(
    report: _ReportBuilder,
    *,
    binding_ref: BindingRef,
    snapshot: RefSnapshot,
    evidence_refs: set[str],
    confidence_warning_threshold: float,
) -> None:
    confidence = snapshot.confidence
    if confidence is None:
        report.add(
            "confidence_present",
            passed=False,
            severity="warning",
            message=f"{binding_ref.ref} has no confidence score",
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Prefer refs with confidence and provenance metadata when repairing bindings.",
        )
    elif not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        report.add(
            "confidence_range",
            passed=False,
            severity="error",
            message=f"{binding_ref.ref} confidence is outside [0, 1]",
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Repair or regenerate grounding metadata with a normalized confidence score.",
        )
    elif confidence < confidence_warning_threshold:
        report.add(
            "confidence_threshold",
            passed=False,
            severity="warning",
            message=f"{binding_ref.ref} confidence {confidence:.3f} is low",
            ref=binding_ref.ref,
            slot=binding_ref.slot,
            hint="Prefer a higher-confidence ref or add evidence before repair retries.",
        )
    else:
        report.add(
            "confidence_present",
            passed=True,
            severity="info",
            message=f"{binding_ref.ref} confidence is present",
            ref=binding_ref.ref,
            slot=binding_ref.slot,
        )

    has_provenance = isinstance(snapshot.provenance, Mapping) and bool(snapshot.provenance)
    report.add(
        "provenance_present",
        passed=has_provenance,
        severity="warning",
        message=(
            f"{binding_ref.ref} has provenance metadata"
            if has_provenance
            else f"{binding_ref.ref} has no provenance metadata"
        ),
        ref=binding_ref.ref,
        slot=binding_ref.slot,
        hint="Prefer refs with provenance so repair can distinguish visual, mesh, and promoted sources.",
    )

    has_visual_evidence = binding_ref.ref in evidence_refs or _entity_has_visual_evidence(snapshot.entity)
    report.add(
        "visual_evidence_available",
        passed=has_visual_evidence,
        severity="warning",
        message=(
            f"{binding_ref.ref} has visual/artifact evidence"
            if has_visual_evidence
            else f"{binding_ref.ref} is not linked to visual evidence"
        ),
        ref=binding_ref.ref,
        slot=binding_ref.slot,
        hint="Attach candidate/visual evidence refs to the binding for repair diagnostics.",
    )


def _entity_has_visual_evidence(entity: Mapping[str, Any]) -> bool:
    visual_keys = {
        "pixel",
        "bbox",
        "mask_id",
        "view_index",
        "patch",
        "support_pixels",
        "source_image",
        "source_artifact",
    }
    if any(key in entity for key in visual_keys):
        return True
    provenance = entity.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    return any(key in provenance for key in visual_keys | {"image_id", "anchor_set_id", "detection_id"})


def _check_point_axis_compatibility(
    report: _ReportBuilder,
    *,
    slot_refs: Sequence[BindingRef],
    snapshots: Mapping[str, RefSnapshot],
) -> None:
    refs_by_slot = {item.slot: item for item in slot_refs if item.ref in snapshots}
    for point_slot, axis_slot in _point_axis_pairs(refs_by_slot):
        point_ref = refs_by_slot[point_slot]
        axis_ref = refs_by_slot[axis_slot]
        point_snapshot = snapshots[point_ref.ref]
        axis_snapshot = snapshots[axis_ref.ref]
        compatibility = _provenance_compatibility(point_snapshot, axis_snapshot)
        if compatibility == "compatible":
            report.add(
                "point_axis_same_part_or_provenance",
                passed=True,
                severity="info",
                message=f"{point_slot} and {axis_slot} share part/provenance",
                ref=axis_ref.ref,
                slot=axis_slot,
            )
        elif compatibility == "unknown":
            report.add(
                "point_axis_same_part_or_provenance",
                passed=False,
                severity="warning",
                message=f"{point_slot} and {axis_slot} lack enough part/provenance metadata to prove compatibility",
                ref=axis_ref.ref,
                slot=axis_slot,
                hint="Use point and axis refs from the same functional part or from artifacts with shared provenance.",
            )
        else:
            report.add(
                "point_axis_same_part_or_provenance",
                passed=False,
                severity="error",
                message=f"{point_slot} and {axis_slot} appear to come from incompatible parts/provenance",
                ref=axis_ref.ref,
                slot=axis_slot,
                hint="Repair the binding so paired point/axis refs come from the same part or compatible provenance.",
            )


def _check_constraint_frame_consistency(
    report: _ReportBuilder,
    *,
    binding: Mapping[str, Any],
    slot_refs: Sequence[BindingRef],
    snapshots: Mapping[str, RefSnapshot],
) -> None:
    binding_type = str(binding.get("type", ""))
    if binding_type != "distance":
        return
    refs_by_slot = {item.slot: item for item in slot_refs if item.ref in snapshots}
    if "a" not in refs_by_slot or "b" not in refs_by_slot:
        return
    a = snapshots[refs_by_slot["a"].ref]
    b = snapshots[refs_by_slot["b"].ref]
    a_tool = a.point_tool is not None
    a_world = a.point_world is not None
    b_tool = b.point_tool is not None
    b_world = b.point_world is not None
    compatible = (a_tool and b_world) or (b_tool and a_world)
    report.add(
        "constraint_frame_consistency",
        passed=compatible,
        severity="error",
        message=(
            "distance binding has one tool-frame point and one world-frame point"
            if compatible
            else "distance binding needs one tool-frame point and one world-frame point"
        ),
        ref=refs_by_slot["b"].ref,
        slot="distance",
        hint="Repair distance refs so one endpoint is a tool point and the other is a scene/target point.",
    )


def _point_axis_pairs(refs_by_slot: Mapping[str, BindingRef]) -> list[tuple[str, str]]:
    explicit_pairs = [
        ("tool_contact_point", "tool_contact_axis"),
        ("tool_contact_point", "tool_surface_normal"),
        ("function_point", "function_axis"),
        ("function_keypoint_ref", "function_axis_ref"),
        ("functional_point", "functional_axis"),
        ("source", "source_axis"),
        ("target_contact", "target_approach_axis"),
        ("target_contact_ref", "motion_axis_ref"),
        ("target", "axis"),
        ("moving", "moving_axis"),
    ]
    pairs: list[tuple[str, str]] = []
    for point_slot, axis_slot in explicit_pairs:
        if point_slot in refs_by_slot and axis_slot in refs_by_slot:
            pairs.append((point_slot, axis_slot))
    slots = sorted(refs_by_slot)
    for point_slot in slots:
        if not any(token in point_slot.lower() for token in ("point", "contact", "source", "target", "moving")):
            continue
        prefix = _slot_prefix(point_slot)
        for axis_slot in slots:
            if axis_slot == point_slot or "axis" not in axis_slot.lower() and "normal" not in axis_slot.lower():
                continue
            if _slot_prefix(axis_slot) == prefix and (point_slot, axis_slot) not in pairs:
                pairs.append((point_slot, axis_slot))
    return pairs


def _slot_prefix(slot: str) -> str:
    lowered = slot.lower()
    for suffix in ("_point", "_center", "_contact", "_axis", "_normal"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _provenance_compatibility(a: RefSnapshot, b: RefSnapshot) -> str:
    a_parts = _part_keys(a)
    b_parts = _part_keys(b)
    if a.ref.startswith("tool.functional_parts.") and b.ref.startswith("tool.functional_parts."):
        return "compatible" if _functional_part_id(a.ref) == _functional_part_id(b.ref) else "conflict"
    if a_parts and b_parts:
        return "compatible" if a_parts & b_parts else "conflict"
    a_prov = _provenance_keys(a)
    b_prov = _provenance_keys(b)
    if a_prov and b_prov:
        return "compatible" if a_prov & b_prov else "unknown"
    return "unknown"


def _functional_part_id(ref: str) -> str | None:
    parts = ref.split(".")
    if len(parts) >= 4 and parts[:2] == ["tool", "functional_parts"]:
        return parts[2]
    return None


def _part_keys(snapshot: RefSnapshot) -> set[str]:
    keys: set[str] = set()
    for container in (snapshot.entity, snapshot.provenance or {}):
        for key in ("part_ref", "object_ref", "target_part_ref", "tool_part_ref", "part_id", "object_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                keys.add(value)
                if key in {"object_ref", "object_id"}:
                    if value.startswith("scene.objects."):
                        keys.add(value[len("scene.objects.") :])
                    else:
                        keys.add(f"scene.objects.{value}")
    functional_id = _functional_part_id(snapshot.ref)
    if functional_id:
        keys.add(f"tool.functional_parts.{functional_id}")
        keys.add(functional_id)
    return keys


def _provenance_keys(snapshot: RefSnapshot) -> set[str]:
    provenance = snapshot.provenance or {}
    keys: set[str] = set()
    for key in ("source_artifact", "observation_id", "anchor_set_id", "detection_id", "method"):
        value = provenance.get(key)
        if isinstance(value, str) and value:
            keys.add(f"{key}:{value}")
    if snapshot.source:
        keys.add(f"source:{snapshot.source}")
    return keys


def _check_secondary_ref_degeneracy(
    report: _ReportBuilder,
    *,
    slot_refs: Sequence[BindingRef],
    secondary_refs: Sequence[BindingRef],
    snapshots: Mapping[str, RefSnapshot],
    degeneracy_epsilon_m: float,
) -> None:
    if not secondary_refs:
        return
    point_groups: dict[str, list[tuple[str, list[float]]]] = {"world": [], "tool": []}
    for binding_ref in [*slot_refs, *secondary_refs]:
        snapshot = snapshots.get(binding_ref.ref)
        if snapshot is None:
            continue
        if snapshot.point_world is not None:
            point_groups["world"].append((binding_ref.ref, snapshot.point_world))
        if snapshot.point_tool is not None:
            point_groups["tool"].append((binding_ref.ref, snapshot.point_tool))
    checked_any = False
    for frame, points in point_groups.items():
        if len(points) < 2:
            continue
        checked_any = True
        max_distance = 0.0
        for i, (_, point_a) in enumerate(points):
            for _, point_b in points[i + 1 :]:
                max_distance = max(max_distance, _distance(point_a, point_b))
        report.add(
            "secondary_refs_non_degenerate",
            passed=max_distance > degeneracy_epsilon_m,
            severity="error",
            message=(
                f"secondary refs have non-degenerate {frame}-frame point spread"
                if max_distance > degeneracy_epsilon_m
                else f"secondary refs collapse to the same {frame}-frame point"
            ),
            ref=points[0][0],
            slot="secondary_refs",
            hint="Use distinct secondary keypoints/refs when a binding declares multi-keypoint evidence.",
            details={"frame": frame, "max_pairwise_distance_m": max_distance},
        )
    if not checked_any:
        report.add(
            "secondary_refs_non_degenerate",
            passed=False,
            severity="warning",
            message="secondary refs were declared but did not resolve to comparable point geometry",
            slot="secondary_refs",
            hint="Use point-capable secondary refs when a binding needs multi-keypoint evidence.",
        )


def _check_point_pair_degeneracy(
    report: _ReportBuilder,
    *,
    slot_refs: Sequence[BindingRef],
    snapshots: Mapping[str, RefSnapshot],
    degeneracy_epsilon_m: float,
) -> None:
    refs_by_slot = {item.slot: item for item in slot_refs if item.ref in snapshots}
    pair_candidates = [
        ("axis_source_ref", "contact_ref", "functional_frame_point_pair"),
        ("axis_source_ref", "primary_ref", "functional_frame_point_pair"),
        ("contact_ref", "secondary_ref", "point_pair_functional_frame"),
        ("primary_ref", "secondary_ref", "point_pair_functional_frame"),
        ("source_start", "source_end", "align_vector_between_points_pair"),
    ]
    checked: set[tuple[str, str, str]] = set()
    for start_slot, end_slot, check_name in pair_candidates:
        if start_slot not in refs_by_slot or end_slot not in refs_by_slot:
            continue
        key = (refs_by_slot[start_slot].ref, refs_by_slot[end_slot].ref, check_name)
        if key in checked:
            continue
        checked.add(key)
        start_snapshot = snapshots[refs_by_slot[start_slot].ref]
        end_snapshot = snapshots[refs_by_slot[end_slot].ref]
        start_point = start_snapshot.point_tool
        end_point = end_snapshot.point_tool
        if start_point is None or end_point is None:
            report.add(
                check_name,
                passed=False,
                severity="warning",
                message=f"{start_slot} and {end_slot} do not both resolve to tool-frame points",
                ref=refs_by_slot[end_slot].ref,
                slot=end_slot,
                hint="Use tool-frame point refs for the derived functional-frame direction.",
            )
            continue
        distance_m = _distance(start_point, end_point)
        report.add(
            check_name,
            passed=distance_m > degeneracy_epsilon_m,
            severity="error",
            message=(
                f"{start_slot} and {end_slot} define a non-degenerate tool-frame direction"
                if distance_m > degeneracy_epsilon_m
                else f"{start_slot} and {end_slot} collapse to the same tool-frame point"
            ),
            ref=refs_by_slot[end_slot].ref,
            slot=end_slot,
            hint="Use distinct axis_source_ref and contact_ref points when a binding or constraint defines a functional direction.",
            details={"distance_m": distance_m},
        )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _check_evidence_only_refs(
    report: _ReportBuilder,
    *,
    evidence_refs: set[str],
    resolver: ReferenceResolver | Any | None,
    symbol_table: Mapping[str, Any] | None,
    symbol_lookup: SymbolLookup | None,
) -> None:
    for ref in sorted(evidence_refs):
        snapshot = _resolve_ref(
            ref,
            resolver=resolver,
            symbol_table=symbol_table,
            symbol_lookup=symbol_lookup,
        )
        report.add(
            "evidence_ref_resolves",
            passed=not isinstance(snapshot, str),
            severity="warning",
            message=snapshot if isinstance(snapshot, str) else f"{ref} evidence ref resolves",
            ref=ref,
            slot="evidence_refs",
            hint="Remove stale evidence refs or replace them with listed candidate refs.",
        )


def _visual_evidence_refs(visual_evidence: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(visual_evidence, Mapping):
        return set()
    refs: set[str] = set()
    refs.update(_collect_refs(visual_evidence.get("numbered_anchor_refs")))
    metadata = visual_evidence.get("metadata")
    if isinstance(metadata, Mapping):
        refs.update(_collect_refs(metadata.get("anchor_ref_table")))
    for item in visual_evidence.get("items", []) if isinstance(visual_evidence.get("items"), list) else []:
        if isinstance(item, Mapping):
            refs.update(_collect_refs(item.get("refs")))
            refs.update(_collect_refs(item.get("anchor_ref_table")))
    return refs
