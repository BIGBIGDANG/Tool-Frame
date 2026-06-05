#!/usr/bin/env python3
"""Program API exposed to restricted PartToolGen Python code.

The API collects symbolic part/scene constraints and objective-function names.
It does not expose simulator state, robot commands, filesystem access, or raw
SimToolReal fixed-goal writing. Metric geometry is resolved later by the
compiler/optimizer from grounded artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable


API_VERSION = "parttoolgen_program_api_v0.3"


class ProgramAPIError(ValueError):
    """Raised when planner code violates the PartToolGen API contract."""


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProgramAPIError(f"{name} must be a non-empty string")
    if "__" in value:
        raise ProgramAPIError(f"{name} must not contain dunder segments")
    return value


def _bounded_number(
    value: float | int,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramAPIError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProgramAPIError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ProgramAPIError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ProgramAPIError(f"{name} must be <= {maximum}")
    return number


@dataclass(frozen=True)
class SymbolRef:
    """Typed symbolic reference selected by the planner program."""

    ref: str

    def __str__(self) -> str:
        return self.ref


@dataclass
class ConstraintCall:
    """Serializable constraint call emitted by the program API."""

    type: str
    refs: dict[str, str]
    params: dict[str, Any] = field(default_factory=dict)
    hard: bool = True
    weight: float = 1.0
    source: str = ""


@dataclass
class ObjectiveCall:
    """Reference to a restricted Python objective function."""

    kind: str
    function_name: str
    weight: float = 1.0
    refs: list[str] = field(default_factory=list)
    source: str = ""


class ConstraintAPI:
    """Constraint builders available as `ctx.constraints.*`."""

    def _ref(self, value: SymbolRef, name: str) -> str:
        if not isinstance(value, SymbolRef):
            raise ProgramAPIError(f"{name} must be created with ctx.ref(...)")
        return _non_empty(value.ref, name)

    def contact(
        self,
        source: SymbolRef,
        target: SymbolRef,
        *,
        tolerance_m: float = 0.005,
        hard: bool = True,
        weight: float = 1.0,
    ) -> ConstraintCall:
        return ConstraintCall(
            type="contact",
            refs={"source": self._ref(source, "source"), "target": self._ref(target, "target")},
            params={"tolerance_m": _bounded_number(tolerance_m, "tolerance_m", minimum=0.0, maximum=0.5)},
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def distance(
        self,
        a: SymbolRef,
        b: SymbolRef,
        *,
        target_m: float,
        tolerance_m: float = 0.01,
        hard: bool = False,
        weight: float = 1.0,
    ) -> ConstraintCall:
        return ConstraintCall(
            type="distance",
            refs={"a": self._ref(a, "a"), "b": self._ref(b, "b")},
            params={
                "target_m": _bounded_number(target_m, "target_m", minimum=0.0, maximum=5.0),
                "tolerance_m": _bounded_number(tolerance_m, "tolerance_m", minimum=0.0, maximum=0.5),
            },
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def align_axis(
        self,
        source_axis: SymbolRef,
        target_axis: SymbolRef,
        *,
        polarity: str = "same",
        min_cosine: float = 0.95,
        hard: bool = True,
        weight: float = 1.0,
    ) -> ConstraintCall:
        if polarity not in {"same", "opposite", "auto"}:
            raise ProgramAPIError("polarity must be same, opposite, or auto")
        return ConstraintCall(
            type="align_axis",
            refs={
                "source_axis": self._ref(source_axis, "source_axis"),
                "target_axis": self._ref(target_axis, "target_axis"),
            },
            params={
                "polarity": polarity,
                "min_cosine": _bounded_number(min_cosine, "min_cosine", minimum=-1.0, maximum=1.0),
            },
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def align_vector_between_points(
        self,
        source_start: SymbolRef,
        source_end: SymbolRef,
        target_axis: SymbolRef,
        *,
        polarity: str = "same",
        min_cosine: float = 0.95,
        hard: bool = True,
        weight: float = 1.0,
    ) -> ConstraintCall:
        if polarity not in {"same", "opposite", "auto"}:
            raise ProgramAPIError("polarity must be same, opposite, or auto")
        return ConstraintCall(
            type="align_vector_between_points",
            refs={
                "source_start": self._ref(source_start, "source_start"),
                "source_end": self._ref(source_end, "source_end"),
                "target_axis": self._ref(target_axis, "target_axis"),
            },
            params={
                "polarity": polarity,
                "min_cosine": _bounded_number(min_cosine, "min_cosine", minimum=-1.0, maximum=1.0),
            },
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def pre_contact_offset(
        self,
        source: SymbolRef,
        target: SymbolRef,
        axis: SymbolRef,
        *,
        distance_m: float,
        direction: str = "opposite_axis",
        hard: bool = False,
        weight: float = 1.0,
    ) -> ConstraintCall:
        """Place a tool point near a target by an explicit signed axis offset.

        `along_axis` means target + axis * distance; `opposite_axis` means
        target - axis * distance. The planner chooses the semantic side.
        """
        if direction not in {"opposite_axis", "along_axis"}:
            raise ProgramAPIError("direction must be opposite_axis or along_axis")
        return ConstraintCall(
            type="pre_contact_offset",
            refs={
                "source": self._ref(source, "source"),
                "target": self._ref(target, "target"),
                "axis": self._ref(axis, "axis"),
            },
            params={
                "distance_m": _bounded_number(distance_m, "distance_m", minimum=0.0, maximum=5.0),
                "direction": direction,
            },
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def clearance(
        self,
        moving: SymbolRef,
        obstacle: SymbolRef,
        *,
        min_distance_m: float,
        hard: bool = True,
        weight: float = 1.0,
    ) -> ConstraintCall:
        return ConstraintCall(
            type="clearance",
            refs={
                "moving": self._ref(moving, "moving"),
                "obstacle": self._ref(obstacle, "obstacle"),
            },
            params={"min_distance_m": _bounded_number(min_distance_m, "min_distance_m", minimum=0.0, maximum=5.0)},
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def move_along(
        self,
        source: SymbolRef,
        start: SymbolRef,
        goal: SymbolRef,
        *,
        profile: str = "linear",
        path_family: str | None = None,
        swing_axis: SymbolRef | None = None,
        swing_angle_rad: float = 0.55,
        swing_lift_m: float = 0.03,
        hard: bool = False,
        weight: float = 1.0,
    ) -> ConstraintCall:
        if profile not in {"linear", "swing_arc"}:
            raise ProgramAPIError("move_along profile must be linear or swing_arc")
        if profile == "swing_arc" and swing_axis is None:
            raise ProgramAPIError("swing_arc move_along requires swing_axis")
        if path_family is not None and path_family not in {"contact_follow", "slide", "swing"}:
            raise ProgramAPIError("path_family must be contact_follow, slide, or swing")
        refs = {
            "source": self._ref(source, "source"),
            "start": self._ref(start, "start"),
            "end": self._ref(goal, "goal"),
        }
        if swing_axis is not None:
            refs["swing_axis"] = self._ref(swing_axis, "swing_axis")
        params: dict[str, Any] = {}
        if path_family is not None:
            params["path_family"] = path_family
        if profile != "linear":
            params["profile"] = profile
            params["swing_angle_rad"] = _bounded_number(
                swing_angle_rad,
                "swing_angle_rad",
                minimum=0.01,
                maximum=1.57,
            )
            params["swing_lift_m"] = _bounded_number(
                swing_lift_m,
                "swing_lift_m",
                minimum=0.0,
                maximum=0.25,
            )
        return ConstraintCall(
            type="move_along",
            refs=refs,
            params=params,
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )

    def tool_occupancy_clearance(
        self,
        tool_part: SymbolRef,
        obstacle: SymbolRef,
        *,
        min_distance_m: float = 0.01,
        samples: str = "obb_corners",
        hard: bool = True,
        weight: float = 1.0,
    ) -> ConstraintCall:
        if samples not in {"centroid", "obb_corners", "surface_patch_centers"}:
            raise ProgramAPIError(
                "samples must be centroid, obb_corners, or surface_patch_centers"
            )
        return ConstraintCall(
            type="tool_occupancy_clearance",
            refs={
                "tool_part": self._ref(tool_part, "tool_part"),
                "obstacle": self._ref(obstacle, "obstacle"),
            },
            params={
                "min_distance_m": _bounded_number(
                    min_distance_m,
                    "min_distance_m",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "samples": samples,
            },
            hard=hard,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
        )


class StageBuilder:
    """Mutable stage builder used inside `build_program(ctx)`."""

    def __init__(self, context: "ProgramContext", name: str, description: str = "") -> None:
        self._context = context
        self.name = _non_empty(name, "stage name")
        self.description = description
        self.repeat_count = 1

    def constraint(self, call: ConstraintCall) -> ConstraintCall:
        if not isinstance(call, ConstraintCall):
            raise ProgramAPIError("stage.constraint expects a ConstraintCall")
        self._context._constraints.append((self.name, call))
        return call

    def objective(
        self,
        function: Callable[..., Any],
        *,
        kind: str = "subgoal",
        weight: float = 1.0,
        refs: list[SymbolRef] | tuple[SymbolRef, ...] | None = None,
    ) -> ObjectiveCall:
        if kind not in {"subgoal", "path"}:
            raise ProgramAPIError("objective kind must be subgoal or path")
        name = getattr(function, "__name__", "")
        _non_empty(name, "objective function name")
        if not refs:
            raise ProgramAPIError("objective refs must be an explicit non-empty list")
        objective_refs = []
        for index, ref in enumerate(refs):
            objective_refs.append(self._context.constraints._ref(ref, f"objective refs[{index}]"))
        call = ObjectiveCall(
            kind=kind,
            function_name=name,
            weight=_bounded_number(weight, "weight", minimum=0.0, maximum=1000.0),
            refs=objective_refs,
        )
        self._context._objectives.append((self.name, call))
        return call

    def repeat(self, count: int) -> "StageBuilder":
        if not isinstance(count, int) or count < 1:
            raise ProgramAPIError("repeat count must be a positive integer")
        self.repeat_count = count
        return self


class ProgramContext:
    """Context object passed to the planner `build_program(ctx)` function."""

    def __init__(
        self,
        *,
        program_id: str,
        task_instruction: str,
        allowed_refs: set[str] | None = None,
    ) -> None:
        self.program_id = _non_empty(program_id, "program_id")
        self.task_instruction = _non_empty(task_instruction, "task_instruction")
        self.allowed_refs = set(allowed_refs or set())
        self.constraints = ConstraintAPI()
        self._stages: list[StageBuilder] = []
        self._constraints: list[tuple[str, ConstraintCall]] = []
        self._objectives: list[tuple[str, ObjectiveCall]] = []

    def ref(self, ref: str) -> SymbolRef:
        ref = _non_empty(ref, "ref")
        if self.allowed_refs and ref not in self.allowed_refs:
            raise ProgramAPIError(f"unknown symbol reference: {ref}")
        return SymbolRef(ref)

    def stage(self, name: str, description: str = "") -> StageBuilder:
        stage = StageBuilder(self, name, description)
        if any(existing.name == stage.name for existing in self._stages):
            raise ProgramAPIError(f"duplicate stage: {stage.name}")
        self._stages.append(stage)
        return stage

    def program(self) -> dict[str, Any]:
        """Return the collected serializable ProgramIR."""

        if not self._stages:
            raise ProgramAPIError("planner program did not declare any stages")
        stage_names = {stage.name for stage in self._stages}
        constraints = []
        for index, (stage_name, call) in enumerate(self._constraints):
            if stage_name not in stage_names:
                raise ProgramAPIError(f"constraint references unknown stage: {stage_name}")
            constraints.append(
                {
                    "id": f"{stage_name}_constraint_{index + 1}",
                    "stage": stage_name,
                    "type": call.type,
                    "refs": call.refs,
                    "params": call.params,
                    "hard": call.hard,
                    "weight": call.weight,
                    "source": call.source,
                }
            )
        objectives = []
        for index, (stage_name, call) in enumerate(self._objectives):
            if stage_name not in stage_names:
                raise ProgramAPIError(f"objective references unknown stage: {stage_name}")
            objectives.append(
                {
                    "id": f"{stage_name}_objective_{index + 1}",
                    "stage": stage_name,
                    "kind": call.kind,
                    "function_name": call.function_name,
                    "weight": call.weight,
                    "refs": call.refs,
                    "source": call.source,
                }
            )
        return {
            "schema_version": "v0.1",
            "program_id": self.program_id,
            "task_instruction": self.task_instruction,
            "api_version": API_VERSION,
            "stages": [
                {
                    "name": stage.name,
                    "description": stage.description,
                    "repeat": stage.repeat_count,
                }
                for stage in self._stages
            ],
            "constraints": constraints,
            "objectives": objectives,
            "diagnostics": {
                "ast_checked": True,
                "executed": True,
            },
        }
