"""Constraint-program data structures for ToolFrame."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


IR_NAME = "Tool-Scene Constraint IR"
COMPILER_NAME = "Stage-wise Geometric Constraint Compiler"
SUBGOAL_NAME = "Whole-tool SE(3) Subgoals"
CONSTRAINT_TARGET_TYPES = (
    "PointTarget",
    "AxisTarget",
    "MoveAlongTarget",
    "OrientationTarget",
    "ClearanceTarget",
)
AXIS_RELATIONS = {
    "parallel",
    "same",
    "anti_parallel",
    "antiparallel",
    "opposite",
}


@dataclass(frozen=True)
class Constraint:
    """A single symbolic geometric constraint.

    The compiler resolves string refs such as `tool.contact_point` and
    `scene.start` after image grounding/lifting has produced concrete 3D refs.
    """

    constraint_type: str
    params: Mapping[str, Any]
    weight: float = 1.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = data.pop("constraint_type")
        return data


@dataclass(frozen=True, init=False)
class ConstraintStage:
    """A named phase of a tool-use interaction.

    `subgoal_constraints` define the terminal pose for a stage. `path_constraints`
    define what must remain true while moving toward that terminal pose. The
    legacy `constraints=` argument is accepted as a subgoal alias so older
    callers can be migrated gradually.
    """

    name: str
    subgoal_constraints: Sequence[Constraint]
    path_constraints: Sequence[Constraint]
    notes: str | None = None

    def __init__(
        self,
        name: str,
        subgoal_constraints: Iterable[Constraint] | None = None,
        path_constraints: Iterable[Constraint] | None = None,
        *,
        constraints: Iterable[Constraint] | None = None,
        notes: str | None = None,
    ) -> None:
        if subgoal_constraints is not None and constraints is not None:
            raise ValueError("use either subgoal_constraints or legacy constraints, not both")
        if subgoal_constraints is None:
            subgoal_constraints = constraints or ()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "subgoal_constraints", tuple(subgoal_constraints))
        object.__setattr__(self, "path_constraints", tuple(path_constraints or ()))
        object.__setattr__(self, "notes", notes)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return all constraints for compatibility with older call sites."""

        return tuple(self.subgoal_constraints) + tuple(self.path_constraints)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "subgoal_constraints": [
                constraint.to_json() for constraint in self.subgoal_constraints
            ],
            "path_constraints": [
                constraint.to_json() for constraint in self.path_constraints
            ],
        }
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True)
class ConstraintProgram:
    """A staged, image-grounded tool-frame constraint program."""

    task_id: str
    instruction: str
    tool_refs: Mapping[str, Any]
    scene_refs: Mapping[str, Any]
    stages: Sequence[ConstraintStage] = field(default_factory=tuple)
    schema_version: str = "toolframe_constraint_program_v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        metadata = {
            "ir_name": IR_NAME,
            "compiler": COMPILER_NAME,
            "subgoal_artifact": SUBGOAL_NAME,
            "constraint_target_types": list(CONSTRAINT_TARGET_TYPES),
            **dict(self.metadata),
        }
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": metadata,
            "tool_refs": dict(self.tool_refs),
            "scene_refs": dict(self.scene_refs),
            "stages": [stage.to_json() for stage in self.stages],
        }


def build_contact_motion_program(
    *,
    task_id: str,
    instruction: str,
    tool_contact_point_ref: str = "tool.contact_point",
    tool_motion_axis_ref: str = "tool.motion_axis",
    scene_start_ref: str = "scene.start",
    scene_end_ref: str = "scene.end",
    scene_direction_ref: str = "scene.direction",
    scene_contact_ref: str | None = None,
    scene_action_end_ref: str | None = None,
    scene_motion_path_ref: str | None = None,
    contact_approach_axis_ref: str | None = None,
    tool_motion_axis_target_ref: str | None = None,
    tool_motion_axis_relation: str = "parallel",
    pre_contact_distance_m: float = 0.08,
    tool_refs: Mapping[str, Any] | None = None,
    scene_refs: Mapping[str, Any] | None = None,
) -> ConstraintProgram:
    """Build the common point/axis/move constraint skeleton.

    This is intentionally not a brush or hammer primitive. The same skeleton can
    represent pushing, sweeping, scraping, or a translational strike once the VLM
    grounds the relevant tool and scene refs.
    """

    if tool_motion_axis_relation not in AXIS_RELATIONS:
        raise ValueError(f"unsupported tool_motion_axis_relation: {tool_motion_axis_relation}")

    start_ref = scene_contact_ref or scene_start_ref
    end_ref = scene_action_end_ref or scene_end_ref
    direction_ref = scene_direction_ref
    motion_path_ref = scene_motion_path_ref
    pre_contact_direction_ref = contact_approach_axis_ref or direction_ref
    pre_contact_distance = (
        float(pre_contact_distance_m)
        if contact_approach_axis_ref
        else -float(pre_contact_distance_m)
    )

    move_params = {
        "tool_ref": tool_contact_point_ref,
        "start_ref": start_ref,
        "end_ref": end_ref,
        "direction_ref": direction_ref,
    }
    if motion_path_ref:
        move_params["path_ref"] = motion_path_ref
    motion_axis_align = Constraint(
        "axis_align",
        {
            "tool_axis_ref": tool_motion_axis_ref,
            "target_axis_ref": tool_motion_axis_target_ref or direction_ref,
            "relation": tool_motion_axis_relation,
            "axis_role": "tool_motion_axis",
        },
        weight=0.8,
    )

    stages = [
        ConstraintStage(
            name="align",
            notes=(
                "Align the selected tool motion axis in free space before approaching the target."
            ),
            subgoal_constraints=[
                motion_axis_align,
            ],
        ),
        ConstraintStage(
            name="approach",
            notes=(
                "Place the contact point before the public start point along the "
                "opposite first VLM motion direction."
            ),
            subgoal_constraints=[
                Constraint(
                    "point_offset",
                    {
                        "tool_ref": tool_contact_point_ref,
                        "target_ref": start_ref,
                        "direction_ref": pre_contact_direction_ref,
                        "distance_m": pre_contact_distance,
                        "axis_role": "contact_approach",
                    },
                    weight=1.0,
                ),
            ],
        ),
        ConstraintStage(
            name="action",
            notes="Advance the active contact point from start toward the VLM-grounded end point.",
            subgoal_constraints=[
                Constraint(
                    "point_match",
                    {"tool_ref": tool_contact_point_ref, "target_ref": end_ref},
                    weight=1.0,
                )
            ],
            path_constraints=[
                Constraint(
                    "move_along",
                    move_params,
                    weight=1.0,
                ),
                motion_axis_align,
            ],
        ),
    ]
    axis_role_bindings = {
        "tool_motion_axis": {
            "role": (
                "tool-side motion axis aligned to the scene movement/path "
                "direction before contact and during use"
            ),
            "tool_axis_ref": tool_motion_axis_ref,
            "target_axis_ref": tool_motion_axis_target_ref or direction_ref,
            "relation": tool_motion_axis_relation,
            "stages": ["align", "action"],
        },
        "scene_direction": {
            "role": "scene path/use direction for the tool contact point",
            "target_axis_ref": direction_ref,
            "stages": ["action"],
        },
        "contact_approach_axis": {
            "role": "offset direction before first contact",
            "target_axis_ref": pre_contact_direction_ref,
            "distance_m": pre_contact_distance,
            "stages": ["approach"],
        },
    }

    return ConstraintProgram(
        task_id=task_id,
        instruction=instruction,
        tool_refs=tool_refs or {},
        scene_refs=scene_refs or {},
        stages=stages,
        metadata={
            "canonical_stage_sequence": ["align", "approach", "action"],
            "semantic_ref_model": "start_end_action_refs",
            "stage_ref_roles": {
                "align": "tool_axis_alignment",
                "approach": "start_approach",
                "action": "end",
            },
            "axis_role_bindings": {
                **axis_role_bindings,
            },
        },
    )
