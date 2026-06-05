"""Bridge legacy benchmark target markers into provider state fields.

This module consumes ``private_eval.scene_fixture`` data for older reports and
providers. It is legacy-only and must not be used by the active image-grounded
tabletop method to snap VLM refs to hidden target geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .task_spec import ToolUseTask


class TargetBridgeError(ValueError):
    """Raised when a target-selection result cannot be mapped to fixture state."""


@dataclass(frozen=True)
class TargetStateSelection:
    marker: str
    target_ref: str
    target_box_id: str
    target_box_origin_local: list[float]
    object_id: str | None = None

    def state_fields(self) -> dict[str, Any]:
        """Return the fields consumed by the SimToolReal URDF provider."""

        return {
            "target_box_id": self.target_box_id,
            "target_box_origin_local": list(self.target_box_origin_local),
        }


def _task_data(task: ToolUseTask | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(task, ToolUseTask):
        return task.data
    return task


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TargetBridgeError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TargetBridgeError(f"{name} must be a list")
    return value


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise TargetBridgeError(f"{name} must be a 3-number list")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise TargetBridgeError(f"{name} must be a 3-number list") from exc


def selected_marker_from_target_selection(target_selection: Mapping[str, Any], *, sequence_index: int = 0) -> str:
    """Extract a single selected target marker from a target-selection payload."""

    marker = target_selection.get("selected_marker") or target_selection.get("selected_target_marker")
    if isinstance(marker, str) and marker.strip():
        return marker.strip()

    markers = target_selection.get("selected_target_markers") or target_selection.get("selected_markers")
    if isinstance(markers, list) and markers:
        try:
            marker = markers[sequence_index]
        except IndexError as exc:
            raise TargetBridgeError(
                f"target selection has {len(markers)} markers; sequence_index={sequence_index} is out of range"
            ) from exc
        if isinstance(marker, str) and marker.strip():
            return marker.strip()

    raise TargetBridgeError("target_selection does not contain a selected target marker")


def _candidate_by_marker_or_ref(public_scene: Mapping[str, Any], *, marker: str | None, target_ref: str | None) -> Mapping[str, Any]:
    candidates = _require_list(public_scene.get("target_candidates"), "public_scene.target_candidates")
    for index, candidate in enumerate(candidates):
        row = _require_mapping(candidate, f"public_scene.target_candidates[{index}]")
        if marker is not None and row.get("marker") == marker:
            return row
        if target_ref is not None and row.get("target_ref") == target_ref:
            return row
    label = marker if marker is not None else target_ref
    raise TargetBridgeError(f"selected target {label!r} is not present in public_scene.target_candidates")


def _fixture_target_objects(task_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    private_eval = _require_mapping(task_data.get("private_eval"), "private_eval")
    fixture = _require_mapping(private_eval.get("scene_fixture"), "private_eval.scene_fixture")
    rows = _require_list(fixture.get("target_objects"), "private_eval.scene_fixture.target_objects")
    return [_require_mapping(row, f"private_eval.scene_fixture.target_objects[{index}]") for index, row in enumerate(rows)]


def _support_origin(task_data: Mapping[str, Any]) -> list[float]:
    private_eval = _require_mapping(task_data.get("private_eval"), "private_eval")
    fixture = _require_mapping(private_eval.get("scene_fixture"), "private_eval.scene_fixture")
    support = fixture.get("support_surface")
    if isinstance(support, Mapping) and "pose_xyz" in support:
        return _vec3(support.get("pose_xyz"), "private_eval.scene_fixture.support_surface.pose_xyz")
    return [0.0, 0.0, 0.0]


def _target_origin_local(target_object: Mapping[str, Any], *, support_origin: list[float]) -> list[float]:
    for key in ("target_box_origin_local", "box_origin_local", "collision_origin_local", "origin_local"):
        if key in target_object:
            return _vec3(target_object.get(key), f"target_object.{key}")
    pose = _vec3(target_object.get("pose_xyz"), "target_object.pose_xyz")
    return [pose[i] - support_origin[i] for i in range(3)]


def resolve_target_state_selection(
    task: ToolUseTask | Mapping[str, Any],
    target_selection: Mapping[str, Any] | None = None,
    *,
    selected_marker: str | None = None,
    selected_target_ref: str | None = None,
    sequence_index: int = 0,
    first_target_collision_index: int = 1,
) -> TargetStateSelection:
    """Resolve a benchmark target selection to the URDF-provider target box.

    The mapping is intentionally category-agnostic: public candidates bind the
    selected marker to an opaque target ref, and the private scene fixture binds
    that marker/ref pair to a target object pose.  The SimToolReal URDF provider
    names parsed collision boxes as ``collision_box_{index}``; benchmark fixture
    URDFs conventionally put the support collision first, followed by target
    objects in ``scene_fixture.target_objects`` order.
    """

    task_data = _task_data(task)
    if target_selection is not None:
        selected_marker = selected_marker or selected_marker_from_target_selection(
            target_selection,
            sequence_index=sequence_index,
        )
        selected_ref_value = target_selection.get("selected_target_ref")
        if selected_target_ref is None and isinstance(selected_ref_value, str) and selected_ref_value.strip():
            selected_target_ref = selected_ref_value.strip()

    if selected_marker is None and selected_target_ref is None:
        raise TargetBridgeError("selected_marker or selected_target_ref is required")

    public_scene = _require_mapping(task_data.get("public_scene"), "public_scene")
    candidate = _candidate_by_marker_or_ref(
        public_scene,
        marker=selected_marker,
        target_ref=selected_target_ref,
    )
    marker = str(candidate.get("marker"))
    target_ref = str(candidate.get("target_ref"))

    target_objects = _fixture_target_objects(task_data)
    matched_index = None
    matched_object = None
    for index, row in enumerate(target_objects):
        if row.get("marker") == marker or row.get("target_ref") == target_ref:
            matched_index = index
            matched_object = row
            break
    if matched_index is None or matched_object is None:
        raise TargetBridgeError(
            f"selected target marker={marker!r} target_ref={target_ref!r} is not present in scene_fixture.target_objects"
        )

    origin_local = _target_origin_local(matched_object, support_origin=_support_origin(task_data))
    return TargetStateSelection(
        marker=marker,
        target_ref=target_ref,
        target_box_id=f"collision_box_{first_target_collision_index + matched_index}",
        target_box_origin_local=origin_local,
        object_id=str(matched_object.get("object_id")) if matched_object.get("object_id") is not None else None,
    )


def target_state_fields_from_selection(
    task: ToolUseTask | Mapping[str, Any],
    target_selection: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return ``target_box_id`` and ``target_box_origin_local`` for a selection."""

    return resolve_target_state_selection(task, target_selection, **kwargs).state_fields()


__all__ = [
    "TargetBridgeError",
    "TargetStateSelection",
    "resolve_target_state_selection",
    "selected_marker_from_target_selection",
    "target_state_fields_from_selection",
]
