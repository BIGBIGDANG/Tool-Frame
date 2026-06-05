#!/usr/bin/env python3
"""Ask a VLM to mark image-space scene keypoints for ToolFrame."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


ROOT = repo_root()
SRC = ROOT / "parttoolgen" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parttoolgen.constraints.vlm_client import build_vlm_client  # noqa: E402

END_POINT_APPLICABILITY_VALUES = {"required", "optional", "not_applicable", "uncertain"}
DIRECTION_SOURCE_VALUES = {"visible_action_direction", "default_axis_candidate"}
DEFAULT_AXIS_IDS = {"support_surface_normal", "opposite_support_surface_normal"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _strip_json_response(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _prompt(
    *,
    instruction: str,
    width: int,
    height: int,
    num_images: int,
    grounding_image_index: int,
    grounding_image_path: Path,
    grounding_frame_convention: Mapping[str, Any] | None = None,
    public_scene_context: Mapping[str, Any] | None = None,
    request_world_action_axis: bool = False,
) -> str:
    # Legacy callers may still pass request_world_action_axis; scene VLM
    # grounding now uses either visible 2D scene references or an explicit
    # public default-axis candidate selected by the VLM.
    _ = request_world_action_axis
    image_note = (
        f"You are given {num_images} images. Use all images for visual context, "
        f"but return pixel coordinates only in image #{grounding_image_index + 1}: "
        f"{grounding_image_path.name}."
        if num_images > 1
        else "You are given one calibrated simulation image."
    )
    frame_note = ""
    if grounding_frame_convention:
        frame_note = (
            "\nGrounding image frame convention:\n"
            f"- image up: {grounding_frame_convention.get('image_up')}\n"
            f"- image down: {grounding_frame_convention.get('image_down')}\n"
            f"- image right: {grounding_frame_convention.get('image_right')}\n"
            "- Draw action_direction arrows in this image frame.\n"
        )
    default_axis_note = ""
    default_axis_schema = ""
    if isinstance(public_scene_context, Mapping) and public_scene_context.get("default_axis_candidates"):
        default_axis_note = (
            "\nPublic default direction-axis candidates are available for cases "
            "where the task acts along a support-surface normal and no reliable "
            "visible 2D endpoint exists. These candidates are task/world priors, "
            "not hidden robot motion plans:\n"
            f"{json.dumps(public_scene_context.get('default_axis_candidates'), indent=2)}\n"
            "Use a default-axis candidate only when the task semantics require "
            "motion into/away from the support surface and a visible action "
            "direction would be misleading. Otherwise choose visible_action_direction."
        )
        default_axis_schema = (
            '\n  "direction_decision": {\n'
            '    "source": "visible_action_direction|default_axis_candidate",\n'
            '    "selected_default_axis_id": "support_surface_normal|opposite_support_surface_normal|null",\n'
            '    "confidence": <0-1>,\n'
            '    "reason": "<why visible 2D direction or the selected default axis is appropriate>"\n'
            "  },"
        )
    return f"""
You are grounding a tool-use task in calibrated simulation images.
{image_note}
{frame_note}

Task:
{instruction}

Grounding image size: width={width}, height={height}.

First identify which visible object is the target according to the task
language. If there are multiple objects, choose the object supported by the
instruction and visible evidence.

Your role is scene grounding, not robot motion planning. Return only sparse
visible scene references that are justified by the pixels and task language.
Do not invent intermediate robot waypoints, smooth trajectories, approach
poses, controller hints, or hidden 3D motion. RGB-D lifting and the compiler
derive metric axes, depths, subgoals, and executable trajectories downstream.
All returned points must lie on the selected target object or target support
surface itself. Do not place start_point, end_point, or path_points on a table,
floor, boundary seam, occluding tool, gripper, shadow, or nearby background just
because it is adjacent to the target. For tasks on a writable, wipeable, or
otherwise usable surface, choose points inside the usable visible surface, not
on its lower/upper edge unless the task explicitly targets that edge.

Then ground two task-level point-reference sets, while returning the compact
fields below for the compiler. The VLM interface is point-only: return one
scene point, two scene points, or a short ordered list of scene points when the
task target itself cannot be represented by a single start/end pair. The
compiler interprets these point refs and turns them into tool motion.
1. start refs: the visible place where the tool first functionally acts on the
   selected target.
2. use-stage refs: point refs on the selected target. If the task needs only a
   single target point, return start_point=end_point. If the task concerns a
   region, return only start/end points that span the region. If the task
   concerns a non-straight 2D locus on the visible surface, return a short
   ordered point list on that locus, whether it is already visible or specified
   by the task language. If the use-stage motion is into/through the target,
   along an unseen depth axis, or otherwise has no separate 2D point endpoint,
   do not draw a 2D motion endpoint; return end_point equal to start_point.

For the current point/direction scaffold, mark 2D scene references:
1. target_object: a short natural-language identifier for the selected target
   object.
2. start_point: the start point on the selected target, decided from the
   task language and current image evidence. Choose the visible scene point
   where the tool first functionally
   acts on the target. This point is the tool contact path start. Contact height
   may be grounded from another view.
3. action_direction: a compact visible use-stage cue. Put the start at
   start_point. If there is a visible endpoint or visible extent, put the end
   on that visible scene reference. If there is no separate visible endpoint
   because the task acts at a point or moves along a hidden/depth direction,
   put the end equal to start_point. The arrow is a scene-reference cue, not a
   robot trajectory.
4. path_points: optional ordered scene point refs on the selected target. Use
   multiple path_points only when the task target itself needs more than a
   start/end pair, such as a non-straight 2D locus on the visible surface. The
   returned rows are still only point refs; they are not robot waypoints,
   controller hints, or a planned trajectory. Do not sample extra points along
   a straight line. Do not return path_points for point targets, holes,
   openings, fasteners, buttons, or any case where downstream 3D geometry
   should generate the use path. When returned, include the start point as the
   first path point and the final endpoint as the last path point.
   For region targets, treat the selected target as one region. Do not trace
   internal visual details such as individual letters, logo strokes, texture
   strokes, stain-boundary wiggles, speckles, pixels, or marks. Return at most
   the start and final endpoint spanning the whole region. Use more than two
   path_points only when the task target itself is a non-straight ordered 2D
   point locus.
5. end_point: final endpoint field. If path_points is returned, end_point may
   repeat the final path point; otherwise use end_point for the single final
   use-stage point. When the task evidence supports no separate endpoint,
   return end_point equal to start_point.
6. end_point_decision: state whether a visible endpoint/path endpoint is
   required, not applicable, or uncertain. This is a VLM semantic judgment from
   the task and images.

Before returning, perform this consistency check:
- If end_point_decision.applicability is "not_applicable", then the task must
  be contact-only or depth-axis-only with no separate 2D point endpoint. In
  that case end_point, action_direction.end, and start_point must be the same
  visible point, and path_points must be empty or contain only that same point.
  Do not mark an ordered 2D point target as "not_applicable" merely because it
  is specified by task language rather than already visible pixels.
- If path_points has more than two points, each intermediate point must be a
  semantically necessary visible scene point, not an interpolation sample.
- If the task is an area-coverage task, path_points must have at most two
  points. The compiler will generate any necessary coverage subgoals from the
  target extent.
- Never use path_points to describe how the robot should travel between scene
  references; that is the compiler's job.

The compiler uses one ordered scene-ref interface:
start_point -> optional path_points -> end_point.
If the task is better represented by a visible path, return the sparse visible
path in path_points. If a single endpoint is clear without a path, return it as
end_point. The downstream parser preserves returned path_points; it does not
remove repeated points or reinterpret intermediate rows as robot waypoints.

Use the attached images, task language, and public frame convention as evidence.
If one view is occluded, use the clearer view, but still return coordinates in
the specified grounding image.
{default_axis_note}
Return only the 2D scene-reference fields in the JSON schema below; calibrated
RGB-D geometry derives world-frame points and directions downstream.
Return 2D points in [y, x] format normalized to 0-1000, following the
VLM-scaffolding convention. Here y is vertical image coordinate and x is
horizontal image coordinate. Use normalized coordinates rather than pixel
coordinates.
Return only this JSON object:
{{
  "target_object": {{"label": "<short name>", "description": "<which object and why>", "confidence": <0-1>}},
  "start_point": {{"point": [<y_0_to_1000>, <x_0_to_1000>], "confidence": <0-1>}},
  "end_point_decision": {{
    "applicability": "required|optional|not_applicable|uncertain",
    "confidence": <0-1>,
    "reason": "<why a separate use-stage endpoint is or is not applicable>"
  }},
  "action_direction": {{
    "start": {{"point": [<y_0_to_1000>, <x_0_to_1000>], "confidence": <0-1>}},
    "end": {{"point": [<y_0_to_1000>, <x_0_to_1000>], "confidence": <0-1>}}
  }},
  "end_point": {{"point": [<y_0_to_1000>, <x_0_to_1000>], "confidence": <0-1>}},
{default_axis_schema}
  "path_points": [
    {{"id": "path0", "point": [<y_0_to_1000>, <x_0_to_1000>], "confidence": <0-1>}}
  ],
  "reason": "<short explanation>"
}}
""".strip()


def _validate_point(name: str, value: Any, *, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    normalized_yx = value.get("point")
    if isinstance(normalized_yx, list) and len(normalized_yx) == 2:
        y_norm = float(normalized_yx[0])
        x_norm = float(normalized_yx[1])
        if not (0.0 <= x_norm <= 1000.0 and 0.0 <= y_norm <= 1000.0):
            raise ValueError(f"{name} normalized point {normalized_yx} outside 0-1000")
        u = x_norm / 1000.0 * float(width)
        v = y_norm / 1000.0 * float(height)
    else:
        # Backwards-compatible path for older local responses.
        u = float(value["u"])
        v = float(value["v"])
        x_norm = u / float(width) * 1000.0
        y_norm = v / float(height) * 1000.0
    if not (0.0 <= u < width and 0.0 <= v < height):
        raise ValueError(f"{name} pixel ({u}, {v}) outside image {width}x{height}")
    confidence = value.get("confidence")
    if confidence is None:
        confidence = 1.0
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "u": u,
        "v": v,
        "confidence": confidence,
        "point_yx_1000": [y_norm, x_norm],
        "coordinate_format": "yx_normalized_0_1000",
    }


def _validate_optional_point(
    name: str,
    value: Any,
    *,
    width: int,
    height: int,
) -> dict[str, float] | None:
    if value is None:
        return None
    return _validate_point(name, value, width=width, height=height)


def _validate_path_points(
    value: Any,
    *,
    width: int,
    height: int,
    field_name: str = "path_points",
) -> list[dict[str, float | str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    path_points: list[dict[str, float | str]] = []
    for index, row in enumerate(value):
        point = _validate_point(f"{field_name}[{index}]", row, width=width, height=height)
        path_points.append(
            {
                "id": (
                    str(row.get("id") or f"path{index}")
                    if isinstance(row, Mapping)
                    else f"path{index}"
                ),
                **point,
            }
        )
    return path_points


def _validate_target_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "label": "target object",
            "description": "not specified by VLM response",
            "confidence": 0.0,
        }
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "label": str(value.get("label") or value.get("name") or "target object"),
        "description": str(value.get("description") or value.get("reason") or ""),
        "confidence": confidence_f,
    }


def _validate_decision(
    name: str,
    value: Any,
    *,
    allowed: set[str],
    default_applicability: str,
    default_reason: str,
) -> dict[str, Any]:
    if value is None:
        return {
            "applicability": default_applicability,
            "confidence": 0.0,
            "reason": default_reason,
            "source": "parser_default_missing_vlm_decision",
        }
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    applicability = str(value.get("applicability") or "").strip().lower()
    applicability = applicability.replace("-", "_").replace(" ", "_")
    if applicability not in allowed:
        raise ValueError(
            f"{name}.applicability must be one of {sorted(allowed)}; got {applicability!r}"
        )
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "applicability": applicability,
        "confidence": confidence_f,
        "reason": str(value.get("reason") or ""),
        "source": "vlm_task_semantic_grounding",
    }


def _validate_direction_decision(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "source": "visible_action_direction",
            "selected_default_axis_id": None,
            "confidence": 1.0,
            "reason": "No default-axis decision was returned; using visible action_direction.",
            "source_defaulted": True,
        }
    if not isinstance(value, Mapping):
        raise ValueError("direction_decision must be an object")
    source = str(value.get("source") or "").strip().lower()
    source = source.replace("-", "_").replace(" ", "_")
    if source not in DIRECTION_SOURCE_VALUES:
        raise ValueError(
            "direction_decision.source must be one of "
            f"{sorted(DIRECTION_SOURCE_VALUES)}"
        )
    selected = value.get("selected_default_axis_id")
    if selected in {"", "null"}:
        selected = None
    if source == "default_axis_candidate":
        if not isinstance(selected, str) or selected not in DEFAULT_AXIS_IDS:
            raise ValueError(
                "direction_decision.selected_default_axis_id must be one of "
                f"{sorted(DEFAULT_AXIS_IDS)} when source is default_axis_candidate"
            )
    elif selected not in {None, ""}:
        raise ValueError(
            "direction_decision.selected_default_axis_id must be null unless "
            "source is default_axis_candidate"
        )
    confidence = value.get("confidence", 1.0)
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_f = 0.0
    return {
        "source": source,
        "selected_default_axis_id": selected,
        "confidence": confidence_f,
        "reason": str(value.get("reason") or ""),
    }


def _parse_grounding_response(data: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    target_object = _validate_target_object(data.get("target_object"))
    start_point = _validate_point("start_point", data.get("start_point"), width=width, height=height)
    action_direction_raw = data.get("action_direction")
    if not isinstance(action_direction_raw, Mapping):
        raise ValueError("action_direction must be an object")
    action_direction = {
        "start": _validate_point(
            "action_direction.start",
            action_direction_raw.get("start"),
            width=width,
            height=height,
        ),
        "end": _validate_point(
            "action_direction.end",
            action_direction_raw.get("end"),
            width=width,
            height=height,
        ),
    }
    raw_end_point = _validate_optional_point(
        "end_point",
        data.get("end_point"),
        width=width,
        height=height,
    )
    end_point_decision = _validate_decision(
        "end_point_decision",
        data.get("end_point_decision"),
        allowed=END_POINT_APPLICABILITY_VALUES,
        default_applicability="required" if raw_end_point is not None else "not_applicable",
        default_reason="VLM response did not include end_point_decision; inferred from end_point.",
    )
    direction_decision = _validate_direction_decision(data.get("direction_decision"))
    raw_path_points = _validate_path_points(
        data.get("path_points"),
        width=width,
        height=height,
        field_name="path_points",
    )
    path_points_source = "vlm_path_points"
    if not raw_path_points and data.get("waypoints") is not None:
        raw_path_points = _validate_path_points(
            data.get("waypoints"),
            width=width,
            height=height,
            field_name="waypoints",
        )
        path_points_source = "legacy_vlm_waypoints_as_path_points"
    end_point = raw_end_point
    end_point_source = "vlm_end_point" if raw_end_point is not None else None
    path_points = list(raw_path_points)
    if path_points and end_point is None:
        end_point = dict(path_points[-1])
        end_point["source"] = "final_path_point_as_end_point"
        end_point_source = "final_path_point"
        end_point_decision = {
            "applicability": "required",
            "confidence": max(float(end_point_decision.get("confidence", 0.0)), float(end_point.get("confidence", 0.0))),
            "reason": (
                "The ordered path_points field includes a final endpoint; the "
                "parser uses the final path point as end_point while preserving "
                "the complete returned path."
            ),
            "source": "parser_final_path_point_as_end_point",
        }
    if end_point is None:
        raise ValueError(
            "end_point is required; for contact-only tasks return end_point equal to start_point"
        )
    intermediate_path_points = path_points[1:-1] if len(path_points) >= 2 else []
    return {
        "target_object": target_object,
        "start_point": start_point,
        "end_point_decision": end_point_decision,
        "direction_decision": direction_decision,
        "action_direction": action_direction,
        "raw_end_point": raw_end_point,
        "raw_path_points": raw_path_points,
        "path_points_source": path_points_source,
        "end_point_source": end_point_source,
        "end_point": end_point,
        "path_points": path_points,
        "intermediate_path_points": intermediate_path_points,
    }


def _semantic_refs(
    *,
    target_object: Mapping[str, Any],
    start_point: Mapping[str, Any],
    action_direction: Mapping[str, Any],
    end_point: Mapping[str, Any] | None,
    end_point_decision: Mapping[str, Any],
    direction_decision: Mapping[str, Any],
    path_points: list[dict[str, float | str]],
) -> dict[str, Any]:
    """Return method-level start/end/direction refs represented by VLM marks."""

    return {
        "schema_version": "toolframe_semantic_refs_v3",
        "source": "vlm_task_conditioned_scene_grounding",
        "start": {
            "role": "functional_start",
            "target_object": dict(target_object),
            "point_ref": "start_point",
            "refs": [
                {
                    "id": "start.start_point",
                    "kind": "point_2d",
                    "source_key": "start_point",
                    "role": "where the tool first functionally acts on the target",
                    "value": dict(start_point),
                }
            ],
        },
        "use_stage": {
            "role": "post_start_use_reference",
            "direction_ref": "action_direction",
            "end_ref": "end_point" if end_point is not None else None,
            "end_point_decision": dict(end_point_decision),
            "direction_decision": dict(direction_decision),
            "refs": [
                {
                    "id": "use_stage.action_direction",
                    "kind": "axis_2d",
                    "source_key": "action_direction",
                    "role": "direction or visual anchor from use-stage point refs",
                    "value": dict(action_direction),
                },
                *(
                    [
                        {
                            "id": "use_stage.default_axis_candidate",
                            "kind": "axis_candidate",
                            "source_key": "direction_decision",
                            "role": "VLM-selected public support-surface default axis",
                            "value": dict(direction_decision),
                        }
                    ]
                    if direction_decision.get("source") == "default_axis_candidate"
                    else []
                ),
                *(
                    [
                        {
                            "id": "use_stage.end_point",
                            "kind": "point_2d",
                            "source_key": "end_point",
                            "role": "visible image endpoint for the executable use-stage path",
                            "value": dict(end_point),
                        }
                    ]
                    if end_point is not None
                    else []
                ),
            ],
            "path_ref": "path_points" if path_points else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-mode", default="openai-compatible")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    evidence = _load_json(args.scene_evidence)
    prompt_images = evidence.get("metadata", {}).get("prompt_images", [])
    if not prompt_images:
        raise FileNotFoundError("scene evidence metadata.prompt_images is empty")
    grounding_image_index = int(evidence.get("metadata", {}).get("grounding_image_index", 0))
    if not (0 <= grounding_image_index < len(prompt_images)):
        raise ValueError(
            f"metadata.grounding_image_index={grounding_image_index} outside prompt_images"
        )
    rgb_path = Path(prompt_images[grounding_image_index])
    image_paths = [Path(path) for path in prompt_images]
    from PIL import Image  # noqa: WPS433

    with Image.open(rgb_path) as image:
        width, height = image.size
    instruction = str(evidence.get("instruction", ""))
    prompt = _prompt(
        instruction=instruction,
        width=width,
        height=height,
        num_images=len(image_paths),
        grounding_image_index=grounding_image_index,
        grounding_image_path=rgb_path,
        grounding_frame_convention=evidence.get("metadata", {}).get(
            "grounding_frame_convention"
        ),
        public_scene_context=evidence.get("metadata", {}).get("public_scene_context"),
        request_world_action_axis=False,
    )
    client = build_vlm_client(
        response_mode=args.response_mode,
        model=args.model,
        image_paths=image_paths,
        max_new_tokens=896,
    )
    response = client.generate(
        prompt=prompt,
        task_instruction=instruction,
        context_id=str(evidence.get("task_id") or "scene_keypoint_grounding"),
    )
    data = json.loads(_strip_json_response(response.raw_text))
    parsed = _parse_grounding_response(data, width=width, height=height)
    output = {
        "schema_version": "toolframe_scene_keypoint_grounding_v3",
        "task_id": evidence.get("task_id"),
        "instruction": instruction,
        "scene_evidence": str(Path(args.scene_evidence)),
        "target_object": parsed["target_object"],
        "start_point": parsed["start_point"],
        "end_point_decision": parsed["end_point_decision"],
        "direction_decision": parsed["direction_decision"],
        "action_direction": parsed["action_direction"],
        "raw_end_point": parsed["raw_end_point"],
        "raw_path_points": parsed["raw_path_points"],
        "path_points_source": parsed["path_points_source"],
        "end_point_source": parsed["end_point_source"],
        "end_point": parsed["end_point"],
        "path_points": parsed["path_points"],
        "intermediate_path_points": parsed["intermediate_path_points"],
        "semantic_refs": _semantic_refs(
            target_object=parsed["target_object"],
            start_point=parsed["start_point"],
            action_direction=parsed["action_direction"],
            end_point=parsed["end_point"],
            end_point_decision=parsed["end_point_decision"],
            direction_decision=parsed["direction_decision"],
            path_points=parsed["path_points"],
        ),
        "reason": data.get("reason", ""),
        "grounding_image_index": grounding_image_index,
        "grounding_image": str(rgb_path),
        "grounding_frame_convention": evidence.get("metadata", {}).get(
            "grounding_frame_convention"
        ),
        "public_scene_context": evidence.get("metadata", {}).get("public_scene_context"),
        "raw_response": response.raw_text,
        "vlm_metadata": response.metadata,
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "scene_keypoints.json", output)
    print(json.dumps({"scene_keypoints": str(output_dir / "scene_keypoints.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
